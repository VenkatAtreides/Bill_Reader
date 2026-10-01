"""
Electricity Bill Reader — for the solar digital twin project.

What it does:
  1. You upload electricity bills in any file type (PDF, photo, Word, Excel, etc.) on a web page.
  2. Each bill is read by an AI model — Ollama on your laptop, or free online models
     (Gemini, Groq) with automatic switching, or Claude (paid).
  3. Automatic checks flag numbers that do not add up.
  4. You review and correct the values in an editable table.
  5. You download everything as an Excel file for the digital twin.

Run with:  streamlit run app.py
"""

import base64
import io
import json
import os
from datetime import datetime

import anthropic
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageSequence

try:  # lets the program open iPhone photos (HEIC); skipped if not installed
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-sonnet-5-5"   # change here if you want a different Claude model
CLAUDE_FALLBACK_MODELS = ["claude-sonnet-5-5", "claude-haiku-4-5-20251001", "claude-opus-5-5"]
MAX_IMAGE_EDGE = 2000                 # pixels; large photos are shrunk to this size
UNIT_TOLERANCE = 0.02                 # 2% tolerance when checking unit totals
AMOUNT_TOLERANCE = 0.05               # 5% tolerance when checking rupee totals

# ---------------------------------------------------------------------------
# The fields Claude must fill. Edit this list to add or remove fields.
# Each entry: (field_name, type, description shown to Claude)
# type is "string" or "number"
# ---------------------------------------------------------------------------

FIELDS = [
    # Identity and period
    ("consumer_number", "string", "Service connection / consumer number"),
    ("consumer_name", "string", "Name of the customer on the bill"),
    ("distribution_company", "string", "Electricity distribution company that issued the bill"),
    ("bill_date", "string", "Date the bill was issued, format YYYY-MM-DD"),
    ("billing_period_from", "string", "Start of the billing period, format YYYY-MM-DD"),
    ("billing_period_to", "string", "End of the billing period, format YYYY-MM-DD"),
    # Connection details
    ("supply_type", "string", "Either 'High Tension' or 'Low Tension'"),
    ("tariff_category", "string", "Tariff category or code exactly as printed, for example 'HT IA' or 'LT IIIB'"),
    ("sanctioned_load_kw", "number", "Sanctioned or connected load in kilowatts"),
    ("contracted_demand_kva", "number", "Contracted or sanctioned demand in kilovolt-amperes"),
    ("recorded_max_demand_kva", "number", "Maximum demand recorded by the meter in this period, in kilovolt-amperes"),
    ("billed_demand_kva", "number", "Demand actually billed, in kilovolt-amperes"),
    ("power_factor", "number", "Average power factor for the period, a number between 0 and 1"),
    # Energy consumption
    ("total_units_kwh", "number", "Total energy consumed in the period, in kilowatt-hours (units)"),
    ("peak_units_kwh", "number", "Units consumed during peak hours, if shown"),
    ("normal_units_kwh", "number", "Units consumed during normal hours, if shown"),
    ("off_peak_units_kwh", "number", "Units consumed during off-peak or night hours, if shown"),
    ("solar_export_units_kwh", "number", "Units exported to the grid from solar, if shown"),
    ("solar_import_units_kwh", "number", "Units imported from the grid on a net or gross meter, if shown separately"),
    # Charges in rupees
    ("energy_charges_rs", "number", "Energy charges in rupees"),
    ("demand_charges_rs", "number", "Demand or fixed charges in rupees"),
    ("time_of_day_charges_rs", "number", "Peak-hour surcharge minus off-peak rebate, in rupees (negative if net rebate)"),
    ("power_factor_adjustment_rs", "number", "Power factor penalty (positive) or incentive (negative), in rupees"),
    ("electricity_tax_rs", "number", "Electricity tax or duty in rupees"),
    ("solar_credit_rs", "number", "Credit given for solar export, in rupees (positive number)"),
    ("other_charges_rs", "number", "All other charges added together, in rupees (meter rent, surcharges, etc.)"),
    ("total_amount_rs", "number", "Net amount payable for this bill, in rupees"),
    # Rates if printed
    ("energy_rate_rs_per_kwh", "number", "Energy charge rate in rupees per unit, if printed"),
    ("demand_rate_rs_per_kva", "number", "Demand charge rate in rupees per kilovolt-ampere, if printed"),
]

NUMERIC_FIELDS = [name for name, kind, _ in FIELDS if kind == "number"]


def build_tool_schema():
    """Builds the structured-output definition Claude is forced to fill."""
    properties = {}
    for name, kind, description in FIELDS:
        properties[name] = {"type": [kind, "null"], "description": description}
    properties["uncertain_fields"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Names of any fields you were unsure about (blurry, ambiguous, or guessed).",
    }
    properties["notes"] = {
        "type": "string",
        "description": "Short notes about anything unusual on the bill (arrears, multiple meters, etc.).",
    }
    return {
        "name": "record_bill",
        "description": "Record the data extracted from one electricity bill.",
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": [name for name, _, _ in FIELDS] + ["uncertain_fields", "notes"],
        },
    }


EXTRACTION_PROMPT = """You are reading an Indian electricity bill (commercial or industrial customer).
Extract the requested fields and record them using the record_bill tool.

Rules:
- Copy numbers exactly as printed. Do not calculate or estimate a value that is not on the bill.
- If a field is not on the bill, use null.
- All money values in rupees, all energy in kilowatt-hours (units), demand in kilovolt-amperes.
- Remove commas from numbers (write 12500.50, not 12,500.50).
- If a value is blurry or you are not confident, still give your best reading but list the field in uncertain_fields.
- If the bill has several pages, use all of them.
"""

# ---------------------------------------------------------------------------
# Preparing files for Claude
# ---------------------------------------------------------------------------


MAX_PAGES = 15          # at most this many images are sent per bill
MAX_TEXT_CHARS = 60000  # text longer than this is cut


def _image_block(image):
    """Turns one picture into a JPEG block Claude accepts (fixes rotation, shrinks if huge)."""
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buffer.getvalue()).decode("utf-8"),
        },
    }


def _text_block(label, text):
    text = text.strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "\n[... text cut because it was very long ...]"
    return {"type": "text", "text": f"--- Content of the bill ({label}) ---\n{text}"}


def _images_from_bytes(raw):
    """Opens any picture format, including every page of a multi-page TIFF."""
    image = Image.open(io.BytesIO(raw))
    blocks = []
    for page in ImageSequence.Iterator(image):
        blocks.append(_image_block(page.copy()))
        if len(blocks) >= MAX_PAGES:
            break
    return blocks


def _from_word(raw):
    import docx  # python-docx

    document = docx.Document(io.BytesIO(raw))
    lines = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    blocks = [_text_block("Word document", "\n".join(lines))] if lines else []
    # Photos or scans pasted inside the Word file
    for rel in document.part.rels.values():
        if "image" in rel.reltype and len(blocks) < MAX_PAGES:
            try:
                blocks.extend(_images_from_bytes(rel.target_part.blob)[:1])
            except Exception:
                pass
    return blocks


def _from_powerpoint(raw):
    from pptx import Presentation

    deck = Presentation(io.BytesIO(raw))
    lines, blocks = [], []
    for number, slide in enumerate(deck.slides, start=1):
        lines.append(f"[Slide {number}]")
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                lines.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    lines.append(" | ".join(c.text.strip() for c in row.cells))
            if shape.shape_type == 13 and len(blocks) < MAX_PAGES:  # picture
                try:
                    blocks.extend(_images_from_bytes(shape.image.blob)[:1])
                except Exception:
                    pass
    return [_text_block("PowerPoint", "\n".join(lines))] + blocks


def _from_spreadsheet(raw, name):
    if name.endswith((".csv", ".tsv")):
        sep = "\t" if name.endswith(".tsv") else ","
        sheets = {"Sheet": pd.read_csv(io.BytesIO(raw), sep=sep, header=None, dtype=str)}
    else:
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, header=None, dtype=str)
    parts = []
    for sheet_name, df in sheets.items():
        parts.append(f"[Sheet: {sheet_name}]\n" + df.fillna("").to_csv(index=False, header=False))
    return [_text_block("spreadsheet", "\n".join(parts))]


def _decode_text(raw):
    for encoding in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _from_web_page(raw):
    import re
    from html import unescape

    text = _decode_text(raw)
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", text)
    text = re.sub(r"(?i)</t[dh]>", " | ", text)
    text = unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"[ \t]+", " ", text)
    return [_text_block("web page", text)]


def _from_email(raw):
    from email import policy
    from email.parser import BytesParser

    message = BytesParser(policy=policy.default).parsebytes(raw)
    blocks = []
    body = message.get_body(preferencelist=("plain", "html"))
    if body is not None:
        content = body.get_content()
        if body.get_content_type() == "text/html":
            blocks.extend(_from_web_page(content.encode("utf-8")))
        else:
            blocks.append(_text_block("email", content))
    # Bills are often sent as email attachments
    for part in message.iter_attachments():
        filename = part.get_filename() or "attachment"
        try:
            blocks.extend(convert_bytes(part.get_payload(decode=True), filename))
        except Exception:
            pass
    return blocks


def convert_bytes(raw, filename):
    """Turns any file into a list of blocks (images, PDF or text) Claude can read."""
    name = filename.lower()

    if name.endswith(".pdf") or raw[:5] == b"%PDF-":
        return [{
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": base64.standard_b64encode(raw).decode("utf-8"),
            },
        }]
    if name.endswith((".docx", ".docm")):
        return _from_word(raw)
    if name.endswith(".pptx"):
        return _from_powerpoint(raw)
    if name.endswith((".xlsx", ".xlsm", ".xls", ".csv", ".tsv")):
        return _from_spreadsheet(raw, name)
    if name.endswith((".html", ".htm")):
        return _from_web_page(raw)
    if name.endswith(".eml"):
        return _from_email(raw)
    if name.endswith((".doc", ".ppt")):
        raise ValueError(
            "This is an old Office format. Open it in Word or PowerPoint and use "
            "'Save As' → PDF (or the newer .docx / .pptx format), then upload that."
        )

    # Anything else: first try it as a picture (JPG, PNG, HEIC, WEBP, TIFF, BMP, GIF ...)
    try:
        return _images_from_bytes(raw)
    except Exception:
        pass
    # Then try it as plain text (TXT, JSON, XML ...)
    text = _decode_text(raw)
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in text[:5000])
    if text.strip() and printable / max(len(text[:5000]), 1) > 0.9:
        return [_text_block("text file", text)]
    raise ValueError(
        "Could not recognise this file type. Try saving or exporting the bill as a PDF or a photo."
    )


def file_to_content_blocks(uploaded_file):
    blocks = convert_bytes(uploaded_file.getvalue(), uploaded_file.name)
    if not blocks:
        raise ValueError("The file seems to be empty — no text or pictures found in it.")
    return blocks


def extract_bill(client, model, uploaded_file):
    """Sends one bill to Claude and returns the extracted fields as a dictionary."""
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        tools=[build_tool_schema()],
        tool_choice={"type": "tool", "name": "record_bill"},
        messages=[
            {
                "role": "user",
                "content": file_to_content_blocks(uploaded_file)
                + [{"type": "text", "text": EXTRACTION_PROMPT}],
            }
        ],
    )
    for block in response.content:
        if block.type == "tool_use":
            return dict(block.input)
    raise ValueError("Claude did not return structured data for this bill.")


# ---------------------------------------------------------------------------
# Free models: Google Gemini free tier, or Ollama running on your own laptop.
# Both speak the same "OpenAI-style" language, so one function handles both.
# ---------------------------------------------------------------------------

AUTO = "Automatic — free online models, switches if one hits its limit"
OLLAMA = "Ollama on this laptop (free, private)"
GEMINI = "Google Gemini (free tier, online)"
GROQ = "Groq — Qwen model (free tier, online)"
CLAUDE = "Claude (paid)"

PROVIDERS = {
    OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5vl:3b",
        "key_name": None,  # Ollama needs no key
        "key_label": None,
        "max_pages": 4,    # local models get slow with many pages
    },
    GEMINI: {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-3.5-flash",
        "key_name": "GEMINI_API_KEY",
        "key_label": "Gemini key (from aistudio.google.com)",
        "max_pages": 10,
    },
    GROQ: {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "qwen/qwen3.8-27b",
        "key_name": "GROQ_API_KEY",
        "key_label": "Groq key (from console.groq.com)",
        "max_pages": 3,    # Groq accepts at most 3 images per request
    },
    CLAUDE: {
        "base_url": None,
        "default_model": DEFAULT_MODEL,
        "key_name": "ANTHROPIC_API_KEY",
        "key_label": "Claude key (starts with sk-ant-)",
        "max_pages": MAX_PAGES,
    },
}

# The order Automatic mode tries the free online models in. Edit to change the order.
AUTO_ORDER = [GEMINI, GROQ]


def _pdf_to_images(raw, max_pages):
    """Turns each PDF page into a picture, because free models read pictures, not PDFs."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(raw)
    blocks = []
    for index in range(min(len(pdf), max_pages)):
        page_image = pdf[index].render(scale=2).to_pil()  # scale 2 ≈ 144 dots per inch
        blocks.append(_image_block(page_image))
    return blocks


def _to_openai_parts(blocks, max_pages):
    """Converts our content blocks into the format Gemini and Ollama accept."""
    parts, image_count = [], 0
    expanded = []
    for block in blocks:
        if block["type"] == "document":
            raw = base64.standard_b64decode(block["source"]["data"])
            expanded.extend(_pdf_to_images(raw, max_pages))
        else:
            expanded.append(block)
    for block in expanded:
        if block["type"] == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image" and image_count < max_pages:
            data = block["source"]["data"]
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
            image_count += 1
    return parts


def _json_instructions():
    lines = [f'  "{name}": {kind} or null   // {description}' for name, kind, description in FIELDS]
    lines.append('  "uncertain_fields": list of field names you were unsure about')
    lines.append('  "notes": short text about anything unusual on the bill')
    return (
        EXTRACTION_PROMPT.replace("using the record_bill tool", "as JSON")
        + "\nReply with ONE JSON object only — no explanation, no markdown — with exactly these keys:\n{\n"
        + "\n".join(lines)
        + "\n}"
    )


def _parse_json(text):
    """Pulls the JSON object out of the model's reply, even if it added extra words."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("The model did not return data in the expected format. Try again or use another model.")
    return json.loads(text[start:end + 1])


def _clean_values(bill):
    """Free models sometimes return numbers as text ('₹12,500'); turn those into real numbers."""
    import re

    cleaned = {}
    for name, kind, _ in FIELDS:
        value = bill.get(name)
        if kind == "number" and isinstance(value, str):
            digits = re.sub(r"[^0-9.\-]", "", value)
            try:
                value = float(digits) if digits not in ("", "-", ".") else None
            except ValueError:
                value = None
        cleaned[name] = value
    uncertain = bill.get("uncertain_fields") or []
    cleaned["uncertain_fields"] = uncertain if isinstance(uncertain, list) else [str(uncertain)]
    cleaned["notes"] = str(bill.get("notes") or "")
    return cleaned


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_CONTEXT = 16384      # how much text + image the local model may take at once
OLLAMA_IMAGE_EDGE = 1280    # smaller pictures for the local model = faster, fits in memory


def extract_bill_ollama(model, uploaded_file, max_pages):
    """Sends one bill to Ollama on this laptop, setting the size limit ourselves."""
    import requests

    images, texts = [], []
    for part in _to_openai_parts(file_to_content_blocks(uploaded_file), max_pages):
        if part["type"] == "text":
            texts.append(part["text"])
        else:
            data = part["image_url"]["url"].split(",", 1)[1]
            picture = Image.open(io.BytesIO(base64.standard_b64decode(data)))
            picture.thumbnail((OLLAMA_IMAGE_EDGE, OLLAMA_IMAGE_EDGE))
            buffer = io.BytesIO()
            picture.convert("RGB").save(buffer, format="JPEG", quality=90)
            images.append(base64.standard_b64encode(buffer.getvalue()).decode("utf-8"))
    texts.append(_json_instructions())
    message = {"role": "user", "content": "\n\n".join(texts)}
    if images:
        message["images"] = images
    try:
        reply = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "messages": [message],
                "stream": False,
                "format": "json",
                "options": {"num_ctx": OLLAMA_CONTEXT, "temperature": 0},
            },
            timeout=900,
        )
    except requests.exceptions.ConnectionError:
        raise ValueError("connection error")
    if reply.status_code != 200:
        raise ValueError(reply.text)
    return _clean_values(_parse_json(reply.json()["message"]["content"]))


def extract_bill_openai_style(client, model, uploaded_file, max_pages):
    """Sends one bill to Gemini or Ollama and returns the extracted fields as a dictionary."""
    content = _to_openai_parts(file_to_content_blocks(uploaded_file), max_pages)
    content.append({"type": "text", "text": _json_instructions()})
    request = dict(model=model, messages=[{"role": "user", "content": content}], temperature=0)
    try:
        response = client.chat.completions.create(response_format={"type": "json_object"}, **request)
    except Exception as error:
        if "response_format" not in str(error) and "json" not in str(error).lower():
            raise
        response = client.chat.completions.create(**request)  # some models reject the JSON switch
    return _clean_values(_parse_json(response.choices[0].message.content))


# ---------------------------------------------------------------------------
# Automatic checks
# ---------------------------------------------------------------------------


def num(value):
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def run_checks(bill):
    """Returns a list of warnings for numbers that do not add up."""
    warnings = []

    # 1. Time-of-day units should add up to total units
    total_units = num(bill.get("total_units_kwh"))
    slots = [num(bill.get(k)) for k in ("peak_units_kwh", "normal_units_kwh", "off_peak_units_kwh")]
    if total_units and any(s is not None for s in slots):
        slot_sum = sum(s for s in slots if s is not None)
        if abs(slot_sum - total_units) > UNIT_TOLERANCE * total_units:
            warnings.append(
                f"Peak + normal + off-peak units ({slot_sum:,.0f}) do not match total units ({total_units:,.0f})."
            )

    # 2. Charges should roughly add up to the total amount
    total_amount = num(bill.get("total_amount_rs"))
    parts = {
        "energy": num(bill.get("energy_charges_rs")),
        "demand": num(bill.get("demand_charges_rs")),
        "time_of_day": num(bill.get("time_of_day_charges_rs")),
        "power_factor": num(bill.get("power_factor_adjustment_rs")),
        "tax": num(bill.get("electricity_tax_rs")),
        "other": num(bill.get("other_charges_rs")),
    }
    solar_credit = num(bill.get("solar_credit_rs")) or 0
    if total_amount and parts["energy"] is not None:
        charge_sum = sum(v for v in parts.values() if v is not None) - solar_credit
        if abs(charge_sum - total_amount) > AMOUNT_TOLERANCE * total_amount:
            warnings.append(
                f"Charges add up to ₹{charge_sum:,.0f} but total amount is ₹{total_amount:,.0f} "
                "(could be arrears or a missed charge — check the bill)."
            )

    # 3. Energy charges vs units × rate
    rate = num(bill.get("energy_rate_rs_per_kwh"))
    energy = num(bill.get("energy_charges_rs"))
    if rate and total_units and energy:
        expected = rate * total_units
        if abs(expected - energy) > AMOUNT_TOLERANCE * energy:
            warnings.append(
                f"Units × rate = ₹{expected:,.0f} but energy charges are ₹{energy:,.0f} "
                "(may be normal if there are time-of-day or slab rates)."
            )

    # 4. Power factor should be between 0 and 1
    pf = num(bill.get("power_factor"))
    if pf is not None and not (0 < pf <= 1):
        warnings.append(f"Power factor {pf} is outside 0 to 1.")

    # 5. Demand exceeding contract
    recorded = num(bill.get("recorded_max_demand_kva"))
    contracted = num(bill.get("contracted_demand_kva"))
    if recorded and contracted and recorded > contracted:
        warnings.append(
            f"Recorded maximum demand ({recorded:,.0f}) exceeds contracted demand ({contracted:,.0f}) — "
            "customer may be paying excess demand penalty."
        )

    # 6. Key fields missing
    for key in ("consumer_number", "billing_period_to", "total_units_kwh", "total_amount_rs"):
        if bill.get(key) in (None, ""):
            warnings.append(f"Key field missing: {key}.")

    # 7. Fields Claude itself was unsure about
    uncertain = bill.get("uncertain_fields") or []
    if uncertain:
        warnings.append("The model was unsure about: " + ", ".join(uncertain) + ".")

    return warnings


# ---------------------------------------------------------------------------
# Excel output
# ---------------------------------------------------------------------------


def to_excel(df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Bills")
        field_help = pd.DataFrame(
            [(n, k, d) for n, k, d in FIELDS], columns=["field", "type", "meaning"]
        )
        field_help.to_excel(writer, index=False, sheet_name="Field meanings")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Web page
# ---------------------------------------------------------------------------


def secret(name):
    """Reads a setting from the Streamlit secrets file or the environment. Returns None if absent."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def get_api_key(provider_name, show_box=True):
    """Finds the key for a provider: secrets file, environment variable, then a sidebar box."""
    settings = PROVIDERS[provider_name]
    if settings["key_name"] is None:
        return "ollama"  # Ollama runs locally and needs no key
    found = secret(settings["key_name"])
    if found:
        return found
    if show_box:
        return st.sidebar.text_input(settings["key_label"], type="password", key=f"key_{provider_name}")
    return None


def make_extractor(provider_name, api_key, model):
    """Returns a function that reads one uploaded bill with the chosen provider."""
    settings = PROVIDERS[provider_name]
    if settings["base_url"] is None:
        client = anthropic.Anthropic(api_key=api_key)

        def run_claude(f):
            last_error = None
            for name in [model] + [m for m in CLAUDE_FALLBACK_MODELS if m != model]:
                try:
                    return extract_bill(client, name, f)
                except anthropic.NotFoundError as error:  # model not available to this key
                    last_error = error
            raise last_error

        return run_claude
    if provider_name == OLLAMA:
        return lambda f: extract_bill_ollama(model, f, settings["max_pages"])
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=settings["base_url"], timeout=300, max_retries=0)
    return lambda f: extract_bill_openai_style(client, model, f, settings["max_pages"])


def make_automatic_extractor(keys):
    """Tries each free online model in AUTO_ORDER; moves to the next if one fails or hits its limit."""
    chain = [
        (name, PROVIDERS[name]["default_model"], make_extractor(name, keys[name], PROVIDERS[name]["default_model"]))
        for name in AUTO_ORDER
        if keys.get(name)
    ]

    def extract(uploaded_file):
        problems = []
        for name, model, extractor in chain:
            try:
                return extractor(uploaded_file), name, model
            except Exception as error:
                problems.append(f"{name.split(' (')[0]}: {explain_error(name, error)}")
        raise ValueError("every model failed — " + " | ".join(problems))

    return extract, [name for name, _, _ in chain]


def explain_error(provider_name, error):
    """Turns a technical error into a plain hint, and always keeps the real reason at the end."""
    text = str(error)
    lowered = text.lower()
    hint = None
    if "connection" in lowered and provider_name == OLLAMA:
        hint = "Could not reach Ollama. Open the Ollama app (or run 'ollama serve') and try again."
    elif "not found" in lowered and provider_name == OLLAMA:
        hint = ("That model is not downloaded yet. Type 'ollama list' in a command window to see your models, "
                "or run: ollama pull <model name>")
    elif "memory" in lowered and provider_name == OLLAMA:
        hint = ("The laptop ran out of memory. Close other apps and try again, or lower OLLAMA_CONTEXT "
                "near the middle of app.py from 16384 to 8192.")
    elif "credit balance" in lowered or "billing" in lowered or "purchase credits" in lowered:
        hint = "No credit on this account. Add credit under Settings → Billing in the console."
    elif any(w in lowered for w in ("401", "authentication", "invalid x-api-key", "invalid api key", "unauthorized")):
        hint = "The key was rejected. Check that you copied it correctly and that it is an API key."
    elif "403" in lowered or "permission" in lowered:
        hint = "This key is not allowed to do this. Check the key's scope/permissions in the console."
    elif "429" in lowered or "quota" in lowered or "rate limit" in lowered or "exhausted" in lowered:
        hint = "Limit reached for now. Wait a minute and try again, or choose another model."
    elif "529" in lowered or "503" in lowered or "overloaded" in lowered or "high demand" in lowered:
        hint = "The service is busy right now. Wait a minute and try again."
    elif "404" in lowered or "not_found" in lowered:
        hint = "Model name not recognised. Check the model name in the sidebar."
    elif "413" in lowered or "too large" in lowered or "too long" in lowered:
        hint = "This file is too large for this model. Try Claude, or split the bill into fewer pages."
    if hint is None:
        return text
    return f"{hint} (Details: {text[:300]})"


def check_password():
    """If APP_PASSWORD is set in the secrets, ask for it before showing anything else."""
    password = secret("APP_PASSWORD")
    if not password or st.session_state.get("password_ok"):
        return
    st.title("⚡ Electricity Bill Reader")
    entered = st.text_input("Password", type="password")
    if entered and entered == password:
        st.session_state.password_ok = True
        st.rerun()
    if entered:
        st.error("Wrong password.")
    st.stop()


def main():
    st.set_page_config(page_title="Electricity Bill Reader", page_icon="⚡", layout="wide")
    check_password()
    st.title("⚡ Electricity Bill Reader")
    st.caption("Upload bills → the AI model extracts the data → check and correct → download Excel for the digital twin.")

    if "rows" not in st.session_state:
        st.session_state.rows = []

    # Sidebar settings. On the website (ONLINE_MODE = "true" in secrets) the laptop-only
    # Ollama option is hidden and Automatic is the default; on the laptop, Ollama is the default.
    online = (secret("ONLINE_MODE") or "").lower() in ("true", "1", "yes")
    if online and secret("ANTHROPIC_API_KEY"):
        choices = [CLAUDE, AUTO, GEMINI, GROQ]  # Claude key added on the website: use it by default
    elif online:
        choices = [AUTO, GEMINI, GROQ, CLAUDE]
    else:
        choices = [OLLAMA, AUTO, GEMINI, GROQ, CLAUDE]

    st.sidebar.header("Settings")
    provider_name = st.sidebar.selectbox("AI model provider", choices)

    if provider_name == AUTO:
        keys = {name: get_api_key(name, show_box=False) for name in AUTO_ORDER}
        missing = [name for name in AUTO_ORDER if not keys[name]]
        if missing:
            with st.sidebar.expander("Add keys (leave blank to skip a model)", expanded=not any(keys.values())):
                for name in missing:
                    keys[name] = st.text_input(PROVIDERS[name]["key_label"], type="password", key=f"key_{name}")
        ready = [name.split(" (")[0].split(" —")[0] for name in AUTO_ORDER if keys.get(name)]
        st.sidebar.caption("Tries in order: " + (" → ".join(ready) if ready else "no keys added yet"))
        api_key, model = ("ok" if ready else ""), None
    else:
        settings = PROVIDERS[provider_name]
        api_key = get_api_key(provider_name)
        model = st.sidebar.text_input("Model name", value=settings["default_model"], key=f"model_{provider_name}")
    if provider_name == OLLAMA:
        st.sidebar.caption("Make sure the Ollama app is running and the model is downloaded (ollama pull " + model + ").")
    if provider_name != OLLAMA and provider_name != CLAUDE:
        st.sidebar.caption("Free tiers may use what you send to improve their products. Avoid confidential customer bills.")
    if st.sidebar.button("Clear all extracted bills"):
        st.session_state.rows = []
        st.rerun()

    # Step 1: upload
    st.subheader("1. Upload bills")
    files = st.file_uploader(
        "Any file type — PDF, photo (JPG, PNG, HEIC, TIFF...), Word, Excel, CSV, PowerPoint, "
        "web page, email or text. You can select many at once.",
        type=None,
        accept_multiple_files=True,
    )

    if st.button("Extract data", type="primary", disabled=not files):
        if not api_key:
            st.error("Add a key in the sidebar first.")
            st.stop()
        if provider_name == AUTO:
            extract, _ = make_automatic_extractor(keys)
        else:
            single = make_extractor(provider_name, api_key, model)
            extract = lambda f: (single(f), provider_name, model)
        progress = st.progress(0.0)
        for i, f in enumerate(files):
            with st.spinner(f"Reading {f.name} ..."):
                try:
                    bill, used_provider, used_model = extract(f)
                    warnings = run_checks(bill)
                    row = {"file_name": f.name}
                    row.update({name: bill.get(name) for name, _, _ in FIELDS})
                    row["uncertain_fields"] = ", ".join(bill.get("uncertain_fields") or [])
                    row["notes"] = bill.get("notes", "")
                    row["check_warnings"] = " | ".join(warnings)
                    row["status"] = "Check" if warnings else "OK"
                    row["read_by"] = f"{used_provider.split(' (')[0]} / {used_model}"
                    row["extracted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                    st.session_state.rows.append(row)
                except Exception as e:
                    message = str(e) if provider_name == AUTO else explain_error(provider_name, e)
                    st.error(f"{f.name}: could not be read — {message}")
            progress.progress((i + 1) / len(files))
        st.success("Done. Review the results below.")

    if not st.session_state.rows:
        st.info("No bills extracted yet.")
        return

    # Step 2: review
    st.subheader("2. Review and correct")
    df = pd.DataFrame(st.session_state.rows)
    flagged = (df["status"] == "Check").sum()
    c1, c2 = st.columns(2)
    c1.metric("Bills extracted", len(df))
    c2.metric("Bills needing a check", int(flagged))

    for row in st.session_state.rows:
        if row["check_warnings"]:
            with st.expander(f"⚠️ {row['file_name']}"):
                for w in row["check_warnings"].split(" | "):
                    st.write("• " + w)

    st.caption("Click any cell to correct it. Your edits are included in the download.")
    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="editor")

    # Step 3: download
    st.subheader("3. Download")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    col_a, col_b, col_c = st.columns(3)
    col_a.download_button(
        "Download Excel",
        data=to_excel(edited),
        file_name=f"bills_extracted_{stamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    col_b.download_button(
        "Download CSV",
        data=edited.to_csv(index=False).encode("utf-8"),
        file_name=f"bills_extracted_{stamp}.csv",
        mime="text/csv",
    )
    col_c.download_button(
        "Download JSON",
        data=json.dumps(edited.to_dict(orient="records"), indent=2, default=str).encode("utf-8"),
        file_name=f"bills_extracted_{stamp}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
