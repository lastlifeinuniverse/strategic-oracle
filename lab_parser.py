"""
Lab Report Parser  Phase 1
============================
Extracts structured lab markers from PDF health reports.

Report types supported:
  innoquest_lipid     Innoquest lipid + special lipids panels
  innoquest_general   Any other Innoquest panel (LLM extraction)
  vo2_master          VO2 Master GXT report
  spot_mas            Gene Solutions SPOT-MAS cancer screening
  longevity           General longevity / executive summary report

Pipeline:
  1. Identify report type by keyword scan
  2. Try pdfplumber table extraction (fast, accurate for structured PDFs)
  3. Fall back to LLM extraction (Gemini Flash, temperature=0, JSON mode)
  4. Validate with Pydantic  reject if required fields missing
  5. Apply longevity flag computation from marker_catalog
"""

from __future__ import annotations

import io
import json
import re
import uuid
from datetime import date, datetime
from typing import Optional

import pdfplumber
from pypdf import PdfReader
from pydantic import BaseModel, field_validator, model_validator

from marker_catalog import NAME_TO_CODE, MARKER_CATALOG, resolve_marker_code, compute_flag


# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class ExtractedMarker(BaseModel):
    marker_code:  str
    value:        float
    unit:         str = ""         # default empty  some reports omit units
    ref_low:      Optional[float] = None
    ref_high:     Optional[float] = None
    flag:         str = "normal"   # computed after validation

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v):
        if isinstance(v, str):
            v = v.replace(",", "").replace("<", "").replace(">", "").strip()
        return float(v)

    @field_validator("unit", mode="before")
    @classmethod
    def coerce_unit(cls, v):
        if v is None:
            return ""
        return str(v)


class ExtractedReport(BaseModel):
    report_type:  str
    collected_at: date
    source:       str
    markers:      list[ExtractedMarker]

    @field_validator("collected_at", mode="before")
    @classmethod
    def parse_date(cls, v):
        if isinstance(v, date):
            return v
        for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d %b %Y", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(str(v).strip(), fmt).date()
            except ValueError:
                continue
        return date.today()  # fallback

    @model_validator(mode="after")
    def apply_flags(self):
        for m in self.markers:
            m.flag = compute_flag(m.marker_code, m.value)
        return self


# =============================================================================
# REPORT TYPE DETECTION
# =============================================================================
REPORT_KEYWORDS: dict[str, list[str]] = {
    # spot_mas first  "gene solutions" / "spot-mas" are unmistakable identifiers
    # and SPOT-MAS cover pages sometimes mention "longevity" in context
    "spot_mas":         ["spot-mas", "spot mas", "gene solutions", "multi-cancer early",
                         "mced", "ctdna"],
    # longevity before vo2_master  longevity reports cite VOmax without being GXT reports
    "longevity":        ["longevity report", "longevity summary", "biological age",
                         "longevity score", "eternami"],
    # vo2_master requires GXT-specific terms, not just "vo2"
    "vo2_master":       ["graded exercise test", "ventilatory threshold", "maximal oxygen",
                         "vo2 master", "vo2master", "gxt"],
    "innoquest_lipid":  ["lipid profile", "lipid panel", "lipoprotein", "apolipoprotein",
                         "innoquest"],
    "innoquest_general":["innoquest", "laboratory report", "biochemistry"],
}


def identify_report_type(text: str) -> str:
    lower = text.lower()
    for report_type, keywords in REPORT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return report_type
    return "unknown"


# =============================================================================
# TEXT EXTRACTION HELPERS
# =============================================================================
def _extract_text_pypdf(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def _extract_tables_pdfplumber(pdf_bytes: bytes) -> list[list[list[str]]]:
    """Return all tables from the PDF as list-of-pages, each a list-of-rows."""
    results = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                results.extend(tables)
    return results


# =============================================================================
# REFERENCE RANGE PARSER
# =============================================================================
_REF_PATTERNS = [
    (r"<\s*([\d.]+)",            lambda m: (None, float(m.group(1)))),
    (r">\s*([\d.]+)",            lambda m: (float(m.group(1)), None)),
    (r"([\d.]+)\s*[-]\s*([\d.]+)", lambda m: (float(m.group(1)), float(m.group(2)))),
    (r"Up to\s*([\d.]+)",        lambda m: (None, float(m.group(1)))),
]

def parse_ref_range(raw: str) -> tuple[Optional[float], Optional[float]]:
    if not raw:
        return None, None
    raw = raw.strip()
    for pattern, extractor in _REF_PATTERNS:
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            try:
                return extractor(m)
            except (ValueError, AttributeError):
                pass
    return None, None


# =============================================================================
# TABLE-BASED EXTRACTOR (pdfplumber, for Innoquest-style structured PDFs)
# =============================================================================
def _parse_table_rows(tables: list[list[list[str]]]) -> list[dict]:
    """
    Walk all table rows, find test-name / result / unit / ref-range columns.
    Returns list of raw dicts before Pydantic validation.
    """
    markers = []
    for table in tables:
        for row in table:
            if not row or len(row) < 2:
                continue
            cells = [str(c or "").strip() for c in row]

            # Heuristic: first non-empty cell is test name, second is value
            name_cell = cells[0]
            if not name_cell or name_cell.lower() in ("test", "analyte", "examination", ""):
                continue

            # Try to find a numeric value in columns 1-3
            value_raw = None
            value_col = None
            for idx in range(1, min(4, len(cells))):
                candidate = cells[idx].replace(",", "").replace("<", "").replace(">", "").strip()
                try:
                    float(candidate)
                    value_raw = cells[idx]
                    value_col = idx
                    break
                except ValueError:
                    continue

            if value_raw is None:
                continue  # no numeric value in this row

            code = resolve_marker_code(name_cell)
            if not code:
                continue  # unknown marker

            # Unit: cell after value column
            unit = cells[value_col + 1] if value_col + 1 < len(cells) else ""
            unit = unit.strip()

            # Ref range: next cell
            ref_raw = cells[value_col + 2] if value_col + 2 < len(cells) else ""
            ref_lo, ref_hi = parse_ref_range(ref_raw)

            markers.append({
                "marker_code": code,
                "value":       value_raw,
                "unit":        unit or MARKER_CATALOG.get(code, {}).get("unit", ""),
                "ref_low":     ref_lo,
                "ref_high":    ref_hi,
            })

    return markers


# =============================================================================
# LLM EXTRACTOR (Gemini Flash fallback)
# =============================================================================
_LLM_EXTRACTION_PROMPT = """Extract all lab test results from the text below.

Return a JSON object with this exact schema:
{{
  "collected_at": "YYYY-MM-DD",
  "markers": [
    {{
      "name": "exact test name from report",
      "value": 1.23,
      "unit": "mmol/L",
      "ref_range": "< 5.20"
    }}
  ]
}}

Rules:
- collected_at: the specimen/collection date (not report date). If not found, use today.
- value: numeric only, no symbols
- unit: as printed (mmol/L, nmol/L, %, etc.)
- ref_range: as printed ("< 3.40", "> 1.00", "0.40 - 4.00")
- Include EVERY test with a numeric result
- Do not add tests that are not in the report
- Return ONLY valid JSON, no markdown, no explanation

Lab report text:
{text}"""


def _llm_extract(text: str, gemini_fn) -> list[dict]:
    """Call Gemini Flash to extract markers from free-form text."""
    prompt = _LLM_EXTRACTION_PROMPT.format(text=text[:8000])
    try:
        raw = gemini_fn(prompt)
        if not raw:
            return []
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
        markers = []
        for m in data.get("markers", []):
            code = resolve_marker_code(m.get("name", ""))
            if not code:
                continue
            ref_lo, ref_hi = parse_ref_range(m.get("ref_range", ""))
            markers.append({
                "marker_code": code,
                "value":       m.get("value"),
                "unit":        m.get("unit", ""),
                "ref_low":     ref_lo,
                "ref_high":    ref_hi,
                "_collected_at": data.get("collected_at"),
            })
        return markers
    except Exception as e:
        print(f"[lab_parser] LLM extraction error: {e}")
        return []


# =============================================================================
# DATE EXTRACTION FROM TEXT
# =============================================================================
_DATE_PATTERNS = [
    r"(?:Collection|Collected|Specimen|Date of Collection|Report Date)[:\s]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
    r"(?:Collection|Collected|Specimen)[:\s]+(\d{1,2}\s+\w+\s+\d{4})",
    r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})",
]

def _extract_date(text: str) -> Optional[str]:
    for pattern in _DATE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


# =============================================================================
# ETERNAMI LONGEVITY REPORT EXTRACTOR
# Values appear after "ABNORMAL NORMAL OPTIMAL" scale lines, e.g.:
#   LDL [mmol/L]
#   ABNORMAL NORMAL OPTIMAL
#   3.30
# =============================================================================
_LONGEVITY_MARKER_NAMES = {
    "ldl":                      ("LDL_C",          "mmol/L"),
    "hdl":                      ("HDL_C",          "mmol/L"),
    "triglycerides":            ("TG",             "mmol/L"),
    "lipoprotein(a)":           ("LP_A",           "nmol/L"),
    "lipoprotein (a)":          ("LP_A",           "nmol/L"),
    "apolipoprotein a1":        ("APO_A1",         "g/L"),
    "apolipoprotein a-1":       ("APO_A1",         "g/L"),
    "apolipoprotein b":         ("APO_B",          "g/L"),
    "glucose (fasting)":        ("GLUCOSE_FASTING","mmol/L"),
    "fasting glucose":          ("GLUCOSE_FASTING","mmol/L"),
    "hba1c":                    ("HBA1C",          "%"),
    "homocysteine":             ("HOMOCYSTEINE",   "mol/L"),
    "hs-crp":                   ("HS_CRP",         "mg/L"),
    "uric acid":                ("URIC_ACID",      "mmol/L"),
    "tsh":                      ("TSH",            "mIU/L"),
    "vitamin d":                ("VIT_D",          "nmol/L"),
    "vitamin b12":              ("B12",            "pmol/L"),
    "folate":                   ("FOLATE",         "nmol/L"),
    "ferritin":                 ("FERRITIN",       "g/L"),
    "egfr":                     ("EGFR",           "mL/min/1.73m"),
}

def _extract_longevity_report(text: str) -> list[dict]:
    """
    Parse ETERNAMI Longevity Report text.
    Handles the visual scale layout:
      MarkerName [unit]
      ABNORMAL NORMAL OPTIMAL
      value
    """
    markers = []
    lines   = [l.strip() for l in text.split("\n")]
    scale_kw = "ABNORMAL NORMAL OPTIMAL"

    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if next line (or current) is the scale indicator
        next_line = lines[i + 1] if i + 1 < len(lines) else ""
        if scale_kw in next_line or scale_kw in line:
            # The marker name is the line before the scale
            name_line  = line if scale_kw not in line else (lines[i - 1] if i > 0 else "")
            # Strip embedded unit like "[mmol/L]"
            unit_match = re.search(r"\[([^\]]+)\]", name_line)
            unit       = unit_match.group(1) if unit_match else ""
            clean_name = re.sub(r"\[.*?\]", "", name_line).strip().lower()

            # Look ahead for the numeric value (skip the scale line itself)
            look_start = i + 2 if scale_kw in next_line else i + 1
            for j in range(look_start, min(look_start + 4, len(lines))):
                candidate = lines[j].replace(",", "").strip()
                try:
                    value = float(candidate)
                    # Find canonical code
                    code, default_unit = None, unit
                    for key, (c, u) in _LONGEVITY_MARKER_NAMES.items():
                        if key in clean_name:
                            code, default_unit = c, u
                            break
                    if code:
                        markers.append({
                            "marker_code": code,
                            "value":       value,
                            "unit":        unit or default_unit,
                        })
                    break
                except ValueError:
                    continue
        i += 1

    return markers


# =============================================================================
# SPOT-MAS EXTRACTOR
# Result is a z-score (<2 = negative, 2-3 = intermediate, >3 = positive)
# =============================================================================
def _extract_spot_mas(text: str) -> list[dict]:
    """Extract the CTDNA z-score from SPOT-MAS report text."""
    markers = []
    # Strip markdown bold/italic markers that Gemini vision sometimes produces
    clean = re.sub(r"\*+", "", text)
    # Look for z-score pattern  (handles "Z-score: 0.07", "Z-score:0.07", etc.)
    m = re.search(r"z[-\s]?score[:\s]+(\d+\.?\d*)", clean, re.IGNORECASE)
    if not m:
        # Also try "0.07" near "z-score" keyword
        m = re.search(r"(\d+\.\d+)\s*\n.*?(?:negative|z.score)", clean, re.IGNORECASE)
    if m:
        markers.append({
            "marker_code": "CTDNA_MCED",
            "value":       float(m.group(1)),
            "unit":        "z-score",
            "ref_low":     None,
            "ref_high":    2.0,
        })
    # Fallback: "No Abnormalities Detected"  z-score 0 (clearly negative)
    elif re.search(r"no abnormalit", text, re.IGNORECASE):
        markers.append({
            "marker_code": "CTDNA_MCED",
            "value":       0.0,
            "unit":        "z-score",
            "ref_low":     None,
            "ref_high":    2.0,
        })
    return markers


# =============================================================================
# VISION EXTRACTION (Gemini multimodal  for scanned / image-only PDFs)
# =============================================================================
_VISION_PROMPT = """This is a page from a health or lab report. Extract every test result you can see.

Return a JSON object with this exact schema:
{
  "collected_at": "YYYY-MM-DD",
  "markers": [
    {
      "name": "exact test name as shown",
      "value": 1.23,
      "unit": "mmol/L",
      "ref_range": "< 5.20"
    }
  ]
}

Rules:
- collected_at: the specimen or collection date shown. If not visible, use "".
- value: numeric only, no symbols or text.
- unit: exactly as printed (mmol/L, nmol/L, %, bpm, mL/kg/min, etc.)
- ref_range: exactly as printed ("< 3.40", "> 1.00", "40 - 75", etc.)
- Include EVERY test that has a numeric result.
- If a field is not visible, use null.
- Return ONLY valid JSON, no markdown fences, no explanation."""


def _pdf_to_images_bytes(pdf_bytes: bytes, max_pages: int = 8) -> list[bytes]:
    """Render each PDF page to a PNG bytes object using pypdfium2."""
    import pypdfium2
    import io as _io
    pdf = pypdfium2.PdfDocument(pdf_bytes)
    images = []
    for i in range(min(len(pdf), max_pages)):
        page   = pdf[i]
        bitmap = page.render(scale=2)          # 2 for readability
        pil    = bitmap.to_pil()
        buf    = _io.BytesIO()
        pil.save(buf, format="PNG")
        images.append(buf.getvalue())
    return images


def _gemini_vision_extract(pdf_bytes: bytes, gemini_vision_fn) -> list[dict]:
    """
    Send each page image to Gemini vision and aggregate extracted markers.
    gemini_vision_fn(prompt, image_bytes_list) -> str
    """
    try:
        page_images = _pdf_to_images_bytes(pdf_bytes)
    except Exception as e:
        print(f"  [lab_parser] PDFimage failed: {e}")
        return []

    all_markers: list[dict] = []
    collected_at = None

    for idx, img_bytes in enumerate(page_images):
        print(f"  [lab_parser] Vision extraction page {idx + 1}/{len(page_images)}...")
        try:
            raw = gemini_vision_fn(_VISION_PROMPT, img_bytes)
            if not raw:
                continue
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            data = json.loads(raw)
            if not collected_at and data.get("collected_at"):
                collected_at = data["collected_at"]
            for m in data.get("markers", []):
                if m.get("value") is None:
                    continue
                code = resolve_marker_code(m.get("name", ""))
                if not code:
                    continue
                ref_lo, ref_hi = parse_ref_range(m.get("ref_range", "") or "")
                entry = {
                    "marker_code": code,
                    "value":       m["value"],
                    "unit":        m.get("unit", ""),
                    "ref_low":     ref_lo,
                    "ref_high":    ref_hi,
                    "_collected_at": collected_at,
                }
                all_markers.append(entry)
        except Exception as e:
            print(f"  [lab_parser] Vision page {idx + 1} error: {e}")
            continue

    return all_markers


# =============================================================================
# VO2 MASTER GXT REPORT EXTRACTOR
# Handles the VO2 Master Graded Exercise Test Report format.
# Extracts: VO2max, max HR, VT1/VT2 (HR + power), training zones (5-zone),
# maximal ventilation, tidal volume, respiratory frequency.
# =============================================================================
def _parse_vo2_master(text: str, pdf_bytes: bytes = None, gemini_vision_fn=None) -> list[dict]:
    """
    Extract all key metrics from a VO2 Master GXT report.
    Uses regex on extracted text; falls back to vision extraction for image-heavy pages.
    """
    markers = []
    collected_at = None

    # --- Date ---
    date_m = re.search(r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})", text)
    if date_m:
        collected_at = date_m.group(1)

    def _add(code, value, unit="", ref_low=None, ref_high=None):
        if value is not None:
            try:
                markers.append({
                    "marker_code": code,
                    "value": float(value),
                    "unit": unit,
                    "ref_low": ref_low,
                    "ref_high": ref_high,
                    "_collected_at": collected_at,
                })
            except (ValueError, TypeError):
                pass

    # ── Maximal Metrics ──────────────────────────────────────────────────────
    # VO2max (mL/kg/min) — look for patterns like "30.0" near "VO2max" or "mL/kg/min"
    m = re.search(r"VO2?max?\s*[:\s]*(\d+\.?\d*)\s*mL/kg/min", text, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d+\.?\d*)\s*mL/kg/min", text)
    if m:
        _add("VO2_MAX", m.group(1), "mL/kg/min")

    # Max Heart Rate (bpm at VO2max)
    # The report lists Power / VO2max / Heart Rate in that order under Maximal Metrics
    m = re.search(r"(?:Heart Rate|HR)\s*\n\s*(\d{2,3})\s*\n\s*bpm", text, re.IGNORECASE)
    if not m:
        # Try "185 bpm" near "Maximal" section
        m = re.search(r"Maximal.*?(\d{3})\s*\n\s*bpm", text, re.IGNORECASE | re.DOTALL)
    if not m:
        # Fallback: Max HR from chart annotation
        m = re.search(r"HR\s+Min:\s+\d+\s+Max:\s+(\d+)", text)
    if m:
        _add("MAX_HR_MEASURED", m.group(1), "bpm")

    # Max Power (W)
    m = re.search(r"Power\s*\n\s*(\d{3,4})\s*\n\s*W", text, re.IGNORECASE)
    if not m:
        m = re.search(r"(\d{3,4})\s*\n\s*W\s*\n\s*Ventilation", text, re.IGNORECASE)
    if m:
        _add("MAX_POWER", m.group(1), "W")

    # ── Ventilatory Thresholds ───────────────────────────────────────────────
    # VT1 HR and VT2 HR — appear together: "Heart Rate  156  173  bpm"
    vt_block = re.search(
        r"(?:Ventilatory Thresholds?|VT1|VT2).*?Heart Rate\s+(\d{2,3})\s+(\d{2,3})\s*bpm",
        text, re.IGNORECASE | re.DOTALL
    )
    if vt_block:
        _add("VT1_HR", vt_block.group(1), "bpm")
        _add("VT2_HR", vt_block.group(2), "bpm")
    else:
        # Try line-by-line: "156 173" on a line after "Heart Rate"
        m = re.search(r"Heart Rate\s+(\d{3})\s+(\d{3})\s*bpm", text)
        if m:
            _add("VT1_HR", m.group(1), "bpm")
            _add("VT2_HR", m.group(2), "bpm")

    # VT1 Power and VT2 Power — "Power  169  259  W"
    m = re.search(r"Power\s+(\d{2,3})\s+(\d{2,3})\s*W", text)
    if m:
        _add("VT1_POWER", m.group(1), "W")
        _add("VT2_POWER", m.group(2), "W")

    # ── Training Zones (5-zone system) ───────────────────────────────────────
    # Zone HR boundaries: Zone 1-2 boundary = 140, Zone 2-3 = 156, Zone 3-4 = 164, Zone 4-5 = 173
    # Format in report: "Heart Rate [bpm]  140  156  164  173"
    zone_hr = re.search(
        r"Heart Rate\s*\[bpm\]\s+(\d{2,3})\s+(\d{2,3})\s+(\d{2,3})\s+(\d{2,3})",
        text
    )
    if zone_hr:
        _add("GXT_ZONE1_HR_MAX", zone_hr.group(1), "bpm")
        _add("GXT_ZONE2_HR_MAX", zone_hr.group(2), "bpm")
        _add("GXT_ZONE3_HR_MAX", zone_hr.group(3), "bpm")
        _add("GXT_ZONE4_HR_MAX", zone_hr.group(4), "bpm")

    # ── VO2max Percentile ────────────────────────────────────────────────────
    # "5%" or "10%" shown on the comparison scale
    pct_m = re.search(r"(\d{1,3})%\s*\n?\s*(?:Very Poor|Poor|Fair|Good)", text, re.IGNORECASE)
    if not pct_m:
        pct_m = re.search(r"\(\s*(\d{1,3})%\s*\)", text)
    if pct_m:
        _add("VO2_MAX_PERCENTILE", pct_m.group(1), "%ile")

    print(f"  [lab_parser] VO2 Master direct extraction: {len(markers)} markers found")

    # ── Vision fallback for image-rendered PDFs ───────────────────────────────
    if len(markers) < 3 and pdf_bytes and gemini_vision_fn:
        print("  [lab_parser] VO2 Master: few markers from text — trying vision on page 1...")
        try:
            page_images = _pdf_to_images_bytes(pdf_bytes, max_pages=1)
            if page_images:
                vision_prompt = (
                    "This is a VO2 Master Graded Exercise Test report. Extract ALL numeric values:\n"
                    "- VO2max (mL/kg/min)\n- Max Heart Rate (bpm)\n- Max Power (W)\n"
                    "- VT1 Heart Rate (bpm)\n- VT2 Heart Rate (bpm)\n"
                    "- VT1 Power (W)\n- VT2 Power (W)\n"
                    "- Training Zone HR boundaries (Zone 1, 2, 3, 4 upper limits in bpm)\n"
                    "- VO2max percentile shown on the comparison scale\n"
                    "Return JSON: {\"collected_at\": \"YYYY-MM-DD\", \"markers\": [{\"name\": \"...\", \"value\": 0.0, \"unit\": \"...\"}]}"
                )
                raw = gemini_vision_fn(vision_prompt, page_images[0])
                if raw:
                    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
                    data = json.loads(raw)
                    if not collected_at:
                        collected_at = data.get("collected_at")
                    for m_item in data.get("markers", []):
                        code = resolve_marker_code(m_item.get("name", ""))
                        if code and m_item.get("value") is not None:
                            _add(code, m_item["value"], m_item.get("unit", ""))
        except Exception as e:
            print(f"  [lab_parser] VO2 Master vision fallback error: {e}")

    return markers


# =============================================================================
# MAIN PARSE FUNCTION
# =============================================================================
def parse_lab_pdf(
    pdf_bytes: bytes,
    gemini_fn=None,
    gemini_vision_fn=None,
) -> tuple[Optional[ExtractedReport], str]:
    """
    Parse a lab PDF into an ExtractedReport.

    Args:
        pdf_bytes: raw PDF file bytes
        gemini_fn: callable(prompt: str) -> str  (Gemini Flash text generation)

    Returns:
        (ExtractedReport | None, error_message)
        On success error_message is "".
    """
    # 1. Extract raw text
    try:
        text = _extract_text_pypdf(pdf_bytes)
    except Exception as e:
        return None, f"Could not read PDF: {e}"

    is_scanned = not text.strip()
    if is_scanned:
        print("  [lab_parser] No text found  attempting vision extraction...")
        if not gemini_vision_fn:
            return None, (
                "PDF is a scanned image and no vision extractor was provided. "
                "Pass gemini_vision_fn to enable OCR via Gemini."
            )

        # Step 1  get a plain-text description of all pages to identify report type
        _desc_prompt = (
            "Describe all visible text, numbers, company names, test names, "
            "and results on this page. Be exhaustive and include every number you see."
        )
        vision_texts = []
        page_images  = _pdf_to_images_bytes(pdf_bytes)
        for idx, img in enumerate(page_images):
            desc = gemini_vision_fn(_desc_prompt, img)
            if desc:
                vision_texts.append(desc)
        combined_vision_text = "\n".join(vision_texts)

        # Step 2  identify report type from vision description
        scanned_type = identify_report_type(combined_vision_text)
        print(f"  [lab_parser] Scanned report type (vision): {scanned_type}")

        raw_markers: list[dict] = []
        scanned_source = "Scanned Report (Vision)"
        scanned_date   = _extract_date(combined_vision_text) or str(date.today())

        if scanned_type == "spot_mas":
            raw_markers   = _extract_spot_mas(combined_vision_text)
            scanned_source = "Gene Solutions SPOT-MAS"
            # Try to get date from SPOT-MAS text (format DD.MM.YYYY)
            m = re.search(r"(\d{2}\.\d{2}\.\d{4})", combined_vision_text)
            if m:
                d, mo, y = m.group(1).split(".")
                scanned_date = f"{y}-{mo}-{d}"
            print(f"  [lab_parser] SPOT-MAS extractor: {len(raw_markers)} markers")
        else:
            # General vision structured extraction
            raw_markers = _gemini_vision_extract(pdf_bytes, gemini_vision_fn)

        if not raw_markers:
            return None, (
                f"Vision extraction found no recognisable lab markers in this scanned PDF "
                f"(detected type: {scanned_type})."
            )

        llm_collected_at = None
        if raw_markers and "_collected_at" in raw_markers[0]:
            llm_collected_at = raw_markers[0].pop("_collected_at", None)

        valid_markers = []
        for m in raw_markers:
            try:
                marker = ExtractedMarker(**{k: v for k, v in m.items()
                                            if k in ExtractedMarker.model_fields})
                valid_markers.append(marker)
            except Exception as e:
                print(f"  [lab_parser] Skipping: {m.get('marker_code')}  {e}")

        if not valid_markers:
            return None, "Vision extraction found markers but none passed validation."

        collected_at_str = llm_collected_at or scanned_date
        try:
            report = ExtractedReport(
                report_type=scanned_type if scanned_type != "unknown" else "scanned_pdf",
                collected_at=collected_at_str,
                source=scanned_source,
                markers=valid_markers,
            )
        except Exception as e:
            return None, f"Report validation failed: {e}"
        print(f"  [lab_parser] Vision parsed {len(report.markers)} markers from {report.collected_at}")
        return report, ""

    # 2. Identify report type
    report_type = identify_report_type(text)
    source_map = {
        "innoquest_lipid":   "Innoquest Lipid Panel",
        "innoquest_general": "Innoquest General",
        "vo2_master":        "VO2 Master GXT",
        "spot_mas":          "Gene Solutions SPOT-MAS",
        "longevity":         "Longevity Report (ETERNAMI)",
        "unknown":           "Unknown Source",
    }
    source = source_map.get(report_type, "Unknown Source")
    print(f"  [lab_parser] Report type: {report_type}")

    raw_markers: list[dict] = []
    llm_collected_at = None

    # 3a. Longevity report  specialized visual-scale extractor, then vision fallback
    if report_type == "longevity":
        raw_markers = _extract_longevity_report(text)
        print(f"  [lab_parser] Longevity extractor: {len(raw_markers)} markers")
        # ETERNAMI visual-scale layout doesn't parse well from text; use vision on marker pages
        if not raw_markers and gemini_vision_fn:
            print("  [lab_parser] Longevity vision fallback (pages 8-10)...")
            raw_markers = _gemini_vision_extract(pdf_bytes, gemini_vision_fn, start_page=8)
        # Final fallback: generic LLM text extraction
        if not raw_markers and gemini_fn:
            print("  [lab_parser] Longevity LLM fallback...")
            raw_markers = _llm_extract(text, gemini_fn)
            if raw_markers:
                llm_collected_at = raw_markers[0].pop("_collected_at", None)

    # 3b. SPOT-MAS  z-score extractor
    elif report_type == "spot_mas":
        raw_markers = _extract_spot_mas(text)
        print(f"  [lab_parser] SPOT-MAS extractor: {len(raw_markers)} markers")

    # 3c. VO2 Master GXT — dedicated regex extractor
    elif report_type == "vo2_master":
        raw_markers = _parse_vo2_master(text, pdf_bytes=pdf_bytes, gemini_vision_fn=gemini_vision_fn)
        print(f"  [lab_parser] VO2 Master extractor: {len(raw_markers)} markers")
        if raw_markers:
            llm_collected_at = raw_markers[0].get("_collected_at")
            for r in raw_markers:
                r.pop("_collected_at", None)

    else:
        # 3d. General: try pdfplumber table extraction first
        try:
            tables = _extract_tables_pdfplumber(pdf_bytes)
            if tables:
                raw_markers = _parse_table_rows(tables)
                print(f"  [lab_parser] Table extraction: {len(raw_markers)} markers")
        except Exception as e:
            print(f"  [lab_parser] Table extraction failed: {e}")

        # 3d. LLM fallback
        if not raw_markers and gemini_fn:
            print("  [lab_parser] Falling back to LLM extraction...")
            raw_markers = _llm_extract(text, gemini_fn)
            if raw_markers:
                llm_collected_at = raw_markers[0].pop("_collected_at", None)
            print(f"  [lab_parser] LLM extraction: {len(raw_markers)} markers")

    if not raw_markers:
        return None, (
            f"No recognised lab markers found in this PDF (type: {report_type}). "
            "Check that the PDF contains selectable text and matches a supported report format."
        )

    # 5. Extract collection date
    collected_at_str = llm_collected_at or _extract_date(text) or str(date.today())

    # 6. Validate with Pydantic
    valid_markers = []
    for m in raw_markers:
        try:
            marker = ExtractedMarker(**{k: v for k, v in m.items()
                                        if k in ExtractedMarker.model_fields})
            valid_markers.append(marker)
        except Exception as e:
            print(f"  [lab_parser] Skipping invalid marker {m.get('marker_code')}: {e}")

    if not valid_markers:
        return None, "Markers were found but none passed validation. Check PDF format."

    try:
        report = ExtractedReport(
            report_type=report_type,
            collected_at=collected_at_str,
            source=source,
            markers=valid_markers,
        )
    except Exception as e:
        return None, f"Report validation failed: {e}"

    print(f"  [lab_parser] Parsed {len(report.markers)} markers from {report.collected_at}")
    return report, ""
