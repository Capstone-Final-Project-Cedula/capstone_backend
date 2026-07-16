"""
app/services/extraction_service.py
Dual-approach NLP extraction engine.
Combines DistilBERT NER + Regex patterns to extract all 14 BIR Form 0016 fields
from a Whisper-generated transcript.
"""
import re
import logging
from typing import Optional, Dict, Any, List
from datetime import date, datetime

logger = logging.getLogger(__name__)

# ─── DistilBERT NER (lazy loaded) ─────────────────────────────────────────────
_ner_pipeline = None


def get_ner_pipeline():
    """Load DistilBERT NER pipeline once and cache it."""
    global _ner_pipeline
    if _ner_pipeline is None:
        from transformers import pipeline
        logger.info("Loading DistilBERT NER pipeline...")
        _ner_pipeline = pipeline(
            "ner",
            model="dslim/distilbert-NER",
            aggregation_strategy="simple",
        )
        logger.info("DistilBERT NER pipeline loaded.")
    return _ner_pipeline


# ─── Regex Patterns ────────────────────────────────────────────────────────────

PATTERNS = {
    # Height: "5 feet 7 inches", "167 cm", "5'7""
    "height": [
        r"(\d{1,3})\s*(?:centimeters?|cm)",
        r"(\d)\s*feet?\s*(?:and\s*)?(\d+)\s*inch(?:es)?",
        r"(\d)'(\d+)\"?",
    ],
    # Weight: "65 kilos", "65 kg", "143 pounds"
    "weight": [
        r"(\d{2,3})\s*(?:kilograms?|kilos?|kg)",
        r"(\d{2,3})\s*(?:pounds?|lbs?)",
    ],
    # Gross Annual Income: "50,000 pesos", "₱50,000", "50000 pesos annually"
    "income": [
        r"(?:₱|PHP|peso[s]?)\s*([\d,]+(?:\.\d{2})?)",
        r"([\d,]+(?:\.\d{2})?)\s*(?:pesos?|PHP)",
        r"(?:gross\s*(?:annual\s*)?income\s*(?:of|is)?\s*)([\d,]+)",
    ],
    # Date of birth: "January 15 1990", "01/15/1990", "15th of January 1990"
    "date": [
        r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})",
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
        r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})",
    ],
    # ICR Number (for aliens)
    "icr": [
        r"ICR\s*(?:number|no\.?)?\s*:?\s*([A-Z0-9\-]{6,20})",
        r"immigrant\s*certificate\s*of\s*residence\s*:?\s*([A-Z0-9\-]{6,20})",
    ],
    # Civil Status
    "civil_status": [
        r"\b(single|married|widowed?|separated)\b",
    ],
    # Sex
    "sex": [
        r"\b(male|female)\b",
        r"\b(man|woman)\b",
    ],
    # Citizenship
    "citizenship": [
        r"\b(Filipino|American|Chinese|Japanese|Korean|British|Australian|Canadian)\b",
    ],
    # Occupation keywords
    "occupation": [
        r"(?:work(?:ing)?\s*as|occupation\s*(?:is)?|employed\s*as|profession\s*(?:is)?)\s+(?:a\s+|an\s+)?([\w\s]+?)(?:\s*[,.]|$)",
    ],
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


# ─── Extraction Helpers ────────────────────────────────────────────────────────

def _extract_height(text: str) -> Optional[float]:
    """Returns height in centimeters."""
    # Try cm first
    m = re.search(PATTERNS["height"][0], text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Try feet + inches
    m = re.search(PATTERNS["height"][1], text, re.IGNORECASE)
    if m:
        feet, inches = int(m.group(1)), int(m.group(2))
        return round((feet * 30.48) + (inches * 2.54), 1)
    m = re.search(PATTERNS["height"][2], text, re.IGNORECASE)
    if m:
        feet, inches = int(m.group(1)), int(m.group(2))
        return round((feet * 30.48) + (inches * 2.54), 1)
    return None


def _extract_weight(text: str) -> Optional[float]:
    """Returns weight in kilograms."""
    m = re.search(PATTERNS["weight"][0], text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(PATTERNS["weight"][1], text, re.IGNORECASE)
    if m:
        lbs = float(m.group(1).replace(",", ""))
        return round(lbs * 0.453592, 1)
    return None


def _extract_income(text: str) -> Optional[float]:
    """Returns gross annual income as float."""
    for pattern in PATTERNS["income"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            val = m.group(1).replace(",", "")
            try:
                return float(val)
            except ValueError:
                continue
    return None


def _extract_date(text: str) -> Optional[date]:
    """Parse date of birth from transcript."""
    # Numeric: MM/DD/YYYY or DD-MM-YYYY
    m = re.search(PATTERNS["date"][0], text)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}/{m.group(2)}/{m.group(3)}", "%m/%d/%Y").date()
        except ValueError:
            pass

    # "January 15, 1990"
    m = re.search(PATTERNS["date"][1], text, re.IGNORECASE)
    if m:
        try:
            month = MONTHS[m.group(1).lower()]
            return date(int(m.group(3)), month, int(m.group(2)))
        except (ValueError, KeyError):
            pass

    # "15th of January 1990"
    m = re.search(PATTERNS["date"][2], text, re.IGNORECASE)
    if m:
        try:
            month = MONTHS[m.group(2).lower()]
            return date(int(m.group(3)), month, int(m.group(1)))
        except (ValueError, KeyError):
            pass

    return None


def _extract_civil_status(text: str) -> Optional[str]:
    m = re.search(PATTERNS["civil_status"][0], text, re.IGNORECASE)
    if m:
        status = m.group(1).lower()
        mapping = {
            "single": "Single",
            "married": "Married",
            "widow": "Widowed",
            "widowed": "Widowed",
            "separated": "Separated",
        }
        return mapping.get(status)
    return None


def _extract_sex(text: str) -> Optional[str]:
    m = re.search(PATTERNS["sex"][0], text, re.IGNORECASE)
    if m:
        return "Male" if m.group(1).lower() == "male" else "Female"
    m = re.search(PATTERNS["sex"][1], text, re.IGNORECASE)
    if m:
        return "Male" if m.group(1).lower() == "man" else "Female"
    return None


def _extract_occupation(text: str) -> Optional[str]:
    for pattern in PATTERNS["occupation"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()
    return None


def _extract_icr(text: str) -> Optional[str]:
    for pattern in PATTERNS["icr"]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().upper()
    return None


def _extract_citizenship(text: str) -> str:
    m = re.search(PATTERNS["citizenship"][0], text, re.IGNORECASE)
    return m.group(1).title() if m else "Filipino"


# ─── DistilBERT NER Extraction ─────────────────────────────────────────────────

def _extract_with_ner(text: str) -> Dict[str, Any]:
    """
    Use DistilBERT NER to extract PER (person names), LOC (locations), and
    ORG entities from the transcript.
    Returns a dict with 'name_parts', 'locations'.
    """
    ner = get_ner_pipeline()
    entities = ner(text)

    names = []
    locations = []

    for entity in entities:
        label = entity.get("entity_group", "")
        word = entity.get("word", "").strip()
        if not word:
            continue
        if label == "PER":
            names.append(word)
        elif label == "LOC":
            locations.append(word)

    return {
        "names": names,
        "locations": locations,
    }


def _parse_name(names: List[str], text: str) -> Dict[str, Optional[str]]:
    """
    Attempt to split NER-detected name tokens into surname / given_name / middle_name.
    Falls back to keyword patterns like "my name is John Doe".
    """
    result = {"surname": None, "given_name": None, "middle_name": None}

    # Try keyword extraction first
    name_pattern = r"(?:my\s+name\s+is|name\s*:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
    m = re.search(name_pattern, text, re.IGNORECASE)
    if m:
        parts = m.group(1).split()
        if len(parts) >= 2:
            result["given_name"] = parts[0].title()
            result["surname"] = parts[-1].title()
            if len(parts) == 3:
                result["middle_name"] = parts[1].title()
        return result

    # Fall back to NER tokens
    if names:
        combined = " ".join(names)
        parts = combined.split()
        if len(parts) >= 2:
            result["given_name"] = parts[0].title()
            result["surname"] = parts[-1].title()
            if len(parts) == 3:
                result["middle_name"] = parts[1].title()

    return result


# ─── Main Extraction Function ──────────────────────────────────────────────────

def extract_fields(transcript: str) -> Dict[str, Any]:
    """
    Full extraction pipeline: DistilBERT NER + Regex.
    Returns all extractable BIR Form 0016 fields plus metadata.

    Args:
        transcript: Raw text from Whisper ASR

    Returns:
        {
            "extracted_fields": { ...ApplicantCreate fields... },
            "confidence_scores": { field: score },
            "unextracted_fields": [fields_not_found],
            "warnings": [warning_messages]
        }
    """
    text = transcript.strip()
    warnings = []
    confidence_scores = {}

    # ── DistilBERT NER pass ─────────────────────────────────────────────────
    try:
        ner_result = _extract_with_ner(text)
        name_result = _parse_name(ner_result["names"], text)
        locations = ner_result["locations"]
    except Exception as e:
        logger.warning(f"NER extraction failed, falling back to regex only: {e}")
        name_result = {"surname": None, "given_name": None, "middle_name": None}
        locations = []
        warnings.append(f"NER extraction unavailable: {str(e)}")

    # ── Regex pass ──────────────────────────────────────────────────────────
    height = _extract_height(text)
    weight = _extract_weight(text)
    income = _extract_income(text)
    dob = _extract_date(text)
    civil_status = _extract_civil_status(text)
    sex = _extract_sex(text)
    occupation = _extract_occupation(text)
    icr_number = _extract_icr(text)
    citizenship = _extract_citizenship(text)

    # Address: use first NER location if found
    address = locations[0] if locations else None
    place_of_birth = locations[1] if len(locations) > 1 else None

    # ── Build extracted fields dict ─────────────────────────────────────────
    extracted = {
        "surname": name_result.get("surname"),
        "given_name": name_result.get("given_name"),
        "middle_name": name_result.get("middle_name"),
        "date_of_birth": dob,
        "address": address or "",
        "sex": sex,
        "civil_status": civil_status,
        "citizenship": citizenship,
        "icr_number": icr_number,
        "place_of_birth": place_of_birth,
        "height_cm": height,
        "weight_kg": weight,
        "occupation": occupation,
        "gross_annual_income": income or 0.0,
        "tax_classification": "A",
    }

    # ── Confidence scoring ─────────────────────────────────────────────────
    all_fields = list(extracted.keys())
    for field, value in extracted.items():
        if value is not None and value != "" and value != 0.0:
            confidence_scores[field] = 1.0
        else:
            confidence_scores[field] = 0.0

    # Fields not extracted
    unextracted = [f for f, v in extracted.items() if not v and f not in ("icr_number", "middle_name")]

    if unextracted:
        warnings.append(f"Could not extract: {', '.join(unextracted)}")

    return {
        "extracted_fields": extracted,
        "confidence_scores": confidence_scores,
        "unextracted_fields": unextracted,
        "warnings": warnings,
    }
