"""
app/services/extraction_service.py
Dual-approach NLP extraction engine.
Combines DistilBERT NER + Regex patterns to extract all 14 BIR Form 0016 fields
from a Whisper-generated transcript.

CHANGED: address extraction now uses the barangay gazetteer (fuzzy match
against real barangay names) instead of trusting raw NER output. See
_extract_address() below.
"""
import re
import logging
from typing import Optional, Dict, Any, List

from datetime import date, datetime

from app.core.config import settings
from app.services.barangay_gazetteer import match_barangay
from app.services.name_gazetteer import match_surname, match_given_name, match_middle_name

logger = logging.getLogger(__name__)

# ─── DistilBERT NER (lazy loaded) ─────────────────────────────────────────────
_ner_pipeline = None


def get_ner_pipeline():
    """Load DistilBERT NER pipeline once and cache it (quantized for low RAM footprint)."""
    global _ner_pipeline
    if _ner_pipeline is None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        logger.info("Loading DistilBERT NER pipeline...")
        model_name = "dslim/distilbert-NER"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        # Dynamic INT8 quantization on Linear layers: reduces RAM from ~260MB to ~65MB
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("DistilBERT dynamically quantized to INT8.")
        except Exception as q_err:
            logger.warning(f"DistilBERT quantization skipped, using full precision: {q_err}")

        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        logger.info("DistilBERT NER pipeline loaded.")
    return _ner_pipeline


# ─── Regex Patterns ────────────────────────────────────────────────────────────

PATTERNS = {
    "height": [
        r"(\d{1,3})\s*(?:centimeters?|cm)",
        r"(\d)\s*feet?\s*(?:and\s*)?(\d+)\s*inch(?:es)?",
        r"(\d)'(\d+)\"?",
    ],
    "weight": [
        r"(\d{2,3})\s*(?:kilograms?|kilos?|kg)",
        r"(\d{2,3})\s*(?:pounds?|lbs?)",
    ],
    "income": [
        r"(?:₱|PHP|peso[s]?)\s*([\d,]+(?:\.\d{2})?)",
        r"([\d,]+(?:\.\d{2})?)\s*(?:pesos?|PHP)",
        r"(?:gross\s*(?:annual\s*)?income\s*(?:of|is)?\s*)([\d,]+)",
    ],
    "date": [
        r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})",
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})",
        r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(January|February|March|April|May|June|July|August|September|October|November|December),?\s+(\d{4})",
    ],
    "icr": [
        r"ICR\s*(?:number|no\.?)?\s*:?\s*([A-Z0-9\-]{6,20})",
        r"immigrant\s*certificate\s*of\s*residence\s*:?\s*([A-Z0-9\-]{6,20})",
    ],
    "civil_status": [
        r"\b(single|married|widowed?|separated)\b",
    ],
    "sex": [
        r"\b(male|female)\b",
        r"\b(man|woman)\b",
    ],
    "citizenship": [
        r"\b(Filipino|American|Chinese|Japanese|Korean|British|Australian|Canadian)\b",
    ],
    "occupation": [
        r"(?:work(?:ing)?\s*as|occupation\s*(?:is)?|employed\s*as|profession\s*(?:is)?)\s+(?:a\s+|an\s+)?([\w\s]+?)(?:\s*[,.]|$)",
    ],
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# ── NEW: address-related patterns ──────────────────────────────────────────

# Matches things like "123 Rizal Street", "Purok 3", "Sitio Maligaya"
LOCAL_DETAIL_PATTERN = re.compile(
    r"((?:\d+[\w\-]*\s*)?"
    r"(?:[A-Za-z0-9\s\.]+?\s*(?:street|st\.?))?[,\s]*"
    r"(?:(?:purok|zone)\s*[\w\-]+)?[,\s]*"
    r"(?:sitio\s+[A-Za-z0-9\s]+)?)",
    re.IGNORECASE,
)

# Matches "barangay X" / "brgy X" / "brgy. X" so we can isolate the
# barangay span before fuzzy-matching it against the gazetteer
BARANGAY_MENTION_PATTERN = re.compile(
    r"(?:barangay|brgy\.?)\s+([A-Za-z0-9\-\s]+?)(?:[,.]|$)",
    re.IGNORECASE,
)


# ─── Extraction Helpers ────────────────────────────────────────────────────────

def _extract_height(text: str) -> Optional[float]:
    m = re.search(PATTERNS["height"][0], text, re.IGNORECASE)
    if m:
        return float(m.group(1))
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
    m = re.search(PATTERNS["weight"][0], text, re.IGNORECASE)
    if m:
        return float(m.group(1).replace(",", ""))
    m = re.search(PATTERNS["weight"][1], text, re.IGNORECASE)
    if m:
        lbs = float(m.group(1).replace(",", ""))
        return round(lbs * 0.453592, 1)
    return None


def _extract_income(text: str) -> Optional[float]:
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
    m = re.search(PATTERNS["date"][0], text)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}/{m.group(2)}/{m.group(3)}", "%m/%d/%Y").date()
        except ValueError:
            pass

    m = re.search(PATTERNS["date"][1], text, re.IGNORECASE)
    if m:
        try:
            month = MONTHS[m.group(1).lower()]
            return date(int(m.group(3)), month, int(m.group(2)))
        except (ValueError, KeyError):
            pass

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


def _extract_local_detail(text: str) -> Optional[str]:
    """Pull out just the house number / street / purok / sitio portion."""
    m = LOCAL_DETAIL_PATTERN.search(text)
    if m and m.group(1).strip():
        return m.group(1).strip(" ,")
    return None


def _extract_address(text: str, ner_locations: List[str]) -> Dict[str, Any]:
    """
    Build a structured, corrected address instead of trusting raw NER.

    Only the barangay is fuzzy-matched against the gazetteer (since it's
    the part most likely to be a locally-specific word Whisper mishears).
    Municipality and province are pulled from settings — they never need
    to be transcribed since this deployment covers one municipality.

    Returns:
        {
            "address": "<full formatted address string>",
            "barangay_confidence": float,  # 0.0-100.0 fuzzy match score
            "barangay_matched": bool,
        }
    """
    barangay_raw = None
    m = BARANGAY_MENTION_PATTERN.search(text)
    if m:
        barangay_raw = m.group(1).strip()
    elif ner_locations:
        barangay_raw = ner_locations[0]

    matched_barangay, score = match_barangay(barangay_raw) if barangay_raw else (None, 0.0)
    final_barangay = matched_barangay or barangay_raw

    local_detail = _extract_local_detail(text)

    parts = [
        local_detail,
        f"Barangay {final_barangay}" if final_barangay else None,
        settings.MUNICIPALITY,
        settings.PROVINCE,
    ]
    full_address = ", ".join(p for p in parts if p)

    return {
        "address": full_address,
        "barangay_confidence": score,
        "barangay_matched": matched_barangay is not None,
    }


# ─── DistilBERT NER Extraction ─────────────────────────────────────────────────

def _extract_with_ner(text: str) -> Dict[str, Any]:
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


def _apply_name_gazetteer(result: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """
    Fuzzy-correct each raw name part against the known local name lists.
    Returns the (possibly corrected) name parts plus a confidence score
    per part, so extract_fields() can blend this into the field
    confidence instead of trusting the raw NER/regex guess as-is.
    """
    corrected = {}
    match_scores = {}

    for key, raw_value, matcher in (
        ("surname", result.get("surname"), match_surname),
        ("given_name", result.get("given_name"), match_given_name),
        ("middle_name", result.get("middle_name"), match_middle_name),
    ):
        if not raw_value:
            corrected[key] = None
            match_scores[key] = 0.0
            continue

        match, score = matcher(raw_value)
        corrected[key] = match or raw_value  # keep raw text if no confident match
        match_scores[key] = score

    return {"parts": corrected, "match_scores": match_scores}


def _parse_name(names: List[str], text: str) -> Dict[str, Optional[str]]:
    """
    Split NER/keyword-detected name tokens into surname / given_name /
    middle_name, then fuzzy-correct each part against the local name
    gazetteer (see _apply_name_gazetteer above).
    """
    raw_result = {"surname": None, "given_name": None, "middle_name": None}

    name_pattern = r"(?:my\s+name\s+is|name\s*:)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})"
    m = re.search(name_pattern, text, re.IGNORECASE)
    if m:
        parts = m.group(1).split()
        if len(parts) >= 2:
            raw_result["given_name"] = parts[0].title()
            raw_result["surname"] = parts[-1].title()
            if len(parts) == 3:
                raw_result["middle_name"] = parts[1].title()
    elif names:
        combined = " ".join(names)
        parts = combined.split()
        if len(parts) >= 2:
            raw_result["given_name"] = parts[0].title()
            raw_result["surname"] = parts[-1].title()
            if len(parts) == 3:
                raw_result["middle_name"] = parts[1].title()

    gazetteer_result = _apply_name_gazetteer(raw_result)
    return {**gazetteer_result["parts"], "_match_scores": gazetteer_result["match_scores"]}


# ─── Main Extraction Function ──────────────────────────────────────────────────

def extract_fields(transcript: str, asr_confidence: float = 1.0) -> Dict[str, Any]:
    """
    Full extraction pipeline: DistilBERT NER + Regex.

    Args:
        transcript: Raw text from Whisper ASR
        asr_confidence: Whisper's confidence score (0.0-1.0) for this
                         transcript, passed through from speech_service.
                         Used so name/address confidence reflects how sure
                         Whisper actually was, not just "found something".

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

    try:
        ner_result = _extract_with_ner(text)
        name_result = _parse_name(ner_result["names"], text)
        locations = ner_result["locations"]
    except Exception as e:
        logger.warning(f"NER extraction failed, falling back to regex only: {e}")
        name_result = {"surname": None, "given_name": None, "middle_name": None, "_match_scores": {}}
        locations = []
        warnings.append(f"NER extraction unavailable: {str(e)}")

    name_match_scores = name_result.pop("_match_scores", {})

    height = _extract_height(text)
    weight = _extract_weight(text)
    income = _extract_income(text)
    dob = _extract_date(text)
    civil_status = _extract_civil_status(text)
    sex = _extract_sex(text)
    occupation = _extract_occupation(text)
    icr_number = _extract_icr(text)
    citizenship = _extract_citizenship(text)

    # ── Address extraction (now gazetteer-corrected) ───────────────────────
    address_result = _extract_address(text, locations)
    place_of_birth = locations[1] if len(locations) > 1 else None

    if not address_result["barangay_matched"] and address_result["barangay_confidence"] > 0:
        warnings.append(
            f"Barangay could not be confidently matched "
            f"(best score: {address_result['barangay_confidence']:.0f}/100) — flag for staff review"
        )

    for name_field in ("surname", "given_name", "middle_name"):
        score = name_match_scores.get(name_field, 0.0)
        if name_result.get(name_field) and score < 80:
            warnings.append(
                f"{name_field.replace('_', ' ').title()} did not confidently match a known "
                f"local name (best score: {score:.0f}/100) — flag for staff review"
            )

    extracted = {
        "surname": name_result.get("surname"),
        "given_name": name_result.get("given_name"),
        "middle_name": name_result.get("middle_name"),
        "date_of_birth": dob,
        "address": address_result["address"],
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
    # Names and address inherit the real ASR confidence instead of a flat
    # 1.0, since these are the fields most likely to be misheard for
    # unfamiliar local words. Address additionally factors in how well the
    # barangay matched the gazetteer.
    for field, value in extracted.items():
        if value is None or value == "" or value == 0.0:
            confidence_scores[field] = 0.0
            continue

        if field in ("surname", "given_name", "middle_name"):
            # Blend the real ASR confidence with how well the name matched
            # a known local name — a name that both sounded clear AND
            # matches a real local name gets the highest confidence.
            gaz_score = name_match_scores.get(field, 0.0) / 100.0
            confidence_scores[field] = round((asr_confidence + gaz_score) / 2, 3) if gaz_score else asr_confidence
        elif field == "address":
            # Blend ASR confidence with how well the barangay matched
            barangay_score = address_result["barangay_confidence"] / 100.0
            confidence_scores[field] = round((asr_confidence + barangay_score) / 2, 3)
        else:
            confidence_scores[field] = 1.0

    unextracted = [f for f, v in extracted.items() if not v and f not in ("icr_number", "middle_name")]

    if unextracted:
        warnings.append(f"Could not extract: {', '.join(unextracted)}")

    return {
        "extracted_fields": extracted,
        "confidence_scores": confidence_scores,
        "unextracted_fields": unextracted,
        "warnings": warnings,
    }