"""
app/services/answer_service.py
Understands ONE interview answer at a time for the live voice interview.

The web interview asks one question per field ("What is your sex?"). Each
spoken or typed answer is sent here with its field key, and this module:

  * cleans it (drops "my surname is", "po", "it's", fillers),
  * extracts the value with regex (dates, numbers, units, sex, civil status,
    ICR, income) and DistilBERT NER (person names, places),
  * checks it is a real answer for that question ("nails" is not a sex),
  * returns a normalized value plus a short spoken message when it is not
    usable, so the interview asks again instead of saving nonsense.

extraction_service.extract_fields() remains the whole-sentence extractor;
this module reuses its DistilBERT pipeline and barangay gazetteer.
"""
from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from app.core.config import settings
from app.services import extraction_service, name_lexicon, place_gazetteer
from app.services.barangay_gazetteer import (
    load_aliases, load_barangays, match_barangay, region_town, town_barangays,
)

logger = logging.getLogger(__name__)


# ─── Result type ──────────────────────────────────────────────────────────────

@dataclass
class AnswerResult:
    field_key: str
    ok: bool
    value: Optional[str] = None          # what the form stores / shows
    display: Optional[str] = None        # how the assistant reads it back
    normalized: Optional[str] = None     # machine form (ISO date, number)
    message: Optional[str] = None        # what to say when not ok / to confirm
    reason: Optional[str] = None         # empty | unclear | invalid | partial
    needs_confirmation: bool = False     # plausible but unrecognized: ask "is that right?"
    methods: List[str] = field(default_factory=list)  # "regex", "distilbert", "gazetteer"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    # Place of birth (plan 001.8): the closest list entries, to tap. {"value", "province"}: province is a
    # PSGC code when the entry is a province (its town is asked next), "" when it is a whole place.
    suggestions: List[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _ok(key, value, display=None, normalized=None, methods=None, confidence=1.0,
        needs_confirmation=False, message=None, warnings=None) -> AnswerResult:
    return AnswerResult(
        field_key=key, ok=True, value=value, display=display or value,
        normalized=normalized if normalized is not None else value,
        methods=methods or ["regex"], confidence=confidence,
        needs_confirmation=needs_confirmation, message=message, warnings=warnings or [],
    )


def _fail(key, message, reason="invalid", methods=None) -> AnswerResult:
    return AnswerResult(field_key=key, ok=False, message=message, reason=reason,
                        methods=methods or ["regex"])


# ─── Text helpers ─────────────────────────────────────────────────────────────

# Whole answers that mean "I didn't understand / say it again".
UNCLEAR_PATTERN = re.compile(
    r"^(?:what|huh|pardon|sorry|excuse me|come again|say (?:it )?again|again|repeat(?: (?:it|that|please))?"
    r"|can you repeat(?: (?:it|that))?|i don'?t (?:know|understand)|not sure|ano|ano po|ha|hindi ko alam"
    r"|pakiulit|ulit|ulitin|what do you mean|i didn'?t (?:hear|catch) (?:you|that))[\s?.!]*$",
    re.IGNORECASE,
)

BLANK_ANSWERS = {
    "none", "no", "nope", "n/a", "na", "not applicable", "nothing", "blank", "skip",
    "wala", "wala po", "wala akong", "no i don't have", "i don't have", "i don't have one",
    "i have none", "no middle name", "i don't have a middle name", "walang", "no icr",
}


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _basic_clean(text: str) -> str:
    """Lowercase-free cleanup shared by every field: polite particles, edge punctuation."""
    t = _squash(text)
    t = re.sub(r"\b(?:po|opo|ho|uhm+|umm+|uh+|ah+|eh+|er+|ahm+|yung|iyong)\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"^(?:yes|yeah|oo|okay|ok|sure|well|so)[,\s]+", "", t, flags=re.IGNORECASE)
    return _squash(t).strip(" ,.;:!?\"'")


def _strip_prefix(text: str, patterns: List[str]) -> str:
    t = text
    changed = True
    while changed:
        changed = False
        for p in patterns:
            new = re.sub(p, "", t, flags=re.IGNORECASE).strip(" ,.:;")
            if new != t:
                t, changed = new, True
    return t


def _is_unclear(text: str) -> bool:
    return bool(UNCLEAR_PATTERN.match(_squash(text)))


def _is_blank_answer(text: str) -> bool:
    t = re.sub(r"[.!?,]", "", _squash(text).lower())
    t = re.sub(r"\bpo\b", "", t).strip()
    return t in BLANK_ANSWERS or bool(re.fullmatch(r"(?:no|wala|none)(?: (?:po|i don'?t|ako|meron|akong))*", t))


def _title(text: str) -> str:
    words = []
    for w in _squash(text).split(" "):
        if not w:
            continue
        # Keep things like "III", "Jr.", keep inner hyphen/apostrophe casing.
        if re.fullmatch(r"(?:i{1,3}|iv|v)", w, re.IGNORECASE):
            words.append(w.upper())
        else:
            words.append("-".join(p[:1].upper() + p[1:].lower() for p in w.split("-")))
    return " ".join(words)


# ─── Spoken numbers ───────────────────────────────────────────────────────────

UNITS = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "k": 1000, "libo": 1000, "million": 1_000_000,
          "milyon": 1_000_000, "billion": 1_000_000_000}
ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}


def _tokenize_numbers(text: str) -> List[object]:
    """
    Turn text into tokens where every number word or digit run becomes a
    number (int/float) and everything else stays a lowercase word.
    "twenty-first" -> 21 (ordinal), "1,500" -> 1500, "250k" -> 250, "k".
    """
    t = text.lower().replace("₱", " ").replace("–", "-")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)          # 1,500,000
    t = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)      # 10th -> 10
    t = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", r"\1 k", t)  # 250k -> 250 k
    t = re.sub(r"\bika-?\s*", "", t)                  # Filipino "ika-10"
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", t)       # twenty-one -> twenty one
    raw = re.findall(r"\d+(?:\.\d+)?|[a-zñ']+|[/\-.:]", t)
    tokens: List[object] = []
    for w in raw:
        if re.fullmatch(r"\d+(?:\.\d+)?", w):
            tokens.append(float(w) if "." in w else int(w))
        elif w in UNITS:
            tokens.append(("num", UNITS[w], w))
        elif w in TENS:
            tokens.append(("num", TENS[w], w))
        elif w in SCALES:
            tokens.append(("scale", SCALES[w], w))
        elif w in ORDINALS:
            tokens.append(("ord", ORDINALS[w], w))
        else:
            tokens.append(w)
    return tokens


def _combine_numbers(tokens: List[object]) -> List[object]:
    """
    Merge number-word tokens into values:
      twenty one -> 21, twenty first -> 21 (ordinal), two hundred fifty thousand -> 250000,
      1.5 million -> 1500000, nineteen ninety five -> 1995 (year pair), nineteen oh five -> 1905.
    Digits pass through. Words are kept as strings.
    """
    tokens = list(tokens)
    for idx in range(len(tokens) - 1):
        # "a thousand", "a hundred" -> one thousand / one hundred
        if tokens[idx] == "a" and isinstance(tokens[idx + 1], tuple) and tokens[idx + 1][0] == "scale":
            tokens[idx] = ("num", 1, "a")

    out: List[object] = []
    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]
        is_numeric = isinstance(tok, (int, float)) or (isinstance(tok, tuple) and tok[0] in ("num", "ord", "scale"))
        if not is_numeric or (isinstance(tok, tuple) and tok[0] == "scale" and tok[2] == "k" and not out):
            out.append(tok if not isinstance(tok, tuple) else tok[2])
            i += 1
            continue

        # Year pair: 19/20 followed by 0..99 ("nineteen ninety five", "twenty ten", "nineteen oh five").
        # Skipped when the NEXT token starts a year pair itself ("May nineteen, nineteen ninety").
        def starts_year_pair(j):
            return (j + 1 < n and isinstance(tokens[j], tuple) and tokens[j][0] == "num" and tokens[j][1] in (19, 20)
                    and isinstance(tokens[j + 1], tuple) and tokens[j + 1][0] in ("num", "ord")
                    and (tokens[j + 1][1] >= 10 or tokens[j + 1][2] == "oh"))

        if isinstance(tok, tuple) and tok[0] == "num" and tok[1] in (19, 20) and i + 1 < n and not starts_year_pair(i + 1):
            nxt = tokens[i + 1]
            # Not a year when the second part is an ordinal ("twenty first") or a lone unit ("twenty one").
            if isinstance(nxt, tuple) and nxt[0] == "num" and not (nxt[1] < 10 and nxt[2] != "oh"):
                j = i + 1
                if nxt[2] == "oh" and j + 1 < n and isinstance(tokens[j + 1], tuple) and tokens[j + 1][0] in ("num", "ord") and tokens[j + 1][1] < 10:
                    out.append(tok[1] * 100 + tokens[j + 1][1])
                    i = j + 2
                    continue
                second = nxt[1]
                k = j + 1
                if second >= 20 and second % 10 == 0 and k < n and isinstance(tokens[k], tuple) \
                        and tokens[k][0] in ("num", "ord") and 1 <= tokens[k][1] <= 9:
                    second += tokens[k][1]
                    k += 1
                if 0 <= second <= 99 and nxt[2] != "oh":
                    out.append(tok[1] * 100 + second)
                    i = k
                    continue

        # General cardinal: accumulate with scales.
        total, current, consumed, is_ord = 0, 0, 0, False
        j = i
        while j < n:
            t = tokens[j]
            if isinstance(t, (int, float)):
                if consumed and (current or total) and not (isinstance(tokens[j - 1], tuple) and tokens[j - 1][0] == "scale"):
                    break
                current += t
            elif isinstance(t, tuple) and t[0] in ("num", "ord"):
                if consumed and current:
                    # "twenty one" (tens + unit), "two hundred fifty" (hundreds + rest);
                    # anything else ("five six") starts a new number.
                    if t[1] < 10 and current % 10 == 0 and current % 100 >= 20:
                        current += t[1]
                    elif t[1] < 100 and current % 100 == 0:
                        current += t[1]
                    else:
                        break
                else:
                    current += t[1]
                if t[0] == "ord":
                    is_ord = True
                    consumed += 1
                    j += 1
                    break
            elif isinstance(t, tuple) and t[0] == "scale":
                if not consumed:
                    current = 1  # "a thousand", "hundred"
                if t[1] == 100:
                    current *= 100
                else:
                    total += current * t[1]
                    current = 0
            elif t == "and" and consumed:
                j += 1
                continue
            else:
                break
            consumed += 1
            j += 1
        value = total + current
        if consumed == 0:
            out.append(tok if not isinstance(tok, tuple) else tok[2])
            i += 1
            continue
        out.append(value if not is_ord else value)
        i = j
    return out


def spoken_numbers(text: str) -> List[float]:
    """All numbers in the text, spoken or written ("two hundred fifty thousand" -> 250000)."""
    return [t for t in _combine_numbers(_tokenize_numbers(text)) if isinstance(t, (int, float))]


def numbers_to_digits(text: str) -> str:
    """'five feet six inches' -> '5 feet 6 inches'; '1.5 million pesos' -> '1500000 pesos'."""
    parts = []
    for tok in _combine_numbers(_tokenize_numbers(text)):
        if isinstance(tok, float) and tok.is_integer():
            parts.append(str(int(tok)))
        else:
            parts.append(str(tok))
    return " ".join(parts)


def _joined_hundreds(nums: List[float]) -> Optional[float]:
    """'one sixty' (160 cm) / 'one thirty' pounds -> 160 / 130."""
    if len(nums) >= 2 and nums[0] in (1, 2) and 10 <= nums[1] <= 99 and float(nums[1]).is_integer():
        return nums[0] * 100 + nums[1]
    return None


# ─── Dates ────────────────────────────────────────────────────────────────────

MONTH_NAMES = {
    1: ["january", "jan", "enero"],
    2: ["february", "feb", "febuary", "pebrero", "febrero"],
    3: ["march", "mar", "marso", "marzo"],
    4: ["april", "apr", "abril"],
    5: ["may", "mayo"],
    6: ["june", "jun", "hunyo", "junio"],
    7: ["july", "jul", "hulyo", "julio"],
    8: ["august", "aug", "agosto"],
    9: ["september", "sep", "sept", "setyembre", "septiyembre", "septembre", "setiembre"],
    10: ["october", "oct", "oktubre", "octubre"],
    11: ["november", "nov", "nobyembre", "noviembre"],
    12: ["december", "dec", "disyembre", "diciembre"],
}
MONTH_LOOKUP = {name: num for num, names in MONTH_NAMES.items() for name in names}
MONTH_DISPLAY = ["", "January", "February", "March", "April", "May", "June", "July",
                 "August", "September", "October", "November", "December"]


def _expand_year(y: int, today: date) -> int:
    if y >= 1000:
        return y
    if y < 100:
        century = 2000 if (2000 + y) <= today.year else 1900
        return century + y
    return y


def parse_date(text: str, today: Optional[date] = None) -> Tuple[Optional[date], Optional[str]]:
    """
    Understand a birth date said or typed in almost any form:
      "May 10 1990", "10 May 1990", "May tenth nineteen ninety", "the 10th of May, 1990",
      "ika-10 ng Mayo 1990", "05/10/1990", "5-10-90", "1990-05-10", "May 10, '90".
    Returns (date, None) or (None, problem) where problem is "no_year", "no_month",
    "invalid", "future" or "too_old".
    """
    today = today or date.today()
    raw = _basic_clean(text).lower()
    raw = re.sub(r"\b(?:i was born (?:on|in)?|born (?:on|in)?|my (?:birthday|birth ?date|date of birth) is|"
                 r"birthday ko ay|ipinanganak ako (?:noong)?|noong|it'?s|the|of|ng|sa)\b", " ", raw)
    raw = raw.replace("'", " ")

    # ISO / numeric forms first (digits typed or recognized as digits).
    iso = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", raw)
    numeric = re.search(r"\b(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2,4})\b", raw)

    tokens = _combine_numbers(_tokenize_numbers(raw))
    month = None
    month_pos = None
    for idx, tok in enumerate(tokens):
        if isinstance(tok, str):
            word = tok.strip(".")
            if word in MONTH_LOOKUP and not (word == "may" and month is not None):
                # "may" is also a Filipino word ("may asawa"); accept it only as the first month word.
                month, month_pos = MONTH_LOOKUP[word], idx
                break

    numbers = [(i, t) for i, t in enumerate(tokens) if isinstance(t, (int, float)) and float(t).is_integer()]
    numbers = [(i, int(t)) for i, t in numbers]

    y = m = d = None
    if iso:
        y, m, d = int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    elif month is not None:
        m = month
        years = [(i, v) for i, v in numbers if v >= 1000 or (v >= 32 and v < 100)]
        days = [(i, v) for i, v in numbers if 1 <= v <= 31]
        if years:
            y = _expand_year(years[-1][1], today)
            days = [(i, v) for i, v in days if i != years[-1][0]]
        elif len(days) >= 2:  # "May 10 90" -> 90 handled above; "May 10 05" -> 2005
            y = _expand_year(days[-1][1], today)
            days = days[:-1]
        if days:
            # Prefer the number right next to the month name.
            d = min(days, key=lambda iv: abs(iv[0] - month_pos))[1]
        if d is None:
            return None, "no_day"
        if y is None:
            return None, "no_year"
    elif numeric:
        a, b, c = int(numeric.group(1)), int(numeric.group(2)), int(numeric.group(3))
        y = _expand_year(c, today)
        if a > 12 and b <= 12:      # 25/12/1990 -> day first
            d, m = a, b
        else:                        # Philippine forms use month/day/year
            m, d = a, b
    elif len(numbers) >= 3:
        a, b, c = numbers[0][1], numbers[1][1], numbers[2][1]
        if a >= 1000:
            y, m, d = a, b, c
        else:
            y = _expand_year(c, today)
            if a > 12 and b <= 12:
                d, m = a, b
            else:
                m, d = a, b
    else:
        if numbers and any(v >= 1000 for _, v in numbers):
            return None, "no_month"
        return None, "invalid"

    try:
        result = date(y, m, d)
    except (TypeError, ValueError):
        return None, "invalid"
    if result > today:
        return None, "future"
    if today.year - result.year > 120:
        return None, "too_old"
    return result, None


def format_date(d: date) -> str:
    return f"{MONTH_DISPLAY[d.month]} {d.day}, {d.year}"


# ─── DistilBERT helpers ───────────────────────────────────────────────────────

@dataclass
class Entity:
    label: str      # PER, LOC, ORG, MISC
    word: str
    score: float
    start: int
    end: int


def _ner_entities(text: str) -> Optional[List[Entity]]:
    """DistilBERT entities with character offsets, or None when the model is off/unavailable."""
    if not settings.NER_ENABLED or not text:
        return None
    try:
        ner = extraction_service.get_ner_pipeline()
        out = []
        for ent in ner(text):
            word = str(ent.get("word", "")).replace(" ##", "").replace("##", "").strip()
            if word:
                out.append(Entity(str(ent.get("entity_group", "")), word, float(ent.get("score", 0.0)),
                                  int(ent.get("start") or 0), int(ent.get("end") or 0)))
        return out
    except Exception as exc:  # model missing / no internet on first download
        logger.warning(f"DistilBERT unavailable, using regex only: {exc}")
        return None


def warm_up_ner() -> None:
    """Load DistilBERT in the background at startup so the first answer is not slow."""
    if settings.NER_ENABLED:
        _ner_entities("Juan Dela Cruz lives in Imbatug, Baungon, Bukidnon.")


# ─── Field handlers ───────────────────────────────────────────────────────────

NAME_PREFIXES = [
    r"^(?:my\s+)?(?:sur\s?name|last\s+name|family\s+name|given\s+name|first\s+name|middle\s+name|full\s+name|name)"
    r"(?:\s+ko)?(?:\s+(?:is|ay))?\s*:?\s*",
    r"^(?:ang\s+)?(?:apelyido|pangalan|gitnang pangalan|middle name)(?:\s+ko)?(?:\s+ay)?\s*",
    r"^(?:it'?s|it\s+is|i'?m|i\s+am|this\s+is|call\s+me|ako\s+(?:si|ay)|si)\s+",
]
# Whisper often mishears the start of "My surname is Santos" ("And what's her name is",
# "A vice surname is"); everything up to "...name is" is dropped.
NAME_LEAD_IN = re.compile(
    r"^.{0,30}?\b(?:sur\s?name|last\s?name|family\s?name|first\s?name|given\s?name|middle\s?name|name"
    r"|apelyido|pangalan)(?:'s|\s+(?:is|ay|ko(?:\s+ay)?))\s*:?\s*",
    re.IGNORECASE,
)
NAME_ABBREVIATIONS = {"ma", "sta", "sto", "jr", "sr", "st"}
HONORIFIC_TAIL = re.compile(r"(?<=\w)[\s,]+(?:po|pô|poe|pa|pah|bo|pu|pe)\W*$", re.IGNORECASE)
GLUED_HONORIFIC = re.compile(r"^(.{3,}?)(?:po|pu|bo|pa|pe|poe|pah)$", re.IGNORECASE)
NAME_ALLOWED = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿñÑ][A-Za-zÀ-ÖØ-öø-ÿñÑ .'\-]*$")
NAME_SOUND_ALIKES = {"1": "Juan", "one": "Juan", "won": "Juan", "wan": "Juan", "hwan": "Juan", "1 1": "Juan"}
NAME_PARTICLES = {"de", "dela", "del", "delos", "los", "la", "las", "san", "santa", "sta", "van", "von", "di", "da", "dos", "mc"}
# Particles of two-word Filipino surnames (Dela Cruz, De Guzman, San Juan, Sta. Maria).
FILIPINO_PARTICLES = {"de", "dela", "del", "delos", "los", "la", "las", "san", "santa", "sta", "sto", "santo"}
NOT_A_NAME = {
    "male", "female", "single", "married", "yes", "no", "okay", "ok", "hello", "hi", "test",
    "testing", "none", "nothing", "name", "surname", "wala",
}
# Everyday words that are never a whole name on their own: "It's." or "Or these." is
# what Whisper made of an unclear answer, not a name.
FILLER_WORDS = {
    "it's", "its", "it", "is", "or", "these", "this", "that", "those", "the", "a", "an", "and", "so",
    "what", "then", "there", "their", "they", "he", "she", "i", "i'm", "im", "me", "my", "you", "your",
    "be", "was", "are", "of", "to", "in", "on", "at", "for", "with", "not", "just", "like", "but",
    "here", "yes", "no", "okay", "ok", "oh", "ah", "um", "uh", "hmm", "well", "sir", "maam", "ma'am",
    "ano", "ito", "iyan", "yan", "yung", "kasi", "lang", "ba", "ang", "ng", "sa", "si",
}


def _spelled_letters(text: str) -> Optional[str]:
    """
    'C R U Z' / 'c-r-u-z' -> 'CRUZ' when the answer is spelled letter by letter;
    "space" (or "next word") separates names: 'M A R Y space A N N' -> 'MARY ANN'.
    """
    words = re.sub(r"\bnext\s+word\b", " space ", text, flags=re.IGNORECASE)
    parts = re.findall(r"[A-Za-zñÑ]+", words)
    singles = [p for p in parts if len(p) == 1]
    spaces = [p for p in parts if p.lower() in ("space", "espasyo")]
    if len(singles) >= 2 and len(singles) + len(spaces) >= len(parts) - 1:
        out = "".join(" " if p.lower() in ("space", "espasyo") else p for p in parts if len(p) == 1 or p in spaces)
        return _squash(out) or None
    return None


def _handle_name(key: str, text: str, label: str, optional: bool, spoken: bool = False) -> AnswerResult:
    if optional and _is_blank_answer(text):
        return _ok(key, "", display="none", methods=["regex"])
    cleaned = _strip_prefix(_basic_clean(text), NAME_PREFIXES)
    lead_in = NAME_LEAD_IN.match(cleaned)
    if lead_in and cleaned[lead_in.end():].strip():
        cleaned = _basic_clean(cleaned[lead_in.end():])
    # "Juan" said as "hwan" is often written as the number one by speech engines.
    if cleaned.lower().strip(" .") in NAME_SOUND_ALIKES:
        return _ok(key, NAME_SOUND_ALIKES[cleaned.lower().strip(" .")], methods=["regex"], confidence=0.7)
    spelled = _spelled_letters(cleaned)
    if spelled:
        return _ok(key, _title(spelled), methods=["regex"], confidence=0.95)
    # ...and "wan" inside a name too: "Juma 1" is "Jumawan".
    cleaned = re.sub(r"(?<=[A-Za-z])\s*\b1\b", "wan", cleaned)
    if spoken:
        # The polite "po" after a name, which Whisper also writes as "Pa", "Poe" or "Bo".
        cleaned = re.sub(HONORIFIC_TAIL, "", cleaned).strip(" ,.;:!?")
    # A pause between two names comes out as a comma or period ("Mary, Ann"): it is a space
    # (but "Ma. Cristina", "Sta. Maria" keep their abbreviation dot).
    cleaned = _squash(re.sub(
        r"\b(\w+)\s*([,.;])\s*(?=[^\W\d_])",
        lambda m: m.group(0) if m.group(2) == "." and m.group(1).lower() in NAME_ABBREVIATIONS else m.group(1) + " ",
        cleaned,
    ))
    if key == "given_name":
        # Two given names said with a joining word ("Maria and Clara", Filipino "at"): only the
        # separate word goes, never letters inside a name (Andrea, Atanacio).
        cleaned = re.sub(r"(?<=[^\W\d_])\s+(?:and|at|&)\s+(?=[^\W\d_])", " ", cleaned, flags=re.IGNORECASE)

    methods = ["regex"]
    titled = _title(cleaned)
    value = titled
    confidence = 0.8
    entities = _ner_entities(titled)
    if entities is not None:
        methods = ["distilbert", "regex"]
        persons = [e for e in entities if e.label == "PER" and e.score >= 0.5 and e.end > e.start]
        if persons:
            # Keep the whole span DistilBERT marked as a person (not just the entity words,
            # which once turned "Dela Cruz" into "Cruz"), plus surname particles before it.
            start, end = min(e.start for e in persons), max(e.end for e in persons)
            # DistilBERT may tag only part of an unfamiliar word ("Tu|mampos" -> "mampos"):
            # always keep whole words.
            while start > 0 and (titled[start - 1].isalpha() or titled[start - 1] in "'-"):
                start -= 1
            while end < len(titled) and (titled[end].isalpha() or titled[end] in "'-"):
                end += 1
            before = titled[:start].split()
            if key == "given_name" and before and all(name_lexicon.is_known(w, "given_name") for w in before):
                # NER may tag only the last given name: keep known names it skipped.
                start = 0
                before = []
            while before and before[-1].lower().strip(".") in NAME_PARTICLES:
                start = titled.rfind(before.pop(), 0, start)
            value = titled[start:end]
            confidence = 0.95

    value = _squash(value).strip(" .,'-")
    if not value or value.lower() in NOT_A_NAME:
        return _fail(key, f"Please tell me your {label}.", methods=methods)
    if all(w.lower().strip(".,!?") in FILLER_WORDS for w in value.split()):
        return _fail(key, f"I didn't catch a name. Please spell your {label}.", reason="not_a_name", methods=methods)
    if spoken:
        # Whisper writes a final "s"/"z" sound after a vowel as "'s" ("Bati's"): names have no "'s".
        value = re.sub(r"(?<=[A-Za-z])'s\b", "s", value)
    if re.search(r"\d", value):
        return _fail(key, f"A name should not have numbers. Please say your {label} again.", methods=methods)
    if not NAME_ALLOWED.match(value) or len(value) > 60 or len(value.split()) > 5:
        return _fail(key, f"I didn't get a name from that. Please say only your {label}.", methods=methods)
    if len(value.replace(" ", "")) < 2:
        return _fail(key, f"That seems too short. Please say your {label} again, or spell it.", methods=methods)

    if not spoken:
        return _ok(key, _title(value), methods=methods, confidence=confidence)

    kind = "given_name" if key == "given_name" else "surname"
    words = value.split()
    # Whisper splits an unfamiliar surname into English words ("Da Gondon", "Lumen Salk").
    # A Filipino surname is one word unless it has a particle (Dela Cruz, De Guzman).
    if kind == "surname" and len(words) > 1 and not any(w.lower().strip(".") in FILIPINO_PARTICLES for w in words):
        value = "".join(words)
    # Spell it the way it is written locally when it sounds exactly like a known name.
    local = name_lexicon.known_spelling(value, kind)
    if not local and not name_lexicon.is_known(value, kind):
        # "po" glued onto the name ("Baculiope", "Ompadbo"): drop it only when what is left
        # is a known name, since real names can end in "po" too (Ocampo).
        glued = GLUED_HONORIFIC.match(value)
        if glued:
            stem = glued.group(1)
            local = (stem if name_lexicon.is_known(stem, kind) else None) or name_lexicon.known_spelling(stem, kind)
    if local:
        if not name_lexicon.is_known(local, kind):
            local = _title(local)
        return _ok(key, local, methods=methods + ["lexicon"], confidence=confidence)
    return _ok(key, _title(value), methods=methods, confidence=confidence)


SEX_FEMALE = {"female", "females", "woman", "women", "girl", "lady", "babae", "babaye", "f", "femail", "feemale"}
SEX_MALE = {"male", "males", "man", "men", "boy", "guy", "lalaki", "lalake", "m", "mail", "mel", "meyl"}


def _handle_sex(key: str, text: str) -> AnswerResult:
    # [a-z']+ keeps "i'm" whole, so its "m" is not read as "male".
    words = set(re.findall(r"[a-z']+", _basic_clean(text).lower()))
    is_f = bool(words & SEX_FEMALE)
    is_m = bool(words & SEX_MALE)
    if is_f and not is_m:
        return _ok(key, "Female")
    if is_m and not is_f:
        return _ok(key, "Male")
    if is_f and is_m:
        return _fail(key, "Please say only one: male or female.")
    return _fail(key, "Sorry, please answer male or female.")


CIVIL_RULES = [
    ("Single", r"\b(?:not married|never married|unmarried|single|walang asawa|wala pang asawa|binata|dalaga|soltera?o?)\b"),
    ("Widowed", r"\b(?:widow(?:ed|er)?|biyud[ao]|byud[ao]|balo|nabalo)\b"),
    ("Separated", r"\b(?:separated|divorced?|annulled|annulment|hiwalay|hiwalay na)\b"),
    ("Married", r"\b(?:married|kasal|may asawa|meron(?:g)? asawa|with (?:a )?(?:husband|wife))\b"),
]


def _handle_civil_status(key: str, text: str) -> AnswerResult:
    t = _basic_clean(text).lower()
    for status, pattern in CIVIL_RULES:
        if re.search(pattern, t):
            return _ok(key, status)
    return _fail(key, "Please answer single, married, widowed, or separated.")


def _handle_date(key: str, text: str) -> AnswerResult:
    d, problem = parse_date(text)
    if d:
        display = format_date(d)
        warnings = []
        age = (date.today() - d).days // 365
        if age < 18:
            warnings.append(f"Applicant would be {age} years old (under 18).")
        return _ok(key, display, display=display, normalized=d.isoformat(), warnings=warnings)
    messages = {
        "no_year": "I didn't hear the year. Please say your full birthday, for example May 10, 1990.",
        "no_day": "I didn't hear the day. Please say your full birthday, for example May 10, 1990.",
        "no_month": "I didn't hear the month. Please say your full birthday, for example May 10, 1990.",
        "future": "That date is in the future. Please say your birthday again.",
        "too_old": "That year seems too far back. Please say your birthday again.",
    }
    return _fail(key, messages.get(problem, "I couldn't understand that date. Please say it like May 10, 1990."),
                 reason="partial" if problem in ("no_year", "no_day", "no_month") else "invalid")


DEMONYMS = {
    "filipino": "Filipino", "filipina": "Filipino", "pilipino": "Filipino", "pilipina": "Filipino",
    "pinoy": "Filipino", "pinay": "Filipino", "philippines": "Filipino", "philippine": "Filipino",
    "american": "American", "usa": "American", "america": "American",
    "chinese": "Chinese", "china": "Chinese", "japanese": "Japanese", "japan": "Japanese",
    "korean": "Korean", "korea": "Korean", "british": "British",
    "australian": "Australian", "australia": "Australian", "canadian": "Canadian", "canada": "Canadian",
    "indian": "Indian", "india": "Indian", "indonesian": "Indonesian", "indonesia": "Indonesian",
    "malaysian": "Malaysian", "malaysia": "Malaysian", "singaporean": "Singaporean",
    "thai": "Thai", "thailand": "Thai", "vietnamese": "Vietnamese", "vietnam": "Vietnamese",
    "taiwanese": "Taiwanese", "taiwan": "Taiwanese", "german": "German", "germany": "German",
    "french": "French", "france": "French", "spanish": "Spanish", "spain": "Spanish",
    "italian": "Italian", "italy": "Italian", "dutch": "Dutch", "swiss": "Swiss",
    "swedish": "Swedish", "norwegian": "Norwegian", "danish": "Danish", "russian": "Russian",
    "irish": "Irish", "mexican": "Mexican", "brazilian": "Brazilian", "saudi": "Saudi",
    "emirati": "Emirati", "israeli": "Israeli", "pakistani": "Pakistani", "bangladeshi": "Bangladeshi",
    "nepalese": "Nepalese", "burmese": "Burmese", "cambodian": "Cambodian", "lao": "Lao",
    "kiwi": "New Zealander", "zealander": "New Zealander", "nigerian": "Nigerian",
    "south african": "South African", "hong konger": "Hong Konger",
}


def _handle_citizenship(key: str, text: str) -> AnswerResult:
    t = _basic_clean(text).lower()
    found: List[str] = []
    for word, label in sorted(DEMONYMS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", t) and label not in found:
            found.append(label)
    if not found:
        # Small mishearings: "filipeno", "pilipeno"
        for word in re.findall(r"[a-z]+", t):
            if len(word) >= 5:
                best = process.extractOne(word, list(DEMONYMS.keys()), scorer=fuzz.ratio)
                if best and best[1] >= 80 and DEMONYMS[best[0]] not in found:
                    found.append(DEMONYMS[best[0]])
    if not found:
        return _fail(key, "Please say your citizenship, for example Filipino.")
    value = " and ".join(found[:2])
    return _ok(key, value)


def _handle_icr(key: str, text: str) -> AnswerResult:
    if _is_blank_answer(text) or re.fullmatch(r"(?:i\s+)?(?:don'?t|do not)\s+have(?: (?:one|it|an icr))?", _squash(text).lower()):
        return _ok(key, "", display="none")
    cleaned = _strip_prefix(_basic_clean(text), [
        r"^(?:yes|yeah|meron|oo)\b[\s,]*", r"^(?:my\s+)?(?:icr|i\s?c\s?r)(?:\s+(?:number|no\.?))?(?:\s+is)?\s*",
        r"^(?:it'?s|it\s+is)\s+",
    ])
    if not cleaned:
        return _fail(key, "Please say your ICR number, or say none.", reason="partial")
    # Read digit by digit so leading zeros survive: "zero one two" -> "012".
    pieces = []
    for word in re.findall(r"[a-z0-9]+", cleaned.lower()):
        if word in UNITS and UNITS[word] < 10:
            pieces.append(str(UNITS[word]))
        elif word in ("dash", "hyphen"):
            continue
        else:
            pieces.append(word)
    value = "".join(pieces).upper()
    if len(value) < 4 or not re.search(r"\d", value) or len(value) > 20:
        return _fail(key, "That doesn't sound like an ICR number. Please say the number, or say none.")
    return _ok(key, value, confidence=0.9)


def _handle_height(key: str, text: str) -> AnswerResult:
    t = numbers_to_digits(_basic_clean(text).lower())
    nums = spoken_numbers(t)
    cm: Optional[float] = None

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:cm|centimet(?:er|re)s?|sentimetro|sentimeters?)\b", t)
    if m:
        cm = float(m.group(1))
    if cm is None:
        m = re.search(r"\b([12](?:\.\d{1,2})?)\s*(?:m|meters?|metres?|metro)\b", t)
        if m:
            cm = float(m.group(1)) * 100
    if cm is None:
        m = re.search(r"\b([3-7])\s*(?:'|feet|foot|ft|talampakan)\s*(?:and\s*)?(\d{1,2})?\s*(?:\"|inches|inch|in|pulgada)?", t)
        if m:
            cm = int(m.group(1)) * 30.48 + int(m.group(2) or 0) * 2.54
    if cm is None:
        m = re.search(r"\b([4-7])\.(\d{1,2})\b", t)   # "5.6" said as five-six = 5 ft 6 in
        if m and int(m.group(2)) <= 11:
            cm = int(m.group(1)) * 30.48 + int(m.group(2)) * 2.54
    if cm is None and len(nums) >= 2 and 3 <= nums[0] <= 7 and 0 <= nums[1] <= 11:
        cm = nums[0] * 30.48 + nums[1] * 2.54    # "five six" = 5 ft 6 in
    if cm is None and _joined_hundreds(nums):
        cm = _joined_hundreds(nums)              # "one sixty" = 160 cm
    if cm is None and nums:
        n = float(nums[0])
        if 50 <= n <= 250:
            cm = n
        elif 1.0 <= n <= 2.5:
            cm = n * 100
        elif 3 <= n <= 7:
            cm = n * 30.48
    if cm is None:
        return _fail(key, "Please say your height in centimeters, or in feet and inches, like 5 feet 4.")
    cm = round(cm)
    if not 50 <= cm <= 250:
        return _fail(key, "That height doesn't seem right. Please say it again, for example 160 centimeters.")
    feet, inches = divmod(round(cm / 2.54), 12)
    return _ok(key, str(cm), display=f"{cm} centimeters ({feet} feet {inches})", normalized=str(cm))


def _handle_weight(key: str, text: str) -> AnswerResult:
    t = numbers_to_digits(_basic_clean(text).lower())
    nums = spoken_numbers(t)
    kg: Optional[float] = None
    amount = _joined_hundreds(nums) or (float(nums[0]) if nums else None)   # "one thirty" = 130
    if amount is not None and re.search(r"\b(?:pounds?|lbs?|libra)\b", t):
        kg = amount * 0.453592
    elif amount is not None:
        kg = amount
    if kg is None:
        return _fail(key, "Please say your weight in kilos, for example 60 kilos.")
    kg = round(kg, 1)
    if not 20 <= kg <= 300:
        return _fail(key, "That weight doesn't seem right. Please say it again in kilos.")
    shown = f"{kg:g}"
    return _ok(key, shown, display=f"{shown} kilos", normalized=shown)


INCOME_NONE = re.compile(
    r"^(?:none|no income|nothing|zero|wala|walang (?:kita|income|trabaho)|unemployed|jobless|no work|"
    r"i don'?t have (?:any )?(?:income|work|a job)|wala po|0)$"
)
# "No, none.", "None, none po", "Wala, wala": only these words, said once or a few times (plan 001.6).
INCOME_NONE_WORDS = {"no", "none", "nope", "nothing", "zero", "wala", "walay", "0"}


def _is_no_income(t: str) -> bool:
    if INCOME_NONE.match(t):
        return True
    words = re.sub(r"[^\w\s']", " ", t).split()
    return bool(words) and all(w in INCOME_NONE_WORDS for w in words)


def _handle_income(key: str, text: str) -> AnswerResult:
    t = _basic_clean(text).lower()
    t = _squash(re.sub(r"\b(?:php|pesos?|piso|peso|p)\b", " ", t))
    if _is_no_income(t):
        return _ok(key, "0", display="no income", normalized="0")
    nums = spoken_numbers(t)
    if not nums:
        return _fail(key, "Please say an amount, for example 120 thousand a year, or 10 thousand a month.")
    amount = float(max(nums))
    period = "year"
    if re.search(r"\b(?:a|per|every|each|kada|isang)\s+(?:month|buwan)\b|\bmonthly\b|\bbuwan[- ]?buwan\b|\bsa isang buwan\b", t):
        period = "month"
    elif re.search(r"\b(?:a|per|every|each|kada)\s+(?:week|linggo)\b|\bweekly\b", t):
        period = "week"
    elif re.search(r"\b(?:a|per|every|each|kada)\s+(?:day|araw)\b|\bdaily\b|\bper day\b", t):
        return _fail(key, "Please tell me your income per month or per year instead of per day.", reason="partial")
    yearly = amount * {"year": 1, "month": 12, "week": 52}[period]
    if yearly < 0 or yearly > 1_000_000_000:
        return _fail(key, "That amount doesn't seem right. Please say it again.")
    yearly = round(yearly, 2)
    shown = f"{yearly:,.0f}" if yearly.is_integer() else f"{yearly:,.2f}"
    display = f"{shown} pesos a year"
    if period != "year":
        display += f" ({amount:,.0f} a {period} times {12 if period == 'month' else 52})"
    return _ok(key, f"{yearly:g}" if not yearly.is_integer() else str(int(yearly)), display=display,
               normalized=str(yearly))


ADDRESS_KEYWORDS = re.compile(
    r"\b(?:purok|sitio|zone|street|st|road|rd|avenue|ave|barangay|brgy|bgy|village|subdivision|subd|"
    r"phase|block|blk|lot|city|municipality|town|province|poblacion|bukidnon|baungon|house)\b"
    r"|\bno\.?\s*\d|\b\d+[a-z]?\s+[a-z]",   # "No. 12", "123 Rizal"
    re.IGNORECASE,
)


def _office_location() -> Tuple[str, str]:
    try:
        from app.services import settings_service
        office = settings_service.get_settings()
        return office.municipality, office.province
    except Exception:
        return settings.MUNICIPALITY, settings.PROVINCE


# How Whisper/Google tend to mishear the address words people say first.
ADDRESS_SOUND_ALIKES = [
    (r"\b(?:purok|purak|purek|purik|puric|puruk|peeric|perok|porok|poroc|purock|pure ok)\b", "Purok"),
    (r"\b(?:barangay|baranggay|barangai|baranay|baranai|baranati|barangey|barrangay|barongay|barana|baranga|"
     r"brgy|bgy)\b\.?", "Barangay"),
    (r"\b(?:sitio|sityo|setio|sicio|seetio)\b", "Sitio"),
]
PUROK_NUMBERS = {"one": 1, "won": 1, "two": 2, "too": 2, "to": 2, "three": 3, "tree": 3, "four": 4, "for": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                 "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
ADDRESS_FILLER_WORDS = {"purok", "sitio", "barangay", "zone", "street", "baungon", "bukidnon", "the", "and",
                        "in", "at", "sa", "of", "city", "municipality"}


def _normalize_address_words(text: str) -> str:
    for pattern, word in ADDRESS_SOUND_ALIKES:
        text = re.sub(pattern, word, text, flags=re.IGNORECASE)
    # Whisper's sentence punctuation inside an address: "Purok, too. Barangay ..." -> "Purok too, Barangay ..."
    text = re.sub(r"\bPurok\s*[,.]\s*", "Purok ", text)
    text = re.sub(r"(?<!\bNo)\.\s+", ", ", text).rstrip(".")
    # "Purok too" / "Purok V" -> "Purok 2" / "Purok 5"
    text = re.sub(r"\b(Purok|Sitio|Zone)\s+([A-Za-z]+)\b",
                  lambda m: f"{m.group(1)} {PUROK_NUMBERS[m.group(2).lower()]}"
                  if m.group(2).lower() in PUROK_NUMBERS else m.group(0), text, flags=re.IGNORECASE)
    return text


def phonetic_key(word: str) -> str:
    """
    Sound-based key for matching local place names that speech engines spell
    differently: vowels dropped, similar consonants merged (m/n, t/d, b/p, k/g/c,
    f/v, s/z). "Imbatug", "Inbaedig" -> "NPTK"; "Salimbalan", "Isolim Balan" -> "SLNPLN".
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    w = re.sub(r"c(?=[eiy])", "s", w).replace("ph", "f").replace("ng", "n")
    groups = {"m": "N", "n": "N", "t": "T", "d": "T", "b": "P", "p": "P", "k": "K", "g": "K", "c": "K", "q": "K",
              "f": "F", "v": "F", "s": "S", "z": "S", "x": "S", "l": "L", "r": "R", "j": "J"}
    key = []
    for ch in w:
        code = groups.get(ch)            # vowels, h, w, y are dropped
        if code and (not key or key[-1] != code):
            key.append(code)
    return "".join(key)


NUMBER_WORD = r"(?:\d+|" + "|".join(sorted({*UNITS, *TENS}, key=len, reverse=True)) + r")"
NUMBERED_BARANGAY = re.compile(rf"\bbarangay\s+({NUMBER_WORD}(?:[\s-]+{NUMBER_WORD})?)\b", re.IGNORECASE)


def _find_barangay(text: str, barangays: Optional[List[str]] = None,
                   aliases: Optional[Dict[str, str]] = None) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Closest barangay: (name, score, the words that matched). Without a list: the office
    municipality's barangays; with one (plan 001.6): only that town's ("Imbatug" of
    Cabanglasan is never taken for Imbatug of Baungon).
    Spelling-based fuzzy matching first, then sound-based (phonetic) matching for
    names a speech engine wrote differently ("in Baedig" -> Imbatug).
    """
    if barangays is None:
        barangays, aliases = load_barangays(), load_aliases()
    aliases = aliases or {}
    # Numbered barangays (Cagayan de Oro's "Barangay 10"): the number said after "Barangay".
    numbered = {b.lower(): b for b in barangays if re.fullmatch(r"barangay\s+\d+", b, re.IGNORECASE)}
    said = NUMBERED_BARANGAY.search(text) if numbered else None
    if said:
        nums = spoken_numbers(said.group(1))
        name = f"barangay {int(nums[0])}" if nums else ""
        if name in numbered:
            return numbered[name], 100.0, said.group(0)
    m = extraction_service.BARANGAY_MENTION_PATTERN.search(text)
    if m:
        found, score = match_barangay(m.group(1).strip(), names=barangays, aliases=aliases)
        if found:
            return found, score, m.group(1).strip()
    names = barangays + list(aliases)
    if not names:
        return None, 0.0, None
    keys = {}
    for name in names:
        keys.setdefault(phonetic_key(name), aliases.get(name, name))

    best: Tuple[Optional[str], float, Optional[str]] = (None, 0.0, None)
    words = re.findall(r"[A-Za-zñÑ]+", text)
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            part = words[i:i + size]
            if part[0].lower() in ADDRESS_FILLER_WORDS and size > 1:
                part = part[1:]          # "in Baedig" -> "Baedig" is tried too
            if not part or any(w.lower() in ADDRESS_FILLER_WORDS for w in part[1:]):
                continue
            full = " ".join(words[i:i + size])
            matched = " ".join(part)
            # (text to compare, the words it came from)
            for chunk, span in ((matched, matched), (matched.replace(" ", ""), matched),
                                (full.replace(" ", ""), full)):
                if len(chunk) < 4:
                    continue
                res = process.extractOne(chunk, names, scorer=fuzz.ratio)
                if res and res[1] > best[1]:
                    best = (aliases.get(res[0], res[0]), float(res[1]), span)
                key = phonetic_key(chunk)
                if len(key) >= 3 and key in keys and best[1] < 85:
                    best = (keys[key], 85.0, span)
    # Longer names tolerate more mishearing ("Imbatog", "Lingatin").
    needed = 80 if best[0] and len(best[0]) >= 6 else 88
    return best if best[1] >= needed else (None, best[1], None)


def _handle_address(key: str, text: str) -> AnswerResult:
    cleaned = _strip_prefix(_basic_clean(text), [
        r"^(?:my\s+)?(?:complete\s+|home\s+)?address(?:\s+is)?\s*:?\s*", r"^i\s+live\s+(?:in|at)\s+",
        r"^(?:nakatira\s+ako\s+sa|taga|sa)\s+",
    ])
    if not cleaned or len(re.sub(r"[^A-Za-z]", "", cleaned)) < 3:
        return _fail(key, "Please tell me your address: your purok or street, and your barangay.")
    cleaned = _normalize_address_words(cleaned)
    methods = ["regex", "gazetteer"]
    entities = _ner_entities(_title(cleaned))
    locations = []
    if entities is not None:
        methods.insert(0, "distilbert")
        locations = [e.word for e in entities if e.label == "LOC" and e.score >= 0.5]

    # The town first (plan 001.6): any explicitly named town supplies its own barangays and province.
    town = _find_address_town(cleaned)
    if town:
        return _address_in_town(key, cleaned, town, methods)

    barangay, score, heard_span = _find_barangay(cleaned)
    if not (barangay or locations or ADDRESS_KEYWORDS.search(cleaned) or "," in cleaned):
        return _fail(key, "I didn't get an address from that. Please say your purok or street, and your barangay.",
                     methods=methods)

    value = _title(cleaned)
    value = re.sub(r"\b(?:Brgy|Bgy)\.?\b", "Barangay", value)
    if barangay:
        if not re.search(rf"\b{re.escape(barangay)}\b", value, re.IGNORECASE):
            # Replace the misheard words ("In Baedig") with the real barangay name,
            # together with a "Barangay"/"in"/"sa" said just before them.
            span = _title(heard_span) if heard_span else None
            pattern = rf"(?:\bBarangay\s+)?(?:\b(?:In|At|Sa)\s+)?{re.escape(span)}" if span else None
            if pattern and re.search(pattern, value):
                value = re.sub(pattern, f"Barangay {barangay}", value, count=1)
            else:
                value = f"{value}, Barangay {barangay}"
        municipality, province = _office_location()
        if municipality and municipality.lower() not in value.lower():
            value = f"{value}, {municipality}"
        if province and province.lower() not in value.lower():
            value = f"{value}, {province}"
    value = _squash(value).strip(" ,")
    if barangay:
        return _ok(key, value, methods=methods, confidence=0.95)
    # No barangay of the municipality recognized: likely misheard, so ask instead of saving it as heard.
    return _ok(key, value, methods=methods, confidence=0.8 if locations else 0.6, needs_confirmation=True,
               message=f"I heard {value}. Is that your complete address?",
               warnings=["Barangay not recognized — please check the address."])


# ─── Address in parts: town or city, barangay, purok or street (plan 001.6) ─────

def _office_town() -> Optional[place_gazetteer.Place]:
    """The office's own city or municipality (Settings), as a PSGC place."""
    municipality, province = _office_location()
    found = place_gazetteer.find_town(f"{municipality}, {province}", home_province=province)
    return found["place"] if found else None


def _office_barangay_words() -> frozenset:
    """The office town's barangays and their aliases, normalised: never taken for a town of the same name."""
    return frozenset(place_gazetteer.normalise(n) for n in load_barangays() + list(load_aliases()))


def _find_address_town(text: str) -> Optional[dict]:
    return place_gazetteer.find_town(text, exclude=_office_barangay_words(), home_province=_office_location()[1])


def _town_value(place: place_gazetteer.Place) -> str:
    """"Cagayan de Oro City, Misamis Oriental": the town with its OWN province (from the Region X list there)."""
    province = (region_town(place.psgc) or {}).get("province") or place.province
    return place.display if not province or province == place.display else f"{place.display}, {province}"


def _barangay_label(name: str) -> str:
    """"Carmen" -> "Barangay Carmen"; "Barangay 10" stays as it is."""
    return name if re.match(r"barangay\b", name, re.IGNORECASE) else f"Barangay {name}"


def _join_address(*parts: str) -> str:
    joined = ", ".join(p.strip(" ,.;") for p in parts if p and p.strip(" ,.;"))
    return _squash(re.sub(r"\s*,(?:\s*,)+\s*", ", ", joined))


def _parse_address(cleaned: str, town: Optional[dict]) -> dict:
    """
    Split a one-sentence address into its parts, the town found first:
    {"place", "ambiguous", "barangay" (or None), "rest" (purok / street, as said)}.
    Without a town, the office town is assumed.
    """
    place = town["place"] if town else _office_town()
    before = cleaned[:town["span"][0]] if town else cleaned
    before = re.sub(r"(?:[\s,]+(?:city|municipality|town|of|in|at|sa))+\s*$", "", f" {before}",
                    flags=re.IGNORECASE).strip(" ,.;")
    names, aliases = town_barangays(place.psgc) if place else ([], {})
    barangay, heard = None, None
    if names and before:
        barangay, _, heard = _find_barangay(before, names, aliases)
    rest = before
    if barangay and heard:
        words = r"[\s-]+".join(re.escape(w) for w in heard.split())
        rest = re.sub(rf"(?:\bBarangay\s+)?(?:\b(?:In|At|Sa)\s+)?{words}", " ", before, count=1, flags=re.IGNORECASE)
    rest = re.sub(r"\b(?:Brgy|Bgy)\.?(?=\s|$)", "Barangay", _title(_squash(rest).strip(" ,.;")))
    rest = re.sub(r"(?:,\s*)?\bBarangay\s*$", "", rest).strip(" ,")
    return {"place": place, "ambiguous": bool(town and town["ambiguous"]), "barangay": barangay, "rest": rest}


def split_address(text: str) -> dict:
    """
    An address's parts, for scoring (eval/run_eval.py) and checks:
    {"town_psgc", "town", "town_said", "barangay", "rest"}. town_said is False when no town was named and the
    office town was assumed. Uses the same town + barangay matching as the interview.
    """
    cleaned = _normalize_address_words(_basic_clean(text))
    town = _find_address_town(cleaned)
    parts = _parse_address(cleaned, town)
    place = parts["place"]
    return {"town_psgc": place.psgc if place and not parts["ambiguous"] else "",
            "town": place.display if place else "", "town_said": town is not None,
            "barangay": parts["barangay"] or "", "rest": parts["rest"]}


def _address_in_town(key: str, cleaned: str, town: dict, methods: List[str]) -> AnswerResult:
    """A one-sentence address with an explicitly named town."""
    parts = _parse_address(cleaned, town)
    place = parts["place"]
    town_value = place.name if parts["ambiguous"] else _town_value(place)
    barangay = parts["barangay"]
    value = _join_address(parts["rest"], _barangay_label(barangay) if barangay else "", town_value)
    if barangay:
        return _ok(key, value, methods=methods, confidence=0.95)
    warning = ("Barangay not recognized — please check the address." if region_town(place.psgc)
               else f"No barangay list for {place.display} (outside Region X) — please check the address.")
    return _ok(key, value, methods=methods, confidence=0.6, needs_confirmation=True,
               message=f"I heard {value}. Is that your complete address?", warnings=[warning])


def _handle_address_town(key: str, text: str, spoken: bool = False) -> AnswerResult:
    """"Which town or city do you live in?" PSGC towns, Region X first. normalized = the town's PSGC code."""
    cleaned = _strip_prefix(_basic_clean(text), [
        r"^(?:my\s+)?(?:town|city|municipality|home\s?town)(?:\s+(?:is|ay|ko))?\s*:?\s*",
        r"^i\s+live\s+(?:in|at)\s+", r"^(?:nakatira\s+ako\s+sa|taga|sa|in|at)\s+",
        r"^(?:the\s+)?(?:city|municipality|town)\s+of\s+",
    ])
    if not cleaned or len(re.sub(r"[^A-Za-z]", "", cleaned)) < 3:
        return _fail(key, "Please tell me the town or city where you live.")
    methods = ["regex", "gazetteer"]
    town = _find_address_town(_normalize_address_words(cleaned))
    if town and not town["ambiguous"]:
        value = _town_value(town["place"])
        return _ok(key, value, normalized=town["place"].psgc, methods=methods, confidence=0.95)
    if town:
        # The same town name in several provinces outside Region X, and no province said.
        name = town["place"].name
        return _ok(key, name, normalized="", methods=methods, confidence=0.6, needs_confirmation=True,
                   message=f"I heard {name}. Is that your town or city?")
    office = _office_town()
    if spoken and office:
        # Nothing matched: most applicants live in the office's own town, so offer it for a yes.
        value = _town_value(office)
        return _ok(key, value, normalized=office.psgc, methods=methods, confidence=0.4, needs_confirmation=True,
                   message=f"I didn't catch the town. Do you live in {value}?",
                   warnings=["Town not recognized; the office town was offered."])
    value = _title(cleaned)
    return _ok(key, value, normalized="", methods=methods, confidence=0.5, needs_confirmation=True,
               message=f"I heard {value}. Is that your town or city?")


def _handle_address_barangay(key: str, text: str, town: Optional[str] = None) -> AnswerResult:
    """"What is your barangay?" Matched only against the barangays of `town` (a PSGC code; default: office)."""
    cleaned = _strip_prefix(_basic_clean(text), [
        r"^my\s+barangay(?:\s+(?:is|ay))?\s*:?\s*", r"^barangay\s+ko\s+(?:ay\s+)?",
        r"^i\s+live\s+(?:in|at)\s+", r"^(?:nakatira\s+ako\s+sa|taga)\s+",
    ])
    if not re.search(r"[A-Za-z0-9]", cleaned or ""):
        return _fail(key, "Please tell me your barangay.")
    cleaned = _normalize_address_words(cleaned)
    stripped = re.sub(r"^(?:in|at|sa)\s+", "", cleaned, flags=re.IGNORECASE)
    if spoken_numbers(stripped) and not re.search(r"[A-Za-z]{3,}", re.sub(NUMBER_WORD, " ", stripped, flags=re.I)):
        stripped = f"Barangay {stripped}"          # "Ten" for "Barangay 10"
    methods = ["regex", "gazetteer"]
    office = _office_town()
    psgc = town or (office.psgc if office else "")
    names, aliases = town_barangays(psgc)
    if names:
        # Keep "in" for sound-alike matching ("in Baedig" -> Imbatug) before trying it as a prefix.
        found, _, _ = _find_barangay(cleaned, names, aliases)
        if not found:
            found, _, _ = _find_barangay(stripped, names, aliases)
        if found:
            return _ok(key, found, display=_barangay_label(found), methods=methods, confidence=0.95)
    heard = re.sub(r"^(?:barangay|brgy|bgy)\.?\s+", "", _title(stripped), flags=re.IGNORECASE).strip(" ,.")
    if not heard or heard.lower() == "barangay":
        return _fail(key, "Please tell me the name of your barangay.")
    label = _barangay_label(heard)
    town_name = (region_town(psgc) or {}).get("town", "")
    warning = (f"Not a barangay of {town_name} — please check it." if names
               else "No barangay list for this town (outside Region X) — please check it.")
    return _ok(key, heard, display=label, methods=methods, confidence=0.5, needs_confirmation=True,
               message=f"I heard {label}. Is that your barangay?", warnings=[warning])


PUROK_LIST_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "office_puroks.csv")
PUROK_PART = re.compile(r"^(?:Purok|Zone|Sitio|Prk|Block|Blk|Lot|Phase)\b\.?\s*\S", re.IGNORECASE)
PUROK_EXAMPLE = "Please say it like: Purok 2, Zone 1 or Sitio Mahogany. Or say none."
TALK_STARTS = {"i", "we", "you", "he", "she", "it", "they", "my", "ako", "the", "a"}
COUNT_WORDS = {"years", "year", "old", "children", "kids", "pesos", "kilos", "times", "months", "days", "people"}


def _purok_names() -> List[str]:
    """The office barangay's puroks (optional app/data/office_puroks.csv, one `name` column); [] without it."""
    try:
        with open(PUROK_LIST_FILE, newline="", encoding="utf-8") as f:
            return [r["name"].strip() for r in csv.DictReader(f) if (r.get("name") or "").strip()]
    except (OSError, KeyError):
        return []


def _handle_address_purok(key: str, text: str) -> AnswerResult:
    """
    "What is your purok, sitio or street?" Purok / zone / sitio + a number or name, a street or house
    number, or a purok of the office list. Other talk ("Sounds good") is refused (plan 001.8).
    """
    if _is_blank_answer(text):
        return _ok(key, "", display="none")
    cleaned = _basic_clean(text)
    said = re.match(r"^(?:my\s+)?(purok|sitio|zone|street)(?:\s+ko)?\s+(?:is|ay)\s+(.+)$", cleaned, re.IGNORECASE)
    if said:
        cleaned = f"{said.group(1)} {said.group(2)}"
    cleaned = _strip_prefix(cleaned, [r"^i\s+live\s+(?:in|at)\s+", r"^(?:nakatira\s+ako\s+sa|sa|in|at)\s+"])
    cleaned = _normalize_address_words(cleaned)
    number = re.fullmatch(r"(?:number\s+)?([A-Za-z]+|\d+[A-Za-z]?)", cleaned, re.IGNORECASE)
    if number and (number.group(1)[0].isdigit() or number.group(1).lower() in PUROK_NUMBERS):
        n = number.group(1)
        cleaned = f"Purok {PUROK_NUMBERS.get(n.lower(), n)}"     # "Two" / "2" alone: Purok 2
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return _fail(key, "Please tell me your purok, sitio or street, or say none.")
    value = re.sub(r"\b(?:Brgy|Bgy)\.?(?=\s|$)", "Barangay", _title(cleaned))
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", value)}
    # A purok/zone/sitio, a street, or a house number with a short name ("123 Mahogany"); not any talk with a
    # number in it ("I have 2 children").
    # A street needs a name before the street word and no "I ..." talk ("Rizal Street", not "I drive a car").
    tokens = [w.lower() for w in re.findall(r"[A-Za-z]+", value)]
    street = (len(tokens) <= 6 and bool(tokens) and tokens[0] not in TALK_STARTS
              and any(w in place_gazetteer.STREET_WORDS for w in tokens[1:]))
    house = (re.fullmatch(r"(?:No\.?\s*)?\d+[A-Za-z]?(?:\s+[A-Za-z]+){1,2}", value)
             and not words & COUNT_WORDS)                          # "123 Mahogany", not "20 years old"
    if PUROK_PART.match(value) or house or street:
        return _ok(key, value, methods=["regex"], confidence=0.9)
    names = _purok_names()
    if names:
        found, _, _ = _find_barangay(cleaned, names, {})
        if found:
            name = found if PUROK_PART.match(found) else f"Purok {found}"
            return _ok(key, name, methods=["regex", "gazetteer"], confidence=0.9)
    return _fail(key, PUROK_EXAMPLE)


BIRTH_LEAD_INS = [
    r"^(?:i\s+was\s+)?born\s+(?:in|at)\s+", r"^(?:my\s+)?place\s+of\s+birth(?:\s+is)?\s*",
    r"^(?:ipinanganak\s+ako\s+sa|pinanganak\s+ako\s+sa|natawo\s+ko\s+sa|taga|sa)\s+", r"^(?:in|at)\s+",
    r"^(?:the\s+)?(?:province|probinsya|city|town|municipality)\s+of\s+",
]


def _province_said(text: str) -> bool:
    """"Cagayan province", "the province of Cagayan": a province is meant, not a city named like it."""
    return bool(re.search(r"\b(?:province|probinsya)\b", text or "", re.IGNORECASE))


def _birth_cleaned(text: str) -> str:
    return _strip_prefix(_basic_clean(text), BIRTH_LEAD_INS)


def _birth_suggestions(places) -> List[dict]:
    return [{"value": p.display, "province": p.psgc} if p.type == "province"
            else {"value": place_gazetteer.place_value(p), "province": ""} for p in places]


def _birth_ok(key: str, found: dict) -> AnswerResult:
    """A whole place: "Malaybalay City, Bukidnon" (normalized ""). A province only: normalized = its PSGC code."""
    methods = ["regex", "gazetteer"]
    if "place" in found:
        return _ok(key, place_gazetteer.place_value(found["place"]), normalized="", methods=methods, confidence=0.95)
    province = found["province"]
    return _ok(key, province.display, normalized=province.psgc, methods=methods, confidence=0.95)


def _handle_birth_province(key: str, text: str) -> AnswerResult:
    """
    "Which province were you born in?" (plan 001.8). A province (the town is asked next), or a whole
    place said at once: a city with no province ("Iligan"), or "Malaybalay, Bukidnon". Talk that names
    no place is refused, never read back as one.
    """
    cleaned = _birth_cleaned(text)
    home = _office_location()[1]
    likely = place_gazetteer.nearby(home)
    if not cleaned or len(re.sub(r"[^A-Za-z]", "", cleaned)) < 3:
        result = _fail(key, "Please tell me the province where you were born.")
        result.suggestions = _birth_suggestions(likely[:3])
        return result
    found = place_gazetteer.find_birth_place(cleaned, home_province=home, province_said=_province_said(text))
    if found:
        return _birth_ok(key, found)
    same_name = place_gazetteer.places_named(cleaned, home_province=home)
    if same_name:
        # "San Jose": a town of that name in many provinces. The closest ones to tap; or say the province.
        result = _fail(key, f"There is more than one {same_name[0].display}. Please say the province first.",
                       methods=["regex", "gazetteer"])
        result.suggestions = _birth_suggestions(same_name[:3])
        return result
    pool = place_gazetteer.provinces() + place_gazetteer.places_without_province()
    result = _fail(key, "Please say the province where you were born. For example: Bukidnon. "
                        "If you were born in a city like Cagayan de Oro, say the city.", methods=["regex", "gazetteer"])
    result.suggestions = _birth_suggestions(place_gazetteer.suggest(cleaned, pool, fill=likely))
    return result


def _handle_birth_town(key: str, text: str, province: Optional[str] = None) -> AnswerResult:
    """"Which town or city in Bukidnon?" Matched inside that province first (near-misses too), then anywhere."""
    prov = place_gazetteer.province_by_code(province or "")
    if not prov:
        return _handle_birth_province(key, text)
    cleaned = _birth_cleaned(text)
    towns = place_gazetteer.towns_in_province(prov.psgc)
    found = None
    if cleaned and len(re.sub(r"[^A-Za-z]", "", cleaned)) >= 3:
        found = place_gazetteer.find_birth_place(cleaned, province_code=prov.psgc, home_province=prov.name,
                                                 province_said=_province_said(text))
    if found and found.get("province") == prov:
        found = None                       # only the province again: the town is still needed
    if found:
        return _birth_ok(key, found)
    example = towns[0].name if towns else ""
    result = _fail(key, f"Please say the town or city in {prov.display} where you were born."
                        + (f" For example: {example}." if example else ""), methods=["regex", "gazetteer"])
    result.suggestions = _birth_suggestions(place_gazetteer.suggest(cleaned, towns, fill=towns))
    return result


def _handle_place_of_birth(key: str, text: str, spoken: bool = False) -> AnswerResult:
    """
    The place of birth in one answer (the manual page, the review, the eval). Spoken: only a real place
    (PSGC list) is taken; talk Whisper heard ("Sounds good") is refused. Typed: kept as typed if it is not
    in the list, and marked so.
    """
    cleaned = _birth_cleaned(text)
    if not cleaned or len(re.sub(r"[^A-Za-z]", "", cleaned)) < 3:
        return _fail(key, "Please tell me the town or city where you were born.")
    if re.fullmatch(r"[\d\s.,/-]+", cleaned):
        return _fail(key, "Please say the name of the town or city where you were born.")
    methods = ["regex", "gazetteer"]
    # Every Philippine province, city and municipality (PSGC): written the official way,
    # with its province ("polo molok" -> "Polomolok, South Cotabato").
    home_province = _office_location()[1]
    found = place_gazetteer.find_birth_place(cleaned, home_province=home_province, province_said=_province_said(text))
    if found:
        return _ok(key, _birth_ok(key, found).value, methods=methods, confidence=0.95)
    place = place_gazetteer.find_place(cleaned, home_province=home_province)
    if place:                       # the same town name in several provinces, none said: the town only
        return _ok(key, place["value"], methods=methods, confidence=0.95)
    pool = place_gazetteer.provinces() + place_gazetteer.places_without_province()
    if spoken:
        result = _fail(key, "Please say the town or city and the province where you were born, "
                            "for example: Malaybalay, Bukidnon.", methods=methods)
        result.suggestions = _birth_suggestions(place_gazetteer.suggest(cleaned, pool,
                                                                        fill=place_gazetteer.nearby(home_province)))
        return result
    entities = _ner_entities(_title(cleaned))
    locations = []
    if entities is not None:
        methods.insert(0, "distilbert")
        locations = [e.word for e in entities if e.label == "LOC" and e.score >= 0.5]
    barangay = _find_barangay(cleaned)[0]
    value = _title(cleaned)
    warning = "Typed place of birth, not in the list of Philippine places — please check it."
    if locations or barangay:
        return _ok(key, value, methods=methods, confidence=0.9, warnings=[warning])
    # Typed, not recognized as a place: kept as typed, but asked to be checked; the closest list entries to tap.
    result = _ok(key, value, methods=methods, confidence=0.5, needs_confirmation=True,
                 message=f"{value} is not in the list of places. Is that the place where you were born?",
                 warnings=[warning])
    all_towns = tuple(p for ps in place_gazetteer._load()[0].values() for p in ps if p.type != "province")
    result.suggestions = _birth_suggestions(place_gazetteer.suggest(cleaned, all_towns + pool))
    return result


OCCUPATION_ALIASES = {
    "Unemployed": r"^(?:wala|none|nothing|no)$|\b(?:unemployed|no work|no job|jobless|wala(?:ng)? trabaho|walang work)\b",
    "Housewife": r"\b(?:house ?wife|homemaker|home maker|plain housewife|maybahay|ina ng tahanan)\b",
    "Student": r"\b(?:student|estudyante|studying|nag-?aaral)\b",
    "Retired": r"\b(?:retired|retiree|pensioner|pensiyonado)\b",
    "Self-employed": r"\b(?:self[- ]?employed|own business|negosyo|businessman|businesswoman|entrepreneur)\b",
}
COMMON_OCCUPATIONS = [
    "farmer", "fisherman", "fisher", "teacher", "driver", "tricycle driver", "jeepney driver",
    "habal-habal driver", "motorcycle driver", "delivery rider", "vendor", "fish vendor", "market vendor",
    "store owner", "sari-sari store owner", "carpenter", "mason", "laborer", "construction worker",
    "nurse", "midwife", "doctor", "engineer", "ofw", "seaman", "seafarer", "security guard", "police officer",
    "soldier", "barangay tanod", "barangay official", "barangay health worker", "government employee",
    "clerk", "office clerk", "cashier", "saleslady", "salesman", "sales clerk", "call center agent",
    "programmer", "accountant", "lawyer", "mechanic", "electrician", "plumber", "welder", "tailor",
    "dressmaker", "beautician", "barber", "cook", "chef", "waiter", "waitress", "housekeeper",
    "domestic helper", "kasambahay", "caregiver", "janitor", "utility worker", "baker", "butcher",
    "painter", "musician", "pastor", "priest", "tour guide", "resort staff", "hotel staff", "receptionist",
    "pharmacist", "dentist", "social worker", "encoder", "secretary", "manager", "supervisor",
    "contractor", "foreman", "coconut farmer", "copra maker", "harvester", "laundry woman", "labandera",
    "tindera", "tindero", "magsasaka", "mangingisda", "guro", "drayber", "karpintero", "panadero",
    "manicurist", "massage therapist", "boatman", "porter", "bookkeeper", "librarian", "technician",
    "employee", "worker", "staff", "helper", "agent", "officer", "assistant", "operator", "attendant",
]


def _handle_occupation(key: str, text: str) -> AnswerResult:
    cleaned = _strip_prefix(_basic_clean(text), [
        r"^(?:my\s+)?(?:occupation|job|work|trabaho)(?:\s+ko)?(?:\s+(?:is|ay))?\s*",
        r"^(?:i\s+work\s+as|i'?m|i\s+am|working\s+as|nagtatrabaho\s+(?:ako\s+)?(?:bilang|as)|ako\s+ay)\s+",
        r"^(?:a|an|isang)\s+",
    ])
    t = cleaned.lower()
    for label, pattern in OCCUPATION_ALIASES.items():
        if re.search(pattern, t):
            return _ok(key, label)
    if not cleaned or len(re.sub(r"[^A-Za-z]", "", cleaned)) < 2:
        return _fail(key, "Please tell me your work or occupation. If you have none, say unemployed.")
    if re.fullmatch(r"[\d\s.,]+", cleaned) or len(cleaned) > 60:
        return _fail(key, "Please say your occupation, for example farmer or teacher.")
    value = _title(cleaned)
    known = any(re.search(rf"\b{re.escape(o)}\b", t) for o in COMMON_OCCUPATIONS)
    if not known:
        best = process.extractOne(t, COMMON_OCCUPATIONS, scorer=fuzz.WRatio)
        known = bool(best and best[1] >= 90)
    if known:
        return _ok(key, value, confidence=0.9)
    return _ok(key, value, confidence=0.5, needs_confirmation=True,
               message=f"I heard {value}. Is that your occupation?")


# ─── Dispatcher ───────────────────────────────────────────────────────────────

FIELD_LABELS = {
    "surname": "surname", "given_name": "given name", "middle_name": "middle name",
    "address": "address", "date": "date of birth", "sex": "sex", "civil_status": "civil status",
    "citizenship": "citizenship", "icr_no": "ICR number", "place_of_birth": "place of birth",
    "height": "height", "weight": "weight", "occupation": "occupation",
    "gross_annual_income": "gross annual income",
    "address_town": "town or city", "address_barangay": "barangay", "address_purok": "purok or street",
    "place_of_birth_province": "province of birth", "place_of_birth_town": "town or city of birth",
}
OPTIONAL_FIELDS = {"middle_name", "icr_no", "address_purok"}

HANDLERS: Dict[str, Callable[..., AnswerResult]] = {
    "surname": lambda k, t, spoken=False: _handle_name(k, t, "surname", optional=False, spoken=spoken),
    "given_name": lambda k, t, spoken=False: _handle_name(k, t, "given name", optional=False, spoken=spoken),
    "middle_name": lambda k, t, spoken=False: _handle_name(k, t, "middle name", optional=True, spoken=spoken),
    "address": _handle_address,
    "date": _handle_date,
    "sex": _handle_sex,
    "civil_status": _handle_civil_status,
    "citizenship": _handle_citizenship,
    "icr_no": _handle_icr,
    "place_of_birth": _handle_place_of_birth,
    "height": _handle_height,
    "weight": _handle_weight,
    "occupation": _handle_occupation,
    "gross_annual_income": _handle_income,
}


# The voice interview asks the address in 3 short questions (plan 001.6); the web joins the answers into
# the one `address` value. Not in HANDLERS: understand_answers() checks the joined address instead.
ADDRESS_PART_HANDLERS: Dict[str, Callable[..., AnswerResult]] = {
    "address_town": _handle_address_town,
    "address_barangay": _handle_address_barangay,
    "address_purok": _handle_address_purok,
    # The place of birth in 2 steps (plan 001.8); the web keeps the result as the one `place_of_birth`.
    "place_of_birth_province": _handle_birth_province,
    "place_of_birth_town": _handle_birth_town,
}

NAME_KEYS = {"surname", "given_name", "middle_name"}


def understand_answer(field_key: str, transcript: str, spoken: bool = False, town: Optional[str] = None,
                      province: Optional[str] = None) -> AnswerResult:
    """
    Interpret one interview answer for one field (see module docstring).
    spoken=True: the text is Whisper's transcript, so a name may also be rejoined
    ("Da Gondon") and given its local spelling ("Bakulio" -> "Baculio"), and a
    place of birth must be a real place. Typed text is kept as typed.
    town: the PSGC code of the town already answered, for "address_barangay".
    province: the PSGC code of the province of birth already answered, for "place_of_birth_town".
    """
    handler = HANDLERS.get(field_key) or ADDRESS_PART_HANDLERS.get(field_key)
    if handler is None:
        raise KeyError(field_key)
    text = _squash(transcript)
    label = FIELD_LABELS[field_key]
    if not text:
        if field_key in OPTIONAL_FIELDS:
            return _ok(field_key, "", display="none")
        result = _fail(field_key, f"I didn't hear an answer. Please tell me your {label}.", reason="empty")
        if field_key == "place_of_birth_province":     # the likeliest places to tap (plan 001.8)
            result.suggestions = _handle_birth_province(field_key, "").suggestions
        elif field_key == "place_of_birth_town":
            result.suggestions = _handle_birth_town(field_key, "", province=province).suggestions
        return result
    if _is_unclear(text):
        return _fail(field_key, "No problem, let me ask again.", reason="unclear")
    try:
        if field_key in NAME_KEYS or field_key in ("address_town", "place_of_birth"):
            return handler(field_key, text, spoken=spoken)
        if field_key == "address_barangay":
            return handler(field_key, text, town=town)
        if field_key == "place_of_birth_town":
            return handler(field_key, text, province=province)
        return handler(field_key, text)
    except Exception as exc:  # never break the interview on a parsing bug
        logger.exception(f"Answer parsing failed for {field_key}: {exc}")
        return _fail(field_key, f"Sorry, I couldn't understand that. Please say your {label} again.")


def understand_answers(answers: Dict[str, str]) -> Dict[str, AnswerResult]:
    """Check every field at once (used before issuing, after manual edits)."""
    return {key: understand_answer(key, answers.get(key) or "") for key in HANDLERS}
