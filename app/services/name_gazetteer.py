"""
app/services/name_gazetteer.py
Loads known local surnames, given names, and middle names for the
deployment area and fuzzy-matches misheard names against them — the
same idea as barangay_gazetteer.py, applied to people's names instead
of barangay names.

Source data: a resident list CSV (id, province, municipality, barangay,
purok_zone, sitio, surname, name, middle_name). Only the name columns
are used here — this file doesn't care about the address columns.

Requires: pip install rapidfuzz
"""
import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

# Path to the CSV. Put mambajao_fictional_residents_1000.csv at app/data/
NAME_GAZETTEER_CSV_PATH = Path(__file__).parent.parent / "data" / "mambajao_fictional_residents_1000.csv"

# Below this fuzzy-match score (0-100), we don't trust the correction.
MATCH_THRESHOLD = 80  # Higher than barangay's 70 — names are shorter and
                       # more varied, so a low-confidence "correction" is
                       # more likely to force a wrong name onto someone.


@lru_cache(maxsize=1)
def _load_name_lists(csv_path: Optional[str] = None) -> dict:
    """
    Load unique surnames, given names, and middle names from the resident
    CSV once and cache them. Returns a dict of three lists.
    """
    path = Path(csv_path) if csv_path else NAME_GAZETTEER_CSV_PATH
    surnames, given_names, middle_names = set(), set(), set()

    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("surname", "").strip():
                    surnames.add(row["surname"].strip())
                if row.get("name", "").strip():
                    given_names.add(row["name"].strip())
                if row.get("middle_name", "").strip():
                    middle_names.add(row["middle_name"].strip())
    except FileNotFoundError:
        logger.error(f"Name gazetteer CSV not found at {path}. "
                     f"Name fuzzy-matching will be disabled.")

    return {
        "surnames": sorted(surnames),
        "given_names": sorted(given_names),
        "middle_names": sorted(middle_names),
    }


def load_surnames() -> list[str]:
    return _load_name_lists()["surnames"]


def load_given_names() -> list[str]:
    return _load_name_lists()["given_names"]


def load_middle_names() -> list[str]:
    return _load_name_lists()["middle_names"]


def _fuzzy_match(heard_text: str, candidates: list[str], threshold: int) -> tuple[Optional[str], float]:
    if not heard_text or not candidates:
        return None, 0.0
    result = process.extractOne(heard_text, candidates, scorer=fuzz.WRatio)
    if result is None:
        return None, 0.0
    match, score, _ = result
    if score >= threshold:
        return match, score
    return None, score


def match_surname(heard_text: str, threshold: int = MATCH_THRESHOLD) -> tuple[Optional[str], float]:
    """Find the closest known surname. Returns (match_or_None, score)."""
    return _fuzzy_match(heard_text, load_surnames(), threshold)


def match_given_name(heard_text: str, threshold: int = MATCH_THRESHOLD) -> tuple[Optional[str], float]:
    """Find the closest known given name. Returns (match_or_None, score)."""
    return _fuzzy_match(heard_text, load_given_names(), threshold)


def match_middle_name(heard_text: str, threshold: int = MATCH_THRESHOLD) -> tuple[Optional[str], float]:
    """Find the closest known middle name. Returns (match_or_None, score)."""
    return _fuzzy_match(heard_text, load_middle_names(), threshold)


def get_surname_list_for_prompt(limit: int = 40) -> str:
    """
    Comma-separated sample of surnames, used to bias Whisper's decoding
    toward local names. Capped at `limit` since Whisper prompts work
    best short — the full 50 is fine here, but this keeps it safe if
    your real resident list grows much larger later.
    """
    surnames = load_surnames()[:limit]
    return ", ".join(surnames) if surnames else ""


def get_given_name_list_for_prompt(limit: int = 40) -> str:
    """Comma-separated sample of given names, for Whisper's prompt."""
    given_names = load_given_names()[:limit]
    return ", ".join(given_names) if given_names else ""