"""
app/services/barangay_gazetteer.py
Loads the known list of barangays for the deployment municipality and
provides fuzzy-matching so a misheard barangay name (e.g. Whisper writes
"Imbatog" or "Salim Balan") is corrected to the closest real barangay
instead of being stored as garbage text.

Data: app/data/baungon_bukidnon_barangays.csv (16 barangays of Baungon,
Bukidnon; see app/data/README.md for sources and how to add puroks/sitios).

Requires: pip install rapidfuzz
"""
import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

GAZETTEER_CSV_PATH = Path(__file__).parent.parent / "data" / "baungon_bukidnon_barangays.csv"

# Below this fuzzy-match score (0-100), we don't trust the correction
# and fall back to the raw text instead of silently rewriting it.
MATCH_THRESHOLD = 70


@lru_cache(maxsize=1)
def _load(csv_path: Optional[str] = None) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    """(barangay names, alias -> barangay, barangay -> puroks/sitios), read once and cached."""
    path = Path(csv_path) if csv_path else GAZETTEER_CSV_PATH
    barangays: list[str] = []
    aliases: dict[str, str] = {}
    puroks: dict[str, list[str]] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                name = (row.get("barangay") or "").strip()
                if not name:
                    continue
                if name not in barangays:
                    barangays.append(name)
                for alias in (row.get("aliases") or "").split(";"):
                    if alias.strip():
                        aliases[alias.strip()] = name
                purok = (row.get("purok") or "").strip()
                if purok:
                    puroks.setdefault(name, []).append(purok)
    except FileNotFoundError:
        logger.error(f"Barangay gazetteer CSV not found at {path}. Barangay fuzzy-matching will be disabled.")
    return barangays, aliases, puroks


def load_barangays(csv_path: Optional[str] = None) -> list[str]:
    """Unique barangay names of the municipality."""
    return list(_load(csv_path)[0])


def load_aliases() -> dict[str, str]:
    """Other names people use for a barangay (e.g. "Poblacion" -> "Imbatug")."""
    return dict(_load()[1])


def load_puroks(barangay: str) -> list[str]:
    """Puroks/sitios of a barangay, if they were added to the CSV."""
    return list(_load()[2].get(barangay, []))


def match_barangay(heard_text: str, threshold: int = MATCH_THRESHOLD) -> tuple[Optional[str], float]:
    """
    Given a raw span of text that's supposed to contain a barangay name,
    find the closest real barangay (aliases included) from the gazetteer.

    Returns:
        (matched_barangay_or_None, match_score)
        If no confident match is found, returns (None, score) so the
        caller can decide to keep the raw text or flag it for review.
    """
    barangays, aliases, _ = _load()
    if not heard_text or not barangays:
        return None, 0.0

    candidates = barangays + list(aliases)
    result = process.extractOne(heard_text, candidates, scorer=fuzz.WRatio)
    if result is None:
        return None, 0.0

    match, score, _ = result
    if score >= threshold:
        return aliases.get(match, match), score
    return None, score


def get_barangay_list_for_prompt() -> str:
    """
    Comma-separated barangay names for Whisper's initial_prompt on the
    address question — biases decoding toward the real local names instead
    of the nearest English-sounding word.
    """
    return ", ".join(load_barangays())
