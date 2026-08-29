"""
app/services/barangay_gazetteer.py
Loads the known list of barangays for the deployment municipality and
provides fuzzy-matching so a misheard barangay name (e.g. Whisper
outputs "boog-ong" instead of "Bug-ong") gets corrected to the closest
real barangay instead of being stored as garbage text.

Requires: pip install rapidfuzz
"""
import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

# Path to the CSV — put mambajao_camiguin_barangays.csv here
GAZETTEER_CSV_PATH = Path(__file__).parent.parent / "data" / "mambajao_camiguin_barangays.csv"

# Below this fuzzy-match score (0-100), we don't trust the correction
# and fall back to the raw text instead of silently rewriting it.
MATCH_THRESHOLD = 70


@lru_cache(maxsize=1)
def load_barangays(csv_path: Optional[str] = None) -> list[str]:
    """
    Load the list of valid barangay names once and cache it.
    Cached with lru_cache so re-deploying doesn't mean re-reading the
    file on every request — restart the process to pick up CSV changes.
    """
    path = Path(csv_path) if csv_path else GAZETTEER_CSV_PATH
    barangays = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("barangay", "").strip()
                if name:
                    barangays.append(name)
    except FileNotFoundError:
        logger.error(f"Barangay gazetteer CSV not found at {path}. "
                     f"Barangay fuzzy-matching will be disabled.")
    return barangays


def match_barangay(heard_text: str, threshold: int = MATCH_THRESHOLD) -> tuple[Optional[str], float]:
    """
    Given a raw span of text that's supposed to contain a barangay name,
    find the closest real barangay from the gazetteer.

    Args:
        heard_text: Whatever text was extracted/heard as the barangay
        threshold: Minimum fuzzy-match score (0-100) to accept a correction

    Returns:
        (matched_barangay_or_None, match_score)
        If no confident match is found, returns (None, score) so the
        caller can decide to keep the raw text or flag it for review.
    """
    barangays = load_barangays()
    if not heard_text or not barangays:
        return None, 0.0

    result = process.extractOne(heard_text, barangays, scorer=fuzz.WRatio)
    if result is None:
        return None, 0.0

    match, score, _ = result
    if score >= threshold:
        return match, score
    return None, score


def get_barangay_list_for_prompt() -> str:
    """
    Build a short comma-separated barangay list to feed into Whisper's
    initial_prompt for the address field — biases decoding toward the
    real local names instead of the nearest English-sounding word.
    """
    barangays = load_barangays()
    return ", ".join(barangays) if barangays else ""