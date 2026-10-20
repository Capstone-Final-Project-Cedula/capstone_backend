"""
app/services/barangay_gazetteer.py
Loads the known list of barangays for the deployment municipality and
provides fuzzy-matching so a misheard barangay name (e.g. Whisper writes
"Imbatog" or "Salim Balan") is corrected to the closest real barangay
instead of being stored as garbage text.

Data: app/data/baungon_bukidnon_barangays.csv (16 barangays of Baungon,
Bukidnon, with aliases and puroks; see app/data/README.md for sources and how
to add puroks/sitios), and app/data/region10_barangays.csv (all 2,022 PSGC
barangays of Region X, so an address in another town of the region is matched
against that town's own barangays; plan 001.6).

Requires: pip install rapidfuzz
"""
import csv
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from rapidfuzz import process, fuzz

logger = logging.getLogger(__name__)

GAZETTEER_CSV_PATH = Path(__file__).parent.parent / "data" / "baungon_bukidnon_barangays.csv"
REGION_CSV_PATH = Path(__file__).parent.parent / "data" / "region10_barangays.csv"

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


@lru_cache(maxsize=1)
def _gazetteer_town() -> tuple[str, str]:
    """(municipality, province) of the barangays in GAZETTEER_CSV_PATH, lower case."""
    try:
        with open(GAZETTEER_CSV_PATH, newline="", encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f), None) or {}
    except FileNotFoundError:
        row = {}
    return (row.get("municipality") or "").strip().lower(), (row.get("province") or "").strip().lower()


def load_barangays(csv_path: Optional[str] = None) -> list[str]:
    """Unique barangay names of the municipality."""
    return list(_load(csv_path)[0])


def load_aliases() -> dict[str, str]:
    """Other names people use for a barangay (e.g. "Poblacion" -> "Imbatug")."""
    return dict(_load()[1])


def load_puroks(barangay: str) -> list[str]:
    """Puroks/sitios of a barangay, if they were added to the CSV."""
    return list(_load()[2].get(barangay, []))


def match_barangay(heard_text: str, threshold: int = MATCH_THRESHOLD, names: Optional[list[str]] = None,
                   aliases: Optional[dict[str, str]] = None) -> tuple[Optional[str], float]:
    """
    Given a raw span of text that's supposed to contain a barangay name,
    find the closest real barangay (aliases included) from the gazetteer.

    Returns:
        (matched_barangay_or_None, match_score)
        If no confident match is found, returns (None, score) so the
        caller can decide to keep the raw text or flag it for review.
    """
    if names is None:
        names, aliases, _ = _load()
    barangays, aliases = names, aliases or {}
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


@lru_cache(maxsize=1)
def _load_region() -> dict[str, dict]:
    """town PSGC -> {"town", "province", "barangays"} for every town of Region X, read once."""
    towns: dict[str, dict] = {}
    try:
        with open(REGION_CSV_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                town = towns.setdefault(row["town_psgc"], {"town": row["town"], "province": row["province"],
                                                           "barangays": []})
                town["barangays"].append(row["barangay"])
    except FileNotFoundError:
        logger.error(f"Region X barangay CSV not found at {REGION_CSV_PATH}. Only the office town's barangays are known.")
    return towns


def region_town(town_psgc: str) -> Optional[dict]:
    """{"town", "province", "barangays"} of a Region X town, or None (outside Region X / unknown)."""
    return _load_region().get(town_psgc or "")


def town_barangays(town_psgc: str) -> tuple[list[str], dict[str, str]]:
    """
    (barangay names, alias -> barangay) of one town. The office town also gets the aliases
    (Imbatug = "Poblacion") of baungon_bukidnon_barangays.csv. Empty outside Region X.
    """
    town = region_town(town_psgc)
    if not town:
        return [], {}
    names, aliases, _ = _load()
    if (town["town"].lower(), town["province"].lower()) == _gazetteer_town():
        return list(dict.fromkeys(town["barangays"] + names)), dict(aliases)
    return list(town["barangays"]), {}


# Whisper keeps only the end of a long initial_prompt (about 220 tokens), and long prompts can backfire:
# a big town's list is cut to this many characters (Cagayan de Oro has 80 barangays).
PROMPT_LIST_CHARS = 600


def get_barangay_list_for_prompt(town_psgc: Optional[str] = None) -> str:
    """
    Comma-separated barangay names for Whisper's initial_prompt on the
    address question — biases decoding toward the real local names instead
    of the nearest English-sounding word. With a town PSGC code, that town's
    barangays (plan 001.6); without one, the office town's.
    """
    names = town_barangays(town_psgc)[0] if town_psgc else []
    names = names or load_barangays()
    # Named barangays first: "Barangay 1" ... "Barangay 40" (Cagayan de Oro) need no hint for their names.
    names = sorted(names, key=lambda n: bool(re.fullmatch(r"barangay\s+\d+", n, re.IGNORECASE)))
    out = ""
    for name in names:
        piece = name if not out else f", {name}"
        if len(out) + len(piece) > PROMPT_LIST_CHARS:
            break
        out += piece
    return out
