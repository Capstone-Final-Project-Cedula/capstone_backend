"""
app/services/name_lexicon.py
Known local names, used to spell a heard name the way it is written locally.

Whisper writes an unfamiliar Filipino name with English spelling habits:
"Baculio" becomes "Bakulio", "Abellana" becomes "Abelana", "Pacana" becomes
"Pakana". These sound exactly the same, so the audio cannot decide; a list of
names that are really used here can. Two sources:

  * names already in this barangay's records (families share surnames), and
  * app/data/common_surnames_ph.csv, common Filipino surnames (see its README).

A heard name is replaced only when it SOUNDS the same as exactly one known name
(same sound key, see sound_key) and is not itself a known name. The applicant
still sees the result on screen and can say "no" and spell it.
"""
from __future__ import annotations

import csv
import logging
import os
import re
import threading
import time
from functools import lru_cache
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "common_surnames_ph.csv")
RECORDS_TTL_SECONDS = 600


def sound_key(name: str) -> str:
    """
    How a name sounds, ignoring spelling choices Filipino names often differ by:
    c/k/q, ph/f, v/b, z/s, double letters, silent h after a consonant (Jhon), y as
    a vowel, and e~i, o~u (spoken alike in Filipino languages). Spaces and hyphens
    are ignored. "Baculio", "Bakulio" -> "bakulyu"; "Abellana", "Abelana" -> "abilana".
    """
    w = (name or "").lower().replace("ñ", "ny")
    w = re.sub(r"[^a-z]", "", w)
    w = w.replace("ph", "f").replace("ck", "k").replace("qu", "k")
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = re.sub(r"(?<=[bcdfgjklmnpqrstvwxz])h", "", w)        # Jhon -> Jon, Thea -> Tea
    w = w.translate(str.maketrans({"c": "k", "q": "k", "v": "b", "z": "s", "e": "i", "o": "u"}))
    w = re.sub(r"(?<=[aiu])y(?![aiu])|^y(?![aiu])", "i", w)    # Ybanez -> Ibanis
    w = re.sub(r"i(?=[aiu])", "y", w)                          # Baculio ~ Bakulyo
    w = re.sub(r"(.)\1+", r"\1", w)                            # Abellana -> Abelana
    return w


@lru_cache(maxsize=1)
def _common_surnames() -> tuple:
    names = []
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n = (row.get("surname") or "").strip()
                if n:
                    names.append(n)
    except OSError as exc:
        logger.warning(f"Common surname list not loaded: {exc}")
    return tuple(names)


_records_lock = threading.Lock()
_records_cache: Dict[str, object] = {"at": 0.0, "names": {}}


def _record_names() -> Dict[str, Set[str]]:
    """Names in this barangay's applicant records, refreshed every few minutes."""
    with _records_lock:
        if time.time() - _records_cache["at"] < RECORDS_TTL_SECONDS:
            return _records_cache["names"]  # type: ignore[return-value]
        names: Dict[str, Set[str]] = {"surname": set(), "given_name": set()}
        try:
            from sqlalchemy import text
            from app.core.database import engine
            with engine.connect() as conn:
                for surname, middle, given in conn.execute(text(
                    "select surname, middle_name, given_name from applicant_records"
                )):
                    for n in (surname, middle):
                        if n and n.strip():
                            names["surname"].add(n.strip())
                    if given and given.strip():
                        names["given_name"].add(given.strip())
        except Exception as exc:  # database unreachable: the common list still works
            logger.warning(f"Could not read names from records: {exc}")
        _records_cache.update(at=time.time(), names=names)
        return names


def refresh_records() -> None:
    """Forget the cached record names (e.g. after a new CTC is issued)."""
    with _records_lock:
        _records_cache["at"] = 0.0


def _index(kind: str) -> Dict[str, Set[str]]:
    known = set(_record_names().get(kind, set()))
    if kind == "surname":
        known.update(_common_surnames())
    index: Dict[str, Set[str]] = {}
    for n in known:
        index.setdefault(sound_key(n), set()).add(n)
    return index


def is_known(name: str, kind: str) -> bool:
    """Is this exact spelling a known local name?"""
    matches = _index(kind).get(sound_key(name), set())
    return any(m.lower() == (name or "").lower() for m in matches)


def known_spelling(heard: str, kind: str) -> Optional[str]:
    """
    The locally used spelling of a heard name, or None to keep what was heard.
    kind: "surname" (also middle names) or "given_name".
    """
    if not heard or len(re.sub(r"[^A-Za-z]", "", heard)) < 3:
        return None
    index = _index(kind)
    matches = index.get(sound_key(heard), set())
    if not matches:
        return None
    if any(m.lower() == heard.lower() for m in matches):
        return None                    # already spelled like a known name
    if len({m.lower() for m in matches}) != 1:
        return None                    # two known spellings sound alike: keep what was heard
    return next(iter(matches))


def names_for_prompt(kind: str, limit: int = 40) -> list:
    """The most frequent names in the records, for Whisper's hint."""
    try:
        from sqlalchemy import text
        from app.core.database import engine
        column = "surname" if kind == "surname" else "given_name"
        with engine.connect() as conn:
            rows = conn.execute(text(
                f"select {column}, count(*) c from applicant_records where {column} is not null "
                f"group by {column} order by c desc limit :n"
            ), {"n": limit}).fetchall()
        return [r[0] for r in rows if r[0]]
    except Exception:
        return []
