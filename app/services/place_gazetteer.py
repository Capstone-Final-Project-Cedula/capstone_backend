"""
app/services/place_gazetteer.py
Every Philippine province, city and municipality (PSA PSGC, see app/data/README.md),
used to recognize a spoken place of birth and write it properly:

  "polomolok south cotabato"  -> "Polomolok, South Cotabato"
  "Polo Molok" (Whisper split) -> "Polomolok, South Cotabato"
  "Gensan"                     -> "General Santos City"
  "San Jose, Batangas"         -> "San Jose, Batangas"   (one of many San Joses)
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process

from app.services.name_lexicon import sound_key

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "ph_places.csv")

# What people commonly call some places instead of the official name.
ALIASES = {
    "gensan": "General Santos", "gen san": "General Santos", "cdo": "Cagayan de Oro",
    "qc": "Quezon City", "metro manila": "Manila", "ncr": "Manila", "davao": "Davao",
}
# Words in an answer that are never (part of) a place name.
STOP = {
    "i", "was", "born", "in", "at", "sa", "ako", "ipinanganak", "pinanganak", "city", "of", "province",
    "probinsya", "municipality", "town", "bayan", "lungsod", "the", "my", "place", "birth", "is", "po",
    "barangay", "purok", "sitio", "philippines", "pilipinas", "mindanao", "luzon", "visayas", "and", "ng",
}


REGION_X = "Region X (Northern Mindanao)"


@dataclass(frozen=True)
class Place:
    name: str        # "Polomolok", "General Santos"
    display: str     # "Polomolok", "General Santos City"
    type: str        # province | city | municipality
    province: str    # "" for a highly urbanized city
    region: str = ""
    psgc: str = ""   # PSGC code ("1001301000" = Baungon)


def _norm(text: str) -> str:
    t = (text or "").lower().replace("ñ", "n")
    t = re.sub(r"[^a-z\s-]", " ", t).replace("-", " ")
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def _load() -> Tuple[Dict[str, List[Place]], Dict[str, List[Place]]]:
    by_name: Dict[str, List[Place]] = {}      # "polomolok" -> [Place]; also the no-space form
    provinces: Dict[str, List[Place]] = {}
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Place(r["name"], r["display"], r["type"], r["province"], r["region"], r["psgc_id"])
            key = _norm(p.name)
            target = provinces if p.type == "province" else by_name
            target.setdefault(key, []).append(p)
            if " " in key:                                  # also "sanjose" for a split/merged hearing
                target.setdefault(key.replace(" ", ""), []).append(p)
    return by_name, provinces


_SOUND_CACHE: Dict[int, Dict[str, List[Place]]] = {}


def _sound_index(cache_id: int, index: Dict[str, List[Place]]) -> Dict[str, List[Place]]:
    if cache_id not in _SOUND_CACHE:
        sounds: Dict[str, List[Place]] = {}
        for key, places in index.items():
            if " " in key:
                continue                    # the no-space form of the same name is indexed too
            for p in places:
                if p not in sounds.setdefault(sound_key(key), []):
                    sounds[sound_key(key)].append(p)
        _SOUND_CACHE[cache_id] = sounds
    return _SOUND_CACHE[cache_id]


def _ngrams(words: List[str], max_n: int = 4):
    """Longest first, so "San Jose del Monte" wins over "San Jose"."""
    for n in range(min(max_n, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            yield i, n, " ".join(words[i:i + n])


def _match(words: List[str], index: Dict[str, List[Place]], fuzzy: bool) -> Tuple[List[Place], set]:
    used: set = set()
    for i, n, gram in _ngrams(words):
        if any(j in used for j in range(i, i + n)) or all(w in STOP for w in words[i:i + n]):
            continue
        for key in (gram, gram.replace(" ", "")):          # "polo molok" -> "polomolok"
            if key in index:
                return index[key], set(range(i, i + n))
    if fuzzy:
        # Same sound, different spelling ("Ozamis" / "Ozamiz", "Kidapawan" / "Kidapauan").
        sounds = _sound_index(id(index), index)
        for i, n, gram in _ngrams(words, 3):
            if any(w in STOP for w in words[i:i + n]):
                continue
            key = sound_key(gram)
            if len(key) >= 4 and key in sounds:
                return sounds[key], set(range(i, i + n))
        keys = list(index)
        for i, n, gram in _ngrams(words, 3):
            squashed = gram.replace(" ", "")
            if len(squashed) < 6 or any(w in STOP for w in words[i:i + n]):
                continue
            res = process.extractOne(squashed, keys, scorer=fuzz.ratio, score_cutoff=88)
            if res:
                return index[res[0]], set(range(i, i + n))
    return [], set()


def find_place(text: str, home_province: str = "") -> Optional[dict]:
    """
    The place named in an answer: {"value", "place", "province"} or None.
    value is how it is written on the form ("Polomolok, South Cotabato").
    home_province (the barangay's own province) decides between towns with the same
    name when the applicant did not say the province ("Valencia" -> Valencia, Bukidnon).
    """
    by_name, provinces = _load()
    t = _norm(text)
    for alias, real in ALIASES.items():
        t = re.sub(rf"\b{alias}\b", _norm(real), t)
    words = t.split()
    if not words:
        return None

    # "Cebu City" is the city, not Cebu province: match "<name> city" against cities first.
    places: List[Place] = []
    used: set = set()
    if "city" in words:
        k = words.index("city")
        for n in (3, 2, 1):
            if k - n < 0:
                continue
            gram = " ".join(words[k - n:k])
            found = by_name.get(gram, []) + by_name.get(gram.replace(" ", ""), []) + by_name.get(f"{gram} city", [])
            cities = [p for p in found if p.type == "city"]
            if cities:
                places, used = cities, set(range(k - n, k + 1))
                break

    rest = [w for i, w in enumerate(words) if i not in used]
    if not places:
        # The longer exact name wins: "Cagayan de Oro" is the city, not Cagayan province.
        town, town_used = _match(rest, by_name, fuzzy=False)
        _, prov_exact_used = _match(rest, provinces, fuzzy=False)
        if town and len(town_used) > len(prov_exact_used):
            places = town
            rest = [w for i, w in enumerate(rest) if i not in town_used]
    prov_matches, prov_used = _match(rest, provinces, fuzzy=True)
    province = prov_matches[0].name if prov_matches else ""
    if not places:
        rest = [w for i, w in enumerate(rest) if i not in prov_used]
        places, _ = _match(rest, by_name, fuzzy=True) if rest else ([], set())

    if places:
        if province:
            same = [p for p in places if p.province == province]
            places = same or places
        elif home_province and len({p.province for p in places}) > 1:
            home = [p for p in places if p.province.lower() == home_province.lower()]
            places = home or places
        unique = {(p.display, p.province) for p in places}
        if len(unique) == 1:
            p = places[0]
            prov = p.province or province
            value = f"{p.display}, {prov}" if prov and prov != p.display else p.display
            return {"value": value, "place": p.display, "province": prov}
        # Same town name in several provinces and none was said: keep the town only.
        return {"value": places[0].name, "place": places[0].name, "province": "",
                "ambiguous": sorted({p.province for p in places if p.province})}
    if province:
        return {"value": province, "place": "", "province": province}
    return None


# ─── The town of an address (plan 001.6) ─────────────────────────────────────

_TOKEN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿñÑ]+")
STREET_WORDS = {"street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "extension", "ext", "highway",
                "boulevard", "blvd", "lane", "subdivision", "subd", "village", "homes", "heights"}


def normalise(text: str) -> str:
    """Lower case, no accents on n, punctuation and hyphens to spaces (how names are compared here)."""
    return _norm(text)


def _tokens(text: str) -> List[Tuple[str, int, int]]:
    """Words of the original text with their character spans, normalised like _norm ("Camaman-an" -> 2 words)."""
    return [(_norm(m.group(0)), m.start(), m.end()) for m in _TOKEN.finditer(text or "")]


def town_by_psgc(psgc: str) -> Optional[Place]:
    """The city or municipality with this PSGC code."""
    by_name, _ = _load()
    for places in by_name.values():
        for p in places:
            if p.psgc == psgc:
                return p
    return None


def find_town(text: str, exclude: frozenset = frozenset(), home_province: str = "",
              prefer_region: str = REGION_X) -> Optional[dict]:
    """
    The city or municipality named in an address, scanning from the END: addresses go from small to big
    ("Purok 2, Kalilangan, Baungon, Bukidnon" -> Baungon), so the last town said is the town.

    exclude: normalised names that are barangays of the office town ("kalilangan", "san vicente" are also
    towns elsewhere); they are skipped unless a province other than home_province is said.
    A name found in several provinces is decided by the province said, then by prefer_region (Region X is
    tried first), then by home_province.

    Returns {"place", "span": (start, end) in text, "province_said", "ambiguous"} or None.
    """
    by_name, provinces = _load()
    toks = _tokens(text)
    words = [ALIASES.get(t[0], t[0]) for t in toks]
    if not words:
        return None

    def towns(places):
        return [p for p in places if p.type != "province"]

    def exact(index, start, end):
        gram = " ".join(words[start:end])
        if all(w in STOP for w in words[start:end]):
            return []
        return index.get(_norm(gram)) or index.get(gram.replace(" ", "")) or []

    spans = [(s, s + n) for n in range(min(4, len(words)), 0, -1) for s in range(len(words) - n + 1)]
    town_hits = [(s, e, towns(exact(by_name, s, e))) for s, e in spans]
    town_hits = [h for h in town_hits if h[2]]
    prov_hits = [(s, e, exact(provinces, s, e)) for s, e in spans]
    prov_hits = [h for h in prov_hits if h[2]]
    # "Cagayan" (a province) inside "Cagayan de Oro" (a city): the longer name wins.
    prov_hits = [p for p in prov_hits if not any(t[0] <= p[0] and p[1] <= t[1] and t[1] - t[0] > p[1] - p[0]
                                                 for t in town_hits)]
    province_said = max(prov_hits, key=lambda p: p[1])[2][0].name if prov_hits else ""
    # Another province said ("San Vicente, Palawan") lifts exclude; a name that is also a town ("Rizal") does not.
    town_spans = {(t[0], t[1]) for t in town_hits}
    other_province = any((p[0], p[1]) not in town_spans and p[2][0].name.lower() != home_province.lower()
                         for p in prov_hits)
    skip = frozenset() if other_province else exclude
    town_hits = [t for t in town_hits if " ".join(words[t[0]:t[1]]) not in skip]
    # "Quezon" is a town and a province: when another town is named, it is the province.
    prov_spans = {(p[0], p[1]) for p in prov_hits}
    if any((t[0], t[1]) not in prov_spans for t in town_hits):
        town_hits = [t for t in town_hits if (t[0], t[1]) not in prov_spans]
    used = {j for p in prov_hits for j in range(p[0], p[1])}

    def fits(start, end):
        """Not a street ("Rizal Street") and not followed by an office barangay ("..., Rizal, Imbatug")."""
        if end < len(words) and words[end] in STREET_WORDS:
            return False
        tail = f" {' '.join(words[end:])} "
        return not any(f" {name} " in tail for name in skip)

    town_hits = [t for t in town_hits if fits(t[0], t[1])]
    found = max(town_hits, key=lambda t: (t[1], t[1] - t[0])) if town_hits else None
    if not found:
        # Same sound, different spelling ("Baongon", "Kagayan de Oro"), then close spelling; last one said wins.
        sounds = _sound_index(id(by_name), by_name)
        keys = list(by_name)
        for end in range(len(words), 0, -1):
            for n in range(min(3, end), 0, -1):
                s = end - n
                part = words[s:end]
                gram = " ".join(part)
                if (any(j in used for j in range(s, end)) or any(w in STOP for w in part) or gram in skip
                        or not fits(s, end)):
                    continue
                key = sound_key(gram)
                if len(key) >= 4 and towns(sounds.get(key, [])):
                    found = (s, end, towns(sounds[key]))
                    break
                squashed = gram.replace(" ", "")
                if len(squashed) >= 6:
                    res = process.extractOne(squashed, keys, scorer=fuzz.ratio, score_cutoff=88)
                    if res and towns(by_name[res[0]]):
                        found = (s, end, towns(by_name[res[0]]))
                        break
            if found:
                break
    if not found:
        return None

    start, end, places = found
    for narrow in (
        lambda p: bool(province_said) and p.province.lower() == province_said.lower(),
        lambda p: bool(prefer_region) and p.region == prefer_region,
        lambda p: bool(home_province) and p.province.lower() == home_province.lower(),
    ):
        if len(places) > 1:
            places = [p for p in places if narrow(p)] or places
    unique = {p.psgc for p in places}
    return {"place": places[0], "span": (toks[start][1], toks[end - 1][2]), "province_said": province_said,
            "ambiguous": len(unique) > 1}
