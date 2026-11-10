"""
app/services/eval_cards.py
Plan 009: the made-up identity cards a participant reads aloud in the normal interview. The card text is
the ground-truth transcript for WER.

  - build_card() turns one made-up person into the {expected, read} shape score_unit reads: `read` is the
    exact sentence the participant says (in one of the answer styles A-E), `expected` the value the system
    should save. The address is asked in 3 steps and the place of birth in 2, so those fields carry `parts`.
    (Moved here from scripts/build_reading_cards.py, which now imports it for the 8 fixed plan 003 cards.)
  - generate_card() draws a new made-up person from fixed lists only (the surname list, the Baungon
    barangays and the Bukidnon places in app/data, and the name lists below). It never reads the database,
    so no real resident's record can end up on a card.
"""
import csv
import random
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

DATA = Path(__file__).resolve().parents[1] / "data"
STYLES = "ABCDE"          # A plain, B English sentence, C Bisaya, D Tagalog, E spelled names
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

FEMALE_GIVEN = (
    "Maria Luisa", "Analyn", "Kristine Joy", "Ma. Cristina", "Rowena", "Jocelyn", "Marites", "Liezl", "Rhea",
    "Mary Grace", "Divina", "Shiela Mae", "Leonora", "Cherry", "Imelda", "Gemma", "Janeth", "Arlene", "Nenita",
    "Rosalie", "Daisy", "Lorna", "Evangeline", "Mylene",
)
MALE_GIVEN = (
    "Jerome", "Ricky", "Rodel", "Jun-jun", "Ramil", "Edgardo", "Arnel", "Romeo", "Dennis", "Joel", "Rolando",
    "Mark Anthony", "Noel", "Ronaldo", "Jaime", "Ernesto", "Rogelio", "Alvin", "Wilfredo", "Danilo", "Renato",
    "John Paul", "Elmer", "Reynaldo",
)
OCCUPATIONS = ("Teacher", "Farmer", "Carpenter", "Student", "Vendor", "Driver", "Welder", "Barangay health worker",
               "Mechanic", "Tailor", "Cook", "Electrician", "Nurse", "Laborer")
CIVIL_STATUSES = ("Single", "Married", "Widowed", "Separated")
INCOMES = (0, 36000, 48000, 60000, 72000, 84000, 96000, 120000, 150000, 180000, 200000, 240000)
BORN_FROM, BORN_TO = date(1950, 1, 1), date(2007, 12, 31)


# ─── The card shape (shared with scripts/build_reading_cards.py) ─────────────

def spelled(text):
    """"Dela Cruz" -> "D-E-L-A C-R-U-Z"."""
    return " ".join("-".join(ch.upper() for ch in word if ch.isalpha()) for word in text.split())


def iso(date_text):
    month, rest = date_text.split(" ", 1)
    day, year = rest.replace(",", "").split(" ")
    return f"{int(year):04d}-{MONTHS.index(month) + 1:02d}-{int(day):02d}"


def name_read(style, value, lead_b, lead_c, lead_d, none_text="none"):
    if style == "E" and value.lower() != "none":
        return f"{value}, {spelled(value)}."
    if style in ("A", "E"):
        return f"{value.capitalize() if value == 'none' else value}."
    if style == "B":
        return f"{lead_b} {value} po."
    if style == "C":
        return f"{lead_c} {value}."
    return f"{lead_d} {value}."


def generic(style, plain, b, c, d):
    return {"A": plain, "B": b, "C": c, "D": d, "E": plain}[style]


def build_card(p1, p2):
    """
    p1 = (no, style, surname, given, middle, purok, barangay, town, birth_province, birth_town or None)
    p2 = (date "March 14, 1990", sex, civil status, citizenship, icr, height, weight, occupation, income)
    """
    no, style, surname, given, middle, purok, brgy, town, bprov, btown = p1
    dob, sex, civil, cit, icr, height, weight, occ, income = p2
    a = {}
    a["surname"] = {"expected": surname,
                    "read": name_read(style, surname, "My surname is", "Akong apelyido kay", "Ang apelyido ko ay")}
    a["given_name"] = {"expected": given,
                       "read": name_read(style, given, "My given name is", "Akong pangalan kay", "Ang pangalan ko ay")}
    a["middle_name"] = {"expected": "" if middle == "none" else middle,
                        "read": name_read(style, middle, "My middle name is", "Akong middle name kay",
                                          "Ang middle name ko ay")}
    full_addr = f"{purok}, Barangay {brgy}, {town}, Bukidnon"
    a["address"] = {
        "expected": full_addr,
        "read": generic(style, f"{full_addr}.", f"My address is {full_addr} po.", f"Sa {full_addr}.",
                        f"Ang tirahan ko ay {full_addr}."),
        "parts": {
            "address_town": {"expected": f"{town}, Bukidnon", "read": generic(
                style, f"{town}.", f"I live in {town} po.", f"Nagpuyo ko sa {town}.", f"Nakatira ako sa {town}.")},
            "address_barangay": {"expected": brgy, "read": generic(
                style, f"{brgy}.", f"My barangay is {brgy} po.", f"Sa barangay {brgy}.",
                f"Ang barangay ko ay {brgy}.")},
            "address_purok": {"expected": purok, "read": generic(
                style, f"{purok}.", f"I live in {purok} po.", f"Sa {purok}.", f"Sa {purok} ako nakatira.")},
        },
    }
    birth_full = f"{btown}, {bprov}" if btown else bprov
    parts = {"place_of_birth_province": {"expected": bprov, "read": generic(
        style, f"{bprov}.", f"I was born in {bprov} po.", f"Natawo ko sa {bprov}.",
        f"Ipinanganak ako sa {bprov}.")}}
    if btown:
        # The extractor returns a town with its province ("Malaybalay City, Bukidnon"); the card says the same.
        parts["place_of_birth_town"] = {"expected": f"{btown}, {bprov}", "read": generic(
            style, f"{btown}.", f"I was born in {btown} po.", f"Natawo ko sa {btown}.",
            f"Ipinanganak ako sa {btown}.")}
    a["place_of_birth"] = {
        "expected": birth_full,
        "read": generic(style, f"{birth_full}.", f"I was born in {birth_full} po.", f"Natawo ko sa {birth_full}.",
                        f"Ipinanganak ako sa {birth_full}."),
        "parts": parts,
    }
    a["date"] = {"expected": iso(dob), "read": generic(
        style, f"{dob}.", f"My birthday is {dob} po.", f"Natawo ko og {dob}.", f"Ipinanganak ako noong {dob}.")}
    low = lambda s: s.lower()  # noqa: E731
    a["sex"] = {"expected": sex, "read": generic(
        style, f"{sex}.", f"I am {low(sex)} po.", f"{sex} ko.", f"{sex} po ako.")}
    a["civil_status"] = {"expected": civil, "read": generic(
        style, f"{civil}.", f"I am {low(civil)} po.", f"{civil} ko.", f"{civil} po ako.")}
    a["citizenship"] = {"expected": cit, "read": generic(
        style, f"{cit}.", f"I am {cit} po.", f"{cit} ko.", f"{cit} po ako.")}
    a["icr_no"] = {"expected": "", "read": generic(
        style, "None.", "My ICR number is none po.", "Walay ICR nako.", "Wala akong ICR.")}
    a["height"] = {"expected": str(height), "read": generic(
        style, f"{height} centimeters.", f"My height is {height} centimeters po.",
        f"Ang gitas-on nako kay {height} centimeters.", f"Ang taas ko ay {height} centimeters.")}
    a["weight"] = {"expected": str(weight), "read": generic(
        style, f"{weight} kilos.", f"My weight is {weight} kilos po.",
        f"Ang gibug-aton nako kay {weight} kilos.", f"Ang timbang ko ay {weight} kilos.")}
    a["occupation"] = {"expected": occ, "read": generic(
        style, f"{occ}.", f"My occupation is {low(occ)} po.", f"Trabaho nako kay {low(occ)}.",
        f"Ang trabaho ko ay {low(occ)}.")}
    amount = f"{income:,} pesos" if income else "none"
    a["gross_annual_income"] = {"expected": str(income), "read": generic(
        style, f"{amount.capitalize() if not income else amount}.", f"My annual income is {amount} po.",
        f"Ang kita nako kay {amount}.", f"Ang kita ko ay {amount}.")}
    return {"no": no, "style": style, "answers": a}


# ─── Generating a new made-up person ─────────────────────────────────────────

def _rows(name: str) -> List[dict]:
    with open(DATA / name, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache()
def surnames() -> Tuple[str, ...]:
    return tuple(r["surname"].strip() for r in _rows("common_surnames_ph.csv") if r["surname"].strip())


@lru_cache()
def baungon_barangays() -> Tuple[str, ...]:
    return tuple(sorted({r["barangay"].strip() for r in _rows("baungon_bukidnon_barangays.csv")}))


@lru_cache()
def bukidnon_towns() -> Tuple[str, ...]:
    """The cities and municipalities of Bukidnon, as the extractor names them ("Malaybalay City")."""
    return tuple(r["display"] for r in _rows("ph_places.csv")
                 if r["province"] == "Bukidnon" and r["type"] in ("city", "municipality"))


def generate_card(style: str, rng: Optional[random.Random] = None) -> dict:
    """One new made-up person in answer style `style` (A-E), as {"style", "answers"}."""
    rng = rng or random.SystemRandom()
    sex = rng.choice(("Female", "Male"))
    given = rng.choice(FEMALE_GIVEN if sex == "Female" else MALE_GIVEN)
    surname = rng.choice(surnames())
    middle = "none" if rng.random() < 0.125 else rng.choice([s for s in surnames() if s != surname])
    purok = f"Purok {rng.randint(1, 9)}"
    barangay = rng.choice(baungon_barangays())
    # Always a Bukidnon town, so the card has both place-of-birth steps (all 5 steps of the interview).
    birth_province, birth_town = "Bukidnon", rng.choice(bukidnon_towns())
    born = BORN_FROM + timedelta(days=rng.randint(0, (BORN_TO - BORN_FROM).days))
    dob = f"{MONTHS[born.month - 1]} {born.day}, {born.year}"
    card = build_card(
        (0, style, surname, given, middle, purok, barangay, "Baungon", birth_province, birth_town),
        (dob, sex, rng.choice(CIVIL_STATUSES), "Filipino", "none", rng.randint(145, 180), rng.randint(42, 90),
         rng.choice(OCCUPATIONS), rng.choice(INCOMES)),
    )
    return {"style": card["style"], "answers": card["answers"]}


def style_for(count: int) -> str:
    """Cards cycle through the five answer styles, so every five participants read all five."""
    return STYLES[count % len(STYLES)]
