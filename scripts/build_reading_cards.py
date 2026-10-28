"""
Builds app/eval/reading_cards.json (plan 003) from the 8 made-up identity cards of
capstone_docs/eval/interview_script.md. Fake people only. Run from capstone_backend:

    python scripts/build_reading_cards.py

For every card and every one of the 14 questions it writes the exact sentence the volunteer reads
(`read`, in the card's answer style A-E) and the value the system should end up with (`expected`).
The address is asked as 3 questions and the place of birth as 2 steps, so those fields also carry
`parts`, each with its own `read` and `expected`; the field's WER pools its parts (plan 003, Codex plan
review finding 2).
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "app" / "eval" / "reading_cards.json"

# (no, style, surname, given, middle, purok, barangay, town, birth_province, birth_town, birth_full)
PART1 = [
    (1, "A", "Abarquez", "Maria Luisa", "Tumampos", "Purok 2", "Imbatug", "Baungon", "Bukidnon", "Malaybalay City"),
    (2, "B", "Dela Cruz", "Jerome", "Bacalso", "Purok 5", "Lingating", "Baungon", "Cagayan de Oro City", None),
    (3, "C", "Jumawan", "Ricky", "none", "Sitio Kibalabag", "Pualas", "Baungon", "Bukidnon", "Valencia City"),
    (4, "D", "Sumagang", "Analyn", "Ocampo", "Purok 1", "Liboran", "Baungon", "Iligan City", None),
    (5, "E", "Abella", "Kristine Joy", "Pacquiao", "Purok 4", "Salimbalan", "Baungon", "Bukidnon", "Talakag"),
    (6, "B", "Gumapac", "Rodel", "Maglinte", "Purok 7", "San Vicente", "Baungon", "Bukidnon", "Manolo Fortich"),
    (7, "C", "Ybañez", "Ma. Cristina", "Lumapas", "Purok 3", "Danatag", "Baungon", "Bukidnon", "Baungon"),
    (8, "A", "Sta. Maria", "Jun-jun", "Cabahug", "Purok 6", "Nicdao", "Baungon", "Davao City", None),
]
# (date, sex, civil, citizenship, icr, height, weight, occupation, income)
PART2 = [
    ("March 14, 1990", "Female", "Married", "Filipino", "none", 158, 55, "Teacher", 180000),
    ("May 2, 1985", "Male", "Single", "Filipino", "none", 170, 68, "Farmer", 96000),
    ("November 30, 1972", "Male", "Widowed", "Filipino", "none", 165, 72, "Carpenter", 120000),
    ("January 9, 2001", "Female", "Single", "Filipino", "none", 152, 48, "Student", 0),
    ("July 21, 1995", "Female", "Separated", "Filipino", "none", 160, 52, "Vendor", 60000),
    ("September 5, 1968", "Male", "Married", "Filipino", "none", 172, 80, "Driver", 150000),
    ("December 25, 1980", "Female", "Widowed", "Filipino", "none", 155, 60, "Barangay health worker", 84000),
    ("April 17, 1999", "Male", "Single", "Filipino", "none", 168, 63, "Welder", 200000),
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


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
            "address_town": {"expected": town, "read": generic(
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
        parts["place_of_birth_town"] = {"expected": btown, "read": generic(
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


def main():
    cards = [build_card(p1, p2) for p1, p2 in zip(PART1, PART2)]
    data = {
        "version": 1,
        "note": "Made-up people only. Built by scripts/build_reading_cards.py from interview_script.md.",
        "sets": {"all": list(range(1, 9)), "set_a": [1, 2, 3, 4], "set_b": [5, 6, 7, 8]},
        "cards": cards,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(cards)} cards)")


if __name__ == "__main__":
    main()
