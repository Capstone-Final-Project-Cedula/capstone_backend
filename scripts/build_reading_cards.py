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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.eval_cards import build_card  # noqa: E402  (plan 009: the card shape lives in the app)

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
