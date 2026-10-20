"""
One-off export of the Region X (Northern Mindanao) barangays to app/data/region10_barangays.csv (plan 001.6).
Not imported by the app: the app only reads the CSV it writes.

The PSGC data comes from the MIT-licensed `barangay` package, which needs Python >= 3.13 (the backend runs on 3.12),
so run this in a separate throwaway environment, never in the backend venv:

    uv venv -p 3.13 .venv-export
    uv pip install -p .venv-export "barangay==2026.10.12.0"
    .venv-export\\Scripts\\python scripts\\export_region10_barangays.py      (from capstone_backend)

Columns: psgc_code, barangay, town, town_psgc, province.
  town      the town's name as written in ph_places.csv ("Cagayan de Oro", not "City of Cagayan De Oro")
  province  the town's province. A highly urbanized city has none in the PSGC; Cagayan de Oro is written with
            Misamis Oriental (the province it lies in), as people write their address.
"""
import csv
import subprocess
import sys
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path

PACKAGE_VERSION = "2026.10.12.0"
REGION = "Region X (Northern Mindanao)"
DATA = Path(__file__).resolve().parent.parent / "app" / "data"
OUT = DATA / "region10_barangays.csv"
PROVINCE_OVERRIDES = {"1030500000": "Misamis Oriental"}   # Cagayan de Oro City (highly urbanized)
EXPECTED = {"total": 2022, "1030500000": 80, "1001301000": 16}   # all, Cagayan de Oro, Baungon


def main() -> None:
    if version("barangay") != PACKAGE_VERSION:
        sys.exit(f"Install barangay=={PACKAGE_VERSION} (found {version('barangay')}).")
    with tempfile.TemporaryDirectory() as tmp:
        flat = Path(tmp) / "flat.csv"
        subprocess.run([sys.executable, "-m", "barangay", "export", "--model", "flat", "-f", "csv", "-o", str(flat)],
                       check=True)
        with open(flat, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    with open(DATA / "ph_places.csv", newline="", encoding="utf-8") as f:
        towns = {r["psgc_id"]: r for r in csv.DictReader(f) if r["type"] != "province"}

    out = []
    for r in rows:
        if r["type"] != "barangay" or r["region"] != REGION:
            continue
        town = towns.get(r["parent_psgc_id"])
        if town is None:
            sys.exit(f"Barangay {r['psgc_id']}: its town {r['parent_psgc_id']} is not in ph_places.csv")
        province = PROVINCE_OVERRIDES.get(r["parent_psgc_id"], r["province"] or town["province"])
        out.append({"psgc_code": r["psgc_id"], "barangay": r["name"], "town": town["name"],
                    "town_psgc": r["parent_psgc_id"], "province": province})
    out.sort(key=lambda r: r["psgc_code"])

    per_town = Counter(r["town_psgc"] for r in out)
    counts = {"total": len(out), "1030500000": per_town["1030500000"], "1001301000": per_town["1001301000"]}
    print(f"Region X barangays: {counts['total']} · Cagayan de Oro: {counts['1030500000']} · "
          f"Baungon: {counts['1001301000']} · towns: {len(per_town)}")
    if counts != EXPECTED:
        sys.exit(f"Row counts differ from the checked ones {EXPECTED}: CSV not written.")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["psgc_code", "barangay", "town", "town_psgc", "province"])
        writer.writeheader()
        writer.writerows(out)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
