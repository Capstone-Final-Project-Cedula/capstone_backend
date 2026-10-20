# Gazetteer data

## baungon_bukidnon_barangays.csv

The 16 barangays of the Municipality of Baungon, Bukidnon (Region X), used to
recognize and correct barangay names in spoken or typed addresses (fuzzy
matching) and as a vocabulary hint for Whisper on the address question.

| Column | Meaning |
|---|---|
| province, municipality, barangay | Official names |
| psgc_code | Philippine Standard Geographic Code (as listed on Wikipedia) |
| population_2020 | 2020 Census of Population and Housing |
| aliases | Other names people say, separated by `;` (Imbatug was formerly "Poblacion") |
| purok | Leave blank, or add one row per purok/sitio (see below) |

Sources (checked October 2026):
- PhilAtlas, "Baungon, Province of Bukidnon": https://www.philatlas.com/mindanao/r10/bukidnon/baungon.html
- Wikipedia, "Baungon": https://en.wikipedia.org/wiki/Baungon

### Adding puroks and sitios

No public source lists the puroks or sitios of Imbatug or the other barangays.
Get the official list from the Imbatug barangay hall (or the Baungon municipal
planning office) and add one row per purok or sitio, for example:

```
Bukidnon,Baungon,Imbatug,101301012,7487,Poblacion;Imbatug Poblacion,Purok 1
Bukidnon,Baungon,Imbatug,101301012,7487,Poblacion;Imbatug Poblacion,Sitio Mando
```

Restart the backend after editing (the list is cached).

**Imbatug's purok list (when Ralph gets it from the barangay hall):** add one row
per purok as above (same first six columns as the Imbatug row, the purok name in
the last column), then restart the backend. No code change is needed; the voice
interview still works without it (the purok is taken as said and read back).

## region10_barangays.csv

All 2,022 barangays of Region X (Northern Mindanao) from the PSA PSGC (plan
001.6), so an address in another town of the region (7 of the 10 speech-sample
speakers live in Cagayan de Oro) is matched against **that town's** barangays and
written with **that town's** province, never Baungon's. A barangay is matched only
inside the town the applicant said: two barangays are named Imbatug in Bukidnon
(Baungon `1001301012`, Cabanglasan `1001322004`).

| Column | Meaning |
|---|---|
| psgc_code | PSGC code of the barangay |
| barangay | Official name ("Barangay 10" for Cagayan de Oro's numbered barangays) |
| town | City or municipality, written as in `ph_places.csv` ("Cagayan de Oro") |
| town_psgc | PSGC code of the town (the voice interview sends it with the barangay question) |
| province | The town's province. Cagayan de Oro (a highly urbanized city, no province in the PSGC) is set to Misamis Oriental; Iligan is left blank as in the PSGC |

Checked counts: 2,022 rows, Cagayan de Oro 80, Baungon 16 (93 towns). Baungon's
aliases (Imbatug = "Poblacion") and puroks stay in `baungon_bukidnon_barangays.csv`.

Source: the MIT-licensed `barangay` package version **2026.10.12.0**
(https://github.com/bendlikeabamboo/barangay), exported once, offline, with
`scripts/export_region10_barangays.py`. That package needs Python 3.13 or newer, so
it is **not** an app dependency: the script runs in a throwaway environment (steps
at the top of the script) and refuses to write the CSV if the row counts change.

## ph_places.csv

All 82 provinces, 150 cities and 1,493 municipalities of the Philippines (1,725
rows), used to recognize a spoken **place of birth** and write it the official way
with its province ("polo molok" -> "Polomolok, South Cotabato", "Gensan" ->
"General Santos City"). Towns with the same name in several provinces are told
apart by the province the applicant says, or else by the barangay's own province.

| Column | Meaning |
|---|---|
| name | Name without "City of" ("General Santos") |
| display | How it is written on the form ("General Santos City") |
| type | province, city or municipality |
| province | Province of the city/municipality (blank for highly urbanized and some independent cities) |
| region, psgc_id | PSA region name and PSGC code |

Source: the PSA Philippine Standard Geographic Code (PSGC) masterlist, exported with
the MIT-licensed `barangay` package version 2026.7.13.0
(https://github.com/bendlikeabamboo/barangay, `barangay export --model flat`),
keeping only provinces, cities and municipalities. Regenerate it the same way when
the PSA publishes a new PSGC release.

## common_surnames_ph.csv

Surnames that are widely used in the Philippines, weighted toward Visayan and
Mindanao (Cebuano) names. It was compiled by hand and is **not** census data or
a frequency ranking. The voice interview uses it (together with the surnames
already in this barangay's records) only to choose the **local spelling** of a
surname Whisper heard: "Bakulio" becomes "Baculio" because the two sound
exactly the same (`name_lexicon.sound_key`). A name that does not sound like
any listed name is kept as heard, and the applicant still confirms every name
on screen. Add the surnames common in Imbatug (one per line) to improve it;
restart the backend after editing.

## resident_names.csv (optional, not included)

The earlier made-up Mambajao resident list was removed: a real name must never be
"corrected" into a fictional one. The old whole-sentence extractor
(`/api/extraction/fields`) can still match names against a list if you add
`resident_names.csv` here (columns `surname`, `name`, `middle_name`), but only use a
real, consented list. Without it, that matching is simply off. The live interview
does not use this file; it uses `common_surnames_ph.csv` and this barangay's own records.
