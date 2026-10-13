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

## mambajao_fictional_residents_1000.csv

1,000 made-up residents from the earlier Mambajao setup. Only the whole-sentence
extractor (`/api/extraction/fields`) uses it; the live interview never corrects
names against it, because a real name must never be "corrected" into a fictional
one. Replace it only with a real, consented list of common local surnames.
