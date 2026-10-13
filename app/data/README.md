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

## mambajao_fictional_residents_1000.csv

1,000 made-up residents from the earlier Mambajao setup. Only the whole-sentence
extractor (`/api/extraction/fields`) uses it; the live interview never corrects
names against it, because a real name must never be "corrected" into a fictional
one. Replace it only with a real, consented list of common local surnames.
