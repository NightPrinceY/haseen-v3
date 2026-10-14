# Source 05: aramb-dev/arabic-vocab-api (checked 2026-10-13)

- Repo: https://github.com/aramb-dev/arabic-vocab-api (no licence, last push 2026-08-04; API at arabic-vocab-api.vercel.app).
  Local copies: `data/sources/arabic-vocab-api/{egyptian,levantine,darija,msa}.csv` (the `all.csv` of each dialect).
- **Real origin: Lingualism** (Matthew Aldrich's "Colloquial Arabic Vocabulary" book series, native-speaker audio on
  media.lingualism.com: `ecav` = Egyptian, `lcav` = Levantine, `mcav` = Moroccan, `pcav` = Palestinian). The books
  use the same English prompts and thematic chapters in every dialect, so the lists are **parallel by concept**.
  The repo scraped the free web chapters only (17–18 of about 57 chapters per book).
- Columns: english, transliteration, arabic, audioUrl, source (section). MSA file: english, transliteration, arabic, source.

| File | Rows | Single-word Arabic | Notes |
|---|---|---|---|
| egyptian | 6,291 | 4,495 | 2,012 Lingualism rows (`ecav`); 3,890 "legacy" rows without audio (other origin, aligned with the MSA file by section); **389 rows are Levantine** (`lcav` audio in around_the_house / work_and_professions: طابِق، سْتأْجر، أجار) → drop by audio prefix |
| levantine | 2,511 | 882 | Lingualism `lcav` + 125 Palestinian (`pcav`) |
| darija | 2,376 | 1,242 | Lingualism Moroccan (`mcav`) |
| msa | 3,876 | — | pairs with the legacy Egyptian sections (3,639 shared English keys) |

## Parallel value (English gloss as the pivot; Egyptian cleaned of the 389 Levantine rows)
- Concepts in Egyptian ∩ Levantine ∩ Moroccan: **1,556**; all four incl. MSA: 362.
- About 170–200 **example sentences per dialect**, parallel across the three (e.g. "I don't feel like chicken":
  مِش عايِز آكُل فِراخ / ما عبالي دْجاج / ما فِيّا لّي ياكُل الدّْجاج).
- Diacritics and plural forms in brackets need stripping; multi-form cells use "، / ( )".

## Verdict
- **High quality, small.** Expert-written, native-checked, concept-aligned across Egyptian / Levantine / Moroccan
  (+ partial MSA). Best use: cross-dialect equivalence sets (substitution like V1 family C for 3 dialects), parallel
  example sentences for alignment, and a clean test list.
- Limits: everyday vocabulary (house, food, health, work); few function words; no Gulf / Iraqi / Sudanese. The full
  books (about 4,500 items each; a Tunisian book also exists) are paid, Yahya's call.
