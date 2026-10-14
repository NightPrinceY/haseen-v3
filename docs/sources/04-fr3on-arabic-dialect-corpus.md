# Source 04: fr3on/arabic-dialect-corpus (checked 2026-10-13)

- HF: https://huggingface.co/datasets/fr3on/arabic-dialect-corpus (MIT, created 2026-01-05, 1 parquet, 123 MB).
- Columns: `text` (YouTube comment), `label` (EG / SA / AR), `score` (like count).
- Real size **1,991,193 rows**, not "150K+" as the card says. Labels: AR 811,095 · EG 764,307 · SA 415,791.
  Text: median 36 characters (about 7 words), mean 61; score median 0 (mean 6).
- Labels are automatic (the card says "dialectal keywords, video metadata"); its own Saudi marker list includes كيف and
  عندك, which are not Saudi markers.

## Label check (1,300 rows sampled at 13 offsets via datasets-server; `data/sources/fr3on/sample_rows.jsonl`)
Rough marker test (closed lists of EGY / GLF / LEV / MAG / MSA markers):

| Label | Rows | With any marker | Marker of the labelled dialect | Other dialect / MSA markers |
|---|---|---|---|---|
| EG | 553 | 193 | 140 (73%) | LEV 21, GLF 17, MSA 31, MAG 3 |
| SA | 250 | 70 | 23 (33%) | MSA 31, LEV 15, EGY 12 |
| AR | 497 | 88 | — | MSA 35, EGY 29, LEV 17 |

Reading the rows confirms it: "SA" holds Levantine (تبعك، بجنن), Algerian (جيت نفاجي، ضرني), Egyptian (زعلان ليه، منفعوش)
and MSA; "EG" holds Gulf, Sudanese and Iraqi comments. 0 exact duplicates in the sample. Topics: news/politics, songs,
religion; many comments are short or MSA-leaning.

## Verdict
- Labels: **not usable as truth** (SA about one third Saudi; EG about three quarters Egyptian on the rows that carry a
  marker; most rows carry none).
- Text: useful as a **large pool of real informal Arabic** (2M comments) if we re-label it ourselves (sentence-level
  dialect ID + our marker evidence), e.g. to find host sentences and frequency evidence for Egyptian, Gulf, Levantine,
  Sudanese and Iraqi. Register is social media, not customer service.
- Priority: medium-low. Download (123 MB) only when the re-labelling step is planned.
