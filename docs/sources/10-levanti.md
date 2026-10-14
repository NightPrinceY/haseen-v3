# Source 10: Levanti (guymorlan/levanti), checked 2026-10-14

- HF: https://huggingface.co/datasets/guymorlan/levanti (CC BY-NC 4.0, not gated, 2024-07). One file
  `levanti_v1.csv` (212 MB), local in `data/sources/levanti/`. Author Guy Mor-Lan (HUJI); used for the
  levantitranslate.com MT / diacritizer models.
- Columns: dialect, arabic, hebrew, english, synthesized, diacritized, hebrew_taatik_EXP, english_translit_EXP.
  English translations were made by GPT-4o from the Arabic + Hebrew; diacritics (113K Palestinian rows) by Claude
  3.5 Sonnet + heuristics.

## Size (508,279 sentences; 1,029 duplicate Arabic strings)
| Dialect | Real (collected, human translated) | Synthetic (Claude 3.5 Sonnet from dictionary entries + examples) |
|---|---|---|
| Palestinian | 12,415 | 157,180 |
| Syrian | 7,636 | 130,096 |
| Egyptian | — | 156,256 |
| Lebanese | 2,093 | 22,550 |
| Jordanian | 6,445 | — |
| "Levantine" (unknown country) | 13,608 | — |
| **Total** | **42,197 sentences / 275K words** | **466,082 sentences / 5.11M words** |

## Quality checks
- Sentences containing hard MSA markers (هذا / هذه / الذي / التي / سوف / لن / لم / ليس): real 0.2–4.2%
  (Levantine 4.2%, Palestinian 3.2%, Jordanian 2.7%); synthetic **≈ 0.0–0.04%**: the synthetic part was clearly
  filtered for MSA.
- Cross-dialect markers in synthetic text are low: Syrian synthetic has 1.4% Lebanese هيدا-type markers; Egyptian
  synthetic has 51% sentences with Egyptian markers and ≈ 0 Levantine ones.
- Domain: real part = subtitles, social media, news-like Palestinian / Jordanian text (some political content);
  synthetic part = everyday sentences built around dictionary words (good lexical diversity: 92K Palestinian,
  83K Syrian, 95K Egyptian word types).

## Verdict
**Useful, with a clear rule.**
- Real 42K: natural Levantine host sentences (Palestinian / Syrian / Jordanian / Lebanese labels).
- Synthetic 466K: large, clean, MSA-free dialect text with high lexical coverage, good for training hosts and
  for corpus frequency of colloquial words, but LLM-written, so **never in dev / test**, and Levantine
  sub-dialects can blend (Syrian ↔ Lebanese).
- Adds no new countries beyond Levant + Egypt; the value is volume for those.
