# Source 09: Alexandria (UBC-NLP, ACL 2026), checked 2026-10-14

- Paper: https://aclanthology.org/2026.acl-long.1503/ · Repo: https://github.com/UBC-NLP/Alexandria (prompts,
  translation / revision guidelines, eval code) · Data: https://huggingface.co/datasets/UBC-NLP/alexandria
  (not gated, CC BY-NC 4.0). Local: `data/sources/alexandria/<COUNTRY>/<split>.parquet` (15 MB, all 34 files).
- How it was made: LLM-generated **English** multi-turn conversations in 11 practical domains → translated by native
  speakers into their **city dialect** → revised by a second native speaker (translator_id, reviewer_id kept).
- Row = one conversation: conv_id, country, domain, dialect (city / sub-dialect), participants (roles),
  english_conversation, dialectal_conversation (turns with speaker, gender direction, text).
- Public splits only (train / dev / public test); the private test is held back. LY and SD have test only; LB has
  no dev.

## Size (public splits)
29,815 conversations · 93,172 turns · **1,236,802 dialect words**.

| Country | Conversations | Words | Sub-dialects |
|---|---|---|---|
| PS | 5,391 | 227K | Nabulsi, Albira (urban) + 9 village (falahi) dialects |
| LB | 3,655 | 146K | Lebanese standard |
| SA | 3,413 | 143K | Southern 2,377 · Hijazi 1,025 · Khaleeji 11 |
| OM | 2,834 | 114K | Suri, Rustaqi, Al-Wafi, Ibri, Seebi |
| SY | 2,569 | 108K | Levantine standard, Homsi |
| MR | 2,457 | 104K | Hassaniya |
| JO | 2,429 | 93K | Irbidi |
| EG | 1,700 | 73K | Cairene |
| YE | 1,704 | 70K | Taizi, San'ani, Central |
| MA | 1,522 | 69K | Standard Darija |
| TN | 1,422 | 58K | Tunisian |
| SD | 345 | 15K | Sudanese standard |
| LY | 374 | 14K | Misrati / Central |

Not parallel across dialects: each English conversation was translated into one dialect only (21 of 29,778
English openings appear in 2+ countries). Parallel = English ↔ one dialect.

## Quality check (hard MSA markers per 1,000 tokens; Latin-script share of turns)
- Almost no MSA in EG 0.0, LB 0.0, MA 0.0, SY 0.1, SD 0.2, JO 0.3, PS 0.6.
- Higher in YE 11.9 (هذا / هذه / الذي / أيضا / التي → some real MSA leakage), SA 6.9, TN 6.8, OM 4.3, MR 4.5
  (mostly هذا, which is natural in Gulf / Yemeni / Tunisian / Hassaniya speech, so not leakage by itself; Omani
  turns also show يمكنني-type service MSA).
- Code-switching (Latin words): TN 38%, MA 33%, LB 17% of turns, others 1–8%.

## Verdict
**High value.** Human-translated and revised, 13 countries with city labels, and the register is close to Tuner's:
commerce, healthcare, logistics, tourism, legal / financial dialogues (customer ↔ provider roles). Best use: natural
host sentences per country (family A) and the main source of **realistic dev / test sentences per dialect** for V2.
Limits: no word-level labels, no cross-dialect parallelism, no Iraqi / Gulf-proper (UAE, Kuwait, Qatar, Bahrain) /
Algerian; LY and SD are small.
