# R2 — Data, R3 — Models (2026-10-12)

## Problem validation (for the README "why this matters")
- **AMIYA shared task (VarDial 2026)**: LLMs asked to answer in a dialect. Baseline Llama-3.1-8B dialect-fidelity (ADI2) = **11.1% Egyptian, 0.9% Saudi**; LLMs fall back to MSA. Some fine-tuned systems "learned to output dialectal words or phrases at the expense of fluency". [overview](https://aclanthology.org/2026.vardial-1.1)
- **AL-QASIDA** (arXiv 2412.04193): LLMs *understand* dialect better than they *produce* it; they are "reluctant to generate DA"; post-training biases toward MSA. Its fidelity metric (ADI2) is **sentence-level** dialect ID.
- → Our detector is the **word-level** complement: it says *which words* broke the dialect, which a sentence-level ADI2 score cannot. Strong framing for Tuner (observability for voice agents).

## Data sources: availability, license, role

| Source | What | Size | Access | License | Role |
|---|---|---|---|---|---|
| MADAR-6 (HF mirror `Elfsong/MADAR_Corpus_Merged`) | Parallel Cairo/Doha/MSA/Beirut/Rabat/Tunis | 11,600 sentences | open mirror | MADAR research license (no redistribution, commercial needs permission) | **Core**: alignment-based mixing + SHARED detection. Do not redistribute raw text |
| MADAR-CODA | 10k raw + CODA spelling, 5 cities | 2k/city | Google form | CMU/NYUAD research license | Spelling normalization reference (Yahya would have to fill the form) |
| MADAR-26 + MADAR Lexicon | 25 cities incl. Riyadh, Jeddah, Cairo, Alexandria; lexicon 47k entries | 2k sentences; 1,045 concepts | request from CAMeL / MADAR site | research license | Nice-to-have (Saudi coverage). Not on HF |
| AOC-ALDi (`arbml/AOC_ALDi`) | 128k real reader comments, dialect label (EGY/GLF/LEV/MSA) + dialectness level | 102.9k / 12.3k / 12.7k | HF parquet (~21 MB) | research (AOC) | **Real mixed text** (dialect with MSA incursions) → teacher-labeling pool + realistic distribution |
| FineWeb2 Egyptian (arz) | Web/blog Egyptian | millions | HF | **ODC-BY** (commercial OK) | Egyptian matrix sentences for injection; production-safe |
| FineWeb2 Najdi (ars) | Saudi forum text | millions | HF | ODC-BY | Gulf source; production-safe |
| Gulf tweets 2018–20 | Cleaned Gulf tweets | 100k+ | HF | CC-BY-4.0 | Gulf source |
| Egyptian↔MSA parallel (`AdhamAshraf/egyptian-2-arabic`) | Colloquial ↔ formal | 18k pairs | HF | MIT | Extra EGY↔MSA alignment pairs (production-safe) |
| SADID+MADAR Egyptian (`RanaGaber/SADID_AND_MADAR`) | Egyptian + English + CODA text | — | HF | Apache-2.0 card (check MADAR part) | CODA-normalized Egyptian |
| AraDiCE (`QCRI/AraDICE-*`) | Post-edited Egyptian/Levantine versions of benchmarks | 14.5k (MMLU-egy) | HF | **CC-BY-NC-SA** | Optional EGY↔MSA pairs (pair with ArabicMMLU by ID); non-commercial |
| MLADI / NADI-2024 MDID | 1,120 tweets, validity in 11 dialects | — | MLADI Space / NADI registration | research | Multi-validity reference; sentence-level |
| LinCE MSA-EA | word-level MSA/EGY | ~20k tokens dev | **host down** | — | Unavailable |

### Licensing point for the production plan
MADAR / AOC / AraDiCE are research-only. For a commercial Tuner deployment the training mix must be rebuilt on production-safe sources (FineWeb2 ODC-BY, MIT/CC-BY corpora, Claude-generated data, Tuner's own call transcripts). The pipeline stays the same; only the inputs change. Say this explicitly in the README.

## Models

### System 1 backbone candidates
| Model | Params | Notes |
|---|---|---|
| `CAMeL-Lab/bert-base-arabic-camelbert-mix` | 110M | MSA+DA+CA; used by AGS for word-in-context dialect generality; POS-EGY/GLF variants exist |
| `UBC-NLP/MARBERTv2` | ~163M | 1B tweets, dialect-heavy; strong DID results (NADI-2023 F1 ~82.9) |
| `rahbi/alclam-base-v2` | base | Dialectal BERT (3.4M social sentences); claims > CAMeLBERT/MARBERT; very low adoption → risk |
| `jhu-clsp/mmBERT-small` / `-base` | small / base | ModernBERT multilingual (1,800+ languages), fast (FlashAttention, unpadding), long context; Arabic-dialect quality unknown → test |
| `bashar-talafha/multi-dialect-bert-base-arabic` | base | NADI-trained multi-dialect BERT (2020) |

Plan: compare 2–3 backbones on the same data (CAMeLBERT-Mix, MARBERTv2, mmBERT-small) — cheap on an L4.

### Auxiliary / feature models
- `VARabi/Sentence-ALDi` (269k downloads): sentence dialectness 0–1 → auxiliary head target or feature.
- `AHAAM/B2BERT`: multi-label sentence DID (11 dialects).
- CAMeLBERT POS-EGY / POS-GLF taggers: POS features (Attia 2019).
- `CAMeL-Lab/arat5-coda`: CODAfication (spelling normalization) — likely too slow for System 1; offline data cleaning only.
- Alignment: AWESOME-align (`aneuraz/awesome-align-with-co`, mBERT-based) or SimAlign-style alignment using CAMeLBERT embeddings (better for Arabic dialects; to test).

### System 2 / teacher candidates (local, open-weight)
ALLaM-7B-Instruct (Apache-2.0), Fanar-1-9B-Instruct (Apache-2.0), SILMA-9B (Gemma license), Qwen3.5-9B (Apache-2.0). JAIS-2 appears in AMIYA as a strong Arabic base. Command-R7B-Arabic is CC-BY-NC (not for production).
Teacher for training data: **Claude in-session** (no API, $0 compute), validated against Yahya's gold set.
