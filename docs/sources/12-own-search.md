# Source 12: own search for the gaps (2026-10-14)

Gaps after sources 01–11: (1) negative evidence = big per-dialect corpora; (2) missing countries: Gulf proper
(UAE, Kuwait, Qatar, Bahrain), Iraq, Algeria; (3) spoken transcripts (closest to voice agents); (4) multi-label
gold (one sentence valid in several dialects); (5) multi-dialect parallel data.

## Found and fetched (`scripts/fetch_text_sources.py`, text only)
| Resource | What | Size | Licence | Gap |
|---|---|---|---|---|
| factlogic/iraqi-arabic-dialect-corpus | Iraqi YouTube / Telegram comments 2010–2026, ensemble-filtered (NADI + MADAR + GlotLID + ALDi + lexicon), tiers high / medium, toxic flag, sub-dialect hint | 326K texts, 4.1M words; "98.5–100% human-verified precision" | CC BY-NC | Iraq (2) |
| UBC-NLP/Casablanca (transcripts only) | Human-transcribed speech, 8 countries: Algeria, Egypt, Jordan, Mauritania, Morocco, Palestine, **UAE**, Yemen | 13.6K segments (dev + test) | CC BY-NC-ND | spoken (3), UAE + Algeria (2) |
| SWivid/Habibi (transcripts only) | TTS benchmark transcripts: ALG, EGY, **IRQ**, MAR, MSA, SAU, **UAE** | 11.4K utterances | Apache-2.0 (SAU from SADA: NC-SA) | spoken (3) |
| touati-kamel/algerian-darja-corpus | Algerian podcast / conversation transcripts, Arabic + Latin script, code-switching | 10K–100K rows | CC BY 4.0 | Algeria (2) |
| NADI-2024 subtask-1 dev (via LahjatBERT repo) | **Human multi-label validity**: 120 sentences × 8 countries (y / n) | 120 | research | multi-label gold (4) |

## Found, not fetched yet (bigger or need Yahya)
| Resource | What | Why not yet |
|---|---|---|
| **FineWeb-2** dialect configs (HuggingFaceFW/fineweb-2, ODC-BY) | GlotLID-labelled web text: ary 2.87 GB, arz 1.14 GB, ars 0.82 GB, apc 65 MB, aeb 32 MB, acm 11 MB, arq / ayp ≈ 0 (parquet sizes) | ~5 GB at 0.3 MB/s ≈ 5 h locally → better counted on an HF sandbox (frequency tables only). This is the main **negative-evidence** source. GlotLID labels are noisy (V1 saw MSA in "ary"). |
| **MADAR Corpus-26** (CAMeL Lab) | 2,000 sentences × **25 cities + MSA, fully parallel** (Corpus-6 we already have) | Google Form on the MADAR download page → Yahya fills it |
| SADA (MohamedRashad/SADA22) | 667 h Saudi speech with transcripts (several Saudi dialects) | large audio parquet; transcripts can be pulled like Casablanca if needed |
| LahjatBERT data (MBZUAI, VarDial 2026) | 18-country multi-label labels (GPT-4o + classifiers) for NADI 2020–2023 tweets | file holds tweet **ids** only; texts need NADI registration |
| QADI (QCRI) | 540K tweets, 18 countries | HF copy has tweet ids only; Twitter hydration no longer possible |
| Gumar (CAMeL) | 110M words of Gulf forum novels (6 GCC countries) | portal / request |
| Bahrain Corpus (LREC 2022) | 620K words Bahraini | "to be released"; not found online |
| GAC-6 | 1.7M Gulf tweets, 6 countries | not found as a download |
| ADI-20, ARCADE (speech DID) | 20 varieties / city-level radio | audio only, no transcripts |

## Takeaways
- **Iraqi** is now covered at scale (4.1M words), plus Iraqi speech transcripts (Habibi IRQ).
- **UAE** has spoken transcripts (Casablanca, Habibi) but little written text; **Kuwait / Qatar / Bahrain** remain
  thin everywhere except mo3jam's lexicon (Kuwaiti 2.7K, Qatari 0.9K, Bahraini 1.1K definitions). Gumar is the
  only big Gulf-proper text source → worth requesting.
- **Negative evidence** at scale = FineWeb-2 counts (sandbox) + our collected corpora (Alexandria, Levanti, Iraqi,
  fr3on, Curras, Casablanca/Habibi transcripts).
- **No word-level multi-dialect dataset exists** (checked 2025–2026 literature: ADI work is sentence / speech level;
  multi-label is sentence level, MLADI / LahjatBERT). A word-level, multi-label, many-dialect resource would be new.

## Fetched (2026-10-14)
- Casablanca transcripts (`data/sources/casablanca_text/`, via `scripts/fetch_transcripts_duckdb.py`, text columns
  only): 13,637 segments / ~153K words — Morocco 2,090 · Mauritania 1,906 · Jordan 1,696 · Egypt 1,692 ·
  Algeria 1,687 · UAE 1,626 · Yemen 1,606 · Palestine 1,334 (≈ 16–24K words each).
- Habibi transcripts (`data/sources/habibi_text/`): 11,424 utterances / ~152K words — MSA 2,962 · IRQ 2,198 ·
  ALG 1,727 · SAU 1,175 (Najdi 705, Gulf 299, Hijazi 171) · MAR 1,083 · UAE 1,048 · EGY 1,037 · UNK 125.
- Iraqi factlogic: `data/sources/iraqi_factlogic/data/{high,medium,medium_extended}` (326K texts).
- Algerian darja corpus: `data/sources/algerian_darja/` (11,151 rows + a domain-conditioned copy).
