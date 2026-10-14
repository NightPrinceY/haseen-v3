# Research Landscape: Word-level Dialect Deviation Detection

Collected 2026-10-12 before choosing the architecture. Everything here was checked against a source (link given).
"Verify" marks things to confirm hands-on before relying on them.

---

## 1. State of the art (what the literature says)

| Finding | Why it matters for us | Source |
|---|---|---|
| Word-level LID is mostly studied for **MSA vs Egyptian**; best reported F1 is **91.9** for diglossic (MSA–dialect) code-switching | Realistic ceiling for a strong system; MSA–DA is the *hardest* pair across all languages studied | [CSW Arabic survey, COLING 2025](https://arxiv.org/pdf/2501.13419) |
| Attia et al. 2019: DL model + **POS features** beat SOTA; **function words tend to come from the dialect, content words from MSA** | Closed-class markers (negation, future, demonstratives, prepositions) carry most of the dialect signal. Content words are often shared → conforming | [Attia et al. 2019](https://aclanthology.org/W19-4603/) |
| Zero-shot LLMs are **weak** at dialect ID (GPT-4 ~45.6 F1-macro; small LLMs as low as 8 F1, biased toward Egyptian); fine-tuned MARBERT/AraBERT do much better (~82.9 F1 on NADI-2023) | Don't use a 7–9B LLM as the main decision maker or as a blind teacher. Fine-tuned encoders are the proven path | [arXiv 2509.13775](https://arxiv.org/pdf/2509.13775), [ArabicNLP 2023](https://aclanthology.org/2023.arabicnlp-1.64) |
| NADI 2024 moved dialect ID to **multi-label**, because varieties overlap | Supports our `SHARED` class instead of forcing one label per word | [NADI 2024](https://arxiv.org/pdf/2407.04910) |
| ALDi: dialectness as a 0–1 score (sentence level) | Useful auxiliary sentence head / feature | [ALDi](https://arxiv.org/html/2310.13747v1) |
| MARBERT (dialect-heavy tweets) > MSA-only models on dialect tasks; CAMeLBERT-Mix recommended for DA | Backbone choice: dialect-aware pretraining matters more than size | [MARBERT](https://arxiv.org/pdf/2101.01785), [CAMeLBERT](https://arxiv.org/pdf/2103.06678) |
| Morphological code-switching (a dialect clitic on an MSA stem) is usually labeled "mixed" | Our morphology rule: flag a word only when it carries a positive marker of another variety | survey §7.1 |
| LLMs mix dialects (the problem Tuner describes) is under-studied at word level | Our task sits in a real research gap: good README story | survey §8 |

---

## 2. Datasets

### 2a. Word-level labels (rare, most valuable)
| Dataset | Content | Size | Access | Use |
|---|---|---|---|---|
| **LinCE `lid_msaea`** (CALCS 2016) | Tweets, labels: `lang1`(MSA) `lang2`(EGY) `ne` `ambiguous` `mixed` `other` | dev: 1,332 tweets, ~22k tokens; train larger | HF `lince-benchmark/lince` (script-based loader; test labels hidden, scored on LinCE site) | Real training data for MSA↔EGY; external eval |
| LCS corpus (Al-Badrashiny & Diab 2016, "LILI") | Forum posts, token-level MSA vs EGY / GLF / LEV | 1,170 posts, ~27k tokens | Paper [C16-1115](https://aclanthology.org/C16-1115/); availability **verify** (likely LDC) | Only resource with Gulf at token level |
| BAEC | Saudi/Egyptian/MSA ↔ English | 45k words | [GitHub TaghreedT/BAEC](https://github.com/TaghreedT/BAEC) | Mostly Arabic–English; low priority |

### 2b. Sentence-level dialect corpora (for synthetic data & weak labels)
| Dataset | Content | Access |
|---|---|---|
| **MADAR corpus** | Same sentences in 25 city dialects + MSA (travel domain) — parallel! | [request form](https://camel.abudhabi.nyu.edu/madar-parallel-corpus/); mirror `Elfsong/MADAR_Corpus_Merged` on HF (check license) |
| **FineWeb2-Egyptian (arz)** | 10M+ docs Egyptian web text | `Omartificial-Intelligence-Space/FineWeb2-Egyptian-Arabic` (ODC-BY) |
| FineWeb2-Najdi (ars), FineWeb2-MSA | Saudi Najdi / MSA web text | same author on HF |
| Gulf tweets 2018–2020 | Cleaned Gulf tweets | `AhmedSSabir/Gulf-Arabic-Tweets-2018-2020` (CC-BY-4.0) |
| Gumar | 110M words Gulf forum novels | [paper](https://arxiv.org/pdf/1609.02960); web interface, download **verify** |
| AOC | 108k comments, MSA / EGY / GLF / LEV + dialectness level | [GitHub sjeblee/AOC](https://github.com/sjeblee/AOC) |
| DART | 25k tweets, 5 dialect groups | [Dropbox link in Arabic-Resources](https://github.com/NNLP-IL/Arabic-Resources) |
| NADI 2020–2024 | Country-level tweets; 2024 multi-label dev | registration required |

### 2c. Parallel & lexical resources (to build the dialect lexicon)
| Resource | Content | Access |
|---|---|---|
| **MADAR lexicon** | 1,045 concepts × 25 cities (~45 words per concept) | MADAR site |
| **Tharwa** | 73k Egyptian ↔ MSA ↔ English entries with POS, root, pattern | [L14-1115](https://aclanthology.org/L14-1115/), "publicly available" — download **verify** |
| Egyptian ↔ MSA parallel (18k pairs) | Slang/colloquial → formal | `AdhamAshraf/egyptian-2-arabic` (MIT) |
| Egyptian ↔ English (50k / 100k+) | Parallel corpora | `oddadmix/egyptian-arabic-english-translation-50k`, `Mo-Abdalkader/Egyptian-Arabic-English-Parallel-Corpus` |
| NileULex | Egyptian + MSA sentiment lexicon | [GitHub](https://github.com/NileTMRG/NileULex) |

### 2d. Guidelines
- **CODA\*** (Conventional Orthography for Dialectal Arabic): spelling conventions for EGY/GLF; base our normalization and annotation guideline on it. [camel-guidelines](https://camel-guidelines.readthedocs.io/en/latest/orthography/)
- CALCS LID label scheme (`lang1/lang2/ne/ambiguous/mixed/other`): map to ours.

---

## 3. Tools & libraries

| Tool | What it gives us | Notes |
|---|---|---|
| **CAMeL Tools** | Morphological analyzers with built-in DBs **`calima-msa-r13`, `calima-egy-r13`, `calima-glf-01`** (all ship with the toolkit); normalization; sentence-level DID (MADAR-26 char/word LMs) | 3-way "is this word analyzable in MSA / EGY / GLF, with which POS?" = strong per-word features. [docs](https://github.com/camel-lab/camel_tools) |
| CAMeLBERT POS taggers (EGY, GLF) | `CAMeL-Lab/bert-base-arabic-camelbert-mix-pos-egy` / `-pos-glf` (token classification) | POS features (Attia 2019 showed POS helps) |
| CAMeL dialect lemmatizers | `CAMeL-Lab/EGY-S2S-lemmatizer`, `GLF-S2S-lemmatizer` (T5) | Lemma-level lexicon lookup |
| GlotLID v3 (fastText) | 2,102 labels incl. Arabic varieties (arz, ars, afb, apc…) | Char n-gram backoff for OOV words; known to confuse Gulf labels. [HF cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid) |
| HF `transformers` token classification | `word_ids()` alignment, first-subword labeling | standard |
| `seqeval` / sklearn | P/R/F1 per label | eval |
| Temperature scaling | Calibrated per-token confidence | few lines of code |
| `optimum` + ONNX Runtime | Export & int8-quantize the encoder; CPU/GPU serving | production |
| **Argilla** (HF Space) | Span/token annotation UI with pre-filled suggestions; annotator corrects | Gold set verification by Yahya. [docs](https://docs.argilla.io/latest/tutorials/token_classification) |
| HF Jobs (`hf jobs uv run`) | Paid GPU runs from the terminal | see `compute-budget.md` |
| HF router (Inference Providers) | 126 chat models incl. Qwen3.5/3.8, Gemma-4, Llama-3/4, DeepSeek-V4, Kimi, Aya-Expanse, Command-R7B-Arabic | Generate *naturally* dialect-mixed Egyptian text for the gold set (the real phenomenon). Account not PRO → needs small billing balance |

## 4. Models

### Encoders (System 1 candidates)
| Model | Size | Pretraining | Notes |
|---|---|---|---|
| `CAMeL-Lab/bert-base-arabic-camelbert-mix` | 110M | MSA + DA + CA | Tagged Egyptian/Gulf; fits local GPU |
| `CAMeL-Lab/bert-base-arabic-camelbert-da` | 110M | DA only | Compare with -mix |
| `UBC-NLP/MARBERTv2` | ~163M | 1B tweets (dialect-heavy) | Strong on dialect tasks; tight on 3.68 GiB |
| AraBERTv02-twitter | 136M | MSA + tweets | Baseline alternative |
| Arabic ModernBERT variants | base | MSA-heavy, experimental | Low adoption; skip unless time |

### Decoders (Arch F backbone / System 2 / teacher candidates)
| Model | Size | License | Notes |
|---|---|---|---|
| `humain-ai/ALLaM-7B-Instruct-preview` | 7B | Apache-2.0 | Arabic-first |
| `QCRI/Fanar-1-9B-Instruct` | 9B | Apache-2.0 | Arabic-first (Gemma-2 based) |
| `silma-ai/SILMA-9B-Instruct-v1.0` | 9B | Gemma license | Arabic-first |
| Qwen3.5-9B / 2B | 2–9B | Apache-2.0 | Multilingual; base of Jev/strands |
| Falcon-H1 Arabic | 7B | **verify** repo name | Tops 2026 Arabic leaderboards (per annota8 blog) |
| `CohereLabs/c4ai-command-r7b-arabic` | 7B | **CC-BY-NC** | Not usable in production |

### System One decision models (studied, see architectures.html)
strands-decider-2B, JEV-9B/27B, CLM-8B: all English-trained, sentence-level decisions; per-word use needs N calls.
Best fit: optional System 2 judge, or a sentence-level gate. Their *design* (frozen backbone + small calibrated head) is what we borrow.

---

## 5. Product context (Tuner = voice agents)

- Agent LLM writes text → **TTS speaks it**. A Gulf/MSA word in an Egyptian agent sounds wrong to the caller.
- Two deployment modes:
  1. **Real-time guard before TTS**: must add only a few ms to the turn → encoder System 1 is mandatory; System 2 only if the budget allows.
  2. **Post-call evaluation** on transcripts / logs (batch): System 2 escalation affordable.
- STT transcripts have no diacritics and noisy spelling → normalization must be robust (CODA-style).

---

## 6. Ideas this research produced

1. **Closed-class marker lexicon first.** Per Attia 2019, function words carry most of the dialect signal. A curated list of EGY/GLF/MSA markers (negation, future, demonstratives, question words, prepositions, discourse particles) gives high-precision features and rules.
2. **3-way morphological evidence.** For each word, run the MSA, EGY and GLF analyzers: "analyzable only in GLF" is a strong deviation signal; "analyzable in all" → likely SHARED.
3. **Claude as teacher, not a 7B LLM.** Build the dialect lexicon and labeled synthetic sentences in-session (no API, $0 compute). Verify lexicon entries against MADAR / Tharwa.
4. **Synthetic injection at scale.** Swap words in real FineWeb2-arz sentences with GLF/MSA equivalents → exact labels for free.
5. **Gold set from the real phenomenon.** Ask 4–6 different LLM families via the HF router to write Egyptian customer-service replies; their natural slips are the test cases. Claude pre-labels, **Yahya verifies** (Argilla or TSV). Never generated by the same process as training data.
6. **Suggest the fix, not just the flag.** For each deviation, return the Egyptian equivalent from the lexicon (أبغى → عايز, سوف → هـ). A real product feature for a pre-TTS guard.
7. **Calibrated abstention.** Temperature-scaled confidences; below a threshold → label `unknown` (the task allows it) or escalate to System 2.
8. **Data flywheel for production.** Customer transcripts → teacher labels → retrain student. Write it in the production plan.

---

## 7. Open items to verify hands-on
- LinCE `lid_msaea` loads with current `datasets` (script loader may need an older version or the raw files from ritual.uh.edu).
- MADAR corpus / lexicon license and download (form).
- Tharwa download location.
- LCS (Gulf token-level) availability.
- Falcon-H1 Arabic exact HF repo id.
- GlotLID Arabic label list.
- Encoder training memory on the 3.68 GiB GPU (CAMeLBERT vs MARBERTv2).
