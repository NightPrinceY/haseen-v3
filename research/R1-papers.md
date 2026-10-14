# R1 — Paper deep-dive (2026-10-12)

Search via the HF Papers API (12 queries, raw results in `papers/s_*.json`). Full markdown of the key papers in `papers/p_*.md`.

## Must-know papers

### 1. The Arabic Generality Score (AGS) — arXiv 2508.17347 (2025) ★★★
The closest published work to our task. Word-level score: how widely a word is used across dialects + MSA.
- **Data:** MADAR-6 (Cairo, Beirut, Doha, Tunis, Rabat + MSA; 12k sentences), MADAR-26, **MADAR-CODA** (2,000 CODA-normalized sentences), **MADAR Lexicon** (1,045 concepts, 47,466 dialect entries with CODA + CAPHI phonology).
- **Word alignment:** AWESOME-align, each dialect sentence aligned to the **MSA anchor**; words aligned to the same MSA word are treated as mutually aligned. Alignments aggregated over the corpus per (word, dialect).
- **Etymology-aware edit distance:** substitution cost = 1 − P(same etymological letter), learned from MADAR Lexicon + CAPHI (so قلب / ألب / جلب count as the same word: q → ʔ in Cairo/Beirut, g in Gulf).
- **Model:** CAMeLBERT-Mix fine-tuned to regress AGS of a word **in context**, target word marked with `[TGT] w [TGT]`.
- **Sentence level:** harmonic mean of the k lowest word scores (one specific word dominates).
- **Eval:** NADI-2024 MDID (multi-label validity) → beats B2BERT and NADI baseline.
- **Stats:** MSA has the most *specific* words (~39%); Doha and Beirut are the most *general* (Doha >43% general).

**What we take:** (a) MSA-anchored alignment over MADAR to get word-level correspondences EGY ↔ GLF ↔ MSA → exact labels for synthetic mixing and for SHARED detection; (b) etymology-aware matching so spelling/phonology variants are not counted as different words; (c) `[TGT]`-marked contextual classifier as a fallback/System-2 style per-word model; (d) "general" ≠ "MSA": ليش، مافي، شوي are dialectal yet general across Gulf/Levant — they still deviate from Egyptian.

### 2. Revisiting Common Assumptions about Arabic Dialects (Keleg et al.) — arXiv 2505.21816 ★★★
MLADI dataset (NADI-2024 extended): 978 DA tweets, validity judged by 3 annotators from each of 11 countries.
- **56%** of dialectal sentences are valid in multiple regional dialects; 116 valid in all 5 regions.
- Sentence length is a poor predictor of ambiguity; **ALDi correlates better (−0.52)**.
- **Published dialect cue lists are unreliable:** Egyptian cues precision 0.60–0.81, *distinctiveness* only 0.35–0.38; **Gulf cue lists precision 0.0** (their Gulf cues شنو، علامك، مواعين are valid elsewhere); recall 0.03–0.15.
- **False friends:** ماشي (Egy "okay" vs MSA "walking"), حد (Egy "someone" vs MSA "limit") → context decides.

**What we take:** a static marker lexicon cannot be the decider; it must be curated by us, validated, and used as a *feature* inside a contextual model. Our SHARED class and multi-validity view are supported by data. Leaderboard / data: [MLADI Space](https://huggingface.co/spaces/AMR-KELEG/MLADI), code [MLADI-assumptions-revisiting](https://github.com/AMR-KELEG/MLADI-assumptions-revisiting).

### 3. POS Tagging for Improving Code-Switching Identification in Arabic (Attia et al. 2019) ★★
Function words tend to come from the dialect, content words from MSA. POS features improved MSA–EGY word-level LID (best reported diglossic word-level F1 ≈ 91.9 per the 2025 survey).

### 4. Can LLMs Understand, Reason About, and Generate Code-Switched Text? — arXiv 2601.07153 (2026) ★★
Human evaluation of synthetic code-switching strategies: **"Grammar forcing (target)"** — keep the matrix language's grammar and insert foreign words — is rated most natural and accurate; random word replacement is least natural. Metrics for describing mixing: **CMI** (Code-Mixing Index) and **SPF** (Switch-Point Fraction).

**What we take:** synthetic data = Egyptian matrix sentence (Egyptian grammar) + inserted Gulf/MSA words at aligned positions, not random swaps. Report CMI/SPF of our synthetic set vs the gold set to show distribution match.

### 5. NuNER: encoder pre-training via LLM-annotated data — arXiv 2402.15343 ★★
GPT-3.5 annotated 1.35M sentences → a 125M encoder that beats GPT-3.5 and competes with GPT-4 after fine-tuning. **Annotation diversity and dataset size matter most.** LLMs are bad at positions → ask for strings and align by exact match.

**What we take:** validates "big model labels → small encoder learns" (our Claude-teacher plan). Diversity of contexts > raw volume. Teacher output format must be per-token with the token string, re-aligned by exact match.

### 6. Automated Annotation with Generative AI Requires Validation — arXiv 2306.00176 ★
LLM annotation quality varies by task; always validate against human labels task-by-task. → Yahya's verified gold set + measuring teacher-vs-gold agreement is mandatory and goes in the README.

### 7. Others worth citing
- ALDi (2310.13747): sentence dialectness 0–1; Sentence-ALDi model on HF (`AMR-KELEG/Sentence-ALDi`).
- Exploiting DID in CODAfication (2407.03020): dialect normalization to CODA for 5 city dialects; code/data/models public.
- AraDiCE (2409.11404): LLMs still weak at dialect ID and generation (Jais/AceGPT > multilingual).
- EZSwitch (2410.22660): equivalence-constraint-guided CS generation with LLMs ([code](https://github.com/gkuwanto/ezswitch)).
- AlcLaM (2407.13097): dialectal BERT trained on 3.4M social-media sentences, claims better dialect performance than CAMeLBERT/MARBERT with far less data ([HF: rahbi](https://huggingface.co/rahbi)) → candidate backbone.
- B2BERT (`AHAAM/B2BERT`): multi-label sentence DID, 11 dialects → possible sentence-level auxiliary signal.
- AMIYA shared task (2602.16290): dialect fidelity in LLM generation (EGY, Saudi, …) → our problem is an active topic in 2026.
- AlexandriaX 2026 (2609.22796): first shared task on dialectal Arabic MT.

## Design consequences
1. **Data generation = MSA-anchored alignment over MADAR (+ etymology-aware matching)**, then grammar-preserving insertion into Egyptian matrix sentences. Exact labels, natural-looking mixes.
2. **Context is required** (false friends, multi-validity). Static lexicons only as features.
3. **Claude as teacher is supported by NuNER-style evidence**, but must be validated against a human gold set (2306.00176).
4. Candidate backbones to compare: CAMeLBERT-Mix (used by AGS), MARBERTv2, AlcLaM.
5. Report synthetic-vs-gold distribution (CMI, SPF) and teacher-vs-gold agreement in the README.
