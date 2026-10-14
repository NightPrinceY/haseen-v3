# R4 — Ready-made code, models, tools (2026-10-12)

## Big finds
### `VARabi/ALDi-Token-DI` (+ seeds -30, -50) ★★★
- **MARBERT fine-tuned for token-level MSA vs DA tagging**, trained on the **CALCS 2016 MSA–DA code-switched dataset** (W16-5805) — the same data LinCE wraps, which we could not download.
- Labels: `ambiguous, lang1 (MSA), lang2 (DA), mixed, ne, other` (CALCS scheme). First-subword labeling, whitespace tokens.
- Size: ~649 MB fp32 (MARBERT base). Part of ALDi (EMNLP 2023), code: [github.com/AMR-KELEG/ALDi](https://github.com/AMR-KELEG/ALDi).
- Uses for us:
  1. **Baseline** for the MSA dimension (target=EGY: `lang1` → deviation "Arabic / MSA").
  2. **Warm start**: initialize System 1 from it (it already knows MSA vs Egyptian tweets at token level), then add GLF/SHARED labels.
  3. **Feature / teacher signal** for MSA-vs-dialect on unlabeled text.
  - Limits: no Gulf vs Egyptian distinction; tweet domain (2016).

### `CAMeL-Lab/codafication` (MIT) ★★
- Repo **contains the MADAR CODA corpus** (Beirut, Cairo, Doha, Rabat, Tunis; 2k sentences each, raw + CODA) and raw↔CODA word alignments → no Google form needed.
- Models: `CAMeL-Lab/arat5-coda`, `arat5-coda-did` (DID-conditioned). Too slow for System 1; use offline to normalize training text / build spelling-variant maps.

### Alignment tools (for MADAR-based mixing)
- **SimAlign** (MIT, `pip install simalign`): embedding-based alignment, no parallel training data, works with any HF BERT → can use CAMeLBERT/MARBERT for Arabic dialect↔MSA.
- **AWESOME-align** (used by AGS): mBERT fine-tuned for alignment (`aneuraz/awesome-align-with-co`).
- Plan: align Cairo↔MSA and Doha↔MSA with SimAlign(itermax) on CAMeLBERT embeddings, MSA as anchor (AGS recipe); spot-check 50 alignments.

### Other ready models
- `VARabi/Sentence-ALDi` (sentence dialectness), `VARabi/NADI2024-baseline` (multi-label DID), `AHAAM/B2BERT` (multi-label DID).
- CAMeLBERT POS-EGY / POS-GLF token classifiers.
- CAMeL Tools: `DIDModel6` (sentence DID, 6 labels incl. Cairo, Doha, MSA), analyzers (msa/egy/glf/lev).

### Code-switching generation
- EZSwitch ([github.com/gkuwanto/ezswitch](https://github.com/gkuwanto/ezswitch)): equivalence-constraint CS generation with LLMs.
- Finding (2601.07153): matrix-language grammar + inserted words = most natural.

### Evaluation / annotation
- `seqeval`, sklearn P/R/F1; CMI and SPF for mixing statistics (ALDi-Token-DI README computes CMI).
- TSV review in a spreadsheet (decided).
