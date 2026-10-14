# Source 15: MADAR Parallel Corpus v1.1 (CAMeL Lab), obtained 2026-10-14

- Registered by Yahya (CAMeL Lab form, Kafrelsheikh AI email); download link from the confirmation email.
  Local: `data/sources/madar_corpus/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/` (README, LICENSE, paper,
  `MADAR_Corpus/MADAR.corpus.<City>.tsv`, columns `sentID.BTEC, split, lang, sent`). English / French are
  index-only (copyright). Git-ignored: licence = internal research and evaluation only, no redistribution,
  no modification of the dataset, commercial use needs CMU.
- **Corpus-26**: 2,000 BTEC sentences in all 25 cities + MSA, fully parallel (all 2,000 ids present in every
  file), MADAR shared-task splits per sentence: 1,600 train / 200 dev / 200 test.
- **Corpus-6**: 12,000 sentences for Cairo, Doha, Beirut, Rabat, Tunis + MSA (9,000 train / 1,000 dev /
  2,000 = the corpus-26 part). Same text as the V1 HF mirror.
- Hard-MSA-marker rate per 1,000 tokens (سوف، لن، لم، ليس، الذي، هذا، هذه، هناك…): near 0 in the Levant, Rabat,
  Tripoli, Benghazi; 2–3 in Egypt, Jordan, Jerusalem, Doha, Jeddah; 13–25 in Baghdad, Basra, Mosul, Muscat,
  Riyadh, Sana'a, Sfax, Tunis (largely هذا / هذه / هناك, which are natural there; part is real MSA leakage in the
  travel-phrasebook register). MSA itself 42.8. No dialect sentence is identical to its MSA sentence.

## Use in V2
- **Challenge gold**: the same sentence in 25 cities → natural, sentence-level substitutions for every pair of
  targets (swap an aligned word from city B into city A's sentence), plus whole-sentence cross-dialect tests.
- **Precision gold**: the 200-sentence test split per city (25 × 200 = 5,000 native sentences) for every target.
- **Alignment**: MSA-anchored word alignment across 26 versions → concept-level equivalents in context
  (complements the MADAR Lexicon's out-of-context concepts).
- Domain is travel phrasebook (short sentences, avg ~6.5 words), not customer service: use for coverage and
  recall tests, keep Alexandria / Casablanca / agent sets for register realism.
