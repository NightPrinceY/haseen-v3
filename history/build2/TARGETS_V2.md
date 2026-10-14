# HaSeen V2 — final targets, gold plan, productisation (approved by Yahya 2026-10-14)

Inputs: `DATASET_OVERVIEW.md` (14 sources, separability), my first call with Tuner, Tuner's docs (docs.usetuner.ai) and the Arabic voice-agent market.

## 1. What Tuner needs (why this shape)
- Tuner = reliability layer for voice agents: **simulation before production, observability + monitoring after**.
  Businesses replacing human support with agents lose QA; Tuner gives evals, red flags and alerts per call.
- From the first call: most agents do not stick to the accent / dialect they are expected to use, evals in this
  area are inaccurate, and Arabic is the hard case. Research must be small challenge, large business gain, fail
  fast on open models, then productionise.
- Product gaps found in the docs: (1) all 21 evals are LLM judges ("they read Arabic, no setup") — LLMs are weak at
  dialect ID (V1 research: AMIYA, GPT-4 ≈ 46 F1 on DID); (2) simulation offers **"Arabic" as one language with no
  dialect**, while English has US / UK / AU accents; (3) no dialect-adherence eval, no caller-dialect field.
- Market: > 90% of GCC calls are in dialect; demand is led by Saudi and Gulf, then Egyptian, then Levantine.

## 2. Final label space
**9 production targets** (each word gets one naturalness bit per target; unknown bits masked):

| Target | Varieties | Verified sentences | Held-out eval | Expert / 2-source forms |
|---|---|---|---|---|
| MSA | Modern Standard Arabic | 35.0K | 3.0K | CALIMA-MSA 38.6K lemmas |
| EGY | Egyptian | 30.1K | 3.8K (+ V1 gold) | 4.8K |
| SAU | Saudi: Najdi, Hijazi, Southern | 12.3K | 2.3K | 1.6K |
| KHL | Gulf-east: UAE, Qatar, Kuwait, Bahrain | 12.7K | 2.7K | 1.0K |
| LEV | Levantine: Palestinian, Jordanian, Lebanese, Syrian (merged 2026-10-14) | 68.5K | 7.4K | 2.6K |
| IRQ | Iraqi | 5.5K (+ 326K filtered web) | 2.2K | 1.4K |
| YEM | Yemeni | 45.5K | 2.7K | 1.8K |
| MOR | Moroccan | 18.6K | 4.2K | 2.3K |
| TUN | Tunisian | 14.5K | 1.1K | 1.2K |
Verified = written, translated or transcribed by native speakers through a documented process (Alexandria, MADAR,
Casablanca, Habibi, Curras / Baladi / Nâbra / Lisan, STS, Dial2MSA, Nile-Chat test). Crowd, LLM-written and
classifier-filtered text are excluded from these counts.

**Recognised as "other dialect" (not offered as targets):** Omani, Sudanese, Libyan, Algerian, Hassaniya. Their data
(MADAR cities, Lisan, Alexandria, Casablanca, mo3jam) is used as deviation evidence: a Libyan word inside a Saudi
agent's reply is flagged "Arabic / other dialect". They can be promoted later when verified data reaches the bar.
Non-Arabic tokens (names, numbers, Latin script) stay "other" as in V1.

## 3. Data policy (verified first)
- **Labels** come from expert evidence: MADAR Lexicon (complete 25-city × 1,042-concept matrix → sense-level
  negative evidence), Birzeit lemma alignments, Lingualism lists, CALIMA (morphology, MSA validity).
  Crowd lexicons (mo3jam, Lahga) count only with net votes ≥ 3 or when two independent sources agree. Corpus
  frequency (human corpora, later FineWeb-2) is a tie-breaker, never the only evidence. Unknown stays masked.
- **Hosts** (sentences the labeler annotates): verified human text first; filtered web (Iraqi factlogic, Algerian
  ASR, fr3on) only after our classifier and labeler agree; LLM-written text (Levanti synthetic, oasst2) only for
  training, never for dev or test.
- **Splits** by source document; every eval set comes from a source or split never used in training.

## 4. Gold (HaSeen-Bench v2), per target
1. **natural** — 300 held-out native sentences (Alexandria test, Casablanca, Habibi, Lisan / Nâbra held-out).
   Expected: almost no flags. A native checker reviews only the flagged words.
2. **challenge** — 300 target sentences with 1–3 controlled substitutions from MADAR's forms of the same concept in
   other varieties (expert forms, exact positions, sense-checked).
3. **agent** — 150 real agent-style replies "in dialect T" written by several open-weight LLMs on an HF sandbox
   (the real failure mode), labelled by Claude, then verified by a native speaker (Yahya for EGY; one paid or
   volunteer native checker per other target).
4. **multi-label sanity** — NADI-2024 dev (120 sentences × 8 countries).
**Ship gate per target** (pre-registered): natural false alarms ≤ 1% of words, challenge F1 ≥ 0.90, agent F1 ≥ 0.85.
A target that misses the gate is demoted to "other dialect" for that release.

## 5. How it plugs into Tuner (productisation + observability)
- **Eval "Dialect adherence"** (new eval category next to Hallucination / Politeness): per agent turn, % of words that
  conform to the configured dialect + the deviating spans with their label. PASS / FAIL by threshold, or SCORE.
- **Red flags / alerts**: "agent drifted to MSA", "Egyptian agent used Gulf words", trend per agent version.
- **Caller dialect** from STT turns (word bits aggregated per turn) → new eval "dialect match" (caller Saudi, agent
  Egyptian) and an analytics dimension.
- **Simulation**: lets Tuner offer "Arabic (Saudi / Egyptian / Levantine …)" and verify that the simulated caller
  actually speaks it.
- **Real-time guard** before TTS (optional): same model, flags or rewrites words.
- Runtime: one encoder, all 9 targets, ONNX int8 on CPU (V1: 11 ms p50 per turn), no LLM call, offline.

## 6. Model implications
- Output head: 9 naturalness bits + other-dialect + other (non-Arabic) + frm; masked BCE as V1.
- Pointer head with 10 options needs the V1 fix (a learned vector per option) — options like SLEV / NLEV are close;
  linear head stays the control run.
- Backbone: MARBERT family again first (dialect-heavy pretraining); test one multi-dialect alternative.

## 7. Next steps
1. MADAR Corpus-26 form (Yahya) — 2,000 sentences × 25 cities, best recall source.
2. Labeler v2: 10-bit scheme, MADAR concept matrix as negative evidence, voted crowd evidence.
3. Build gold natural + challenge sets per target (no annotators needed), then the agent set (sandbox, ~$1).
4. FineWeb-2 frequency tables on a sandbox (~$0.1–0.2) for tie-breaking.

## Update 2026-10-14: South and North Levantine merged into LEV (Yahya's proposal, accepted on evidence)
MADAR concept distance SLEV–NLEV 0.10–0.12 (lowest of all non-merged pairs), Alexandria cross density 2–5 words per
100, and only 20 of 934 Levant gold swaps were intra-Levantine; competitors also sell "Levantine" as one dialect.
Cost: function words differ (closed-class distance 0.23: هيدا / هاد, هلق / هسا), so a Lebanese هيدا in a Jordanian
agent is not flagged. Country is kept as metadata, so LEV can be split again later. Production targets: **9**.
