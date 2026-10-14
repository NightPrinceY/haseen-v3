# HaSeen V2 — dataset overview and target decision (2026-10-14)

Evidence: 14 source studies (`research/sources/01–14`), the unified inventory (`scripts/build_inventory.py` →
`data/inventory/`), and three measurements (`scripts/separability.py`, `scripts/word_separability.py`,
`scripts/target_table.py`). Every number below comes from those files.

## 1. What we hold
- **Sentences: 2.74M rows** (`data/inventory/sentences.jsonl.gz`; 1.87M after exact de-duplication), each tagged
  with country-level variety, sub-dialect when known, source and kind (native / translated / spoken / synthetic).
- **Lexicon: 1.31M rows** (`data/inventory/lexicon.jsonl`): MADAR Lexicon (44K, expert, 25 cities), mo3jam (25K,
  crowd, 24 countries, voted), Lahga (21K, 24 dialects), Lingualism (7.5K, EGY / LEV / MAR) + token rows from the
  Birzeit corpora (Curras, Baladi, Nâbra, Lisan ×4; 1.17M tokens with lemma / POS / gloss).
- **Morphology**: CALIMA EGY / GLF / LEV / MSA (inflection, MSA validity).
- **Gaps**: no word-level labels anywhere (that is HaSeen's job); Kuwait / Bahrain / Qatar / UAE text is thin;
  MADAR Corpus-26 and FineWeb-2 counts are not yet in.

## 2. Which varieties can be told apart at the WORD level
Sentence-level dialect ID is the wrong yardstick: HaSeen flags words. Two controlled word-level measures:

**MADAR concept distance** (expert, complete 1,032 READY concepts × 25 cities): share of concepts where two
countries share no form. Closed-class only (167 concepts) in brackets.
- Practically identical: **Palestine–Jordan 0.00 (0.01)**, **Lebanon–Syria 0.01 (0.02)**,
  **Riyadh–Jeddah 0.03 (0.04)**.
- Every other pair ≥ 0.10 (most 0.15–0.45); e.g. Egypt–Sudan 0.18 (0.13), Saudi–Qatar 0.15 (0.10),
  Saudi–Oman 0.15, Yemen–Saudi 0.13, Iraq–Levant 0.13–0.20, Maghreb pairs 0.34–0.41, MSA vs any dialect 0.34–0.64.
- Function words: Egypt and South Levant share many (EG–JO 0.08 closed) but differ on open-class words (0.15).

**Alexandria deviation density** (same translation pipeline and domains for 11 countries): share of a speaker's
test tokens that never occur in the target's training text although they would be expected ≥ 5 times there.
- PS→JO 0.01, JO→PS 0.01, LB→SY 0.02, SY→LB 0.02 — and the "absent" words are place names (البترا، نابلس، شيكل),
  not dialect words → confirms the merges.
- Gulf-side pairs are low but real: SA→OM 0.03 (ابغي، بكره، المويه، كويس), YE→SA 0.05, OM→SA 0.06.
- Clear pairs: EG→SA 0.13 (ده، مش، كده، اوي، دلوقتي، حضرتك), MA→TN 0.21 (ديال، هاد، واش، بزاف، دابا),
  Maghreb ↔ East 0.18–0.31.

**Sentence-level ID** (char 1–5-gram SVM), for reference: Alexandria-only macro-F1 0.62 (MA 0.96, MR 0.96, TN 0.89,
YE 0.76, OM 0.76, LB 0.72, EG 0.70; JO 0.52, PS 0.59, SA 0.60; LY / SD untrainable there). Main confusions
JO↔PS, SY↔LB, SD→EG, LY→SA/TN. Cross-source to speech (Casablanca / Habibi): macro-F1 0.46 — domain shift is large,
so evaluation must include spoken text.

Gulf-east grouping (no MADAR city except Doha): crowd lexicons (mo3jam + Lahga) show KW–QA 0.67, QA–BH 0.61,
QA–AE 0.59, BH–KW 0.51 overlap vs 0.32–0.34 with Oman → Kuwait, Qatar, Bahrain, UAE form one target.

**Merge rule (pre-set):** merge two countries into one target when MADAR distance ≤ 0.05 on all concepts and on
closed-class concepts; keep them separate otherwise.

## 3. Recommended targets: 15 (+ "other / unknown")
| Target | Countries | Human sentences (native + translated + spoken) | Human words | Lexicon forms | Eval sentences* | Tier |
|---|---|---|---|---|---|---|
| MSA | — | 70.9K | 957K | CALIMA-MSA 38.6K lemmas | 3.0K | 1 |
| EGY Egyptian | EG | 31.6K (+192K synthetic) | 314K | 7.5K | 3.8K (+ V1 gold 794) | 1 |
| SLEV South Levantine | PS, JO | 51.7K (+157K synth.) | 525K | 3.9K | 5.2K | 1 |
| NLEV North Levantine | LB, SY | 45.9K (+152K synth.) | 446K | 2.6K | 2.2K | 1 |
| IRQ Iraqi | IQ | 343.8K | 3.98M | 4.9K | 2.2K | 1 |
| SAU Saudi | SA (Najdi, Hijazi, Southern) | 17.6K (+22K synth.) | 199K | 6.6K | 2.3K | 1 |
| YEM Yemeni | YE | 46.5K | 1.20M | 4.2K | 2.7K | 1 |
| MOR Moroccan | MA | 21.0K | 190K | 5.5K | 4.2K | 1 |
| TUN Tunisian | TN | 15.3K | 124K | 3.6K | 1.1K | 1 |
| ALG Algerian | DZ | 640.6K (mostly noisy ASR-like transcripts) | 9.39M | 5.4K | 3.3K | 2 |
| KHL Gulf-east | KW, QA, BH, AE | 19.0K | 137K | 5.9K | 2.7K | 2 |
| OMN Omani | OM | 8.9K | 117K | 2.2K | 1.1K | 2 |
| SDN Sudanese | SD | 4.9K | 73K | 3.4K | 1.1K | 2 |
| LBY Libyan | LY | 5.0K | 71K | 2.5K | 1.1K | 2 |
| HAS Hassaniya | MR | 9.8K | 125K | 0.8K | 3.0K | 2 |
*Eval sentences = held-out human sets already in hand (Alexandria test, Casablanca, Habibi). Unassigned: 15.6K
"Levantine" and 3K "Maghrebi" sentences without a country.

- **Tier 1 (9)**: clearly separable, ≥ 15K human sentences, ≥ 2.5K lexicon forms, MADAR city coverage
  (function words), held-out native eval.
- **Tier 2 (6)**: separable but thin or noisy data (one or two sources, < 10K clean human sentences, weak lexicon
  or noisy corpus). Ship as "beta" with honest per-target numbers.
- 15 targets × multi-label bits per word is the V1 scheme (3 bits) scaled to 15; unknown bits stay masked.

## 4. How to evaluate 15 targets without 15 annotators
1. **Precision per target** (V1 egy_natural idea): native held-out sentences of T judged against target T should
   produce ~0 flags (Alexandria test, Casablanca, Habibi, Lisan / Nâbra held-out, V1 gold for EGY).
2. **Recall per pair** (V1 challenge idea): MADAR concept-parallel substitutions (expert forms) inserted into
   T-sentences → known deviating positions for every (T, other) pair.
3. **Real cross-dialect text**: A-speaker sentences judged against T; flag density compared with the measured
   Alexandria density (e.g. EG text vs SAU target ≈ 13 flags / 100 words).
4. **Native spot checks**: Yahya for EGY; a small paid or volunteer check per Tier-1 target (≈ 100 sentences).
5. Multi-label sanity: NADI-2024 dev (120 sentences × 8 countries, human validity labels).

## 5. Decisions for Yahya
1. Approve 15 targets (9 Tier 1 + 6 Tier 2) and the merge rule, or choose another level (5 regions / 11 MADAR
   sub-regions / 18 countries).
2. Request MADAR Corpus-26 (2,000 sentences × 25 cities, fully parallel) via the form — biggest remaining gain
   for recall tests.
3. Approve ≈ $0.1–0.2 sandbox for FineWeb-2 frequency tables (negative evidence at web scale for EGY, MOR, SAU,
   LEV, TUN, IRQ).
4. Whether to download fr3on (2M YouTube comments) for re-labelling as extra hosts.
