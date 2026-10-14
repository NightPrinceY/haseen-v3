# HaSeen-3-Bench (test_v3, v3.3) — natural, challenge, agent and lexical-probe parts

Word-level dialect-adherence gold for the six V3 targets: **MSA, Egyptian (EGY), Saudi (SAU), Gulf (GLF = Saudi +
Gulf-east + Yemeni), Levantine (LEV), Iraqi (IRQ)**. It was built and frozen **before any V3 training data existed**,
from human-transcribed sources that no earlier HaSeen model ever saw. Integrity: `MANIFEST.json` (sha256 per file).
Build: `pipeline/scripts/build_test_v3.py`, `build_challenge_v3.py`, `build_probe_v3.py`, `gen_agent_v3.py`,
`apply_msa_review_v33.py`.

## Parts (sentences / scored words / gold deviations)
| Part | dev | test | What it measures |
|---|---|---|---|
| natural | 3,176 / 36.9K / 0 | 6,254 / 71.3K / 0 | false alarms on native speech (every word conforms) |
| challenge | 1,758 / 22.4K / 2,568 | 3,484 / 43.2K / 4,996 | recall: 1–3 words swapped for another variety's word for the same MADAR concept, plus MSA-drift injections (اللي→الذي، مش→لن، رح→سوف، هـ→سـ، ده→هذا) |
| agent | — | 200 replies / 3.9K words / 121 | real LLM replies asked to answer "in dialect T" (Qwen2.5-7B, SILMA-9B, Fanar-1-9B, Gemma-2-9B), labelled word by word |
| probe | — | 900 items / 3,964 checks | 1,782 held-out mo3jam dictionary entries: the head word must deviate from targets it does not belong to and conform to its own |

Per target (test): natural MSA 754 / EGY 1,200 / GLF 1,500 / SAU 1,200 / LEV 800 / IRQ 800 sentences; challenge
≈ 550–600 sentences per target.

## Sources (human-transcribed only; model-transcribed or synthetic text is never used in this bench)
| Target | natural sources |
|---|---|
| MSA | MGB-2 (Al Jazeera, test / validation), QCRI ArabicCulturalQA (MSA side) |
| Egyptian | MasriSpeech validation, MGB-3, ArabicCulturalQA (Egyptian side, native post-edited) |
| Saudi | SADA held-out shows (Najdi, Hijazi), SDAIA SCC |
| Gulf | SADA held-out shows (Najdi, Hijazi, Khaliji), SCC, ArabicCulturalQA (Gulf side), Lisan Yemeni (held-out hashes) |
| Levantine | Curras, Nâbra, Baladi, Levanti native, ArabicCulturalQA (Levantine side) |
| Iraqi | Lisan Iraqi, Iraqi factlogic (held-out hashes) |

SADA dev / test shows are speaker-disjoint from training (speaker ids from `mabahboh/sada2022-speakers`; 89 shared
speakers removed). Every held-out sentence hash is excluded from the training inventory.

## Gold rules
- natural: every word conforms; sentences with a hard MSA marker, ≥ 2 formal function words or > 30% MSA-skewed
  words are dropped from dialect sets (MSA drift is tested by challenge). Words whose status is "mixed" for the
  target are marked `ignore` and never scored.
- challenge: a swapped word is kept only if it is never listed for the target's cities, unused in the target's
  human training text (≥ 3 hits) and not a valid MSA word (formal open-class words are shared by every variety)
  unless it is an MSA closed-class form or a hard MSA marker. Maghrebi swaps are capped at 25%, MSA at 30%.
  25% of MADAR concepts are held out of training: `heldout_concept` marks them (generalisation, not memorisation).
- agent: labelled by Claude (not native-verified; the native reviewer was unavailable). Deviations include
  Fanar's MSA futures (سنقوم، سأراجع), Egyptian words in "Saudi" replies, and Chinese / Russian text.

## Change log
- v3.1: counts rebuilt with SADA (v3.0 had counted Saudi words such as معلش as non-Saudi).
- v3.2: agent part added.
- v3.3 (before any test scoring): the MSA natural sets held colloquial MGB-2 talk-show sentences labelled as clean
  MSA («مصر هي اللي صدرت…», «زي ما قلت من شوية»). Every MGB-2 sentence of MSA natural dev (184) and test (374) was
  read; 24 dev and 46 test colloquial sentences were removed with their challenge hosts. Same rule on dev and test;
  no model output was used for test.

## Licence and leakage
Built from research-licensed sources (SADA CC BY-NC-SA; QCRI MGB-2 / MGB-3 / ArabicCulturalQA; SDAIA SCC; Birzeit
corpora; MADAR Lexicon: CMU-Q / NYUAD research licence). Private dataset, internal evaluation only, no redistribution.
The training mix shares **no sentence and no normalized 6-gram** with any file of this bench
(`pipeline/scripts/filter_overlap_v3.py`).
