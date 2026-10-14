# Data study for the training set (2026-10-12, in progress)

## Sources downloaded locally (data/sources, data/lexicons/raw)
| Source | Content | Size | License | Use |
|---|---|---|---|---|
| Dial2MSA-Verified (github khered20) | EGY 9.1k / GLF 6.6k / LEV 4.1k / MGR 3.3k tweet→MSA pairs, human-verified | 7 MB | research only | parallel injection EGY/GLF↔MSA; LEV/MGR → DIA label |
| oasst2-9k-translation (HF ahmedsamirio) | 9.4k assistant replies: EN / MSA / EGY (GPT-4o) | 10 MB | oasst2 Apache + GPT-4o output | **assistant-register** EGY hosts + MSA parallel |
| Arabic STS (PLOS ONE S1) | ~2.7k sentences EN→MSA+EGY, and →Saudi (professional) | 0.9 MB | CC-BY | 3-way parallel EGY/GLF/MSA |
| Nile-Chat SFT test split (MBZUAI-Paris) | 35.8k rows; train has short-translation_egy_msa 83k, oasst2 27k, ultrachat/tulu Egyptian assistant text | 10 MB (train 1.9 GB, not downloaded) | per subset | more EGY hosts / EGY-MSA pairs |
| Kaikki Wiktionary | Egyptian 1,481 entries, Gulf 817 (Kuwaiti-heavy) | 7 MB | CC-BY-SA | lexicon seed (EGY 698 / GLF 137 corpus-distinct forms) |
| CAMeL calima DBs | egy-r13 (GPL), msa-r13 (GPL), glf-01 + lev-01 (CC-BY) | 125 MB zips | mixed | inflection paradigms; GLF/LEV have **no glosses** → no gloss alignment |
| Not yet downloaded | HeshamHaroon/saudi-dialect-conversations (3.5k Najdi convs, Apache) | — | Apache | GLF hosts |

## Findings
- CAMeL EGY: 86k of 161k stems are ARAMORPH (MSA imports) → not a dialect lexicon by itself. Value = inflection.
- Corpus-distinct lists have topical noise (food words as "GLF") → lexicon tiers must be hand-curated; corpus test only as tie-breaker.
- **Label-scheme conflict in gold:** هل/يمكنك/حاليا are SHARED for EGY target (v1.2) but MSA in glf_experimental.
  Proposal (guideline v1.3, deviation flags unchanged): add `FRM` (MSA form with a distinct colloquial equivalent, accepted in service register) and `DIA` (Levantine/Maghrebi → output "unknown").
  Decision layer: EGY target conforms on FRM; GLF target (strict) deviates on FRM; MSA target conforms. Register policy becomes a runtime option (service | strict).
- الثانية (ordinal time) = SHARED, الثاني (= the other) = MSA: consistent in gold.
- Context matters: ممكن/تقدر/معاك labeled EGY inside MSA sentences → the model must learn context, not only a lexicon.

## Plan (next session)
1. Labeler v1.3: closed MSA marker list, ث list, curated EGY/GLF/DIA/FRM tiers, lexstats tie-breaker; uncertain tokens → ignore (-100). Validate on anchors + a new Claude-labeled dev set (≥98%).
2. Hosts: oasst2 EGY, Dial2MSA, STS, MADAR, Saudi conversations, Nile-Chat subsets (sandbox ~$0.01, ask first).
3. Injection from aligned parallel pairs (SimAlign in sandbox); mix ≈40% clean / 40% injected / 20% MSA+GLF matrices; DIA from LEV/MGR.
4. Hold out 25% of GLF lexicon lemmas from training → challenge v2 measures generalization (the frozen challenge reuses common markers).
5. D3: Claude labels ~1.5k real sentences (open-class coverage + dev/calibration).
6. G2 leakage checks (8-gram) vs gold; G3 Yahya rates 100 injections.
