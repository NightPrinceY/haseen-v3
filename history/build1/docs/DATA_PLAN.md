# Training data plan: HaSeen v1 (2026-10-12)

Status: study finished. The labeler and lexicons are built and audited. Building the dataset is waiting on 3 decisions (§9).
Companion files:
- `docs/DATA_STUDY.md` (sources)
- `src/haseen/labeler.py`
- `data/lexicons/`

## 1. What changed in the label scheme (guideline v1.3 draft)
The gold set exposed a conflict in single-label varieties. The same word gets a different label depending on the target:
- **هل / يمكنك / حاليا:** SHARED for Egyptian (Yahya's register answers), but MSA in the Gulf gold.
- **هذا / أكثر:** a deviation for Egyptian, but SHARED in the Gulf gold.

**Fix:** each word gets three naturalness bits, one per variety (EGY, GLF, MSA), each 1, 0 or unknown. It also gets three flags:
- `frm`: MSA-register word that casual Egyptian replaces.
- `dia`: Levantine or Maghrebi word.
- `other`: names, digits, Latin script.

How the bits are used:
- A word **deviates** for target t if and only if its t-bit is 0. Words with `dia` always deviate.
- The **output label** is a variety where the word is natural. Preference order:
  - Egyptian target: MSA, then Gulf. This matches Tuner's example (سوف → MSA).
  - MSA target: Egyptian, then Gulf.
  - Gulf target: MSA, then Egyptian.
  - `dia` words get "unknown".
- **Unknown bits are masked in the loss.** The labeler never guesses.

Examples:

| Word | EGY GLF MSA |
|---|---|
| سوف | 0 0 1 |
| هذا | 0 1 1 |
| هل | 1 0 1 (frm) |
| عشان | 1 1 0 |
| وش | 0 1 0 |
| شو | dia |
| ممكن | 1 ? ? |

Model impact: the head becomes three sigmoid outputs plus `dia`/`other`, instead of a 5-way softmax. The pointer variant scores the three variety options independently. The gold deviation flags are unchanged. Gold eval stays as it is.

## 2. Labeler (src/haseen/labeler.py): evidence order
1. **No Arabic letter → other.** Then the brand gazetteer and the CAMeL proper-name gazetteer: 4,037 names that are never common words. A frequent word is never treated as a name.
2. **Core lexicon** `core_v2.tsv`: about 230 hand-curated entries, with allowed proclitics and 20 context rules for homographs. Examples:

   | Word | Senses the rule separates |
   |---|---|
   | وش | what / face |
   | ابي | want / my father |
   | كذا | like this / several |
   | الحين | now / في ذلك الحين |
   | دول | those / countries |
   | لست | not / for a woman |
   | ودي | I'd like / and this |
   | هل | question / Gulf هالـ |
   | فلم | film |
   | سيبني | leave me / will build |
   | بده | with this / Levantine |
   | راح | went / Gulf future |

3. **Morphology.**
   - **سـ + imperfect → MSA-only.** Checks: the verb's other person prefixes exist in the corpus; 1st person keeps its hamza in writing; the word is not noun + possessive and not a loanword in the host corpus.
   - **هـ/حـ + imperfect, and ما…ش → Egyptian-only.** This rule is off in MSA hosts.
4. **Corpus evidence.** Two sources:
   - `lexstats.json`: web corpora.
   - `domain_counts.json`: matched-domain parallel data. Assistant replies, tweets and Najdi conversations; 715k EGY, 252k GLF and 840k MSA tokens.

   How the evidence is read:
   - **Validity.** A word counts as used in a variety if r ≥ 0.25 with at least 3 occurrences, or if it is simply frequent (≥ 30 in domain data, ≥ 50 on the web). It counts as avoided only if r < 0.10 while at least 10 occurrences are expected.
   - **MSA bit.** The matched-domain parallel data decides first, because translators to MSA drop colloquial ممكن.
   - **EGY bit.** Follows v1.2: formal open-class words are natural in Egyptian.
   - **Gulf-only words.** Need all of the following:
     - Egyptian avoidance in both Egyptian corpora.
     - Not a name.
     - Gulf morphology (يسوون، تقولين، هالـ) or one of the 240 corpus candidates Claude approved (`glf_corpus_approved_v1.txt`). 460 candidates were reviewed; places, foods and shared verbs were rejected.

### Audit results
- **Anchors (40 sentences):** every deviation caught, 0 false positives. One miss: وش at the start of an Egyptian host is read as "face"; injected Gulf words get their labels from their source instead. One token is unknown.
- **Every flagged type in about 800k host tokens** was reviewed with its contexts in 3 audit rounds. That found and fixed about 25 homograph or loanword false-positive patterns.
- **Remaining host deviation rate:** 0.1–0.3% of tokens, and these are real leaks. Examples: GPT-4o "Egyptian" writing أكثر or الثاني, tweets with لم or الذي.

## 3. Sources (all local, in data/sources)
| Source | Variety | Size | Register | License |
|---|---|---|---|---|
| oasst2-9k-translation | EGY + MSA parallel | 9.4k pairs (about 250k tokens per side) | assistant replies | Apache-2.0 + GPT-4o |
| Nile-Chat test, egy_msa + oasst2 | EGY + MSA parallel | 6k pairs | mixed / assistant | per subset |
| Dial2MSA-Verified | EGY/GLF/LEV/MGR + MSA | 3.3k / 3.2k / 3.1k / 2.8k tweets | social media | research only |
| Arabic STS | EGY + MSA; Saudi + MSA | 2.7k + 0.5k | captions | CC-BY |
| Saudi Najdi conversations | GLF | 22.5k turns | dialogue, includes customer service | Apache-2.0 |
| Kaikki Wiktionary | EGY / GLF lexicon | 1.5k / 0.8k entries | — | CC-BY-SA |
| CAMeL calima DBs | EGY / MSA / GLF / LEV | inflections, MSA stems, names | — | GPL / CC-BY |

- **Not downloaded:**
  - Nile-Chat train (1.9 GB). It has 83k more EGY↔MSA translation rows. Needs a sandbox.
  - MADAR. Already used in the pilot; research license.
- **Leakage (G2):** 0 shared 8-grams between any source and the frozen gold (7,933 gold 8-grams).
- **Dial2MSA "verified" MSA sides contain dialect words:** مش, حضرتك, and فاشلون as a Gulf leak. MSA hosts with a flagged dialect word are dropped from the clean-MSA pool.

## 4. Dataset recipe: D-v3, about 60k sentences
### D-A: natural hosts, about 45%
Real sentences labeled by the labeler, with the host variety known. They teach two things:
- What conforms: the formal register in Egyptian, Gulf words in Gulf.
- Open-class Gulf words, through Gulf hosts.

Mix:
- **EGY:** about 60% (oasst2 EGY, Nile-Chat, Dial2MSA, STS).
- **GLF:** about 25% (Najdi conversations, Dial2MSA, STS Saudi).
- **MSA:** about 15%.

Hosts are cut to 4–40 words. Sentences with more than 30% unknown bits are dropped.

### D-B: aligned injection, about 35%
Word substitutions between parallel sides, which gives real context for the swap.
- EGY host + aligned MSA word or phrase. It becomes MSA or FRM depending on the labeler: an injected سوف is a deviation, an injected يمكنك is not.
- EGY host + Gulf word, through MSA-anchored alignment (STS triples, Dial2MSA pivot).
- MSA host + aligned Egyptian word. This teaches context, like ممكن + verb in MSA.
- Up to 3 substitutions per sentence. 30% of sentences are left clean. Labels for the injected word come from the labeler run on its source sentence.

Alignment: SimAlign (MARBERTv2, itermax) with an MSA anchor, as in the pilot. **This needs a GPU sandbox (§9).**

### D-C: lexicon substitution, about 10%
- The core lexicon gets equivalence groups. Examples: عايز ↔ أريد ↔ أبغى، دلوقتي ↔ الآن ↔ الحين، هـ+verb ↔ سـ+verb ↔ بـ/راح+verb.
- This covers closed-class markers in every inflection, using CAMeL paradigms.
- **Hold-out for generalization:** 25% of the Gulf lexical lemmas and 25% of the corpus-approved Gulf words are never used in training. Challenge v2 is built from them, so it measures generalization to unseen words. The frozen challenge v1 reuses common markers.

### D-D: Claude-written service corpus, about 10%
- Claude writes about 1,500 Egyptian customer-service turns in the session, at zero cost.
- Each turn also gets an MSA and a Gulf version, giving domain-matched triples.
- Claude marks the aligned spans directly, so this is the domain-matched part of D-B.
- Topics follow Tuner's verticals: telecom, banking, delivery, clinics, travel.
- Deduplicated against the gold set.

### Splits and controls
- **Splits:** train 90% / dev 10%, grouped by source sentence.
- **Calibration and early stopping:** dev plus a 300-sentence Claude-labeled real dev set, D3, double-pass for G4. The gold set is never used.
- **Balance:** about 10–12% of tokens deviate per target. The pilot ran at about 15%.

## 5. Data Guard status
| Gate | Status |
|---|---|
| G0 | Guideline v1.2 plus the v1.3 bits proposal, which needs Yahya's OK. Anchors pass. |
| G1 | Done: provenance per row in `parallel_v1.jsonl`, license column, exact dedup. |
| G2 | Done for the sources (0 shared 8-grams). To be re-run on the built set before training. |
| G3 | Planned. Yahya answers yes/no on 100 injected sentences ("natural Egyptian except the marked words?"). Target ≥ 90%. |
| G4 | Planned. D3 double-pass, Claude self-agreement ≥ 97%. |
| G6 | Planned. After the first training run: cartography + cleanlab, review the top 300 flagged tokens. |

## 6. Gold issues found (logged, not changed; gold v1 stays frozen)
- **glf_experimental:** حاليا is labeled MSA once and SHARED once. Candidate fix for gold v1.1, to apply before any model is evaluated.
- **ambiguity_log.tsv:** the entries for قادمة and موعد (MSA) are superseded by v1.2 (Q35 and Q24 were both answered yes). Update the log.

## 7. Model changes implied
- Three sigmoid naturalness heads plus `dia` and `other`, with a masked BCE loss.
- The decision layer becomes a few lines: deviation is 1 − bit[target], and the label comes from `output_label`.
- A register policy can come later. A "strict colloquial Egyptian" option would treat frm=1 as a deviation, with no retraining.

## 8. Cost
| Step | Where | Estimated cost |
|---|---|---|
| Labeler, hosts, lexicon substitution, D-D | local / in session | $0 |
| SimAlign on about 30k pairs | HF sandbox, T4/L4, about 30–45 min | about $0.3–0.5 |
| Nile-Chat full egy_msa rows (+83k pairs, optional) | CPU sandbox, about 10 min | about $0.01 |

## 9. Decisions needed from Yahya
1. **Approve guideline v1.3:** three naturalness bits plus frm/dia. Deviation flags in the gold do not change.
2. **Gulf target stays strict** (هل / يمكنك / هناك deviate in Gulf), the way the gold is labeled. Egyptian keeps the validated service register.
3. **Sandbox spend:** about $0.5 for alignment, plus $0.01 for the full Nile-Chat egy_msa rows.
