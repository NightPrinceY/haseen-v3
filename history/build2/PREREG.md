# HaSeen V2 — pre-registered expectations (written 2026-10-14, before any real training)

Script: `train_v2.py` (smoke-tested locally on a tiny random BERT: both heads, calibration, gold eval, latency).
Data: `data/train_v7` (344,426 train / 17,979 dev rows). Gold: `data/gold_v2/frozen` (scored once, thresholds from
train_v7 dev only). Backbone: `VARabi/ALDi-Token-DI` (as V1). 2 epochs, bs 32, lr 4e-5 / head 1e-3.

## Runs (HF L4 sandbox, ~7 steps/s from V1 → ~55 min per run incl. eval)
| Run | Head | Seeds | Cost |
|---|---|---|---|
| V2-R1 | pointer (9 options + learned vectors) | 13 first, then 21, 42 | ~$0.75 each |
| V2-R2 | linear (control) | 13 | ~$0.75 |
Stop rule: if R1-s13 collapses (identical option vectors / dev macro F1 < 0.6) → fix before more spend.

## Expectations (test splits)
| Target | natural FA | challenge F1 | held-out-concept recall |
|---|---|---|---|
| MSA, EGY, LEV, MOR, TUN | ≤ 1% | ≥ 0.90 | 0.60–0.80 |
| SAU, IRQ, YEM | ≤ 1% | 0.85–0.92 | 0.50–0.75 |
| KHL (weakest data) | ≤ 1.5% | 0.80–0.88 | 0.40–0.65 |
- Dev macro F1 (9 targets) ≥ 0.90; ECE ≤ 0.01; pointer ≈ linear (within 1 F1), pointer better calibrated.
- Ship gate (TARGETS_V2.md): natural FA ≤ 1%, challenge F1 ≥ 0.90, agent F1 ≥ 0.85 (agent set = G4, later).
  Targets that miss it ship as "experimental" or are demoted to "other dialect".

## Addendum (2026-10-14, before any final run finished): decision policy fitted on gold dev
A probe of a mid-training checkpoint (pointer s13, step 11K) showed that detection is strong but (1) thresholds fitted
for F1 on train_v7 dev do not respect the 1% natural false-alarm gate and (2) a single source label is right only
~52% of the time, because a deviating word is usually natural in several other varieties. Decided before the final
results, with no use of gold test:
- `posthoc_policy.py`: per-target tau chosen on **gold dev** = max challenge F1 subject to natural FA ≤ 1%; global
  tau_odia chosen the same way. Temperatures stay from train_v7 dev.
- Source label = **set** {v ≠ t : p_v ≥ 0.5} (MSA first as primary). Reported: top-1, set hit, mean set size.
- Run choice: most targets within the FA gate, then highest mean gold-dev challenge F1. Gold test scored **once**,
  for the chosen run, with the frozen policy (`--test`).

## Addendum 2 (2026-10-14, after the v7 runs, before any v8 training): labeler v8 + one more round
Error analysis of gold-**dev** false alarms (never test) traced the confident errors to labeler bugs, now fixed and
ablated on gold dev (`scripts/audit_labeler_v2.py`; labeler false alarms 0.62% → 0.34%, same 55 wrong labels,
hits 1,886 → 1,766 with the lost ones masked as unknown):
1. odia = 1 only for words natural in none of the nine targets (was set before corpus positives; families B/C
   forced it on "other dialect" injections that are natural in a target, e.g. نبغي, سيمانة).
2. Valid MSA open-class words get no dialect negatives (register rule, same as the gold): سائح، اتصل، كريمة.
3. Spelling variants are the same word in sense negatives (ست / سته, تمانيه / ثمانيه).
4. msa_valid compares against every dialect's rate (أبغي leaked into MSA text 4x and passed against an empty set).
5. V1 core lexicon (hand-curated, Egyptian bits validated by Yahya) overrides EGY / MSA bits on every path
   (أبغى, الحين, شو, بدي are not Egyptian).
Rejected by the ablation: corpus-use dominance, corpus use as evidence, MSA-bit dominance, exact-form-first.

Plan: train_v8 → V3-R2-linear-s13 and V3-R1-pointer-s42 (same hyper-parameters as v7) in parallel on one L4
(~$1.2). Policy: posthoc_policy.py with **odia_gate 0.5** for every candidate (fixed now, from the v7 dev analysis).
Choice among {v7 linear, v8 linear, v8 pointer}: most targets within the 1% FA gate on gold dev, then highest gold-dev
mean F1. Gold test scored once for the chosen model. Expectation: fewer natural false alarms (≥ 8/9 within the gate
on dev), challenge F1 within ±0.01 of v7.

## Addendum 3 (2026-10-14, before the v9 run): known-answer gate + MSA-future labels
Round-2 gold-dev results (odia_gate 0.5): v7 linear 0.9293, v8 linear 0.9225, v8 pointer 0.9219 mean F1, all 9/9 within
the FA gate. A known-answer suite run through the SDK (the task's own example + guideline core cases, 43 word checks,
`training/known_answers.py`) showed v7 misses أبغى / الحين / شو / ولن for Egyptian, i.e. fails the task example
(V1 passed it); v8 runs pass 41/43 and miss the سـ-future (سيصل). The gold has few such words, so the gold rule alone
would have kept v7. Added rule (applies to every release from now on): **a candidate must pass the task example and
≥ 95% of the known-answer suite; among those, the gold-dev rule decides.**
Labeler fix (v9 data): سـ-future verbs are hard MSA markers for every dialect (markers.tsv had سيتم as "formal";
the future rule was blocked by small leaks of formal text into dialect corpora); perfect verbs whose root starts with
س (سألت, سافر) are guarded by CALIMA stems. Labeler audit unchanged (FA 0.35%, hits 1,766, wrong 55).
Plan: train_v9 → V4-R2-linear-s13 alone on the L4 (~50 min). Candidates: v8 linear, v8 pointer, v9 linear.

## Decision record (2026-10-14, before scoring gold test): release candidate = V4-R2-linear-s13 (train_v9)
| Candidate | gold-dev mean F1 (odia_gate 0.5) | targets ≤ 1% FA | known answers |
|---|---|---|---|
| V2-R2-linear-s13 (v7) | 0.9293 | 9/9 | 48/54, fails the task example (أبغى) → out |
| V3-R2-linear-s13 (v8) | 0.9225 | 9/9 | 52/54 (misses سيتم, سيصل) |
| V3-R1-pointer-s42 (v8) | 0.9219 | 9/9 | 52/54 (misses سيتم, سيصل) |
| **V4-R2-linear-s13 (v9)** | 0.9208 | 9/9 | **54/54** |
**Stated deviation from addendum 3:** the written rule picks v8 linear by 0.0017 mean F1, which is below the seed spread
measured on the v7 pointer seeds (0.003). v9 is chosen because it is the only candidate that handles the سـ-future
(سيتم، سيصل), a hard marker in the guideline that the gold barely contains. From the next release: when gold-dev
differences are within seed noise, the candidate with more known answers wins. v8 linear stays in the bucket.
Gold test is scored once, for v9 only, on the shipped int8 model with its own dev-fitted policy.

## Addendum 4 (2026-10-14, release 2.1.0): Gulf as a virtual target, no retraining
Question (Yahya): merge Saudi with Gulf-east and retrain, and retire the non-production targets to "other dialect"?
- MADAR distance Saudi–Qatar is 0.15 (closed class 0.10), above the merge rule (≤ 0.05) and above the Levant merge
  (0.10–0.12): a merged label would erase real differences (وش / شو). A retrained "Gulf" bit would mean "natural in
  Saudi or Gulf-east", which the model already outputs. So: **GLF = max(p_SAU, p_KHL)**, threshold fitted on dev
  (natural FA ≤ 1%), gold = SAU + KHL sets keeping only swaps listed for no Gulf city in MADAR
  (`training/virtual_targets.py`). Dev: FA 0.57%, F1 0.865 (tau 0.8). **Test (once): FA 0.89%, F1 0.912 → passes the
  gate**, while Gulf-east alone (FA 3.14%) does not.
- Retiring KHL / YEM / MOR as labels would lose information ("this word is Moroccan" beats "other dialect") and would
  not improve the other targets; their weakness is only as *targets*. They stay as experimental targets (flagged in
  calibration.json and in every result) until the agent set and a native review of their false alarms.
