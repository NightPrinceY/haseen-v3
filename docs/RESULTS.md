# HaSeen 3 · results

I report everything on **HaSeen-3-Bench**, a test I froze before any V3 training data existed, from human speech no
HaSeen model had seen (Saudi TV, Egyptian speech, news, Levantine and Iraqi corpora). Every number below links to the
file it comes from. Thresholds were fitted on **dev** only; the test was scored **once**, on the int8 model that ships.

- Model: `VARabi/ALDi-Token-DI` (MARBERT, BERT-base) + 7 naturalness logits (MSA, EGY, SAU, GLF, LEV, IRQ, MAG) +
  other-dialect, non-Arabic and register heads · 162,263,056 parameters ([report.json](results/report.json)).
- Runtime: ONNX int8 on CPU, **7.12 ms p50** / 9.09 ms p95 per 15-word sentence (4 threads, AMD EPYC 7R13);
  PyTorch fp32 on the same CPU 29.82 ms. int8 and fp32 take the same decision on 99.41% of 59,733 dev words.
  155.5 MB. No GPU, no LLM, offline.

## Test (HaSeen-3-Bench v3.3, scored once)
[posthoc_v3_int8_test.json](results/posthoc_v3_int8_test.json) · policy: [policy_v3_int8.json](results/policy_v3_int8.json)

| Target | natural false alarms | challenge P | R | **F1** | label set holds the source | unseen-concept recall | gate (FA ≤ 1%, F1 ≥ 0.90) |
|---|---|---|---|---|---|---|---|
| MSA | 0.76% | 0.927 | 0.889 | **0.908** | 0.86 | 0.63 | pass |
| Egyptian | 0.38% | 0.966 | 0.852 | **0.905** | 0.91 | 0.39 | pass |
| Saudi | 0.76% | 0.925 | 0.840 | **0.881** | 0.88 | 0.51 | FA pass · F1 below |
| Gulf (Saudi, Gulf-east, Yemeni) | 0.36% | 0.963 | 0.815 | **0.883** | 0.92 | 0.43 | FA pass · F1 below |
| Levantine | 0.78% | 0.909 | 0.753 | **0.824** | 0.83 | 0.32 | FA pass · F1 below |
| Iraqi | 0.43% | 0.932 | 0.725 | **0.816** | 0.87 | 0.33 | FA pass · F1 below |

Mean challenge F1 **0.869** (dev 0.874: no sign of over-fitting the policy). All six targets stay under 1% false
alarms on native speech, and what the model flags is right 91–97% of the time. The gap is recall: Levantine and
Iraqi miss about a quarter of the swapped words, mostly words from concepts held out of training.

## V3 against V2 on the same test
I ran the shipped V2 model (HaSeen-2-ONNX) on this test, with its thresholds re-fitted on the V3 dev the same way
([v2_baseline_on_v3.json](results/v2_baseline_on_v3.json)):

| Target | V2 F1 (FA) | **V3 F1 (FA)** |
|---|---|---|
| MSA | 0.763 (2.24%, fails the FA gate) | **0.908** (0.76%) |
| Egyptian | 0.888 (0.49%) | **0.905** (0.38%) |
| Saudi | 0.819 (0.77%) | **0.881** (0.76%) |
| Gulf | 0.852 (0.52%) | **0.883** (0.36%) |
| Levantine | 0.795 (0.32%) | **0.824** (0.78%) |
| Iraqi | 0.797 (0.37%) | **0.816** (0.43%) |
| mean | 0.819 | **0.869** |

V2 had scored 0.932 on its own bench. On this harder, fresher test it drops to 0.819, and V3 is better on every
target. That is why I report V3 on this test only.

## Hard sets
- **Agent replies** (200 LLM replies asked to answer "in dialect T", 121 deviations, Claude-labelled): precision
  0.48–0.94, recall 0.17–0.58; Gulf F1 0.71, Saudi 0.64, Iraqi 0.51, Levantine 0.49. False alarms 0.0–0.6% of
  words, except Iraqi 2.98%. Egyptian (6 deviations) and MSA (2) are too small to read.
- **mo3jam probe** (900 held-out dictionary entries, 3,964 checks): own-variety words kept 84%, other-variety
  words caught 57% (MSA 75%, Saudi 73%, Iraqi 61%, Gulf 54%, Levantine 49%, Egyptian 36%).
These are below what I pre-registered (agent ≥ 0.85, probe 0.70–0.85 / ≥ 0.90): rare, unseen and LLM-style words are
where V3 is weakest.

## How I got here (dev only)
| Step | dev mean F1 | targets ≤ 1% FA | known answers |
|---|---|---|---|
| Round 1: V5 linear / pointer on train_v10 (labeler v9) | 0.828 / 0.831 | 6/6 | 69/73 (fails the 95% gate) |
| Diagnosis: the labeler itself scores 0.815 on dev → the model had reached its teacher | | | |
| Round 2: V6 linear / pointer on train_v11 (labeler v10, teacher 0.864) | **0.876** / 0.871 | 6/6 | **71/73** / 71/73 |
| Shipped: V6 linear, int8, policy refit on int8 outputs | 0.874 | 6/6 | |
Files: [round 1 dev](results/round1_V5-linear_dev.json), [round 2 dev](results/posthoc_V6-linear-s13.txt),
[pointer](results/posthoc_V6-pointer-s42.txt), [selection](results/selection.json),
[known answers](results/ka_V6-linear-s13.txt). Rule (pre-registered in `training/PREREG_V3.md`): pass the known-answer
gate, then the higher dev F1; within 0.003 the one with more known answers.

## What I would fix next (3.1)
1. **Gulf negatives**: today a Gulf bit is 0 only if Saudi, Gulf-east and Yemeni are all 0, so an unknown Yemeni bit
   blocks it; Egyptian service words (عايز، إحنا، مفيش) are then unknown for Gulf, and the model accepts them (8 of 31
   frequent Egyptian words flagged under target Gulf in my probe). Fix: treat a Yemeni "unknown" as "not natural"
   when Saudi and Gulf-east are both 0 and Yemeni text uses the word at < 8% of its home rate, then retrain.
2. **Label order**: rank the label set by how specific a variety is to the word (a word natural only in Levantine
   should be called Levantine first), not only by probability; the set is right 83–92%, its first item 38–79%.
3. **Unseen words**: more lexicon coverage (Lahga, Lisan, mo3jam held-in entries) for Levantine and Iraqi.

## Definitions
- **natural**: native sentences; the share of their words wrongly flagged.
- **challenge**: native sentences with 1–3 words swapped for another variety's word for the same MADAR concept, plus
  MSA-drift injections; precision / recall / F1 over words.
- **label set**: the deviating word's source variety is among the varieties HaSeen says it belongs to.
- **unseen-concept recall**: 25% of MADAR concepts never appear in training; recall on swaps from them.
