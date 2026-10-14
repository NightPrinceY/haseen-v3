# HaSeen 2 · results

Word-level Arabic dialect deviation detection for **nine varieties**: given a sentence and a target variety, every
word is marked as conforming or deviating, and a deviating word gets the varieties it does belong to (MSA, another
dialect, or "other dialect" for Omani, Sudanese, Libyan, Algerian, Hassaniya). One encoder pass gives the answer for
all nine targets; the target is a runtime parameter.

- Backbone: `VARabi/ALDi-Token-DI` (MARBERT, BERT-base) + 9 naturalness logits + other-dialect, non-Arabic and
  register heads · 162,266,132 parameters.
- Runtime: ONNX int8 on CPU, **6.97 ms p50** / 8.47 ms p95 per 15-word sentence (4 threads,
  AMD EPYC 7R13 Processor); PyTorch fp32 CPU 28.41 ms. No GPU, no LLM, works offline.
- Trained on HaSeen-2-Mix (343K sentences), evaluated on HaSeen-2-Bench (10,155 gold sentences).

## Model selection (gold dev only)
| Candidate | gold-dev mean F1 | targets ≤ 1% FA |
|---|---|---|
| v7 linear (train_v7) · known answers 48/54, fails the task example | 0.9293 | 9/9 |
| v8 linear (train_v8) · known answers 52/54 | 0.9225 | 9/9 |
| v8 pointer (train_v8) · known answers 52/54 | 0.9219 | 9/9 |
| v9 linear (train_v9) · shipped · known answers 54/54 | 0.9208 | 9/9 |

Rule (pre-registered, training/PREREG.md): most targets within the gate, then highest mean F1 → **v9 linear (V4-R2-linear-s13), see the decision record in training/PREREG.md**.

## Results on HaSeen-2-Bench **test** (scored once; policy fitted on dev)
ONNX int8 (what ships):

| Target | natural false alarms | challenge P | R | F1 | label set holds the source | held-out concepts recall |
|---|---|---|---|---|---|---|
| MSA | 0.44% | 0.969 | 0.969 | **0.969** | 0.90 (3.2 varieties) | 1.00 |
| Egyptian | 0.96% | 0.938 | 0.938 | **0.938** | 0.87 (2.5 varieties) | 0.66 |
| Saudi (Najdi, Hijazi, Southern) | 0.61% | 0.969 | 0.934 | **0.951** | 0.93 (2.5 varieties) | 0.96 |
| Gulf (UAE, Qatar, Kuwait, Bahrain) * | 3.14% | 0.985 | 0.948 | **0.967** | 0.89 (2.5 varieties) | 0.91 |
| Levantine (PS, JO, LB, SY) | 0.10% | 0.985 | 0.833 | **0.903** | 0.89 (1.7 varieties) | 0.71 |
| Iraqi | 0.89% | 0.949 | 0.939 | **0.944** | 0.92 (2.3 varieties) | 0.88 |
| Yemeni * | 1.34% | 0.915 | 0.939 | **0.927** | 0.92 (1.8 varieties) | 0.75 |
| Moroccan * | 1.40% | 0.924 | 0.825 | **0.872** | 0.94 (2.3 varieties) | 0.57 |
| Tunisian | 0.75% | 0.996 | 0.853 | **0.919** | 0.90 (2.4 varieties) | 0.29 |

Mean challenge F1 **0.932** · targets within the 1% natural false-alarm gate: **6/9**.

\* experimental in this release: natural false alarms above 1% on test. PyTorch fp32 with its own dev-fitted policy:
mean F1 0.935, 7/9 within the gate.

- **natural**: native sentences (MADAR, Alexandria, Casablanca test splits); share of words wrongly flagged.
- **challenge**: native sentences with 1–3 words swapped for another variety's word for the same MADAR concept.
- **label set**: the deviating word's source variety is among the varieties the model says it belongs to.
- **held-out concepts**: 25% of MADAR concepts never seen in training (generalization, not memorization).

## Decision policy (calibration.json)
`deviates = p_t < tau_t`, or other-dialect: `p_odia > 0.4` and `p_t < 0.5`; names, digits and
Latin script always conform. Temperatures T_var 1.397, T_odia 1.008; per-target tau
{"MSA": 0.175, "EGY": 0.85, "SAU": 0.625, "KHL": 0.95, "LEV": 0.5, "IRQ": 0.225, "YEM": 0.3, "MOR": 0.9, "TUN": 0.675}. Fitted on HaSeen-2-Bench dev on this exact int8 model: max challenge F1 subject to ≤ 1%
natural false alarms. Int8 vs PyTorch decision agreement with one shared policy: 98.8%.

## Quantization
Dynamic int8 per-tensor (per-channel tried: lower agreement). {"per_tensor": {"decision_agreement_vs_torch": 0.98805, "latency_ort_int8_cpu": {"p50": 6.97, "p95": 8.47}}, "per_channel": {"decision_agreement_vs_torch": 0.62562, "latency_ort_int8_cpu": {"p50": 7.32, "p95": 9.03}}}

## Limits
- Text only: accent and pronunciation are out of scope (they come through STT output).
- Labels come mostly from lexical evidence; context is used by the encoder but supervised mainly for homographs.
- Levantine is one class (a Lebanese word inside a Jordanian reply is not flagged); Gulf (KHL) evidence is Doha-heavy,
  and its dev natural set is small (1,704 words): the dev-fitted threshold (0.95) gave 0.82% false alarms on dev but
  3.14% on test. Gulf, Yemeni and Moroccan are experimental until the agent set and a native review of their alarms.
- The natural sets are travel / general domain; the agent set (LLM replies "in dialect T", native-verified) is next.
- Private: several training sources (MADAR, Birzeit corpora) are research-only.
