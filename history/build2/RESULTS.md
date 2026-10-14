# HaSeen V2 — training results (2026-10-14)

Pre-registration: `PREREG.md` (expectations + addendum on the gold-dev policy, written before final results).
Backbone `VARabi/ALDi-Token-DI`, train_v7 (344,286 train / 17,974 dev rows), 2 epochs, bs 32, 4 runs in parallel on
one HF L4 (~3.2 h sandbox, ≈ $2.6). Weights + logs: `hf://buckets/NightPrince/HaSeen-V2/runs/` (private).

## Run selection (gold **dev** only)
| Run | train_v7 dev macro F1 (calibrated) | ECE | gold-dev mean challenge F1 | targets ≤ 1% FA | train min |
|---|---|---|---|---|---|
| V2-R1-pointer-s13 | 0.9781 | 0.00014 | 0.9184 | 8/9 | 170 |
| V2-R1-pointer-s21 | 0.9775 | 7e-05 | 0.9226 | 8/9 | 173 |
| V2-R1-pointer-s42 | 0.9783 | 0.00011 | 0.9238 | 8/9 | 173 |
| V2-R2-linear-s13 | 0.9775 | 5e-05 | 0.9242 | 8/9 | 160 |

Rule (PREREG addendum): most targets within the FA gate, then highest gold-dev mean F1 → **V2-R2-linear-s13**
(pointer seeds 0.9216 ± 0.003, linear 0.9242: a tie within seed noise; the rule was applied as written).

## Gold test, scored once (frozen policy from gold dev)
Policy: T_var 1.422, T_odia 1.279, tau_odia 0.9, per-target tau {"MSA": 0.02, "EGY": 0.675, "SAU": 0.15, "KHL": 0.25, "LEV": 0.875, "IRQ": 0.005, "YEM": 0.075, "MOR": 0.01, "TUN": 0.01}.

| Target | natural FA | P | R | challenge F1 | label set hit (size) | label top-1 | held-out concept recall | expected F1 / held-out | ship gate |
|---|---|---|---|---|---|---|---|---|---|
| MSA | 0.13% | 0.943 | 0.931 | **0.937** | 0.90 (3.3) | 0.45 | 0.93 (n=28) | ≥ 0.90 / 0.60–0.80 | pass |
| EGY | 0.84% | 0.936 | 0.957 | **0.946** | 0.89 (2.4) | 0.57 | 0.66 (n=35) | ≥ 0.90 / 0.60–0.80 | pass |
| SAU | 0.67% | 0.941 | 0.941 | **0.941** | 0.94 (2.4) | 0.51 | 0.83 (n=47) | 0.85–0.92 / 0.50–0.75 | pass |
| KHL | 1.55% | 0.949 | 0.962 | **0.956** | 0.90 (2.6) | 0.48 | 0.82 (n=33) | 0.80–0.88 / 0.40–0.65 | experimental (FA) |
| LEV | 0.77% | 0.949 | 0.899 | **0.923** | 0.91 (1.8) | 0.70 | 0.71 (n=34) | ≥ 0.90 / 0.60–0.80 | pass |
| IRQ | 0.76% | 0.969 | 0.872 | **0.918** | 0.94 (2.2) | 0.53 | 0.71 (n=34) | 0.85–0.92 / 0.50–0.75 | pass |
| YEM | 0.92% | 0.940 | 0.957 | **0.948** | 0.93 (2.0) | 0.61 | 0.62 (n=8) | 0.85–0.92 / 0.50–0.75 | pass |
| MOR | 1.27% | 0.932 | 0.880 | **0.906** | 0.95 (2.6) | 0.56 | 0.50 (n=28) | ≥ 0.90 / 0.60–0.80 | experimental (FA) |
| TUN | 1.04% | 0.920 | 0.910 | **0.915** | 0.91 (2.7) | 0.44 | 0.68 (n=34) | ≥ 0.90 / 0.60–0.80 | experimental (FA) |

Mean challenge F1 **0.932**. All 9 targets pass challenge F1 ≥ 0.90; 6/9 pass natural FA ≤ 1%.

## Against the pre-registered expectations
- Challenge F1: met or beaten everywhere. SAU / IRQ / YEM (expected 0.85–0.92) reached 0.918–0.948; KHL (expected
  0.80–0.88, weakest data) reached 0.956.
- Natural false alarms: expected ≤ 1% everywhere (≤ 1.5% KHL). **Missed** for KHL 1.55%, MOR 1.27%, TUN 1.04%. These
  three ship as *experimental* until the agent set (G4) and a native review of their false alarms.
- Held-out concept recall: inside the expected range for most targets; MSA 0.93 above; MOR 0.50 below its 0.60 floor.
- Pointer ≈ linear (as in V1); pointer did not collapse with 9 options (learned orthogonal option vectors).
- Calibration: ECE ≤ 0.00014 on train_v7 dev for all runs.

## Known weaknesses
- Source label: top-1 only 0.44–0.70 because deviating words are usually natural in several varieties; the label set
  holds the right source 89–95% of the time, at 1.8–3.3 varieties per word. Present it as a region / set, not one dialect.
- Several thresholds sit at the bottom of the grid (0.005–0.02): false alarms that remain are confident ones (rare
  foreign words, names, numbers written as words, polite MSA forms like اعذرني). Next: review them with natives and
  feed back into the labeler, not lower thresholds.
- Natural-set gate is measured on MADAR / Alexandria / Casablanca text (travel and general domain); the agent set (G4)
  is still needed to check the service register.
- Latency not yet measured for V2 (GPU was shared by 5 processes). Next: ONNX int8 export + CPU latency.

---
**Superseded (2026-10-14 morning):** the shipped model is V4-R2-linear-s13 on train_v9 (labeler v9), release 2.0.0.
Final results, selection and the stated deviation from the pre-registered rule: `RESULTS_2.0.0.md`, `PREREG.md`
(decision record), `eval_sdk_test.txt` (same test split through the package).
