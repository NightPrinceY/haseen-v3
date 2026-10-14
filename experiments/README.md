# Build-3 runs (all kept)

| Run | Data | Head | Outcome |
|---|---|---|---|
| `V5-linear-s13`, `V5-pointer-s42` | train_v10 (labeler v9) | linear / pointer | round 1: dev 0.828 / 0.831, failed the known-answer gate (69/73) → labeler review |
| `V6-linear-s13` | train_v11 (labeler v10) | linear | round 2: dev 0.876, known answers 71/73 → **shipped** (int8 ONNX) |
| `V6-pointer-s42` | train_v11 | pointer | round 2: dev 0.871, 71/73 |

Files per run: `config.json` (all hyper-parameters), `train.out` (training log as printed), `log.jsonl` (loss per step),
`dev_curve.jsonl` (dev scores every 1,000 steps), `eval_raw.out` (raw-logit evaluation log), `calibration.json`,
`dev_at_save.json`, `metrics.json`, `posthoc_*.json` / `policy_*.json` (thresholds fitted on dev, and the test scored
once for the shipped run), `ka_*.txt` (known answers), `selection.json` (the pre-registered choice), `report.json` and
`export.txt` (int8 export: decision agreement, latency, size). Model weights are in the private model repos.
Build-1 and Build-2 runs, with their logs, are in `../history/`.
