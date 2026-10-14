# V1 — release snapshot (2026-10-13)

Everything in this folder as of git tag `takehome-v1` is **V1**. Later work goes into V2 and does not change V1 files
(the gold set, the datasets and the run folders are frozen; new versions get new folders).

## What V1 contains
| Part | Where | Version |
|---|---|---|
| Annotation guideline | `HaSeen-V1/docs/GUIDELINE.md` | v1.2 + v1.3 naturalness-bits proposal (`docs/DATA_PLAN.md` §1) |
| Gold evaluation set (frozen, sha256) | `HaSeen-V1/data/gold_v1/` | gold v1: 794 sentences, 14,534 words |
| Training data | `HaSeen-V1/data/train_v6/` (shipping), `train_v3`–`train_v5` (history) | train_v6: 92.3k train / 6.9k dev / 4.3k held-out sentences |
| Service corpus (Claude-written) | `HaSeen-V1/data/service_v1/` | 197 templates × 3 varieties |
| Lexicons | `HaSeen-V1/data/lexicons/` | core lexicon v2 (~270 entries), Lahga map, gazetteers |
| Labeler | `HaSeen-V1/src/haseen/labeler.py` | v1.3 (3 naturalness bits, never guesses) |
| Inference | `HaSeen-V1/src/haseen/detector.py`, `cli.py` | detect(text, target) → task JSON |
| Training + evaluation | `training/` | train.py, export_onnx.py, analyze_run.py, rescore.py, checklist (G7) |
| Shipping model | `training/models/HaSeen-1-R5v6-s42/` (+ bucket) | R5v6-pointer-s42 (chosen by dev score) |
| ONNX int8 | `training/models/HaSeen-1-R4v5-s21/onnx/` | same architecture; export of R5v6-s42 is a V1.1 task |
| Results vs pre-registered expectations | `training/RESULTS.md` | all runs R1–R5 |
| Blueprint (interactive) | `research/artifacts/training-blueprint.html` → https://claude.ai/artifact/GUxvVmhaSXwZoqh9yU3qHt | v4 |
| Storage | `hf://buckets/NightPrince/HaSeen-V1` (private) | data + every run |

## V1 numbers (R5v6, mean ± 95% CI over 3 seeds; shipping seed s42 in brackets)
- Egyptian, natural + stress + challenge: F1 0.966 ± 0.005; natural Egyptian false alarms 1–2 / 4,761 words
- egy_stress F1 0.868 ± 0.021 · egy_challenge 0.985–0.988 · held-out unseen Gulf words recall 0.84
- MSA F1 0.702 ± 0.005 · Gulf (experimental) F1 0.780 ± 0.024
- Calibration ECE 0.0006 · CheckList: INV 100%, DIR 100%, MFT 24–25/26
- Latency (batch 1, 15 words): ONNX int8 CPU 11.3 ms p50 / 12.2 p95 (163 MB); GPU fp16 8.5 ms
- Compute spent: ≈ $3 of HF credits (pilot ≈ $1, alignment $0.12, training ≈ $3 incl. one collapsed run)

## Known V1 limitations
- أبعثلك-type ث-spellings are missed (rare in data); "من حقك" sometimes flagged.
- MSA target over-flags some service nouns; Gulf has no native annotator (experimental).
- GPU latency target (< 5 ms) not met without TensorRT.
- MADAR and Dial2MSA are research-only licences; a commercial V2 must replace them.

## Release: HaSeen V1 (name chosen 2026-10-13: هـ · سين — Egyptian هـ vs MSA سـ; also حصين "guarded")
All private.
| Where | Name |
|---|---|
| HF model | https://huggingface.co/NightPrince/HaSeen-1 (self-contained, offline load verified) |
| HF model | https://huggingface.co/NightPrince/HaSeen-1-ONNX (int8, onnxruntime-only inference) |
| HF dataset | https://huggingface.co/datasets/NightPrince/HaSeen-Mix-92K (train_v6) |
| HF dataset | https://huggingface.co/datasets/NightPrince/HaSeen-Bench (gold v1) |
| HF dataset | https://huggingface.co/datasets/NightPrince/HaSeen-Service-Triples |
| HF Space | https://huggingface.co/spaces/NightPrince/HaSeen-Playground (static; model runs in the browser) |
| GitHub | https://github.com/NightPrinceY/haseen (tag v1.0.0; local clone `TakeHome/haseen/`) |
Notes: Gradio Spaces need HF PRO (402), so the playground is static + onnxruntime-web. GitHub Actions runs fail at
startup on this account (workflow file is valid; likely billing / Actions settings for private repos).
