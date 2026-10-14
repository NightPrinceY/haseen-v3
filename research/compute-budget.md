# Compute Budget Plan ($15 HF Jobs)

Rule: every paid run answers one measurable question, has a stop rule, a hard `--timeout`, and is first dry-run locally on a tiny sample.

## Key research finding that shapes the budget
Zero-shot LLMs are weak at Arabic dialect ID; fine-tuned MARBERT/AraBERT beat them clearly
(GPT-4 zero-shot ~45.6 F1-macro; fine-tuned MARBERTv2 ~82.9 F1 on NADI-2023; small LLMs zero-shot as low as 8 F1).
So **blind "teacher LLM labels 100k sentences" is NOT a safe spend**: the teacher may be worse than the student.

Sources: arxiv.org/pdf/2509.13775, aclanthology.org/2023.arabicnlp-1.64

## Free (local GPU 3.68 GiB / CPU) — Gate 0
- Data pipeline, synthetic injection (MADAR lexicon / parallel corpora), LinCE MSA-EA, gold eval set.
- Baseline A (lexicon + CAMeL morphology).
- Encoder B/C training: CAMeLBERT-Mix (110M) fits locally with fp16, seq 64–128, batch 8–16 (+ grad accumulation).
  MARBERTv2 (163M) is tight: batch 8 + gradient checkpointing.
- Latency measurement on local GPU and CPU.
Output: F1 on gold set + error analysis. This decides what (if anything) is worth paying for.

## Paid gates
| Gate | Question | Flavor | Timeout | Max cost | Go only if |
|---|---|---|---|---|---|
| 1 | Does a local Arabic LLM (ALLaM-7B / Fanar-9B / SILMA-9B / Qwen) fix the encoder's low-confidence words? | L4 ($0.80/h) | 45m | ~$1 | Gate 0 done; error analysis shows context/knowledge errors |
| 2 | Does a decoder backbone + LoRA token head (Arch F) beat the encoder on gold F1? | L40S ($1.80/h) or A100 ($2.50/h) | 2h | ~$5 | Encoder errors are capacity/rare-word, not label noise |
| 3 | Final training runs / seeds of the chosen System 1 | L4 | 1h | ~$1 | A winner is chosen |
| — | Reserve for a failed or repeated run | | | ~$4–5 | |

Gate 1 can be done for free on Kaggle instead if we want to keep the money for Gate 2.

## Safety rails
- `hf jobs uv run ... --timeout <x>` on every job (default is 30 min).
- Scripts run end-to-end on CPU with 20 samples before submission.
- Checkpoints and outputs pushed to the Hub during the run, so a crash keeps partial results.
- Spend tracked in this file after each job.

## Spend log (starting balance $12.66)
| Date | Job | Flavor | Minutes | Cost | Result |
|---|---|---|---|---|---|
| 2026-10-12 | Spike sandbox (data checks, analyzer coverage) | cpu-upgrade | ~30 | ~$0.02 | LinCE unavailable; MADAR parallel OK; CAMeL analyzers weak (rule F1 0.16) |
| 2026-10-12 | Spike 3 sandbox (ALDi-Token-DI probe, tokenizers, latency) | cpu-upgrade | ~20 | ~$0.01 | MARBERT family best tokenizer; ALDi-Token-DI misses single inserted words; CPU 30 ms fp32 |
| 2026-10-12 | Pilot: D1 build + 4 variants + D1v2 rebuild + 2 variants | l4x1 (+cpu) | ~65 (+40 cpu) | ~$0.95 | Data labeling is the lever: realistic F1 0.63→0.80; pointer ≈ linear, best ECE |
| 2026-10-12 | HF router: gold pool A (560) + MSA stress (100) + EGY stress (220) + DeepSeek committee (~430 calls) | router | — | UNKNOWN — provider cost fields mostly missing; Yahya reports it was a lot; likely driven by reasoning-model judges (failed run). Router use stopped. | gold candidates built |
| 2026-10-12 | Word alignment (batched SimAlign-style, MARBERTv2 L8) of 158k dialect↔MSA pairs (MADAR-6, Dial2MSA, oasst2, STS, Nile-Chat test + 39k Nile-Chat train) | l4x1 | ~9 | ~$0.12 | data/align/aligned_v1.jsonl.gz |
| 2026-10-13 | Training: smoke, R1-s13 (collapsed), R1-s21 (stopped), R2, R1v2, R4v5 ×3 seeds, G7 checklist on 5 runs | l4x1 sandbox 6acd42c7… | ~140 (20:36–22:55 UTC) | ~$1.9 (estimate, over the ~$1.6 plan by ~$0.3 because of the collapsed pointer run) | best: R4v5-pointer, Egyptian F1 0.970 ± 0.003. Cancel API returned 500 at the end; sandbox relies on its 30 min idle timeout, cancel retried in a loop |
| 2026-10-13 | R5v6 ×3 seeds + ONNX/int8 export + detector/eval checks | l4x1 sandbox 6acd9949… | ~80 | ~$1.1 | shipping model R5v6-s42 (by dev); int8 CPU 11.3 ms. Sandbox killed, no jobs running |
| 2026-10-14 | HaSeen V2 training: pointer s13/s21/s42 + linear s13 on train_v7 (344K rows, 2 ep), **4 runs in parallel on one L4** (GPU 19/23 GB, 100% util; 2 jobs = 8.4 steps/s vs 7.8 alone) | l4x1 sandbox 6acecb42… | ~190 | ~$2.6 | done (~190 min incl. probes ≈ $2.6); chosen V2-R2-linear-s13, gold test mean F1 0.932 |
| 2026-10-14 | HaSeen V2 round 2: labeler v8 → train_v8; V3-R2-linear-s13 + V3-R1-pointer-s42 in parallel on one L4, then raw gold logits + export | l4x1 sandbox 6acefe32… | ~172 (07:00–09:52) | ~$2.3 | V3 linear + pointer (v8), V4 linear (v9, shipped), v7 + v9 exports, int8 logits, fresh-user SDK test, HF uploads; killed |
| 2026-10-14 | HaSeen V3 round 1: train_v10 (500K rows, 1 ep), V5-linear-s13 + V5-pointer-s42 in parallel on one L4, raw gold logits | l4x1 sandbox 6acf5781… | ~84 (13:20–14:44) | ~$1.1 | dev mean F1 0.819 / 0.818 = the labeler's own ceiling (teacher F1 0.76–0.89); weights in bucket; killed |
| 2026-10-14 | HaSeen V3 known answers (V5 linear + pointer, CPU) | cpu-upgrade sandbox 6acf6eb3… | ~25 | ~$0.02 | both 69/73 (< 95% gate): GLF misses Egyptian حضرتك / بتاعك / هيوصل, SAU misses شو → labeler leak fix; killed |
| 2026-10-14 | HaSeen V3 round 2: train_v11 (labeler v10, 501K rows, 1 ep), V6-linear-s13 + V6-pointer-s42 on one L4, raw logits, dev policy, known answers, selection, ONNX int8 export, HF uploads (HaSeen-3, -ONNX, -Bench, -Mix, Space) | l4x1 sandbox 6acf8de4… | ~95 (17:13–18:48) | ~$1.3 | chosen V6-linear-s13 (dev 0.876, known answers 71/73); test once (int8): mean F1 0.869, 6/6 within 1% FA; bucket backups; killed |
