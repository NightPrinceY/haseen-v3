# Pilot report — System-1 variants on D1-lite (2026-10-12)

Compute: HF Sandbox L4 (~65 min) + CPU sandboxes. Approx. cost ≈ $0.95 (balance before: $12.66).
Raw results: `run1/results/*.json` (D1 v1 labels), `run2/*.json` (D1 v2 labels).

## Data
- **D1 v1:** MADAR-6 (Cairo / Doha / MSA), SimAlign (itermax, MARBERTv2) with MSA as anchor, 1–3 aligned-word insertions into a matrix sentence. 30% of sentences unmixed. Train 36,000 / dev 4,000 / test 6,400. SHARED defined by **sentence-level** co-occurrence.
- **D1 v2:** same, but SHARED and insertion filtering use **corpus-level validity** (`build_lexstats.py`): relative frequency r_v(w) from FineWeb2-arz (marker-filtered) + egyptian-2-arabic, FineWeb2-Najdi + Gulf tweets (marker-filtered), FineWeb2-MSA. Valid if count ≥ 3 and r ≥ 0.25. Reject an insertion if it is valid in the matrix variety at r ≥ 0.10.
  - Stats sanity: سوف MSA-only (EGY r = 0.05), الذي MSA-only, ابغي / وايد / الحين GLF-only, عايز / مش EGY-only, خطأ / فورا / تسجيل / الدفع shared.
  - Labels: SHARED 173,949 · EGY 42,936 · MSA 29,512 · GLF 25,479.
- **Realistic check:** 30 Claude-written Egyptian customer-service sentences, 23 deviations (`realish_check.jsonl`). **Not gold** (Claude labels, small n): directional signal only.

## Results (2 epochs, bs 32, lr 4e-5, temperature-calibrated)
Deviation = word variety not in {target, SHARED, OTHER}; target = matrix variety.

| Run | Variant | Backbone / head | Synthetic P / R / F1 | ECE | Realistic P / R / F1 | GPU p50 |
|---|---|---|---|---|---|---|
| D1 v1 | v1 | MARBERTv2 / linear | 0.817 / 0.866 / **0.841** | 0.020 | 0.457 / 0.913 / 0.609 | 7.8 ms |
| D1 v1 | v2 | ALDi-Token-DI / linear | 0.799 / 0.876 / 0.836 | 0.012 | 0.477 / 0.913 / 0.627 | 8.0 ms |
| D1 v1 | v3 | ALDi-Token-DI / pointer | 0.797 / 0.875 / 0.834 | 0.010 | 0.423 / 0.957 / 0.587 | 8.9 ms |
| D1 v1 | v4 | CAMeLBERT-Mix / linear | 0.801 / 0.861 / 0.830 | 0.008 | 0.412 / 0.913 / 0.568 | 8.1 ms |
| **D1 v2** | v2 | ALDi-Token-DI / linear | 0.875 / 0.866 / 0.871 | 0.012 | **0.741 / 0.870 / 0.800** | 8.0 ms |
| **D1 v2** | v3 | ALDi-Token-DI / pointer (A★) | 0.879 / 0.868 / **0.874** | **0.008** | 0.690 / 0.870 / 0.769 | 8.4 ms |

Label accuracy on correctly detected deviations (GLF vs MSA): 0.92–1.00.
Latency = PyTorch eager, fp16, batch 1, 15-word sentence (18 subwords), no ONNX/TensorRT yet.

## Findings
1. **Data labeling is the main lever.** Switching SHARED from sentence-level to corpus-level validity: realistic F1 0.627 → 0.800 (precision 0.48 → 0.74), synthetic F1 0.836 → 0.871. No model change.
2. **Backbone, warm start and head type barely matter** on the same data (all within ~1 F1 on 6.4k-sentence synthetic test). MARBERT family ≥ CAMeLBERT.
3. **Pointer head (Jev-style A★) = linear head on quality** (0.874 vs 0.871 synthetic; the realistic gap is about one word out of 23) and gives the **best calibration** (ECE 0.008) plus open options.
4. **The single-word insertion problem is solved by the data:** سوف, الحين, أبغاك, وايد, تبي, زين, راح are caught (realistic recall 0.87).
5. **Remaining errors** (D1 v2, v2):
   - False MSA on formal-but-normal service words: بتحويلك, وصلتك, تحدث, الخاصية, الشكوى, and بحضرتك. MADAR is travel domain → need customer-service domain data (D2/D3).
   - Missed function-word ambiguity: لا (MSA verbal negation vs Egyptian "no"), إذا, معنا → context + lexicon features + teacher data.
6. **Latency:** GPU eager 8 ms p50. Above the 5 ms GPU target → ONNX / TensorRT / fp16 kernels, or revise the target. CPU int8 still to measure.

## Recommendation from the pilot
- Ship **A★: MARBERT-family encoder (warm start from ALDi-Token-DI) + pointer head** = a real System-1-style typed decision model, calibrated. Keep the linear head as an ablation; if Yahya's gold set shows linear > pointer by > 2 F1, ship linear.
- Spend the remaining effort on **data** (D2 domain injection into customer-service-like real sentences, D3 Claude-labeled real text, gold set), not on bigger models.
- **B (decoder)**: not justified by current errors (they are domain/label errors, not capacity). Run only if gold F1 < 0.85 after D2/D3.
- **C (cascade)**: optional; most remaining errors are function-word ambiguity that a System-2 check could fix → evaluate after D2/D3 with τ tuned on dev.
