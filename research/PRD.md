# PRD: Dialect Deviation Detector

Status: **v2 (2026-10-12)** — architecture decided by the pilot (see `pilot/PILOT_REPORT.md`); Data Guard and day-by-day plan added. Interactive blueprint: research/artifacts/
Owner: Yahya. Deadline: 2026-10-19 15:00 Cairo (target 2026-10-17).

---

## 1. Problem
LLM-powered voice agents mix varieties of Arabic: a Gulf word inside Egyptian speech, or formal MSA slipping into colloquial replies. When the reply is spoken by TTS, the caller hears an agent that "doesn't sound local". We need to detect, **per word**, every word that does not belong to the expected variety, say what it is instead, and do it fast enough to run on every agent turn.

## 2. Users and use cases
| Use case | Mode | Latency need |
|---|---|---|
| Guard on agent text **before TTS** | real-time, per turn | few ms (must not add noticeable turn latency) |
| Post-call evaluation of transcripts / logs (Tuner Observability) | batch | throughput matters, latency does not |
| Dataset auditing / regression tests for agent prompts | batch | none |

## 3. Constraints
- **No cloud LLM at inference.** Must run fully offline (air-gapped). Self-hosted open-weight models are allowed.
- A GPU is available in deployment. The design must still run on CPU for System 1.
- Any open-source tool, model or dataset is allowed and must be credited.
- LLM help is allowed at development time (data creation and annotation). Stated explicitly in the README.

## 4. Inputs and outputs
**Input**
```json
{"text": "أنا أبغى أروح الاجتماع بكرة بس سوف أتأخر شوية", "target": "Arabic / Egyptian"}
```
**Output** (task schema, extended; the extensions are optional fields and explained in the README)
```json
{
  "target": "Arabic / Egyptian",
  "tokens": [
    {"word": "أنا", "index": 0, "start": 0, "end": 3, "conforms": true, "variety": "SHARED"},
    {"word": "أبغى", "index": 1, "start": 4, "end": 8, "conforms": false,
     "label": "Arabic / Gulf", "confidence": 0.92, "suggestion": "عايز"}
  ],
  "deviations": ["أبغى", "سوف"],
  "meta": {"model": "...", "latency_ms": 4.1, "escalated_tokens": 0}
}
```
Extensions and why:
- `start`/`end`: character offsets, so a UI or TTS pipeline can highlight or replace the exact span (the word alone is ambiguous when it repeats).
- `variety`: the predicted variety for every token, including conforming ones (SHARED vs EGY), useful for analytics.
- `suggestion`: the target-variety equivalent when the lexicon has one. Makes the output actionable for a pre-TTS guard.
- `meta`: latency and how many tokens went to System 2, for observability.

Interfaces: Python function `detect(text, target) -> dict`, and a CLI (`dialect-detect --target egy "..."`, plus `--jsonl` batch mode). An HTTP endpoint is in the production plan.

## 5. Supported targets
| Target | Status | Evaluation |
|---|---|---|
| Arabic / Egyptian | **Primary** | Full gold set (verified by a native Egyptian speaker) |
| Arabic / MSA | Supported | Gold subset (dialect words inside formal text) |
| Arabic / Gulf | **Experimental** | Small set; no native Gulf annotator; "Gulf" covers several sub-dialects. Stated as a limitation |

Deviation labels: `Arabic / Egyptian`, `Arabic / Gulf`, `Arabic / MSA`, `Arabic / Levantine` (if confident), `unknown`.

## 6. Annotation guideline (labels per token)
Internal per-token variety: `EGY`, `GLF`, `MSA`, `SHARED`, `OTHER`.

1. **Shared words conform.** A word that a native speaker of the target would naturally use is conforming, even if MSA or other dialects use it too (confirmed by Tuner: "الاجتماع" conforms).
2. **Deviation = a native speaker of the target would not use this word in this context.** Context matters: "سوف" is fine in an MSA sentence and deviating in an Egyptian one.
3. **Morphology: flag positive evidence only.** A word deviates if it carries a marker of another variety (MSA "سوف/لن/لم/الذي/ليس", Gulf "أبغى/وش/ليش/حق/يبيله", Levantine "عم/هيك"). A word that merely lacks the target's marker is conforming ("أتأخر" without هـ, as in Tuner's example).
4. **Mixed words** (target clitic on a foreign stem, or the reverse) are deviating, labeled by the foreign part.
5. **Named entities** (people, places, brands, products) conform and get `OTHER`. They are not dialect evidence.
6. **Numbers, punctuation, emojis, URLs** are not scored (`OTHER`, `conforms: true`). Kept in the token list so indices match the input.
7. **Latin script / English words / Arabizi** get `OTHER`, conform, and are not scored. Arabizi transliteration is future work.
8. **Spelling variation** is normalized before classification (alef/hamza forms, ta marbuta/ha, alef maqsura/ya, tatweel, diacritics removed), following CODA* conventions. The output always returns the original word.
9. **Ambiguous** cases the annotator cannot resolve get `SHARED` (benefit of the doubt), and are listed in the annotation notes.

## 7. Tokenization
Whitespace + punctuation split on the original text, keeping character offsets. Clitics stay attached (the task asks for words, and "هتأخر" is one word). Subword handling is internal to the model.

## 8. Evaluation protocol
- **Gold set** (never used for training or tuning):
  - Source: Egyptian customer-service style replies written by 4–6 different LLM families via the HF router (the real phenomenon), plus a few hand-written hard cases.
  - Size: ~250–300 sentences for the Egyptian target, ~60 for MSA, ~60 for Gulf.
  - Process: Claude pre-labels → Yahya reviews every token in a CoNLL-style TSV (`index, word, variety, deviating, label, notes`).
  - Kept separate from all training data sources and generation prompts.
- **Dev set**: synthetic + LinCE dev, used for tuning and calibration.
- **External check**: LinCE `lid_msaea` dev (MSA vs Egyptian), mapped to our labels.
- **Metrics**:
  - Primary: precision, recall, F1 on **deviating tokens** (positive class = deviating), micro-averaged.
  - Label accuracy on correctly detected deviations (is it GLF vs MSA?).
  - Per-target and per-label breakdown; confusion matrix.
  - Calibration: ECE of the confidence.
  - Latency: p50 / p95 per sentence on CPU and GPU, plus throughput.
- **Baselines** reported next to the final system: lexicon+morphology rules; and the final system with System 2 off.

## 9. Success criteria
| Metric | Target |
|---|---|
| Deviating-token F1, Egyptian gold set | ≥ 0.85 (literature ceiling for MSA–dialect word-level LID is ~0.92) |
| System 1 latency, 15-word sentence | p95 < 20 ms on CPU, < 5 ms on GPU |
| Fresh clone → working CLI | following the README only, one command for setup |

## 10. Deliverables (from the task)
GitHub repo: source and inference pipeline; data-prep and training scripts; dependency file + Dockerfile; evaluation set + script; README (setup, exact commands with expected output, supported dialects, results and how to reproduce, limitations, future work, credits); a production plan.

## 11. Out of scope
Arabizi, speech input, dialects other than EGY/GLF/MSA as targets, automatic rewriting of whole sentences (we only suggest single-word replacements).

## 12. Architecture (decided)
**A★ — HaSeen:** MARBERT-family encoder (163M), warm start from `VARabi/ALDi-Token-DI`, **pointer head** (word vector · cached option-description vector, Jev-style), temperature calibration, decision layer vs target, lexicon suggestions. One forward pass for all words.
- Pilot (D1 v2): synthetic F1 0.874, ECE 0.008; linear-head twin 0.871 / realistic 0.80.
- Linear head kept as ablation; ship linear instead only if gold shows linear > pointer by > 2 F1.
- Aux sentence heads dropped (not measured, simpler). System 2 (cascade) optional, batch mode, evaluated after D2/D3. Decoder B only if gold F1 < 0.85 after D2/D3.

## 13. Data Guard (every dataset must pass before training or eval)
| Gate | Name | Owner | Pass criterion |
|---|---|---|---|
| G0 | Guideline + 40 anchor examples + ambiguity log | Yahya | guideline versioned; anchors agreed 100% |
| G1 | Provenance + hygiene (source, generator, prompt id, license; dedup; filters; PII) | auto | 0 rows without provenance; dup rate reported |
| G2 | Split isolation: gold from LLM family Set A, training from Set B; no 8-gram overlap; gold frozen with sha256 | auto | overlap = 0; hash committed before training |
| G3 | Generation QA: 100 D1 alignments, 100 D2 injections rated by Yahya, 60-word lexstats sanity list | Yahya | align ≥ 95%; natural ≥ 90%; sanity 60/60 |
| G4 | Teacher consistency: double-pass labeling, rationale per deviation, Claude vs Yahya on 200 sentences | Claude | self-agreement ≥ 97%; κ ≥ 0.80 |
| G5 | Gold review: Yahya reviews 100% of tokens; 20% re-review after 48h; ambiguous tracked separately | Yahya | intra κ ≥ 0.85; ambiguous ≤ 3% |
| G6 | Label-error mining: dataset cartography + cleanlab; review top 300 flagged tokens; retrain | auto | reviewed, fix rate logged |
| G7 | CheckList: MFT (markers), INV (diacritics/punct/NE), DIR (هـ→سوف flips) | Claude | MFT ≥ 95%, INV ≥ 98%, DIR ≥ 95%; in CI |
| G8 | Eval integrity: no tuning on gold; 3 seeds; bootstrap CIs; per-label breakdown | auto | mean ± 95% CI reported |
| G9 | Reproducibility: pinned versions, seeds, data/model cards, one-command rebuild | auto | fresh clone reproduces ±0.5 F1 |
A failed gate sends the data back to its source step. No exceptions.

## 14. Plan
| Day | Goal | Key tasks (gate) |
|---|---|---|
| Mon 13 Oct | Foundations | guideline v1 + anchors (G0); repo skeleton (G9); gold Set A generation (G2); Claude pre-label batch 1; Yahya blind calibration 50 → κ (G4); guideline v1.1 |
| Tue 14 Oct | Gold done + lexicon + D2 | Yahya reviews all gold (G5); freeze gold + sha256 (G2); lexicon (G3); D2 from Set B + AOC/FineWeb2 (G1); Yahya rates 100 injections (G3); leakage checks (G2) |
| Wed 15 Oct | D3 + training + cartography | D3 1–2k double-pass (G4); train A★ + linear, 3 seeds (G8); cartography + cleanlab, review top-300, retrain (G6); first gold eval with CIs |
| Thu 16 Oct | Quality, speed, engineering | CheckList + CI (G7); error analysis, linear vs pointer; ONNX + CPU/GPU latency (G9); System 2 eval (optional); CLI/API |
| Fri 17 Oct | Delivery | README, production plan, Dockerfile, fresh-clone test (G9); Yahya final read; submit |
| Sat–Sun 18–19 | Buffer | fixes only; deadline Sun 19 Oct 15:00 Cairo |

## 15. Beyond the take-home
Submission repo stays private (access granted to Tuner). Anything public comes after the hiring decision and is retrained on redistributable data only (MADAR/AOC are research-only). Roadmap: open model + data cards + HF Space demo → technical report / benchmark for word-level dialect fidelity → more dialects and STT transcripts → SDK, typed /v1/systemone API, data flywheel.
