# HaSeen 3 · هـ · سين

**Which Arabic words do not belong to the dialect your voice agent should speak?** HaSeen checks every word of a
turn against a target variety and, for each word that does not fit, says which varieties it does belong to.

Six targets: **MSA, Egyptian, Saudi, Gulf (Saudi + Gulf-east + Yemeni), Levantine, Iraqi**. Maghrebi and other
dialects (Sudanese, Libyan, Algerian, …) are named when they appear. One encoder pass answers for every target, on a
CPU, offline, in **7 ms** per sentence. I built it for voice-agent QA: dialect-adherence evals, red flags ("the agent
drifted to MSA", "a Gulf agent used Egyptian words"), caller-dialect estimates and simulation checks.

> Private repository. Model, data and Space are private on Hugging Face (`NightPrince/HaSeen-3*`); set `HF_TOKEN`.
> The full story (task, research, data, architecture, three builds, results, decisions, limits): https://claude.ai/artifact/FPTDppf8nPKVPh2ZjSoeg9

## Setup
```bash
uv tool install "git+https://github.com/NightPrinceY/haseen-v3"     # or: pipx install … / pip install -e . in a venv
export HF_TOKEN=hf_…                                                 # the model repo is private
haseen download                                                      # once (~156 MB); later runs are offline
```

## Use
```bash
haseen "أنا أبغى أروح الاجتماع بكرة بس سوف أتأخر شوية" -t egy          # coloured view
haseen "حضرتك الطلب بتاعك هيوصل بكرة" -t glf --json                   # result JSON
cat replies.txt | haseen -t lev                                       # one sentence per line -> JSON lines
haseen serve --port 8000                                              # HTTP: POST /detect {"text", "target"}
```
Expected output of the first line:
```
HaSeen · target Egyptian · 2 of 9 words deviate
  أبغى  Saudi   1.000  (also Gulf)
  سوف   MSA     1.000   → هـ + الفعل
  adherence 78% · reads as Gulf 0.80, Saudi 0.80, Levantine 0.65
```
```python
import haseen
r = haseen.detect("حضرتك الطلب بتاعك هيوصل بكرة", "glf")
r["deviations"]                 # ['حضرتك', 'هيوصل']
r["tokens"][0]["labels"]        # varieties a deviating word belongs to, most likely first
r["summary"]["adherence"]       # share of Arabic words that fit the target
r["summary"]["reads_as"]        # which varieties the turn reads as (caller-dialect estimate)
```
Targets: `msa egy sau glf lev irq` (also `Egyptian`, `gulf`, `khaleeji`, `najdi`, `shami`, `مصري`, `خليجي`, …).
`glf` accepts a word if it is natural in any Gulf variety; `sau` is stricter (Saudi only).

Docker (model baked in, runs with `--network none`):
```bash
DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN -t haseen3 .
docker run --rm -p 8000:8000 haseen3
```

<!-- RESULTS -->
## Results on HaSeen-3-Bench **test** (scored once, int8 model that ships)
I froze this test before building any V3 training data, from human speech no HaSeen model had seen (Saudi TV shows,
Egyptian speech, Al Jazeera, Levantine and Iraqi corpora). Thresholds were fitted on dev only.

| Target | natural false alarms | challenge P | R | **F1** | label set holds the source |
|---|---|---|---|---|---|
| MSA | 0.76% | 0.927 | 0.889 | **0.908** | 0.86 |
| Egyptian | 0.38% | 0.966 | 0.852 | **0.905** | 0.91 |
| Saudi | 0.76% | 0.925 | 0.840 | **0.881** | 0.88 |
| Gulf | 0.36% | 0.963 | 0.815 | **0.883** | 0.92 |
| Levantine | 0.78% | 0.909 | 0.753 | **0.824** | 0.83 |
| Iraqi | 0.43% | 0.932 | 0.725 | **0.816** | 0.87 |

Mean F1 **0.869** · all six targets under 1% false alarms · MSA and Egyptian also pass F1 ≥ 0.90. On the same test,
the previous release (V2) scores 0.819 and fails the false-alarm gate for MSA; V3 is better on every target.
Full results, hard sets (LLM agent replies, held-out dictionary probe) and how I chose the model:
[docs/RESULTS.md](docs/RESULTS.md).
<!-- /RESULTS -->

Reproduce: `make bench && make eval` runs the test split through this package; expected output in
[docs/eval_sdk_test.txt](docs/eval_sdk_test.txt) (mean F1 0.869, 6/6 within the gate; the package strips edge
punctuation first, so single targets differ from the table by ≤ 0.004).

## Known limits (read before using)
- **Gulf vs Egyptian**: with target Gulf, frequent Egyptian service words are flagged only partly (8 of 31 in my
  probe: حضرتك، دلوقتي، هيوصل are caught; عايز، إحنا، أوي، مفيش، برضه are not). Some of the 31 are also Hijazi
  (كده، فين، لسه), but most misses are real. Cause: Gulf = Saudi ∪ Gulf-east ∪ Yemeni, and Yemeni "unknown" blocks
  the Gulf negative in training. Fix planned for 3.1 (see docs/RESULTS.md).
- **Recall on unseen words**: words from concepts held out of training are caught 32–63% of the time; Levantine and
  Iraqi miss about a quarter of swapped words. What HaSeen flags is right 91–97% of the time.
- **Which variety a word is**: the label *set* holds the right variety 83–92% of the time, but its first label is
  right only 38–79%; many dialect words are natural in several varieties, so read the set, not only the first item.
- **LLM-style replies**: on real agent replies, F1 is 0.49–0.71 for the dialect targets, and Iraqi raised false
  alarms on 3% of words.

## How it works
- **Labels**: for every word and every variety, one bit: natural (1), not natural (0) or unknown (masked). A word
  deviates when its target bit is 0. Formal open-class words are shared by every variety (service register); MSA
  grammar markers (سوف، لن، الذي، سيتم) and other varieties' words are not ([docs/GUIDELINE.md](docs/GUIDELINE.md)).
- **Model**: `VARabi/ALDi-Token-DI` (MARBERT) encoder, first sub-word per word, 7 naturalness logits + other-dialect,
  non-Arabic and register heads; masked multi-task BCE; temperature and per-target thresholds fitted on dev
  ([training/train_v3.py](training/train_v3.py), [training/posthoc_v3.py](training/posthoc_v3.py)).
- **Data**: HaSeen-3-Mix, 501K sentences labelled by an evidence-ordered labeler (MADAR Lexicon concept matrix,
  per-variety MSA markers, a native-validated core lexicon, CALIMA-MSA, voted crowd lexicons, corpus rates)
  ([docs/MIX_CARD.md](docs/MIX_CARD.md), [pipeline/](pipeline/)).
- **Gold**: HaSeen-3-Bench, frozen before training with a sha256 manifest ([docs/BENCH_CARD.md](docs/BENCH_CARD.md)).
  Expectations and every decision were written down before the results ([training/PREREG_V3.md](training/PREREG_V3.md)).
- **Release gate**: natural false alarms ≤ 1%, challenge F1 ≥ 0.90, and a known-answer suite (the task example plus
  73 word checks; this release: 71/73) ([training/known_answers.py](training/known_answers.py)).

## Repository
| Path | What |
|---|---|
| `src/haseen/` | the package: ONNX runtime (default), PyTorch runtime, decision policy, CLI, HTTP server |
| `training/` | training, post-hoc policy, known answers, release export (ONNX int8 + parity + latency), PREREG_V3 |
| `pipeline/` | labeler v10, test and data builders, leakage filter, audits (run against the private data layout) |
| `scripts/evaluate.py` | benchmark through the public API |
| `demo/web/` | the Space: the model in the browser (onnxruntime-web), same decisions as Python (`tests/js_parity.js`) |
| `docs/` | targets, guideline, data cards, results and the files behind every number; `docs/sources/` one note per data source |
| `experiments/build3/` | every Build-3 run: config, dev curve, metrics, dev-fitted policy, known answers, selection, export report |
| `research/` | the research trail: plan, literature, data survey, architecture study, spikes, pilot, data audits, GPU spend log, report pages |
| `history/` | the two earlier builds kept as they were (guidelines, gold-building trail, all runs, results, reports) |
| `brief/` | the task and the scope questions with their answers |

## Data and licences
Training and evaluation use research-licensed sources (MADAR: CMU-Q / NYUAD; SADA CC BY-NC-SA; QCRI MGB-2 / MGB-3 /
ArabicCulturalQA; SDAIA SCC; Birzeit corpora; Alexandria CC BY-NC 4.0). Models and datasets are private; no
redistribution. The MADAR-derived suggestion map (`lexicons/concept_map.json`) ships only inside the private model
repos and is never committed here, so this code repository holds no licensed data. Code: all rights reserved
(see LICENSE).
