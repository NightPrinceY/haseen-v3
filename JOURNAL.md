# Work journal (12–14 Oct 2026)

What I did, in order, with the files that hold the evidence. I worked with Claude Code as a research assistant
(reading, coding, running GPU sandboxes, drafting labels for me to verify); the decisions, data access and native
checks are mine. GPU spend per session: [research/compute-budget.md](research/compute-budget.md).

## 12 Oct · task in, research, Build 1 plan
- Task received ([brief/task.md](brief/task.md)). Sent five scope questions; all answered within 16 minutes
  ([brief/clarifications.md](brief/clarifications.md)).
- Literature, datasets, models and tools surveyed, each with a source ([research/landscape.md](research/landscape.md),
  [R1](research/R1-papers.md), [R2–R3](research/R2-R3-data-models.md), [R4](research/R4-repos-tools.md),
  [paper search records](research/papers/)).
- Plan with targets fixed before any work (F1 ≥ 0.85, p95 < 20 ms CPU) and a 10-gate Data Guard ([research/PRD.md](research/PRD.md)).
- Three spikes (~$0.03): LinCE offline; MADAR parallel available; "deviating if the Egyptian analyzer cannot parse it"
  scores F1 0.16; MARBERT tokenizes dialect best ([research/spike/](research/spike/)).
- Nine architectures scored, three finalists, the choosing rule written before any training
  ([research/architectures.html](research/architectures.html)).
- Pilot, six runs (~$0.95): backbones within 1 F1; changing the labels lifted realistic F1 0.63 → 0.80
  ([research/pilot/PILOT_REPORT.md](research/pilot/PILOT_REPORT.md)).
- Build-1 gold: 560 replies from five open LLM families, labelled with Claude's help; I fixed the Egyptian
  service register with 42 yes/no checks; a DeepSeek committee was recorded but not trusted (κ 0.21).
  Gold frozen with sha256 ([history/build1/annotation/](history/build1/annotation/)).
- I stopped the hosted LLM router after one costly judging run; every later label came from evidence I could inspect.

## 12–13 Oct · Build 1 data and training
- Data study without any LLM API: Dial2MSA, oasst2, Arabic STS, Nile-Chat, MADAR-6, Saudi customer-service chats,
  Najdi conversations, the Lahga dictionary ([history/build1/docs/DATA_STUDY.md](history/build1/docs/DATA_STUDY.md)).
- Labeler with three naturalness bits per word, audited on ~800K tokens in three rounds; 158K dialect↔MSA pairs
  aligned on an L4 (~$0.12); 92K training sentences; I checked 100 of 100 sampled sentences
  ([history/build1/docs/DATA_PLAN.md](history/build1/docs/DATA_PLAN.md), [g3 check](history/build1/annotation/g3_check_v1.md)).
- Nine runs; the first pointer head collapsed and was fixed; three seeds: Egyptian F1 0.966 ± 0.005, 74 behaviour
  tests, int8 ONNX 11.3 ms on CPU. Shipped seed chosen by dev, not gold
  ([history/build1/docs/RESULTS.md](history/build1/docs/RESULTS.md), [runs](history/build1/experiments/)).
- Packaged as an SDK, server and Docker image; tested from a fresh environment
  ([history/build1/report.html](history/build1/report.html)).

## 13–14 Oct · Build 2: the whole region
- Fifteen sources studied one by one ([docs/sources/](docs/sources/)): crawled mo3jam (31.6K definitions, 24 countries);
  obtained the full MADAR Lexicon (47.7K entries, 25 cities × 1,042 concepts) and filled the MADAR Corpus-26 form;
  emailed Prof. Mustafa Jarrar at Birzeit, who replied the same day, then filled the per-corpus forms with my
  university email and got Curras, Baladi, Nâbra and Lisan Iraqi / Yemeni / Sudanese / Libyan; added Alexandria,
  Levanti, Casablanca, Iraqi factlogic, Algerian darja. Ruled out sites that block AI crawlers or need a CAPTCHA.
- Inventory of 2.74M sentences and word-level separability per variety ([research/DATASET_OVERVIEW.md](research/DATASET_OVERVIEW.md)).
- Targets chosen from verified data; I merged Levantine into one target (MADAR distance 0.10–0.12 inside the Levant)
  ([history/build2/TARGETS_V2.md](history/build2/TARGETS_V2.md)).
- Bench (10,155 sentences, MADAR swaps + native sentences) frozen before the labeler; labeler v2 with nine bits;
  expectations pre-registered ([history/build2/PREREG.md](history/build2/PREREG.md)).
- Seven runs; the best-dev run failed the task example, so I added a known-answer gate and shipped the only run with
  54/54, recording the deviation from my own rule. 0.932 mean F1 on its bench
  ([history/build2/RESULTS_2.0.0.md](history/build2/RESULTS_2.0.0.md), [runs](history/build2/runs/)).

## 14 Oct · Build 3: fresh test, six targets, shipped
- Data audit: Gulf, Saudi and Egyptian native text was thin. New human-transcribed sources: SADA (Saudi TV, speaker
  dialect), my own MasriSpeech (52.9K Egyptian utterances), MGB-2 / MGB-3, SDAIA SCC, QCRI ArabicCulturalQA;
  model-transcribed sets kept to training; TTS and synthetic sets rejected ([research/VISION_V3.md](research/VISION_V3.md)).
- Label space: Gulf = Saudi + Gulf-east + Yemeni, Saudi also a target of its own; weak varieties become labels.
- Fresh test built and frozen before any Build-3 training data: speaker-disjoint SADA shows, Egyptian speech,
  Al Jazeera, Levantine and Iraqi corpora, MSA-drift injections, 200 LLM agent replies, and a held-out mo3jam probe
  (900 entries), my idea for testing words no model has seen ([docs/BENCH_CARD.md](docs/BENCH_CARD.md)).
- Round 1 (~$1.1): dev 0.83 = the labeler's own score (0.815). I found the teacher, not the model, was the ceiling.
- Gold cleaning: read all 558 MGB-2 sentences of the MSA sets, removed 70 colloquial ones by one rule on dev and test,
  without model output for test ([pipeline/scripts/apply_msa_review_v33.py](pipeline/scripts/apply_msa_review_v33.py)).
- Labeler v10, reviewed by a type-level label diff on 20K words; rules that hurt were rolled back; teacher 0.815 → 0.864.
  Clean rebuild, strict 6-gram leakage filter, label spot check, full pipeline smoke test on a tiny model; only then
  the GPU ([training/PREREG_V3.md](training/PREREG_V3.md), addenda 1–4).
- Round 2 (~$1.3): dev 0.876, known answers 71/73, linear chosen by the pre-registered rule; int8 export (99.41% same
  decisions, 7.1 ms); thresholds fitted on int8 dev; **test scored once: mean F1 0.869, 6/6 under 1% false alarms**.
  Build 2 on the same test: 0.819 ([docs/RESULTS.md](docs/RESULTS.md), [runs](experiments/build3/)).
- Released as HaSeen 3.0.0: package, CLI, server, Docker (offline, non-root), browser Space, model and data cards;
  verified from a fresh clone.
