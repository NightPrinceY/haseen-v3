# Spike results (2026-10-12)

Ran on an HF Sandbox (cpu-upgrade, $0.03/h, ~30 min ≈ $0.02).

## 1. LinCE MSA-EA: NOT available
`lince-benchmark/lince` is only a loader script; the data lives on ritual.uh.edu, which times out
(from Yahya's machine and from HF). No HF mirror, no Wayback snapshot.
→ There is **no real word-level labeled MSA/EGY/GLF data** we can use. Synthetic + teacher-labeled data is mandatory.

## 2. MADAR parallel corpus: available (key resource)
`Elfsong/MADAR_Corpus_Merged`: **11,600 sentences**, each in Cairo, Doha (Gulf), MSA, Beirut, Rabat, Tunis + English.
Splits: 9,000 train / 1,000 dev / 1,600 test (corpus-6). Travel domain, short sentences.
Example: عايزين ترابيزة جنب الشباك (Cairo) / بغينا طاولة يم الدريشة (Doha) / نريد مائدة بجانب النافذة (MSA).
→ Parallel triples let us build mixed sentences with exact labels and auto-detect SHARED words.
(License of the mirror: check MADAR terms before redistributing; we redistribute only derived synthetic data + scripts.)

## 3. Real monolingual text: usable
- FineWeb2 Egyptian: ~75% of rows are usable Arabic paragraphs (≥6 words, >90% Arabic letters); real Egyptian blogs.
- FineWeb2 Najdi: ~59% usable; real Saudi Gulf forum text.
- Gulf tweets (CC-BY-4.0), Egyptian↔MSA parallel 18k (MIT), SADID+MADAR Egyptian with CODA-fied text.

## 4. CAMeL morphological analyzers: weak discriminators
Pseudo labels from MADAR set differences (noisy: paraphrases inflate the MSA/GLF classes).
% of words analyzable by each DB:

| class | n | msa db | egy db | glf db |
|---|---|---|---|---|
| SHARED | 7,870 | 97.2 | 98.1 | 84.2 |
| EGY-specific | 40,978 | 66.8 | 82.9 | 58.2 |
| GLF-specific | 35,934 | 71.8 | **76.9** | 70.7 |
| MSA-specific | 63,014 | 98.4 | **98.9** | 78.1 |

The EGY database analyzes ~99% of MSA words and ~77% of Gulf words (it over-generates).
Rule "deviating if not analyzable in EGY db": **P=0.56 R=0.09 F1=0.16**.
→ Morphology cannot be the main signal. Keep it only as a weak feature (signature like "glf only").
→ The decision must come from lexical + contextual knowledge learned from data: the encoder, trained on good data.

## Consequences for the design
1. Data is the core of the solution: (a) MADAR-based mixing with exact labels, (b) lexicon injection into real FineWeb2-arz sentences, (c) Claude-labeled sentences.
2. A curated closed-class **marker lexicon** (high precision) replaces morphology as the rule-based layer.
3. Baseline to beat for the README: the morphology rule (F1 0.16) and the marker lexicon alone.

---
# Spike 3 (2026-10-12): ready-made token model, tokenizers, CPU latency (~$0.01)

## ALDi-Token-DI (MARBERT fine-tuned on CALCS-2016 MSA/DA token labels)
| Sentence | Result |
|---|---|
| Task example (EGY + أبغى GLF + سوف MSA) | every word DA, **سوف missed** (DA 0.87) |
| EGY + MSA phrase (لن نستطيع ... الذي) | MSA phrase found, but MSA label **spreads** to neighbours (نغير، الميعاد، اخترته) |
| Pure MSA | all MSA ✓ |
| EGY + Gulf (وايد، الحين) / pure Gulf | all DA (no EGY vs GLF distinction, as expected) |

Lessons:
1. Models trained on natural code-switching learn **span-level smoothing**: a single foreign word inside a dialect sentence gets absorbed. That is exactly the LLM failure mode Tuner cares about → our training data must contain **single-word insertions** (synthetic injection does this).
2. Useful as warm start / MSA-vs-DA signal, not as the solution.

## CPU latency (MARBERT-base token classifier, fp32, 4 threads, 15 subwords)
p50 = 30.1 ms, p95 = 33.9 ms → above our 20 ms CPU target → need ONNX + int8 (or a smaller model) for CPU; GPU is fine.

## Tokenizer fertility on 30k real Egyptian words (FineWeb2-arz)
| Model | subwords/word | words with [UNK] | vocab |
|---|---|---|---|
| CAMeLBERT-Mix | 1.93 | 17.0%* | 30k |
| MARBERTv2 | **1.86** | **2.3%** | 100k |
| AlcLaM-v2 | 1.86 | 2.3% | 100k (same vocab as MARBERT) |
| AraBERTv02-twitter | 1.87 | 17.0%* | 64k |
| mmBERT-small | 2.80 | 0.0% | 256k |

*Caused by diacritics (shadda, fatha, tanween) and tatweel in the text: CAMeLBERT/AraBERT vocabularies have none. Normalization (dediacritize + remove tatweel) fixes it → normalization is mandatory in the pipeline.
mmBERT makes ~50% longer sequences → slower and worse word alignment for Arabic.
