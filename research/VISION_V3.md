# HaSeen V3 — data audit and label-space decision (2026-10-14)

Goal: train only what we can make strong, judged on data nobody has looked at. V3 is the version that ships.

## 1. What the data audit found
The label space was not the main problem; **native data for the markets that matter was**:

| Variety | Sentences in the inventory | What they really are |
|---|---|---|
| Saudi | 40K | 22K synthetic (saudi_conversations), ~11K Alexandria translations, 5K mo3jam examples |
| Gulf-east (AE, QA, KW, BH) | ~16K | 10K MADAR translations (Doha), 3K Casablanca / Habibi, 2K mo3jam |
| Omani | 9K | Alexandria translations |
| Yemeni | 47K | 39K Lisan (native) — the best Gulf-side source we had |
| Egyptian | 226K | mostly synthetic (Levanti 156K, oasst2 36K); ~25K verified |
| Levantine | ~400K | 466K synthetic Levanti across PS / SY / LB / JO; ~70K native |
| Iraqi | 344K | factlogic web text (native) |
| Algerian | 1.5M | spoken (not a target) |
Our strongest targets were the ones with native text; Gulf-east, Yemeni and Moroccan were weak because their text
was thin or translated.

## 2. New native sources found (never used, verified accessible)
| Source | Content | Size | Use |
|---|---|---|---|
| **SADA** (SDAIA + Saudi Broadcasting Authority), text-only mirror `khaledalganem/sada2022` | transcribed speech from 64 Saudi TV shows, speaker dialect per utterance | Najdi 90.7K, Hijazi 34.8K, Khaliji 28.5K, Southern / Northern 0.2K (+ MSA 4K, Egyptian 2K, Levantine 0.9K, Yemeni 0.4K) | Saudi + Gulf training **and** a fresh test (split by show) |
| **MasriSpeech-Full** (Yahya's own) | transcribed native Egyptian speech | 52.9K utterances | Egyptian training + fresh test |
| Annotated Gumar (CAMeL Lab) | Emirati novels, 200K words, sentence-level dialect | small | Gulf-east check set (to request) |
| Rejected after inspection | Gulf tweets 2018–20 (mixed MSA / Libyan), ASLC-448 (scripted, TTS audio), QADI (tweet IDs only) | | |
SADA is CC BY-NC-SA (research use); MasriSpeech is ours.

## 3. Label space
Production targets (what a customer can choose):
| Target | Members | Why |
|---|---|---|
| MSA | — | strong |
| Egyptian | — | strong; + MasriSpeech |
| **Gulf** | Saudi, Gulf-east, Yemeni, Omani | distances 0.13–0.15 between members; one big native pool (SADA + Lisan) |
| **Saudi** | (inside Gulf) | strong alone; largest Gulf market; SADA gives 125K+ native utterances |
| Levantine | PS, JO, LB, SY | strong (lowest false alarms) |
| Iraqi | — | strong |
Labels only (a deviating word is named, but they are not targets): **Maghrebi** (Moroccan, Tunisian, Algerian,
Libyan, Hassaniya), **other dialect** (Sudanese, anything else).
How "Saudi inside Gulf" works: the model answers separate yes / no questions per word: "natural in Saudi?" and
"natural in any Gulf variety?". شو → Saudi no, Gulf yes; وش → both yes. A Saudi word is always Gulf; not the reverse.

## 4. Data rules
1. Native text first; synthetic text (Levanti synthetic, oasst2, saudi_conversations) capped and train-only.
2. Every source is split by document / show / speaker **before** anything is built; test documents never enter
   training, lexicons or counts.
3. Corpus counts for the labeler are recomputed without the test documents.
4. Each new source gets a 100-sentence quality check before use (dialect correct? transcription clean?).

## 5. Fresh test (built and frozen before training; never used for any decision)
| Part | From | Measures |
|---|---|---|
| natural | held-out SADA shows (Najdi, Hijazi, Khaliji), held-out MasriSpeech, held-out Lisan / Levanti-native / Iraqi documents | false alarms on real native speech |
| challenge | the same held-out sentences with 1–3 words swapped for another variety's word of the same MADAR concept (new concept hold-out) | catching deviations, naming their source |
| lexical probe | 20% of mo3jam entries (net votes ≥ 3) held out with their examples before training, e.g. دبوج (Najdi) | unseen dialect words: must deviate for other targets, conform for their own |
| agent (optional) | LLM replies "in dialect T", reviewed by natives (Yahya: Egyptian) | the real failure mode |
The V2 gold (HaSeen-2-Bench, test included) becomes the **dev** set for thresholds and model choice.

## 6. Decisions fixed before training
- Ship gate per target: natural false alarms ≤ 1%, challenge F1 ≥ 0.90, known answers 100%, lexical probe ≥ 0.80.
- Model choice on dev only; when dev differences are within seed noise, more known answers wins.
- One training round with the chosen configuration (+ a linear and a pointer seed), one test evaluation, published as is.

## 7. Cost and time
Sources: free (text-only reads). Building + labeling: local. Training: ~$1–1.5 (L4, two runs in parallel).
Optional agent set: ~$1 (open-weight LLMs on a sandbox) + native reviewers.
