# HaSeen 3 — targets, label space and how it plugs into a voice-agent QA product

## 1. Why this shape
- A voice-agent reliability product needs to know, per agent turn, whether the agent kept the dialect it was
  configured for, and which words broke it. Most agents drift (to MSA, to Egyptian, to another Gulf variety), and
  LLM-judge evals are weak at Arabic dialect identification.
- V3 offers only targets that the data can support at the release gate, groups the Gulf varieties the way agents
  are configured ("Khaleeji"), and keeps the weak varieties as **labels** instead of targets.

## 2. Targets (what a user can ask for)
| Target | Varieties | Notes |
|---|---|---|
| `msa` | Modern Standard Arabic | formal open-class words are shared with every dialect (service register) |
| `egy` | Egyptian | |
| `sau` | Saudi: Najdi, Hijazi, Southern | standalone target **and** a member of Gulf |
| `glf` | Gulf: Saudi, Gulf-east (UAE, Qatar, Kuwait, Bahrain, Oman), Yemeni | a word conforms if it is natural in any member |
| `lev` | Levantine: Palestinian, Jordanian, Lebanese, Syrian | |
| `irq` | Iraqi | |

Aliases: `gulf`, `khaleeji`, `emirati`, `kuwaiti`, `yemeni`, `omani` → `glf`; `najdi`, `hijazi` → `sau`; `shami` → `lev`;
Arabic names (مصري، سعودي، خليجي، شامي، عراقي، فصحى) are accepted too.

## 3. Labels (what a deviating word can be said to be)
The model outputs seven naturalness probabilities — **MSA, EGY, SAU, GLF, LEV, IRQ, MAG** (Maghrebi: Moroccan,
Tunisian) — plus *other dialect* (natural in none: Sudanese, Libyan, Algerian, Hassaniya …) and *other* (names,
numbers, Latin script). A deviating word gets the set of varieties with p ≥ 0.5, most likely first. Maghrebi is a
label only: there is not enough native Maghrebi data to promise a Maghrebi target at the gate. For target Gulf,
Saudi is never named as the source (it is inside Gulf).

## 4. Release gate (pre-registered in `training/PREREG_V3.md`)
Per target on HaSeen-3-Bench test: natural false alarms ≤ 1% of words, challenge F1 ≥ 0.90; plus the known-answer
suite (the task example and ≥ 95% of 73 word checks). Results and what passed: `docs/RESULTS.md`.

## 5. How it plugs into a voice-agent QA product
- **Eval "Dialect adherence"**: per agent turn, share of Arabic words that fit the configured dialect plus the
  deviating words with their labels; PASS / FAIL by threshold or a score.
- **Red flags / alerts**: "agent drifted to MSA", "Gulf agent used Egyptian words", trend per agent version.
- **Caller dialect**: `summary.reads_as` ranks the varieties a turn reads as (from STT text).
- **Simulation**: verify that a simulated "Saudi caller" actually speaks Saudi.
- **Real-time guard** before TTS: one encoder pass, ONNX int8 on CPU, offline, no LLM call.
