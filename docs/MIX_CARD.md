---
license: other
license_name: research-only-see-card
language: [ar]
pretty_name: HaSeen-3-Mix
task_categories: [token-classification]
tags: [arabic, dialects, code-switching, weak-supervision, voice-agents]
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: dev.jsonl
---
# HaSeen-3-Mix · word-level naturalness labels for Arabic varieties (private)

Training data of **HaSeen 3**: 501,080 train / 25,935 validation sentences, about 6.2M words. Every word carries one
naturalness bit per variety (1 natural, 0 not natural, -1 unknown = masked in the loss), a per-variety register
flag (`frm`: formal but acceptable in the service register), `odia` (natural in none of the varieties: Sudanese,
Libyan, Algerian, Hassaniya, …) and `other` (names, digits, Latin script).

Row: `{"id", "family", "source", "host", "sub", "tokens", "bits" [[7]], "frm" [[7]], "odia", "other", "inj", "pid",
"split", "bits9", "host9"}` with varieties in the order **MSA, EGY, SAU, GLF, LEV, IRQ, MAG**.
`bits9` keeps the labeler's nine fine-grained bits (MSA, EGY, SAU, KHL, LEV, IRQ, YEM, MOR, TUN). The V3 bits are
derived from them: **GLF = Saudi ∪ Gulf-east ∪ Yemeni**, **MAG = Moroccan ∪ Tunisian** (1 if any member is 1,
0 only if every member is 0, else unknown). Saudi is both its own variety and a Gulf member.

## Families
| Family | What | Train rows |
|---|---|---|
| A | native sentences of a known variety, labelled word by word | 420,123 |
| B | MADAR Corpus-26 train: words swapped with the aligned word of another city for the same concept | 13,001 |
| C | lexicon substitution into native sentences (MADAR concept forms of other varieties) | 56,319 |
| D | MSA future transforms (سوف / سـ + verb in place of the dialect future) | 11,637 |

New native sources in V3 (training parts only, every bench hold-out removed): SADA TV shows (51 training shows;
Najdi, Hijazi, Khaliji speakers), MasriSpeech (Egyptian), MGB-3, SDAIA SCC, QCRI ArabicCulturalQA (parallel,
native post-edited: Egyptian, Levantine, Gulf, MSA, Maghrebi).

## Labels (words, train + validation)
| Variety | natural (1) | not natural (0) | unknown (-1) |
|---|---|---|---|
| MSA | 4,344,123 | 934,580 | 938,955 |
| EGY | 4,840,621 | 465,813 | 911,224 |
| SAU | 4,951,806 | 296,877 | 968,975 |
| GLF | 5,368,879 | 189,489 | 659,290 |
| LEV | 5,264,363 | 224,739 | 728,556 |
| IRQ | 4,758,847 | 435,026 | 1,023,785 |
| MAG | 4,550,085 | 287,287 | 1,380,286 |
other-dialect words: 6,347 (before the leakage filter).

## Labeler (v10)
`pipeline/haseen2/labeler.py`: expert evidence first (MADAR Lexicon 25 cities × 1,042 concepts with 25% of concepts
held out for evaluation, per-variety MSA marker statuses, a core lexicon validated by a native Egyptian speaker,
CALIMA-MSA), voted crowd lexicons, corpus frequency; no evidence → unknown. V10 rules: exact MADAR city forms win
over clitic stripping (لعند is not عند), names and news words used in MSA text stay MSA (العراق), corpus positives
need a real rate (a word leaking into another variety's text at < 8% of its home rate is not counted: حضرتك in
Yemeni text), strong corpus absence of a known dialect form is negative evidence, the هـ-future is Egyptian only,
سيتم is a hard MSA marker. Teacher score on HaSeen-3-Bench **dev**: challenge F1 0.84–0.91, natural false alarms
≤ 0.81% (all six targets).

## Leakage
No sentence and no normalized 6-gram shared with any HaSeen-3-Bench file (869 rows removed; a second pass removes
0). Bench sentence hashes, mo3jam probe entries (examples and lexicon rows) and SADA test / dev shows are excluded
from the inventory.

## Sources and licences
MADAR Corpus + Lexicon (CMU-Q / NYUAD, research licence: internal research only, no redistribution), SADA (CC BY-NC-SA),
QCRI MGB-3 and ArabicCulturalQA, SDAIA SCC, MasriSpeech, Alexandria (UBC, CC BY-NC 4.0), Birzeit Curras / Baladi /
Nâbra / Lisan (research licences), Levanti, Arabic STS, Dial2MSA, Nile-Chat test, Iraqi factlogic, Algerian darja,
mo3jam / Lahga (crowd, voted), oasst2 subset. **This dataset is private and must not be redistributed**; several
sources are research-only or non-commercial.
