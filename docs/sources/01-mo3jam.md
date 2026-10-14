# Source 01: mo3jam.com (crowd-sourced dialect dictionary), crawled 2026-10-13

- Crawl: `scripts/crawl_mo3jam.py` (term list from the 51 `/index/<letter>` pages; robots.txt allows /term and /index).
  23,114 term pages, 0 failures, 44 min, 183 MB gzipped HTML in `data/sources/mo3jam/raw/`.
  Parsed: `data/sources/mo3jam/definitions.jsonl` (one row per definition). Profile: `profile.json`
  (`scripts/analyze_mo3jam.py`). en.mo3jam.com and ar.mo3jam.com serve the same definitions (same ids).
- Row: def_id, term, kind (word / phrase / proverb), dialects (codes, several allowed), transliteration,
  defs {ar, en, fr: definition + example}, related (def ids + flags), votes up / down, user, date.

## Size
- 31,608 definitions · 23,083 terms · word 26,635 / phrase 3,916 / proverb 944.
- Single-word definitions 24,678 → **16,246 distinct normalized forms**; 14,328 (88%) are in neither Lahga nor the
  V1 core lexicon (only 126 overlap with core_v2 → few function words).
- Definition languages: ar 31,424, en 2,843, fr 1,636. Years 2009–2026 (peak 2021).

## Per dialect (definitions · single-word forms · with example · net-positive votes)
| Region | Countries (definitions) | Region single-word forms |
|---|---|---|
| Maghrebi MAA | Algerian 5,877 · Moroccan 3,760 · Tunisian 1,935 · Libyan 1,389 · Hassaniya 574 · "Maghrebi" 1,762 | 7,046 |
| Saudi SAA | Saudi 3,727 · Najdi 2,313 · Hijazi 1,980 | 4,548 |
| Gulf GAA | Kuwaiti 2,665 · Emirati 1,531 · Bahraini 1,088 · Qatari 913 · Omani 904 · "Khaliji" 2,072 | 3,963 |
| Levantine LEA | Palestinian 1,818 · Syrian 1,580 · Jordanian 1,264 · Lebanese 882 · "Levantine" 1,652 | 3,453 |
| Iraqi IRA | 3,827 | 2,477 |
| Egyptian EGA | 3,479 | 2,149 |
| Yemeni YEA | 2,780 | 1,913 |
| Sudanese SUA | 2,439 | 1,627 |

About 75% of definitions per dialect have an example sentence; 55–65% have a short or marked MSA gloss.

## Quality signals
- Votes: net ≥ 3 for 16,226 (51%), net ≥ 10 for 3,865; net negative 2,583 (8%); no votes 192.
- 68% written by guests; 1,803 named authors.
- Multi-tag: 26,512 definitions have one dialect; 1,260 have 8 tags and 138 have 20 ("all dialects") → 1,017 forms
  appear in all 8 regions. Pairwise region overlaps (1,100–1,900 forms) are mostly these pan-dialect forms.
- Related-terms graph: one giant component (13,254 definitions) + 517 small ones; 106 small components span ≥ 3
  regions. Use each definition's own related list (a concept neighbourhood), not the transitive closure.
- Content includes insults and vulgar terms (needs a filter), names, and phrases / proverbs.

## Verdict
Positive-evidence lexicon for 8 regions / 24 countries: strong for open-class and slang words, weak for function
words, no negative evidence. Best uses: (1) per-region lexical evidence for the labeler (weighted by votes);
(2) about 23K example sentences as dialect hosts; (3) MSA glosses for substitution families. Region level (8) is well
covered (1.6K–7K forms each); country level is thin for Lebanese, Qatari, Omani, Bahraini, Hassaniya (< 1K forms).
