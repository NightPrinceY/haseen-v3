# Source 07: the printed dictionaries behind Living Arabic, and open alternatives (checked 2026-10-13)

Option 3 after Living Arabic was ruled out: find the source dictionaries through other channels.

## Printed dictionaries (archive.org metadata search; nothing downloaded)
| Dictionary | Year | Status | Script | Value for V2 |
|---|---|---|---|---|
| Hinds & Badawi, *A Dictionary of Egyptian Arabic* | 1986 | in copyright; no archive.org item | Arabic + transcription | high, but no legal digital copy |
| Qafisheh, *NTC's Gulf Arabic-English Dictionary* | 1997 | in copyright; archive.org **lending only** | Latin transcription | — |
| Woodhead & Beene / Clarity et al., Iraqi (Georgetown) | 1964–67 (repr. 2003) | in copyright; archive.org lending only | Arabic + transcription | — |
| Harrell, *A Dictionary of Moroccan Arabic* (Georgetown) | 1963–66 | in copyright; lending copies + some unofficial uploads | Latin transcription | low (transcription) |
| Stowasser & Ani, *A Dictionary of Syrian Arabic* (Georgetown) | 1964 | archive.org item `dictionaryofsyri00karl` | Latin transcription | low (transcription) |
| Spiro, Egyptian colloquial vocabularies | 1895 / 1897 | public domain (several archive.org scans) | Arabic | low: 130 years old, needs Arabic OCR |
| English-Arabic vocabulary, Anglo-Egyptian Sudan | 1905 | public domain | Arabic | low: old, OCR |
| Peace Corps English-Tunisian Arabic Dictionary (ERIC ED183017) | 1977 | US government, free | mostly transcription | low |

Conclusion: the strong printed dictionaries are in copyright and only lendable page by page; the free ones are old
scans or Latin transcription. OCR + transcription conversion would cost a lot for little gain. Not pursued.

## Open, structured alternatives found on the way (better than the printed books)
| Resource | Dialects | Size | Access | Why it matters |
|---|---|---|---|---|
| **Currasat portal** (Birzeit; Curras, Baladi, Nâbra, Lisan) https://sina.birzeit.edu/currasat/ | Palestinian 56K tokens, Lebanese 9.6K, Syrian 60K (10 local dialects), Yemeni 1.05M, Iraqi / Libyan / Sudanese ~50K each | ~1.3M annotated tokens | free licensed copy by **Google form** request (Yahya fills it); portal copyright Birzeit, robots allows | Real sentences, and **every token has a dialect lemma, an MSA lemma and an English gloss** → word-level dialect↔MSA alignment for 7 countries with no alignment tool |
| **Kaikki / Wiktionary** (CC BY-SA, already used for EGY / GLF in V1) | South Levantine 4,361 senses, Hijazi 3,222, Moroccan 2,804, Egyptian 1,955, North Levantine 1,156, Gulf 1,021, Tunisian 885, Cypriot 503 | ~16K senses | open JSONL download | V1 used only Egyptian + Gulf |
| **DODa** Darija Open Dataset https://github.com/darija-open-dataset/dataset | Moroccan | ~150K entries, 86K translated sentences (4.7 MB sentences.csv) | open GitHub | the largest Moroccan lexicon + sentences |
| **TUNICO** Tunis Arabic dictionary https://tunico.acdh.oeaw.ac.at | Tunisian | 7,543 entries | open web (ACDH Vienna) | expert, corpus-based |
| Tharwa (Columbia) | Egyptian ↔ MSA ↔ English | 73K entries | paper says public; old download URL dead (404) | would be strong for EGY↔MSA |
| MADAR Lexicon | 25 cities | 1,045 concepts × 25 | request form (research licence) | concept-parallel across 25 cities |
