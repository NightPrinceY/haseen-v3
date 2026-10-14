# Source 13: MADAR Lexicon (CAMeL Lab, NYU Abu Dhabi), checked 2026-10-14

## Did V1 use it?
No. V1 used only the **MADAR-6 parallel corpus** (HF mirror `Elfsong/MADAR_Corpus_Merged`: 11,600 sentences,
Cairo / Doha / Beirut / Rabat / Tunis + MSA). The lexicon was listed as "nice to have, needs a request" in
`research/R2-R3-data-models.md` and was never obtained.

## What it is (Bouamor et al., LREC 2018; used by the AGS paper, arXiv 2508.17347)
- 1,045 concepts, each defined by an (English, French, MSA) triplet, with dialect forms for **25 cities**
  (Aleppo, Alexandria, Algiers, Amman, Aswan, Baghdad, Basra, Beirut, Benghazi, Cairo, Damascus, Doha, Fes, Jeddah,
  Jerusalem, Khartoum, Mosul, Muscat, Rabat, Riyadh, Salt, Sanaa, Sfax, Tripoli, Tunis).
- 47,466 dialect entries (about 45 per concept); each form has **CODA spelling + CAPHI pronunciation**.
- Expert-built (native annotators per city), concept-parallel across all 25 cities and MSA.

## Access
- Download = Google Form "CAMeL Lab Registration Form: MADAR Lexicon"
  https://docs.google.com/forms/d/e/1FAIpQLSe2LHYmHsxdkHPYHgcZDz25dTNbnygPkmClIaLd_fwud-XnTQ/viewform
  (open on 2026-10-14; non-commercial licence). Yahya fills it.
- The same site has the MADAR Corpus-26 form (2,000 sentences × 25 cities + MSA, fully parallel) and a searchable
  lexicon interface. Not on HF or GitHub.

## Why it matters for V2
The only resource with the same concepts in 25 cities + MSA → cross-dialect equivalence sets for substitution
(V1 family C, now for 25 cities), "same concept, different form" evidence (a strong kind of negative evidence:
if Cairo's form for a concept is X, the Riyadh form Y is not Cairene), and CODA spellings for normalization.
Limit: travel-domain concepts (from BTEC), 1,045 concepts.

## Bonus found on the way
- **Maknuune** (CAMeL Lab): open Palestinian lexicon, ~36K entries; the GitHub repo
  `CAMeL-Lab/maknuune_lexicon` holds the full dictionary as LaTeX per letter (parsable).

## The paper itself (Bouamor et al., LREC 2018, pp. 3387–3396; local `research/papers/lrec2018-351-madar.pdf`)
What it adds beyond the summary above:
- **Lexicon row format**: concept (En, Fr, MSA lemma triplet + POS, plus a "phrasal" frequent inflected form) →
  dialect rows (CODA, CAPHI, **list of city codes that use this form**). Example "very": برشا Tunis/Sfax · بزاف
  Rabat/Fes/Algiers · جدا Cairo/Alexandria (g) and Jeddah/Khartoum/Riyadh (j) · خالص Cairo/Alex/Aswan · كتير
  Beirut/Jerusalem/Damascus/Aleppo/Amman/Fez/Rabat · كلش Basra/Baghdad/Mosul/Doha · واجد Basra/Baghdad/Doha/Benghazi/
  Tripoli/Muscat · مرة Jeddah · هلبة Tripoli · عوم Muscat. → one word maps to **several cities**, the exact multi-label
  shape V2 needs.
- **How it was built**: automatic seeding from Tharwa (Egyptian / Levantine), Karmous (Tunisian, ~3.8K words),
  a Moroccan textbook and the LDC Iraqi lexicon; then **13 native linguists** validated every row, deleted
  irrelevant entries, added missing ones, and were told to "think of more than one translation and carefully specify
  the city". Average **1.8 forms per concept per dialect**; 1,899 forms per dialect. Three quarters of concepts are
  open class; covers 85–88% of BTEC lemma tokens.
- **Negative evidence**: because each of the 25 cities was filled in for every concept, "city X is not listed for
  form F of concept C, but X has its own forms for C" is real, sense-level evidence that F is not X's word for C.
  This is stronger than "absent from a dictionary".
- **Dialect distances (Corpus-25 vocabularies)**: average pairwise overlap 26.3%; closest Amman–Jerusalem 54.4%;
  most distant Sfax–Alexandria (87.4% different); Muscat is the closest city to MSA (37.5% overlap), Sfax the
  furthest. → cities inside a sub-region are close; regions are far apart.
- **Taxonomy** used by MADAR: 5 regions → **10 sub-regions** (Morocco, Algeria, Tunisia, Libya, Egypt/Sudan,
  South Levant, North Levant, Iraq, Gulf, Yemen) → 25 cities. A principled starting point for V2 target levels.
- Corpus guidelines: translations made from English/French (not MSA, to avoid priming), no code-switching, and a
  widely used MSA word is kept when a dialect uses it (useful for our "shared" label).
- New leads named in the paper: **AVIA** multi-dialect lexicon (Al-Ain UAE, Baghdad, Jeddah, Jerusalem, **Kuwait
  City, Doha**, Sana'a), the **Manchester Database of Arabic Dialects** (15 countries), **Tharwa**, LDC ECAL (Egyptian)
  and IAML (Iraqi), YouDACC (YouTube dialect comments), ArapTweet (multi-dialect Twitter), Gumar (Gulf).

## We have it: MADAR Lexicon v1.0 (release 2022-11-10), shared by Yahya on 2026-10-14
Local: `data/sources/madar_lexicon/MADAR_Lexicon_v1.0/{MADAR_Lexicon_v1.0.tsv, README.txt, LICENSE.txt}`
(research / evaluation licence, CMU-Q + NYUAD). Columns: ID, Concept_ID, Class (Open / Closed / Mixed), English,
French, MSA, Dialect (city code; Sfax = `SFA`), Example, En/Fr/MSA lemma+POS, CODA, Tokenization (D3), CAPHI,
Release Status (READY / CHECK), Release Note.

Profile:
- 47,721 rows · 1,042 concepts · **every concept is filled for all 25 cities** (1,042 / 1,042) → complete
  concept × city matrix = sense-level negative evidence everywhere.
- Status: READY 42,335 · CHECK 5,386 (11%, flagged for further QC → lower weight).
- Rows per city 1,471 (Tripoli) – 2,825 (Amman). Class: open 32,261 · mixed 9,413 · **closed 6,047 (167 closed-class
  concepts × 25 cities)** → the function words mo3jam lacks: what = ايش 11 cities / شنو 7 / شو 6 / ايه 5 / وش 2…;
  now = هسا / هلق / الحين / توا / دلوقتي / دابا / هلقيت…; why = ليش 15 / ليه 11 / علاش 6…; where = وين 18 / فين 12;
  not = ما 22 / مش 11 / مو 7 / موب / موش…; how = كيف 14 / شلون 7 / كيفاش 5 / ازاي 3.
- 5,012 distinct single-word forms (normalized CODA). By number of sub-regions using a form: 1 → 2,579, 2 → 685,
  … all 11 → 223 (pan-Arabic). Sub-region-exclusive forms: Gulf 565, Yemen 339, Algeria 284, Tunisia 250,
  Morocco 239, Iraq 193, N. Levant 184, Egypt 178, S. Levant 153, Libya 109, Sudan 85.
- City-exclusive forms are mostly single-city sub-regions (Algiers 284, Sana'a 339, Muscat 326, Khartoum 85,
  Doha 73); inside multi-city sub-regions, cities rarely own a form (Cairo 2, Amman 4, Baghdad 4) → **city level is
  too fine for word-level targets; sub-region level is where words separate.**
- Overlap with other sources: 1,453 forms also in mo3jam, 1,406 in Lahga, **2,877 (57%) in neither**.

Verdict: **top-tier source.** Expert, complete 25-city × 1,042-concept matrix with closed-class coverage and CODA
spellings. Backbone for (1) negative evidence, (2) cross-dialect substitution sets for 25 cities, (3) the
function-word layer, (4) choosing the target granularity (sub-region).
