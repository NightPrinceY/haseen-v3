# Source 14: BAMA & CALIMA morphological dictionaries (CAMeL Lab), checked 2026-10-14

## What exists (CAMeL Tools data catalogue 1.6, checked online)
| Package | Variety | Licence | Status for us |
|---|---|---|---|
| morphology-db-msa-r13 (BAMA / SAMA-derived, "calima-msa") | MSA | GPL v2 | **already local** since V1 (`HaSeen-V1/data/lexicons/raw/camel_msa`) |
| morphology-db-msa-s31 | MSA (SAMA 3.1) | LDC | needs the LDC SAMA licence; not needed |
| morphology-db-egy-r13 (CALIMA-EGY) | Egyptian | GPL v2 | already local (`camel_egy`) |
| morphology-db-glf-01 | Gulf | CC BY 4.0 | already local (`camel_glf`) |
| morphology-db-lev-01 | Levantine | CC BY 4.0 | already local (`camel_lev`) |
| camel_morph repo extras | EGY verb lexicon CSVs (SIGMORPHON 2022), `egy_lexprob.tsv` (60K lemma+POS counts), `glf_lexprob.tsv` (8.2K) | open GitHub | not downloaded yet (small) |
BAMA itself (Buckwalter 1.0/2.0) is MSA-only and is the ancestor of calima-msa / SAMA; nothing dialectal to add.
No CALIMA databases exist for Maghrebi, Iraqi, Yemeni or Sudanese.

## Profile of the local databases (stems section)
| DB | Stem entries | Stem forms | Lemmas | Glosses | Notes |
|---|---|---|---|---|---|
| EGY r13 | 161,101 | 59,651 | 69,793 (47,847 normalized) | 74,842 stems tagged `[CALIMA]` (Egyptian, from ECAL / Tharwa), **86,259 `[ARAMORPH]`** (MSA imports) | 18,097 normalized lemmas come only from CALIMA and none of them is in the MSA DB |
| GLF 01 | 22,911 | 11,073 | 5,330 normalized | none ("TBA") | 1,256 lemmas not in MSA DB (تكشخ, تغشمر, صاقع, متزاعل…) |
| LEV 01 | 30,348 | 11,300 | 5,433 normalized | none | 1,157 not in MSA DB (فرفح, كلشي, مزيونه…); some noise (هذه listed) |
| MSA r13 | 72,734 | 37,301 | 38,568 | yes | the MSA reference lexicon |
Overlaps: GLF ∩ LEV lemmas 2,348; GLF ∩ EGY-only 519; LEV ∩ EGY-only 457.

## What it is good for in V2
1. **Morphology, not dialect labels.** The analyzers can generate every inflected surface form of a lemma
   (clitics, person, aspect). That is how lexicon entries (MADAR, mo3jam, Lahga) can be expanded from a lemma to all
   the forms that appear in real text (e.g. عايز → عايزة / عايزين / مش عايزك…).
2. **MSA reference**: "analysable by calima-msa" is evidence a form is MSA-valid (used in V1 for the msa stem list).
3. **Egyptian-specific lemmas** (18K CALIMA-only) and the lexprob counts are extra Egyptian evidence.
Limits (seen in V1): analyzers over-generate (morph-rule F1 0.16 as a detector), EGY mixes MSA imports, GLF / LEV
have no glosses and are small, and only 3 dialects are covered.

Verdict: **support tool, already in hand.** Use for inflection expansion and MSA validity; not a dialect lexicon
on its own. Optional: fetch `egy_lexprob.tsv` / `glf_lexprob.tsv` from camel_morph for lemma frequencies.
