# Source 08: Currasat corpora (Birzeit SinaLab), checked 2026-10-13

Portal: https://sina.birzeit.edu/currasat/ (Curras = Palestinian, Baladi = Lebanese, Nâbra = Syrian, Lisan = Yemeni /
Iraqi / Libyan / Sudanese). The licensed-copy Google form is closed ("no longer accepting responses").

## What we got: Curras (Palestinian), complete
- The lab publishes it in its own public GitHub site repo: `SinaLab/sinalab.github.io/currasat/curras/curras.zip`
  (3.9 MB → `curras.csv`, 44.8 MB, 2020-09-30). Local: `data/sources/currasat/curras.{zip,csv}`.
- 55,783 tokens · 46 documents · 5,228 sentences · 46,415 Arabic tokens. 80 columns per token: sentence, word,
  CODA spelling, **Palestinian lemma, MSA lemma**, English gloss, prefixes / stem / suffixes with POS, person, aspect,
  gender, number.
- 7,705 tokens (16.6%) have a Palestinian lemma different from the MSA lemma → 1,835 distinct dialect→MSA pairs,
  dominated by **function words**, exactly what mo3jam lacks: مش→ليس 319, شو→ماذا 221, بس→لكن/فقط, بد→أراد/يريد,
  اللي→الذي, وين→أين, احنا→نحن, هيك→هكذا, زي→مثل, عشان→أجل, ليش→لماذا, حدا→أحد, كمان→أيضا, فش→لا يوجد, هلقيت→الآن.
- Use: real Palestinian host sentences with **token-level MSA alignment built in** (no SimAlign needed), plus a
  high-precision Levantine function-word lexicon.

## What we could not get
- Baladi (Lebanese 9.6K), Nâbra (Syrian 60K), Lisan (Yemeni 1.05M, Iraqi / Libyan / Sudanese ~50K each) are only
  served through the portal's search API (`ontology.birzeit.edu/sina/v2/api/FilterCurrasBiladiResults`), which sits
  behind a **Cloudflare bot challenge** (HTTP 403 `cf-mitigated: challenge`). Not bypassed.
- No copies on Hugging Face or GitHub (searched: curras, lisan, nabra, baladi, dialect corpus names). SinaTools'
  downloader only fetches MSA resources (lemmas_dic, n-grams, NER).
- Route left: email the authors (Mustafa Jarrar, Tymaa Hammouda; Birzeit SinaLab) asking for the Lisan / Nâbra /
  Baladi files, since the form is closed.

## Update 2026-10-14: per-corpus download forms (Prof. Jarrar replied "Try this" + the about page)
The about page (https://sina.birzeit.edu/currasat/about-en.html) has one Google Form per corpus, licence
**CC BY 4.0**: Curras, Baladi, Lisan Iraqi (45K), Lisan Yemeni (~1.05M), Lisan Sudanese (52K), Lisan Libyan (51K),
Nâbra (Syrian, 10 local dialects). The forms are open, but each says: access only for users **affiliated with a
company, university or institution**; fields = full name, organization, **official (non-Gmail) email** (the
download link is sent there), city / country. Yahya has to submit them himself.
Form ids: Curras 1FAIpQLSehTz…, Baladi 1FAIpQLSeYjV…, Iraqi 1FAIpQLSe3lz…, Yemeni 1FAIpQLSfIhp…,
Sudanese 1FAIpQLSe30a…, Libyan 1FAIpQLSdcbw…, Nâbra 1FAIpQLSeo18….

## Obtained (2026-10-14): all six free editions (forms submitted by Yahya + Claude in the browser pane)
Local: `data/sources/currasat/{baladi,nabra,lisan_iraqi,lisan_libyan,lisan_sudanese,lisan_yemeni}/x_*/` — each has
`*-dataset.csv` (one row per token), `* RowText_sentences.csv`, ReadMe.pdf, license.pdf (CC BY 4.0, cite),
tagset_translation.xlsx. Download links came from Dropbox on the form confirmation page.

| Corpus | Tokens | Sentences | Token-level DA lemma ≠ MSA lemma | Notes |
|---|---|---|---|---|
| Baladi (Lebanese) | 9,561 | 457 | 1,627 (19.4%), 437 pairs | الي→الذي, اشي→شيء, عم, مش→ليس, هيدا→هذا, بد→أراد, شو, هيك, كرمال, رح→سوف; train/val/test split |
| Nâbra (Syrian) | 60,021 | 5,937 | 8,563 (15.6%), 1,837 pairs | **sub-dialect per token**: Shami 17.2K, Halabi 9.2K, Sahili 7.9K, Raqqawi 6.3K, Deiri 4.3K, Homsi 4.1K, Daraawi 3.8K, Suwayda 3.1K, Hamawi 2.3K, Mardini 1.6K; uses a CODA column |
| Lisan Iraqi | 45,881 | 3,328 | 259 (DALemma rarely filled) | free edition: dialect words often keep their own form as "MSALemma" (شلون, اكو, كلش) or map straight to MSA (هواية→جدا); gloss + POS always present |
| Lisan Libyan | 51,686 | 3,053 | 40 | same scheme as Iraqi |
| Lisan Sudanese | 52,616 | 3,000 | 25 | same scheme |
| Lisan Yemeni | 994,413 | 35,318 | 1,668 | same scheme; Twitter, political topics (Houthi, Afash) |

Reading: Curras / Baladi / Nâbra give clean token-level dialect→MSA lemma alignment (Levant). Lisan's free edition
is mainly **real native sentences with POS + English gloss per token** for Iraq, Libya, Sudan and Yemen (the Premium
edition adds "additional lemmas and features"); its MSALemma column mixes dialect forms and MSA equivalents, so
use the gloss / POS, not the lemma pair, as alignment evidence.
