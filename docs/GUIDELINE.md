# HaSeen — Annotation guideline v3.0 (2026-10-14)

V3 keeps the v2.0 guideline below word for word: the labeler still answers for the nine fine-grained varieties
(MSA, EGY, SAU, KHL, LEV, IRQ, YEM, MOR, TUN). What V3 changes is the label space the model learns and the targets a
user can choose:
- **Gulf (GLF) = Saudi ∪ Gulf-east (KHL) ∪ Yemeni (YEM)**: a word is natural in Gulf if it is natural in any member,
  not natural only if it is not natural in every member, otherwise unknown. Saudi stays a standalone target too.
- **Maghrebi (MAG) = Moroccan ∪ Tunisian**, same rule; Maghrebi is a label, not a target.
- Targets: MSA, EGY, SAU, GLF, LEV, IRQ (see `TARGETS.md`). Omani is a Gulf-east member.
- Labeler v10 evidence rules (exact MADAR city forms first, corpus rates, Egyptian هـ-future, سيتم as a hard MSA
  marker) are described in `MIX_CARD.md`; the guideline's questions and register decisions are unchanged.

---

## Guideline v2.0 (2026-10-14), unchanged

Supersedes V1 v1.2 for V2. V1 decisions for Egyptian stay valid (Yahya's 42 register answers).

## 1. The question
For every word and every target T in {MSA, EGY, SAU, KHL, LEV, IRQ, YEM, MOR, TUN}:
> Would a native speaker of T, speaking as a customer-service / voice agent, naturally use this word, in this form,
> in this sentence?
Answer per target: **1** natural · **0** not natural · **?** unknown (no reliable evidence; never guessed).
A word **deviates** for T iff its T-bit is 0. One annotation serves all nine targets.

## 2. Word labels shown to the user
When a word deviates for T, the output label is the variety it belongs to: one of the nine targets
(preference: MSA first, then the closest region), **"Arabic / other dialect"** (Omani, Sudanese, Libyan, Algerian,
Hassaniya, or any dialect outside the ten), or "unknown". Non-Arabic tokens (names, brands, digits, Latin script,
emoji, URLs) are **other** and never deviate.

## 3. Register (service register, all targets)
Target = the colloquial variety as an agent speaks it. Following V1's validated Egyptian decision, **formal
open-class vocabulary is shared** (الفاتورة، التحويل، فوراً، نعتذر). For MSA function words, each target has a
status in `data/lexicons_v2/markers.tsv` (built from native text, `scripts/build_markers.py`):
- **hard** — grammatical MSA markers the target never uses: future (سوف، سـ + imperfect), verbal negation
  (لن، لم، ليس، ليست), relatives (الذي، التي، الذين), demonstratives where the target does not use them. Deviate.
  Differences per target are real: هذا is natural in SAU, IRQ, YEM, TUN (and mixed in KHL), not in EGY / LEV / MOR;
  الذي is used in Yemeni.
- **frm** — other MSA function words the target avoids in casual speech (هل، نحن، الآن، فقط، يمكنك، لدينا، يوجد،
  ماذا، لماذا…). **Natural in the service register** (bit 1 + frm flag); a runtime "strict colloquial" policy can
  flag them without retraining.
- **shared** — used by the target (e.g. كيف in SAU / LEV / YEM, لو everywhere in the East).
- **mixed** — evidence in between, or a spelling variant of the target's own word (Gulf هذا / هذه for hādha /
  hādhi): unknown, decided by context.

## 4. Evidence order (the labeler and the annotator use the same)
1. Not Arabic / name / brand → other.
2. **MADAR concept matrix** (expert, 1,042 concepts × 25 cities): a listed form of concept C for a city of T → 1;
   T's cities list only other forms for C and never this form (in any concept) → 0.
3. Markers file (§3) and the V1 core lexicon re-mapped to the nine targets.
4. Expert corpora with lemma alignment (Curras, Baladi, Nâbra), Lingualism lists.
5. Crowd lexicons (mo3jam, Lahga) only with net votes ≥ 3 or two independent sources.
6. Corpus frequency per target (verified hosts) as tie-breaker.
Contradictions → unknown and logged in `annotation/ambiguity_log_v2.tsv`.

## 5. Spelling and morphology
Normalize أإآ→ا، ى→ي، ة→ه, strip diacritics and tatweel. Clitics follow the stem's verdict, except dialect clitics
that are themselves markers (هـ/حـ future = EGY, بـ + imperfect = EGY / LEV / GLF, راح / رح future = LEV / GLF / IRQ,
ما…ش = EGY / MAG, كـ / تـ progressive = MOR, باش = MAG). Morphology flags only positive evidence; a word is never
labelled by what it lacks.

## 6. Gold specifics
- natural sets: native sentences; only words with explicit evidence are marked deviating (true leaks happen,
  e.g. MSA hard markers in Yemeni or Iraqi travel sentences); everything else conforms.
- challenge sets: deviation positions come from the substitution itself; labels from the source variety.
- agent sets: Claude labels with evidence, native verifies; disagreements resolved by a second native or marked
  ambiguous and excluded from the headline metric (target ≤ 3% of tokens).
