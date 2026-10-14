# HaSeen — Annotation Guideline v1.2

Status: v1.0 (2026-10-12). Every change is versioned at the bottom (changelog) and every new hard case goes to `annotation/ambiguity_log.tsv`.
Applies to: gold set, calibration set, anchors, teacher labels (D3), and the definition the model is trained toward.

---

## 0. Target register (v1.2, validated by a native Egyptian speaker)
The Egyptian target is **Egyptian customer-service / voice-agent Arabic**, a semi-formal register. A native judge (Yahya, 42 yes/no checks) accepted as natural: نحن، هل، تم، إذا، لكن، فقط، الآن، حالياً، جداً، نعتذر، برجاء، يمكن، أهلاً بك، تؤكدلي، على دراية، بأسرع ما يمكن، عبر، تزودنا، احذر، يا عزيزي، للغاية، فوراً، formal service nouns، موعد، هاتفك، معنا، أخبرك، يمكنني/يمكنك، أستطيع، كيف، متى، لا + verb (لا تقلق)، ذلك، القادمة، اليوم، ordinal hours، tanween adverbs (مسبقاً، أولاً)، تخبرنا/توفرلي، أعطيني/ترسلها، يُفضل/تتعلق.
For the Egyptian target a word **deviates** only if it is:
1. a **hard MSA grammatical marker**: سوف، سـ + imperfect (سنقوم، سأرسل، سيتم)، لن، لم، ليس، الذي/التي/الذين، هذا/هذه/هؤلاء (Tuner's own example marks سوف);
2. a **pronunciation-changing MSA spelling** where Egyptian has a different form: ث for ت (أكثر vs أكتر، أبعثلك vs أبعتلك، الثاني vs التاني);
3. a word of **another dialect** (Gulf, Levantine, Maghrebi…).
Everything else that a native agent would say is `SHARED` or `EGY`.

## 1. The question we answer
For each word in a sentence and a **target variety** (Egyptian, MSA or Gulf):

> **Would a fluent native speaker of the target naturally use this word, in this form, in this sentence?**
> Yes → the word **conforms**. No → it **deviates**, and we say which variety it belongs to.

We label each word with its **variety**; conform/deviate is derived from the variety and the target (§5). This lets one annotation serve every target.

## 2. Labels (one per word)
| Label | Meaning |
|---|---|
| `EGY` | Used in Egyptian Arabic and **not** natural in the other two varieties (عايز، مش، ده، دلوقتي، إزاي، هتأخر) |
| `GLF` | Gulf Arabic (Saudi/Gulf), not natural in Egyptian or MSA (أبغى، وش، الحين، وايد، زين = okay، تبي) |
| `MSA` | Modern Standard Arabic, not natural in spoken Egyptian / Gulf (سوف، لن، الذي، ليس، هذا، أيضاً) |
| `SHARED` | Natural in the target **and** in at least one other variety (أنا، الطلب، خلال، إن شاء الله، تمام) |
| `OTHER` | Not dialect evidence: names, brands, numbers, Latin-script words, emojis, URLs, punctuation-only |

Optional flags per word: `ambiguous=1` (annotator cannot decide, §8), `DA_OTHER` in notes for Levantine/Maghrebi/Iraqi words (§7.6).

## 3. Decision procedure (apply in this order, stop at the first rule that fires)
1. **Not Arabic-script, a number, a proper name, a brand, an emoji or a URL?** → `OTHER`.
2. **Positive marker of a variety?** The word *contains* a form listed in §6 (whole word or clitic/prefix) → that variety.
   - Morphology rule: a word that merely **lacks** the target's marker is not marked by its absence (أتأخر without هـ → not MSA; see §7.2).
3. **Natural in the target?** Decide by native intuition. If unsure, use the **corpus validity test** (§4).
   - Natural in the target and also in another variety → `SHARED`.
   - Natural only in the target → the target's label (`EGY` when the target is Egyptian).
4. **Not natural in the target** → the variety where it is natural (`GLF` / `MSA`). If none of the three → note `DA_OTHER` or mark `ambiguous`.

## 4. Corpus validity test (tie-breaker, not a replacement for intuition)
For a word `w` and variety `v`: `r_v(w) = p_v(w) / max_u p_u(w)` over large marker-filtered corpora (FineWeb2 arz / ars / MSA + Gulf tweets + egyptian-2-arabic). `w` is **valid in v** if `count_v(w) ≥ 3` and `r_v(w) ≥ 0.25`.
- Example: خطأ r_EGY = 0.40 → valid in Egyptian → `SHARED`. سوف r_EGY = 0.05 → not valid → `MSA`.
- The test is only a tie-breaker for formal-but-common words. Native intuition wins when it is confident; when intuition and the test disagree, log the case in the ambiguity log and decide with the committee (§9).

## 5. From variety to conform / deviate
| Target | Conforms | Deviates (label in output) |
|---|---|---|
| Egyptian | `EGY`, `SHARED`, `OTHER` | `GLF` → "Arabic / Gulf", `MSA` → "Arabic / MSA", DA_OTHER → "unknown" |
| MSA | `MSA`, `SHARED`, `OTHER` | `EGY` → "Arabic / Egyptian", `GLF` → "Arabic / Gulf" |
| Gulf | `GLF`, `SHARED`, `OTHER` | `EGY`, `MSA` |

## 6. Marker lists (positive evidence)
Normalized forms (أ/إ/آ → ا، ى → ي، ة → ه) are matched; clitics (و، ف، ب، ل، ك، ال) may be attached.

**MSA markers**
- Future: سوف، سـ + imperfect verb (سأرسل، سنقوم، سيتم، ستصلك)
- Negation of verbs: لن، لم، ليس/ليست/لست، لا + imperfect verb (لا يمكن، لا نستطيع)
- Relatives: الذي، التي، الذين، اللذان، اللتان، اللواتي
- Demonstratives: هذا، هذه، ذلك، تلك، هؤلاء، أولئك
- Others: أيضاً، إذ، حيث (as conjunction)، كي، لكي، ثمة، إنما، بيد أن، لدى/لدينا، يتم/تم (passive auxiliary)، نحن، أنتم، ماذا، لماذا، كيف (in speech Egyptian uses إزاي)، أين، متى (Egyptian: إمتى)، هل (question particle)
- Case endings written as tanween in running speech (يوماً سعيداً) → MSA

**EGY markers**
- Future هـ/حـ + verb (هتوصلك، حيترجع)، progressive بـ + verb (بيتواصل) when the verb itself is Egyptian or neutral
- Negation مش، ما…ش (ماوصلش)، مفيش
- Demonstratives ده، دي، دول؛ question words إيه، إزاي، فين، إمتى، ليه
- Lexical: عايز/عاوز، دلوقتي، النهارده، بكرة، كده، أوي/قوي، بتاع/بتاعة/بتوع، لسه، عشان/علشان (shared with Gulf → see §7.4)، برضه، خالص، معلش، حضرتك، يا فندم، تاني (= again/other)، إحنا

**GLF markers**
- Want: أبغى/أبي/تبي/يبي/نبي/أبا
- Question: وش، ايش، شلون، ليش، متى (shared with MSA → see §7.4)
- Time/degree: الحين، ذحين، وايد/واجد
- Discourse: زين (= okay)، مب/مو (negation)، حق (= for/belonging)، هذي/ذي، هالـ (هالحين، هالشي)
- Future/intent: راح + verb، بـ + verb with Gulf verb (بسوي، بعطيك)
- Lexical: سوى/يسوي (= do)، عطني، شفيك، خلاص (shared)

## 7. Hard cases (decisions)
### 7.1 Formal service vocabulary
Words common in Egyptian customer-service speech even if they look formal (تحويل، تسجيل، الخاصية، الشكوى، فوراً، نعتذر، إلغاء، المختص، متاح) → `SHARED`. Use §4 when unsure. A formal **construction** is still MSA: "سنقوم بتحويلك" → سنقوم = `MSA` (marker سـ)، بتحويلك = `SHARED`.

### 7.2 Missing target marker
"أتأخر" (no هـ), "أقوم بتحويلك" (no هـ) → `SHARED`. We never label by absence. Tuner's own example confirms this.

### 7.3 لا
- "لا" alone = "no" → `SHARED`.
- "لا" + imperfect verb as negation (لا يمكننا، لا نستطيع) → `MSA` (Egyptian: مش هنقدر / ما نقدرش).
- "لا" in fixed phrases (لا شكر على واجب) → `SHARED`.

### 7.4 Words shared by two varieties
عشان (EGY + GLF), ليه/ليش differ (ليه EGY, ليش GLF), متى (MSA + GLF; Egyptian says إمتى) → if natural in the target → `SHARED`; if natural only in two non-target varieties → label the one that the sentence context points to; else `GLF` for متى when the target is Egyptian.

### 7.5 Borrowings
Arabic-script loanwords used in Egyptian speech (السيستم، الأبلكيشن، الموبايل، الباقة، جيجا) → `SHARED`. Latin-script words → `OTHER`.

### 7.6 Other dialects
Levantine (شو، هلق، بدي، عم + verb), Maghrebi, Iraqi words are none of our three varieties. In the gold set write them as variety `GLF` **plus** the note `DA_OTHER:LEV` (or `:MAG`, `:IRQ`). Evaluation treats every `DA_OTHER` word as a deviation **for any target, including Gulf** (detection counts), and excludes it from label accuracy, because the output label for it should be "unknown". كتير is natural in Egyptian → `SHARED`.

### 7.11 إذا، معنا and similar near-MSA forms
- إذا: Egyptian speech uses لو. Decide by §4 (corpus test); until measured, `MSA` with `ambiguous=1` when the rest of the sentence is fully colloquial, `SHARED` when the sentence is formal service register.
- معنا، معكم، لدينا → `MSA` (Egyptian: معانا، معاكو، عندنا). معاك/معايا → `EGY`.

### 7.7 Religious and fixed expressions
إن شاء الله، الحمد لله، السلام عليكم، بإذن الله، الله يخليك → `SHARED`.

### 7.8 Pronouns
أنا، هو، هي → `SHARED`. نحن، أنتم، هم (as in MSA) → `MSA` when target is Egyptian (Egyptian: إحنا، إنتو، هما). إنت/إنتي → `EGY`/`SHARED` (Gulf uses إنت too → `SHARED`).

### 7.9 Spelling
Normalize before deciding (§6). Misspelled Egyptian words are still `EGY`. Arabizi is out of scope (`OTHER`).

### 7.10 Names and numbers
Brand names (فودافون، أمازون), person names, cities, product codes, digits, written numbers used as quantities (خمسين، يومين) → names/digits `OTHER`, number words `SHARED`.

## 8. Ambiguity
If after §3–§7 the annotator is still unsure: set `ambiguous=1`, pick the most likely label, write a note. Ambiguous tokens are reported separately and excluded from the headline metric (target ≤ 3% of tokens).

## 9. Disagreement resolution (Claude-led)
1. Claude label + reason.
2. LLM committee (≥ 3 strong models via HF router, prompted with this guideline) votes per word.
3. Corpus validity test (§4).
4. If Claude, the committee majority and the corpus test agree → final. If not → one yes/no native-speaker question to Yahya ("هل المصري بيقول X في الجملة دي؟"). If still unresolved → `ambiguous=1`.
Every resolved case is added to the anchors or the hard-case list above in the next version.

## Changelog
- v1.2 (2026-10-12): §0 target register from Yahya's 42 yes/no answers. 32 gold tokens and 8 anchor tokens relabelled MSA → SHARED. Natural Egyptian gold becomes mainly a precision set (6 deviations / 277 sentences); deviation-rich Egyptian data comes from egy_mirror stress.
- v1.1 (2026-10-12): DA_OTHER words deviate for every target (fixes Gulf-target case found while labeling g-glf-0408). Gold MSA/GLF targets: only deviations marked; MSA/GLF vs SHARED collapsed. Corpus validity test is a tie-breaker only: written Egyptian (blogs) over-validates formal words (أكثر، قادمة، عبر) — spoken-register intuition wins, logged in ambiguity_log.tsv.
- v1.0 (2026-10-12): initial version from pilot error analysis (لا، إذا، معنا، تحدث، الخاصية، الشكوى، أقوم).
