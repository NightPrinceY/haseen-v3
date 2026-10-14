# HaSeen V2 — full plan (written 2026-10-14, gold first)

Scope: 9 production targets (MSA, EGY, SAU, KHL, LEV, IRQ, YEM, MOR, TUN); Omani, Sudanese, Libyan,
Algerian and Hassaniya are recognised as "other dialect". Spec: `research/TARGETS_V2.md`.
Rule from V1: data is the lever, and gold decides everything. Gold is built and frozen **before** any training.

## City → target map (MADAR cities)
EGY: Cairo, Alexandria, Aswan · SAU: Riyadh, Jeddah · KHL: Doha · LEV: Jerusalem, Amman, Salt, Beirut, Damascus, Aleppo · IRQ: Baghdad, Basra, Mosul · YEM: Sana'a · MOR: Rabat, Fes · TUN: Tunis, Sfax ·
other dialect: Khartoum, Tripoli, Benghazi, Algiers, Muscat.

## Phase G — Gold: HaSeen-Bench v2 (frozen before training)
| Step | What | Evidence / who | Output |
|---|---|---|---|
| G0 | Guideline v2: one question per word and target ("would a native speaker of T say this word here?"); per-target **hard-MSA marker lists derived from MADAR closed-class concepts** (e.g. هذا is natural in Gulf, Iraqi, Yemeni, Tunisian but not in Egyptian or Levantine); register = service register (formal open-class words are shared) | MADAR Lexicon closed class (167 concepts × 25 cities), V1 guideline | `docs/GUIDELINE_V2.md`, `data/lexicons_v2/markers_<T>.tsv` |
| G1 | **natural** set per target, 300 sentences from held-out verified sources never used in training (MADAR-26 test, Alexandria test, Casablanca, Habibi, Lisan / Nâbra held-out) | source = native text; flagged words reviewed with evidence; native spot-check | `data/gold_v2/natural_<T>.jsonl` |
| G2 | **challenge** set per target, 300 sentences: MADAR-26 test sentence of T with 1–3 words replaced by the **lexicon-anchored aligned word** from another variety's version of the same sentence (same concept in MADAR Lexicon), plus MSA swaps from the MSA version | expert parallel corpus + expert lexicon; exact positions; no annotator | `data/gold_v2/challenge_<T>.jsonl` |
| G3 | **cross** set: whole sentences of variety A judged against target B (MADAR-26 test, all pairs) — density sanity, multi-label check | parallel corpus | `data/gold_v2/cross.jsonl` |
| G4 | **agent** set per target, 150 replies "in dialect T" from several open-weight LLMs on an HF sandbox (the real failure mode), labelled by Claude with evidence, verified by natives (Yahya for EGY; one checker per other target) | sandbox ≈ $1 (needs Yahya's OK) | `data/gold_v2/agent_<T>.jsonl` |
| G5 | Freeze: sha256 manifest, data card, leakage check (8-gram) against every training source | — | `data/gold_v2/MANIFEST.json` |
Ship gate per target (pre-registered): natural false alarms ≤ 1% of words · challenge F1 ≥ 0.90 · agent F1 ≥ 0.85.

## Phase L — Labeler v2 (9 bits)
Per word: 9 naturalness bits (1 / 0 / unknown), `other_dialect`, `other`, `frm`. Evidence order:
1. not Arabic → other; names / brands gazetteers.
2. **MADAR concept matrix**: the form is a listed form of concept C for cities of T → bit 1; T's cities list only
   other forms for C (and the form is not listed for T) → bit 0 (sense-level negative evidence).
3. Hard-MSA markers per target (G0 lists) and V1 core lexicon (EGY / GLF / MSA, re-mapped).
4. Birzeit lemma alignments, Lingualism, crowd lexicons with votes ≥ 3 or 2 independent sources.
5. Corpus frequency per target from verified hosts (and FineWeb-2 later) as tie-breaker only.
Never guess: no evidence → unknown (masked). Audited on gold **dev** parts only, never on test.

## Phase D — Training data v7
Hosts: verified text per target first (MADAR train splits, Alexandria train, Curras / Baladi / Nâbra, Lisan, STS,
Dial2MSA, Nile-Chat test), filtered web second (Iraqi factlogic, Algerian ASR, mo3jam examples), LLM text last
(Levanti synthetic, oasst2) — train only. Families as V1: A natural, B aligned swaps (MADAR-26 train + Corpus-6 +
Dial2MSA + STS), C lexicon substitution (MADAR concept sets, all 25 cities), D per-target marker transforms,
E service templates per target. Splits by source document; 8-gram leakage check against gold.

## Phase M — Model
MARBERTv2-family encoder; heads: pointer (10 options + learned per-option vector, V1 fix) and linear control;
masked BCE; temperature + per-target thresholds from dev. HF L4 sandbox, 3 seeds, pre-registered expectations.
ONNX int8 CPU target ≤ 15 ms p50 per turn.

## Phase P — Product (Tuner)
`detect(text, target)` with 9 targets → per word conforms / deviates + label (variety, MSA, other dialect, other);
per-turn adherence score; caller-dialect estimate; CLI / server / Docker as V1.1; Tuner mapping: eval
"Dialect adherence" (PASS / FAIL or 1–5), red flags, simulation check. Release HaSeen-2 privately.

## Order of work
G0 → G2 (no annotators) → G1 → G3 → G5 freeze of these parts → L → D → M → G4 (agent set; needs sandbox OK and
natives) → final eval → P.
