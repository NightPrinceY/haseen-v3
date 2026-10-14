# HaSeen gold v1 (frozen 2026-10-12) — evaluation only, never trained or tuned on

| File | What | Measures |
|---|---|---|
| egy_natural.jsonl | Egyptian replies from 5 LLM families (Set A: Gemma-4-31B, Llama-3.3-70B, DeepSeek-V4-Flash, Aya-Expanse-32B, Llama-4-Scout), 4 prompt styles | **precision** on natural agent output (few deviations: LLMs rarely break the Egyptian agent register) |
| egy_stress.jsonl | Egyptian replies with MSA knowledge-base context (egy_mirror) | realistic MSA leakage (سيتم، سيقوم، لن، هذا) |
| egy_challenge.jsonl | verified-clean egy_natural hosts with 1-2 controlled Gulf / hard-MSA substitutions (grammar-checked) | **recall** on known deviation types; reported separately |
| msa.jsonl | MSA-target replies + msa_mirror / msa_rewrite stress | Egyptian leakage into MSA |
| glf_experimental.jsonl | Saudi-target replies | experimental: no native Gulf annotator |

Labels: per-token variety (EGY/GLF/MSA/SHARED/OTHER), deviates (0/1 vs the row's target), ambiguous flag.
Annotation: Claude labels following docs/GUIDELINE.md v1.2; target register validated by a native Egyptian speaker (42 yes/no checks); LLM second opinion (DeepSeek-V4-Pro) recorded in annotation/raw (raw agreement 97.7%, κ 0.21 — it over-flags formal service vocabulary).
Removed: 38 corrupted generations (CJK/Hangul/Cyrillic/Thai/Latin inside Arabic, incoherent text).
Integrity: sha256 per file in MANIFEST.json; any change → new version.
