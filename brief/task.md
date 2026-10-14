# Take-Home Task: Dialect Deviation Detection

## Role: AI/ML Engineer

## Overview

Build a system that takes a text and a target dialect, then flags and
extracts every word that does not belong to that dialect. Your system
**MUST** **NOT** use any cloud based LLM provider at inference time.
Assume the system will be hosted in the Air Gapped Environment at
deployment time.

We expect this to take roughly 10--16 hours of focused work. You have
**7 calendar days** from receipt to submit. We value a working,
well-reasoned solution over an exhaustive one. The submission should
prove your design concept then followed with a documented plan on how to
convert the concept to a production ready system.

## Problem statement

LLMs often mixes dialects of the same language: a Gulf word inside
Egyptian text, or formal MSA slipping into colloquial speech. Your
system should detect these deviations at the word level.

**Inputs**

- text: a string of free-form text.

- target: the expected dialect, e.g. Arabic / Egyptian, Arabic / Gulf,
  Arabic / MSA.

**Outputs**

- Every word (token) in the input labeled as either conforming to the
  target or deviating from it.

- For each deviating word: the word, its position in the text, and a
  label describing what it is instead (the dialect it belongs to, MSA,
  or unknown).

- An extracted list of all deviating words.

## Requirements

1.  Accept a text input and a target dialect as parameters.

2.  Tokenize the text into words

3.  For simplicity you can ignore handling punctuation, numbers and
    spelling variation sensibly.

4.  Classify each word as conforming or deviating relative to the target
    dialect.

5.  Return the results as structured JSON (schema below) and expose them
    through at least one interface: a CLI, a Python function.

6.  Support at least **one dialect** of one language as targets (e.g.
    Egyptian, Saudi with MSA recognized as a deviation label.

7.  Include a small evaluation: a labeled test set (yours or public) and
    the metrics you measured, such as precision, recall and F1 on
    deviating words.

## Nice to Have

1.  Label each deviating word with the dialect it most likely belongs to
    (e.g. Arabic / Gulf, Arabic / MSA, unknown).

2.  Achieve reasonable latency and cost at the inference time.

## Note:

Treat named entities, numbers and punctuation however you think is
right, but state your choice in the README.

## Example

Illustrative only; you may refine the schema if you explain why.

**Input**

{\
\"text\": \"[أنا أبغى أروح الاجتماع بكرة بس سوف أتأخر
شوية]{dir="rtl"}\",\
\"target\": \"Arabic / Egyptian\"\
}

**Output**

{\
\"target\": \"Arabic / Egyptian\",\
\"tokens\": \[\
{\"word\": \"[أنا]{dir="rtl"}\", \"index\": 0, \"conforms\": true},\
{\"word\": \"[أبغى]{dir="rtl"}\", \"index\": 1, \"conforms\": false,
\"label\": \"Arabic / Gulf\", \"confidence\": 0.92},\
{\"word\": \"[أروح]{dir="rtl"}\", \"index\": 2, \"conforms\": true},\
{\"word\": \"[الاجتماع]{dir="rtl"}\", \"index\": 3, \"conforms\":
true},\
{\"word\": \"[بكرة]{dir="rtl"}\", \"index\": 4, \"conforms\": true},\
{\"word\": \"[بس]{dir="rtl"}\", \"index\": 5, \"conforms\": true},\
{\"word\": \"[سوف]{dir="rtl"}\", \"index\": 6, \"conforms\": false,
\"label\": \"Arabic / MSA\", \"confidence\": 0.87},\
{\"word\": \"[أتأخر]{dir="rtl"}\", \"index\": 7, \"conforms\": true},\
{\"word\": \"[شوية]{dir="rtl"}\", \"index\": 8, \"conforms\": true}\
\],\
\"deviations\": \[\"[أبغى\", \"سوف]{dir="rtl"}\"\]\
}

Here [أبغى]{dir="rtl"} is Gulf Arabic (an Egyptian speaker would say
[عايز]{dir="rtl"}), and [سوف]{dir="rtl"} is Modern Standard Arabic
(Egyptian would use the [هـ]{dir="rtl"} prefix: [هتأخر]{dir="rtl"}).

## Deliverables

Submit a **GitHub repository** (public, or private with access granted
to us) containing:

- [ ] Source code for the system and the inference pipeline.

- [ ] Any training or data-preparation scripts, or clear instructions
  for obtaining the data.

- [ ] A dependency file (requirements.txt, pyproject.toml or similar),
  or a Dockerfile.

- [ ] Your evaluation set and evaluation script.

- [ ] A clear **README.md**.

The README must cover:

- [ ] Setup: prerequisites and installation steps.

- [ ] How to run it: exact commands, with at least one copy-paste
  example and its expected output.

- [ ] Supported languages and dialects.

- [ ] Evaluation results and how to reproduce them.

- [ ] Limitations, known failure cases, and what you would do with more
  time.

We should be able to clone the repo and run it by following the README
alone.

## Tools and resources

You may use any existing open-source tools, libraries, models and
datasets, such as Hugging Face models, fastText language ID, spaCy, or
public dialect corpora. Fine-tuning, rules, heuristics, LLM prompting or
a hybrid are all acceptable. Credit what you use in the README.

**API keys and compute credits:** if your approach needs a paid AI model
API key or compute credits, **reach out to us before you start** and we
will provide them.

## How we will evaluate

  -----------------------------------------------------------------------
  Area                                What we look for
  ----------------------------------- -----------------------------------
  Architecture                        The quality of the Architecture
                                      used

  Productisation                      Same Solution can be used in
                                      production if more resource
                                      available

  ML judgment                         A sensible model choice, sound
                                      evaluation, and honest discussion
                                      of trade-offs.

  Code quality                        Readable, modular, tested where it
                                      matters.

  Reproducibility                     Runs cleanly from the README on a
                                      fresh machine.

  Communication                       A clear README that explains
                                      decisions and limitations.
  -----------------------------------------------------------------------

We may ask you to walk through your solution in a follow-up interview.

## Submission and questions

Reply to the email that sent you this task with your repository link. If
anything is unclear contact us.
