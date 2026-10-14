"""Spike 2: how well do CAMeL morphological analyzers separate EGY / GLF / MSA words?

Uses MADAR parallel sentences (Cairo, Doha, MSA). Word-level pseudo labels from set
differences within each parallel triple:
  EGY-specific = in the Cairo sentence but in neither the Doha nor the MSA sentence
  GLF-specific = in Doha but not Cairo / MSA
  MSA-specific = in MSA but not Cairo / Doha
  SHARED       = in all three
Then: for each class, how often is the word analyzable by each DB, and how good is a
simple "signature" rule at predicting the class?
"""
import json, re, sys
from collections import Counter, defaultdict
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.normalize import normalize_alef_ar, normalize_alef_maksura_ar, normalize_teh_marbuta_ar
from camel_tools.utils.dediac import dediac_ar

PUNCT = re.compile(r"[^\w؀-ۿ]+")


def norm(w):
    w = dediac_ar(w).replace("ـ", "")
    w = normalize_alef_ar(normalize_alef_maksura_ar(normalize_teh_marbuta_ar(w)))
    return w


def toks(s):
    return [t for t in (PUNCT.sub("", x) for x in s.split()) if t and not t.isdigit()]


rows = [json.loads(l) for l in open("/work/madar_merged.jsonl")]
items = []  # (word, cls)
for r in rows:
    c, d, m = toks(r["Cairo"]), toks(r["Doha"]), toks(r["MSA"])
    C, D, M = {norm(x) for x in c}, {norm(x) for x in d}, {norm(x) for x in m}
    for w in c:
        n = norm(w)
        if n in D and n in M: items.append((w, "SHARED"))
        elif n not in D and n not in M: items.append((w, "EGY"))
    for w in d:
        n = norm(w)
        if n not in C and n not in M: items.append((w, "GLF"))
    for w in m:
        n = norm(w)
        if n not in C and n not in D: items.append((w, "MSA"))

print("pseudo-labeled tokens:", Counter(c for _, c in items))

dbs = {k: Analyzer(MorphologyDB.builtin_db(v), backoff="NONE", cache_size=200000)
       for k, v in {"msa": "calima-msa-r13", "egy": "calima-egy-r13", "glf": "calima-glf-01"}.items()}

cov = defaultdict(Counter)
sig_by_cls = defaultdict(Counter)
for w, cls in items:
    sig = tuple(k for k in ("msa", "egy", "glf") if dbs[k].analyze(w))
    for k in sig: cov[cls][k] += 1
    cov[cls]["_n"] += 1
    sig_by_cls[cls][sig] += 1

print("\n% analyzable per class x DB")
for cls in ("SHARED", "EGY", "GLF", "MSA"):
    n = cov[cls]["_n"]
    print(f"{cls:7s} n={n:6d} " + "  ".join(f"{k}={100*cov[cls][k]/n:5.1f}" for k in ("msa", "egy", "glf")))

print("\nTop analyzer signatures per class")
for cls in ("SHARED", "EGY", "GLF", "MSA"):
    n = cov[cls]["_n"]
    top = ", ".join(f"{'+'.join(s) or 'none'}:{100*c/n:.0f}%" for s, c in sig_by_cls[cls].most_common(5))
    print(f"{cls:7s} {top}")

# Simple rule baseline for target = EGY: deviating if NOT analyzable by egy DB
tp = fp = fn = tn = 0
for w, cls in items:
    dev_true = cls in ("GLF", "MSA")
    dev_pred = not dbs["egy"].analyze(w)
    tp += dev_true and dev_pred; fp += (not dev_true) and dev_pred
    fn += dev_true and (not dev_pred); tn += (not dev_true) and (not dev_pred)
p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
print(f"\nRule 'deviates if not analyzable in EGY db' (target EGY): P={p:.3f} R={r:.3f} F1={2*p*r/max(1e-9,p+r):.3f}")
