"""Corpus-level word validity per variety (fixes D1's sentence-level SHARED definition).

For each normalized word w and variety v we estimate p_v(w) from large monolingual corpora, then
  r_v(w) = p_v(w) / max_u p_u(w)
A word is "valid in v" when count_v(w) >= MIN_COUNT and r_v(w) >= VALID_R.
Dialect corpora are filtered to sentences carrying strong dialect markers, to reduce MSA contamination
(Egyptian blogs and Gulf forums contain MSA passages).

Sources (production-safe licenses): FineWeb2 arz / ars / MSA (ODC-BY), Gulf tweets (CC-BY-4.0),
Egyptian side of egyptian-2-arabic (MIT).
"""
import json, itertools, re, argparse
from collections import Counter
from datasets import load_dataset
from common import norm, words

EGY_MARK = {norm(w) for w in "مش ده دي دول عايز عايزة عاوز كده إزاي ازاي ليه فين امتى إمتى دلوقتي النهارده بتاع بتاعت اللي عشان علشان خالص قوي أوي اوي برضه بقى معلش ازيك".split()}
GLF_MARK = {norm(w) for w in "وش ابغى أبغى ابي أبي تبي الحين وايد شلون ليش زين هالحين عشان حق يبي ودي مب مو اشوف شفيك وشلون هذي ذي".split()}
MSA_MARK = {norm(w) for w in "سوف لن لم الذي التي الذين ليس هذا هذه إن أن قد لقد إذ حيث كي".split()}

p = argparse.ArgumentParser()
p.add_argument("--docs", type=int, default=120000)
p.add_argument("--out", default="lexstats.json")
a = p.parse_args()


def sentences(text):
    for s in re.split(r"[.!?؟\n]+", text):
        ws = words(s)
        if len(ws) >= 4:
            yield [norm(w) for w in ws]


def collect(name, markers, limit, field="text", split="train"):
    c, n_s = Counter(), 0
    ds = load_dataset(name, split=split, streaming=True)
    for r in itertools.islice(ds, limit):
        for ws in sentences(r[field] or ""):
            if markers is None or (len(markers.intersection(ws)) >= 1 and not (MSA_MARK.intersection(ws) and markers is not MSA_MARK and len(markers.intersection(ws)) < 2)):
                c.update(ws); n_s += 1
    print(name, "sentences kept", n_s, "tokens", sum(c.values()), flush=True)
    return c


EGY = collect("Omartificial-Intelligence-Space/FineWeb2-Egyptian-Arabic", EGY_MARK, a.docs)
EGY.update(collect("AdhamAshraf/egyptian-2-arabic", None, 100000, field="egyptian_arabic"))
GLF = collect("Omartificial-Intelligence-Space/FineWeb2-Najdi-Arabic", GLF_MARK, a.docs)
GLF.update(collect("AhmedSSabir/Gulf-Arabic-Tweets-2018-2020", GLF_MARK, 400000))
MSA = collect("Omartificial-Intelligence-Space/FineWeb2-MSA", None, a.docs // 2)

C = {"EGY": EGY, "GLF": GLF, "MSA": MSA}
tot = {v: sum(c.values()) for v, c in C.items()}
vocab = set().union(*[{w for w, n in c.items() if n >= 3} for c in C.values()])
stats = {}
for w in vocab:
    p_ = {v: (C[v][w] + 0.5) / tot[v] for v in C}
    m = max(p_.values())
    stats[w] = {v: [C[v][w], round(p_[v] / m, 4)] for v in C}
json.dump({"totals": tot, "words": stats}, open(a.out, "w"), ensure_ascii=False)
print("vocab", len(stats), "totals", tot)
for w in ["سوف", "لن", "الذي", "خطا", "فورا", "تسجيل", "الدفع", "ابغي", "وايد", "الحين", "عايز", "مش", "اقوم", "الله", "شاء", "تمام", "حاضر"]:
    print(w, stats.get(norm(w)))
