"""Build D1-lite: word-labeled, dialect-mixed sentences from MADAR parallel Cairo / Doha / MSA.

Recipe (after AGS, arXiv 2508.17347):
1. Align Cairo->MSA and Doha->MSA with SimAlign (itermax) on MARBERTv2 embeddings. MSA is the anchor.
2. Variety of every word in its own sentence:
   SHARED if its normalized form also appears in one of the other two parallel sentences, else the
   sentence's own variety (EGY / GLF / MSA).
3. Mixing (keeps the matrix sentence's grammar): for a matrix sentence, replace 1-3 words with the
   word aligned to them in another variety. A replacement is kept only if the inserted word is never
   used anywhere in the matrix variety's side of the corpus (corpus-level vocabulary filter), so it is
   very likely a real deviation, and it gets that variety's label.
4. 30% of matrix sentences stay unmixed (no deviation), so the model learns "nothing to flag".
Matrices: Cairo (main), plus Doha and MSA (so one model serves several targets).

Output: {split}.jsonl with {"tokens": [...], "labels": [...], "matrix": "EGY", "n_dev": k}
"""
import json, random, sys, argparse
from collections import Counter
from datasets import load_dataset
from simalign import SentenceAligner
from common import norm, words

p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=0)
p.add_argument("--out", default="data_d1")
p.add_argument("--device", default="cpu")
p.add_argument("--seed", type=int, default=13)
p.add_argument("--lexstats", default="")
p.add_argument("--valid_r", type=float, default=0.25)
p.add_argument("--reject_r", type=float, default=0.10)
a = p.parse_args()
random.seed(a.seed)

ds = load_dataset("Elfsong/MADAR_Corpus_Merged", split="train")
rows = list(ds)
if a.limit:
    random.shuffle(rows)
    rows = rows[: a.limit]
SPLIT = {"corpus-6-train": "train", "corpus-6-dev": "dev", "corpus-6-test-corpus-26-train": "test"}
COL = {"EGY": "Cairo", "GLF": "Doha", "MSA": "MSA"}

LEX = json.load(open(a.lexstats))["words"] if a.lexstats else {}


def valid(nw, v, r):
    """corpus-level: is normalized word nw used in variety v (count >= 3 and relative freq >= r)?"""
    st = LEX.get(nw)
    return bool(st) and st[v][0] >= 3 and st[v][1] >= r


aligner = SentenceAligner(model="UBC-NLP/MARBERTv2", token_type="bpe", matching_methods="i", device=a.device)

# corpus-level normalized vocabularies per variety (train split only, to avoid leaking dev/test)
vocab = {v: set() for v in COL}
for r in rows:
    if SPLIT[r["split"]] == "train":
        for v, c in COL.items():
            vocab[v].update(norm(w) for w in words(r[c]))

stats = Counter()
out = {s: [] for s in SPLIT.values()}
align_samples = []

for k, r in enumerate(rows):
    W = {v: words(r[c]) for v, c in COL.items()}
    if any(len(x) == 0 for x in W.values()):
        continue
    N = {v: [norm(w) for w in W[v]] for v in COL}
    S = {v: set(N[v]) for v in COL}
    # alignment to MSA anchor: dict msa_index -> word index in EGY / GLF
    to_msa = {}
    for v in ("EGY", "GLF"):
        al = aligner.get_word_aligns(W[v], W["MSA"])["itermax"]
        to_msa[v] = {i: j for i, j in al}
    if len(align_samples) < 40:
        align_samples.append({"EGY": W["EGY"], "MSA": W["MSA"], "GLF": W["GLF"],
                              "EGY->MSA": sorted(to_msa["EGY"].items()), "GLF->MSA": sorted(to_msa["GLF"].items())})
    # word-level counterparts through the MSA anchor
    msa_from = {v: {} for v in ("EGY", "GLF")}
    for v in ("EGY", "GLF"):
        for i, j in to_msa[v].items():
            msa_from[v].setdefault(j, i)

    def own_label(v, i):
        others = [o for o in COL if o != v]
        if any(N[v][i] in S[o] for o in others):
            return "SHARED"
        if LEX and any(valid(N[v][i], o, a.valid_r) for o in others):
            return "SHARED"
        return v

    def counterpart(src_v, i, dst_v):
        """word in dst_v aligned (through MSA) to word i of src_v"""
        j = i if src_v == "MSA" else to_msa[src_v].get(i)
        if j is None:
            return None
        if dst_v == "MSA":
            return W["MSA"][j]
        ii = msa_from[dst_v].get(j)
        return None if ii is None else W[dst_v][ii]

    split = SPLIT[r["split"]]
    for matrix, n_copies in (("EGY", 2), ("GLF", 1), ("MSA", 1)):
        for _ in range(n_copies):
            toks = list(W[matrix])
            labels = [own_label(matrix, i) for i in range(len(toks))]
            n_dev = 0
            if random.random() > 0.30:
                cand = list(range(len(toks)))
                random.shuffle(cand)
                want = random.choice([1, 1, 1, 2, 2, 3])
                for i in cand:
                    if n_dev >= want:
                        break
                    srcs = [v for v in COL if v != matrix]
                    random.shuffle(srcs)
                    for src in srcs:
                        w2 = counterpart(matrix, i, src)
                        if not w2:
                            continue
                        n2 = norm(w2)
                        if n2 == N[matrix][i] or n2 in vocab[matrix] or n2 in S[matrix] or (LEX and valid(n2, matrix, a.reject_r)):
                            stats[f"reject_{matrix}<-{src}"] += 1
                            continue
                        toks[i] = w2
                        labels[i] = src
                        n_dev += 1
                        stats[f"insert_{matrix}<-{src}"] += 1
                        break
            out[split].append({"tokens": toks, "labels": labels, "matrix": matrix, "n_dev": n_dev})
            stats[f"{split}_sents"] += 1
    if k % 500 == 0:
        print(k, dict(stats), flush=True)

import os
os.makedirs(a.out, exist_ok=True)
for s, xs in out.items():
    with open(f"{a.out}/{s}.jsonl", "w") as f:
        for x in xs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
lab = Counter(l for xs in out.values() for x in xs for l in x["labels"])
json.dump({"stats": dict(stats), "label_counts": dict(lab)}, open(f"{a.out}/stats.json", "w"), ensure_ascii=False, indent=1)
json.dump(align_samples, open(f"{a.out}/align_samples.json", "w"), ensure_ascii=False, indent=1)
print("labels:", dict(lab))
print("done:", {s: len(x) for s, x in out.items()})
