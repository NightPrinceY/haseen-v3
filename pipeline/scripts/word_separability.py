"""Word-level separability between varieties (2026-10-14). Sentence-level ID is the wrong yardstick for HaSeen:
what matters is how many WORDS of variety A a target T would flag. Two controlled measures:

1. MADAR concept distance (expert, complete 25-city x 1,042-concept matrix): for countries A, T the share of
   concepts where A's forms and T's forms do not overlap at all (READY rows; cities pooled per country).
   Reported for all concepts and for closed-class concepts only (function words).
2. Alexandria deviation density (same pipeline, same domains): train counts per country; for each test token of
   country A, it is "T-absent" if it is seen >= 3 times in A-train and its expected count in T-train is >= 5 but
   it never occurs there. Density = share of A test tokens that are T-absent = how many words per 100 a T-target
   model could flag in an A speaker's text, using corpus evidence only.
Output: data/inventory/word_separability.json + printed matrices."""
import csv, gzip, json, os, sys, collections, itertools
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "HaSeen-V1", "src"))
from haseen.text import norm, words  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
CITY = {"RAB": "MA", "FES": "MA", "ALG": "DZ", "TUN": "TN", "SFA": "TN", "TRI": "LY", "BEN": "LY", "CAI": "EG",
        "ALX": "EG", "ASW": "EG", "KHA": "SD", "JER": "PS", "AMM": "JO", "SAL": "JO", "BEI": "LB", "DAM": "SY",
        "ALE": "SY", "MOS": "IQ", "BAG": "IQ", "BAS": "IQ", "DOH": "QA", "MUS": "OM", "RIY": "SA_RIY", "JED": "SA_JED",
        "SAN": "YE"}
out = {}

# ---- 1. MADAR concept distance
C = collections.defaultdict(lambda: collections.defaultdict(set))
cls = {}
msa = {}
for r in csv.DictReader(open(f"{ROOT}/sources/madar_lexicon/MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0.tsv"), delimiter="\t"):
    if r["Release Status"] != "READY":
        continue
    C[r["Concept_ID"]][CITY[r["Dialect"]]].add(norm(r["CODA"].strip()))
    cls[r["Concept_ID"]] = r["Class"]
    msa[r["Concept_ID"]] = {norm(x.strip()) for x in r["MSA"].replace("،", ",").split(",")}
for cid in C:
    C[cid]["MSA"] = msa[cid]
V = ["EG", "SD", "PS", "JO", "LB", "SY", "IQ", "SA_RIY", "SA_JED", "QA", "OM", "YE", "LY", "TN", "DZ", "MA", "MSA"]


def dist(subset):
    M = np.zeros((len(V), len(V)))
    for i, a in enumerate(V):
        for j, b in enumerate(V):
            cs = [c for c in subset if C[c].get(a) and C[c].get(b)]
            M[i, j] = sum(1 for c in cs if not (C[c][a] & C[c][b])) / max(1, len(cs))
    return M


for name, subset in [("all", list(C)), ("closed", [c for c in C if cls[c] == "ClosedClass"])]:
    M = dist(subset)
    out[f"madar_{name}"] = {"labels": V, "matrix": M.round(3).tolist(), "n_concepts": len(subset)}
    print(f"\n== MADAR concept distance ({name}, {len(subset)} concepts): share of concepts with NO shared form")
    print("      " + " ".join(f"{v[:6]:>6}" for v in V))
    for i, a in enumerate(V):
        print(f"{a[:6]:>6}" + " ".join(f"{M[i, j]:6.2f}" for j in range(len(V))))
    iu = np.triu_indices(len(V), 1)
    Z = linkage(M[iu], "average")
    out[f"madar_{name}"]["groups"] = {str(k): sorted(([V[i] for i in np.where(fcluster(Z, k, "maxclust") == t)[0]]
                                                      for t in set(fcluster(Z, k, "maxclust"))), key=len, reverse=True)
                                      for k in (3, 4, 5, 6, 8, 10, 12)}
    for k, g in out[f"madar_{name}"]["groups"].items():
        print(f"  k={k}: {g}")

# ---- 2. Alexandria deviation density
cnt = collections.defaultdict(collections.Counter)
test = collections.defaultdict(list)
for l in gzip.open(f"{ROOT}/inventory/sentences.jsonl.gz", "rt"):
    r = json.loads(l)
    if r["source"] != "alexandria":
        continue
    toks = [norm(w) for w in words(r["text"])]
    if r["split"] == "test":
        test[r["var"]].append(toks)
    else:
        cnt[r["var"]].update(toks)
A = sorted(v for v in cnt if sum(cnt[v].values()) > 20000)
tot = {v: sum(cnt[v].values()) for v in A}
D = np.zeros((len(A), len(A)))
examples = {}
for i, a in enumerate(A):
    toks = [t for s in test[a] for t in s]
    for j, t in enumerate(A):
        if a == t:
            continue
        absent = collections.Counter()
        n = 0
        for w in toks:
            ca = cnt[a][w]
            if ca >= 3 and cnt[t][w] == 0 and ca / tot[a] * tot[t] >= 5:
                n += 1
                absent[w] += 1
        D[i, j] = n / max(1, len(toks))
        examples[f"{a}->{t}"] = [w for w, _ in absent.most_common(12)]
out["alexandria_density"] = {"labels": A, "matrix": D.round(4).tolist(), "train_tokens": tot, "examples": examples}
print("\n== Alexandria deviation density: share of A-test tokens a T-target would flag (corpus evidence)")
print("rows = speaker A, cols = target T")
print("     " + " ".join(f"{v:>5}" for v in A))
for i, a in enumerate(A):
    print(f"{a:>5}" + " ".join(f"{D[i, j]:5.2f}" for j in range(len(A))))
for p in ["JO->PS", "PS->JO", "SY->LB", "LB->SY", "SA->OM", "EG->SA", "SA->EG", "MA->TN"]:
    if p in examples:
        print(p, examples[p])
json.dump(out, open(f"{ROOT}/inventory/word_separability.json", "w"), indent=1, ensure_ascii=False)
