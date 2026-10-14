"""G5: freeze HaSeen-Bench v2 (natural + challenge parts) -> data/gold_v2/frozen/ (2026-10-14).

1. Applies the documented fixes in annotation/gold_v2_fixes.tsv (each with a reason).
2. Adds an `ignore` mask: words whose status for the target is "mixed" in markers.tsv (also behind a proclitic,
   e.g. لهذه for SAU) are excluded from scoring, unless they are a gold deviation. Guideline v2 §3.
3. Writes MANIFEST.json (sha256 per file, counts) and gold_8grams.txt for leakage checks of every future
   training set (no training text may share an 8-gram with the gold).
After freezing, files are never edited; a change means a new version folder."""
import csv, glob, hashlib, json, os, collections
from gold_common import ROOT, norm, key, markers, PROCLITICS

G = os.path.join(ROOT, "data", "gold_v2")
F = os.path.join(G, "frozen")
os.makedirs(F, exist_ok=True)
MK = markers()
fixes = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(ROOT, "annotation", "gold_v2_fixes.tsv")), delimiter="\t"):
    fixes[(r["set"], r["id"])].append(r)


def mixed(T, tok):
    if T == "MSA":
        return False
    k = key(tok)
    cands = {k}
    m = PROCLITICS.match(k)
    if m:
        cands.add(k[m.end():])
    return any(c in MK and MK[c].get(T) == "mixed" for c in cands)


man, grams = {}, set()
applied = 0
for path in sorted(glob.glob(f"{G}/natural_*_*.jsonl") + glob.glob(f"{G}/challenge_*_*.jsonl")):
    name = os.path.basename(path)[:-6]
    if name.endswith("stats"):
        continue
    rows = [json.loads(l) for l in open(path)]
    for r in rows:
        for fx in fixes.get((name, r["id"]), []):
            i = int(fx["index"])
            r["dev"][i] = int(fx["dev"])
            r["label"][i] = fx["label"] or None
            r.setdefault("fixes", []).append(fx["reason"])
            applied += 1
        r["ignore"] = [int(mixed(r["target"], w) and not d) for w, d in zip(r["tokens"], r["dev"])]
        toks = [norm(w) for w in r["tokens"]]
        for i in range(len(toks) - 7):
            grams.add(" ".join(toks[i:i + 8]))
    out = os.path.join(F, name + ".jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man[name + ".jsonl"] = {"sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(), "sentences": len(rows),
                            "words": sum(len(r["tokens"]) for r in rows),
                            "deviations": sum(sum(r["dev"]) for r in rows),
                            "ignored": sum(sum(r["ignore"]) for r in rows)}
open(os.path.join(F, "gold_8grams.txt"), "w").write("\n".join(sorted(grams)))
json.dump({"version": "bench-v2.0", "date": "2026-10-14", "fixes_applied": applied, "files": man},
          open(os.path.join(F, "MANIFEST.json"), "w"), indent=1, ensure_ascii=False)
tot = collections.Counter()
for k, v in man.items():
    part = "natural" if k.startswith("natural") else "challenge"
    tot[part + "_sent"] += v["sentences"]; tot[part + "_words"] += v["words"]; tot[part + "_dev"] += v["deviations"]
print(f"frozen {len(man)} files, fixes applied {applied}, 8-grams {len(grams)}", dict(tot))
