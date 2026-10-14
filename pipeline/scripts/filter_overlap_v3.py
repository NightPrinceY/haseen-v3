"""Strict leakage filter for a V3 training set (in place): drop train/dev rows that share a whole normalized sentence
or any normalized 6-gram with ANY file of data/test_v3 (natural / challenge dev + test, agent, mo3jam probe).
Usage: python scripts/filter_overlap_v3.py data/train_v11"""
import glob, json, os, re, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "HaSeen-V1", "src"))
from haseen.text import norm  # noqa: E402
PERSIAN = str.maketrans({"ی": "ي", "ک": "ك", "گ": "ك"})
N = lambda w: norm(re.sub(r"[^\w]", "", w.translate(PERSIAN)))
sents, grams = set(), set()
for p in glob.glob(f"{ROOT}/data/test_v3/*.jsonl"):
    for l in open(p):
        if not l.strip():
            continue
        t = [x for x in map(N, json.loads(l)["tokens"]) if x]
        sents.add(" ".join(t))
        grams.update(" ".join(t[i:i + 6]) for i in range(len(t) - 5))
d = sys.argv[1]
out = {}
for part in ("train", "dev"):
    keep, drop = [], 0
    for l in open(f"{d}/{part}.jsonl"):
        t = [x for x in map(N, json.loads(l)["tokens"]) if x]
        if " ".join(t) in sents or any(" ".join(t[i:i + 6]) in grams for i in range(len(t) - 5)):
            drop += 1; continue
        keep.append(l)
    open(f"{d}/{part}.jsonl", "w").writelines(keep)
    out[part] = {"kept": len(keep), "dropped": drop}
json.dump(out, open(f"{d}/overlap_filter.json", "w"), indent=1)
print(out, "| test sentences", len(sents), "6-grams", len(grams))
