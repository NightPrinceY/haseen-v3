"""Audit labeler v2 on gold DEV parts only (never test). natural: words of the host target labelled 0 = labeler
false alarms; challenge: swapped words labelled 0 for the target = hits, 1 = wrong, None = unknown."""
import json, os, sys, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from haseen2.labeler import Labeler, TARGETS
L = Labeler()
G = os.path.join(ROOT, "data", "gold_v2", "frozen")
fa_words = collections.Counter()
print(f"{'T':5s}{'nat words':>10}{'FA':>6}{'FA%':>7}{'known%':>8} | {'swaps':>6}{'hit':>6}{'wrong':>6}{'unk':>6}")
for T in TARGETS:
    n = fa = known = 0
    for l in open(f"{G}/natural_{T}_dev.jsonl"):
        r = json.loads(l)
        for w, lab, ig in zip(r["tokens"], L.label(r["tokens"], host=None), r["ignore"]):
            if ig:
                continue
            n += 1
            b = lab["bits"][T]
            known += b is not None
            if b == 0:
                fa += 1; fa_words[(T, w, lab["src"])] += 1
    sw = hit = wrong = unk = 0
    for l in open(f"{G}/challenge_{T}_dev.jsonl"):
        r = json.loads(l)
        labs = L.label(r["tokens"], host=None)
        for i in r["swapped"]:
            sw += 1
            b = labs[i]["bits"][T]
            hit += b == 0; wrong += b == 1; unk += b is None
    print(f"{T:5s}{n:>10}{fa:>6}{fa / n:>7.2%}{known / n:>8.1%} | {sw:>6}{hit:>6}{wrong:>6}{unk:>6}")
print("top labeler false alarms:", [(t, w, s, c) for (t, w, s), c in fa_words.most_common(30)])
