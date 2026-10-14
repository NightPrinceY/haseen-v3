"""Data available per candidate V2 target (2026-10-14) -> data/inventory/targets.json

Targets follow the separability results (word_separability.py): countries whose words cannot be told apart are
merged (PS+JO, LB+SY, Riyadh+Jeddah), the rest stay separate. Counts are after exact de-duplication."""
import gzip, json, os, collections, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "HaSeen-V1", "src"))
from haseen.text import norm, words  # noqa: E402

INV = os.path.join(os.path.dirname(__file__), "..", "data", "inventory")
T = {"MSA": ["MSA"], "EGY": ["EG"], "SDN": ["SD"], "SLEV": ["PS", "JO"], "NLEV": ["LB", "SY"], "IRQ": ["IQ"],
     "SAU": ["SA"], "KHL": ["KW", "QA", "BH", "AE", "GLF"], "OMN": ["OM"], "YEM": ["YE"], "LBY": ["LY"],
     "TUN": ["TN"], "ALG": ["DZ"], "MOR": ["MA"], "HAS": ["MR"], "(LEV, country unknown)": ["LEV"],
     "(MAG, country unknown)": ["MAG"]}
V2T = {v: t for t, vs in T.items() for v in vs}
MADAR = {"EGY": "Cairo, Alexandria, Aswan", "SDN": "Khartoum", "SLEV": "Jerusalem, Amman, Salt",
         "NLEV": "Beirut, Damascus, Aleppo", "IRQ": "Baghdad, Basra, Mosul", "SAU": "Riyadh, Jeddah", "KHL": "Doha",
         "OMN": "Muscat", "YEM": "Sana'a", "LBY": "Tripoli, Benghazi", "TUN": "Tunis, Sfax", "ALG": "Algiers",
         "MOR": "Rabat, Fes", "HAS": "-", "MSA": "MSA column"}
EVAL = {"alexandria:test", "casablanca:test", "habibi:test"}

seen = set()
S = {t: collections.Counter() for t in T}
src = {t: collections.Counter() for t in T}
words_n = collections.Counter()
evaln = collections.Counter()
for l in gzip.open(os.path.join(INV, "sentences.jsonl.gz"), "rt"):
    r = json.loads(l)
    t = V2T.get(r["var"])
    if not t or r["text"] in seen:
        continue
    seen.add(r["text"])
    S[t][r["kind"]] += 1
    src[t][r["source"]] += 1
    if r["kind"] != "synthetic":
        words_n[t] += len(r["text"].split())
    if f"{r['source']}:{r['split']}" in EVAL:
        evaln[t] += 1

lex = collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(os.path.join(INV, "lexicon.jsonl")):
    r = json.loads(l)
    if r["source"] in ("curras", "baladi", "nabra") or r["source"].startswith("lisan"):
        continue  # corpus tokens are not lexicon entries
    for v in r["vars"]:
        if v in V2T:
            lex[V2T[v]][r["form"]].add(r["source"])
out = {}
print(f"{'target':24s}{'native':>8}{'transl':>8}{'spoken':>8}{'synth':>8}{'human words':>12}{'eval sents':>11}{'lex forms':>10}{'≥2 lexicons':>12}  sources")
for t in T:
    k = S[t]
    lf = lex[t]
    multi = sum(1 for f, s in lf.items() if len(s) >= 2)
    out[t] = {"countries": T[t], "sentences": dict(k), "human_words": words_n[t], "eval_sentences": evaln[t],
              "lexicon_forms": len(lf), "lexicon_forms_2plus": multi, "sources": dict(src[t].most_common()),
              "madar_cities": MADAR.get(t, "-")}
    print(f"{t:24s}{k['native']:>8}{k['translated']:>8}{k['spoken']:>8}{k['synthetic']:>8}{words_n[t]:>12}{evaln[t]:>11}{len(lf):>10}{multi:>12}  {len(src[t])}")
json.dump(out, open(os.path.join(INV, "targets.json"), "w"), indent=1, ensure_ascii=False)
