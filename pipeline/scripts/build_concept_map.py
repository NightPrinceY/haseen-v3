"""Suggestion map for the SDK: form -> {target: [forms the target's cities use for the same MADAR concept]}.

Only open- and closed-class concepts with READY rows; forms ordered by how many of the target's cities list them.
Output: data/lexicons_v2/release/concept_map.json, shipped only inside the private model repos (MADAR is research-only;
never commit it to a code repository)."""
import collections, csv, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gold_common import CITY2T, norm  # noqa: E402

TSV = os.path.join(ROOT, "data", "sources", "madar_lexicon", "MADAR_Lexicon_v1.0", "MADAR_Lexicon_v1.0.tsv")
CODE = {"ALE": "Aleppo", "ALG": "Algiers", "ALX": "Alexandria", "AMM": "Amman", "ASW": "Aswan", "BAG": "Baghdad",
        "BAS": "Basra", "BEI": "Beirut", "BEN": "Benghazi", "CAI": "Cairo", "DAM": "Damascus", "DOH": "Doha",
        "FES": "Fes", "JED": "Jeddah", "JER": "Jerusalem", "KHA": "Khartoum", "MOS": "Mosul", "MUS": "Muscat",
        "RAB": "Rabat", "RIY": "Riyadh", "SAL": "Salt", "SAN": "Sanaa", "SFA": "Sfax", "TRI": "Tripoli", "TUN": "Tunis"}
conc = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for r in csv.DictReader(open(TSV), delimiter="\t"):
    if r["Release Status"] != "READY":
        continue
    f = norm(r["CODA"].strip())
    city = CODE.get(r["Dialect"], r["Dialect"])
    t = CITY2T.get(city)
    if f and " " not in f and t and t not in ("OTHER_DIA",):
        conc[r["Concept_ID"]][t][f] += 1
    for m in r["MSA"].replace("،", ",").split(","):
        m = norm(m.strip())
        if m and " " not in m and "+" not in m and "#" not in m:
            conc[r["Concept_ID"]]["MSA"][m] += 1
out = collections.defaultdict(dict)
for c, d in conc.items():
    forms = {f for cnt in d.values() for f in cnt}
    for f in forms:
        for t, cnt in d.items():
            if f not in cnt:
                out[f].setdefault(t, [x for x, _ in cnt.most_common(3)])
dst = os.path.join(ROOT, "data", "lexicons_v2", "release")  # MADAR-derived: ships only inside the private model repos
os.makedirs(dst, exist_ok=True)
json.dump(out, open(os.path.join(dst, "concept_map.json"), "w"), ensure_ascii=False, separators=(",", ":"))
print(len(out), "forms;", "example ابغي ->", out.get("ابغي"), "| سوف ->", out.get("سوف", {}).get("EGY"))
