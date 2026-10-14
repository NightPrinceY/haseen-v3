"""Profile mo3jam definitions.jsonl for the V2 target decision -> data/sources/mo3jam/profile.json (+ printed summary).

Questions: how big per dialect (words vs phrases), how trustworthy (votes, authors, dates), how many give an MSA
gloss and a real example sentence, which dialects share forms, how the related-terms graph groups equivalents
across dialects, and how much is new against Lahga and the V1 core lexicon."""
import collections, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "HaSeen-V1", "src"))
from haseen.text import norm, words  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "sources", "mo3jam")
V1 = os.path.join(os.path.dirname(__file__), "..", "..", "HaSeen-V1", "data", "lexicons")
NAMES = {"MAA": "Maghrebi", "MAA_DZ": "Algerian", "MAA_LY": "Libyan", "MAA_MA": "Moroccan", "MAA_MR": "Hassaniya",
         "MAA_TN": "Tunisian", "EGA": "Egyptian", "SUA": "Sudanese", "LEA": "Levantine", "LEA_JO": "Jordanian",
         "LEA_LB": "Lebanese", "LEA_PS": "Palestinian", "LEA_SY": "Syrian", "IRA": "Iraqi", "SAA": "Saudi",
         "SAA_HI": "Hijazi", "SAA_NA": "Najdi", "YEA": "Yemeni", "GAA": "Khaliji", "GAA_AE": "Emirati",
         "GAA_BH": "Bahraini", "GAA_KW": "Kuwaiti", "GAA_OM": "Omani", "GAA_QA": "Qatari"}
REGION = lambda c: c.split("_")[0]  # MAA, EGA, SUA, LEA, IRA, SAA, YEA, GAA
GLOSS = re.compile(r"(?:=|تعني|يعني|بمعنى|معناها|معناه|أي|اي)\s*[:：]?\s*([^\n.،,؛]+)")

D = [json.loads(l) for l in open(os.path.join(ROOT, "definitions.jsonl"))]
P = {}
P["definitions"] = len(D)
P["terms"] = len({d["index_term"] for d in D})
P["kind"] = collections.Counter(d["kind"] for d in D)
single = [d for d in D if len(words(d["term"])) == 1]
P["single_word_definitions"] = len(single)
P["single_word_forms"] = len({norm(words(d["term"])[0]) for d in single})

# votes / authors / dates
net = [d["votes_up"] - d["votes_down"] for d in D]
P["votes"] = {"no_votes": sum(1 for d in D if d["votes_up"] + d["votes_down"] == 0),
              "net_negative": sum(1 for x in net if x < 0), "net_ge_3": sum(1 for x in net if x >= 3),
              "net_ge_10": sum(1 for x in net if x >= 10)}
P["guest_share"] = round(sum(1 for d in D if d["user"] == "a guest") / len(D), 3)
P["authors"] = len({d["user"] for d in D if d["user"] != "a guest"})
P["top_authors"] = collections.Counter(d["user"] for d in D if d["user"] != "a guest").most_common(10)
P["years"] = dict(sorted(collections.Counter(d["date"][:4] for d in D).items()))
P["def_langs"] = collections.Counter(l for d in D for l in d["defs"])
P["n_dialect_tags"] = dict(sorted(collections.Counter(len(d["dialects"]) for d in D).items()))

# per dialect (leaf codes; a parent-only tag such as SAA counts under the parent)
per = collections.defaultdict(lambda: collections.Counter())
forms = collections.defaultdict(set)
for d in D:
    w1 = len(words(d["term"])) == 1
    ex = any(v["example"] for v in d["defs"].values())
    ar = d["defs"].get("ar", {}).get("definition", "")
    gl = bool(GLOSS.search(ar)) or (0 < len(ar.split()) <= 3)
    good = d["votes_up"] - d["votes_down"] >= 1
    for c in d["dialects"]:
        p = per[c]
        p["defs"] += 1; p["word"] += w1; p["with_example"] += ex; p["short_or_marked_gloss"] += gl; p["net_pos"] += good
        if w1:
            forms[c].add(norm(words(d["term"])[0]))
P["per_dialect"] = {f"{c} {NAMES.get(c, c)}": dict(v, single_word_forms=len(forms[c]))
                    for c, v in sorted(per.items(), key=lambda kv: -kv[1]["defs"])}

# region level and overlap (same normalized single-word form tagged in two regions)
reg_forms = collections.defaultdict(set)
for c, fs in forms.items():
    reg_forms[REGION(c)] |= fs
regs = sorted(reg_forms, key=lambda r: -len(reg_forms[r]))
P["region_single_word_forms"] = {r: len(reg_forms[r]) for r in regs}
P["region_overlap"] = {f"{a}&{b}": len(reg_forms[a] & reg_forms[b]) for i, a in enumerate(regs) for b in regs[i + 1:]}
owners = collections.Counter()
for f in set().union(*reg_forms.values()):
    owners[sum(f in reg_forms[r] for r in regs)] += 1
P["forms_by_number_of_regions"] = dict(sorted(owners.items()))

# related-terms graph: connected components = concept groups across dialects
by_id = {d["def_id"]: d for d in D}
adj = collections.defaultdict(set)
for d in D:
    for r in d["related"]:
        adj[d["def_id"]].add(r["def_id"]); adj[r["def_id"]].add(d["def_id"])
seen, comps = set(), []
for s in adj:
    if s in seen:
        continue
    st, comp = [s], []
    seen.add(s)
    while st:
        x = st.pop(); comp.append(x)
        for y in adj[x] - seen:
            seen.add(y); st.append(y)
    comps.append(comp)
sizes = sorted((len(c) for c in comps), reverse=True)
multi = [c for c in comps if len({REGION(t) for i in c if i in by_id for t in by_id[i]["dialects"]}) >= 3]
P["related_graph"] = {"defs_with_related": sum(1 for d in D if d["related"]), "components": len(comps),
                      "largest": sizes[:10], "components_spanning_3plus_regions": len(multi)}

# novelty against V1 resources
lahga = json.load(open(os.path.join(V1, "lahga_map.json")))
core = {l.split("\t")[0] for l in open(os.path.join(V1, "core_v2.tsv")) if l.strip() and not l.startswith(("#", "form\t"))}
allf = set().union(*forms.values())
P["vs_v1"] = {"single_word_forms": len(allf), "in_lahga": len(allf & set(lahga)), "in_core_v2": len(allf & core),
              "new": len(allf - set(lahga) - core)}

json.dump(P, open(os.path.join(ROOT, "profile.json"), "w"), ensure_ascii=False, indent=1, default=dict)
print(json.dumps(P, ensure_ascii=False, indent=1, default=dict)[:12000])
