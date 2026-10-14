"""V3 lexical probe from mo3jam (built and frozen BEFORE any V3 data or training).

20% of eligible mo3jam entries (single word, one dialect tag, net votes >= 3, with an example) are held out by
sha1(def_id); their entries and example sentences must never enter V3 lexicons, counts or hosts
(data/test_v3/probe_heldout_def_ids.json is read by the V3 inventory / labeler builders).
Each probe item = the entry's own example sentence with the head word located in it, and two kinds of checks:
  own:   target = the word's own V3 group (and Saudi for Saudi words)       -> the word must CONFORM
  cross: target = a V3 production target where no source lists the word  -> the word must DEVIATE
Cross targets require: the form (or its proclitic stem) is listed for none of the target's varieties in any lexicon
(MADAR, Lahga, mo3jam, Birzeit, ...), is used < 3 times in the target's human text, and for MSA is no CALIMA-MSA stem
and < 3 uses in human MSA text."""
import collections, hashlib, json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "HaSeen-V1", "src"))
from haseen.text import norm  # noqa: E402

PERSIAN = str.maketrans({"ی": "ي", "ک": "ك", "ە": "ه", "ۀ": "ه", "ي": "ي"})
n = lambda w: norm(re.sub(r"[^\w]", "", w.translate(PERSIAN)))
PROC = re.compile(r"^(و|ف|ب|ل|ك|ال|وال|بال|فال|لل|ول)")

MJ2V = {"EGA": "EG", "SUA": "SD", "LEA_PS": "PS", "LEA_JO": "JO", "LEA_LB": "LB", "LEA_SY": "SY", "LEA": "LEV",
        "IRA": "IQ", "SAA": "SA", "SAA_HI": "SA", "SAA_NA": "SA", "YEA": "YE", "GAA": "GLF", "GAA_AE": "AE",
        "GAA_BH": "BH", "GAA_KW": "KW", "GAA_OM": "OM", "GAA_QA": "QA", "MAA": "MAG", "MAA_DZ": "DZ", "MAA_LY": "LY",
        "MAA_MA": "MA", "MAA_MR": "MR", "MAA_TN": "TN"}
# V3 label space: production targets MSA, EGY, GLF (Saudi, Gulf-east, Yemeni, Omani), SAU (inside GLF), LEV, IRQ
V3 = {"EG": {"EGY"}, "SA": {"SAU", "GLF"}, "AE": {"GLF"}, "QA": {"GLF"}, "KW": {"GLF"}, "BH": {"GLF"}, "OM": {"GLF"},
      "GLF": {"GLF"}, "YE": {"GLF"}, "PS": {"LEV"}, "JO": {"LEV"}, "LB": {"LEV"}, "SY": {"LEV"}, "LEV": {"LEV"},
      "IQ": {"IRQ"}, "DZ": {"MAG"}, "MA": {"MAG"}, "TN": {"MAG"}, "LY": {"MAG"}, "MR": {"MAG"}, "MAG": {"MAG"},
      "SD": {"OTHER"}, "MSA": {"MSA"}}
TARGETS = ("MSA", "EGY", "GLF", "SAU", "LEV", "IRQ")
T_VARS = {"EGY": {"EG"}, "SAU": {"SA"}, "GLF": {"SA", "AE", "QA", "KW", "BH", "OM", "GLF", "YE"},
          "LEV": {"PS", "JO", "LB", "SY", "LEV"}, "IRQ": {"IQ"}, "MSA": {"MSA"}}
T_COUNTS = {"EGY": ["EGY"], "SAU": ["SAU"], "GLF": ["SAU", "KHL", "YEM"], "LEV": ["LEV"], "IRQ": ["IRQ"], "MSA": ["MSA"]}

defs = [json.loads(l) for l in open(f"{ROOT}/data/sources/mo3jam/definitions.jsonl")]
# every variety each normalized form is listed for, across all lexicons (incl. every mo3jam entry)
listed = collections.defaultdict(set)
for l in open(f"{ROOT}/data/inventory/lexicon.jsonl"):
    r = json.loads(l)
    listed[n(r["form"])] |= set(r["vars"])
for d in defs:
    for c in d["dialects"]:
        listed[n(d["term"])].add(MJ2V.get(c, c))
cnt = json.load(open(f"{ROOT}/data/inventory/target_counts_human_nogold.json"))
calima = {norm(l.strip()) for l in open(f"{ROOT}/../HaSeen-V1/data/lexicons/msa_stems_camel.txt")}


def stems(w):
    s = {w}
    m = PROC.match(w)
    if m and len(w) - m.end() >= 2:
        s.add(w[m.end():])
    return s


def cross_ok(w, t):
    S = stems(w)
    if any(listed.get(x, set()) & T_VARS[t] for x in S):
        return False
    if any(cnt.get(c, {}).get(x, 0) >= 3 for c in T_COUNTS[t] for x in S):
        return False
    if t == "MSA" and S & calima:
        return False
    return True


eligible = [d for d in defs if d["kind"] == "word" and len(d["dialects"]) == 1 and d["votes_up"] - d["votes_down"] >= 3
            and (d["defs"].get("ar") or {}).get("example", "").strip() and MJ2V.get(d["dialects"][0])]
held = [d for d in eligible if int(hashlib.sha1(f"haseen-v3-probe-{d['def_id']}".encode()).hexdigest(), 16) % 5 == 0]
items, stats = [], collections.Counter()
for d in held:
    var = MJ2V[d["dialects"][0]]
    w = n(d["term"])
    if len(w) < 2:
        continue
    ex = (d["defs"]["ar"]["example"]).split("\n")[0].split("=")[0].strip()
    toks = ex.split()
    pos = next((i for i, t in enumerate(toks) if w in stems(n(t)) or n(t) == w), None)
    if pos is None or len(toks) < 3:
        stats["word_not_in_example"] += 1
        continue
    own = sorted(V3[var] & set(TARGETS))
    cross = [t for t in TARGETS if t not in V3[var] and not (var in T_VARS["GLF"] and t == "SAU") and cross_ok(n(toks[pos]), t)]
    checks = [{"target": t, "expect": "conform"} for t in own] + [{"target": t, "expect": "deviate"} for t in cross]
    if not checks:
        continue
    items.append({"id": f"probe-{d['def_id']}", "def_id": d["def_id"], "word": toks[pos], "variety": var,
                  "group": sorted(V3[var]), "tokens": toks, "pos": pos, "checks": checks,
                  "votes": d["votes_up"] - d["votes_down"]})
    for c in checks:
        stats[(c["expect"], c["target"])] += 1
    stats[("items", var)] += 1
out = f"{ROOT}/data/test_v3"
os.makedirs(out, exist_ok=True)
json.dump(sorted(d["def_id"] for d in held), open(f"{out}/probe_heldout_def_ids.json", "w"))
with open(f"{out}/probe_mo3jam.jsonl", "w") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(len(eligible), "eligible;", len(held), "held out;", len(items), "probe items")
for k, v in sorted(stats.items(), key=lambda x: str(x[0])):
    print(k, v)
