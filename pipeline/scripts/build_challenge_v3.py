"""HaSeen V3 fresh challenge sets (recall + source label), built from the V3 natural sentences (held-out, native).
In a target sentence, 1–3 words that MADAR lists for the target's cities (concept c) are replaced by another variety's
form of the same concept (expert, exact position). A swap is kept only if the new form is listed for none of the
target's cities, used < 3 times in the target's human text, and is not a valid MSA word unless it comes from MSA as a
hard marker or closed-class form (formal words are shared in the service register: the V2 gold rule).
MSA target: dialect forms swapped into MGB-2 sentences (form must not be MSA-valid).
Label of a swap = the V3 variety group of the source city (MSA, EGY, SAU, GLF, LEV, IRQ, MAG, OTHER); for the Saudi
target a Gulf-east / Yemeni / Omani form gets GLF. Concepts in heldout_concepts.json are flagged for the
generalization metric (they are absent from every V3 lexicon).
Output: data/test_v3/challenge_<T>_{dev,test}.jsonl"""
import collections, json, os, random, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gold_common import key as wkey, stems, norm, madar_lexicon, markers  # noqa: E402

random.seed(2029)
OUT = f"{ROOT}/data/test_v3"
L, BY_CITY = madar_lexicon()
MK = markers()
HELD = set(json.load(open(f"{ROOT}/data/lexicons_v2/heldout_concepts.json"))["heldout"])
GROUP = {"Riyadh": "SAU", "Jeddah": "SAU", "Doha": "GLF", "Sanaa": "GLF", "Muscat": "GLF",
         "Cairo": "EGY", "Alexandria": "EGY", "Aswan": "EGY", "Jerusalem": "LEV", "Amman": "LEV", "Salt": "LEV",
         "Beirut": "LEV", "Damascus": "LEV", "Aleppo": "LEV", "Baghdad": "IRQ", "Basra": "IRQ", "Mosul": "IRQ",
         "Rabat": "MAG", "Fes": "MAG", "Tunis": "MAG", "Sfax": "MAG", "Algiers": "MAG", "Tripoli": "MAG",
         "Benghazi": "MAG", "Khartoum": "OTHER", "MSA": "MSA"}
T_CITIES = {"SAU": {"Riyadh", "Jeddah"}, "GLF": {"Riyadh", "Jeddah", "Doha", "Sanaa", "Muscat"},
            "EGY": {"Cairo", "Alexandria", "Aswan"}, "LEV": {"Jerusalem", "Amman", "Salt", "Beirut", "Damascus", "Aleppo"},
            "IRQ": {"Baghdad", "Basra", "Mosul"}, "MSA": {"MSA"}}
HUM = json.load(open(f"{ROOT}/data/inventory_v3/target_counts_human_nogold.json"))  # v3.1: train-only counts incl. SADA
T_COUNTS = {"SAU": ["SAU"], "GLF": ["SAU", "KHL", "YEM"], "EGY": ["EGY"], "LEV": ["LEV"], "IRQ": ["IRQ"], "MSA": ["MSA"]}
MSA_OK = BY_CITY["MSA"] | {w for w, n in HUM["MSA"].items() if n >= 3} | \
    {norm(l.strip()) for l in open(f"{ROOT}/../HaSeen-V1/data/lexicons/msa_stems_camel.txt")}
CLASS = {}
import csv  # noqa: E402
for r in csv.DictReader(open(f"{ROOT}/data/sources/madar_lexicon/MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0.tsv"), delimiter="\t"):
    CLASS[r["Concept_ID"]] = r["Class"]
# form -> concepts listed for the target's cities
T_FORM_C = {T: collections.defaultdict(set) for T in T_CITIES}
for c, d in L.items():
    for city, fs in d.items():
        for T, cs in T_CITIES.items():
            if city in cs:
                for f in fs:
                    T_FORM_C[T][f].add(c)
T_FORMS = {T: set(T_FORM_C[T]) for T in T_CITIES}


def mk_hard(T, f):
    f = norm(f)
    if f not in MK:
        return False
    return MK[f].get(T) == "hard" if T != "GLF" else all(MK[f].get(m) == "hard" for m in ("SAU", "KHL", "YEM"))


def ok_swap(T, f, src_city, cid):
    if "+" in f or "#" in f or len(f) < 2:
        return False
    if f in T_FORMS[T] or any(HUM.get(m, {}).get(f, 0) >= 3 for m in T_COUNTS[T]):
        return False
    if T == "MSA":
        return f not in MSA_OK
    if f in MSA_OK:
        hard = f in MK and (MK[f].get(T) == "hard" if T != "GLF" else all(MK[f].get(m) == "hard" for m in ("SAU", "KHL", "YEM")))
        return src_city == "MSA" and (hard or CLASS.get(cid) == "ClosedClass")
    return True


for T in T_CITIES:
    for sp in ("dev", "test"):
        nat = [json.loads(l) for l in open(f"{OUT}/natural_{T}_{sp}.jsonl")]
        random.shuffle(nat)
        rows, src_n = [], collections.Counter()
        tot_swaps = 0
        for x in nat:
            toks = list(x["tokens"])
            cands = []
            for i, w in enumerate(toks):
                for s in stems(w):
                    for cid in T_FORM_C[T].get(s, ()):
                        opts = [(city, f) for city, fs in L[cid].items() if city not in T_CITIES[T] and GROUP.get(city)
                                for f in fs if ok_swap(T, f, city, cid)]
                        if opts:
                            cands.append((i, cid, opts))
            msa_inj = []
            if T != "MSA":
                for i, w in enumerate(toks):
                    k = wkey(w)
                    nxt = wkey(toks[i + 1]) if i + 1 < len(toks) else ""
                    rep = None
                    if k == "اللي":
                        rep = random.choice(["الذي", "التي"])
                    elif k in ("مش", "مو", "مب", "موب", "مهوب") and re.match(r"^(ي|ت|ن|ا)[ء-ي]{2,}$", nxt or ""):
                        rep = "لن"
                    elif k in ("رح", "راح", "حا", "حـ") and re.match(r"^(ي|ت|ن|ا)[ء-ي]{2,}$", nxt or ""):
                        rep = "سوف"
                    elif T == "EGY" and re.match(r"^ه(ي|ت|ن)[ء-ي]{2,}$", k) and HUM["EGY"].get(k, 0) >= 3:
                        rep = "س" + w[1:]
                    elif T in ("EGY", "LEV") and k in ("ده", "دا", "هاد", "هيدا"):
                        rep = "هذا"
                    elif T in ("EGY", "LEV") and k in ("دي", "هاي", "هيدي"):
                        rep = "هذه"
                    if rep and (T == "SAU" or mk_hard(T, rep)):
                        msa_inj.append((i, rep))
            if not cands and not msa_inj:
                continue
            # prefer a mix of sources: ~25% MSA swaps for dialect targets
            random.shuffle(cands)
            k = random.choice([1, 1, 2, 3])
            used, dev, lab, conc, swapped = set(), [0] * len(toks), [None] * len(toks), [], []
            if msa_inj and src_n["MSA"] < 0.3 * max(1, tot_swaps) + 1:
                i, rep = random.choice(msa_inj)
                toks[i] = rep; dev[i], lab[i] = 1, "MSA"
                used.add(i); swapped.append(i); conc.append("MSA-marker"); src_n["MSA"] += 1; tot_swaps += 1
            for i, cid, opts in cands:
                if len(swapped) >= k or i in used:
                    continue
                groups = collections.defaultdict(list)
                for o in opts:
                    groups[GROUP[o[0]]].append(o)
                if T == "SAU":
                    groups.pop("SAU", None)
                if not groups:
                    continue
                # balanced source groups: MSA drift weighs 30% for dialect targets, Maghrebi is down-weighted
                gs = sorted(groups)
                wts = [(0.3 * len(gs) if g == "MSA" and T != "MSA" else 0.4 if g == "MAG" else 1.0) for g in gs]
                over = src_n["MAG"] >= 0.25 * max(1, tot_swaps)
                if over:
                    gs = [x for x in gs if x != "MAG"]
                    if not gs:
                        continue  # Maghrebi-only position while Maghrebi is already at its 25% cap
                    wts = [(0.3 * len(gs) if g == "MSA" and T != "MSA" else 1.0) for g in gs]
                g = random.choices(gs, weights=wts)[0]
                pick = random.choice(groups[g])
                city, f = pick
                g = GROUP[city]
                if T == "SAU" and g == "SAU":
                    continue
                toks[i] = f
                dev[i], lab[i] = 1, ("GLF" if (T == "SAU" and g == "GLF") else g)
                used.add(i); swapped.append(i); conc.append(cid); src_n[lab[i]] += 1; tot_swaps += 1
            if not swapped:
                continue
            rows.append({"id": x["id"].replace("nat", "ch"), "target": T, "source": x["source"], "tokens": toks, "dev": dev,
                         "label": lab, "ignore": x["ignore"], "swapped": swapped, "concepts": conc,
                         "heldout_concept": [c in HELD for c in conc]})
            if len(rows) >= (300 if sp == "dev" else 600):
                break
        with open(f"{OUT}/challenge_{T}_{sp}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(T, sp, len(rows), "sentences", sum(len(r["swapped"]) for r in rows), "swaps",
              "held-out-concept swaps", sum(sum(r["heldout_concept"]) for r in rows), dict(src_n))
