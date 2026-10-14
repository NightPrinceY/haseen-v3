"""HaSeen V3 training set v10 (2026-10-14): the v7 builder (families A-D, same safety rules) run on inventory v3
(new native sources, every V3-test hold-out removed), with labels converted at the end to the V3 label space:
  bits order V3 = MSA, EGY, SAU, GLF, LEV, IRQ, MAG
  GLF = natural in Saudi OR Gulf-east OR Yemeni (1 if any member is 1; 0 only if all three are 0; else unknown)
  MAG = Moroccan OR Tunisian (same rule); odia = natural in none of the nine V2 varieties (Sudanese, Libyan, ...)
Run with HASEEN_INV=inventory_v3 HASEEN_OUT=train_v10 HASEEN_V8_OFF=corpus_ev,msabit_dom,exact_first (dominance on).
Train v11 (labeler v10: exact MADAR forms first, msa_median, msa0_single, dom_all, absent_neg, ha_future, peak_big, سيتم hard; lex_dom opt-in):
  HASEEN_INV=inventory_v3 HASEEN_OUT=train_v11 HASEEN_V8_OFF=corpus_ev,msabit_dom

--- v7 description ---
HaSeen V2 training set v7 (2026-10-14). Phase D of docs/PLAN_V2.md.

Row: {id, family, source, host, sub, tokens, bits[[9 x 1/0/-1]], frm[[9 x 0/1]], odia[0/1], other[0/1], inj[...],
      split, pid}   bits / frm follow TARGETS order; -1 = unknown (masked in the loss).
A word deviates for target t iff bits[t] == 0; odia words deviate for every target.

Families
  A  natural hosts: verified human text first (MADAR train splits, Alexandria train, Curras / Baladi / Nâbra-west,
     Lisan, STS, Dial2MSA, Nile-Chat test), filtered web second (Iraqi factlogic high tier, mo3jam examples),
     LLM text last (Levanti synthetic, oasst2 Egyptian, Saudi conversations; capped). Other-dialect hosts
     (Khartoum, Tripoli, Benghazi, Algiers, Muscat MADAR rows, Lisan Libyan / Sudanese) with host = None.
  B  aligned swaps on MADAR TRAIN ids: host city sentence, 1-3 words replaced by another variety's word for the
     same (non-held-out) MADAR concept, taken from that variety's parallel sentence (real inflection).
  C  lexicon substitution into non-MADAR hosts (Alexandria / Curras / Baladi / Nâbra / Lisan): a host word that is a
     MADAR form of concept C for the host target is replaced by another variety's form of C.
  D  MSA future transforms: dialect future particle / prefix -> سوف / سـ (hard MSA marker for every dialect).
Labels: the labeler (src/haseen2/labeler.py) in the word's own context — host words with the host prior, injected
words labelled in their source sentence (where they are natural) without the host prior.
Safety: gold sentences never enter (MADAR / Alexandria dev+test, Casablanca, Habibi); no row shares an 8-gram
with data/gold_v2/frozen/gold_8grams.txt; held-out MADAR concepts are never a swap source; 4..40 tokens;
<= 30% unknown bits for the host target; exact dedup; split 95/5 by source sentence (pid)."""
import csv, glob, gzip, hashlib, json, os, random, re, sys, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from haseen2.labeler import Labeler, TARGETS, stems  # noqa: E402
from haseen.text import norm, words  # noqa: E402

rnd = random.Random(20261014)
L = Labeler()
OUT = os.path.join(ROOT, "data", os.environ.get("HASEEN_OUT", "train_v8"))
os.makedirs(OUT, exist_ok=True)
stats = collections.Counter()
TI = {t: i for i, t in enumerate(TARGETS)}
GOLD8 = set(open(os.path.join(ROOT, "data", "gold_v2", "frozen", "gold_8grams.txt")).read().split("\n"))
# V3: the frozen V3 test must not leak either (8-grams of every natural / challenge / probe sentence)
for _f in glob.glob(os.path.join(ROOT, "data", "test_v3", "*.jsonl")):
    for _l in open(_f):
        _t = [norm(w) for w in json.loads(_l)["tokens"]]
        GOLD8 |= {" ".join(_t[i:i + 8]) for i in range(len(_t) - 7)}
HELD = set(json.load(open(os.path.join(ROOT, "data", "lexicons_v2", "heldout_concepts.json")))["heldout"])
CITY2T = {"Cairo": "EGY", "Alexandria": "EGY", "Aswan": "EGY", "Riyadh": "SAU", "Jeddah": "SAU", "Doha": "KHL",
          "Jerusalem": "LEV", "Amman": "LEV", "Salt": "LEV", "Beirut": "LEV", "Damascus": "LEV", "Aleppo": "LEV",
          "Baghdad": "IRQ", "Basra": "IRQ", "Mosul": "IRQ", "Sanaa": "YEM", "Rabat": "MOR", "Fes": "MOR",
          "Tunis": "TUN", "Sfax": "TUN", "MSA": "MSA", "Khartoum": None, "Tripoli": None, "Benghazi": None,
          "Algiers": None, "Muscat": None}
CODE2CITY = {"ALE": "Aleppo", "ALG": "Algiers", "ALX": "Alexandria", "AMM": "Amman", "ASW": "Aswan", "BAG": "Baghdad",
             "BAS": "Basra", "BEI": "Beirut", "BEN": "Benghazi", "CAI": "Cairo", "DAM": "Damascus", "DOH": "Doha",
             "FES": "Fes", "JED": "Jeddah", "JER": "Jerusalem", "KHA": "Khartoum", "MOS": "Mosul", "MUS": "Muscat",
             "RAB": "Rabat", "RIY": "Riyadh", "SAL": "Salt", "SAN": "Sanaa", "SFA": "Sfax", "TRI": "Tripoli",
             "TUN": "Tunis"}


def leaks(toks):
    n = [norm(w) for w in toks]
    return any(" ".join(n[i:i + 8]) in GOLD8 for i in range(len(n) - 7))


def enc(labs):
    bits = [[-1 if l["bits"][t] is None else int(l["bits"][t]) for t in TARGETS] for l in labs]
    frm = [[int(t in l["frm"]) for t in TARGETS] for l in labs]
    return bits, frm, [l["odia"] for l in labs], [l["other"] for l in labs]


rows, seen = [], set()


def add(family, source, host, toks, labs, inj=(), pid="", sub=""):
    if not (4 <= len(toks) <= 40):
        stats["drop_len"] += 1; return
    key = " ".join(norm(w) for w in toks)
    if key in seen:
        stats["dup"] += 1; return
    if leaks(toks):
        stats["drop_gold_8gram"] += 1; return
    if host in TI:
        unk = sum(1 for l in labs if l["bits"][host] is None and not l["odia"]) / len(toks)
        if unk > 0.30:
            stats["drop_unknown"] += 1; return
    seen.add(key)
    b, f, o, ot = enc(labs)
    rows.append({"id": f"{family}-{hashlib.md5((pid + key).encode()).hexdigest()[:12]}", "family": family,
                 "source": source, "host": host or "ODIA", "sub": sub, "tokens": toks, "bits": b, "frm": f,
                 "odia": o, "other": ot, "inj": list(inj), "pid": pid or key[:40]})
    stats[f"fam_{family}_{host or 'ODIA'}"] += 1


# ---------------------------------------------------------------- MADAR corpus (train splits only)
MC = collections.defaultdict(dict)
for f in glob.glob(f"{ROOT}/data/sources/madar_corpus/*/MADAR_Corpus/MADAR.corpus.*.tsv"):
    city = f.split(".")[-2]
    if city in CITY2T:
        for r in csv.DictReader(open(f), delimiter="\t"):
            if r["split"].split("-")[-1] == "train":
                MC[city][r["sentID.BTEC"]] = r["sent"]

# ---------------------------------------------------------------- A: natural hosts
CAP = {"iraqi_factlogic_high": 20000, "iraqi_factlogic_medium": 6000, "iraqi_factlogic_medium_extended": 0,
       "levanti_real": 25000, "levanti_syn": 12000, "oasst2_9k": 6000, "saudi_conversations": 8000,
       "mo3jam_examples": 12000, "lisan_yemeni": 20000, "algerian_darja": 4000, "nilechat_egy_msa_test": 6000,
       "sada": 70000, "masrispeech": 40000, "mgb3": 10 ** 9, "scc": 10 ** 9, "acqa": 10 ** 9}
V2T = {"EG": "EGY", "SA": "SAU", "KW": "KHL", "QA": "KHL", "BH": "KHL", "AE": "KHL", "PS": "LEV", "JO": "LEV",
       "LB": "LEV", "SY": "LEV", "LEV": "LEV", "IQ": "IRQ", "YE": "YEM", "MA": "MOR", "TN": "TUN", "MSA": "MSA",
       "GLF": "KHL", "OM": "KHL"}  # V3: generic Gulf (SADA Khaliji, ArabicCulturalQA gulf) and Omani are Gulf members
for city, d in MC.items():
    for sid, sent in d.items():
        toks = sent.split()
        add("A", f"madar:{city}", CITY2T[city], toks, L.label(toks, CITY2T[city]), pid=f"madar-{sid}", sub=city)
pool = collections.defaultdict(list)
for l in gzip.open(f"{ROOT}/data/{os.environ.get('HASEEN_INV', 'inventory')}/sentences.jsonl.gz", "rt"):
    r = json.loads(l)
    src = r["source"]
    if src in ("casablanca", "habibi", "madar6") or r["split"] in ("dev", "test"):
        continue
    if src == "levanti":
        src = "levanti_syn" if r["kind"] == "synthetic" else "levanti_real"
    pool[src].append(r)
for src, rs in pool.items():
    rnd.shuffle(rs)
    for r in rs[:CAP.get(src, 10 ** 9)]:
        host = V2T.get(r["var"])  # None for other dialects (SD, LY, DZ, MR, MAG)
        toks = r["text"].split()
        add("A", src, host, toks, L.label(toks, host), pid=f"{src}-{hashlib.md5(r['text'].encode()).hexdigest()[:10]}",
            sub=r.get("sub", "") or r["var"])

# ---------------------------------------------------------------- B: aligned swaps on MADAR train ids
CONC = collections.defaultdict(lambda: collections.defaultdict(set))
FORM_C = collections.defaultdict(lambda: collections.defaultdict(set))  # city -> form -> concepts
for r in csv.DictReader(open(f"{ROOT}/data/sources/madar_lexicon/MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0.tsv"), delimiter="\t"):
    if r["Release Status"] != "READY" or r["Concept_ID"] in HELD:
        continue
    f = norm(r["CODA"].strip())
    if f and " " not in f:
        city = CODE2CITY[r["Dialect"]]
        CONC[r["Concept_ID"]][city].add(f)
        FORM_C[city][f].add(r["Concept_ID"])
    for m in r["MSA"].replace("،", ",").split(","):  # MSA representative forms act as the "MSA city"
        m = norm(m.strip())
        if m and " " not in m and "+" not in m and "#" not in m:
            CONC[r["Concept_ID"]]["MSA"].add(m)
            FORM_C["MSA"][m].add(r["Concept_ID"])


def tgt_forms(T):
    return {f for c, d in CONC.items() for city, fs in d.items() if CITY2T[city] == T for f in fs}


TFORMS = {T: tgt_forms(T) for T in TARGETS}


def swaps(host_city, htoks, src_city, stoks):
    T = CITY2T[host_city]
    out, used_i, used_j = [], set(), set()
    for i, w in enumerate(htoks):
        for f in stems(norm(w)) & set(FORM_C[host_city]):
            for cid in FORM_C[host_city][f]:
                for j, v in enumerate(stoks):
                    if i in used_i or j in used_j or norm(v) == norm(w):
                        continue
                    if not (stems(norm(v)) & CONC[cid].get(src_city, set())):
                        continue
                    if stems(norm(v)) & TFORMS[T]:
                        continue
                    out.append((i, j)); used_i.add(i); used_j.add(j)
    return out


cities = [c for c in MC if c != "MSA"] + ["MSA"]
nB = collections.Counter()
ids = sorted(set(MC["Cairo"]))
for sid in ids:
    hosts = [c for c in cities if sid in MC[c] and CITY2T[c]]
    rnd.shuffle(hosts)
    for hc in hosts:
        T = CITY2T[hc]
        if nB[T] >= 12000:
            continue
        srcs = [c for c in cities if c != hc and sid in MC[c] and CITY2T[c] != T]
        rnd.shuffle(srcs)
        for sc in srcs[:3]:
            ht, st = MC[hc][sid].split(), MC[sc][sid].split()
            pr = swaps(hc, ht, sc, st)
            if not pr:
                continue
            hl, sl = L.label(ht, T), L.label(st, CITY2T[sc])
            # keep only swaps the labeler confirms as deviations for the host target (V1 rule: inject deviants)
            pr = [(i, j) for i, j in pr if sl[j]["bits"][T] == 0 or (CITY2T[sc] is None and sl[j]["bits"][T] is None)]
            if CITY2T[sc] is None:  # v8: "other dialect" injections only for words natural in no target
                pr = [(i, j) for i, j in pr if not any(v == 1 for v in sl[j]["bits"].values())]
            if not pr:
                continue
            pr = rnd.sample(pr, min(len(pr), rnd.choice([1, 1, 2, 3])))
            toks, labs = list(ht), list(hl)
            for i, j in pr:
                toks[i] = st[j]
                lab = dict(sl[j]); lab["bits"] = dict(lab["bits"])
                if CITY2T[sc] is None:
                    lab["odia"] = 1
                labs[i] = lab
            add("B", f"madar:{hc}<{sc}", T, toks, labs, inj=[i for i, _ in pr], pid=f"madar-{sid}", sub=hc)
            nB[T] += 1

# ---------------------------------------------------------------- C: lexicon substitution into non-MADAR hosts
FORM_T = collections.defaultdict(set)  # form -> targets (incl. None for other dialects)
for c, d in CONC.items():
    for city, fs in d.items():
        for f in fs:
            FORM_T[f].add(CITY2T[city])
nC = collections.Counter()
hostsC = [r for r in rows if r["family"] == "A" and not r["source"].startswith("madar") and r["host"] in TI
          and "syn" not in r["source"] and "oasst2" not in r["source"]]
rnd.shuffle(hostsC)
for r in hostsC:
    T = r["host"]
    if nC[T] >= 8000:
        continue
    toks = list(r["tokens"])
    cands = []
    for i, w in enumerate(toks):
        k = norm(w)
        for cid in set().union(*(FORM_C[c].get(k, set()) for c in FORM_C if CITY2T.get(c) == T)):
            others = [(city, f) for city, fs in CONC[cid].items() for f in fs
                      if CITY2T[city] != T and f not in TFORMS[T] and f != k]
            if others:
                cands.append((i, rnd.choice(others)))
    if not cands:
        continue
    cands = [(i, (city, f)) for i, (city, f) in cands
             if L.label_word(f, CITY2T[city])["bits"][T] == 0
             or (CITY2T[city] is None and L.label_word(f, None)["bits"][T] is None)]
    # v8: "other dialect" injections only for words natural in no target
    cands = [(i, (city, f)) for i, (city, f) in cands
             if CITY2T[city] is not None or not any(v == 1 for v in L.label_word(f, None)["bits"].values())]
    if not cands:
        continue
    pick = rnd.sample(cands, min(len(cands), rnd.choice([1, 1, 2])))
    labs = L.label(toks, T)
    inj = []
    for i, (city, f) in pick:
        toks[i] = f
        lab = L.label_word(f, CITY2T[city])
        if CITY2T[city] is None:
            lab["odia"] = 1
        labs[i] = lab
        inj.append(i)
    add("C", r["source"] + ">lex", T, toks, labs, inj=inj, pid=r["pid"], sub=r["sub"])
    nC[T] += 1

# ---------------------------------------------------------------- D: MSA future transforms
FUT_PART = {"رح", "راح", "ح", "باش", "غادي", "غا", "بش", "حا", "ها", "بيش"}
nD = collections.Counter()
for r in [x for x in rows if x["family"] == "A" and x["host"] in TI and x["host"] != "MSA"]:
    T = r["host"]
    if nD[T] >= 3000:
        continue
    toks = list(r["tokens"])
    for i in range(len(toks) - 1):
        w = norm(toks[i])
        nxt = norm(toks[i + 1])
        if w in FUT_PART and re.match(r"^(ي|ت|ن|ا)[ء-ي]{2,}$", nxt):
            toks[i] = "سوف"
            break
        if T == "EGY" and re.match(r"^ه(ي|ت|ن)[ء-ي]{2,}$", w) and L.cnt["EGY"].get(w, 0) >= 3:
            toks[i] = "س" + toks[i][1:]
            break
    else:
        continue
    labs = L.label(toks, T)
    inj = [j for j, (a, b) in enumerate(zip(toks, r["tokens"])) if a != b]
    if any(labs[j]["bits"][T] != 0 for j in inj):
        continue  # the transformed word must be a confirmed MSA deviation for the host
    add("D", r["source"] + ">fut", T, toks, labs, inj=inj, pid=r["pid"], sub=r["sub"])
    nD[T] += 1

# ---------------------------------------------------------------- V3 label space
V3T = ("MSA", "EGY", "SAU", "GLF", "LEV", "IRQ", "MAG")
I9 = {t: i for i, t in enumerate(TARGETS)}
GROUPS = {"MSA": ("MSA",), "EGY": ("EGY",), "SAU": ("SAU",), "GLF": ("SAU", "KHL", "YEM"), "LEV": ("LEV",),
          "IRQ": ("IRQ",), "MAG": ("MOR", "TUN")}
H3 = {"MSA": "MSA", "EGY": "EGY", "SAU": "SAU", "KHL": "GLF", "YEM": "GLF", "LEV": "LEV", "IRQ": "IRQ",
      "MOR": "MAG", "TUN": "MAG", "ODIA": "ODIA"}


def to_v3(b9):
    out = []
    for t in V3T:
        vals = [b9[I9[m]] for m in GROUPS[t]]
        out.append(1 if 1 in vals else 0 if all(v == 0 for v in vals) else -1)
    return out


for r in rows:
    r["bits9"] = r["bits"]
    r["bits"] = [to_v3(b) for b in r["bits"]]
    r["frm"] = [[int(any(f[I9[m]] for m in GROUPS[t])) for t in V3T] for f in r["frm"]]
    r["host9"] = r["host"]
    r["host"] = H3.get(r["host"], r["host"])
TARGETS = V3T  # stats below in V3 terms
# ---------------------------------------------------------------- split by pid, write
pids = sorted({r["pid"] for r in rows})
rnd.shuffle(pids)
dev_p = set(pids[: len(pids) // 20])
with open(f"{OUT}/train.jsonl", "w") as ft, open(f"{OUT}/dev.jsonl", "w") as fd:
    for r in rows:
        r["split"] = "dev" if r["pid"] in dev_p else "train"
        (fd if r["split"] == "dev" else ft).write(json.dumps(r, ensure_ascii=False) + "\n")
tok = collections.Counter()
for r in rows:
    for b, o in zip(r["bits"], r["odia"]):
        tok["tokens"] += 1
        tok["odia"] += o
        for t, v in zip(TARGETS, b):
            tok[f"{t}={v}"] += 1
json.dump({"stats": stats, "tokens": tok, "rows": len(rows), "train": sum(r["split"] == "train" for r in rows),
           "dev": sum(r["split"] == "dev" for r in rows)}, open(f"{OUT}/stats.json", "w"), indent=1)
print(json.dumps({"rows": len(rows), **{k: v for k, v in stats.items() if not k.startswith("fam_")}}, ensure_ascii=False))
fam = collections.Counter()
for k, v in stats.items():
    if k.startswith("fam_"):
        fam[k] = v
for k, v in sorted(fam.items()):
    print(k, v)
print({t: (tok[f"{t}=1"], tok[f"{t}=0"], tok[f"{t}=-1"]) for t in TARGETS}, "odia", tok["odia"], "of", tok["tokens"])
