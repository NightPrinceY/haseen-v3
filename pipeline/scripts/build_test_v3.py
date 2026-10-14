"""HaSeen V3 fresh evaluation: natural sets (precision) for the V3 production targets, built and frozen BEFORE any V3
data or training. Sources are human-transcribed / human-written native text that V3 training will never see:
  SAU / GLF  SADA held-out shows, speaker-disjoint from training (sada_clean_v3.jsonl: split3 dev / test), SDAIA SCC
             (20% by hash), Lisan Yemeni (held-out hash) for the Yemeni part of GLF
  EGY        MGB-3 test (dev: validation), MasriSpeech validation (dev: 4% of train by hash)
  LEV        Curras / Nâbra / Baladi / Levanti-native held out by sentence hash
  IRQ        Lisan Iraqi / factlogic-high held out by sentence hash
  MSA        MGB-2 test (dev: validation), colloquial sentences removed
  + ArabicCulturalQA (QCRI; parallel, post-edited by natives) test / validation for EGY, LEV, GLF, MSA
Gold: every word conforms (dialect targets: sentences with a hard MSA marker, >= 2 formal function words or > 30% MSA-
skewed words are dropped; MSA: explicit dialect leaks are labelled). Words whose status is "mixed" are ignored.
Hashed hold-outs are written to data/test_v3/heldout_text_sha1.json; the V3 inventory must drop them.
Output: data/test_v3/natural_<T>_{dev,test}.jsonl"""
import collections, gzip, hashlib, json, os, random, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from gold_common import key as wkey, norm, markers  # noqa: E402

random.seed(2028)
S = f"{ROOT}/data/sources"
OUT = f"{ROOT}/data/test_v3"
os.makedirs(OUT, exist_ok=True)
H = lambda t: hashlib.sha1(("haseen-v3-" + re.sub(r"\s+", " ", t.strip())).encode()).hexdigest()
bucket = lambda t: int(H(t), 16) % 100
MK = markers()
GLF_M = ("SAU", "KHL", "YEM")


def mk_status(T, w):
    n = wkey(w)
    if n not in MK:
        return None
    if T == "GLF":
        st = [MK[n].get(m) for m in GLF_M]
        return "hard" if all(s == "hard" for s in st) else "mixed" if any(s in ("hard", "mixed") for s in st) else st[0]
    return MK[n].get(T)


HUM = json.load(open(f"{ROOT}/data/inventory_v3/target_counts_human_nogold.json"))  # v3.1: train-only counts incl. SADA
HUM["GLF"] = collections.Counter()
for m in GLF_M:
    HUM["GLF"].update(HUM.get(m, {}))
HTOT = {t: max(1, sum(c.values())) for t, c in HUM.items()}


def msa_like(T, toks):
    n = 0
    for w in toks:
        w = norm(w)
        rm = (HUM["MSA"].get(w, 0) + 1) / HTOT["MSA"]
        rt = (HUM[T].get(w, 0) + 1) / HTOT[T]
        n += HUM["MSA"].get(w, 0) >= 3 and rm >= 5 * rt
    return n / max(1, len(toks))


CORE_DIA = set()
for line in open(f"{ROOT}/../HaSeen-V1/data/lexicons/core_v2.tsv"):
    p = line.rstrip("\n").split("\t")
    if not line.startswith(("#", "form\t")) and len(p) >= 2 and p[1][2:3] == "0":
        CORE_DIA.add(norm(p[0]))
COLLOQ_VERB = re.compile(r"^(و|ف)?(ب|ه|ح|عم)(ي|ت|ن|ا)[ء-ي]{2,}$")


def colloquial(toks):
    return any((wkey(w) in CORE_DIA) or (COLLOQ_VERB.match(wkey(w) or "") and HUM["MSA"].get(wkey(w), 0) == 0)
               for w in toks if wkey(w))


pool = {(T, sp): collections.defaultdict(list) for T in ("MSA", "EGY", "GLF", "SAU", "LEV", "IRQ") for sp in ("dev", "test")}
held = set()
# SADA (show + speaker split)
for l in open(f"{S}/sada/sada_clean_v3.jsonl"):
    r = json.loads(l)
    if r["split3"] in ("dev", "test"):
        if r["var"] == "SAU":
            pool[("SAU", r["split3"])][f"sada:{r['sub']}"].append(r["text"])
            pool[("GLF", r["split3"])][f"sada:{r['sub']}"].append(r["text"])
        elif r["var"] in ("KHL", "YEM"):
            pool[("GLF", r["split3"])][f"sada:{r['sub']}"].append(r["text"])
# SCC (Saudi code-switching podcasts): 20% test, 10% dev
for l in open(f"{S}/scc/scc_text.jsonl"):
    t = (json.loads(l).get("Original_text") or "").strip()
    b = bucket(t)
    if b < 30 and t:
        held.add(H(t))
        sp = "test" if b < 20 else "dev"
        pool[("SAU", sp)]["scc"].append(t); pool[("GLF", sp)]["scc"].append(t)
# Egyptian
for l in open(f"{S}/mgb3/mgb3_text.jsonl"):
    r = json.loads(l)
    sp = {"test": "test", "validation": "dev"}.get(r["split"])
    if sp and r.get("text"):
        pool[("EGY", sp)]["mgb3"].append(re.sub(r"\([^)]*\)", " ", r["text"]))
for l in open(f"{S}/masrispeech/masrispeech_text.jsonl", encoding="utf-8"):
    r = json.loads(l)
    t = (r["text"] or "").strip()
    if not t:
        continue
    if r["split"] == "validation":
        pool[("EGY", "test")]["masrispeech"].append(t)
    elif bucket(t) < 4:
        held.add(H(t)); pool[("EGY", "dev")]["masrispeech"].append(t)
# ArabicCulturalQA (QCRI, native post-edited, parallel; assistant-style answers): test -> test, validation -> dev
ACQA_T = {"egyptian": ["EGY"], "levantine": ["LEV"], "gulf": ["GLF"], "msa": ["MSA"]}
for l in open(f"{S}/arabic_cultural_qa/acqa_text.jsonl"):
    r = json.loads(l)
    sp = {"test": "test", "validation": "dev"}.get(r["split"])
    for T in ACQA_T.get(r["dialect"], []):
        if sp:
            pool[(T, sp)]["acqa"].append(r["text"])
# MSA: MGB-2
for l in open(f"{S}/mgb2/mgb2_text.jsonl"):
    r = json.loads(l)
    sp = {"test": "test", "validation": "dev"}.get(r["split"])
    if sp and r.get("transcript"):
        pool[("MSA", sp)]["mgb2"].append(r["transcript"])
# hashed hold-outs from shared human pools (LEV, IRQ, Yemeni part of GLF)
SRC_T = {"curras": "LEV", "nabra": "LEV", "baladi": "LEV", "levanti": "LEV", "lisan_iraqi": "IRQ",
         "iraqi_factlogic_high": "IRQ", "lisan_yemeni": "GLF"}
for l in gzip.open(f"{ROOT}/data/inventory/sentences.jsonl.gz", "rt"):
    r = json.loads(l)
    T = SRC_T.get(r["source"])
    if not T or r.get("kind") != "native":
        continue
    if r["source"] == "nabra" and r.get("sub") in ("رقاوية", "ديرية", "ماردلية"):
        continue
    b = bucket(r["text"])
    if b < 15:
        held.add(H(r["text"]))
        pool[(T, "test" if b < 10 else "dev")][r["source"]].append(r["text"])

WANT = {"test": {"MSA": 800, "EGY": 1200, "GLF": 1500, "SAU": 1200, "LEV": 800, "IRQ": 800},
        "dev": {"MSA": 400, "EGY": 600, "GLF": 800, "SAU": 600, "LEV": 400, "IRQ": 400}}
dropped, stats = collections.Counter(), {}
for (T, sp), srcs in pool.items():
    keys = sorted(srcs)
    for k in keys:
        random.shuffle(srcs[k])
    rows, i, seen = [], 0, set()
    while len(rows) < WANT[sp][T] and any(srcs[k] for k in keys):
        k = keys[i % len(keys)]; i += 1
        if not srcs[k]:
            continue
        sent = re.sub(r"\s+", " ", srcs[k].pop()).strip()
        toks = sent.split()
        if not (3 <= len(toks) <= 40) or sent in seen:
            continue
        if T != "MSA":
            st = [mk_status(T, w) for w in toks]
            if st.count("hard") or st.count("frm") >= 2 or msa_like(T, toks) > 0.30:
                dropped[(T, sp, "msa")] += 1
                continue
            ignore = [int(s == "mixed") for s in st]
        else:
            if colloquial(toks):
                dropped[(T, sp, "colloquial")] += 1
                continue
            ignore = [0] * len(toks)
        seen.add(sent)
        rows.append({"id": f"v3-nat-{T}-{sp}-{len(rows)}", "target": T, "source": k, "tokens": toks,
                     "dev": [0] * len(toks), "label": [None] * len(toks), "ignore": ignore})
    with open(f"{OUT}/natural_{T}_{sp}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats[f"{T}_{sp}"] = {"sentences": len(rows), "words": sum(len(r["tokens"]) for r in rows),
                          "by_source": dict(collections.Counter(r["source"] for r in rows))}
json.dump(sorted(held), open(f"{OUT}/heldout_text_sha1.json", "w"))
json.dump({"natural": stats, "dropped": {str(k): v for k, v in dropped.items()}, "heldout_hashes": len(held)},
          open(f"{OUT}/natural_stats.json", "w"), ensure_ascii=False, indent=1)
for k, v in stats.items():
    print(k, v)
print("dropped", dict(dropped))
