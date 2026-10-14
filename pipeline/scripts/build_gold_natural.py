"""G1: HaSeen-Bench v2 natural sets (precision), 2026-10-14.

Native sentences of each target from sources / splits never used for training statistics:
  MADAR Corpus-26 dev / test (and Corpus-6 dev for Cairo, Doha, Beirut, Rabat, Tunis), Alexandria dev / test,
  Casablanca transcripts (dialects) and Habibi MSA transcripts -> test only.
Only clearly colloquial sentences are kept: a sentence is dropped when it holds a hard MSA marker for the target or
two or more of its "frm" (formal MSA) function words; Habibi in particular has whole MSA narration / news / sports
sentences. MSA leaks are tested by the challenge sets instead. MSA target: kept as is.
Gold: every word conforms.
A model flag on any other word counts as a false alarm; flagged words are reviewed after the first run and
real leaks are added with evidence (the set then gets a new version).
Output: data/gold_v2/natural_<T>_{dev,test}.jsonl rows {id, target, source, sub, tokens, dev, label}."""
import gzip, json, os, random, re, collections
from gold_common import key as wkey, ROOT, TARGETS, CITY2T, norm, madar_corpus, madar_lexicon, markers

random.seed(2027)
C = madar_corpus()
MK = markers()
OUT = os.path.join(ROOT, "data", "gold_v2")
A2T = {"EG": "EGY", "SA": "SAU", "AE": "KHL", "PS": "LEV", "JO": "LEV", "LB": "LEV", "SY": "LEV", "IQ": "IRQ",
       "YE": "YEM", "MA": "MOR", "TN": "TUN", "MSA": "MSA"}
pool = {(T, s): collections.defaultdict(list) for T in TARGETS for s in ("dev", "test")}
for city, d in C.items():
    T = CITY2T[city]
    if T not in TARGETS:
        continue
    for sid, (split, sent) in d.items():
        sp = "test" if split.endswith("corpus-26-test") else "dev" if split.endswith(("corpus-26-dev", "corpus-6-dev")) else None
        if sp:
            pool[(T, sp)][f"madar:{city}"].append((f"madar-{city}-{sid}", sent))
for l in gzip.open(os.path.join(ROOT, "data", "inventory", "sentences.jsonl.gz"), "rt"):
    r = json.loads(l)
    T = A2T.get(r["var"])
    if not T:
        continue
    if r["source"] == "alexandria" and r["split"] in ("dev", "test"):
        pool[(T, r["split"])][f"alexandria:{r['var']}"].append((None, r["text"]))
    elif r["source"] == "casablanca":
        # Habibi is not used at all: its dialect subsets are narration close to MSA and its "MSA" subset holds
        # colloquial talk-show speech (checked by hand and with the V1 baseline, 2026-10-14)
        # Habibi dialect subsets are mostly narration / documentary / sports Arabic close to MSA (checked by hand
        # on UAE and Iraqi samples), so they only serve the MSA target
        pool[(T, "test")][f"{r['source']}:{r['var']}"].append((None, r["text"]))


HUM = json.load(open(os.path.join(ROOT, "data", "inventory", "target_counts_human_nogold.json")))
HTOT = {t: sum(c.values()) for t, c in HUM.items()}


def msa_like(T, toks):
    """Share of tokens whose rate in human MSA text is >= 5x their rate in the target's human text (add-1)."""
    n = 0
    for w in toks:
        w = norm(w)
        rm = (HUM["MSA"].get(w, 0) + 1) / HTOT["MSA"]
        rt = (HUM[T].get(w, 0) + 1) / HTOT[T]
        n += HUM["MSA"].get(w, 0) >= 3 and rm >= 5 * rt
    return n / len(toks)


_L, BY_CITY = madar_lexicon()
DIA_FORMS, DIA_CITIES = {}, {}  # form -> varieties / cities that list it (MADAR, dialect cities only)
for city, fs in BY_CITY.items():
    if city != "MSA":
        for f in fs:
            DIA_FORMS.setdefault(f, set()).add(CITY2T[city])
            DIA_CITIES.setdefault(f, set()).add(city)
MSA_OK = BY_CITY["MSA"] | {w for w, n in HUM["MSA"].items() if n >= 3} | \
    {norm(l.strip()) for l in open(os.path.join(ROOT, "..", "HaSeen-V1", "data", "lexicons", "msa_stems_camel.txt"))}


CORE_DIA = set()  # V1 core lexicon: forms whose MSA bit is 0 (Egyptian / Gulf / other-dialect markers)
for line in open(os.path.join(ROOT, "..", "HaSeen-V1", "data", "lexicons", "core_v2.tsv")):
    p = line.rstrip("\n").split("\t")
    if line.startswith(("#", "form\t")) or len(p) < 2:
        continue
    if p[1][2:3] == "0":
        CORE_DIA.add(norm(p[0]))
COLLOQ_VERB = re.compile(r"^(و|ف)?(ب|ه|ح|عم)(ي|ت|ن|ا)[ء-ي]{2,}$")


def colloquial(toks):
    """Habibi MSA filter: any V1 dialect marker or a colloquial-prefixed verb not used in human MSA text."""
    for w in toks:
        k = wkey(w)
        if not k:
            continue
        if k in CORE_DIA or (COLLOQ_VERB.match(k) and HUM["MSA"].get(k, 0) == 0 and k not in MSA_OK):
            return True
    return False


def msa_leak(w):
    """MSA target: a word MADAR lists only for dialect cities and real MSA text does not use -> leak label."""
    k = wkey(w)
    # strong evidence only: listed for >= 3 dialect cities and never seen in 809K tokens of human MSA text
    # (rare MSA words such as انعطف / خطف are listed for single cities and would be false leaks)
    if not k or k in MSA_OK or len(DIA_CITIES.get(k, ())) < 3 or HUM["MSA"].get(k, 0) > 0:
        return None
    vs = sorted(DIA_FORMS[k])
    return vs[0] if len(vs) == 1 else "DIA"


def hard(T, w):
    n = wkey(w)
    return T != "MSA" and n in MK and MK[n].get(T) == "hard"


stats = {}
dropped = collections.Counter()
for (T, sp), srcs in pool.items():
    n_want = 300 if sp == "dev" else 400
    keys = sorted(srcs)
    rows, k = [], 0
    for key in keys:
        random.shuffle(srcs[key])
    while len(rows) < n_want and any(srcs[key] for key in keys):
        key = keys[k % len(keys)]
        k += 1
        if not srcs[key]:
            continue
        rid, sent = srcs[key].pop()
        toks = sent.split()
        if not (3 <= len(toks) <= 40):
            continue
        if T != "MSA":
            nh = sum(1 for w in toks if hard(T, w))
            nf = sum(1 for w in toks if wkey(w) in MK and MK[wkey(w)].get(T) == "frm")
            if nh or nf >= 2 or msa_like(T, toks) > 0.30:
                dropped[(T, sp)] += 1
                continue
        lab = [msa_leak(w) if T == "MSA" else None for w in toks]
        if T == "MSA" and key.startswith("habibi") and (any(lab) or colloquial(toks)):
            dropped[(T, sp)] += 1  # Habibi "MSA" includes colloquial interview speech: keep only clean MSA
            continue
        dev = [1 if x else 0 for x in lab]
        rows.append({"id": rid or f"{key}-{len(rows)}", "target": T, "source": key, "tokens": toks, "dev": dev,
                     "label": lab})
    with open(f"{OUT}/natural_{T}_{sp}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats[f"{T}_{sp}"] = {"sentences": len(rows), "words": sum(len(r["tokens"]) for r in rows),
                          "dropped_msa_like": dropped[(T, sp)], "leaks_marked": sum(sum(r["dev"]) for r in rows),
                          "sources": dict(collections.Counter(r["source"].split(":")[0] for r in rows))}
    print(T, sp, stats[f"{T}_{sp}"])
json.dump(stats, open(f"{OUT}/natural_stats.json", "w"), indent=1)
