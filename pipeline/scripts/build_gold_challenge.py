"""G2: HaSeen-Bench v2 challenge sets (2026-10-14). No annotator needed; every deviation comes from experts.

For target T and a MADAR Corpus-26 sentence id (dev split -> dev set, test split -> test set):
  host   = the sentence as written by a native translator from a city of T
  source = the same sentence from a city of another variety (another target, MSA, or an "other dialect" city)
A host word w and a source word v are swapped only when the MADAR Lexicon lists w for (concept C, host city) and
v for (C, source city), and v is never listed for any city of T (in any concept) and is not used in T's native
training text. So v is the other variety's word for the same meaning, in the same slot: a deviation with an exact
position and a known label. 1-3 swaps per sentence, one source per sentence.
Original host words that are hard MSA markers for T (markers.tsv) are also gold deviations (true leaks).
Output: data/gold_v2/challenge_<T>_{dev,test}.jsonl  rows {id, target, host_city, source_city, source_var,
tokens, dev[0/1], label[str|None], swapped[pos...]}."""
import json, os, random, collections
from gold_common import key, ROOT, TARGETS, CITY2T, stems, norm, words, madar_corpus, madar_lexicon, markers, target_counts

random.seed(2026)
C = madar_corpus()
L, by_city = madar_lexicon()
MK = markers()
CNT, TOT = target_counts()
OUT = os.path.join(ROOT, "data", "gold_v2")
os.makedirs(OUT, exist_ok=True)
cities_of = collections.defaultdict(list)
for c, t in CITY2T.items():
    cities_of[t].append(c)
form_concepts = collections.defaultdict(lambda: collections.defaultdict(set))  # city -> form -> concepts
for cid, d in L.items():
    for city, fs in d.items():
        for f in fs:
            form_concepts[city][f].add(cid)


def t_forms(T):
    s = set()
    for c in cities_of[T]:
        s |= by_city[c]
    return s


def used_in_T(T, tok):
    """True if T's native train text uses this form (>= 2 per 100K tokens) or MADAR lists it for T."""
    n = norm(tok)
    if T == "MSA":
        # valid MSA = MADAR MSA forms, the clean MADAR MSA translations, or a CALIMA-MSA stem (an independent
        # morphological lexicon; with a pronoun suffix removed). Homographs (هون، نص) are dropped on purpose:
        # gold keeps only words that cannot be MSA.
        S = stems(tok) | {SUFFIX.sub("", x) for x in stems(tok)}
        return CNT["MSA"].get(n, 0) > 0 or bool(S & by_city["MSA"]) or bool(S & CALIMA_MSA)
    return bool(stems(tok) & (TF[T] | CROWD[T])) or CNT[T].get(n, 0) / TOT[T] * 1e5 >= 2 \
        or any(HUM.get(T, {}).get(x, 0) >= 2 for x in stems(tok))


TF = {T: t_forms(T) for T in TARGETS}
import re as _re
SUFFIX = _re.compile(r"(ي|ك|ه|ها|نا|كم|هم|كما|هما|ني)$")
CALIMA_MSA = {norm(l.strip()) for l in open(os.path.join(ROOT, "..", "HaSeen-V1", "data", "lexicons", "msa_stems_camel.txt"))}
HUM = json.load(open(os.path.join(ROOT, "data", "inventory", "target_counts_human_nogold.json")))
CROWD = collections.defaultdict(set)  # target -> forms from voted crowd lexicons (mo3jam votes >= 3, Lahga reviewed)
_V2T = {"EG": "EGY", "SA": "SAU", "KW": "KHL", "QA": "KHL", "BH": "KHL", "AE": "KHL", "GLF": "KHL", "PS": "LEV",
        "JO": "LEV", "LB": "LEV", "SY": "LEV", "LEV": "LEV", "IQ": "IRQ", "YE": "YEM", "MA": "MOR", "TN": "TUN"}
for l in open(os.path.join(ROOT, "data", "inventory", "lexicon.jsonl")):
    r = json.loads(l)
    if r["source"] in ("mo3jam", "lahga", "lingualism") and r["weight"] >= 1.0:
        for v in r["vars"]:
            for t in _V2T.get(v, "").split("|"):
                if t:
                    CROWD[t].add(r["form"])
# MSA-valid = MADAR MSA concept forms + words used >= 3 times in 809K tokens of human MSA text. (CALIMA-MSA stems
# were tried and rejected: as bare stems they match dialect homographs such as تنجم "can" / "results from".)
MSA_OK = set(by_city["MSA"]) | {w for w, n in HUM["MSA"].items() if n >= 3}
CLASS = {}
import csv as _csv
for r in _csv.DictReader(open(f"{ROOT}/data/sources/madar_lexicon/MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0.tsv"), delimiter="\t"):
    CLASS[r["Concept_ID"]] = r["Class"]


def msa_valid(v):
    return bool(stems(v) & MSA_OK)


def pairs(host_city, host_toks, src_city, src_toks, T):
    out = []
    for i, w in enumerate(host_toks):
        for f in stems(w) & set(form_concepts[host_city]):
            for cid in form_concepts[host_city][f]:
                for j, v in enumerate(src_toks):
                    if norm(v) == norm(w) or not (stems(v) & L[cid].get(src_city, set())):
                        continue
                    if used_in_T(T, v):
                        continue
                    if T != "MSA" and msa_valid(v):
                        # formal open-class words are shared in the service register: only hard markers or
                        # MSA closed-class forms (numbers, pronouns, particles) count as MSA deviations
                        nv = norm(v)
                        if not ((nv in MK and MK[nv].get(T) == "hard") or (src_city == "MSA" and CLASS.get(cid) == "ClosedClass")):
                            continue
                    out.append((i, j, cid))
    seen_i, seen_j, res = set(), set(), []
    for i, j, cid in out:
        if i not in seen_i and j not in seen_j:
            seen_i.add(i); seen_j.add(j); res.append((i, j, cid))
    return res


def hard_dev(T, tok):
    if T == "MSA":
        return False
    n = key(tok)
    if n in MK and MK[n].get(T) == "hard":
        return True
    # سـ + imperfect: 3+ letters after سي/ست/سن/سا and not a known form of T
    return len(n) >= 5 and n[0] == "س" and n[1] in "يتنا" and n not in TF[T] and CNT[T].get(n, 0) == 0 \
        and CNT["MSA"].get(n, 0) > 0


def src_label(city):
    return CITY2T[city]


stats = {}
for T in TARGETS:
    for split in ("dev", "test"):
        ids = sorted(i for i in C["Cairo"] if C["Cairo"][i][0].endswith(f"corpus-26-{split}"))
        others = [c for c in C if CITY2T[c] != T]
        rows, used = [], collections.Counter()
        form_n = collections.Counter()  # diversity: one swapped form at most 4 times per set
        want = 300
        for rnd in range(16):
            random.shuffle(ids)
            for k, sid in enumerate(ids):
                if len(rows) >= want:
                    break
                host = cities_of[T][(k + rnd) % len(cities_of[T])]
                if sid not in C[host]:
                    continue
                htoks = C[host][sid][1].split()
                # source mix: ~25% MSA, ~15% other dialect, rest other targets (MSA target: dialects only)
                r = random.random()
                if T != "MSA" and r < 0.25:
                    pool = ["MSA"]
                elif r < 0.40:
                    pool = [c for c in others if CITY2T[c] == "OTHER_DIA"]
                else:
                    pool = [c for c in others if CITY2T[c] not in ("OTHER_DIA", "MSA")]
                random.shuffle(pool)
                for src in pool:
                    if (sid, host, src) in used or sid not in C[src]:
                        continue
                    stoks = C[src][sid][1].split()
                    pr = pairs(host, [norm(w) for w in htoks], src, stoks, T)
                    if not pr:
                        continue
                    pr = [p for p in pr if form_n[norm(stoks[p[1]])] < 4]
                    if not pr:
                        continue
                    pr = random.sample(pr, min(len(pr), random.choice([1, 1, 2, 2, 3])))
                    for _, j, _ in pr:
                        form_n[norm(stoks[j])] += 1
                    toks = list(htoks)
                    dev = [1 if hard_dev(T, w) else 0 for w in toks]
                    lab = ["MSA" if d else None for d in dev]
                    for i, j, cid in pr:
                        toks[i] = stoks[j]
                        dev[i] = 1
                        lab[i] = src_label(src)
                    used[(sid, host, src)] += 1
                    rows.append({"id": f"ch-{T}-{split}-{sid}-{host}-{src}", "target": T, "host_city": host,
                                 "source_city": src, "source_var": src_label(src), "tokens": toks, "dev": dev,
                                 "label": lab, "swapped": sorted(i for i, _, _ in pr), "concepts": [c for _, _, c in pr]})
                    break
        with open(f"{OUT}/challenge_{T}_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sw = sum(len(r["swapped"]) for r in rows)
        leaks = sum(sum(r["dev"]) for r in rows) - sw
        stats[f"{T}_{split}"] = {"sentences": len(rows), "swaps": sw, "native_hard_leaks": leaks,
                                 "sources": dict(collections.Counter(r["source_var"] for r in rows))}
        print(T, split, stats[f"{T}_{split}"])
json.dump(stats, open(f"{OUT}/challenge_stats.json", "w"), indent=1)
