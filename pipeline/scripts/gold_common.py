"""Shared resources for building HaSeen-Bench v2 (gold): targets, MADAR corpus / lexicon indexes, marker statuses."""
import csv, glob, json, os, re, sys, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "HaSeen-V1", "src"))
from haseen.text import norm, words  # noqa: E402,F401

TARGETS = ["MSA", "EGY", "SAU", "KHL", "LEV", "IRQ", "YEM", "MOR", "TUN"]
CITY2T = {"Cairo": "EGY", "Alexandria": "EGY", "Aswan": "EGY", "Riyadh": "SAU", "Jeddah": "SAU", "Doha": "KHL",
          "Jerusalem": "LEV", "Amman": "LEV", "Salt": "LEV", "Beirut": "LEV", "Damascus": "LEV",
          "Aleppo": "LEV", "Baghdad": "IRQ", "Basra": "IRQ", "Mosul": "IRQ", "Sanaa": "YEM", "Rabat": "MOR",
          "Fes": "MOR", "Tunis": "TUN", "Sfax": "TUN", "MSA": "MSA",
          "Khartoum": "OTHER_DIA", "Tripoli": "OTHER_DIA", "Benghazi": "OTHER_DIA", "Algiers": "OTHER_DIA",
          "Muscat": "OTHER_DIA"}
CODE2CITY = {"ALE": "Aleppo", "ALG": "Algiers", "ALX": "Alexandria", "AMM": "Amman", "ASW": "Aswan", "BAG": "Baghdad",
             "BAS": "Basra", "BEI": "Beirut", "BEN": "Benghazi", "CAI": "Cairo", "DAM": "Damascus", "DOH": "Doha",
             "FES": "Fes", "JED": "Jeddah", "JER": "Jerusalem", "KHA": "Khartoum", "MOS": "Mosul", "MUS": "Muscat",
             "RAB": "Rabat", "RIY": "Riyadh", "SAL": "Salt", "SAN": "Sanaa", "SFA": "Sfax", "TRI": "Tripoli",
             "TUN": "Tunis"}
PROCLITICS = re.compile(r"^(وال|فال|بال|كال|لل|وب|ول|فب|فل|وك|ال|و|ف|ب|ل|ك)(?=..)")


def key(tok):
    """Normalized word without edge punctuation ('ماذا؟' -> 'ماذا'); '' if nothing Arabic is left."""
    w = words(tok)
    return norm(w[0]) if w else ""


def stems(tok):
    """Normalized token and its proclitic-stripped stem(s)."""
    t = key(tok) or norm(tok)
    out = {t}
    m = PROCLITICS.match(t)
    if m:
        out.add(t[m.end():])
    return out


def madar_corpus():
    """{city: {sentID: (split, sentence)}} for Corpus-26 rows (+ Corpus-6 rows for the 5 cities)."""
    C = collections.defaultdict(dict)
    for f in glob.glob(f"{ROOT}/data/sources/madar_corpus/*/MADAR_Corpus/MADAR.corpus.*.tsv"):
        city = f.split(".")[-2]
        if city in CITY2T:
            for r in csv.DictReader(open(f), delimiter="\t"):
                C[city][r["sentID.BTEC"]] = (r["split"], r["sent"])
    return C


def madar_lexicon():
    """concept -> city -> set(normalized single-word forms); READY rows only. Also form -> set(cities)."""
    L = collections.defaultdict(lambda: collections.defaultdict(set))
    msa = collections.defaultdict(set)
    for r in csv.DictReader(open(f"{ROOT}/data/sources/madar_lexicon/MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0.tsv"),
                           delimiter="\t"):
        if r["Release Status"] != "READY":
            continue
        f = norm(r["CODA"].strip())
        if f and " " not in f:
            L[r["Concept_ID"]][CODE2CITY[r["Dialect"]]].add(f)
        for m in r["MSA"].replace("،", ",").split(","):
            m = norm(m.strip())
            if m and " " not in m and "+" not in m and "#" not in m:
                msa[r["Concept_ID"]].add(m)
    for c, ms in msa.items():
        L[c]["MSA"] |= ms
    by_city = collections.defaultdict(set)
    for c in L:
        for city, fs in L[c].items():
            by_city[city] |= fs
    return L, by_city


def markers():
    """form -> {target: status}; MSA is 'shared' for all of its own forms."""
    out = {}
    for line in open(f"{ROOT}/data/lexicons_v2/markers.tsv"):
        if line.startswith(("#", "form\t")):
            continue
        p = line.rstrip("\n").split("\t")
        T = ["EGY", "SAU", "KHL", "LEV", "IRQ", "YEM", "MOR", "TUN"]
        out[p[0]] = {t: p[3 + i].split(":")[0] for i, t in enumerate(T)}
    return out


def target_counts():
    C = json.load(open(f"{ROOT}/data/inventory/target_counts_madar_alex_train.json"))
    return C, C.pop("_tot")
