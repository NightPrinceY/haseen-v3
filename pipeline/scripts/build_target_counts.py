"""Per-target word counts used by markers / gold / labeler (2026-10-14; LEV = PS + JO + LB + SY).
  data/inventory/target_counts_madar_alex_train.json : MADAR Corpus-26/6 + Alexandria TRAIN splits (clean, same domain)
  data/inventory/target_counts_human_nogold.json     : all non-synthetic human text except gold-reserved splits
                                                       and except Casablanca / Habibi / madar6 rows"""
import csv, glob, gzip, json, os, sys, collections
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "HaSeen-V1", "src"))
from haseen.text import norm, words
CITY2T = {"Cairo": "EGY", "Alexandria": "EGY", "Aswan": "EGY", "Riyadh": "SAU", "Jeddah": "SAU", "Doha": "KHL",
          "Jerusalem": "LEV", "Amman": "LEV", "Salt": "LEV", "Beirut": "LEV", "Damascus": "LEV", "Aleppo": "LEV",
          "Baghdad": "IRQ", "Basra": "IRQ", "Mosul": "IRQ", "Sanaa": "YEM", "Rabat": "MOR", "Fes": "MOR",
          "Tunis": "TUN", "Sfax": "TUN", "MSA": "MSA"}
V2T = {"EG": "EGY", "SA": "SAU", "KW": "KHL", "QA": "KHL", "BH": "KHL", "AE": "KHL", "GLF": "KHL", "PS": "LEV",
       "JO": "LEV", "LB": "LEV", "SY": "LEV", "LEV": "LEV", "IQ": "IRQ", "YE": "YEM", "MA": "MOR", "TN": "TUN", "MSA": "MSA"}
clean, tot = collections.defaultdict(collections.Counter), collections.Counter()
for f in glob.glob(f"{ROOT}/data/sources/madar_corpus/*/MADAR_Corpus/MADAR.corpus.*.tsv"):
    c = f.split(".")[-2]
    if c in CITY2T:
        for r in csv.DictReader(open(f), delimiter="\t"):
            if "train" in r["split"].split("-")[-1] or r["split"] == "corpus-6-train":
                for w in words(r["sent"]):
                    clean[CITY2T[c]][norm(w)] += 1; tot[CITY2T[c]] += 1
hum = collections.defaultdict(collections.Counter)
INV = os.environ.get("HASEEN_INV", "inventory")  # v3: HASEEN_INV=inventory_v3
for l in gzip.open(f"{ROOT}/data/{INV}/sentences.jsonl.gz", "rt"):
    r = json.loads(l)
    t = V2T.get(r["var"])
    if not t:
        continue
    if r["source"] == "alexandria" and r["split"] == "train":
        for w in words(r["text"]):
            clean[t][norm(w)] += 1; tot[t] += 1
    if r["kind"] == "synthetic" or r["split"] in ("test", "dev") or r["source"] in ("casablanca", "habibi", "madar6"):
        continue
    for w in words(r["text"]):
        hum[t][norm(w)] += 1
json.dump({t: dict(c) for t, c in clean.items()} | {"_tot": dict(tot)},
          open(f"{ROOT}/data/{INV}/target_counts_madar_alex_train.json", "w"), ensure_ascii=False)
json.dump({t: dict(c) for t, c in hum.items()}, open(f"{ROOT}/data/{INV}/target_counts_human_nogold.json", "w"),
          ensure_ascii=False)
print("clean", dict(tot)); print("human", {t: sum(c.values()) for t, c in hum.items()})
