"""Unify every V2 source into two files (2026-10-14):

  data/inventory/sentences.jsonl.gz  one row per sentence:
      {text, var, sub, source, kind, split}
      var  = country-level variety code (EG SD PS JO LB SY IQ SA KW AE QA BH OM YE LY TN DZ MA MR MSA)
             or a region code when the source only knows the region (LEV GLF MAG)
      sub  = city / sub-dialect when known (e.g. SA:najdi, SY:halabi, PS:nabulsi)
      kind = native (written by speakers) | translated (human translation into the dialect) |
             spoken (human transcript) | synthetic (LLM-written)
  data/inventory/lexicon.jsonl       one row per (form, source, varieties):
      {form, vars, source, concept, msa, weight}
      form = normalized single word (V1 norm); vars = varieties that use it (positive evidence)

Sources: sections 01-14 of research/sources. Varieties that a source names only by region stay at region level."""
import csv, glob, gzip, json, os, re, sys, collections

csv.field_size_limit(10 ** 9)
HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, "..")
V1 = os.path.join(V2, "..", "HaSeen-V1")
sys.path.insert(0, os.path.join(V1, "src"))
from haseen.text import norm, words  # noqa: E402

S = os.path.join(V2, "data", "sources")
S1 = os.path.join(V1, "data", "sources")
OUT = os.path.join(V2, "data", "inventory")
os.makedirs(OUT, exist_ok=True)
ARABIC = re.compile(r"[ء-ي]")
SPLIT = re.compile(r"(?<=[.!?؟\n])\s+|\n+")

stats = collections.Counter()
fo = gzip.open(os.path.join(OUT, "sentences.jsonl.gz"), "wt")


def add(text, var, source, kind, sub="", split=""):
    """Write one sentence (or several, if the text is long) when it is mostly Arabic and 3-60 words."""
    if not text or not isinstance(text, str):
        return
    parts = [text] if len(text.split()) <= 60 else [p for p in SPLIT.split(text) if p]
    for p in parts:
        p = " ".join(p.split())
        w = p.split()
        if not (3 <= len(w) <= 60):
            continue
        ar = sum(1 for x in w if ARABIC.search(x))
        if ar / len(w) < 0.6:
            continue
        fo.write(json.dumps({"text": p, "var": var, "sub": sub, "source": source, "kind": kind, "split": split},
                            ensure_ascii=False) + "\n")
        stats[(source, var, kind)] += 1


def pq(path, cols=None):
    import pandas as pd
    return pd.read_parquet(path, columns=cols)


# ---------------------------------------------------------------- sentences
# Alexandria (UBC-NLP): human translation English -> city dialect, revised by a second native speaker
for f in glob.glob(f"{S}/alexandria/*/*.parquet"):
    d = pq(f)
    sp = os.path.basename(f).split("-")[0]
    for _, r in d.iterrows():
        dia = r["dialect"].lower()
        sub = ("najdi" if "khaleeji" in dia else "hijazi" if "hijazi" in dia else "southern" if "southern" in dia
               else re.sub(r"[^a-z' ]", "", dia).replace(" dialect", "").strip())
        for t in r["dialectal_conversation"]:
            add(t["text"], r["country"], "alexandria", "translated", sub, sp)

# Levanti: real (collected, human translated) vs synthetic (Claude 3.5)
LV = {"Palestinian": "PS", "Syrian": "SY", "Lebanese": "LB", "Jordanian": "JO", "Levantine": "LEV", "Egyptian": "EG"}
with open(f"{S}/levanti/levanti_v1.csv", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        add(r["arabic"], LV.get(r["dialect"], "LEV"), "levanti",
            "synthetic" if r["synthesized"] == "True" else "native")

# Birzeit Currasat raw sentences (native social-media / scripts)
BZ = [("curras", "PS", None), ("baladi", "LB", "baladi/x_Baladi/Baladi/Baladi RowText_sentences.csv"),
      ("nabra", "SY", "nabra/x_Nabra/Nabra/Nabra RowText_sentences.csv"),
      ("lisan_iraqi", "IQ", "lisan_iraqi/Shared/x_Iraqi/Iraqi/Lisan-Iraqi RowText_sentences.csv"),
      ("lisan_libyan", "LY", "lisan_libyan/Shared/x_Libyan/Libyan/Lisan-Libyan RowText_sentences.csv"),
      ("lisan_sudanese", "SD", "lisan_sudanese/x_Sudanese/Sudanese/Lisan-Sudanese RowText_sentences.csv"),
      ("lisan_yemeni", "YE", "lisan_yemeni/x_Yemeni/Yemeni/Lisan-Yemeni RowText_sentences.csv")]
# Nâbra: eastern Syrian sub-dialects (Raqqawi, Deiri, Mardini) are Mesopotamian (qeltu) varieties, not Levantine
NABRA_EAST = {"رقاوية", "ديرية", "ماردلية"}
nabra_sub = {}
for r in csv.DictReader(open(f"{S}/currasat/nabra/x_Nabra/Nabra/Nabra-dataset.csv", encoding="utf-8-sig")):
    nabra_sub[r["sentenceId"]] = r.get("subdialect") or ""
for name, var, path in BZ:
    if path:
        for r in csv.DictReader(open(f"{S}/currasat/{path}", encoding="utf-8-sig")):
            if name == "nabra":
                sub = nabra_sub.get(r["Sentence_id"], "")
                if sub in NABRA_EAST:
                    continue
                add(r["sentence"], var, name, "native", sub)
            else:
                add(r["sentence"], var, name, "native")
seen = set()
for r in csv.DictReader(open(f"{S}/currasat/curras.csv", encoding="utf-8")):
    k = (r["DocID"], r["Sentence_ID"])
    if k not in seen:
        seen.add(k)
        add(r["sentence"], "PS", "curras", "native")

# Iraqi factlogic (YouTube / Telegram comments, ensemble-filtered)
for tier in ["high", "medium", "medium_extended"]:
    for t in pq(glob.glob(f"{S}/iraqi_factlogic/data/{tier}/*.parquet")[0], ["text"])["text"]:
        add(t, "IQ", f"iraqi_factlogic_{tier}", "native")

# Algerian darja corpus (podcast / conversation transcripts, heavy code-switching)
for l in open(f"{S}/algerian_darja/algerian_darja_corpus.jsonl"):
    add(json.loads(l).get("text"), "DZ", "algerian_darja", "spoken")

# Casablanca + Habibi transcripts (spoken)
CB = {"Algeria": "DZ", "Egypt": "EG", "Jordan": "JO", "Mauritania": "MR", "Morocco": "MA", "Palestine": "PS",
      "UAE": "AE", "Yemen": "YE"}
for f in glob.glob(f"{S}/casablanca_text/*.jsonl"):
    c = os.path.basename(f).split("__")[0]
    for l in open(f):
        add(json.loads(l)["transcription"], CB[c], "casablanca", "spoken", split="test")
HB = {"ALG": ("DZ", ""), "EGY": ("EG", ""), "IRQ": ("IQ", ""), "MAR": ("MA", ""), "MSA": ("MSA", ""),
      "SAU-Najdi": ("SA", "najdi"), "SAU-Hijazi": ("SA", "hijazi"), "SAU-Gulf": ("SA", "gulf"), "UAE": ("AE", "")}
for f in glob.glob(f"{S}/habibi_text/*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["dialect"] in HB:
            add(r["text"], HB[r["dialect"]][0], "habibi", "spoken", HB[r["dialect"]][1], "test")

# mo3jam example sentences (crowd, one dialect tag only)
MJ = {"EGA": "EG", "SUA": "SD", "LEA_PS": "PS", "LEA_JO": "JO", "LEA_LB": "LB", "LEA_SY": "SY", "LEA": "LEV",
      "IRA": "IQ", "SAA": "SA", "SAA_HI": "SA", "SAA_NA": "SA", "YEA": "YE", "GAA": "GLF", "GAA_AE": "AE",
      "GAA_BH": "BH", "GAA_KW": "KW", "GAA_OM": "OM", "GAA_QA": "QA", "MAA": "MAG", "MAA_DZ": "DZ", "MAA_LY": "LY",
      "MAA_MA": "MA", "MAA_MR": "MR", "MAA_TN": "TN"}
MJ_SUB = {"SAA_HI": "hijazi", "SAA_NA": "najdi"}
for l in open(f"{S}/mo3jam/definitions.jsonl"):
    d = json.loads(l)
    if len(d["dialects"]) == 1 and d["votes_up"] - d["votes_down"] >= 0:
        ex = (d["defs"].get("ar") or {}).get("example", "")
        for line in ex.split("\n"):
            add(line.split("=")[0], MJ[d["dialects"][0]], "mo3jam_examples", "native", MJ_SUB.get(d["dialects"][0], ""))

# V1 sources
md = pq(f"{S1}/madar6.parquet")
for col, var in [("Cairo", "EG"), ("Doha", "QA"), ("Beirut", "LB"), ("Rabat", "MA"), ("Tunis", "TN"), ("MSA", "MSA")]:
    for t in md[col]:
        add(t, var, "madar6", "translated")
for l in open(f"{S1}/parallel_v1.jsonl"):
    r = json.loads(l)
    src = r["source"]
    if src == "madar6":
        continue
    kind = "native" if src.startswith("dial2msa") else "translated"
    if r.get("EGY"):
        add(r["EGY"], "EG", src, "synthetic" if src.startswith(("oasst2", "nilechat_oasst2")) else kind)
    if r.get("GLF"):
        add(r["GLF"], "SA" if src == "arabic_sts_saudi" else "GLF", src, kind)
    if r.get("DIA"):
        add(r["DIA"], "LEV" if src == "dial2msa_lev" else "MAG", src, kind)
    if r.get("MSA"):
        add(r["MSA"], "MSA", src, "translated")
for l in open(f"{S1}/saudi_conversations.jsonl"):
    for m in json.loads(l)["messages"]:
        add(m["content"], "SA", "saudi_conversations", "synthetic", "najdi")
fo.close()

# ---------------------------------------------------------------- lexicon
lex = open(os.path.join(OUT, "lexicon.jsonl"), "w")
lstats = collections.Counter()


def L(form, vars_, source, concept="", msa="", weight=1.0):
    f = norm((form or "").strip())
    if not f or " " in f or not ARABIC.search(f):
        return
    lex.write(json.dumps({"form": f, "vars": sorted(set(vars_)), "source": source, "concept": concept, "msa": msa,
                          "weight": weight}, ensure_ascii=False) + "\n")
    lstats[source] += 1


CITY = {"RAB": "MA", "FES": "MA", "ALG": "DZ", "TUN": "TN", "SFA": "TN", "TRI": "LY", "BEN": "LY", "CAI": "EG",
        "ALX": "EG", "ASW": "EG", "KHA": "SD", "JER": "PS", "AMM": "JO", "SAL": "JO", "BEI": "LB", "DAM": "SY",
        "ALE": "SY", "MOS": "IQ", "BAG": "IQ", "BAS": "IQ", "DOH": "QA", "MUS": "OM", "RIY": "SA", "JED": "SA",
        "SAN": "YE"}
for r in csv.DictReader(open(f"{S}/madar_lexicon/MADAR_Lexicon_v1.0/MADAR_Lexicon_v1.0.tsv"), delimiter="\t"):
    L(r["CODA"], [CITY[r["Dialect"]]], "madar_lexicon", r["Concept_ID"], r["MSA"],
      1.0 if r["Release Status"] == "READY" else 0.5)
for l in open(f"{S}/mo3jam/definitions.jsonl"):
    d = json.loads(l)
    t = d["term"].split()
    if len(t) == 1:
        net = d["votes_up"] - d["votes_down"]
        L(t[0], [MJ[c] for c in d["dialects"]], "mo3jam", str(d["def_id"]),
          (d["defs"].get("ar") or {}).get("definition", "")[:60], 1.0 if net >= 3 else 0.5 if net >= 0 else 0.1)
LH = {"egyptian": "EG", "cairene": "EG", "saidi": "EG", "levantine": "LEV", "palestinian": "PS", "syrian": "SY",
      "lebanese": "LB", "jordanian": "JO", "iraqi": "IQ", "gulf": "GLF", "najdi": "SA", "hejazi": "SA",
      "kuwaiti": "KW", "emirati": "AE", "bahraini": "BH", "omani": "OM", "yemeni": "YE", "sudanese": "SD",
      "moroccan": "MA", "algerian": "DZ", "tunisian": "TN", "libyan": "LY", "hassaniya": "MR", "maghrebi": "MAG"}
lj = json.load(open(f"{V1}/data/lexicons/raw/lahga.json"))
for w in lj["words"]:
    for e in w["entries"]:
        if e["dialect"] in LH:
            L(e["form"], [LH[e["dialect"]]], "lahga", w["headword"], w["headword"], 0.5 if e["needs_review"] else 1.0)
for d, var in [("egyptian", "EG"), ("levantine", "LEV"), ("darija", "MA")]:
    for r in csv.DictReader(open(f"{S}/arabic-vocab-api/{d}.csv")):
        if var == "EG" and "lcav" in r.get("audioUrl", ""):
            continue
        L(r["arabic"].split("(")[0].split("،")[0].split("/")[0], [var], "lingualism", r["english"])
for name, var, path in BZ:
    if not path:
        rows = csv.DictReader(open(f"{S}/currasat/curras.csv", encoding="utf-8"))
        get = lambda r: (r["CODA_Surface_ar"] or r["Word_Ar"], r["Lemma_PAL_ar"], r["Lemma_MSA_ar"])
    else:
        rows = csv.DictReader(open(f"{S}/currasat/{path.replace(' RowText_sentences', '-dataset')}", encoding="utf-8-sig"))
        get = lambda r: (r.get("Token") or r.get("CODA") or "", r.get("DALemma", ""), r.get("MSALemma", ""))
    for r in rows:
        if name == "nabra" and (r.get("subdialect") or "") in NABRA_EAST:
            continue
        tok, da, msa = get(r)
        L(tok, [var], name, da, msa, 0.3)  # corpus tokens: weak positive evidence (frequency handled later)
lex.close()

json.dump({"sentences": {f"{s}|{v}|{k}": n for (s, v, k), n in sorted(stats.items())},
           "lexicon": dict(lstats)}, open(os.path.join(OUT, "build_stats.json"), "w"), indent=1, ensure_ascii=False)
print(sum(stats.values()), "sentences;", sum(lstats.values()), "lexicon rows")
