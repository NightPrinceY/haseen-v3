"""Inventory v3 = inventory v2 minus everything the frozen V3 test holds out, plus the new human sources (train parts).

Removed (data/test_v3/MANIFEST.json "exclusions_for_v3_training"):
  - sentences whose hash is in data/test_v3/heldout_text_sha1.json (MasriSpeech dev, SCC, Curras / Nâbra / Baladi /
    Levanti-native / Lisan / factlogic hold-outs)
  - mo3jam probe entries: their example sentences (hosts) and their lexicon rows
Added (train parts only):
  SADA (shows with split3 = train), MasriSpeech train (minus held-out hashes), MGB-3 train, SDAIA SCC (not held out),
  QCRI ArabicCulturalQA train (egyptian, levantine, gulf, msa, maghrebi; parallel ids kept in `pid`)
Output: data/inventory_v3/{sentences.jsonl.gz, lexicon.jsonl, build_stats.json}"""
import collections, gzip, hashlib, json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "HaSeen-V1", "src"))
from haseen.text import norm  # noqa: E402

S, T3 = f"{ROOT}/data/sources", f"{ROOT}/data/test_v3"
OUT = f"{ROOT}/data/inventory_v3"
os.makedirs(OUT, exist_ok=True)
H = lambda t: hashlib.sha1(("haseen-v3-" + re.sub(r"\s+", " ", t.strip())).encode()).hexdigest()
HELD = set(json.load(open(f"{T3}/heldout_text_sha1.json")))
PROBE = set(json.load(open(f"{T3}/probe_heldout_def_ids.json")))
PERSIAN = str.maketrans({"ی": "ي", "ک": "ك"})
# probe example sentences (as the v2 inventory stored them) and probe forms
probe_ex, probe_forms = set(), set()
for l in open(f"{S}/mo3jam/definitions.jsonl"):
    d = json.loads(l)
    if d["def_id"] in PROBE:
        ex = (d["defs"].get("ar") or {}).get("example", "")
        for line in ex.split("\n"):
            t = line.split("=")[0].strip()
            if t:
                probe_ex.add(re.sub(r"\s+", " ", t))
        probe_forms.add((norm(re.sub(r"[^\w]", "", d["term"].translate(PERSIAN))), str(d["def_id"])))
stats = collections.Counter()
fo = gzip.open(f"{OUT}/sentences.jsonl.gz", "wt")


def add(text, var, source, kind, sub="", split="", pid=""):
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text.split()) < 3 or not re.search(r"[ء-ي]", text):
        return
    fo.write(json.dumps({"text": text, "var": var, "sub": sub, "source": source, "kind": kind, "split": split,
                         "pid": pid}, ensure_ascii=False) + "\n")
    stats[(source, var)] += 1


for l in gzip.open(f"{ROOT}/data/inventory/sentences.jsonl.gz", "rt"):
    r = json.loads(l)
    if H(r["text"]) in HELD:
        stats["drop_heldout_hash"] += 1; continue
    if r["source"] == "mo3jam_examples" and re.sub(r"\s+", " ", r["text"].strip()) in probe_ex:
        stats["drop_probe_example"] += 1; continue
    fo.write(json.dumps({**r, "pid": r.get("pid", "")}, ensure_ascii=False) + "\n")
    stats[("v2", r["source"])] += 1
# SADA train shows (speaker label -> variety)
SADA_V = {"SAU": "SA", "KHL": "GLF", "YEM": "YE", "EGY": "EG", "LEV": "LEV", "MSA": "MSA", "IRQ": "IQ", "MAG": "MAG"}
for l in open(f"{S}/sada/sada_clean_v3.jsonl"):
    r = json.loads(l)
    if r["split3"] == "train":
        add(r["text"], SADA_V[r["var"]], "sada", "spoken", r["sub"].lower(), pid="sada:" + r["doc"])
for l in open(f"{S}/masrispeech/masrispeech_text.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["split"] == "train" and r["text"] and H(r["text"]) not in HELD:
        add(r["text"], "EG", "masrispeech", "spoken", pid=f"masri:{r['file']}")
for l in open(f"{S}/mgb3/mgb3_text.jsonl"):
    r = json.loads(l)
    if r["split"] == "train" and r.get("text"):
        add(re.sub(r"\([^)]*\)", " ", r["text"]), "EG", "mgb3", "spoken", pid="mgb3:" + r["id"].rsplit("_", 2)[0])
for l in open(f"{S}/scc/scc_text.jsonl"):
    t = (json.loads(l).get("Original_text") or "").strip()
    if t and H(t) not in HELD:
        add(t, "SA", "scc", "spoken", pid="scc")
ACQA_V = {"egyptian": "EG", "levantine": "LEV", "gulf": "GLF", "msa": "MSA", "maghrebi": "MAG"}
for l in open(f"{S}/arabic_cultural_qa/acqa_text.jsonl"):
    r = json.loads(l)
    if r["split"] == "train" and r["dialect"] in ACQA_V:
        add(r["text"], ACQA_V[r["dialect"]], "acqa", "translated", pid="acqa:" + r["pid"])
fo.close()
# lexicon: drop the probe entries' own rows (mo3jam rows carry the def_id as concept)
n = 0
with open(f"{OUT}/lexicon.jsonl", "w") as f:
    for l in open(f"{ROOT}/data/inventory/lexicon.jsonl"):
        r = json.loads(l)
        if r["source"] == "mo3jam" and (r["form"], str(r.get("concept", ""))) in probe_forms:
            stats["drop_probe_lexicon_row"] += 1; continue
        f.write(l); n += 1
json.dump({str(k): v for k, v in stats.items()}, open(f"{OUT}/build_stats.json", "w"), ensure_ascii=False, indent=1)
for k in sorted(stats, key=str):
    if not (isinstance(k, tuple) and k[0] == "v2"):
        print(k, stats[k])
print("lexicon rows kept", n)
