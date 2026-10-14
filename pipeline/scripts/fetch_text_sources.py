"""Fetch the text parts of HF dialect datasets found in the V2 source search (2026-10-14).

Small text datasets are downloaded as files. Speech datasets (Casablanca, Habibi) are large because of audio, so
only their transcripts are pulled, row by row, through the public datasets-server /rows API (100 rows per call).
Output: data/sources/<name>/...  Re-runnable: finished files are skipped."""
import json, os, time, urllib.parse, urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "sources")
DS = "https://datasets-server.huggingface.co"


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "HaSeen-research"}), timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            print("retry", k, url[:120], e, flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError(url)


def files(repo, out, keep=lambda p: True):
    sib = json.loads(get(f"https://huggingface.co/api/datasets/{repo}"))["siblings"]
    for s in sib:
        p = s["rfilename"]
        if not keep(p):
            continue
        dst = os.path.join(ROOT, out, p)
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(get(f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(p)}"))
        print("got", repo, p, flush=True)


def transcripts(repo, out, cols):
    info = json.loads(get(f"{DS}/splits?dataset={repo}"))["splits"]
    for s in info:
        dst = os.path.join(ROOT, out, f"{s['config']}_{s['split']}.jsonl")
        if os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        rows, off = [], 0
        while True:
            q = f"{DS}/rows?dataset={repo}&config={s['config']}&split={s['split']}&offset={off}&length=100"
            d = json.loads(get(q))
            rows += [{c: r["row"].get(c) for c in cols} for r in d["rows"]]
            off += 100
            if off >= d["num_rows_total"]:
                break
            time.sleep(0.3)
        with open(dst + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(dst + ".tmp", dst)
        print("got", repo, s["config"], s["split"], len(rows), flush=True)


if __name__ == "__main__":
    files("factlogic/iraqi-arabic-dialect-corpus", "iraqi_factlogic", lambda p: p.endswith((".parquet", ".md")))
    files("touati-kamel/algerian-darja-corpus", "algerian_darja", lambda p: not p.startswith("."))
    transcripts("UBC-NLP/Casablanca", "casablanca_text", ["seg_id", "transcription", "gender", "duration"])
    transcripts("SWivid/Habibi", "habibi_text", ["text", "dialect", "speaker_id", "duration"])
    print("done", flush=True)
