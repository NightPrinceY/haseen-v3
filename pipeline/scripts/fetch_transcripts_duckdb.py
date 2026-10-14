"""Transcripts only from speech datasets (Casablanca, Habibi): DuckDB reads just the text columns of the remote
parquet files with HTTP range requests, so the audio is never downloaded. Output: data/sources/<name>/*.jsonl"""
import duckdb, json, os, urllib.request

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
for repo, out, cols in [("UBC-NLP/Casablanca", "casablanca_text", "seg_id, transcription, gender, duration"),
                        ("SWivid/Habibi", "habibi_text", "text, dialect, speaker_id, duration")]:
    sib = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{repo}"))["siblings"]
    os.makedirs(f"data/sources/{out}", exist_ok=True)
    for f in [s["rfilename"] for s in sib if s["rfilename"].endswith(".parquet")]:
        dst = f"data/sources/{out}/" + f.replace("/", "__").replace(".parquet", ".jsonl")
        if os.path.exists(dst):
            continue
        rows = con.execute(f"SELECT {cols} FROM read_parquet('https://huggingface.co/datasets/{repo}/resolve/main/{f}')").fetchall()
        names = [c.strip() for c in cols.split(",")]
        with open(dst + ".tmp", "w") as o:
            for r in rows:
                o.write(json.dumps(dict(zip(names, r)), ensure_ascii=False) + "\n")
        os.replace(dst + ".tmp", dst)
        print(repo, f, len(rows), flush=True)
print("done", flush=True)
