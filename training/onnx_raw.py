"""Raw per-word logits of a released ONNX model on every gold file, in the format of train_v3.py --dump_raw, so that
posthoc_v3.py fits the decision policy (dev) and scores test on exactly the model that ships.

Usage: python onnx_raw.py --model release_models/HaSeen-3-ONNX --gold ../data/test_v3 --out run_dir/raw_gold_onnx.jsonl.gz"""
import argparse, glob, gzip, json, os, sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from haseen.text import norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--gold", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
tok.enable_truncation(json.load(open(os.path.join(a.model, "config.json"))).get("max_len", 128))
tok.no_padding()
so = ort.SessionOptions(); so.intra_op_num_threads = 4
S = ort.InferenceSession(os.path.join(a.model, "model.int8.onnx"), so, providers=["CPUExecutionProvider"])
with gzip.open(a.out, "wt") as f:
    for p in sorted(glob.glob(f"{a.gold}/*.jsonl")):
        name = os.path.basename(p)[:-6]
        for line in open(p):
            x = json.loads(line)
            enc = tok.encode([norm(w) for w in x["tokens"]], is_pretokenized=True)
            ids = np.array([enc.ids], dtype=np.int64)
            v, d, o = S.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
            ps, seen = [None] * len(x["tokens"]), set()
            for pos, wid in enumerate(enc.word_ids):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    ps[wid] = {"v": [float(z) for z in v[0, pos]], "odia": float(d[0, pos]), "other": float(o[0, pos])}
            f.write(json.dumps({"set": name, "id": x["id"], "p": ps}) + "\n")
print("wrote", a.out)
