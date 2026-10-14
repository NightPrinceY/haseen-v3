"""The default HaSeen runtime: HaSeen-3-ONNX (int8) on CPU with onnxruntime and the `tokenizers` library only.

No PyTorch, no transformers. Agreement with the PyTorch model is reported in docs/RESULTS.md."""
import json, math, os, time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .decision import ARABIC, VARS, Suggester, build_result, resolve_target, split_words
from .text import norm


def _sig(x):
    return 1.0 / (1.0 + math.exp(-x))


class OnnxDetector:
    backend = "onnx"

    def __init__(self, model_dir, threads=None):
        self.model_dir = model_dir
        self.cal = json.load(open(os.path.join(model_dir, "calibration.json")))
        self.cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.enable_truncation(self.cfg.get("max_len", 128))
        self.tok.no_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.int8.onnx"), so, providers=["CPUExecutionProvider"])
        self.suggest = Suggester(os.path.join(model_dir, "lexicons"))
        self.name = self.cfg.get("name", "HaSeen-3-ONNX") + " (int8)"

    def _probs(self, toks):
        arabic = [i for i, (w, _, _) in enumerate(toks) if ARABIC.search(w)]
        if not arabic:
            return {}
        enc = self.tok.encode([norm(toks[i][0]) for i in arabic], is_pretokenized=True)
        ids = np.array([enc.ids], dtype=np.int64)
        v, d, o = self.sess.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        probs, seen = {}, set()
        for pos, wid in enumerate(enc.word_ids):
            if wid is not None and wid not in seen:  # first subword of each word
                seen.add(wid)
                pv = {x: _sig(float(v[0, pos, k]) / self.cal["T_var"]) for k, x in enumerate(VARS)}
                probs[arabic[wid]] = (pv, _sig(float(d[0, pos]) / self.cal["T_odia"]), _sig(float(o[0, pos])))
        return probs

    def detect(self, text, target="egy", return_probs=False):
        t0 = time.perf_counter()
        t = resolve_target(target)
        toks = split_words(text)
        probs = self._probs(toks)
        meta = {"model": self.name, "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "device": "cpu"}
        return build_result(toks, probs, t, self.cal, self.suggest, meta, return_probs)

    def detect_many(self, texts, target="egy", return_probs=False):
        return [self.detect(x, target, return_probs) for x in texts]
