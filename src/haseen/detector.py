"""HaSeen-3 in PyTorch (pip install "haseen[gpu]"); same decisions as the ONNX runtime, CUDA if present.

A model directory holds config.json (init, head, max_len), calibration.json (decision policy), tokenizer files,
encoder/config.json and model.safetensors (or a training run's best/model.pt)."""
import json, math, os, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .decision import ARABIC, VARS, Suggester, build_result, resolve_target, split_words
from .text import norm

OPTION_TEXT = {"MSA": "كلمة طبيعية في العربية الفصحى", "EGY": "كلمة طبيعية في اللهجة المصرية",
               "SAU": "كلمة طبيعية في اللهجة السعودية", "GLF": "كلمة طبيعية في إحدى لهجات الخليج واليمن",
               "LEV": "كلمة طبيعية في اللهجة الشامية", "IRQ": "كلمة طبيعية في اللهجة العراقية",
               "MAG": "كلمة طبيعية في إحدى لهجات المغرب العربي"}
NV = len(VARS)


class HaSeen2Model(nn.Module):
    """Encoder + naturalness head (pointer or linear, 7 varieties) + other-dialect, non-Arabic and register heads."""

    def __init__(self, init, head, tok, encoder_config=None):
        super().__init__()
        self.head = head
        self.enc = (AutoModel.from_config(encoder_config, add_pooling_layer=False) if encoder_config is not None
                    else AutoModel.from_pretrained(init, add_pooling_layer=False))
        H = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        if head == "linear":
            self.var = nn.Linear(H, NV)
        else:
            self.pw, self.po = nn.Linear(H, 256), nn.Linear(H, 256)
            self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.bias = nn.Parameter(torch.zeros(NV))
            e = tok([OPTION_TEXT[v] for v in VARS], padding=True, return_tensors="pt")
            self.register_buffer("opt_ids", e["input_ids"], persistent=False)
            self.register_buffer("opt_mask", e["attention_mask"], persistent=False)
            self.opt_res = nn.Parameter(torch.zeros(NV, 256))
        self.odia, self.oth, self.frm = nn.Linear(H, 1), nn.Linear(H, 1), nn.Linear(H, NV)
        self._opt = None

    def options(self):
        if self._opt is None or self._opt.device != self.opt_ids.device or self._opt.dtype != self.po.weight.dtype:
            h = self.enc(input_ids=self.opt_ids, attention_mask=self.opt_mask).last_hidden_state
            m = self.opt_mask.unsqueeze(-1).to(h.dtype)
            self._opt = F.normalize(F.normalize(self.po((h * m).sum(1) / m.sum(1)), dim=-1) + self.opt_res, dim=-1)
        return self._opt

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        v = self.var(h) if self.head == "linear" else \
            self.scale.exp() * F.normalize(self.pw(h), dim=-1) @ self.options().T + self.bias
        return v, self.odia(h).squeeze(-1), self.oth(h).squeeze(-1)


class Detector:
    backend = "torch"

    def __init__(self, model_dir, device=None):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        self.cal = json.load(open(os.path.join(model_dir, "calibration.json")))
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        enc_dir = os.path.join(model_dir, "encoder")
        offline = os.path.exists(os.path.join(enc_dir, "config.json"))
        self.tok = AutoTokenizer.from_pretrained(model_dir if offline else cfg["init"])
        self.model = HaSeen2Model(cfg["init"], cfg["head"], self.tok, AutoConfig.from_pretrained(enc_dir) if offline else None)
        st = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(st):
            from safetensors.torch import load_file
            self.model.load_state_dict(load_file(st))
        else:
            self.model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu"))
        self.model.to(self.dev).eval()
        self.max_len = cfg.get("max_len", 128)
        self.name = cfg.get("name", os.path.basename(os.path.normpath(model_dir)))
        self.suggest = Suggester(os.path.join(model_dir, "lexicons"))
        with torch.no_grad():
            if cfg["head"] != "linear":
                self.model.options()

    @torch.no_grad()
    def _probs(self, toks):
        arabic = [i for i, (w, _, _) in enumerate(toks) if ARABIC.search(w)]
        if not arabic:
            return {}
        enc = self.tok([[norm(toks[i][0]) for i in arabic]], is_split_into_words=True, truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        v, d, o = self.model(enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev))
        probs, seen = {}, set()
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None and wid not in seen:
                seen.add(wid)
                pv = {x: torch.sigmoid(v[0, pos, k] / self.cal["T_var"]).item() for k, x in enumerate(VARS)}
                probs[arabic[wid]] = (pv, torch.sigmoid(d[0, pos] / self.cal["T_odia"]).item(), torch.sigmoid(o[0, pos]).item())
        return probs

    def detect(self, text, target="egy", return_probs=False):
        t0 = time.perf_counter()
        t = resolve_target(target)
        toks = split_words(text)
        probs = self._probs(toks)
        meta = {"model": self.name, "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "device": self.dev}
        return build_result(toks, probs, t, self.cal, self.suggest, meta, return_probs)
