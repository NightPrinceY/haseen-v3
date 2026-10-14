"""Turn a training run into the two release folders, check them, and measure CPU latency.

Usage: python export_release.py --run runs/V6-pointer-s42 --policy runs/V6-pointer-s42/policy_v3_t.json \
           --gold ../data/test_v3 --lexicons ../data/lexicons_v2/release --out release_models --name HaSeen-3
Writes
  <out>/HaSeen-3/       model.safetensors, encoder/config.json, tokenizer files, config.json, calibration.json, lexicons/
  <out>/HaSeen-3-ONNX/  model.int8.onnx, tokenizer.json, config.json, calibration.json, lexicons/, report.json
report.json: decision agreement ONNX int8 vs PyTorch fp32 on every gold dev word, and latency p50/p95 (batch 1,
15 words) for torch fp32 CPU, ORT fp32 CPU and ORT int8 CPU (4 threads)."""
import argparse, glob, json, os, shutil, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from haseen.decision import VARS, decide  # noqa: E402
from haseen.detector import HaSeen2Model  # noqa: E402
from haseen.text import norm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--policy", required=True)
ap.add_argument("--gold", required=True)
ap.add_argument("--lexicons", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--experimental", default="")
ap.add_argument("--name", default="HaSeen-3")
a = ap.parse_args()

cfg = json.load(open(f"{a.run}/config.json"))
pol = json.load(open(a.policy))
cal = {k: pol[k] for k in ("T_var", "T_odia", "tau", "tau_odia", "odia_gate")}
cal["experimental"] = [x for x in a.experimental.split(",") if x]
cal["fitted_on"] = pol.get("fitted_on", "")
tok = AutoTokenizer.from_pretrained(cfg["init"])
m = HaSeen2Model(cfg["init"], cfg["head"], tok)
sd = torch.load(f"{a.run}/best/model.pt", map_location="cpu")
missing, unexpected = m.load_state_dict(sd, strict=False)
assert not unexpected and all(k.startswith("drop") for k in missing), (missing, unexpected)
m.eval()

tdir, odir = f"{a.out}/{a.name}", f"{a.out}/{a.name}-ONNX"
for d in (tdir, odir):
    os.makedirs(d, exist_ok=True)
    shutil.copytree(a.lexicons, f"{d}/lexicons", dirs_exist_ok=True)
    json.dump(cal, open(f"{d}/calibration.json", "w"), indent=1)
    json.dump({"name": a.name if d == tdir else a.name + "-ONNX", "init": cfg["init"], "head": cfg["head"],
               "max_len": cfg["max_len"], "targets": list(VARS), "seed": cfg["seed"], "data": os.path.basename(cfg["data"])},
              open(f"{d}/config.json", "w"), indent=1)
tok.save_pretrained(tdir)
os.makedirs(f"{tdir}/encoder", exist_ok=True)
m.enc.config.save_pretrained(f"{tdir}/encoder")
from safetensors.torch import save_file  # noqa: E402
save_file({k: v.contiguous() for k, v in m.state_dict().items()}, f"{tdir}/model.safetensors")
shutil.copy(f"{tdir}/tokenizer.json", f"{odir}/tokenizer.json")

with torch.no_grad():
    if cfg["head"] != "linear":
        m.options()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids, attention_mask)


w = Wrap(m).eval()
s15 = "حضرتك الطلب بتاعك هيوصل بكرة الصبح بإذن الله ولو فيه أي مشكلة كلمنا على الرقم ده".split()
enc = tok([[norm(x) for x in s15]], is_split_into_words=True, return_tensors="pt")
tmp = f"{a.out}/model.fp32.onnx"
torch.onnx.export(w, (enc["input_ids"], enc["attention_mask"]), tmp, input_names=["input_ids", "attention_mask"],
                  output_names=["variety", "odia", "other"], opset_version=17, dynamo=False,
                  dynamic_axes={k: {0: "b", 1: "t"} for k in ("input_ids", "attention_mask", "variety", "odia", "other")})
import onnxruntime as ort  # noqa: E402
from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: E402
QUANT = {"per_tensor": dict(per_channel=False), "per_channel": dict(per_channel=True)}
for qn, kw in QUANT.items():
    quantize_dynamic(tmp, f"{a.out}/model.{qn}.onnx", weight_type=QuantType.QInt8, **kw)

so = ort.SessionOptions(); so.intra_op_num_threads = 4
S32 = ort.InferenceSession(tmp, so, providers=["CPUExecutionProvider"])
torch.set_num_threads(4)


def lat(fn, n=100):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1000)
    return {"p50": round(float(np.percentile(ts, 50)), 2), "p95": round(float(np.percentile(ts, 95)), 2)}


feeds = {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
rep = {"cpu_threads": 4, "cpu": os.popen("lscpu | grep 'Model name' | head -1").read().split(":")[-1].strip()}
with torch.no_grad():
    rep["latency_torch_fp32_cpu"] = lat(lambda: w(enc["input_ids"], enc["attention_mask"]))
rep["latency_ort_fp32_cpu"] = lat(lambda: S32.run(None, feeds))

sig = lambda x: 1 / (1 + np.exp(-x))


def probs(v, d, o, pos):
    pv = {x: float(sig(v[pos, k] / cal["T_var"])) for k, x in enumerate(VARS)}
    return pv, float(sig(d[pos] / cal["T_odia"])), float(sig(o[pos]))


def agreement(S8):
    agree = n = 0
    for p in sorted(glob.glob(f"{a.gold}/*_dev.jsonl")):
        for line in open(p):
            x = json.loads(line)
            e = tok([[norm(t) for t in x["tokens"]]], is_split_into_words=True, truncation=True, max_length=cfg["max_len"],
                    return_tensors="pt")
            with torch.no_grad():
                vt, dt, ot = [y[0].numpy() for y in w(e["input_ids"], e["attention_mask"])]
            v8, d8, o8 = [y[0] for y in S8.run(None, {"input_ids": e["input_ids"].numpy(),
                                                       "attention_mask": e["attention_mask"].numpy()})]
            seen = set()
            for pos, wid in enumerate(e.word_ids(0)):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                n += 1
                agree += decide(*probs(vt, dt, ot, pos), x["target"], cal)[:2] == decide(*probs(v8, d8, o8, pos), x["target"], cal)[:2]
    return agree / n, n


rep["quantization"] = {}
for qn in QUANT:
    S8 = ort.InferenceSession(f"{a.out}/model.{qn}.onnx", so, providers=["CPUExecutionProvider"])
    ag, n = agreement(S8)
    rep["quantization"][qn] = {"decision_agreement_vs_torch": round(ag, 5), "latency_ort_int8_cpu": lat(lambda: S8.run(None, feeds))}
best = max(QUANT, key=lambda q: rep["quantization"][q]["decision_agreement_vs_torch"])
shutil.move(f"{a.out}/model.{best}.onnx", f"{odir}/model.int8.onnx")
rep["chosen_quantization"] = best
rep["latency_ort_int8_cpu"] = rep["quantization"][best]["latency_ort_int8_cpu"]
rep["int8_vs_torch_decision_agreement"] = rep["quantization"][best]["decision_agreement_vs_torch"]
rep["words_compared"] = n
rep["size_mb"] = {"onnx_int8": round(os.path.getsize(f"{odir}/model.int8.onnx") / 2 ** 20, 1),
                  "safetensors": round(os.path.getsize(f"{tdir}/model.safetensors") / 2 ** 20, 1)}
rep["params"] = sum(p.numel() for p in m.parameters())
json.dump(rep, open(f"{odir}/report.json", "w"), indent=1)
for f in [tmp] + glob.glob(f"{a.out}/model.per_*.onnx"):
    os.remove(f)
print(json.dumps(rep, indent=1))
