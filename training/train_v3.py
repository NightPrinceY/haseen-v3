"""Train HaSeen V3: Arabic encoder + 7 naturalness bits (MSA EGY SAU GLF LEV IRQ MAG) + odia / other /
frm heads. Adapted from V1 training/train.py (same pointer-head fix, masked BCE, dev-only calibration).

Data : data/train_v11/{train,dev}.jsonl (scripts/build_dataset_v10.py with labeler v10; V3 label space)
Gold : data/test_v3/{natural,challenge}_<T>_{dev,test}.jsonl (HaSeen-3-Bench v3.3); the decision policy is fitted
       on its dev split afterwards by posthoc_v3.py (test scored once). `ignore` words are not scored.
Out  : <out>/config.json, log.jsonl, dev_curve.jsonl, best/{model.pt, calibration.json}, metrics.json, preds_*.jsonl
Loss : masked BCE on the 7 bits (unknown -1 -> weight 0; pos weight on "not natural") + BCE(odia) + BCE(other)
       + 0.2 masked BCE(frm, 7).
Decide (target t): see posthoc_v3.py / src/haseen/decision.py (other dialect only if p_t < 0.5; labels = set of
       varieties with p >= 0.5, most likely first; Saudi is never named as a source for target Gulf)."""
import argparse, json, math, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "HaSeen-V1", "src"))
from haseen.text import norm  # noqa: E402

VARS = ("MSA", "EGY", "SAU", "GLF", "LEV", "IRQ", "MAG")  # V3 label space (train_v10)
OPTION_TEXT = {"MSA": "كلمة طبيعية في العربية الفصحى", "EGY": "كلمة طبيعية في اللهجة المصرية",
               "SAU": "كلمة طبيعية في اللهجة السعودية", "GLF": "كلمة طبيعية في إحدى لهجات الخليج واليمن",
               "LEV": "كلمة طبيعية في اللهجة الشامية", "IRQ": "كلمة طبيعية في اللهجة العراقية",
               "MAG": "كلمة طبيعية في إحدى لهجات المغرب العربي"}
NV = len(VARS)

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "train_v10"))
ap.add_argument("--gold", default=os.path.join(HERE, "..", "data", "test_v3"))
ap.add_argument("--heldout_concepts", default=os.path.join(HERE, "..", "data", "lexicons_v2", "heldout_concepts.json"))
ap.add_argument("--init", default="VARabi/ALDi-Token-DI")
ap.add_argument("--head", choices=["pointer", "linear"], default="pointer")
ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=4e-5)
ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--pos_weight", type=float, default=3.0)
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--dev_limit", type=int, default=6000, help="dev rows used during training (all used for calibration)")
ap.add_argument("--limit", type=int, default=0, help="smoke test: N train / N dev rows, gold capped")
ap.add_argument("--out", required=True)
ap.add_argument("--bucket", default="")
ap.add_argument("--dump_raw", action="store_true", help="also write raw per-word logits for every gold file (raw_gold.jsonl.gz)")
ap.add_argument("--eval_only", default="", help="path to a model.pt: skip training, run calibration + gold eval")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)


def sync():
    if a.bucket:
        subprocess.run(["hf", "sync", a.out, a.bucket], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


load = lambda p: [json.loads(l) for l in open(p)]
train, dev_rows = load(f"{a.data}/train.jsonl"), load(f"{a.data}/dev.jsonl")
if a.limit:
    random.shuffle(train); random.shuffle(dev_rows)
    train, dev_rows = train[: a.limit], dev_rows[: max(200, a.limit // 10)]
random.Random(7).shuffle(dev_rows)
dev_small = dev_rows[: a.dev_limit]
HELD = set(json.load(open(a.heldout_concepts))["heldout"])
gold = {}
for T in VARS:
    for part in ("natural", "challenge"):
        for sp in ("dev", "test"):
            p = f"{a.gold}/{part}_{T}_{sp}.jsonl"
            if os.path.exists(p):
                rows = load(p)
                gold[f"{part}_{T}_{sp}"] = rows[:50] if a.limit else rows

json.dump(dict(vars(a), device=dev, torch=torch.__version__, n_train=len(train), n_dev=len(dev_rows),
               option_text=OPTION_TEXT, started=time.strftime("%Y-%m-%d %H:%M:%S")),
          open(f"{a.out}/config.json", "w"), ensure_ascii=False, indent=1)
tok = AutoTokenizer.from_pretrained(a.init)


class HaSeen2(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(a.init, add_pooling_layer=False)
        H = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        if a.head == "linear":
            self.var = nn.Linear(H, NV)
        else:
            self.pw, self.po = nn.Linear(H, 256), nn.Linear(H, 256)
            self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.bias = nn.Parameter(torch.zeros(NV))
            e = tok([OPTION_TEXT[v] for v in VARS], padding=True, return_tensors="pt")
            self.register_buffer("opt_ids", e["input_ids"], persistent=False)
            self.register_buffer("opt_mask", e["attention_mask"], persistent=False)
            q, _ = torch.linalg.qr(torch.randn(256, NV))  # V1 fix: a learned orthogonal vector per option
            self.opt_res = nn.Parameter(q.T.contiguous() * 2.0)
        self.odia, self.oth, self.frm = nn.Linear(H, 1), nn.Linear(H, 1), nn.Linear(H, NV)

    def options(self):
        h = self.enc(input_ids=self.opt_ids, attention_mask=self.opt_mask).last_hidden_state
        m = self.opt_mask.unsqueeze(-1).float()
        return F.normalize(F.normalize(self.po((h * m).sum(1) / m.sum(1)), dim=-1) + self.opt_res, dim=-1)

    def forward(self, ids, mask, opt=None):
        h = self.drop(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state)
        if a.head == "linear":
            v = self.var(h)
        else:
            o = opt if opt is not None else self.options()
            v = self.scale.exp() * F.normalize(self.pw(h), dim=-1) @ o.T + self.bias
        return v, self.odia(h).squeeze(-1), self.oth(h).squeeze(-1), self.frm(h)


def encode(batch, labels=True):
    words = [[norm(w) for w in x["tokens"]] for x in batch]
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
    B, L = enc["input_ids"].shape
    first = torch.zeros(B, L, dtype=torch.bool)
    y = torch.full((B, L, NV), -1.0); yd = torch.zeros(B, L); yo = torch.zeros(B, L); yf = torch.zeros(B, L, NV)
    pos = []
    for b, x in enumerate(batch):
        seen, p = set(), {}
        for i, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid not in seen:
                seen.add(wid); p[wid] = i; first[b, i] = True
                if labels:
                    y[b, i] = torch.tensor(x["bits"][wid], dtype=torch.float)
                    yd[b, i], yo[b, i] = x["odia"][wid], x["other"][wid]
                    yf[b, i] = torch.tensor(x["frm"][wid], dtype=torch.float)
        pos.append(p)
    return enc, first, y, yd, yo, yf, pos


def loss_fn(out, first, y, yd, yo, yf):
    v, d, o, f = out
    known = (y >= 0) & first.unsqueeze(-1)
    w = torch.where(y == 0, a.pos_weight, 1.0) * known
    lv = (F.binary_cross_entropy_with_logits(v, y.clamp(min=0), reduction="none") * w).sum() / known.sum().clamp(min=1)
    fm = first.float()
    ld = (F.binary_cross_entropy_with_logits(d, yd, reduction="none", pos_weight=torch.tensor(5.0, device=d.device)) * fm).sum() / fm.sum()
    lo = (F.binary_cross_entropy_with_logits(o, yo, reduction="none") * fm).sum() / fm.sum()
    lf = (F.binary_cross_entropy_with_logits(f, yf, reduction="none") * fm.unsqueeze(-1)).sum() / (fm.sum() * NV)
    return lv + ld + lo + 0.2 * lf


@torch.no_grad()
def predict(model, rows, bs=64, opt=None):
    model.eval()
    if opt is None and a.head == "pointer":
        opt = model.options()
    out = []
    for k in range(0, len(rows), bs):
        batch = rows[k:k + bs]
        enc, first, *_, pos = encode(batch, labels=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            v, d, o, f = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev), opt)
        v, d, o = v.float().cpu(), d.float().cpu(), o.float().cpu()
        for b, x in enumerate(batch):
            pw = [None] * len(x["tokens"])
            for wid, i in pos[b].items():
                pw[wid] = {"v": v[b, i].tolist(), "odia": float(d[b, i]), "other": float(o[b, i])}
            out.append(pw)
    model.train()
    return out


sig = lambda x: 1 / (1 + math.exp(-x))


def decide(p, t, cal):
    if p is None:  # word cut by max_len
        return False, None, 0.0
    pv = {v: sig(p["v"][k] / cal["T_var"]) for k, v in enumerate(VARS)}
    if sig(p["other"]) > 0.5:
        return False, None, pv[t]
    pd = sig(p["odia"] / cal["T_odia"])
    if pd > cal["tau_odia"]:
        return True, "OTHER_DIA", pd
    if pv[t] < cal["tau"][t]:
        if t != "MSA" and pv["MSA"] >= 0.5:
            return True, "MSA", 1 - pv[t]
        best = max((v for v in VARS if v != t), key=lambda v: pv[v])
        return True, best, 1 - pv[t]
    return False, None, pv[t]


def prf(tp, fp, fn):
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
    return round(P, 4), round(R, 4), round(2 * P * R / max(1e-9, P + R), 4)


def dev_scores(preds, rows, cal, targets=VARS):
    res = {}
    for k, t in enumerate(VARS):
        if t not in targets:
            continue
        tp = fp = fn = 0
        for x, ps in zip(rows, preds):
            for i, p in enumerate(ps):
                g = 1 if x["odia"][i] else (None if x["bits"][i][k] == -1 else int(x["bits"][i][k] == 0))
                if g is None or p is None:
                    continue
                d = decide(p, t, cal)[0]
                tp += d and g; fp += d and not g; fn += (not d) and g
        res[t] = prf(tp, fp, fn)
    res["score"] = round(float(np.mean([res[t][2] for t in targets])), 4)
    return res


DEFAULT_CAL = {"T_var": 1.0, "T_odia": 1.0, "tau": {v: 0.5 for v in VARS}, "tau_odia": 0.5}


def fit_calibration(preds, rows):
    lv, yv, ld, yd = [], [], [], []
    for x, ps in zip(rows, preds):
        for i, p in enumerate(ps):
            if p is None:
                continue
            for k in range(NV):
                if x["bits"][i][k] != -1:
                    lv.append(p["v"][k]); yv.append(float(x["bits"][i][k]))
            ld.append(p["odia"]); yd.append(float(x["odia"][i]))

    def temp(l, y):
        l, y = torch.tensor(l), torch.tensor(y)
        t = torch.zeros(1, requires_grad=True)
        o = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

        def c():
            o.zero_grad(); Lo = F.binary_cross_entropy_with_logits(l / t.exp(), y); Lo.backward(); return Lo
        o.step(c)
        return float(t.exp())
    cal = {"T_var": temp(lv, yv), "T_odia": temp(ld, yd), "tau": {}, "tau_odia": 0.5}
    for t in VARS:
        best = (0, 0.5)
        for tau in np.arange(0.2, 0.91, 0.05):
            c2 = dict(cal, tau={**{v: 0.5 for v in VARS}, t: float(tau)})
            best = max(best, (dev_scores(preds, rows, c2, (t,))[t][2], round(float(tau), 2)))
        cal["tau"][t] = best[1]
    return cal


def ece(preds, rows, cal, bins=10):
    cnt, conf, acc = np.zeros(bins), np.zeros(bins), np.zeros(bins)
    for x, ps in zip(rows, preds):
        for i, p in enumerate(ps):
            if p is None:
                continue
            for k in range(NV):
                if x["bits"][i][k] == -1:
                    continue
                q = sig(p["v"][k] / cal["T_var"]); c = max(q, 1 - q)
                b = min(bins - 1, int(c * bins)); cnt[b] += 1; conf[b] += c; acc[b] += (q >= 0.5) == (x["bits"][i][k] == 1)
    n = cnt.sum()
    return round(float(sum(abs(acc[b] - conf[b]) for b in range(bins) if cnt[b]) / n), 5) if n else None


def gold_eval(name, rows, preds, cal):
    T = rows[0]["target"]
    if name.startswith("natural"):
        fa = n = 0
        for x, ps in zip(rows, preds):
            for i, p in enumerate(ps):
                if x["ignore"][i]:
                    continue
                n += 1
                d = decide(p, T, cal)[0]
                fa += d and not x["dev"][i]
        return {"words": n, "false_alarms": fa, "rate": round(fa / max(1, n), 5)}
    tp = fp = fn = lab_ok = 0
    held_tp = held_n = 0
    for x, ps in zip(rows, preds):
        cmap = dict(zip(x.get("swapped", []), x.get("concepts", [])))
        for i, p in enumerate(ps):
            if x["ignore"][i]:
                continue
            d, lab, _ = decide(p, T, cal)
            g = x["dev"][i]
            tp += d and g; fp += d and not g; fn += (not d) and g
            if d and g:
                lab_ok += (lab == x["label"][i])
            if g and cmap.get(i) in HELD:
                held_n += 1; held_tp += d
    P, R, F1 = prf(tp, fp, fn)
    return {"P": P, "R": R, "F1": F1, "tp": tp, "fp": fp, "fn": fn, "label_acc": round(lab_ok / max(1, tp), 4),
            "heldout_concept_recall": round(held_tp / max(1, held_n), 4), "heldout_n": held_n}


# ------------------------------------------------------------------ training
model = HaSeen2().to(dev)
enc_p = [p for n, p in model.named_parameters() if n.startswith("enc.")]
head_p = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
optim = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": head_p, "lr": a.head_lr}], weight_decay=0.01)
steps = int(math.ceil(len(train) / a.bs) * a.epochs)
sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
log = open(f"{a.out}/log.jsonl", "a"); curve = open(f"{a.out}/dev_curve.jsonl", "a")
best, step, t0 = -1, 0, time.time()
order = list(range(len(train)))
print(f"train {len(train)} rows, {steps} steps, device {dev}", flush=True)
while step < steps and not a.eval_only:
    random.shuffle(order)
    for k in range(0, len(order), a.bs):
        if step >= steps:
            break
        enc, first, y, yd, yo, yf, _ = encode([train[j] for j in order[k:k + a.bs]])
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        loss = loss_fn([o.float() for o in out], first.to(dev), y.to(dev), yd.to(dev), yo.to(dev), yf.to(dev))
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim); scaler.update(); sched.step(); step += 1
        if step % 50 == 0:
            log.write(json.dumps({"step": step, "loss": round(loss.item(), 5), "sec": round(time.time() - t0)}) + "\n"); log.flush()
        if step % a.eval_every == 0 or step == steps:
            sc = dev_scores(predict(model, dev_small), dev_small, DEFAULT_CAL)
            curve.write(json.dumps({"step": step, **sc}) + "\n"); curve.flush()
            print(f"step {step}/{steps} loss {loss.item():.4f} dev {sc} {time.time() - t0:.0f}s", flush=True)
            if sc["score"] > best:
                best = sc["score"]
                os.makedirs(f"{a.out}/best", exist_ok=True)
                torch.save(model.state_dict(), f"{a.out}/best/model.pt")
                json.dump({"step": step, **sc}, open(f"{a.out}/best/dev_at_save.json", "w"))
            sync()

# ------------------------------------------------------------------ calibration (dev) + gold
model.load_state_dict(torch.load(a.eval_only or f"{a.out}/best/model.pt", map_location=dev))
os.makedirs(f"{a.out}/best", exist_ok=True)
pd = predict(model, dev_rows)
cal = fit_calibration(pd, dev_rows)
json.dump(cal, open(f"{a.out}/best/calibration.json", "w"), indent=1)
metrics = {"dev_default": dev_scores(pd, dev_rows, DEFAULT_CAL), "dev_calibrated": dev_scores(pd, dev_rows, cal),
           "ece_dev": ece(pd, dev_rows, cal), "calibration": cal, "gold": {}}
import gzip
raw = gzip.open(f"{a.out}/raw_gold.jsonl.gz", "wt") if a.dump_raw else None
for name, rows in gold.items():
    pg = predict(model, rows)
    if raw:
        for x, ps in zip(rows, pg):
            raw.write(json.dumps({"set": name, "id": x["id"], "p": ps}) + "\n")
    metrics["gold"][name] = gold_eval(name, rows, pg, cal)
    with open(f"{a.out}/preds_{name}.jsonl", "w") as f:
        for x, ps in zip(rows, pg):
            f.write(json.dumps({"id": x["id"], "tokens": x["tokens"], "dev": x["dev"],
                                "decisions": [decide(p, x["target"], cal) for p in ps]}, ensure_ascii=False) + "\n")
if raw:
    raw.close()
s15 = [{"tokens": "حضرتك الطلب بتاعك هيوصل بكرة الصبح بإذن الله ولو فيه أي مشكلة كلمنا على الرقم ده".split()}]
opt_c = None
with torch.no_grad():
    opt_c = model.options() if a.head == "pointer" else None
for _ in range(5):
    predict(model, s15, opt=opt_c)
ts = []
for _ in range(50):
    t1 = time.perf_counter(); predict(model, s15, opt=opt_c); ts.append((time.perf_counter() - t1) * 1000)
metrics["latency_ms_batch1_15w"] = {"p50": round(float(np.percentile(ts, 50)), 2), "p95": round(float(np.percentile(ts, 95)), 2), "device": dev}
metrics["train_seconds"] = round(time.time() - t0)
json.dump(metrics, open(f"{a.out}/metrics.json", "w"), ensure_ascii=False, indent=1)
summ = {k: (v.get("F1"), v.get("R"), v.get("heldout_concept_recall")) if "F1" in v else v.get("rate")
        for k, v in metrics["gold"].items() if k.endswith("test")}
print(json.dumps({"dev_calibrated": metrics["dev_calibrated"], "gold_test": summ}, ensure_ascii=False, indent=1))
sync()
