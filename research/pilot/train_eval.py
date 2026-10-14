"""Pilot: train + evaluate System-1 variants on D1-lite.

Variants
  v1  MARBERTv2            + linear head   (A without warm start)
  v2  ALDi-Token-DI        + linear head   (A: warm start from an MSA/DA token tagger)
  v3  ALDi-Token-DI        + pointer head  (A*: Jev-style, options are text descriptions scored by dot product)
  v4  CAMeLBERT-Mix        + linear head   (backbone comparison)

Metrics
  - deviation P/R/F1 per target (target = sentence matrix variety; deviating = word variety not in {target, SHARED, OTHER})
  - label accuracy on correctly detected deviations
  - token variety accuracy, ECE before/after temperature scaling (fit on dev)
  - same metrics on realish_check.jsonl (Claude-written, Egyptian target)
  - GPU latency, 15-word sentence, batch 1
"""
import json, time, math, argparse, random, statistics, os
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from common import LABELS, L2I, norm

VARIANTS = {
    "v1": ("UBC-NLP/MARBERTv2", "linear"),
    "v2": ("VARabi/ALDi-Token-DI", "linear"),
    "v3": ("VARabi/ALDi-Token-DI", "pointer"),
    "v4": ("CAMeL-Lab/bert-base-arabic-camelbert-mix", "linear"),
}
OPTION_TEXT = {  # pointer head: each option is a short Arabic description, encoded by the same encoder
    "EGY": "كلمة باللهجة المصرية",
    "GLF": "كلمة باللهجة الخليجية",
    "MSA": "كلمة بالعربية الفصحى",
    "SHARED": "كلمة مشتركة بين اللهجات والفصحى",
    "OTHER": "اسم علم أو كلمة أجنبية أو رقم",
}
DEV_SET = {"EGY": {"GLF", "MSA"}, "GLF": {"EGY", "MSA"}, "MSA": {"EGY", "GLF"}}


class Tagger(nn.Module):
    def __init__(self, base, head, tok):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base, add_pooling_layer=False)
        H = self.enc.config.hidden_size
        self.head_type = head
        self.drop = nn.Dropout(0.1)
        if head == "linear":
            self.head = nn.Linear(H, len(LABELS))
        else:
            self.pw = nn.Linear(H, 256)
            self.po = nn.Linear(H, 256)
            self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
            enc = tok([OPTION_TEXT[l] for l in LABELS], padding=True, return_tensors="pt")
            self.register_buffer("opt_ids", enc["input_ids"], persistent=False)
            self.register_buffer("opt_mask", enc["attention_mask"], persistent=False)
        self._opt_cache = None

    def option_vectors(self):
        if self._opt_cache is not None and not self.training:
            return self._opt_cache
        h = self.enc(input_ids=self.opt_ids, attention_mask=self.opt_mask).last_hidden_state
        m = self.opt_mask.unsqueeze(-1).float()
        o = F.normalize(self.po((h * m).sum(1) / m.sum(1)), dim=-1)
        if not self.training:
            self._opt_cache = o
        return o

    def forward(self, input_ids, attention_mask):
        h = self.drop(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        if self.head_type == "linear":
            return self.head(h)
        w = F.normalize(self.pw(h), dim=-1)
        return self.scale.exp() * w @ self.option_vectors().T


def load(path):
    return [json.loads(l) for l in open(path)]


def encode(batch, tok, max_len):
    words = [[norm(w) for w in x["tokens"]] for x in batch]
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
    labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    first = []
    for b, x in enumerate(batch):
        seen, pos = set(), {}
        for i, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid not in seen:
                seen.add(wid); pos[wid] = i
                labels[b, i] = L2I[x["labels"][wid]]
        first.append(pos)
    return enc, labels, first


@torch.no_grad()
def predict(model, data, tok, dev, max_len, bs=64):
    model.eval()
    out = []
    for k in range(0, len(data), bs):
        batch = data[k:k + bs]
        enc, labels, first = encode(batch, tok, max_len)
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev)).float().cpu()
        for b, x in enumerate(batch):
            out.append([(x["labels"][w], logits[b, i]) for w, i in sorted(first[b].items())])
    return out  # per sentence: list of (gold_label, logits)


def fit_temperature(preds):
    logits = torch.stack([lg for s in preds for _, lg in s])
    gold = torch.tensor([L2I[g] for s in preds for g, _ in s])
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logits / t.exp(), gold); loss.backward(); return loss
    opt.step(closure)
    return float(t.exp())


def metrics(preds, data, T=1.0):
    tp = fp = fn = 0; lab_ok = 0; acc = n = 0
    conf, corr = [], []
    per_t = {}
    for s, x in zip(preds, data):
        tgt = x["matrix"]; devs = DEV_SET[tgt]
        c = per_t.setdefault(tgt, [0, 0, 0])
        for g, lg in s:
            p = (lg / T).softmax(-1); k = int(p.argmax()); pl = LABELS[k]
            n += 1; acc += pl == g
            conf.append(float(p[k])); corr.append(pl == g)
            gd, pd = g in devs, pl in devs
            if gd and pd: tp += 1; c[0] += 1; lab_ok += pl == g
            elif pd: fp += 1; c[1] += 1
            elif gd: fn += 1; c[2] += 1
    P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); F1 = 2 * P * R / max(1e-9, P + R)
    # ECE, 15 bins
    bins = [[] for _ in range(15)]
    for cf, cr in zip(conf, corr):
        bins[min(14, int(cf * 15))].append((cf, cr))
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(r for _, r in b) / len(b)) for b in bins if b)
    tf1 = {}
    for t, (a, b, c_) in per_t.items():
        p_, r_ = a / max(1, a + b), a / max(1, a + c_)
        tf1[t] = round(2 * p_ * r_ / max(1e-9, p_ + r_), 4)
    return {"dev_P": round(P, 4), "dev_R": round(R, 4), "dev_F1": round(F1, 4), "label_acc_on_TP": round(lab_ok / max(1, tp), 4),
            "token_acc": round(acc / n, 4), "ECE": round(ece, 4), "F1_by_target": tf1, "n_tokens": n}


def errors(preds, data, k=25):
    out = []
    for s, x in zip(preds, data):
        row = []
        bad = False
        for (g, lg), w in zip(s, x["tokens"]):
            p = lg.softmax(-1); pl = LABELS[int(p.argmax())]
            row.append(f"{w}/{g}" + ("" if pl == g else f"→{pl}({p.max():.2f})"))
            bad |= pl != g
        if bad:
            out.append(" ".join(row))
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--data", default="d1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base, head = VARIANTS[a.variant]
    tok = AutoTokenizer.from_pretrained(base)
    model = Tagger(base, head, tok).to(dev)
    train, dvl, test = (load(f"{a.data}/{s}.jsonl") for s in ("train", "dev", "test"))
    real = load("realish_check.jsonl")
    if a.limit:
        train, dvl, test = train[:a.limit], dvl[:a.limit // 4], test[:a.limit // 4]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    t0 = time.time(); step = 0
    for ep in range(a.epochs):
        random.shuffle(train); model.train()
        for k in range(0, len(train), a.bs):
            enc, labels, _ = encode(train[k:k + a.bs], tok, a.max_len)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            loss = F.cross_entropy(logits.float().view(-1, len(LABELS)), labels.to(dev).view(-1), ignore_index=-100)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); step += 1
            if step % 200 == 0:
                print(f"[{a.variant}] ep{ep} step {step}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    train_s = time.time() - t0
    pd_dev = predict(model, dvl, tok, dev, a.max_len)
    T = fit_temperature(pd_dev)
    pd_test = predict(model, test, tok, dev, a.max_len)
    pd_real = predict(model, real, tok, dev, a.max_len)
    # latency: 15-word sentence, batch 1
    sent = [{"tokens": real[1]["tokens"] + real[2]["tokens"][:6], "labels": real[1]["labels"] + real[2]["labels"][:6], "matrix": "EGY"}]
    enc, _, _ = encode(sent, tok, a.max_len)
    ids, am = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
    model.eval(); ts = []
    with torch.no_grad():
        for i in range(60):
            if dev == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                model(ids, am)
            if dev == "cuda": torch.cuda.synchronize()
            if i >= 10: ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    res = {"variant": a.variant, "base": base, "head": head, "train_sents": len(train), "epochs": a.epochs,
           "train_seconds": round(train_s), "temperature": round(T, 3),
           "test_synthetic": metrics(pd_test, test, 1.0), "test_synthetic_calibrated": metrics(pd_test, test, T),
           "realish_check": metrics(pd_real, real, 1.0), "realish_check_calibrated": metrics(pd_real, real, T),
           "latency_ms": {"device": dev, "p50": round(statistics.median(ts), 2), "p95": round(ts[int(.95 * len(ts)) - 1], 2), "subwords": int(ids.shape[1])},
           "realish_errors": errors(pd_real, real), "test_errors_sample": errors(pd_test, test, 15)}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/{a.variant}.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if "errors" not in k}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
