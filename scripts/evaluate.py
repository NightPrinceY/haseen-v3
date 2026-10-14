"""Evaluate HaSeen 3 on HaSeen-3-Bench through the public SDK (the numbers in docs/RESULTS.md).

    python scripts/evaluate.py                       # test split, ONNX int8 on CPU
    python scripts/evaluate.py --split dev --backend torch
    python scripts/evaluate.py --bench path/to/bench --model path/to/model

Bench layout (dataset NightPrince/HaSeen-3-Bench): natural_<T>_<split>.jsonl, challenge_<T>_<split>.jsonl with
tokens, dev (gold deviation per word), label, ignore. Words are passed pre-split, so gold and model align 1:1.
Metrics per target: natural false-alarm rate (share of native words flagged), challenge P / R / F1, and whether the
gold source variety is in the predicted label set."""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import haseen  # noqa: E402
from haseen.decision import NAME, TARGETS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--bench", default=os.path.join(HERE, "..", "data", "bench"))
ap.add_argument("--split", default="test", choices=["dev", "test"])
ap.add_argument("--model", default=None)
ap.add_argument("--backend", default="auto")
ap.add_argument("--out", default="")
a = ap.parse_args()
if not os.path.isdir(a.bench):
    sys.exit(f"no bench at {a.bench}: run `make bench` (private dataset NightPrince/HaSeen-3-Bench, needs HF_TOKEN)")
det = haseen.load(a.model, backend=a.backend)
INV = {v: k for k, v in NAME.items()}


def run(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    for r in rows:
        # one word per gold token: join with spaces; split_words strips edge punctuation, so map by order
        res = det.detect(" ".join(r["tokens"]), r["target"])
        toks = res["tokens"]
        if len(toks) != len(r["tokens"]):  # a token that was pure punctuation: align by skipping it
            keep = [i for i, w in enumerate(r["tokens"]) if any(ch.isalnum() for ch in w)]
            it = iter(toks)
            toks = [next(it) if i in keep else {"conforms": True} for i in range(len(r["tokens"]))]
        yield r, toks


t0 = time.time()
table, out = [], {}
for T in TARGETS:
    fa = n = 0
    for r, toks in run(f"{a.bench}/natural_{T}_{a.split}.jsonl"):
        for i, tk in enumerate(toks):
            if r["ignore"][i]:
                continue
            n += 1
            fa += (not tk["conforms"]) and not r["dev"][i]
    tp = fp = fn = hit = 0
    for r, toks in run(f"{a.bench}/challenge_{T}_{a.split}.jsonl"):
        for i, tk in enumerate(toks):
            if r["ignore"][i]:
                continue
            d, g = not tk["conforms"], bool(r["dev"][i])
            tp += d and g; fp += d and not g; fn += (not d) and g
            if d and g:
                got = [INV[x] for x in tk.get("labels", [])]
                hit += r["label"][i] in got or (r["label"][i] == "SAU" and "GLF" in got)
    P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
    F = 2 * P * R / max(1e-9, P + R)
    out[T] = {"natural_fa": round(fa / max(1, n), 5), "P": round(P, 4), "R": round(R, 4), "F1": round(F, 4),
              "label_set_hit": round(hit / max(1, tp), 4)}
    agent = ""
    ap_ = f"{a.bench}/agent_{T}_{a.split}.jsonl"
    if os.path.exists(ap_):  # LLM replies asked to speak T (test only); word-level P / R / F1
        atp = afp = afn = 0
        for r, toks in run(ap_):
            for i, tk in enumerate(toks):
                if r["ignore"][i]:
                    continue
                d, g = not tk["conforms"], bool(r["dev"][i])
                atp += d and g; afp += d and not g; afn += (not d) and g
        aP, aR = atp / max(1, atp + afp), atp / max(1, atp + afn)
        out[T]["agent_F1"] = round(2 * aP * aR / max(1e-9, aP + aR), 4)
        agent = f"   {out[T]['agent_F1']:.3f}"
    table.append(f"{T:4}  {fa / max(1, n) * 100:6.2f}%  {P:6.3f} {R:6.3f} {F:6.3f}   {hit / max(1, tp):5.2f}{agent}")
print(f"HaSeen 3 · {det.name} · HaSeen-3-Bench {a.split} · {time.time() - t0:.0f}s\n")
print("T     nat FA      P      R     F1   label-set" + ("   agent F1" if any("agent_F1" in o for o in out.values()) else ""))
print("\n".join(table))
print(f"\nmean F1 {sum(o['F1'] for o in out.values()) / len(out):.3f} · targets with natural FA <= 1%: "
      f"{sum(o['natural_fa'] <= 0.01 for o in out.values())}/{len(out)}")
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
