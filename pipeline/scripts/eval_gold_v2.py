"""Evaluate a detector on HaSeen-Bench v2 (2026-10-14).

  python scripts/eval_gold_v2.py --model v1 [--split dev|test] [--targets EGY,MSA,SAU,KHL]
Metrics per target: natural = false alarms / words (precision side); challenge = word-level P / R / F1 on
deviations. Writes data/gold_v2/eval_<model>_<split>.json and a review list of every natural false alarm
(candidates for gold fixes: real leaks get added with evidence and a new gold version).
Model "v1" = HaSeen V1 (installed `haseen` package, ONNX); targets map EGY->egy, MSA->msa, SAU/KHL->glf."""
import argparse, json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
G = os.path.join(ROOT, "data", "gold_v2", "frozen")


def spans(tokens):
    out, pos = [], 0
    for t in tokens:
        out.append((pos, pos + len(t)))
        pos += len(t) + 1
    return out


def v1_predictor():
    import haseen
    det = haseen.load(quiet=True)
    MAP = {"EGY": "egy", "MSA": "msa", "SAU": "glf", "KHL": "glf"}

    def pred(tokens, T):
        text = " ".join(tokens)
        res = det.detect(text, MAP[T])
        flags = [0] * len(tokens)
        sp = spans(tokens)
        for tk in res["tokens"]:
            if not tk["conforms"]:
                for i, (a, b) in enumerate(sp):
                    if a <= tk["start"] < b:
                        flags[i] = 1
        return flags
    return pred, list(MAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--targets", default=None)
    a = ap.parse_args()
    pred, targets = v1_predictor() if a.model == "v1" else (None, None)
    if a.targets:
        targets = a.targets.split(",")
    out, review = {}, []
    for T in targets:
        res = {}
        nat = [json.loads(l) for l in open(f"{G}/natural_{T}_{a.split}.jsonl")]
        fa = words = 0
        for r in nat:
            f = pred(r["tokens"], T)
            for i, (p, g) in enumerate(zip(f, r["dev"])):
                if r["ignore"][i]:
                    continue
                words += 1
                if p and g:
                    continue
                if p and not g:
                    fa += 1
                    review.append({"target": T, "id": r["id"], "word": r["tokens"][i], "i": i,
                                   "sentence": " ".join(r["tokens"])})
        res["natural"] = {"sentences": len(nat), "words": words, "false_alarms": fa, "rate": fa / max(words, 1)}
        ch = [json.loads(l) for l in open(f"{G}/challenge_{T}_{a.split}.jsonl")]
        tp = fp = fn = 0
        for r in ch:
            f = pred(r["tokens"], T)
            for p, g, ig in zip(f, r["dev"], r["ignore"]):
                if ig:
                    continue
                tp += p and g; fp += p and not g; fn += (not p) and g
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res["challenge"] = {"sentences": len(ch), "tp": tp, "fp": fp, "fn": fn, "P": P, "R": R,
                            "F1": 2 * P * R / max(P + R, 1e-9)}
        out[T] = res
        print(f"{T:5s} natural FA {fa}/{words} = {fa / max(words, 1):.3%}   challenge P {P:.3f} R {R:.3f} F1 {res['challenge']['F1']:.3f}")
    json.dump(out, open(f"{G}/../evals/eval_{a.model}_{a.split}.json", "w"), indent=1)
    with open(f"{G}/../evals/review_{a.model}_{a.split}.jsonl", "w") as f:
        for x in review:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
