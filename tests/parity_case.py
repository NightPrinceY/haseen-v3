"""Writes random logits + the Python result, for tests/test_js_parity (node) to compare against demo/web/haseen.js."""
import json, math, random, sys
sys.path.insert(0, "src")
from haseen.decision import ARABIC, TARGETS, VARS, build_result, split_words  # noqa: E402

random.seed(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
cal = {"T_var": 0.9, "T_odia": 1.1, "tau": {v: random.choice([0.01, 0.1, 0.5, 0.8]) for v in TARGETS}, "tau_odia": 0.6,
       "odia_gate": 0.5}
text = "أبغى أساعدك، سوف أحوّلك للقسم المختص الحين Vodafone ١٢٣ شو بدي هلق دلوقتي واش"
toks = split_words(text)
variety, odia, other, probs, first = [], [], [], {}, []
for i, (w, _, _) in enumerate(toks):
    if not ARABIC.search(w):  # like the engines: only Arabic words go through the model
        first.append(-1); continue
    first.append(len(odia))
    v = [random.gauss(0, 4) for _ in VARS]; d = random.gauss(-2, 3); o = random.gauss(-3, 3)
    variety += v; odia.append(d); other.append(o)
    sig = lambda x: 1 / (1 + math.exp(-x))
    probs[i] = ({x: sig(v[k] / cal["T_var"]) for k, x in enumerate(VARS)}, sig(d / cal["T_odia"]), sig(o))
res = {t: build_result(toks, probs, t, cal, lambda w, tt: None, {}) for t in TARGETS}
json.dump({"text": text, "cal": cal, "variety": variety, "odia": odia, "other": other,
           "first": first, "res": res}, sys.stdout, ensure_ascii=False)
