"""V3 decision policy + evaluation from raw per-word logits (train_v3.py --dump_raw or release onnx_raw.py).

Label space: MSA EGY SAU GLF LEV IRQ MAG (model outputs) + odia + other. Production targets: MSA EGY SAU GLF LEV IRQ.
Policy (dev only): per target tau_t = max challenge F1 on V3 dev subject to natural false alarms <= --fa_max; global
tau_odia the same way (mean F1); odia gate fixed at 0.5 (PREREG_V3). Temperatures come from calibration.json.
Labels: a deviating word gets the set {v != t : p_v >= 0.5} (members of t excluded: SAU is inside GLF), most
likely first. --test scores (once) natural / challenge / agent test sets and the mo3jam probe.
Usage: python posthoc_v3.py <run> [--raw raw_gold.jsonl.gz] [--cal best/calibration.json] [--tag _x] [--test]"""
import argparse, gzip, json, math, os

VARS = ("MSA", "EGY", "SAU", "GLF", "LEV", "IRQ", "MAG")
TARGETS = ("MSA", "EGY", "SAU", "GLF", "LEV", "IRQ")
INSIDE = {"GLF": {"SAU"}, "SAU": set()}  # members a target's labels must not name (SAU is a Gulf variety)
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--gold", default=os.path.join(HERE, "..", "data", "test_v3"))
ap.add_argument("--raw", default="raw_gold.jsonl.gz")
ap.add_argument("--cal", default="best/calibration.json")
ap.add_argument("--fa_max", type=float, default=0.01)
ap.add_argument("--odia_gate", type=float, default=0.5)
ap.add_argument("--tag", default="")
ap.add_argument("--test", action="store_true")
a = ap.parse_args()
sig = lambda x: 1 / (1 + math.exp(-max(-60, min(60, x))))
cal = json.load(open(os.path.join(a.run, a.cal)))
raw = {}
for l in gzip.open(os.path.join(a.run, a.raw), "rt"):
    r = json.loads(l); raw[(r["set"], r["id"])] = r["p"]


def probs(p):
    pv = {v: sig(p["v"][k] / cal["T_var"]) for k, v in enumerate(VARS)}
    return pv, sig(p["odia"] / cal["T_odia"]), sig(p["other"])


def decide(pv, pd, po, t, tau, tau_odia):
    if po > 0.5:
        return False, None, ()
    if pd > tau_odia and pv[t] < a.odia_gate:
        return True, "OTHER_DIA", ("OTHER_DIA",)
    if pv[t] < tau:
        others = sorted([v for v in VARS if v != t and v not in INSIDE.get(t, set())], key=lambda v: -pv[v])
        S = tuple(v for v in others if pv[v] >= 0.5) or (others[0],)
        return True, S[0], S
    return False, None, ()


def words(name):
    path = os.path.join(a.gold, name + ".jsonl")
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        x = json.loads(l)
        ps = raw.get((name, x["id"]))
        if ps is None:
            continue
        hidx = set(i for i, h in zip(x.get("swapped", []), x.get("heldout_concept", [])) if h)
        for i, p in enumerate(ps):
            if x["ignore"][i]:
                continue
            pv, pd, po = probs(p) if p else ({v: 1.0 for v in VARS}, 0.0, 0.0)
            out.append({"pv": pv, "pd": pd, "po": po, "g": x["dev"][i], "lab": x["label"][i], "held": i in hidx})
    return out


def score(nat, ch, t, tau, tau_odia):
    fa = sum(decide(w["pv"], w["pd"], w["po"], t, tau, tau_odia)[0] and not w["g"] for w in nat) / max(1, len(nat))
    tp = fp = fn = top1 = hit = htp = hn = 0
    for w in ch:
        d, prim, S = decide(w["pv"], w["pd"], w["po"], t, tau, tau_odia)
        tp += d and w["g"]; fp += d and not w["g"]; fn += (not d) and w["g"]
        if d and w["g"]:
            top1 += prim == w["lab"]; hit += (w["lab"] in S) or (w["lab"] == "SAU" and "GLF" in S)
        if w["g"] and w["held"]:
            hn += 1; htp += d
    P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return {"FA": round(fa, 5), "P": round(P, 4), "R": round(R, 4), "F1": round(2 * P * R / max(1e-9, P + R), 4),
            "label_top1": round(top1 / max(1, tp), 4), "label_set_hit": round(hit / max(1, tp), 4),
            "heldout_recall": round(htp / max(1, hn), 4), "heldout_n": hn, "n_dev": tp + fn}


dev = {t: (words(f"natural_{t}_dev"), words(f"challenge_{t}_dev")) for t in TARGETS}
GRID = [0.005, 0.01, 0.02, 0.03, 0.04] + [round(0.05 + 0.025 * k, 3) for k in range(37)]


def fit(tau_odia):
    taus, res = {}, {}
    for t in TARGETS:
        cands = [(score(*dev[t], t, tau, tau_odia), tau) for tau in GRID]
        ok = [c for c in cands if c[0]["FA"] <= a.fa_max]
        best = max(ok, key=lambda c: c[0]["F1"]) if ok else min(cands, key=lambda c: c[0]["FA"])
        taus[t], res[t] = best[1], best[0]
    return taus, res


best = None
for tau_odia in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98):
    taus, res = fit(tau_odia)
    m = sum(r["F1"] for r in res.values()) / len(TARGETS)
    gate = sum(r["FA"] <= a.fa_max for r in res.values())
    if best is None or (gate, m) > (best[0], best[1]):
        best = (gate, m, tau_odia, taus, res)
gate, m, tau_odia, taus, res = best
policy = {"T_var": cal["T_var"], "T_odia": cal["T_odia"], "tau": taus, "tau_odia": tau_odia, "odia_gate": a.odia_gate,
          "fitted_on": "V3 dev (natural + challenge)", "label_rule": "set {v != t, v not inside t: p_v >= 0.5}; most likely first"}
out = {"policy": policy, "dev": res, "dev_mean_F1": round(m, 4), "dev_gate": gate}


def table(title, rs, extra=""):
    print(f"\n{title}\n{'T':4} {'tau':>5} {'natFA':>7} {'P':>6} {'R':>6} {'F1':>6} {'top1':>5} {'set':>5} {'held':>5}")
    for t in TARGETS:
        r = rs[t]
        print(f"{t:4} {taus[t]:5.3f} {r['FA']*100:6.2f}% {r['P']:6.3f} {r['R']:6.3f} {r['F1']:6.3f} {r['label_top1']:5.2f} "
              f"{r['label_set_hit']:5.2f} {r['heldout_recall']:5.2f}")
    print(f"mean F1 {sum(rs[t]['F1'] for t in TARGETS) / len(TARGETS):.4f} | within FA gate: "
          f"{sum(rs[t]['FA'] <= a.fa_max for t in TARGETS)}/6 | tau_odia {tau_odia} {extra}")


table(f"V3 DEV (policy fitted here) — {a.run}", res)
if a.test:
    test = {t: score(words(f"natural_{t}_test"), words(f"challenge_{t}_test"), t, taus[t], tau_odia) for t in TARGETS}
    agent = {}
    for t in TARGETS:
        ws = words(f"agent_{t}_test")
        if ws:
            agent[t] = score([], ws, t, taus[t], tau_odia)
            agent[t]["FA_words"] = round(sum(decide(w["pv"], w["pd"], w["po"], t, taus[t], tau_odia)[0] and not w["g"]
                                              for w in ws) / max(1, sum(not w["g"] for w in ws)), 5)
    # mo3jam probe: per check, the head word must conform (own) or deviate (cross)
    pr = {"conform": [0, 0], "deviate": [0, 0]}
    pr_t = {}
    for l in open(os.path.join(a.gold, "probe_mo3jam.jsonl")):
        x = json.loads(l)
        ps = raw.get(("probe_mo3jam", x["id"]))
        if not ps or ps[x["pos"]] is None:
            continue
        pv, pd, po = probs(ps[x["pos"]])
        for c in x["checks"]:
            t = c["target"]
            d = decide(pv, pd, po, t, taus[t], tau_odia)[0]
            ok = d if c["expect"] == "deviate" else not d
            pr[c["expect"]][0] += ok; pr[c["expect"]][1] += 1
            k = (t, c["expect"]); pr_t.setdefault(k, [0, 0]); pr_t[k][0] += ok; pr_t[k][1] += 1
    out["test"] = test; out["agent"] = agent
    out["probe"] = {k: round(v[0] / max(1, v[1]), 4) for k, v in pr.items()} | {"n": {k: v[1] for k, v in pr.items()}}
    out["probe_by_target"] = {f"{t}:{e}": round(v[0] / max(1, v[1]), 4) for (t, e), v in pr_t.items()}
    table("V3 TEST (scored once, frozen policy)", test)
    print("\nAGENT (LLM replies 'in dialect T', Claude-labelled)")
    for t, r in agent.items():
        print(f"{t:4} P {r['P']:.3f} R {r['R']:.3f} F1 {r['F1']:.3f} · false alarms {r['FA_words']*100:.2f}% · deviations {r['n_dev']}")
    print("\nPROBE (mo3jam, held out):", out["probe"])
json.dump(out, open(os.path.join(a.run, f"posthoc_v3{a.tag}{'_test' if a.test else ''}.json"), "w"), indent=1)
json.dump(policy, open(os.path.join(a.run, f"policy_v3{a.tag}.json"), "w"), indent=1)
