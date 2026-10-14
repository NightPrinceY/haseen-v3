"""Targets, word splitting, suggestions and the per-word decision, shared by every HaSeen runtime.

Decision for target t (policy fitted on the V3 dev sets, see docs/RESULTS.md):
  other     p_name > 0.5                                  -> names, digits, Latin: always conform
  deviates  p_other_dialect > tau_odia and p_t < odia_gate -> label "Arabic / other dialect"
            p_t < tau_t                                    -> labels = the varieties where the word is natural
                                                              (p_v >= 0.5), most likely first (gold dev: top-1 0.501
                                                              vs 0.479 for "MSA first")
  conforms  otherwise
"""
import json, os, re

from .text import norm

VARS = ("MSA", "EGY", "SAU", "GLF", "LEV", "IRQ", "MAG")  # model outputs (one naturalness bit each)
TARGETS = ("MSA", "EGY", "SAU", "GLF", "LEV", "IRQ")  # what a customer can choose; MAG is a label only
INSIDE = {"GLF": ("SAU",)}  # Saudi is a Gulf variety: never named as the source of a deviation from Gulf
NAME = {"MSA": "Arabic / MSA", "EGY": "Arabic / Egyptian", "SAU": "Arabic / Saudi", "GLF": "Arabic / Gulf",
        "LEV": "Arabic / Levantine", "IRQ": "Arabic / Iraqi", "MAG": "Arabic / Maghrebi", "OTHER_DIA": "Arabic / other dialect"}
SHORT = {"MSA": "MSA", "EGY": "Egyptian", "SAU": "Saudi", "GLF": "Gulf", "LEV": "Levantine", "IRQ": "Iraqi",
         "MAG": "Maghrebi", "OTHER_DIA": "other dialect"}
_AL = {
    "MSA": ["msa", "fusha", "fus7a", "standard", "modern standard arabic", "فصحى", "الفصحى"],
    "EGY": ["egy", "egyptian", "egypt", "masri", "مصري", "المصرية"],
    "SAU": ["sau", "saudi", "ksa", "najdi", "hijazi", "سعودي", "السعودية"],
    "GLF": ["glf", "gulf", "khaleeji", "khaliji", "emirati", "uae", "qatari", "kuwaiti", "bahraini", "omani", "yemeni",
            "خليجي", "الخليجية"],
    "LEV": ["lev", "levantine", "shami", "syrian", "lebanese", "jordanian", "palestinian", "شامي", "الشامية"],
    "IRQ": ["irq", "iraqi", "iraq", "عراقي", "العراقية"],
}
ALIASES = {a: t for t, al in _AL.items() for a in al + [NAME[t].lower()]}
_WORD = re.compile(r"\S+")
_LETTER = r"ء-ْٰٱ-ۓ\w"  # letters + harakat/tanween, so a final tanween stays in the word
_EDGE = re.compile(rf"^[^{_LETTER}]+|[^{_LETTER}]+$")
ARABIC = re.compile(r"[ء-يٱ-ۓ]")


def resolve_target(target):
    """'egy' / 'Egyptian' / 'Arabic / Egyptian' / 'مصري' -> 'EGY'."""
    t = ALIASES.get(str(target).strip().lower())
    if t is None:
        raise ValueError(f"unsupported target {target!r}: use one of {', '.join(v.lower() for v in TARGETS)}")
    return t


def split_words(text):
    """Words with character offsets; punctuation at word edges is not part of the word."""
    out = []
    for m in _WORD.finditer(text):
        raw = m.group(0)
        core = _EDGE.sub("", raw)
        if core:
            s = m.start() + raw.find(core)
            out.append((core, s, s + len(core)))
    return out


class Suggester:
    """Target-variety replacement for a deviating word: future-marker rules, then the expert concept map
    (form -> the forms each variety uses for the same concept)."""

    FUTURE = {"EGY": "هـ + الفعل", "LEV": "رح + الفعل", "SAU": "بـ / راح + الفعل", "GLF": "بـ / راح + الفعل",
              "IRQ": "راح + الفعل"}
    CMAP = {"GLF": ("SAU", "KHL", "YEM")}  # concept-map keys (MADAR varieties) for each target

    def __init__(self, lexicon_dir):
        try:
            self.cmap = json.load(open(os.path.join(lexicon_dir, "concept_map.json"), encoding="utf-8"))
        except OSError:
            self.cmap = {}

    def __call__(self, word, t):
        w = norm(word)
        if t != "MSA" and w in ("سوف", "وسوف", "فسوف"):
            return self.FUTURE.get(t)
        if t == "EGY" and len(w) >= 4 and w[0] == "س" and w[1] in "ينت":
            return "ه" + word[1:]
        for m in self.CMAP.get(t, (t,)):
            forms = [f for f in self.cmap.get(w, {}).get(m, []) if f != w]
            if forms:
                return forms[0]
        return None


def decide(pv, pd, po, t, cal):
    """-> (conforms, primary label or None, label set, confidence)."""
    pt = pv[t]
    if po > 0.5:
        return True, None, [], None
    if pd > cal["tau_odia"] and pt < cal.get("odia_gate", 1.0):
        return False, "OTHER_DIA", ["OTHER_DIA"], pd
    if pt < cal["tau"][t]:
        others = [v for v in VARS if v != t and v not in INSIDE.get(t, ())]
        S = [v for v in sorted(others, key=lambda v: -pv[v]) if pv[v] >= 0.5] or [max(others, key=lambda v: pv[v])]
        return False, S[0], S, 1 - pt
    return True, None, [], pt


def build_result(toks, probs, t, cal, suggest, meta, return_probs=False):
    """toks: [(word, start, end)]; probs: {word index: ({VARS: p}, p_other_dialect, p_name)} -> result JSON."""
    out, devs, arabic, informative = [], [], 0, 0
    caller, caller_all = {v: 0.0 for v in VARS}, {v: 0.0 for v in VARS}
    for i, (w, s, e) in enumerate(toks):
        item = {"word": w, "index": i, "start": s, "end": e, "conforms": True}
        if i in probs:
            pv, pd, po = probs[i]
            if return_probs:
                item["p"] = {**{v: round(pv[v], 4) for v in VARS}, "other_dialect": round(pd, 4), "name": round(po, 4)}
            ok, prim, S, conf = decide(pv, pd, po, t, cal)
            if po <= 0.5:
                arabic += 1
                for v in VARS:
                    caller_all[v] += pv[v]
                if min(pv.values()) < 0.5:  # only words that some variety would not say tell varieties apart
                    informative += 1
                    for v in VARS:
                        caller[v] += pv[v]
            if not ok:
                item.update(conforms=False, label=NAME[prim], labels=[NAME[x] for x in S], confidence=round(conf, 3))
                sug = suggest(w, t) if prim != "OTHER_DIA" else None
                if sug:
                    item["suggestion"] = sug
            elif conf is not None:
                item["confidence"] = round(conf, 3)
        if not item["conforms"]:
            devs.append(w)
        out.append(item)
    summary = {"arabic_words": arabic, "deviating_words": len(devs),
               "adherence": round(1 - len(devs) / arabic, 4) if arabic else 1.0}
    if arabic:  # which variety the turn reads as (caller-dialect estimate): mean naturalness over informative words
        sc, k = (caller, informative) if informative else (caller_all, arabic)
        ranked = sorted(VARS, key=lambda v: -sc[v])
        summary["reads_as"] = [{"variety": NAME[v], "score": round(sc[v] / k, 3)} for v in ranked[:3]]
        summary["informative_words"] = informative
    if t in cal.get("experimental", []):
        meta = {**meta, "note": f"{NAME[t]} is experimental in this release (see docs/RESULTS.md)"}
    return {"target": NAME[t], "tokens": out, "deviations": devs, "summary": summary, "meta": meta}
