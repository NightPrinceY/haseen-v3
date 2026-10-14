"""Decision policy, targets and word splitting (no model needed)."""
import pytest

from haseen.decision import NAME, VARS, build_result, decide, resolve_target, split_words

CAL = {"T_var": 1.0, "T_odia": 1.0, "tau": {v: 0.5 for v in VARS}, "tau_odia": 0.9, "odia_gate": 0.5,
       "experimental": ["IRQ"]}


def pv(**kw):
    p = {v: 0.99 for v in VARS}
    p.update(kw)
    return p


@pytest.mark.parametrize("alias,t", [("egy", "EGY"), ("Egyptian", "EGY"), ("Arabic / Saudi", "SAU"), ("gulf", "GLF"),
                                     ("emirati", "GLF"), ("yemeni", "GLF"), ("خليجي", "GLF"),
                                     ("shami", "LEV"), ("fusha", "MSA"), ("مصري", "EGY"),
                                     ("iraqi", "IRQ"), ("najdi", "SAU")])
def test_aliases(alias, t):
    assert resolve_target(alias) == t


def test_unknown_target():
    for bad in ("klingon", "darija", "tun"):  # Maghrebi is a label, not a target, in this release
        with pytest.raises(ValueError):
            resolve_target(bad)


def test_split_keeps_offsets_and_strips_punctuation():
    toks = split_words("أبغى، سوف!")
    assert [w for w, _, _ in toks] == ["أبغى", "سوف"]
    assert toks[1][1] == 6


def test_names_always_conform():
    assert decide(pv(EGY=0.01), 0.99, 0.9, "EGY", CAL)[0] is True


def test_most_likely_label_first():
    ok, prim, S, conf = decide(pv(EGY=0.05, SAU=0.9, MSA=0.8, GLF=0.1, LEV=0.1, IRQ=0.1, MAG=0.1), 0.0, 0.0, "EGY", CAL)
    assert not ok and prim == "SAU" and S == ["SAU", "MSA"] and conf == pytest.approx(0.95)


def test_label_set_sorted_by_probability():
    ok, prim, S, _ = decide(pv(EGY=0.05, MSA=0.1, SAU=0.95, GLF=0.97, LEV=0.1, IRQ=0.6, MAG=0.1), 0.0, 0.0, "EGY", CAL)
    assert not ok and prim == "GLF" and S == ["GLF", "SAU", "IRQ"]


def test_other_dialect_needs_target_unnatural():
    # odia fires only when the word is not natural in the target (odia_gate)
    assert decide(pv(LEV=0.9), 0.95, 0.0, "LEV", CAL)[0] is True
    ok, prim, _, _ = decide(pv(LEV=0.1), 0.95, 0.0, "LEV", CAL)
    assert not ok and prim == "OTHER_DIA"


def test_result_summary_and_experimental_note():
    toks = split_words("أبغى أساعدك")
    probs = {0: (pv(EGY=0.05, MSA=0.1, SAU=0.95, GLF=0.9, LEV=0.1, IRQ=0.2, MAG=0.1), 0.0, 0.0),
             1: (pv(), 0.0, 0.0)}
    r = build_result(toks, probs, "GLF", CAL, lambda w, t: None, {"model": "test"})
    assert r["target"] == NAME["GLF"] and r["summary"]["adherence"] == 1.0
    r = build_result(toks, {0: (pv(IRQ=0.9), 0.0, 0.0)}, "IRQ", CAL, lambda w, t: None, {"model": "test"})
    assert "note" in r["meta"]
    r = build_result(toks, probs, "EGY", CAL, lambda w, t: None, {"model": "test"})
    assert r["deviations"] == ["أبغى"] and r["summary"]["adherence"] == 0.5
    assert r["tokens"][0]["labels"][0] == NAME["SAU"]


def test_saudi_inside_gulf():
    # a Gulf-east word: deviates for Saudi with label GLF; a Saudi-only deviation from Gulf never names SAU
    p = pv(SAU=0.05, GLF=0.95, EGY=0.1, MSA=0.1, LEV=0.1, IRQ=0.1, MAG=0.1)
    assert decide(p, 0.0, 0.0, "GLF", CAL)[0] is True and decide(p, 0.0, 0.0, "SAU", CAL)[1] == "GLF"
    ok, prim, S, _ = decide(pv(GLF=0.05, SAU=0.6, EGY=0.9, MSA=0.1, LEV=0.6, IRQ=0.1, MAG=0.1), 0.0, 0.0, "GLF", CAL)
    assert not ok and prim == "EGY" and "SAU" not in S
