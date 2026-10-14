"""G0: per-target status of MSA function words -> data/lexicons_v2/markers.tsv (2026-10-14).

Evidence: native text of each target (MADAR Corpus-26/6 *train* splits + Alexandria *train*; dev / test splits are
reserved for gold) vs MSA (MADAR MSA column). ratio = rate per 1k tokens in T / rate in MSA.
Status per target (guideline v2 §3):
  hard   — grammatical MSA marker (future, verbal negation, relative, demonstrative) with ratio <= 0.05:
           deviates in every register
  frm    — other MSA function word with ratio <= 0.05: natural in the service register (V1 policy, validated by
           Yahya for Egyptian), flagged only under the optional "strict colloquial" policy
  shared — ratio >= 0.20: the target uses it
  mixed  — in between: left to context / the model (label unknown in training)
EGY keeps Yahya's 42 validated register answers (V1 guideline v1.2) where they differ."""
import json, os

ROOT = os.path.join(os.path.dirname(__file__), "..")
C = json.load(open(f"{ROOT}/data/inventory/target_counts_madar_alex_train.json"))
TOT = C.pop("_tot")
T = ["EGY", "SAU", "KHL", "LEV", "IRQ", "YEM", "MOR", "TUN"]
GRAM = {  # closed grammatical classes whose MSA forms deviate when the target does not use them
    "future": ["سوف"], "negation": ["لن", "لم", "ليس", "ليست", "لسنا", "ليسوا", "اليس", "اليست"],  # الم / الن left out: homographs of ألم "pain"
    "relative": ["الذي", "التي", "الذين", "اللذان", "اللتان", "اللواتي", "اللاتي"],
    "demonstrative": ["هذا", "هذه", "هؤلاء", "ذلك", "تلك", "اولئك", "هذان", "هاتان"]}
OTHER = ["هناك", "هنا", "نحن", "انتم", "ماذا", "لماذا", "كيف", "اين", "متي", "هل", "الان", "ايضا", "فقط", "لكن",
         "اذا", "يوجد", "توجد", "لدي", "لدينا", "لديك", "عندما", "حيث", "جدا", "كثيرا", "اريد", "نريد", "تريد",
         "يمكن", "يمكنك", "يمكنني", "يمكننا", "حسنا", "ان", "قد", "تم", "سيتم", "ثم", "لذلك", "بينما", "اي",
         "كي", "لكي", "اذ", "انما", "بل", "منذ", "ربما"]
EGY_V12 = {"نحن": "frm", "هل": "frm", "تم": "frm", "اذا": "shared", "لكن": "shared", "فقط": "frm", "الان": "frm",
           "يمكن": "frm", "يمكنك": "frm", "يمكنني": "frm", "كيف": "frm", "متي": "frm", "اين": "frm", "ذلك": "frm"}
# Spelling variants: Gulf / Iraqi writers often spell their own demonstrative the MSA way (هاذا -> هذا). If the
# target uses a variant, the MSA spelling is not evidence of MSA -> "mixed" (left to context), never "hard".
VARIANT = {"هذا": ["هاذا", "هذا"], "هذه": ["هذي", "هاذي", "هاذه", "هاذيه"], "ذلك": ["ذاك", "هذاك", "هاذاك"],
           "تلك": ["ذيك", "هذيك", "هاذيك"], "هؤلاء": ["هذول", "هاذول", "هذولا", "هاذولا"]}
# ذلك: V1.2 accepted it in the Egyptian service register (Q-list), unlike هذا / هذه.

rate = lambda t, w: C[t].get(w, 0) / TOT[t] * 1000
rows = []
for cat, ws in list(GRAM.items()) + [("function", OTHER)]:
    for w in ws:
        m = rate("MSA", w)
        if m < 0.05:
            continue
        st = {}
        for t in T:
            r = rate(t, w) / m
            if r >= 0.20:
                s = "shared"
            elif r <= 0.05:
                s = "hard" if cat != "function" else "frm"
            else:
                s = "mixed"
            if s == "hard" and any(rate(t, v) / m >= 0.20 for v in VARIANT.get(w, []) if v != w):
                s = "mixed"
            if t == "KHL" and w in ("هذا", "هذه") and s == "hard":
                s = "mixed"  # Doha has only MADAR (one translator); Gulf writers spell hādha / hādhi as هذا / هذه
            if t == "EGY" and w in EGY_V12:
                s = EGY_V12[w]
            st[t] = (s, round(r, 3))
        rows.append((w, cat, round(m, 2), st))
os.makedirs(f"{ROOT}/data/lexicons_v2", exist_ok=True)
with open(f"{ROOT}/data/lexicons_v2/markers.tsv", "w") as f:
    f.write("# MSA function words: status per target (hard / frm / shared / mixed) and ratio vs MSA. Built by "
            "scripts/build_markers.py from MADAR + Alexandria train splits.\n")
    f.write("form\tclass\tmsa_per_1k\t" + "\t".join(T) + "\n")
    for w, cat, m, st in rows:
        f.write(f"{w}\t{cat}\t{m}\t" + "\t".join(f"{st[t][0]}:{st[t][1]}" for t in T) + "\n")
print(f"{len(rows)} forms")
for t in T:
    hard = [w for w, c, m, st in rows if st[t][0] == "hard"]
    shared = [w for w, c, m, st in rows if st[t][0] == "shared" and c != "function"]
    print(f"{t:5s} hard: {' '.join(hard)}   | grammatical forms the target uses: {' '.join(shared) or '-'}")
