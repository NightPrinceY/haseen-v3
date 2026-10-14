"""HaSeen V2 training labeler (guideline v2). Never guesses: no evidence -> None (masked in the loss).

label(tokens, host) -> list of dicts per token:
    bits:  {target: 1 | 0 | None} for the 10 targets
    frm:   [targets for which the word is a formal-MSA function word (natural in the service register)]
    odia:  1 if the word belongs to an "other dialect" (Omani, Sudanese, Libyan, Algerian, Hassaniya, Levantine
           or Maghrebi forms outside the targets) and to no target
    other: 1 for non-Arabic tokens (never deviate)
    src:   evidence tag (for audits)
host = the target code of the sentence's own variety (or None). A native host sentence makes its words natural
for the host target when nothing says otherwise ("natural hosts", V1 family A).

Evidence order (GUIDELINE_V2 §4):
 1 non-Arabic -> other
 2 markers.tsv for MSA function words (hard -> 0, frm -> 1 + frm, shared -> 1, mixed -> None); MSA bit 1
 3 سـ + imperfect (MSA future) -> 0 for every dialect target, 1 for MSA
 4 MSA-valid open-class words (human MSA text, MADAR MSA forms) -> shared in the service register (all 1)
 5 dialect forms (MADAR Lexicon minus held-out concepts, voted crowd lexicons, Birzeit lemmas, Lingualism) ->
   1 for the targets that list them; 0 for a target whose cities fill the same concept with other forms and whose
   human text never uses the form (sense-level negative evidence)
 6 per-target human-text counts: used >= 3 times -> 1; never used although a 0-evidence peer uses it -> 0 only
   if the form already has positive dialect evidence elsewhere (absence alone is not evidence)
 7 colloquial verb prefixes (بـ / هـ / حـ / عم + imperfect) -> MSA 0
 8 host prior -> host bit 1 when still None"""
import csv, json, os, re, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(V2, "..", "HaSeen-V1", "src"))
from haseen.text import norm, words  # noqa: E402

TARGETS = ["MSA", "EGY", "SAU", "KHL", "LEV", "IRQ", "YEM", "MOR", "TUN"]
DIA_T = TARGETS[1:]
BIG_T = ("EGY", "SAU", "LEV", "IRQ", "YEM")  # dialect targets with >= 500K tokens of human text (inventory v3)
CITY2T = {"CAI": "EGY", "ALX": "EGY", "ASW": "EGY", "RIY": "SAU", "JED": "SAU", "DOH": "KHL", "JER": "LEV",
          "AMM": "LEV", "SAL": "LEV", "BEI": "LEV", "DAM": "LEV", "ALE": "LEV", "BAG": "IRQ", "BAS": "IRQ",
          "MOS": "IRQ", "SAN": "YEM", "RAB": "MOR", "FES": "MOR", "TUN": "TUN", "SFA": "TUN",
          "KHA": "ODIA", "TRI": "ODIA", "BEN": "ODIA", "ALG": "ODIA", "MUS": "ODIA"}
VAR2T = {"EG": "EGY", "SA": "SAU", "KW": "KHL", "QA": "KHL", "BH": "KHL", "AE": "KHL", "GLF": "KHL", "PS": "LEV",
         "JO": "LEV", "LB": "LEV", "SY": "LEV", "IQ": "IRQ", "YE": "YEM", "MA": "MOR", "TN": "TUN",
         "OM": "ODIA", "SD": "ODIA", "LY": "ODIA", "DZ": "ODIA", "MR": "ODIA", "MAG": "ODIA", "LEV": "LEV"}
ARABIC = re.compile(r"[ء-ي]")
PROCLITICS = re.compile(r"^(وال|فال|بال|كال|لل|وب|ول|فب|فل|وك|ال|و|ف|ب|ل|ك)(?=..)")
SA_FUT_RAW = re.compile(r"^(و|ف)?س(ي|ت|ن|أ)[ء-ي]{2,}$")  # on the raw word: 1st person keeps its hamza
HA_FUT = re.compile(r"^(?:و|ف)?ه((?:ي|ت|ن)[ء-ي]{2,})$")  # not هـا: هالسالفة / هاشم  # v10: Egyptian هـ-future
COLLOQ_VERB = re.compile(r"^(و|ف)?(ب|ه|ح)(ي|ت|ن|ا)[ء-ي]{2,}$|^(و|ف)?عم$")


def stems(w):
    out = {w}
    m = PROCLITICS.match(w)
    if m and len(w) - m.end() >= 3:  # never strip down to a 1-2 letter word (اللي -> لي, كام -> ام)
        out.add(w[m.end():])
    return out


def variants(w):
    """Spelling variants that are the same word: final ه / none, ت / ث, د / ذ."""
    out = {w, w + "ه", w[:-1] if w.endswith("ه") else w}
    for a, b in (("ت", "ث"), ("ث", "ت"), ("د", "ذ"), ("ذ", "د")):
        out |= {x.replace(a, b) for x in list(out)}
    return out


SUFFIX = re.compile(r"(ي|ك|ه|ها|نا|كم|هم|كما|هما|ني)$")


# v8 candidate changes, ablated on gold dev (audit_labeler_v2.py). Kept: odia only for words natural in no target,
# formal open-class words shared (formal_open), spelling variants in sense negatives, msa_valid peak over all
# dialects (أبغى). Rejected by the audit (more false alarms / fewer hits): used_dom, corpus_ev, msabit_dom,
# exact_first. Override with env HASEEN_V8_OFF="..." (comma list of switches that are OFF).
V10_ON = set(filter(None, os.environ.get("HASEEN_V10_ON", "").split(",")))  # opt-in experimental rules
V8_OFF = set(filter(None, os.environ.get("HASEEN_V8_OFF", "used_dom,corpus_ev,msabit_dom,exact_first").split(",")))


class Labeler:
    def __init__(self, root=V2):
        d = lambda *p: os.path.join(root, *p)
        # markers
        self.mk = {}
        for line in open(d("data", "lexicons_v2", "markers.tsv")):
            if line.startswith(("#", "form\t")):
                continue
            p = line.rstrip("\n").split("\t")
            self.mk[p[0]] = {t: p[3 + i].split(":")[0] for i, t in enumerate(DIA_T)}
        held = set(json.load(open(d("data", "lexicons_v2", "heldout_concepts.json")))["heldout"])
        # MADAR lexicon (READY rows, held-out concepts removed): concept -> target -> forms
        self.concept = collections.defaultdict(lambda: collections.defaultdict(set))
        self.form_concepts = collections.defaultdict(set)
        msa_forms = set()
        for r in csv.DictReader(open(d("data", "sources", "madar_lexicon", "MADAR_Lexicon_v1.0", "MADAR_Lexicon_v1.0.tsv")),
                                delimiter="\t"):
            if r["Release Status"] != "READY" or r["Concept_ID"] in held:
                continue
            f = norm(r["CODA"].strip())
            if f and " " not in f:
                self.concept[r["Concept_ID"]][CITY2T[r["Dialect"]]].add(f)
                self.form_concepts[f].add(r["Concept_ID"])
            for m in r["MSA"].replace("،", ",").split(","):
                m = norm(m.strip())
                if m and " " not in m and "+" not in m and "#" not in m:
                    msa_forms.add(m)
        # closed-class concepts (numbers, pronouns, particles): MSA forms no city of target t lists -> hard for t
        cl_msa, cl_t = collections.defaultdict(set), collections.defaultdict(lambda: collections.defaultdict(set))
        self.closed_concepts = set()
        for r in csv.DictReader(open(d("data", "sources", "madar_lexicon", "MADAR_Lexicon_v1.0", "MADAR_Lexicon_v1.0.tsv")),
                                delimiter="\t"):
            if r["Class"] != "ClosedClass" or r["Release Status"] != "READY":
                continue
            self.closed_concepts.add(r["Concept_ID"])
            for m in r["MSA"].replace("،", ",").split(","):
                m = norm(m.strip())
                if m and " " not in m and "+" not in m and "#" not in m:
                    cl_msa[r["Concept_ID"]].add(m)
            f = norm(r["CODA"].strip())
            if f:
                cl_t[r["Concept_ID"]][CITY2T[r["Dialect"]]].add(f)
        self.closed_msa = collections.defaultdict(set)  # MSA form -> dialect targets that never use it
        all_dialect_forms = {f for c in cl_t.values() for fs in c.values() for f in fs}
        for c, ms in cl_msa.items():
            for m in ms:
                if m in all_dialect_forms:
                    continue
                for t in DIA_T:
                    if cl_t[c].get(t):
                        self.closed_msa[m].add(t)
        self.madar_t = collections.defaultdict(set)  # form -> targets / ODIA
        for c, dd in self.concept.items():
            for t, fs in dd.items():
                for f in fs:
                    self.madar_t[f].add(t)
        # human counts per target (no gold splits)
        inv = os.environ.get("HASEEN_INV", "inventory")  # v3: HASEEN_INV=inventory_v3 (test hold-outs removed)
        self.cnt = json.load(open(d("data", inv, "target_counts_human_nogold.json")))
        clean = json.load(open(d("data", inv, "target_counts_madar_alex_train.json")))
        self.clean_tot = clean.pop("_tot")
        self.clean = clean  # MADAR + Alexandria train splits: same domain for every target, used for rate tests
        self.msa_forms = msa_forms
        self.calima_msa = {norm(l.strip()) for l in open(os.path.join(V2, "..", "HaSeen-V1", "data", "lexicons",
                                                                        "msa_stems_camel.txt"))}
        self.tot = {t: max(1, sum(c.values())) for t, c in self.cnt.items()}
        # v8: V1 core lexicon (hand-curated, Egyptian bits validated by Yahya): EGY / MSA bits override, Gulf
        # positives extend to SAU and KHL. Context-ruled entries only give their Egyptian bit (الحين: ctx hin).
        self.core = {}
        for line in open(os.path.join(V2, "..", "HaSeen-V1", "data", "lexicons", "core_v2.tsv")):
            if line.startswith(("#", "form\t")) or not line.strip():
                continue
            f = line.rstrip("\n").split("\t") + [""] * 7
            B = {"1": 1, "0": 0, "?": None}
            self.core[f[0]] = {"EGY": B[f[1][0]], "GLF": B[f[1][1]], "MSA": B[f[1][2]], "clitics": f[4], "ctx": f[5]}
        # crowd / expert lexicons (positive evidence only)
        self.lex_t = collections.defaultdict(set)
        for l in open(d("data", inv, "lexicon.jsonl")):
            r = json.loads(l)
            src = r["source"]
            if src in ("mo3jam", "lahga") and r["weight"] < 1.0:
                continue
            if src in ("lisan_iraqi", "lisan_libyan", "lisan_sudanese", "lisan_yemeni"):
                continue  # Lisan free edition: lemma pairs unreliable; its text enters through counts
            if src == "madar_lexicon":
                continue  # handled above with held-out concepts removed
            for v in r["vars"]:
                t = VAR2T.get(v)
                if t:
                    self.lex_t[r["form"]].add(t)

    # --------------------------------------------------------------------------------------------------
    def rate(self, t, w):
        return max(self.cnt.get(t, {}).get(s, 0) for s in self.S(w)) / self.tot.get(t, 1)

    _S_cache = {}

    def S(self, w):
        """Stems of w, or w alone when the whole form is a listed dialect form (v10, see exact_dialect)."""
        if w not in self._S_cache:
            self._S_cache[w] = {w} if self.exact_dialect(w) else stems(w)
        return self._S_cache[w]

    def exact_dialect(self, w):
        """v10: the whole form is listed as a dialect form by a MADAR city and MSA text barely
        uses it: then clitic-stripped stems must not make it MSA (لعند -> عند, فوقت -> وقت, كاين -> أين)."""
        if "exact_dia" in V8_OFF or w in self.msa_forms or self.cnt["MSA"].get(w, 0) >= 3:
            return False
        # MADAR city forms only: crowd lexicons also list regular clitic compounds (فالبيت) that are plain MSA
        return bool(self.madar_t.get(w, set()) - {"ODIA"})

    def msa_valid(self, w, dia_t):
        """MSA forms of MADAR concepts, or used >= 3 times in human MSA text at a rate not far below its rate in
        the dialects that list it (guards against dialect words leaking into 'MSA' sides of parallel data)."""
        S = self.S(w)
        if self.exact_dialect(w):
            S = {w}
        if S & self.msa_forms:
            return True
        if max(self.cnt["MSA"].get(s, 0) for s in S) < 3:
            return False
        # v8: peak over all dialect targets, not only the listed ones (an unlisted Gulf form like أبغي leaked into
        # 'MSA' text 4 times and passed against an empty set)
        peak = max([self.rate(t, w) for t in (dia_t if "msavalid_all" in V8_OFF else DIA_T) if t in self.cnt] or [0])
        if self.rate("MSA", w) >= 0.2 * peak:
            return True
        # v10: names and news words (العراق, بغداد, الرياض) are dense in one dialect's text but used in MSA text at
        # least half as often as in the typical large dialect corpus; leaked dialect words are far below it
        # (عشان 0.007x, اللي 0.01x, يعني 0.23x)
        if "msa_median" not in V8_OFF and max(self.cnt["MSA"].get(s, 0) for s in S) >= 10:
            rs = sorted(self.rate(t, w) for t in BIG_T if t in self.cnt)
            return self.rate("MSA", w) >= 0.5 * rs[len(rs) // 2]
        return False

    def msa_bit(self, w, dia_t=()):
        """MSA bit for a word with dialect evidence (v8: every positive test needs MSA dominance, because dialect
        words leak into the 'MSA' sides of parallel data: بزاف, برشا, أبغي).
        1: a MADAR MSA concept form; or used in the clean MSA text at >= 20% of its peak clean dialect rate;
        0: listed for >= 2 dialect targets and its human-MSA rate is < 5% of its dialect peak, or never used in MSA
           text and not a CALIMA-MSA stem;
        1: used >= 3 times in human MSA text at >= 20% of its dialect peak; else unknown."""
        S = self.S(w)
        if (S | variants(w)) & self.msa_forms:
            return 1
        crate = lambda t: max(self.clean.get(t, {}).get(x, 0) for x in S) / self.clean_tot.get(t, 1)
        cpeak = max([crate(t) for t in DIA_T if t in self.clean] or [0])
        if crate("MSA") > 0 and (crate("MSA") >= 0.2 * cpeak or "msabit_dom" in V8_OFF):
            return 1
        peak = max([self.rate(t, w) for t in DIA_T if t in self.cnt] or [0])
        if len(set(dia_t) - {"ODIA"}) >= 2 and peak and self.rate("MSA", w) < 0.05 * peak:
            return 0
        # v10: one dialect is enough when its text uses the form >= 30 times and MSA text almost never does
        # (ياهو 370x Iraqi / 1x MSA, هاذه 945x / 1x)
        pk_n = max([max(self.cnt[t].get(x, 0) for x in S) for t in DIA_T if t in self.cnt] or [0])
        n = max(self.cnt["MSA"].get(x, 0) for x in S)
        # (MSA text is news: 1st-person verbs like نقدر / صارت are rare there but valid, so at most 2 MSA hits)
        if "msa0_single" not in V8_OFF and dia_t - {"ODIA"} and pk_n >= 30 and n <= 2 and self.rate("MSA", w) < 0.03 * peak \
                and not ((S | {SUFFIX.sub("", x) for x in S}) & self.calima_msa):
            return 0
        n = max(self.cnt["MSA"].get(x, 0) for x in S)
        if n >= 3 and (self.rate("MSA", w) >= 0.2 * peak or "msabit_dom" in V8_OFF):
            return 1
        if n == 0 and not ((S | {SUFFIX.sub("", x) for x in S}) & self.calima_msa):
            return 0
        return None

    def used(self, t, w):
        """Used by target t: >= 3 occurrences AND >= 10 per million tokens (3 hits in Iraq's 3.8M tokens is noise)."""
        c = max(self.cnt.get(t, {}).get(s, 0) for s in self.S(w))
        if not (c >= 3 and c / self.tot.get(t, 1) >= 1e-5):
            return False
        # v8 dominance: a handful of hits (< 30) is a leak when another dialect uses the word >= 30x more
        # (شو 9x in EGY text, دلوقتي 29x in Yemeni text); frequent words are kept (لي is a homograph in MOR)
        if "used_dom" in V8_OFF:
            return True
        peak = self.peak(w)
        if c >= 30:
            # v10: frequent words too, at a lower bar: Egyptian media and speakers leak into Yemeni / Saudi text
            # (حضرتك at 0.03x, عايز 0.03x, بتاعك 0.08x of the Egyptian rate), real shared use is >= 0.1x (كده in
            # Saudi 0.24x, شو in Gulf-east 0.10x)
            # hundreds of hits are real use even when another dialect uses the word far more (شو 1,733x in Iraqi text)
            return "dom_all" in V8_OFF or c >= 300 or self.rate(t, w) >= 0.08 * peak
        return self.rate(t, w) >= 0.03 * peak

    def peak(self, w):
        """Peak dialect rate; v10: over corpora with >= 150K tokens only (3 hits in Tunisian's 30K tokens = 1e-4)."""
        return max([self.rate(u, w) for u in DIA_T if u in self.cnt and
                    (self.tot.get(u, 0) >= 150000 or "peak_big" in V8_OFF)] or [0])

    def absent(self, t, w):
        """v10: t's text would hold the form >= 50 times at the peak dialect's rate but uses it at < 3% of that rate."""
        if "absent_neg" in V8_OFF or t not in self.cnt:
            return False
        peak = self.peak(w)
        return peak * self.tot.get(t, 0) >= 50 and self.rate(t, w) < 0.03 * peak

    def lex_dom(self, t, w):
        """v10: a crowd-lexicon listing for target t is dropped when t has a large human corpus (>= 500K tokens), the
        form is frequent elsewhere (>= 30 hits) and t uses it at < 3% of that peak rate (موش listed for Saudi:
        1 hit in 1.3M Saudi tokens vs 60 in 30K Tunisian). MADAR city forms are never dropped."""
        # off by default after the v9->v10 label diff: corpus genres differ (Iraqi text is news-heavy), so a low rate
        # cannot overrule a crowd lexicon (خلاص listed for Iraqi and Levantine was dropped)
        if "lex_dom" not in V10_ON or t not in BIG_T or t not in self.cnt:
            return False
        S = self.S(w)
        pk_n = max(max(self.cnt[u].get(x, 0) for x in S) for u in DIA_T if u in self.cnt)
        peak = max(self.rate(u, w) for u in DIA_T if u in self.cnt)
        return pk_n >= 30 and self.rate(t, w) < 0.03 * peak

    def label_word(self, raw, host=None):
        r = self._label_word(raw, host)
        ws = words(raw)
        if r["other"] or not ws:
            return r
        return self._core_override(norm(ws[0]), r)

    def _core_override(self, w, r):
        """v8: V1 core lexicon (hand-curated, Egyptian bits validated by Yahya) over every evidence path: EGY / MSA bits
        override, Gulf positives extend to SAU and KHL; context-ruled entries give only their Egyptian bit."""
        e = self.core_lookup(w) if "core" not in V8_OFF else None
        if not e or e["ctx"] not in ("", "hin"):
            return r
        bits = dict(r["bits"])
        if e["EGY"] is not None:
            bits["EGY"] = e["EGY"]
        if e["ctx"] == "" and e["MSA"] is not None:
            bits["MSA"] = e["MSA"]
        if e["GLF"] == 1:
            for t in ("SAU", "KHL"):
                bits[t] = 1 if bits[t] is None else bits[t]
        odia = r["odia"] if not any(v == 1 for v in bits.values()) else 0
        return {**r, "bits": bits, "odia": odia, "src": r["src"] + "+core"}

    def _label_word(self, raw, host=None):
        ws = words(raw)
        if not ws or not ARABIC.search(ws[0]):
            return {"bits": {t: 1 for t in TARGETS}, "frm": [], "odia": 0, "other": 1, "src": "other"}
        w = norm(ws[0])
        S = self.S(w)
        bits = {t: None for t in TARGETS}
        frm, src = [], []
        # 2 markers (exact form or behind a proclitic)
        mk = self.mk.get(w) or (None if self.exact_dialect(w) else next((self.mk[s] for s in S if s in self.mk), None))
        rawk0 = re.sub(r"[ًٌٍَُِّْ]", "", ws[0])
        if mk and SA_FUT_RAW.match(rawk0) and "fut_hard" not in V8_OFF:
            # v8: سيتم was classed as a formal function word; a سـ-future verb is a hard MSA marker (guideline §3)
            # v10: also "mixed" (سيتم at 6.7% in Saudi TV text is formal drift, not Saudi)
            mk = {t: ("hard" if st in ("frm", "mixed") else st) for t, st in mk.items()}
        if mk:
            bits["MSA"] = 1
            for t, st in mk.items():
                bits[t] = {"hard": 0, "frm": 1, "shared": 1}.get(st)
                if st == "frm":
                    frm.append(t)
            return {"bits": bits, "frm": frm, "odia": 0, "other": 0, "src": "markers"}
        # 2b MSA closed-class forms a target never uses (خمسون، اثنان، ثلاثون)
        cm = self.closed_msa.get(w) if len(w) >= 3 else None  # exact form only: clitic stripping makes كام -> أم
        if cm:
            crate = lambda t: self.clean.get(t, {}).get(w, 0) / self.clean_tot.get(t, 1)
            cm = {t for t in cm if crate(t) < 0.2 * crate("MSA")}  # dialects that do use it keep it
        if cm:
            bits["MSA"] = 1
            for t in DIA_T:
                bits[t] = 0 if t in cm else None
            return {"bits": bits, "frm": [], "odia": 0, "other": 0, "src": "closed_msa"}
        # 3 MSA future سـ + imperfect (not a dialect form, used in MSA)
        rawk = re.sub(r"[ًٌٍَُِّْ]", "", ws[0])
        bare = w[2:] if w[:1] in "وف" else w[1:]  # the imperfect without سـ must itself be a known word
        # v8: formal سـ-future leaks into dialect text (سيتم in Egyptian corpora), so "used in a dialect" no longer
        # blocks the rule; instead MSA must use the form at least as much as any dialect (سيدي stays out)
        dia_use = [t for t in DIA_T if self.used(t, w)]
        msa_dom = not dia_use or ("fut_dom" not in V8_OFF and
                                  self.rate("MSA", w) >= 0.3 * max(self.rate(t, w) for t in DIA_T if t in self.cnt))
        in_msa = any(self.cnt["MSA"].get(s, 0) for s in S) or ("fut_bare" not in V8_OFF and (
            self.cnt["MSA"].get(bare, 0) >= 3 or (not dia_use and sum(self.cnt[t].get(bare, 0) for t in self.cnt) >= 10)))
        # a perfect verb whose root starts with س is not a future (سألت = سأل + ت): check CALIMA-MSA stems
        perfect = {re.sub(r"(تما|تم|تن|نا|وا|ت|ا|ني|ه|ها|هم|ك|كم)$", "", x) for x in S} | S
        if SA_FUT_RAW.match(rawk) and not (S & set(self.madar_t)) and in_msa and not (perfect & self.calima_msa) \
                and sum(self.cnt[t].get(bare, 0) for t in self.cnt) >= 3 and msa_dom:
            bits.update({t: 0 for t in DIA_T})
            bits["MSA"] = 1
            return {"bits": bits, "frm": [], "odia": 0, "other": 0, "src": "msa_future"}
        dia_t = set()
        # v8: evidence for the exact form wins; stems only when the form itself is unlisted (الحين -> حين)
        SE = {w} if (self.madar_t.get(w) or self.lex_t.get(w)) and "exact_first" not in V8_OFF else S
        for s in SE:
            dia_t |= self.madar_t.get(s, set()) | {t for t in self.lex_t.get(s, set()) if not self.lex_dom(t, w)}
        if "corpus_ev" not in V8_OFF:
            dia_t |= {t for t in DIA_T if self.used(t, w)}  # v8: corpus use is evidence too (enables negatives)
        msa_valid = self.msa_valid(w, dia_t)
        # 4 formal / shared open-class word: natural everywhere in the service register
        if msa_valid and not COLLOQ_VERB.match(w):
            return {"bits": {t: 1 for t in TARGETS}, "frm": [], "odia": 0, "other": 0, "src": "msa_shared"}
        # 5 dialect evidence
        if dia_t:
            src.append("lexicon")
            for t in DIA_T:
                if t in dia_t:
                    bits[t] = 1
            bits["MSA"] = self.msa_bit(w, dia_t)
            # negative: target t fills a shared concept with other forms and never uses this form
            concepts = set().union(*(self.form_concepts.get(s, set()) for s in SE))
            peak = max([self.rate(u, w) for u in dia_t if u in self.cnt] or [0])
            # v8: a valid MSA open-class word is shared in the service register (GUIDELINE_V2 §3, same rule as the
            # gold): no dialect negatives for it; closed-class MSA forms keep their negatives
            formal_open = bits["MSA"] == 1 and not (concepts & self.closed_concepts) and "formal_open" not in V8_OFF
            V = variants(w)
            for t in DIA_T:
                if formal_open:
                    break
                if bits[t] is None and not self.used(t, w) and not (SE & self._t_lex(t)):
                    # MADAR: t lists other forms for the concept, and none of them is a spelling variant of w
                    # (v8: ست / سته, تمانيه / ثمانيه are the same word)
                    sense = any(self.concept[c].get(t) and not (V & self.concept[c][t]) for c in concepts)
                    # corpus absence counts only for targets with >= 150K tokens of human text (KHL, TUN, MOR are
                    # too small: common Gulf words like نقدر / نرجع are simply missing there)
                    expect = self.tot.get(t, 0) >= 150000 and peak * self.tot[t] >= 5 and self.rate(t, w) == 0
                    if sense or expect:
                        bits[t] = 0
            odia = int(dia_t == {"ODIA"})
        else:
            odia = 0
        # 6 corpus positives
        for t in DIA_T:
            if bits[t] is None and self.used(t, w):
                bits[t] = 1
                src.append("corpus")
        # 6b v10: strong corpus absence is negative evidence for a word that is not natural in MSA
        if bits["MSA"] != 1 and dia_t - {"ODIA"}:  # only known dialect forms; topic words skew corpora too
            for t in DIA_T:
                if bits[t] is None and t not in dia_t and t != host and self.absent(t, w):
                    bits[t] = 0
                    src.append("absent")
        # 6c v10: هـ + imperfect is the Egyptian future (هيوصل, هنبعت); no other target forms its future this way
        if HA_FUT.match(w) and not msa_valid and "ha_future" not in V8_OFF:
            bare = HA_FUT.sub(r"\1", w)
            tot_w = sum(self.cnt[t].get(w, 0) for t in self.cnt)
            # a frequent form must be mainly Egyptian (هيدي is Levantine 'this', هيجي Iraqi 'like this', هيبه a noun);
            # a rare form needs its بـ-imperfect twin or bare form in Egyptian text (هيوصل <- بيوصل)
            egy_main = self.rate("EGY", w) >= 0.5 * self.peak(w) if tot_w >= 5 else \
                (self.cnt["EGY"].get("ب" + bare, 0) + self.cnt["EGY"].get(bare, 0) >= 2)
            if egy_main and sum(self.cnt[t].get(bare, 0) for t in self.cnt) >= 3:
                if bits["EGY"] is None:
                    bits["EGY"] = 1
                for t in DIA_T:
                    if t != "EGY" and bits[t] is None and t not in dia_t:
                        bits[t] = 0
                src.append("ha_future")
        # 7 colloquial verb morphology
        if COLLOQ_VERB.match(w) and not msa_valid and bits["MSA"] is None:
            bits["MSA"] = self.msa_bit(w, dia_t)
            src.append("morph")
        # 8 host prior
        if host in TARGETS and bits[host] is None:
            bits[host] = 1
            src.append("host")
        # v8: "other dialect" only for words natural in none of the nine targets (MSA included)
        if odia and any(v == 1 for v in bits.values()):
            odia = 0
        return {"bits": bits, "frm": frm, "odia": odia, "other": 0, "src": "+".join(src) or "none"}

    PROCLITIC = {"و": "w", "ف": "f", "ب": "b", "ل": "l", "ك": "k"}

    def core_lookup(self, w):
        if w in self.core:
            return self.core[w]
        for k in (1, 2):
            pre, rest = w[:k], w[k:]
            if len(rest) >= 2 and all(c in self.PROCLITIC for c in pre):
                e = self.core.get(rest)
                if e and all(self.PROCLITIC[c] in e["clitics"] for c in pre):
                    return e
        return None

    _tl_cache = {}

    def _t_lex(self, t):
        if t not in self._tl_cache:
            self._tl_cache[t] = {f for f, ts in self.lex_t.items() if t in ts}
        return self._tl_cache[t]

    def label(self, tokens, host=None):
        return [self.label_word(w, host) for w in tokens]
