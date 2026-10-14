// HaSeen 3 in the browser: the same normalization, word splitting, WordPiece tokenization and decision policy as the
// Python package (src/haseen/text.py, decision.py), so results match HaSeen-3-ONNX run by onnxruntime in Python.
(function (root) {
  const DIAC = /[ؐ-ًؚ-ٰٟۖ-ۭ]/g;
  const ARABIC = /[ء-يٱ-ۓ]/;
  const EDGE = /^[^ء-ْٰٱ-ۓ\p{L}\p{M}\p{N}_]+|[^ء-ْٰٱ-ۓ\p{L}\p{M}\p{N}_]+$/gu; // harakat / tanween stay in the word
  const VARS = ["MSA", "EGY", "SAU", "GLF", "LEV", "IRQ", "MAG"];
  const TARGETS = ["MSA", "EGY", "SAU", "GLF", "LEV", "IRQ"];
  const INSIDE = { GLF: ["SAU"] }; // decision.py INSIDE
  const NAME = { MSA: "Arabic / MSA", EGY: "Arabic / Egyptian", SAU: "Arabic / Saudi", GLF: "Arabic / Gulf",
    LEV: "Arabic / Levantine", IRQ: "Arabic / Iraqi", MAG: "Arabic / Maghrebi", OTHER_DIA: "Arabic / other dialect" };
  const FUTURE = { EGY: "هـ + الفعل", LEV: "رح + الفعل", SAU: "بـ / راح + الفعل", GLF: "بـ / راح + الفعل",
    IRQ: "راح + الفعل" };
  const CMAP = { GLF: ["SAU", "KHL", "YEM"] };

  function norm(w) { // text.py norm()
    return w.replace(DIAC, "").replace(/ـ/g, "").replace(/[إأآٱ]/g, "ا")
      .replace(/ى/g, "ي").replace(/ة/g, "ه");
  }

  function splitWords(text) { // detector.py tokenize(): words with character offsets
    const out = []; const re = /\S+/g; let m;
    while ((m = re.exec(text))) {
      const raw = m[0]; const core = raw.replace(EDGE, "");
      if (!core) continue;
      const start = m.index + raw.indexOf(core);
      out.push({ word: core, start, end: start + core.length });
    }
    return out;
  }

  function isPunct(ch) { // BERT: ASCII non-alphanumerics + Unicode P*
    const c = ch.codePointAt(0);
    if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) return true;
    return /\p{P}/u.test(ch);
  }

  class WordPiece { // BertNormalizer(lowercase, strip accents) + BertPreTokenizer + WordPiece, from tokenizer.json
    constructor(tj) {
      this.vocab = tj.model.vocab; this.unk = this.vocab[tj.model.unk_token];
      this.maxChars = tj.model.max_input_chars_per_word || 100;
      this.cls = this.vocab["[CLS]"]; this.sep = this.vocab["[SEP]"];
    }
    normalize(s) {
      s = s.replace(/[\u0000�]|\p{Cc}/gu, (c) => (/\s/.test(c) ? " " : ""));
      return s.normalize("NFD").replace(/\p{Mn}/gu, "").toLowerCase();
    }
    preTokens(s) {
      const out = []; let cur = "";
      for (const ch of s) {
        if (/\s/.test(ch)) { if (cur) out.push(cur); cur = ""; }
        else if (isPunct(ch)) { if (cur) out.push(cur); out.push(ch); cur = ""; }
        else cur += ch;
      }
      if (cur) out.push(cur);
      return out;
    }
    pieces(tok) {
      const chars = Array.from(tok);
      if (chars.length > this.maxChars) return [this.unk];
      const ids = []; let start = 0;
      while (start < chars.length) {
        let end = chars.length, found = null;
        while (start < end) {
          let sub = chars.slice(start, end).join("");
          if (start > 0) sub = "##" + sub;
          if (sub in this.vocab) { found = this.vocab[sub]; break; }
          end--;
        }
        if (found === null) return [this.unk];
        ids.push(found); start = end;
      }
      return ids;
    }
    encodeWords(words, maxLen) { // like tokenizer(words, is_split_into_words=True): ids + first-subword index per word
      const ids = [this.cls]; const first = [];
      for (const w of words) {
        const sub = [];
        for (const t of this.preTokens(this.normalize(w))) sub.push(...this.pieces(t));
        first.push(sub.length && ids.length < maxLen - 1 ? ids.length : -1);
        for (const id of sub) if (ids.length < maxLen - 1) ids.push(id);
      }
      ids.push(this.sep);
      return { ids, first };
    }
  }

  const sig = (x) => 1 / (1 + Math.exp(-x));

  function suggest(word, t, cmap) { // decision.py Suggester
    const w = norm(word);
    if (t !== "MSA" && ["سوف", "وسوف", "فسوف"].includes(w)) return FUTURE[t] || null;
    if (t === "EGY" && w.length >= 4 && w[0] === "س" && "ينت".includes(w[1])) return "ه" + word.slice(1);
    for (const m of CMAP[t] || [t]) {
      const forms = (((cmap || {})[w] || {})[m] || []).filter((f) => f !== w);
      if (forms.length) return forms[0];
    }
    return null;
  }

  function decideWord(pv, pd, po, t, cal) { // decision.py decide()
    const pt = pv[t];
    if (po > 0.5) return { ok: true, prim: null, set: [], conf: null };
    if (pd > cal.tau_odia && pt < (cal.odia_gate ?? 1)) return { ok: false, prim: "OTHER_DIA", set: ["OTHER_DIA"], conf: pd };
    if (pt < cal.tau[t]) {
      const others = VARS.filter((v) => v !== t && !(INSIDE[t] || []).includes(v)).sort((a, b) => pv[b] - pv[a]);
      let S = others.filter((v) => pv[v] >= 0.5); if (!S.length) S = [others[0]];
      return { ok: false, prim: S[0], set: S, conf: 1 - pt };
    }
    return { ok: true, prim: null, set: [], conf: pt };
  }

  // decide(): logits per first subword -> result JSON (decision.py build_result)
  function decide(words, first, variety, odia, other, cal, target, cmap) {
    const tokens = []; const deviations = []; let arabic = 0, informative = 0; const caller = {}, callerAll = {};
    VARS.forEach((v) => { caller[v] = 0; callerAll[v] = 0; });
    const nv = VARS.length;
    words.forEach((wd, i) => {
      const item = { word: wd.word, index: i, start: wd.start, end: wd.end, conforms: true };
      const p = first[i];
      if (p >= 0 && ARABIC.test(wd.word)) {
        const pv = {}; VARS.forEach((v, k) => (pv[v] = sig(variety[p * nv + k] / cal.T_var)));
        const pd = sig(odia[p] / cal.T_odia), po = sig(other[p]);
        item.p = { ...pv, other_dialect: pd, name: po };
        const d = decideWord(pv, pd, po, target, cal);
        if (po <= 0.5) {
          arabic++; VARS.forEach((v) => (callerAll[v] += pv[v]));
          if (Math.min(...VARS.map((v) => pv[v])) < 0.5) { informative++; VARS.forEach((v) => (caller[v] += pv[v])); }
        }
        if (!d.ok) {
          Object.assign(item, { conforms: false, label: NAME[d.prim], labels: d.set.map((x) => NAME[x]), confidence: +d.conf.toFixed(3) });
          const s = d.prim !== "OTHER_DIA" ? suggest(wd.word, target, cmap) : null; if (s) item.suggestion = s;
        } else if (d.conf !== null) item.confidence = +d.conf.toFixed(3);
      }
      if (!item.conforms) deviations.push(wd.word);
      tokens.push(item);
    });
    const summary = { arabic_words: arabic, deviating_words: deviations.length,
      adherence: arabic ? +(1 - deviations.length / arabic).toFixed(4) : 1 };
    if (arabic) {
      const sc = informative ? caller : callerAll, k = informative || arabic;
      summary.reads_as = VARS.slice().sort((a, b) => sc[b] - sc[a]).slice(0, 3).map((v) => ({ variety: NAME[v], score: +(sc[v] / k).toFixed(3) }));
      summary.informative_words = informative;
    }
    return { target: NAME[target], tokens, deviations, summary };
  }

  root.HaSeen = { norm, splitWords, WordPiece, decide, ARABIC, VARS, TARGETS, NAME };
  if (typeof module !== "undefined") module.exports = root.HaSeen;
})(typeof window !== "undefined" ? window : globalThis);
