// node tests/js_parity.js < case.json : demo/web/haseen.js decide() must equal the Python build_result()
const H = require("../demo/web/haseen.js");
let raw = ""; process.stdin.on("data", (c) => (raw += c)).on("end", () => {
  const c = JSON.parse(raw); let bad = 0;
  const words = H.splitWords(c.text);
  for (const t of H.TARGETS) {
    const js = H.decide(words, c.first, c.variety, c.odia, c.other, c.cal, t, {});
    const py = c.res[t];
    js.tokens.forEach((a, i) => {
      const b = py.tokens[i];
      const ka = JSON.stringify([a.conforms, a.label || null, a.labels || null]), kb = JSON.stringify([b.conforms, b.label || null, b.labels || null]);
      if (ka !== kb) { bad++; console.log(t, a.word, ka, kb); }
    });
    if (JSON.stringify(js.summary.reads_as) !== JSON.stringify(py.summary.reads_as) || js.summary.adherence !== py.summary.adherence) { bad++; console.log(t, "summary", js.summary, py.summary); }
  }
  console.log(bad ? `MISMATCH ${bad}` : "parity ok"); process.exit(bad ? 1 : 0);
});
