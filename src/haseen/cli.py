"""The `haseen` command.

    haseen "أبغى أساعدك، سوف أحوّلك للقسم المختص الحين"               # target Egyptian by default
    haseen "شو بدك نعمل هلق؟" -t sau --json                           # result JSON
    cat replies.txt | haseen -t egy                                  # one sentence per line -> JSON lines
    haseen --jsonl in.jsonl > out.jsonl                              # rows {"text", "target"?}
    haseen serve --port 8000                                         # HTTP: POST /detect
    haseen download | haseen info
"""
import argparse, json, os, sys

from . import __version__

from .decision import NAME, SHORT as _S

# identity colours (HaSeen brand ramps), light steps so they read on dark and light terminals
_RGB = {"Arabic / MSA": (133, 193, 225), "Arabic / Egyptian": (246, 160, 107), "Arabic / Saudi": (174, 191, 146),
        "Arabic / Gulf": (143, 160, 115), "Arabic / Maghrebi": (255, 198, 165), "Arabic / Levantine": (223, 175, 90), "Arabic / Iraqi": (204, 219, 178),
        "Arabic / Yemeni": (238, 207, 156), "Arabic / Moroccan": (255, 198, 165), "Arabic / Tunisian": (181, 218, 239),
        "Arabic / other dialect": (222, 164, 190)}
_SHORT = {NAME[k]: v for k, v in _S.items()}


def _paint(s, rgb, on):
    return f"\033[1;38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{s}\033[0m" if on else s


def _dim(s, on):
    return f"\033[2m{s}\033[0m" if on else s


def pretty(res, color):
    toks, devs = res["tokens"], [t for t in res["tokens"] if not t["conforms"]]
    head = f"HaSeen · target {_SHORT[res['target']]} · {len(devs)} of {len(toks)} words deviate"
    if res.get("meta", {}).get("latency_ms") is not None:
        head += f" · {res['meta']['latency_ms']} ms"
    lines = [_dim(head, color), ""]
    lines.append("  " + " ".join(t["word"] if t["conforms"] else _paint(t["word"], _RGB[t["label"]], color) for t in toks))
    lines.append("")
    if not devs:
        lines.append(f"  ✓ every word fits {_SHORT[res['target']]}")
    width = max([len(t["word"]) for t in devs] + [4])
    for t in devs:
        also = [_SHORT[x] for x in t.get("labels", [])[1:]]
        row = f"  {t['word']:<{width}}  {_paint(_SHORT[t['label']].ljust(13), _RGB[t['label']], color)} {t['confidence']:.3f}"
        if also:
            row += _dim(f"  (also {', '.join(also)})", color)
        if t.get("suggestion"):
            row += f"   → {t['suggestion']}"
        lines.append(row)
    sm = res.get("summary", {})
    if sm.get("reads_as"):
        lines.append("")
        lines.append(_dim(f"  adherence {sm['adherence']:.0%} · reads as " +
                          ", ".join(f"{_SHORT[r['variety']]} {r['score']:.2f}" for r in sm["reads_as"]), color))
    if res.get("meta", {}).get("note"):
        lines.append(_dim("  note: " + res["meta"]["note"], color))
    return "\n".join(lines)


def _serve(argv):
    ap = argparse.ArgumentParser(prog="haseen serve", description="HTTP API: POST /detect {text | texts, target}")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"])
    a = ap.parse_args(argv)
    from .server import serve
    serve(a.host, a.port, model=a.model, backend=a.backend)


def _download(argv):
    ap = argparse.ArgumentParser(prog="haseen download", description="fetch the model now (later runs are offline)")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    a = ap.parse_args(argv)
    from .hub import ONNX_REPO, TORCH_REPO, model_path
    print(model_path(repo=TORCH_REPO if a.backend == "torch" else ONNX_REPO))


def _info(_argv):
    from .hub import ONNX_REPO, model_path
    try:
        path = model_path(repo=ONNX_REPO, quiet=True) if not os.environ.get("HASEEN_MODEL") else os.environ["HASEEN_MODEL"]
    except Exception as e:  # not downloaded yet and offline
        path = f"not available ({type(e).__name__})"
    print(json.dumps({"haseen": __version__, "model": ONNX_REPO, "model_path": path,
                      "targets": ["msa", "egy", "glf", "sau", "lev", "irq"], "labels": list(_SHORT)}, ensure_ascii=False, indent=1))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    sub = {"serve": _serve, "download": _download, "info": _info}
    if argv and argv[0] in sub:
        return sub[argv[0]](argv[1:])
    ap = argparse.ArgumentParser(prog="haseen", description="HaSeen 3: which Arabic words do not belong to the target variety?",
                                 epilog="subcommands: haseen serve | haseen download | haseen info")
    ap.add_argument("text", nargs="?", help="Arabic text (omit to read one sentence per line from stdin)")
    ap.add_argument("-t", "--target", default="egy", help="msa | egy | glf (Gulf: Saudi, Gulf-east, Yemeni, Omani) | sau | lev | irq (or 'Egyptian', 'gulf', ...); default egy")
    ap.add_argument("--json", action="store_true", help="print the task JSON instead of the coloured view")
    ap.add_argument("--probs", action="store_true", help="add per-word probabilities for every variety (ONNX backend)")
    ap.add_argument("--jsonl", help="batch file: one {\"text\", \"target\"?} per line -> JSON lines on stdout")
    ap.add_argument("--model", default=None, help="local model folder or Hub repo id (default: NightPrince/HaSeen-3-ONNX)")
    ap.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"])
    ap.add_argument("--version", action="version", version=f"haseen {__version__}")
    a = ap.parse_args(argv)

    from .api import load
    from .decision import resolve_target
    try:
        resolve_target(a.target)
    except ValueError as e:
        ap.error(str(e))
    if not a.text and not a.jsonl and sys.stdin.isatty():
        ap.error("give a sentence, --jsonl FILE, or pipe sentences on stdin")
    try:
        det = load(a.model, backend=a.backend)
    except (RuntimeError, FileNotFoundError, ImportError) as e:
        ap.exit(1, f"haseen: {e}\n")
    kw = {"return_probs": True} if a.probs and getattr(det, "backend", "") == "onnx" else {}

    if a.jsonl:
        for line in open(a.jsonl, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                print(json.dumps(det.detect(r["text"], r.get("target", a.target), **kw), ensure_ascii=False), flush=True)
    elif a.text:
        res = det.detect(a.text, a.target, **kw)
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else pretty(res, color))
    else:  # stdin, one sentence per line
        for line in sys.stdin:
            if line.strip():
                print(json.dumps(det.detect(line.strip(), a.target, **kw), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
