"""`haseen serve`: a small HTTP API with the standard library only (no web framework to install).

    POST /detect   {"text": "...", "target": "egy"}             -> task JSON
                   {"texts": ["...", "..."], "target": "egy"}   -> {"results": [...]}
    GET  /health   -> {"status": "ok", "model": ...}

The model is loaded once at start; requests are served in threads (onnxruntime sessions are thread-safe)."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__

MAX_BODY = 1 << 20  # 1 MB


def make_handler(det):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"haseen/{__version__}"

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/health", "/"):
                return self._send(200, {"status": "ok", "haseen": __version__, "model": getattr(det, "name", "HaSeen-3"),
                                        "usage": "POST /detect {\"text\": \"...\", \"target\": \"msa|egy|sau|glf|lev|irq\"}"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/detect":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._send(413 if n > MAX_BODY else 400, {"error": "send a JSON body up to 1 MB"})
            try:
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                target = req.get("target", "egy")
                if "texts" in req:
                    return self._send(200, {"results": [det.detect(x, target) for x in req["texts"]]})
                if not isinstance(req.get("text"), str):
                    return self._send(400, {"error": "field 'text' (string) or 'texts' (list) is required"})
                return self._send(200, det.detect(req["text"], target))
            except ValueError as e:  # bad JSON or unsupported target
                return self._send(400, {"error": str(e)})

        def log_message(self, format, *args):  # noqa: A002 (base-class name)  # one short line per request
            print(f"haseen serve: {self.address_string()} {format % args}")

    return Handler


def serve(host="127.0.0.1", port=8000, model=None, backend="auto"):
    from .api import load
    det = load(model, backend=backend)
    srv = ThreadingHTTPServer((host, port), make_handler(det))
    print(f"haseen serve: {getattr(det, 'name', 'HaSeen-3')} on http://{host}:{port}  (POST /detect, GET /health)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
