"""Crawl every term page of mo3jam.com (crowd-sourced Arabic dialect dictionary) and parse it.

  python scripts/crawl_mo3jam.py fetch   -> data/sources/mo3jam/raw/<sha1>.html.gz (resumable, 10 workers, about 6 req/s)
  python scripts/crawl_mo3jam.py parse   -> data/sources/mo3jam/definitions.jsonl (one row per definition)

Term list: data/sources/mo3jam/terms_index.json, built from the 51 /index/<letter> pages (23,114 terms, 31,608
definitions on 2026-10-13). robots.txt allows /term and /index (it disallows /search, /profile, /home).
en.mo3jam.com and ar.mo3jam.com serve the same definitions (same ids); only the interface language differs."""
import concurrent.futures as cf, gzip, hashlib, html, json, os, re, sys, time, urllib.parse, urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "sources", "mo3jam")
RAW = os.path.join(ROOT, "raw")
UA = "Mozilla/5.0 (HaSeen research crawler; one request per term; github.com/NightPrinceY)"


def path(term):
    return os.path.join(RAW, hashlib.sha1(term.encode()).hexdigest() + ".html.gz")


def fetch(term):
    p = path(term)
    if os.path.exists(p):
        return "skip"
    url = "https://en.mo3jam.com/term/" + urllib.parse.quote(term)
    err = ""
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
            with gzip.open(p + ".tmp", "wb") as f:
                f.write(json.dumps({"term": term, "url": url}).encode() + b"\n" + body)
            os.replace(p + ".tmp", p)
            time.sleep(0.25)
            return "ok"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 * (k + 1))
    return "fail " + err


def txt(s):
    s = re.sub(r"<br\s*/?>", "\n", s or "")
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    return "\n".join(" ".join(l.split()) for l in s.split("\n") if l.strip())


def parse_page(raw):
    head, body = raw.split(b"\n", 1)
    meta = json.loads(head)
    h = body.decode("utf-8", "replace")
    out = []
    for m in re.finditer(r'<article id="d(\d+)" class="def ?([^"]*)">(.*?)</article>', h, re.S):
        b = m.group(3)
        g = lambda p: (re.search(p, b, re.S) or [None, ""])[1]
        dia = g(r'<div class="dialects">(.*?)</div>')
        kind = (re.search(r"<span>\s*(\w+) in\s*</span>", dia) or [None, ""])[1]
        rel = re.findall(r'<li><a href="/term/[^"#]*#d(\d+)"><span class="term-flags">(.*?)</span></span>\s*([^<]*)</a>', g(r'<ol class="related-terms">(.*?)</ol>'))
        votes = re.findall(r'vote-(down|up)"><span class="digit-text">(\d+)', b)
        user = g(r'<span class="def-user">(.*?)</span>') or (re.search(r'href="/profile/([^"]+)"', b) or [None, ""])[1]
        out.append({
            "def_id": int(m.group(1)), "term": txt(g(r'<h2 class="def-term">(.*?)</h2>')) or meta["term"],
            "index_term": meta["term"], "kind": kind, "dialects": m.group(2).split(),
            "translit": txt(g(r'def-term-tr[^>]*>(.*?)</span>')),
            # one block per language (ar / en / fr); the English site shows en first when it exists
            "defs": {c.split('"', 1)[0].split()[0]: {
                "definition": txt((re.search(r'<div class="def-body">(.*?)</div>', c, re.S) or [None, ""])[1]),
                "example": txt((re.search(r'<div class="example">(.*?)</div>', c, re.S) or [None, ""])[1])}
                for c in b.split('<div class="def-content ')[1:]},
            "related": [{"def_id": int(i), "flags": re.findall(r'flag (\w+)', f), "term": t.strip()} for i, f, t in rel],
            "votes_down": int(dict((k, v) for k, v in votes).get("down", 0)),
            "votes_up": int(dict((k, v) for k, v in votes).get("up", 0)),
            "user": txt(user), "date": g(r'<time datetime="([^"]+)"'),
        })
    return out


if __name__ == "__main__":
    terms = list(json.load(open(os.path.join(ROOT, "terms_index.json"))))
    if sys.argv[1] == "fetch":
        os.makedirs(RAW, exist_ok=True)
        stats, t0 = {}, time.time()
        with cf.ThreadPoolExecutor(10) as ex:
            for i, r in enumerate(ex.map(fetch, terms)):
                k = r.split()[0]
                stats[k] = stats.get(k, 0) + 1
                if r.startswith("fail"):
                    print("FAIL", terms[i], r, flush=True)
                if i % 1000 == 0:
                    print(i, stats, f"{time.time() - t0:.0f}s", flush=True)
        print("done", stats, flush=True)
    else:
        n = 0
        with open(os.path.join(ROOT, "definitions.jsonl"), "w") as f:
            for t in terms:
                p = path(t)
                if os.path.exists(p):
                    for d in parse_page(gzip.open(p).read()):
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
                        n += 1
        print(n, "definitions")
