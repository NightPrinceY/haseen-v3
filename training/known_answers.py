"""Known-answer sanity suite (the task's example + guideline core cases), run on a training run with a policy file.

Usage: python known_answers.py <run_dir with config.json + best/model.pt> <policy.json>
Release gate (PREREG addendum 3 / PREREG_V3): pass the task example and >= 95% of the word checks."""
import json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from haseen.detector import Detector
CASES = [  # (text, target, words that must deviate, words that must conform)
    ("أنا أبغى أروح الاجتماع بكرة بس سوف أتأخر شوية", "egy", ["أبغى", "سوف"], ["أروح", "الاجتماع", "بكرة", "شوية"]),
    ("ابشر يا غالي، الحين بنسوي لك الطلب ونرسل لك الكود", "egy", ["الحين"], ["الطلب", "الكود"]),
    ("ابشر يا غالي، الحين بنسوي لك الطلب ونرسل لك الكود", "sau", [], ["الحين", "الطلب", "الكود"]),
    ("شو المشكلة اللي عندك؟ بدي ساعدك هلق", "egy", ["شو", "هلق"], ["المشكلة"]),
    ("شو المشكلة اللي عندك؟ بدي ساعدك هلق", "lev", [], ["شو", "بدي", "هلق", "المشكلة"]),
    ("حضرتك الطلب بتاعك هيوصل بكرة الصبح، وهل تحب نبعتلك رسالة تأكيد؟", "egy", [], ["حضرتك", "بتاعك", "هيوصل", "وهل", "رسالة"]),
    ("هل تحب نبعتلك الكود دلوقتي؟", "egy", [], ["هل", "نبعتلك", "دلوقتي"]),
    ("سيتم تحويلك للقسم المختص، وسنرسل لك رسالة", "egy", ["سيتم", "وسنرسل"], ["تحويلك", "المختص", "رسالة"]),
    ("سيتم تحويلك للقسم المختص، وسنرسل لك رسالة", "msa", [], ["سيتم", "وسنرسل", "تحويلك"]),
    ("الفني سيصل لحضرتك الساعة اتنين، ولو فيه أي مشكلة كلمنا", "egy", ["سيصل"], ["الفني", "اتنين", "مشكلة"]),
    ("يمكنك دفع الفاتورة عبر التطبيق، ولن تحتاج إلى زيارة الفرع", "msa", [], ["يمكنك", "الفاتورة", "التطبيق", "ولن"]),
    ("يمكنك دفع الفاتورة عبر التطبيق، ولن تحتاج إلى زيارة الفرع", "egy", ["ولن"], ["الفاتورة", "التطبيق"]),
    ("واش بغيتي نصيفط ليك الكود دابا؟", "egy", ["واش", "بغيتي", "دابا"], ["الكود"]),
    ("شكو ماكو؟ هسه أدزلك الرابط", "egy", ["شكو", "ماكو", "هسه"], ["الرابط"]),
    ("شلونك عيني؟ هسه أدزلك الرابط", "irq", [], ["شلونك", "هسه", "الرابط"]),
    ("ابشر يا غالي، الحين بنسوي لك الطلب ونرسل لك الكود", "glf", [], ["الحين", "بنسوي", "الطلب"]),
    ("شو تبي؟ الحين نسويها لك", "glf", [], ["شو", "تبي", "الحين"]),
    ("شو تبي؟ الحين نسويها لك", "sau", ["شو"], ["تبي", "الحين"]),
    ("سنقوم بمعالجة طلبك وسيتم التواصل معك", "glf", ["سنقوم", "وسيتم"], ["طلبك"]),
    ("حضرتك الطلب بتاعك هيوصل بكرة", "glf", ["حضرتك", "بتاعك", "هيوصل"], ["الطلب"]),
]
run, pol = sys.argv[1], sys.argv[2]
d = os.path.join(tempfile.gettempdir(), "haseen_ka", os.path.basename(os.path.normpath(run)))
os.makedirs(d, exist_ok=True)
shutil.copy(f"{run}/config.json", d)
p = json.load(open(pol)); json.dump({k: p[k] for k in ("T_var", "T_odia", "tau", "tau_odia", "odia_gate")}, open(f"{d}/calibration.json", "w"))
if not os.path.exists(f"{d}/model.pt"):
    os.symlink(os.path.abspath(f"{run}/best/model.pt"), f"{d}/model.pt")
det = Detector(d)
ok = tot = 0
for text, t, dev, conf in CASES:
    r = det.detect(text, t)
    got = {x["word"]: x["conforms"] for x in r["tokens"]}
    for w in dev:
        tot += 1; ok += got.get(w) is False
        if got.get(w) is not False: print(f"  MISS  {t} {w!r} should deviate")
    for w in conf:
        tot += 1; ok += got.get(w) is True
        if got.get(w) is not True: print(f"  FALSE {t} {w!r} should conform")
print(f"{os.path.basename(run)}: {ok}/{tot} known answers")
