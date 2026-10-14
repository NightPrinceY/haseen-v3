"""Spike 3: probe ready-made ALDi-Token-DI + compare backbone tokenizers and CPU latency."""
import time, itertools, statistics, re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from datasets import load_dataset

torch.set_num_threads(4)
TAGS = ["ambiguous", "lang1", "lang2", "mixed", "ne", "other"]  # lang1=MSA, lang2=DA

SENTS = {
    "task example (EGY + GLF أبغى + MSA سوف)": "أنا أبغى أروح الاجتماع بكرة بس سوف أتأخر شوية",
    "pure EGY": "أنا عايز أروح الاجتماع بكرة بس هتأخر شوية",
    "EGY + GLF (وايد، الحين)": "الخدمة كانت وايد حلوة بس الحين مش عارف أوصل للموظف",
    "EGY + MSA (لن، الذي)": "معلش يا فندم لن نستطيع نغير الميعاد الذي حضرتك اخترته",
    "pure MSA": "سوف نقوم بإرسال الفاتورة إلى بريدك الإلكتروني خلال يومين",
    "pure GLF": "ابغى اعرف وش صار على طلبي لأني انتظرت وايد",
}

name = "VARabi/ALDi-Token-DI"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForTokenClassification.from_pretrained(name, num_labels=len(TAGS)).eval()


def tag(text):
    words = text.split()
    enc = tok(words, is_split_into_words=True, return_tensors="pt")
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)[0]
    out, seen = [], set()
    for i, w in enumerate(enc.word_ids()):
        if w is None or w in seen:
            continue
        seen.add(w)
        p = probs[i]
        out.append(f"{words[w]}:{TAGS[int(p.argmax())]}({p.max():.2f})")
    return out


print("=== ALDi-Token-DI (MSA=lang1, DA=lang2) ===")
for k, s in SENTS.items():
    print(f"[{k}]\n   " + "  ".join(tag(s)))

# Latency on CPU (4 threads), 15-word sentence
s = SENTS["task example (EGY + GLF أبغى + MSA سوف)"] + " لو سمحت ابعتلي الفاتورة"
enc = tok(s, return_tensors="pt")
for _ in range(5):
    model(**enc)
ts = []
for _ in range(50):
    t = time.perf_counter(); model(**enc); ts.append((time.perf_counter() - t) * 1000)
ts.sort()
print(f"\nMARBERT-base token clf CPU(4 threads) fp32: p50={statistics.median(ts):.1f}ms p95={ts[int(.95*len(ts))-1]:.1f}ms (len {enc['input_ids'].shape[1]} subwords)")

# Tokenizer fertility on real Egyptian text
ds = load_dataset("Omartificial-Intelligence-Space/FineWeb2-Egyptian-Arabic", split="train", streaming=True)
ar = re.compile(r"^[؀-ۿ]+$")
words = []
for r in itertools.islice(ds, 3000):
    words += [w for w in r["text"].split() if ar.match(w)]
    if len(words) > 30000:
        break
words = words[:30000]
print(f"\n=== subwords per word on {len(words)} Egyptian words (lower = better) ===")
for m in ["CAMeL-Lab/bert-base-arabic-camelbert-mix", "UBC-NLP/MARBERTv2", "jhu-clsp/mmBERT-small", "rahbi/alclam-base-v2", "aubmindlab/bert-base-arabertv02-twitter"]:
    try:
        t = AutoTokenizer.from_pretrained(m)
        n = sum(len(t.tokenize(w)) for w in words)
        unk = sum(1 for w in words if t.unk_token and t.unk_token in t.tokenize(w))
        print(f"{m:45s} fertility={n/len(words):.2f}  unk_words={100*unk/len(words):.2f}%  vocab={t.vocab_size}")
    except Exception as e:
        print(m, "ERR", repr(e)[:120])
