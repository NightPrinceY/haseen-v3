"""Shared text utilities for the pilot (normalization, tokenization, labels)."""
import re

LABELS = ["EGY", "GLF", "MSA", "SHARED", "OTHER"]
L2I = {l: i for i, l in enumerate(LABELS)}

_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
# Arabic letters only (excludes Arabic punctuation such as ، ؛ ؟ which live in the same block)
_LETTER = r"ء-يٱ-ۓ\w"
_PUNCT_EDGES = re.compile(rf"^[^{_LETTER}]+|[^{_LETTER}]+$")
_ARABIC = re.compile(r"[ء-يٱ-ۓ]")


def norm(w: str) -> str:
    """CODA-like light normalization used for matching and as model input."""
    w = _DIAC.sub("", w).replace("ـ", "")
    w = re.sub("[إأآٱ]", "ا", w)
    w = w.replace("ى", "ي").replace("ة", "ه")
    return w


def words(sentence: str):
    """Whitespace split, strip punctuation at word edges, drop empty / non-Arabic-only tokens."""
    out = []
    for raw in sentence.split():
        w = _PUNCT_EDGES.sub("", raw)
        if w and _ARABIC.search(w):
            out.append(w)
    return out
