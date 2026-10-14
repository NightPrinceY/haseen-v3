"""Public API.

    import haseen
    haseen.detect("أبغى أساعدك، سوف أحوّلك للقسم المختص الحين", "egy")   # ONNX on CPU, model fetched once

    det = haseen.load()                       # keep a detector; same .detect(text, target)
    det = haseen.load(backend="torch")        # HaSeen-3 in PyTorch (pip install "haseen[gpu]"), uses CUDA if present
    det = haseen.load("path/to/model")        # a local HaSeen-3 or HaSeen-3-ONNX folder
"""
import os

from .hub import ONNX_REPO, TORCH_REPO, model_path


def load(model=None, backend="auto", device=None, threads=None, quiet=False):
    """Return a detector. backend: 'auto' (ONNX unless the folder only has PyTorch weights), 'onnx' or 'torch'."""
    if backend not in ("auto", "onnx", "torch"):
        raise ValueError("backend must be 'auto', 'onnx' or 'torch'")
    path = model_path(model, repo=TORCH_REPO if backend == "torch" else ONNX_REPO, quiet=quiet)
    has_onnx = os.path.exists(os.path.join(path, "model.int8.onnx"))
    if backend == "onnx" or (backend == "auto" and has_onnx):
        if not has_onnx:
            raise FileNotFoundError(f"{path} has no model.int8.onnx")
        from .engine import OnnxDetector
        return OnnxDetector(path, threads=threads)
    if backend == "auto" and not any(os.path.exists(os.path.join(path, n)) for n in ("model.safetensors", "model.pt")):
        raise FileNotFoundError(f"no HaSeen model in {path} (expected model.int8.onnx or model.safetensors)")
    try:
        from .detector import Detector as TorchDetector
    except ImportError as e:
        raise ImportError('the PyTorch backend needs: pip install "haseen[gpu]"') from e
    return TorchDetector(path, device=device)


Detector = load  # `haseen.Detector(...)` reads naturally too

_DEFAULT = {}


def detect(text, target="egy", return_probs=False, **load_kw):
    """Check one sentence. The detector is loaded on the first call and reused."""
    key = tuple(sorted(load_kw.items()))
    if key not in _DEFAULT:
        _DEFAULT[key] = load(**load_kw)
    det = _DEFAULT[key]
    if return_probs and getattr(det, "backend", "") == "onnx":
        return det.detect(text, target, return_probs=True)
    return det.detect(text, target)
