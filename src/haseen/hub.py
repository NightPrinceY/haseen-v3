"""Find the model: an explicit folder, $HASEEN_MODEL, or HaSeen-3-ONNX from Hugging Face (downloaded once, then cached).

After the first download HaSeen runs fully offline: the cached copy is used without touching the network."""
import os, sys

ONNX_REPO = "NightPrince/HaSeen-3-ONNX"
TORCH_REPO = "NightPrince/HaSeen-3"
REVISION = os.environ.get("HASEEN_REVISION", "main")
# a cached snapshot only counts if it holds the weights (a cache can contain just a README, e.g. after reading a card)
WEIGHTS = {ONNX_REPO: ("model.int8.onnx",), TORCH_REPO: ("model.safetensors",)}
ANY_WEIGHTS = ("model.int8.onnx", "model.safetensors", "model.pt")


def has_weights(path, repo=None):
    names = WEIGHTS.get(repo or "", ANY_WEIGHTS)
    return any(os.path.exists(os.path.join(path, n)) for n in names)


def model_path(model=None, repo=ONNX_REPO, quiet=False):
    """Return a local folder holding the model files (weights included)."""
    model = model or os.environ.get("HASEEN_MODEL")
    if model and os.path.isdir(model):
        return model
    if model and "/" in model and not os.path.exists(model):
        repo = model  # a Hub repo id
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(repo, revision=REVISION, local_files_only=True)
        if has_weights(path, repo):
            return path
    except Exception:
        pass
    if not quiet:
        print(f"haseen: downloading {repo} (once, then cached; ~164 MB for the ONNX model)…", file=sys.stderr)
    try:
        path = snapshot_download(repo, revision=REVISION)
    except Exception as e:
        raise RuntimeError(f"could not download {repo} ({type(e).__name__}). The repo is private: set HF_TOKEN (or run `hf auth login`), connect once and run "
                           "`haseen download`, or set HASEEN_MODEL to a local model folder.") from e
    if not has_weights(path, repo):  # offline, the Hub client can hand back the incomplete cached copy
        raise RuntimeError(f"could not download {repo}: the cached copy at {path} has no model weights. Connect to the "
                           "internet once and run `haseen download`, or set HASEEN_MODEL to a local model folder.")
    return path
