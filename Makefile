install:          ## install the haseen package (CPU, ONNX; no PyTorch)
	pip install -e .
model:            ## download HaSeen-3-ONNX once (private repo: needs HF_TOKEN); later runs are offline
	haseen download
bench:            ## download HaSeen-3-Bench (private) into data/bench
	hf download NightPrince/HaSeen-3-Bench --repo-type dataset --local-dir data/bench
data:             ## download the training mix (private)
	hf download NightPrince/HaSeen-3-Mix --repo-type dataset --local-dir data/train
test:             ## unit tests (no model needed) + SDK smoke test if a model is available
	pytest -q tests
eval-public:      ## the public evaluation set in this repo (200 LLM agent replies, word-level gold)
	python scripts/evaluate.py --bench data/eval_public --split test
eval:             ## HaSeen-3-Bench test split through the SDK, ONNX int8 on CPU
	python scripts/evaluate.py --split test
demo:             ## one sentence, three targets
	haseen "أبغى أساعدك، سوف أحوّلك للقسم المختص الحين" -t egy
	haseen "أبغى أساعدك، سوف أحوّلك للقسم المختص الحين" -t sau
	haseen "أبغى أساعدك، سوف أحوّلك للقسم المختص الحين" -t msa
serve:            ## HTTP API on :8000 (POST /detect)
	haseen serve --port 8000
