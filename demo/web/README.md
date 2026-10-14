---
title: HaSeen 3 Playground
emoji: 🪶
colorFrom: yellow
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: other
short_description: Word-level Arabic dialect checks, 6 targets, in-browser
---
# HaSeen 3 Playground (private)

Type an Arabic sentence, pick the variety it should be in, and see which words do not belong, which varieties they
do belong to, the adherence score and which variety the sentence reads as. The model (HaSeen-3-ONNX, int8, about 160 MB)
runs in your browser with onnxruntime-web: the text never leaves the device. Same decisions as the Python package
(`haseen`, parity test `tests/js_parity.js`).
