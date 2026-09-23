# Side-model comparison

Check that a quantized side model in dohnuts.cpp agrees with its PyTorch
reference on the same cases. Choices are compared exactly; `noul` and `score`
use a small tolerance (quantization and rounding). Both 0.8B models agree with
their references on all five cases.

## decider-0.8b

Needs the model directory (contains the `decider/` package and weights):

```sh
PYTHON=../.venv/bin/python scripts/compare/compare.sh decider \
  work/side/decider-0.8b-q8_0.gguf work/side/decider.json work/side/decider-0.8b
```

## kev-0.8b

Needs the checkpoint directory (`adapter_model.safetensors` + `head.pt`) and a
checkout of [jaredpalmer/kev](https://github.com/jaredpalmer/kev) on PYTHONPATH
(the directory containing its `kev/` package):

```sh
PYTHONPATH=/path/to/kev-repo PYTHON=../.venv/bin/python \
  scripts/compare/compare.sh kev \
  work/side/kev-0.8b-q8_0.gguf work/side/kev.json work/side/kev-0.8b \
  work/side/kev-head.f32
```

## Files

```
cases.jsonl     one /v1/systemone request per line
ref_decider.py  PyTorch reference (decider.infer.Decider)
ref_kev.py      PyTorch reference (kev.checkpoint.Checkpoint)
score.py        per-case agreement report
compare.sh      runs both and reports
```
