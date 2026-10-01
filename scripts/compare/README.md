# Side-model comparison

Check that a quantized side model in dohnuts.cpp agrees with its PyTorch
reference on the same cases. Choices are compared exactly; `noul` and `score`
use a small tolerance (quantization and rounding).

## decider-0.8b

Needs the model directory (contains the `decider/` package and weights):

```sh
PYTHON=../.venv/bin/python scripts/compare/compare.sh decider \
  work/side/decider-0.8b-q8_0.gguf work/side/decider.json work/side/decider-0.8b
```

## thisthat-1.2

Needs a checkout (or `pip install -e`) of
[FLock-io/this-that-model](https://github.com/FLock-io/this-that-model); point
`THISTHAT_SRC` at it if it is not installed:

```sh
CASES=scripts/compare/cases_thisthat.jsonl THISTHAT_SRC=/path/to/this-that-model \
  PYTHON=../.venv/bin/python scripts/compare/compare.sh thisthat \
  work/side/thisthat-1.2-q8_0.gguf work/side/thisthat.json work/side/thisthat-1.2
```

The default `cases.jsonl` only exercises its simpler questions; the model's own
recorded cohort under `cases_thisthat.jsonl` (and the wide-option
`cases_wide.jsonl`) is the more useful check. The two agree on every choice and
`noul`; the only reported difference is the `score` *scalar*, because upstream
returns the argmax level as an integer where the dohnuts answer shape reports
the expected level `sum(k · p[k])`. The probabilities and the argmax agree.

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
cases.jsonl          one /v1/systemone request per line (the shared smoke set)
cases_thisthat.jsonl this-that-model's recorded cohort, grouped by state
cases_wide.jsonl     a 12-option choice plus a mixed request (thisthat)
ref_decider.py       PyTorch reference (decider.infer.Decider)
ref_thisthat.py      PyTorch reference (thisthat.TypedDecider)
ref_kev.py           PyTorch reference (kev.checkpoint.Checkpoint)
score.py             per-case agreement report
compare.sh           runs both and reports
```
