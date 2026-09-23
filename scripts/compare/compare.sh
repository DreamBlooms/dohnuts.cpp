#!/usr/bin/env bash
# Compare a quantized side model running in dohnuts.cpp against its PyTorch
# reference on the shared cases, then print agreement per case.
#
# Usage:
#   scripts/compare/compare.sh decider <model.gguf> <decider.json> <decider-dir>
#   scripts/compare/compare.sh kev <model.gguf> <kev.json> <kev-dir> <head.f32>
#
# The reference runs need their upstream packages importable:
#   decider: the `decider/` folder inside the model download
#   kev:     a checkout of github.com/jaredpalmer/kev (its parent dir is searched)
# Set PYTHON to pick the interpreter (default: python3).
set -euo pipefail

PROFILE=${1:?decider or kev}
GGUF=${2:?gguf}
META=${3:?metadata json}
MODEL_DIR=${4:?pytorch model directory}
HEAD=${5:-}

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
CASES=${CASES:-$DIR/cases.jsonl}

echo "==> dohnuts.cpp ($PROFILE)"
"$ROOT/build/dohnuts-cli" --model "$GGUF" --metadata "$META" ${HEAD:+--head "$HEAD"} \
    --input "$CASES" > "$DIR/cpp_$PROFILE.jsonl" 2>/dev/null

if [ "$PROFILE" = "decider" ]; then
    echo "==> PyTorch reference"
    "$PYTHON" "$DIR/ref_decider.py" "$MODEL_DIR" "$CASES" > "$DIR/py_$PROFILE.jsonl"
else
    echo "==> PyTorch reference"
    "$PYTHON" "$DIR/ref_kev.py" "$MODEL_DIR" "$CASES" > "$DIR/py_$PROFILE.jsonl"
fi

echo "==> agreement"
"$PYTHON" "$DIR/score.py" "$DIR/cpp_$PROFILE.jsonl" "$DIR/py_$PROFILE.jsonl"
