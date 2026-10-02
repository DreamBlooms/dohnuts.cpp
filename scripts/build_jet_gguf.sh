#!/usr/bin/env bash
# Convert michaljach/jet (a text-only Qwen3.5 decision model) to a quantized
# GGUF. Vision is not used.
#
# Usage: build_jet_gguf.sh <model_dir> <out_file> [llama_cpp_dir]
set -euo pipefail

MODEL=${1:?jet model directory (config.json + model.safetensors)}
OUT=${2:?output .gguf path}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${3:-$ROOT/third_party/llama.cpp}

PYTHON=${PYTHON:-python3}
mkdir -p "$(dirname "$OUT")"

echo "==> converting to Q8_0 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$MODEL" \
    --outfile "$OUT" --outtype q8_0 --no-mtp

echo "==> done: $OUT"
