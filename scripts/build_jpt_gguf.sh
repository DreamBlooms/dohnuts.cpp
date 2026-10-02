#!/usr/bin/env bash
# Convert kirp/jpt-4b (a merged LoRA decision model on Qwen3.5-4B) to a
# quantized GGUF. The vision tower is dropped; only the text path is used.
#
# Usage: build_jpt_gguf.sh <model_dir> <out_file> [llama_cpp_dir]
set -euo pipefail

MODEL=${1:?jpt-4b model directory (config.json + model.safetensors)}
OUT=${2:?output .gguf path}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${3:-$ROOT/third_party/llama.cpp}

PYTHON=${PYTHON:-python3}
mkdir -p "$(dirname "$OUT")"

echo "==> converting to Q8_0 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$MODEL" \
    --outfile "$OUT" --outtype q8_0 --no-mtp

echo "==> done: $OUT"
