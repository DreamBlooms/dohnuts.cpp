#!/usr/bin/env bash
# Merge a Linnaeus checkpoint (LoRA + scalar head) and convert it to quantized GGUF.
#
# Usage: build_linnaeus_gguf.sh <base_dir> <checkpoint_dir> <out_dir> [llama_cpp_dir]
set -euo pipefail

BASE=${1:?base model directory (Qwen3.5-2B)}
CHECKPOINT=${2:?Linnaeus checkpoint directory (adapter + linnaeus.json)}
OUT=${3:?output directory}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${4:-$ROOT/third_party/llama.cpp}

PYTHON=${PYTHON:-python3}
NAME=${NAME:-Linnaeus-0.1.0-2B}
mkdir -p "$OUT"

# llama-quantize is not part of the library build; build it on demand.
QUANTIZE=$LLAMA/build/bin/llama-quantize
if [ ! -x "$QUANTIZE" ]; then
    echo "==> building llama-quantize"
    cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
    cmake --build "$LLAMA/build" -j --target llama-quantize
fi

echo "==> merging LoRA (bf16) and exporting the head"
"$PYTHON" "$ROOT/scripts/export_linnaeus.py" --base "$BASE" --adapter "$CHECKPOINT" \
    --out "$OUT/linnaeus-merged" --head "$OUT/head.f32"

echo "==> converting to f16 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$OUT/linnaeus-merged" \
    --outfile "$OUT/$NAME-F16.gguf" --outtype f16 --no-mtp

echo "==> converting vision encoder (mmproj)"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$OUT/linnaeus-merged" \
    --mmproj --outfile "$OUT/mmproj-$NAME-bf16.gguf" --outtype bf16

echo "==> quantizing Q8_0"
"$QUANTIZE" "$OUT/$NAME-F16.gguf" "$OUT/$NAME-Q8_0.gguf" Q8_0

echo "==> quantizing Q4_K_M"
"$QUANTIZE" "$OUT/$NAME-F16.gguf" "$OUT/$NAME-Q4_K_M.gguf" Q4_K_M

echo "==> done: $OUT"
