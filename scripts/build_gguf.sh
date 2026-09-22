#!/usr/bin/env bash
# Convert a merged Dohnuts export to GGUF (f16) and quantize to Q8_0 / Q4_K_M.
#
# Usage: build_gguf.sh <merged_dir> <out_dir> [llama_cpp_dir]
set -euo pipefail

MERGED=${1:?merged model directory}
OUT=${2:?output directory}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${3:-$ROOT/third_party/llama.cpp}

PYTHON=${PYTHON:-python3}
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

echo "==> converting to f16 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$MERGED" \
    --outfile "$OUT/dohnuts-f16.gguf" --outtype f16 --no-mtp

# The vision tower is separate and optional; convert it only when present.
if "$PYTHON" -c "import json,sys; sys.exit(0 if 'vision_config' in json.load(open('$MERGED/config.json')) else 1)"; then
    echo "==> converting vision encoder (mmproj)"
    "$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$MERGED" \
        --mmproj --outfile "$OUT/mmproj-dohnuts-0.1.0-bf16.gguf" --outtype bf16
fi

echo "==> quantizing Q8_0"
"$QUANTIZE" "$OUT/dohnuts-f16.gguf" "$OUT/dohnuts-Q8_0.gguf" Q8_0

echo "==> quantizing Q4_K_M"
"$QUANTIZE" "$OUT/dohnuts-f16.gguf" "$OUT/dohnuts-Q4_K_M.gguf" Q4_K_M

echo "==> done: $OUT"
