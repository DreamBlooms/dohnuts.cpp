#!/usr/bin/env bash
# Convert the NeoHorse-Jev-4B backbone (a Qwen3.5 hybrid) to a quantized GGUF.
# The decision pointer head is separate: export it with
# scripts/export_neohorsejev_head.py and pass it as --head. Vision is not used.
#
# Usage: build_neohorsejev_gguf.sh <model_dir> <out_dir> [llama_cpp_dir]
#   writes <out_dir>/neohorsejev-4b-q8_0.gguf
set -euo pipefail

MODEL=${1:?NeoHorse-Jev-4B directory (backbone/ or root config.json + weights)}
OUTDIR=${2:?output directory}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${3:-$ROOT/third_party/llama.cpp}

PYTHON=${PYTHON:-python3}
mkdir -p "$OUTDIR"
OUT="$OUTDIR/neohorsejev-4b-q8_0.gguf"

# The backbone lives under backbone/ when the bundle ships the head alongside.
SRC="$MODEL"
if [ -f "$MODEL/backbone/config.json" ]; then SRC="$MODEL/backbone"; fi

echo "==> converting backbone to Q8_0 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$SRC" \
    --outfile "$OUT" --outtype q8_0 --no-mtp

echo "==> done: $OUT"
if [ -f "$MODEL/pointer_head.safetensors" ]; then
    echo "==> export the pointer head next:"
    echo "    python3 $ROOT/scripts/export_neohorsejev_head.py \\"
    echo "      $MODEL/pointer_head.safetensors $OUTDIR/neohorsejev-head.f32"
fi
