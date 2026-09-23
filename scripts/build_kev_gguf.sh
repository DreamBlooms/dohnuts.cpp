#!/usr/bin/env bash
# Merge a kev checkpoint (LoRA + pointer head) and convert it to a quantized GGUF.
#
# Usage: build_kev_gguf.sh <base_dir> <checkpoint_dir> <out_dir> [llama_cpp_dir]
#
# Set STREAM=1 to merge one tensor at a time (low memory; for large checkpoints
# like kev-4b, whose fp32 merge does not fit on a small machine).
set -euo pipefail

BASE=${1:?base model snapshot directory}
CHECKPOINT=${2:?kev checkpoint directory (adapter + head.pt)}
OUT=${3:?output directory}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${4:-$ROOT/third_party/llama.cpp}

PYTHON=${PYTHON:-python3}
VERSION=${VERSION:-$(basename "$CHECKPOINT" | sed 's/^kev-//')}
mkdir -p "$OUT"

# 1. Merge the adapter into the base and export the bilinear pointer head.
#    finalize() in the exporter copies the tokenizer files over.
echo "==> merging LoRA and exporting the pointer head"
"$PYTHON" "$ROOT/scripts/export_kev.py" "$BASE" "$CHECKPOINT" "$OUT/kev-merged" \
    ${STREAM:+--stream} --version "$VERSION"

# 2. Convert to a quantized GGUF.
echo "==> converting to Q8_0 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$OUT/kev-merged" \
    --outfile "$OUT/kev-$VERSION-q8_0.gguf" --outtype q8_0 --no-mtp

echo "==> done: $OUT"
