#!/usr/bin/env bash
# Merge the jaredpalmer/kev-0.8b LoRA and export the pointer head, then convert
# the merged model to a quantized GGUF.
#
# Usage: build_kev_gguf.sh <base_dir> <checkpoint_dir> <out_dir> [llama_cpp_dir]
set -euo pipefail

BASE=${1:?Qwen3.5-0.8B-Base snapshot directory}
CHECKPOINT=${2:?kev-0.8b snapshot directory (adapter + head.pt)}
OUT=${3:?output directory}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${4:-$ROOT/third_party/llama.cpp}

PYTHON=${PYTHON:-python3}
mkdir -p "$OUT"

# 1. Merge the adapter into the base and export the bilinear pointer head.
echo "==> merging LoRA and exporting the pointer head"
"$PYTHON" "$ROOT/scripts/export_kev.py" "$BASE" "$CHECKPOINT" "$OUT/kev-merged"

# 2. The merged export omits tokenizer files; take them from the base.
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -f "$BASE/$f" ] && cp "$BASE/$f" "$OUT/kev-merged/"
done

# 3. Convert to a quantized GGUF.
echo "==> converting to Q8_0 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$OUT/kev-merged" \
    --outfile "$OUT/kev-0.8b-q8_0.gguf" --outtype q8_0 --no-mtp

echo "==> done: $OUT"
