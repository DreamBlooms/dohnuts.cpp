#!/usr/bin/env bash
# Convert Mapika/decider-0.8b (full fine-tune) to a quantized GGUF.
#
# Usage: build_decider_gguf.sh <model_dir> <out_file> [llama_cpp_dir]
set -euo pipefail

MODEL=${1:?decider model directory (config.json + model.safetensors)}
OUT=${2:?output .gguf path}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${3:-$ROOT/third_party/llama.cpp}

PYTHON=${PYTHON:-python3}
mkdir -p "$(dirname "$OUT")"

# The base declares mtp_num_hidden_layers but ships no MTP weights; --no-mtp
# keeps the block count at 24 like the reference conversion.
echo "==> converting to Q8_0 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$MODEL" \
    --outfile "$OUT" --outtype q8_0 --no-mtp

# Opt-in (DECISION=1): also write the official decision metadata + systemone
# template, so the same GGUF is servable by upstream llama-server and by
# dohnuts --profile official.
if [ "${DECISION:-0}" = "1" ]; then
    echo "==> annotating with official decision metadata"
    "$PYTHON" "$ROOT/scripts/annotate_decision.py" "$OUT" "$OUT" --profile decider \
        --temperature choice=1.0 --temperature score=1.0 --temperature noul=1.0
fi

echo "==> done: $OUT"
