#!/usr/bin/env bash
# Merge the JAD-S1 LoRA into LLaDA-MoE-7B-A1B and convert it to a quantized GGUF.
#
# Usage: build_jad_gguf.sh <base_dir> <checkpoint_dir> <out_q8_0_file> [llama_cpp_dir]
#
# Writes both the merged F16 GGUF (same directory, ...-F16.gguf) and the Q8_0 one.
#
# The adapter is applied with llama.cpp's own LoRA tooling (convert_lora_to_gguf.py
# + llama-export-lora), so no torch/peft model load is needed and the LLaDA-MoE
# expert tensors keep the converter's mapping.
set -euo pipefail

BASE=${1:?base model directory (LLaDA-MoE-7B-A1B-Instruct snapshot)}
CHECKPOINT=${2:?JAD-S1 checkpoint directory (adapter_config.json + adapter_model.safetensors)}
OUT=${3:?output Q8_0 .gguf path}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=${4:-$ROOT/third_party/llama.cpp}

# The F16 file ships alongside the Q8_0 one; derive its name from OUT.
case "$OUT" in
    *Q8_0.gguf) OUT_F16="${OUT%Q8_0.gguf}F16.gguf" ;;
    *) OUT_F16="${OUT%.gguf}-F16.gguf" ;;
esac

PYTHON=${PYTHON:-python3}
WORK=${WORK:-$ROOT/work/jad-gguf}
mkdir -p "$WORK" "$(dirname "$OUT")"

# llama-export-lora / llama-quantize are tools, not part of the library build.
TOOLS=$LLAMA/build/bin
if [ ! -x "$TOOLS/llama-export-lora" ] || [ ! -x "$TOOLS/llama-quantize" ]; then
    echo "==> building llama tools"
    cmake -S "$LLAMA" -B "$LLAMA/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF >/dev/null
    cmake --build "$LLAMA/build" -j --target llama-export-lora llama-quantize
fi

echo "==> converting the LLaDA-MoE base to f16 GGUF"
"$PYTHON" "$LLAMA/convert_hf_to_gguf.py" "$BASE" \
    --outfile "$WORK/jad-base-f16.gguf" --outtype f16

echo "==> converting the JAD LoRA to GGUF"
"$PYTHON" "$LLAMA/convert_lora_to_gguf.py" "$CHECKPOINT" \
    --base "$BASE" --trust-remote-code --outfile "$WORK/jad-lora.gguf"

echo "==> merging the LoRA into the base"
"$TOOLS/llama-export-lora" -m "$WORK/jad-base-f16.gguf" \
    --lora "$WORK/jad-lora.gguf" -o "$WORK/jad-f16.gguf"

echo "==> keeping the merged F16 model: $OUT_F16"
mv "$WORK/jad-f16.gguf" "$OUT_F16"
rm -f "$WORK/jad-base-f16.gguf"

echo "==> quantizing Q8_0"
"$TOOLS/llama-quantize" "$OUT_F16" "$OUT" Q8_0

echo "==> done: $OUT_F16"
echo "==> done: $OUT"
