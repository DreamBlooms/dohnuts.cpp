# dohnuts.cpp

English | [简体中文](README.zh-CN.md)

**The same decisions, on CPU.**

Native C++ inference for [Dohnuts](https://github.com/PsiACE/dohnuts), built on
[llama.cpp](third_party/llama.cpp). It runs the released Qwen3.5-0.8B base with
the merged Dohnuts LoRA, scores candidate markers with the scalar decision head,
and returns probabilities from a single forward pass. No GPU, no generation.

Dohnuts 0.1.0 is text-only. The released adapter carries language LoRA and the
scorer head, and no vision weights.

## One message, several decisions

Serve the model:

```sh
build/dohnuts-cli --server --port 8080 \
  --model work/dohnuts-Q8_0.gguf --head work/head.f32 \
  --metadata models/dohnuts-0.1.0/dohnuts.json
```

Route a support request and check whether it asks for a refund in the same call:

```sh
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":{"message":"I was charged twice. Please refund the duplicate."},
       "questions":{"route":{"type":"choice","instructions":"Which team should handle this?",
         "criteria":["billing","technical support","sales"]},
         "refund":{"type":"noul","instructions":"Is a refund requested?"}}}'
```

```json
{"model":"dohnuts",
 "answers":{
   "route":{"type":"choice","confidence":0.2632,
     "probabilities":{"billing":0.6223,"technical support":0.3283,"sales":0.0494},
     "choice":"billing"},
   "refund":{"type":"noul","confidence":0.9572,"noul":0.9572}},
 "usage":{"input_tokens":93,"images":0}}
```

`choice`, `score`, and `noul` behave as in
[Dohnuts](https://github.com/PsiACE/dohnuts). `/predict` takes one request or an
array; `/health` and `/v1/models` describe the server. Pass `--api-key KEY` to
require `Authorization: Bearer KEY` on predictions. CORS is open to any origin by
default; set `--cors-origin ORIGIN` to restrict it.

Run the CLI on a file of requests with `--input requests.jsonl`, or add `--raw`
to print uncalibrated scorer logits.

## Build

Requires CMake 3.14+, a C++20 compiler, and `nlohmann-json3-dev`.

On Ubuntu / Debian the two scripts below install the dependencies and build the
CLI (extra CMake arguments are forwarded, e.g. a GPU backend):

```sh
sudo scripts/setup.sh
scripts/build.sh
```

Manual build:

```sh
git submodule update --init --depth 1
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j --target dohnuts-cli
```

llama.cpp is pinned to release `v0.4.1` as a submodule. `-DLLAMA_DIR=...`
points the build at another checkout.

## Models

Pre-converted GGUF files are published at
[DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF](https://huggingface.co/DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF):

| File | Quantization |
| --- | --- |
| `Dohnuts-0.1.0-0.8B-f16.gguf` | F16 |
| `Dohnuts-0.1.0-0.8B-Q8_0.gguf` | Q8_0 |
| `Dohnuts-0.1.0-0.8B-Q6_K.gguf` | Q6_K |
| `Dohnuts-0.1.0-0.8B-Q4_K_M.gguf` | Q4_K_M |

`head.f32` (the scorer head) and `dohnuts.json` (calibration) are required
alongside any GGUF. Download one quantization plus both small files:

```sh
hf download DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF \
  Dohnuts-0.1.0-0.8B-Q8_0.gguf head.f32 dohnuts.json --local-dir models
```

## Build the GGUF yourself

The release ships a compact checkpoint, not a standalone language model. Merge
it into the base once, then convert and quantize:

```sh
python3 scripts/export_dohnuts.py \
    --base models/Qwen3.5-0.8B \
    --adapter models/dohnuts-0.1.0 \
    --out work/merged --head work/head.f32

scripts/build_gguf.sh work/merged work
```

This writes `dohnuts-f16.gguf`, `dohnuts-Q8_0.gguf`, and `dohnuts-Q4_K_M.gguf`.

## Accuracy

Against the f32 Hugging Face reference over six text cases and one long
shared-prefix case, every candidate choice agrees:

| Precision | Max logit deviation | Max probability deviation |
| --- | ---: | ---: |
| f16 | 0.573 | 0.097 |
| Q8_0 | 0.656 | 0.094 |
| Q6_K | 0.879 | 0.116 |
| Q4_K_M | 1.878 | 0.161 |

The deviations come from weight rounding and grow with quantization. Use f16,
Q8_0, or Q6_K when probabilities matter, Q4_K_M for coarse decisions.

## Speed

Qwen3.5 is a hybrid: 18 gated-delta-net layers and 6 full-attention layers. The
delta-net is the CPU bottleneck, so prefill runs at roughly 21 tokens/s on eight
threads and a two-question request takes a few seconds. Dohnuts never generates
tokens, so only prefill matters. Numbers vary with host load; use `llama-bench`
for a stable reference.

## GPU

The default build is CPU only. CUDA, Vulkan, ROCm (HIP), and Metal are optional
backends; enable one at configure time:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_CUDA=ON     # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_VULKAN=ON   # AMD or Intel
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_HIP=ON      # AMD ROCm
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_METAL=ON    # Apple
cmake --build build -j --target dohnuts-cli
```

Each backend needs its own toolchain: the CUDA Toolkit for CUDA, the Vulkan SDK
(`glslc` and the loader) for Vulkan, ROCm for HIP, and the Xcode command line
tools for Metal. When cross-compiling, pin the target GPU with
`-DCMAKE_CUDA_ARCHITECTURES=89` (CUDA) or `-DGPU_TARGETS=gfx1100` (HIP).

Offload at runtime:

```sh
build/dohnuts-cli --model Dohnuts-0.1.0-0.8B-Q8_0.gguf --head head.f32 \
  --metadata dohnuts.json --gpu-layers -1
```

`--gpu-layers -1` keeps every layer in VRAM; a positive number keeps that many.
`--device CUDA0` or a comma-separated list selects devices. `--list-devices`
prints what the build can use. A CPU-only build ignores `--gpu-layers`, so the
same command line works everywhere.

## How it works

Dohnuts reads the post-norm hidden state at each candidate marker, applies one
scalar head, and temperature-scales a softmax per question. llama.cpp supports
the architecture (`qwen35`) and exposes all of it through the public API, so the
core is unchanged:

| Dohnuts | Here |
| --- | --- |
| Prompt template with a reserved marker token per candidate | `protocol.cpp`, tokenized without special tokens |
| Hidden state at each marker | `llama_get_embeddings_ith` with `embeddings=true`, pooling off |
| Scalar scorer `Linear(1024, 1)` | `head.f32` dot product in `engine.cpp` |
| Temperature softmax, entropy confidence, expected score | `calibrate_answer` in `protocol.cpp` |
| Shared input prefix | common prefix decoded once, then `llama_memory_seq_cp` |

Norm weights are stored as `weight + 1`, matching the Dohnuts fused kernels.

## Layout

```
include/dohnuts/engine.hpp    engine interface
include/dohnuts/protocol.hpp  prompt rendering, calibration, predictor
src/engine.cpp                llama.cpp wrapper: load, tokenize, batched scoring
src/protocol.cpp              Dohnuts templates and answers
src/main.cpp                  CLI and HTTP server
scripts/                      model export and GGUF build
```

## License

The code is licensed under [Apache-2.0](LICENSE). The HTTP layer is adapted from
[laya.cpp](https://github.com/lkarlslund/laya.cpp) under MIT. See [NOTICE](NOTICE)
for third-party terms. Dohnuts model weights keep their own license.
