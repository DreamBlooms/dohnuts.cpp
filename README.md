# dohnuts.cpp

English | [简体中文](README.zh-CN.md)

**The same decisions, on CPU.**

Native C++ inference for [Dohnuts](https://github.com/PsiACE/dohnuts), built on
[llama.cpp](third_party/llama.cpp). It runs the released Qwen3.5-0.8B base with the
merged Dohnuts LoRA, scores candidate markers with the scalar decision head, and
returns probabilities from a single forward pass. No GPU, no generation.

The base keeps its frozen vision tower, so an optional mmproj file adds image
decisions (see [Vision](#vision)). The same binary also runs two related decision
models, `decider-0.8b` and `kev-0.8b`, through profiles (see
[Other decision models](#other-decision-models)).

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
require `Authorization: Bearer KEY`, and `--cors-origin ORIGIN` to restrict CORS
(open by default). Run a file of requests with `--input requests.jsonl`, or add
`--raw` to print uncalibrated scorer logits.

## Build

Requires CMake 3.14+ and a C++20 compiler. On Ubuntu / Debian the two scripts
below install the dependencies and build the CLI (extra CMake arguments are
forwarded, e.g. a GPU backend):

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

llama.cpp is pinned to release `v0.4.1` as a submodule; `-DLLAMA_DIR=...` points
the build at another checkout.

### Windows (cross-compile)

From Ubuntu / Debian, MinGW-w64 cross-compiles a self-contained
`dohnuts-cli.exe` (no extra DLLs):

```sh
sudo scripts/setup.sh --with-mingw
scripts/build_windows.sh
```

## Models

Pre-converted GGUF files are published at
[DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF](https://huggingface.co/DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF):

| File | Quantization |
| --- | --- |
| `Dohnuts-0.1.0-0.8B-f16.gguf` | F16 |
| `Dohnuts-0.1.0-0.8B-Q8_0.gguf` | Q8_0 |
| `Dohnuts-0.1.0-0.8B-Q6_K.gguf` | Q6_K |
| `Dohnuts-0.1.0-0.8B-Q4_K_M.gguf` | Q4_K_M |
| `mmproj-dohnuts-0.1.0-bf16.gguf` | Vision encoder (BF16, optional) |

`head.f32` (the scorer head) and `dohnuts.json` (calibration) are required
alongside any GGUF. Download one quantization plus both small files:

```sh
hf download DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF \
  Dohnuts-0.1.0-0.8B-Q8_0.gguf head.f32 dohnuts.json --local-dir models
```

### Build the GGUF yourself

The release ships a compact checkpoint, not a standalone language model. Merge it
into the base once, then convert and quantize:

```sh
python3 scripts/export_dohnuts.py \
    --base models/Qwen3.5-0.8B \
    --adapter models/dohnuts-0.1.0 \
    --out work/merged --head work/head.f32

scripts/build_gguf.sh work/merged work
```

This writes `dohnuts-f16.gguf`, `dohnuts-Q8_0.gguf`, and `dohnuts-Q4_K_M.gguf`.

## Vision

The Qwen3.5 base keeps its vision tower and the Dohnuts LoRA only adapts the
language side, so image decisions work without retraining. The tower ships
separately as `mmproj-dohnuts-0.1.0-bf16.gguf`; pass it with `--mmproj`:

```sh
build/dohnuts-cli --server --port 8080 \
  --model work/dohnuts-Q8_0.gguf --head work/head.f32 \
  --metadata models/dohnuts-0.1.0/dohnuts.json \
  --mmproj work/mmproj-dohnuts-0.1.0-bf16.gguf
```

Put one image in `state.image` as a base64 PNG or JPEG data URL. It is scored
exactly like the Python `state["image"]`, and the image key is dropped from the
text state:

```sh
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":{"image":"data:image/png;base64,iVBORw0KG..."},
       "questions":{"shape":{"type":"choice","criteria":["square","circle"]}}}'
```

`usage.images` is 1 for image requests, 0 otherwise. One image per request is
supported, matching Dohnuts. Without `--mmproj` any image request is rejected.

Images are expensive on CPU, so a request with several questions about one image
encodes the image once and runs the leading text and its 256 vision tokens
through the language model once, then shares that state across the questions.
Images are resized to 512x512 (256 tokens) to match the Python preprocessing.

## Other decision models

The CLI also runs two decision models that share the Qwen3.5-0.8B backbone but
use a different prompt and readout. They are profiles: the Dohnuts path is
untouched, and neither takes images.

| Profile | Model | Readout | Weights |
| --- | --- | --- | --- |
| `decider` | [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) | LM head restricted to the option letters at an `Answer: (` slot | full fine-tune |
| `kev` | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | bilinear pointer head over the decide and option-end markers | LoRA + pointer head |

Pass the matching model config as `--metadata`; the file names its own profile
(`"profile": "decider"` or `"profile": "kev"`), so no flag is needed. `--profile`
overrides the file, and `dohnuts` is the default. `kev` additionally needs
`--head`:

```sh
# decider: no scorer head, one temperature from decider.json
build/dohnuts-cli --model work/side/decider-0.8b-q8_0.gguf \
  --metadata work/side/decider.json

# kev: bilinear head plus one temperature from kev.json
build/dohnuts-cli --model work/side/kev-0.8b-q8_0.gguf \
  --head work/side/kev-head.f32 --metadata work/side/kev.json
```

The response keeps the same core fields (`type`, `choice`, `probabilities`,
`noul`, `score`, `confidence`), so clients work unchanged. Each profile adds its
native statistics under `native`: `certainty`, and `legend` / `level_fit` /
`fit_mass` for isolated `score` levels on decider; the model's own confidence for
kev.

GGUF conversions and the merged kev head are published at
[DreamBlooms/decider-0.8b-GGUF](https://huggingface.co/DreamBlooms/decider-0.8b-GGUF)
and [DreamBlooms/kev-0.8b-GGUF](https://huggingface.co/DreamBlooms/kev-0.8b-GGUF).
Rebuild them from the upstream checkpoints with:

```sh
# decider-0.8b: a full fine-tune, converted directly
scripts/build_decider_gguf.sh <decider-0.8b-dir> work/side/decider-0.8b-q8_0.gguf

# kev-0.8b: merge the LoRA into the base, export the head, then convert
scripts/build_kev_gguf.sh <Qwen3.5-0.8B-Base-dir> <kev-0.8b-dir> work/side
```

Both exports are quantized to Q8_0. `kev` merges the LoRA in fp32 before
conversion and writes `kev-head.f32` (the q and k pointer rows with their biases)
plus `kev.json`. Pass `--stream` for a low-memory merge (one tensor at a time) on
checkpoints too large to hold in RAM.

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

Each backend needs its own toolchain: the CUDA Toolkit, the Vulkan SDK (`glslc`
and the loader), ROCm, or the Xcode command line tools. When cross-compiling, pin
the target GPU with `-DCMAKE_CUDA_ARCHITECTURES=89` (CUDA) or
`-DGPU_TARGETS=gfx1100` (HIP).

Offload at runtime with `--gpu-layers -1` (all layers) or a positive layer count,
and select devices with `--device CUDA0` or a comma-separated list.
`--list-devices` prints what the build can use. A CPU-only build ignores
`--gpu-layers`, so the same command line works everywhere.

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
| Frozen vision tower + merger | mtmd with `mmproj-dohnuts-0.1.0-bf16.gguf`; M-RoPE positions injected per image chunk |

The side profiles use the same backend but their own readout: decider restricts
the LM head to the option letters, and kev projects the decide and option-end
hidden states through a bilinear pointer head.

Norm weights are stored as `weight + 1`, matching the Dohnuts fused kernels.

## Layout

```
include/dohnuts/engine.hpp    engine interface
include/dohnuts/protocol.hpp  prompt rendering, calibration, predictor
include/dohnuts/profile.hpp   side model profile switch
include/dohnuts/side.hpp      side engine facade
include/dohnuts/side/         runner, decider and kev profiles
include/dohnuts/http.hpp      HTTP transport
src/engine.cpp                llama.cpp/mtmd wrapper: load, tokenize, batched scoring
src/protocol.cpp              Dohnuts templates and answers
src/side.cpp                  side profile dispatch
src/side/                     shared runner and the two side profiles
src/http.cpp                  server routes and CORS
src/main.cpp                  CLI and HTTP server
scripts/                      setup, native/Windows builds, model export, GGUF
cmake/                        MinGW-w64 cross toolchain
```

## License

The code is licensed under [Apache-2.0](LICENSE). The HTTP layer is adapted from
[laya.cpp](https://github.com/lkarlslund/laya.cpp) under MIT. See [NOTICE](NOTICE)
for third-party terms. Dohnuts model weights keep their own license; the side
models keep theirs.
