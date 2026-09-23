# dohnuts.cpp

[English](README.md) | 简体中文

**同样的决策，纯 CPU 运行。**

用原生 C++ 在 [llama.cpp](third_party/llama.cpp) 上运行
[Dohnuts](https://github.com/PsiACE/dohnuts) 推理。加载已发布的 Qwen3.5-0.8B 基座与
合并后的 Dohnuts LoRA，用标量决策头对候选标记打分，单次前向返回概率分布。不需要 GPU，
也不生成文本。

基座保留了冻结的视觉塔，因此额外提供一个 mmproj 文件即可做图像决策（见[视觉](#视觉)）。
同一个二进制还通过 profile 运行两个相关决策模型 `decider-0.8b` 与 `kev-0.8b`
（见[其他决策模型](#其他决策模型)）。

## 一条消息，多个决策

启动服务：

```sh
build/dohnuts-cli --server --port 8080 \
  --model work/dohnuts-Q8_0.gguf --head work/head.f32 \
  --metadata models/dohnuts-0.1.0/dohnuts.json
```

一次调用同时路由工单并判断是否要求退款：

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

`choice`、`score`、`noul` 的语义与 [Dohnuts](https://github.com/PsiACE/dohnuts) 一致。
`/predict` 接受单个请求或数组；`/health` 和 `/v1/models` 描述服务状态。加
`--api-key KEY` 后预测接口需要 `Authorization: Bearer KEY`，用 `--cors-origin ORIGIN`
可限制 CORS（默认开放）。用 `--input requests.jsonl` 对文件逐行跑 CLI，加 `--raw`
则输出未校准的打分 logits。

## 构建

需要 CMake 3.14+ 和支持 C++20 的编译器。在 Ubuntu / Debian 上，下面两个脚本会安装
依赖并编译 CLI（多余参数会转交给 CMake，例如指定 GPU 后端）：

```sh
sudo scripts/setup.sh
scripts/build.sh
```

手动构建：

```sh
git submodule update --init --depth 1
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build -j --target dohnuts-cli
```

llama.cpp 以子模块固定在发行版 `v0.4.1`；用 `-DLLAMA_DIR=...` 可指向其他 checkout。

### Windows（交叉编译）

在 Ubuntu / Debian 上用 MinGW-w64 交叉编译出独立的 `dohnuts-cli.exe`
（无需额外 DLL）：

```sh
sudo scripts/setup.sh --with-mingw
scripts/build_windows.sh
```

## 模型

预转换好的 GGUF 发布在
[DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF](https://huggingface.co/DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF)：

| 文件 | 量化 |
| --- | --- |
| `Dohnuts-0.1.0-0.8B-f16.gguf` | F16 |
| `Dohnuts-0.1.0-0.8B-Q8_0.gguf` | Q8_0 |
| `Dohnuts-0.1.0-0.8B-Q6_K.gguf` | Q6_K |
| `Dohnuts-0.1.0-0.8B-Q4_K_M.gguf` | Q4_K_M |
| `mmproj-dohnuts-0.1.0-bf16.gguf` | 视觉塔（BF16，可选） |

`head.f32`（打分头）和 `dohnuts.json`（校准）是任何 GGUF 都必须搭配的文件。下载一个
量化版本加这两个小文件：

```sh
hf download DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF \
  Dohnuts-0.1.0-0.8B-Q8_0.gguf head.f32 dohnuts.json --local-dir models
```

### 自己构建 GGUF

发布的是紧凑 checkpoint，不是独立的语言模型。先合并到基座，再转换与量化：

```sh
python3 scripts/export_dohnuts.py \
    --base models/Qwen3.5-0.8B \
    --adapter models/dohnuts-0.1.0 \
    --out work/merged --head work/head.f32

scripts/build_gguf.sh work/merged work
```

会生成 `dohnuts-f16.gguf`、`dohnuts-Q8_0.gguf`、`dohnuts-Q4_K_M.gguf`。

## 视觉

Qwen3.5 基座保留了视觉塔，而 Dohnuts LoRA 只调整语言侧，所以无需重训即可做图像决策。
视觉塔单独发布为 `mmproj-dohnuts-0.1.0-bf16.gguf`，用 `--mmproj` 传入：

```sh
build/dohnuts-cli --server --port 8080 \
  --model work/dohnuts-Q8_0.gguf --head work/head.f32 \
  --metadata models/dohnuts-0.1.0/dohnuts.json \
  --mmproj work/mmproj-dohnuts-0.1.0-bf16.gguf
```

把一张图片以 base64 PNG/JPEG data URL 放进 `state.image`，其打分方式与 Python 的
`state["image"]` 完全一致，并且该 key 会从文本 state 中剔除：

```sh
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":{"image":"data:image/png;base64,iVBORw0KG..."},
       "questions":{"shape":{"type":"choice","criteria":["square","circle"]}}}'
```

图像请求的 `usage.images` 为 1，否则为 0。每请求只支持一张图片，与 Dohnuts 一致。不加
`--mmproj` 时，任何图像请求都会被拒绝。

CPU 上图像开销较大，因此同一张图问多个问题时，图片只编码一次，前导文本与 256 个视觉
token 也只过一次语言模型，再把这份状态共享给各个问题。图片会缩放到 512x512（256
token），与 Python 预处理一致。

## 其他决策模型

CLI 还能运行两个共用 Qwen3.5-0.8B 基座、但提示与读出方式不同的决策模型。它们以 profile
形式提供：Dohnuts 路径保持不变，二者都不支持图像。

| Profile | 模型 | 读出方式 | 权重 |
| --- | --- | --- | --- |
| `decider` | [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) | 在 `Answer: (` 槽位把 LM head 限制到选项字母 | 全量微调 |
| `kev` | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | 对 decide 与选项结束标记做双线性 pointer head | LoRA + pointer head |

把对应模型配置作为 `--metadata` 传入即可；配置文件自带 profile
（`"profile": "decider"` 或 `"profile": "kev"`），无需额外开关。`--profile` 可覆盖
文件设置，默认 `dohnuts`。`kev` 还需 `--head`：

```sh
# decider：无需打分头，温度来自 decider.json
build/dohnuts-cli --model work/side/decider-0.8b-q8_0.gguf \
  --metadata work/side/decider.json

# kev：双线性头 + kev.json 中的温度
build/dohnuts-cli --model work/side/kev-0.8b-q8_0.gguf \
  --head work/side/kev-head.f32 --metadata work/side/kev.json
```

响应保留同一套核心字段（`type`、`choice`、`probabilities`、`noul`、`score`、
`confidence`），客户端无需改动。各 profile 在 `native` 下附加自己的统计量：decider 有
`certainty`，以及 isolated `score` 层级的 `legend` / `level_fit` / `fit_mass`；kev 有
其自身的 confidence。

GGUF 转换产物与合并后的 kev 头发布在
[DreamBlooms/decider-0.8b-GGUF](https://huggingface.co/DreamBlooms/decider-0.8b-GGUF) 和
[DreamBlooms/kev-0.8b-GGUF](https://huggingface.co/DreamBlooms/kev-0.8b-GGUF)。从上游
checkpoint 重建：

```sh
# decider-0.8b：全量微调，直接转换
scripts/build_decider_gguf.sh <decider-0.8b-dir> work/side/decider-0.8b-q8_0.gguf

# kev-0.8b：先把 LoRA 合并进基座、导出头，再转换
scripts/build_kev_gguf.sh <Qwen3.5-0.8B-Base-dir> <kev-0.8b-dir> work/side
```

二者都量化到 Q8_0。`kev` 会先在 fp32 下合并 LoRA 再转换，并写出 `kev-head.f32`
（q 与 k 的 pointer 行及其偏置）和 `kev.json`。对无法一次性放入内存的大 checkpoint，
加 `--stream` 可逐张量低内存合并。

## 精度

与 f32 的 Hugging Face 原版对比（6 个文本样例 + 1 个长共享前缀样例），所有候选选择
完全一致：

| 精度 | 最大 logit 偏差 | 最大概率偏差 |
| --- | ---: | ---: |
| f16 | 0.573 | 0.097 |
| Q8_0 | 0.656 | 0.094 |
| Q6_K | 0.879 | 0.116 |
| Q4_K_M | 1.878 | 0.161 |

偏差来自权重量化舍入，随量化程度增大。对概率敏感时用 f16、Q8_0 或 Q6_K，粗判可用
Q4_K_M。

## 速度

Qwen3.5 是混合架构：18 层 gated delta net + 6 层全注意力。CPU 上的瓶颈是 delta net，
八线程 prefill 约 21 tokens/s，一个两问题的请求需要数秒。Dohnuts 从不生成 token，
所以只有 prefill 有意义。数值会随主机负载波动，稳定参考请用 `llama-bench`。

## GPU

默认只编译 CPU。CUDA、Vulkan、ROCm（HIP）、Metal 是可选后端，配置时选一个开启：

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_CUDA=ON     # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_VULKAN=ON   # AMD 或 Intel
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_HIP=ON      # AMD ROCm
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_METAL=ON    # Apple
cmake --build build -j --target dohnuts-cli
```

各后端需要各自的工具链：CUDA Toolkit、Vulkan SDK（`glslc` 与 loader）、ROCm、或
Xcode 命令行工具。交叉编译时用 `-DCMAKE_CUDA_ARCHITECTURES=89`（CUDA）或
`-DGPU_TARGETS=gfx1100`（HIP）指定目标 GPU。

运行时用 `--gpu-layers -1`（全部层）或正数（层数）卸载到 GPU，用 `--device CUDA0`
或逗号分隔列表选择设备；`--list-devices` 打印当前构建可用的设备。纯 CPU 构建会忽略
`--gpu-layers`，所以同一条命令在任何构建下都能用。

## 实现方式

Dohnuts 在每个候选标记处取 post-norm 隐状态，乘一个标量头，再按问题做温度缩放 softmax。
llama.cpp 已支持该架构（`qwen35`），并通过公共 API 暴露了全部所需能力，核心无需改动：

| Dohnuts | 这里 |
| --- | --- |
| 每个候选带保留标记 token 的提示模板 | `protocol.cpp`，tokenize 时不加特殊 token |
| 各标记处的隐状态 | `llama_get_embeddings_ith`，开启 `embeddings=true`、关闭 pooling |
| 标量打分头 `Linear(1024, 1)` | `engine.cpp` 中 `head.f32` 点积 |
| 温度 softmax、熵置信度、期望分数 | `protocol.cpp` 的 `calibrate_answer` |
| 共享输入前缀 | 公共前缀只算一次，再用 `llama_memory_seq_cp` 复用 |
| 冻结视觉塔 + merger | mtmd 加载 `mmproj-dohnuts-0.1.0-bf16.gguf`，逐图像块注入 M-RoPE 位置 |

side profile 复用同一后端，但有自己的读出：decider 把 LM head 限制到选项字母，kev 把
decide 与选项结束处的隐状态过双线性 pointer head。

Norm 权重按 `weight + 1` 存储，与 Dohnuts 的融合算子一致。

## 目录

```
include/dohnuts/engine.hpp    引擎接口
include/dohnuts/protocol.hpp  提示渲染、校准、predictor
include/dohnuts/profile.hpp   side 模型 profile 开关
include/dohnuts/side.hpp      side 引擎门面
include/dohnuts/side/         runner、decider 与 kev profile
include/dohnuts/http.hpp      HTTP 传输层
src/engine.cpp                llama.cpp/mtmd 封装：加载、tokenize、批量打分
src/protocol.cpp              Dohnuts 模板与答案
src/side.cpp                  side profile 分发
src/side/                     共享 runner 与两个 side profile
src/http.cpp                  服务路由与 CORS
src/main.cpp                  CLI 与 HTTP 服务
scripts/                      环境安装、原生/Windows 构建、模型导出与 GGUF
cmake/                        MinGW-w64 交叉工具链
```

## 许可证

代码采用 [Apache-2.0](LICENSE)。HTTP 层改编自 [laya.cpp](https://github.com/lkarlslund/laya.cpp)，
遵循 MIT；第三方条款见 [NOTICE](NOTICE)。Dohnuts 模型权重保留其原有许可；side 模型
保留各自的许可。
