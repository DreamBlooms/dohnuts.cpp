# dohnuts.cpp

[English](README.md) | 简体中文

**同样的决策能力，纯 CPU 推理。**

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的
[Dohnuts](https://github.com/PsiACE/dohnuts) 原生 C++ 推理实现。加载已发布的
Qwen3.5-0.8B 基座与合并后的 Dohnuts LoRA，以标量决策头对候选标记打分，单次前向即可输出
概率分布；全程无需 GPU，也不生成文本。

基座保留了冻结的视觉塔，配合额外的 mmproj 文件即可完成图像决策（见[视觉](#视觉)）。
同一可执行文件还通过 profile 支持另外两个同类决策模型 `decider-0.8b` 与 `kev-0.8b`
（见[其他决策模型](#其他决策模型)）。

## 一条消息，多个决策

启动服务：

```sh
build/dohnuts-cli --server --port 8080 \
  --model work/dohnuts-Q8_0.gguf --head work/head.f32 \
  --metadata models/dohnuts-0.1.0/dohnuts.json
```

在同一次调用中完成工单路由并判断是否需要退款：

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

`choice`、`score` 与 `noul` 的行为与 [Dohnuts](https://github.com/PsiACE/dohnuts) 一致。
`/predict` 接受单个请求或请求数组；`/health` 与 `/v1/models` 用于查询服务状态。传入
`--api-key KEY` 后，预测接口需携带 `Authorization: Bearer KEY`；`--cors-origin ORIGIN`
可限制 CORS 来源（默认允许所有来源）。使用 `--input requests.jsonl` 可按行处理请求文件；
加上 `--raw` 则输出未经校准的打分 logits。

## 构建

需要 CMake 3.14+ 与支持 C++20 的编译器。在 Ubuntu / Debian 上，以下两个脚本会安装依赖并
编译 CLI（额外参数将转交给 CMake，例如指定 GPU 后端）：

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

llama.cpp 以子模块形式固定在 `v0.4.1` 发行版；`-DLLAMA_DIR=...` 可指定其他 checkout。

### Windows（交叉编译）

在 Ubuntu / Debian 上，可用 MinGW-w64 交叉编译出独立的 `dohnuts-cli.exe`
（无需额外 DLL）：

```sh
sudo scripts/setup.sh --with-mingw
scripts/build_windows.sh
```

### GPU

默认构建仅支持 CPU。CUDA、Vulkan、ROCm（HIP）与 Metal 为可选后端，在配置阶段启用其一：

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_CUDA=ON     # NVIDIA
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_VULKAN=ON   # AMD 或 Intel
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_HIP=ON      # AMD ROCm
cmake -B build -DCMAKE_BUILD_TYPE=Release -DDOHNUTS_METAL=ON    # Apple
cmake --build build -j --target dohnuts-cli
```

各后端需要对应的工具链：CUDA Toolkit、Vulkan SDK（`glslc` 与 loader）、ROCm，或 Xcode
命令行工具。交叉编译时，可用 `-DCMAKE_CUDA_ARCHITECTURES=89`（CUDA）或
`-DGPU_TARGETS=gfx1100`（HIP）指定目标 GPU。

运行时通过 `--gpu-layers -1`（全部层）或指定层数卸载到 GPU，并通过 `--device CUDA0`
或逗号分隔的设备列表选择设备；`--list-devices` 会列出当前构建可用的设备。纯 CPU 构建会
忽略 `--gpu-layers`，因此同一条命令在各类构建下均可使用。

## 模型

预转换的 GGUF 文件发布于
[DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF](https://huggingface.co/DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF)：

| 文件 | 量化 |
| --- | --- |
| `Dohnuts-0.1.0-0.8B-f16.gguf` | F16 |
| `Dohnuts-0.1.0-0.8B-Q8_0.gguf` | Q8_0 |
| `Dohnuts-0.1.0-0.8B-Q6_K.gguf` | Q6_K |
| `Dohnuts-0.1.0-0.8B-Q4_K_M.gguf` | Q4_K_M |
| `mmproj-dohnuts-0.1.0-bf16.gguf` | 视觉塔（BF16，可选） |

`head.f32`（打分头）与 `dohnuts.json`（校准参数）是任何量化版本都必须搭配的文件。下载一个
量化版本及这两个小文件：

```sh
hf download DreamBlooms/Dohnuts-0.1.0-0.8B-GGUF \
  Dohnuts-0.1.0-0.8B-Q8_0.gguf head.f32 dohnuts.json --local-dir models
```

### 自己构建 GGUF

发布的是紧凑 checkpoint，而非独立的语言模型。先将其合并进基座，再进行转换与量化：

```sh
python3 scripts/export_dohnuts.py \
    --base models/Qwen3.5-0.8B \
    --adapter models/dohnuts-0.1.0 \
    --out work/merged --head work/head.f32

scripts/build_gguf.sh work/merged work
```

该命令会生成 `dohnuts-f16.gguf`、`dohnuts-Q8_0.gguf` 与 `dohnuts-Q4_K_M.gguf`。

## 视觉

Qwen3.5 基座保留了视觉塔，而 Dohnuts LoRA 仅调整语言部分，因此无需重新训练即可支持图像
决策。视觉塔单独发布为 `mmproj-dohnuts-0.1.0-bf16.gguf`，通过 `--mmproj` 传入：

```sh
build/dohnuts-cli --server --port 8080 \
  --model work/dohnuts-Q8_0.gguf --head work/head.f32 \
  --metadata models/dohnuts-0.1.0/dohnuts.json \
  --mmproj work/mmproj-dohnuts-0.1.0-bf16.gguf
```

将一张图片以 base64 PNG/JPEG data URL 的形式放入 `state.image`，其打分方式与 Python 的
`state["image"]` 完全一致，且该字段会从文本 state 中移除：

```sh
curl http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":{"image":"data:image/png;base64,iVBORw0KG..."},
       "questions":{"shape":{"type":"choice","criteria":["square","circle"]}}}'
```

图像请求的 `usage.images` 为 1，否则为 0。每次请求支持一张图片，与 Dohnuts 保持一致；
未传入 `--mmproj` 时，任何图像请求都会被拒绝。

图像在 CPU 上开销较大，因此当同一张图片对应多个问题时，图片只编码一次，前导文本与其
256 个视觉 token 也只经过一次语言模型，随后将这份状态共享给各个问题。图片会缩放至
512×512（256 token），与 Python 预处理保持一致。

## 精度

在 6 个文本样例与 1 个长共享前缀样例上，与 f32 的 Hugging Face 原版对比，所有候选选择
完全一致：

| 精度 | 最大 logit 偏差 | 最大概率偏差 |
| --- | ---: | ---: |
| f16 | 0.573 | 0.097 |
| Q8_0 | 0.656 | 0.094 |
| Q6_K | 0.879 | 0.116 |
| Q4_K_M | 1.878 | 0.161 |

偏差源自权重量化舍入，并随量化程度增大。对概率精度敏感时建议使用 f16、Q8_0 或 Q6_K，
粗略判断则可用 Q4_K_M。

## 速度

Qwen3.5 为混合架构：18 层 gated delta net 与 6 层全注意力。CPU 上的瓶颈在于 delta net，
八线程下 prefill 约为 21 tokens/s，一个包含两个问题的请求需要数秒。Dohnuts 不生成 token，
因此只有 prefill 有意义。具体数值会随主机负载波动，稳定的基准请使用 `llama-bench`。

## 其他决策模型

CLI 还可运行两个决策模型，它们与 Dohnuts 共用 Qwen3.5-0.8B 基座，但使用各自的提示模板与
读出方式。二者以 profile 形式集成：Dohnuts 路径保持不变，且均不支持图像输入。

| Profile | 模型 | 读出方式 | 权重 |
| --- | --- | --- | --- |
| `decider` | [Mapika/decider-0.8b](https://huggingface.co/Mapika/decider-0.8b) | 在 `Answer: (` 槽位将 LM head 限制到选项字母 | 全量微调 |
| `kev` | [jaredpalmer/kev-0.8b](https://huggingface.co/jaredpalmer/kev-0.8b) | 对 decide 与选项结束标记做双线性 pointer head | LoRA + pointer head |

传入对应模型的配置文件作为 `--metadata` 即可；配置文件自身声明了 profile
（`"profile": "decider"` 或 `"profile": "kev"`），无需额外开关。`--profile` 可覆盖文件中的
声明，默认值为 `dohnuts`。`kev` 还需额外传入 `--head`：

```sh
# decider：无需打分头，温度来自 decider.json
build/dohnuts-cli --model work/side/decider-0.8b-q8_0.gguf \
  --metadata work/side/decider.json

# kev：双线性头 + kev.json 中的温度
build/dohnuts-cli --model work/side/kev-0.8b-q8_0.gguf \
  --head work/side/kev-head.f32 --metadata work/side/kev.json
```

响应保留同一套核心字段（`type`、`choice`、`probabilities`、`noul`、`score`、
`confidence`），客户端无需改动。各 profile 会在 `native` 下附加自身的统计量：decider 提供
`certainty`，以及 isolated `score` 层级的 `legend`、`level_fit` 与 `fit_mass`；kev 提供其
自身的 confidence。

GGUF 转换产物与合并后的 kev 头发布于
[DreamBlooms/decider-0.8b-GGUF](https://huggingface.co/DreamBlooms/decider-0.8b-GGUF) 与
[DreamBlooms/kev-0.8b-GGUF](https://huggingface.co/DreamBlooms/kev-0.8b-GGUF)。可使用以下
脚本从上游 checkpoint 重新构建：

```sh
# decider-0.8b：全量微调，直接转换
scripts/build_decider_gguf.sh <decider-0.8b-dir> work/side/decider-0.8b-q8_0.gguf

# kev-0.8b：先将 LoRA 合并进基座、导出头，再转换
scripts/build_kev_gguf.sh <Qwen3.5-0.8B-Base-dir> <kev-0.8b-dir> work/side
```

两者均量化为 Q8_0。`kev` 会在 fp32 下先合并 LoRA 再转换，并写出 `kev-head.f32`
（q 与 k 的 pointer 行及其偏置）与 `kev.json`。对于无法一次性载入内存的大型 checkpoint，
可加 `--stream` 以逐张量低内存合并。

## 实现方式

Dohnuts 在每个候选标记处读取 post-norm 隐状态，乘以一个标量头，并按问题做温度缩放
softmax。llama.cpp 已支持该架构（`qwen35`），并通过公共 API 暴露了全部所需能力，因此核心
部分无需改动：

| Dohnuts | 这里 |
| --- | --- |
| 每个候选带保留标记 token 的提示模板 | `protocol.cpp`，tokenize 时不加特殊 token |
| 各标记处的隐状态 | `llama_get_embeddings_ith`，开启 `embeddings=true`、关闭 pooling |
| 标量打分头 `Linear(1024, 1)` | `engine.cpp` 中 `head.f32` 点积 |
| 温度 softmax、熵置信度、期望分数 | `protocol.cpp` 的 `calibrate_answer` |
| 共享输入前缀 | 公共前缀只算一次，再用 `llama_memory_seq_cp` 复用 |
| 冻结视觉塔 + merger | mtmd 加载 `mmproj-dohnuts-0.1.0-bf16.gguf`，逐图像块注入 M-RoPE 位置 |

side profile 复用同一后端，但拥有各自的读出方式：decider 将 LM head 限制到选项字母，kev
则将 decide 与选项结束处的隐状态通过双线性 pointer head 投影。

归一化权重以 `weight + 1` 的形式存储，与 Dohnuts 的融合算子保持一致。

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

代码采用 [Apache-2.0](LICENSE) 许可。HTTP 层改编自
[laya.cpp](https://github.com/lkarlslund/laya.cpp)，遵循 MIT 许可；第三方条款见
[NOTICE](NOTICE)。[Dohnuts 模型](https://huggingface.co/PsiACE/Dohnuts-0.1.0-0.8B) 权重
保留其原有许可，side 模型亦保留各自的许可。
