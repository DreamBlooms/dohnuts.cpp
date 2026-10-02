---
name: dohnuts-llm-adapt
description: Adapt a new decision-model (typed-decision / judgment / Jev-style model) into dohnuts.cpp as a side profile. Use when adding support for a new HF model that answers choice/score/noul questions with one forward pass, or when touching dohnuts.cpp profiles, prompt rendering, label readout, head export, GGUF conversion, or HF GGUF upload. Covers the profile registration checklist, readout modes (LM-head vs bilinear pointer head), prompt/token handling, build + smoke-test, and the GGUF upload convention. Does NOT contain any specific model's weights/temperatures/prompt text (those are fetched per-migration).
---

# Adapting a decision model into dohnuts.cpp

This is the reusable, feed-level workflow for adding a **new decision model** to
dohnuts.cpp as a `side` profile. It encodes everything that is the *same every
time* (architecture, registration points, readout patterns, token handling,
build/test/upload). It deliberately **omits non-reusable model specifics** — a
given model's prompt text, temperatures, head tensor names, GGUF filenames, and
HF repo must be fetched fresh each migration (see [Fetching model
specifics](#fetching-model-specifics)).

A "decision model" here = takes a `state` + typed `questions` (each `choice`,
`score`, or `noul`), answers each with a calibrated probability distribution in
**one forward pass**, no token generation.

## Ground rules (read first)

- **Never print or echo special/control tokens** (chat markers like the
  assistant/user markers, think blocks, FIM/box markers) into terminal output or
  tool output. This breaks the tool-call envelope. When a source file must
  contain them, assemble them at write-time from `chr(...)` / string concat /
  Unicode escapes in a generator script, and never `cat`/`print` the assembled
  token back. Verify by *counting* occurrences, not by printing them.
- **Side profiles ignore vision.** Even if the model ships a vision tower, drop
  it (text path only). Vision is only in the core `dohnuts` engine.
- **Side models need no comparison/parity harness** unless explicitly requested —
  smoke-test the pipeline (all 3 question types) and eyeball a sample of answers
  for sanity. Only do byte-level parity against a PyTorch reference if asked.
- **Public schema is fixed.** `common_answer` always emits `type`,
  `confidence`, `probabilities`/`choice`/`score`/`legend`/`noul`. `confidence`
  is the normalized entropy. Model-specific semantics go under `native`.
- **Explore the environment before assuming it.** Check whether a GPU is present,
  how many cores, and how much free disk; long conversions and 4B+ downloads need
  headroom. If the build backend (CPU/CUDA/Vulkan/…), the parallelism, or the
  upload destination is ambiguous, ask the user rather than guessing. Run long
  jobs in the background and poll instead of blocking.

## The profile model (what a profile IS)

One profile ≈ **one GGUF + one config json + prompt rendering + output
readout**. Two independent engines exist:

| Path | Models | Where |
| --- | --- | --- |
| core `dohnuts` profile | Dohnuts-0.8B, Linnaeus (scalar pointer head + vision) | `src/engine.cpp`, `src/protocol.cpp` |
| side profiles | every decision model (decider/thisthat/tev1/kev/jet/jpt/neohorsejev, …) | `src/side/*.cpp` + `src/side/runner.cpp` |

**You only ever touch the side layer.** The core engine is untouched. Read
`include/dohnuts/side/profile.hpp` first — it is the contract.

### `profile` interface (implement these)

```cpp
class profile {
  virtual void plan(id, state, question, rows, questions) const = 0;   // 1 question -> 1..N rows
  virtual void plan_request(state, questions, rows, out) const;        // default loops plan(); override for multi-question shared-prefix prompts
  virtual std::vector<double> score(const planned_row&) const = 0;     // -> softmax distribution
  virtual json native(const planned_question&, p, level_fit, fit_mass) const = 0;  // private stats under "native"
  virtual std::string model_name() const = 0;
};
```

### `planned_row` (one decoded forward pass)

| field | meaning |
| --- | --- |
| `ids` | full token sequence |
| `option_rel` | per-option hidden-state token positions (pointer-head profiles) |
| `slot_rel` | readout position: the logits index (LM-head) or the decide marker (pointer-head) |
| `n_options` | number of answer options |
| `letters` | per-option label **token ids** (LM-head profiles) |
| `prefix` | leading shared state-token count (prefix cache) |
| `keep` | reuse/checkpoint the prefix (set true for multi-row requests) |
| `temperature` | per-row override; **0 = use profile default** (sentinel, never fed to softmax) |

### `planned_question` (maps rows back to the answer)

`id`, `type`, `keys` (answer labels), `legend` (score level descriptions),
`isolated`, `first_row`, `n_rows` (a `score` may expand to several rows).

## Registration checklist (the 6 mandatory edits)

Add a profile named `<name>`. Each edit is required; missing one breaks the
build or silently falls through to `decider`.

1. **`include/dohnuts/profile.hpp`** — add `enum class model_profile` member +
   comment describing the readout. Update the `profile_from_string` doc comment.
2. **`src/profile.cpp`** — add the string→enum branch in `profile_from_string`
   (`if (value == "<name>") return model_profile::<name>;`) and a `case` in
   `profile_name`'s switch.
3. **`include/dohnuts/side/profile.hpp`** — declare the factory:
   `std::unique_ptr<profile> make_<name>_profile(runner&, const json& config);`
   (add `const std::filesystem::path & head_path` if it needs an external head).
4. **`src/side.cpp`** — dispatch: add the `else if (p->kind ==
   model_profile::<name>)` branch calling the factory. **If the profile reads
   hidden states** (pointer head), extend `ropts.embeddings = … ||
   p->kind == model_profile::<name>;` and add a `--head` requirement
   check alongside `kev`/`neohorsejev`. `embeddings=true` turns on
   `llama_get_embeddings_ith`; LM-head profiles leave it false.
5. **`src/main.cpp`** — update `usage()` text and, if it needs `--head`, the
   `needs_head` condition. (Config auto-selects the profile from
   `metadata["profile"]`; `--profile` overrides.)
6. **`CMakeLists.txt`** — add `src/side/<name>.cpp` to `add_executable`.

Then implement `src/side/<name>.cpp` (the profile class + a
`make_<name>_profile` factory at the bottom) and write the model config json.

## Readout modes (pick one)

### Mode A — LM head restricted to label tokens (decider/thisthat/tev1/jet/jpt)

The model is a normal causal LM. Run one forward pass and read the **next-token
logits at `slot_rel`**, restricted to a per-option label token.

- Build labels with `build_single_token_labels(back, MAX_OPTIONS, "<name>")`
  (`common.hpp`) → returns single-token ids in order **A..Z, then AA, AB, …**
  The label *text* (`A`, `B`, `AA`, …) is what you put in the prompt; the id is
  what you index logits with. Add a `letter_label(j)` helper for the text.
- Prompt ends at the answer slot; `slot_rel = ids.size() - 1` (the token whose
  logits you read). For chat prompts with an assistant prefill that ends in a
  literal like `Answer:`, the label comes *right after* it, so the last token is
  the prefill end and you read its logits.
- `score()`:
  ```cpp
  back.decode(row.ids, row.slot_rel, row.prefix, row.keep);
  const float* logits = back.logits_at(row.slot_rel);
  std::vector<double> raw(row.n_options);
  for (int j = 0; j < row.n_options; ++j) raw[j] = (double) logits[row.letters[j]];
  return softmax(raw, temperature);   // or per-row override
  ```
- **Per-type temperature**: if the model calibrates one temperature per question
  type, set `planned_row.temperature` in `plan()` and in `score()` use
  `softmax(raw, row.temperature > 0 ? row.temperature : temperature)`.

### Mode B — bilinear pointer head (kev/neohorsejev)

The model is a backbone + an **external bilinear head** `PointerHead(q,k)` with
`pointer_dim = dp`. `logit[j] = (k(h_opt[j]) · q(h_decide)) / sqrt(dp)`.

- Needs `--head` pointing at a `.f32` file and `embeddings=true` (hidden states).
- Prompt lays out FIM/box markers: state under a prefix marker, then per question
  `middle … instruction … [box_start option box_end]×K … suffix`. Read the hidden
  state at the suffix (`slot_rel` = decide marker) and at each `box_end`
  (`option_rel[j]`).
- Head file layout (match `scripts/export_kev.py`): `2 * dp * (hidden + 1)`
  floats, **q rows first then k rows**, each row `[hidden…, bias]`. Load with
  `read_floats(head_path, 2 * weights)` and split into `head_q`, `head_k`.
- `score()`:
  ```cpp
  back.decode(row.ids, -1, row.prefix, row.keep);   // -1: no single logits index
  const float* h_decide = back.embeddings_at(row.slot_rel);
  auto qv = project(head_q, h_decide);
  // per option: kv = project(head_k, embeddings_at(row.option_rel[j]))
  // raw[j] = dot(qv, kv) * (1/sqrt(dp));  return softmax(raw, temperature);
  ```
- **`project(w, h)`**: for each `d` in `dp`, `out[d] = bias + Σ_i w[d][i]*h[i]`,
  where `w` row `d` has `hidden` weights then 1 bias.
- If the model uses a **shared state + per-question branches** (block-causal
  layout), override `plan_request` and emit one row per question, each row =
  shared state prefix + that question's branch, `row.prefix = prefix.size()` so
  the prefix is decoded once and cached (this is exactly what `thisthat` and
  `neohorsejev` do; it is mathematically equivalent to the packed block-causal
  form on hybrid/recurrent backbones).

## Prompt & token handling

- **Rendering state**: strings verbatim; objects/arrays flattened to indented
  text (usually fewer tokens than JSON). Pick the renderer that matches the
  upstream reference (`dump_python`/`render` in `common.hpp` for JSON-ish;
  `kev_render`/`jp_render` style for indented key/value). If the upstream
  escapes `<|name|>` spans so callers can't forge control tokens, replicate that
  escape.
- **Chat template**: if the model uses a chat template with thinking disabled,
  hand-roll the prefix as
  `marker(system) + system + end + marker(user) + user + end + generation-suffix`
  where `generation-suffix` is the assistant marker + an **empty think block**.
  The empty think block is usually required — dropping it collapses accuracy.
  Assemble markers via `chr()`/escapes in your generator; never print them.
- **Label tokenization boundaries matter.** If the reference tokenizes the
  header / option block / answer slot as separate calls (so BPE must not merge
  across boundaries), mirror that by tokenizing in the same chunks and
  concatenating token-id vectors (`back.tokenize(text, /*parse_special=*/true)`).
- **`parse_special`**: pass `true` when the prompt embeds model special tokens
  (so they map to their ids), `false` when checking whether a bare label is a
  single token.
- **Context limits**: `runner.max_length` (default 4096) is a hard cap and there
  is no CLI flag for it. For long-prompt models raise it via the profile config
  (e.g. `max_state_tokens`) or `side_options.max_length`, and note the cap in the
  config json.

## Config json (`--metadata`)

Self-describing: `"profile": "<name>"` selects the profile; the rest are profile
hyperparams read in the constructor. Example shape:

```json
{
  "profile": "<name>",
  "temperature": 1.0,
  "version": "<version-label>",
  "base": "<upstream hf repo>",
  "max_options": 255,
  "max_state_tokens": 4096
}
```

- `temperature` is the only value that changes numeric output. Use the upstream
  calibrated value(s). For per-type temps use `t_choice`/`t_score`/`t_noul`.
  `max_options` is documentation-only (code uses its own `constexpr`).
- Do **not** put a `temperatures.{choice,score,noul}` object here — that key is
  only read by the core `dohnuts` path and is ignored by side profiles.

## Build & smoke test

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release   # add DOHNUTS_CUDA=ON / _VULKAN / _HIP / _METAL for a GPU backend
cmake --build build -j --target dohnuts-cli # match -j to the available cores
```

Smoke (all three types in one request):

```jsonc
// one line of a .jsonl
{"state":"<some state>","questions":{
  "q1":{"type":"choice","instructions":"...","criteria":{"a":"A","b":"B"}},
  "q2":{"type":"noul","instructions":"..."},
  "q3":{"type":"score","instructions":"...","criteria":["low","med","high"]}}}
```

```sh
build/dohnuts-cli --model <model>.gguf --metadata <name>.json \
  [--head <name>-head.f32] [--gpu-layers N] --input smoke.jsonl
```

Pass criteria: JSON parses, all 3 answers present with `type`,
`probabilities`/`choice`/`score`/`noul`, `confidence`, and a populated `native`.
Sanity-check a few answers against the state (does the choice follow from the
rules?). A random/untrained head yields near-uniform probabilities — fine for a
pipeline smoke, not for quality.

## GGUF conversion

- Converter: `third_party/llama.cpp/convert_hf_to_gguf.py` (needs `transformers`
  with the model's arch support). Check free disk first — a 4B checkpoint and its
  Q8_0 output together need roughly the source size plus ~5 GB, more if you keep
  the source and any intermediate copies.
- `--outtype q8_0` for the shipping quant. **Pass `--no-mtp`** when the HF config
  declares `mtp_num_hidden_layers` but ships no MTP weights (common on Qwen3.5
  derivatives) — it keeps the block count correct.
- Text-only / vision-dropped models convert directly. Merged-LoRA models convert
  directly (already merged). LoRA+external-head models need the head exported
  separately (`.f32`, Mode B layout) and passed as `--head` at run time.
- **Watch the tensor-name prefix.** Some checkpoints nest the text tower under
  `language_model.*` or a `text_config.architectures` that names a backbone-only
  class. If the converter rejects the arch or cannot map a tensor, either patch
  a conversion copy's `config.json` (`architectures` and
  `text_config.architectures` → the causal-LM class) or rewrite a single-shard
  copy renaming `language_model.`→`model.` and dropping `visual.`/`vision.*`.
- Mirror the existing `scripts/build_<model>_gguf.sh` wrappers (see
  `build_thisthat_gguf.sh` for the canonical shape) and add a
  `scripts/build_<name>_gguf.sh`.

## HF GGUF upload

Pick the target repo/namespace with the user (org, privacy, and any naming
convention are theirs to choose). Files: the Q8_0 `.gguf`, the config json, any
head `.f32`, and a `README.md`. The README is **the upstream model card body,
unchanged**, with two additions:

1. **YAML front matter** (updated for the quantized copy):
   ```yaml
   ---
   license: <upstream license, e.g. cc-by-nc-4.0>
   pipeline_tag: text-classification
   base_model: <upstream repo>
   base_model_relation: quantized
   library_name: llama.cpp
   tags: [typed-decision, structured-output, calibration, decision-making, gguf, llama.cpp]
   ---
   ```
2. A **`## GGUF`** section at the end: how to build dohnuts.cpp, a file table
   (name / quant / size), the config json note, the `dohnuts-cli --server` start
   command (+ `--head` if any), a `curl /v1/systemone` example, a one-line note on
   `native` fields, and a "rebuild with `scripts/build_<name>_gguf.sh`" line. Keep
   the tone/structure of an existing quantized model card in the same namespace.

Upload with the `huggingface_hub` API / `hf` CLI using whatever token/endpoint
the environment provides (do not hardcode a host or a mirror). If the upstream
license is non-commercial (e.g. CC BY-NC), call that out in the card. Uploads can
run in the background while you continue other work.

## Project README updates (both `README.md` and `README.zh-CN.md`)

- The **intro** names ~3 representative models then "etc." — do **not** list
  every family like a roll-call.
- The **"Other decision models"** table gets one row per new profile (models,
  readout, weights, GGUF link). Keep both language files consistent.

## Fetching model specifics (per-migration, do NOT cache here)

Fetch these fresh each time (from the HF `config.json`, `model_manifest.json`,
`calibration.json`, and the upstream inference/prompt source, or the serving lib
such as `llm2jev`):

- **Readout type**: does it use a plain LM head (Mode A) or an external pointer
  head (Mode B)? (Look for a `pointer_head.safetensors` / `PointerHead` / a head
  dim in the manifest → Mode B.)
- **Prompt format**: the exact system/user text, option layout, answer slot, and
  whether it's chat-templated or raw completion. Replicate byte-for-byte; this
  is where numeric parity lives.
- **Temperatures**: single value vs per-type, and the exact calibrated numbers.
- **Head tensor names/shape** (Mode B): the `q`/`k` weight+bias names and
  `pointer_dim`, to write the export script.
- **Answer semantics**: how `score` is reduced (expected level vs argmax index),
  what `native` should carry (e.g. a model-specific confidence formula),
  label limits (options/levels).
- **Token caps**: state/branch/prompt limits → set `max_*_tokens` in the config.

## When you're done

- `git status` / `git diff` shows only intended files (the 6 registration edits,
  `src/side/<name>.cpp`, scripts, both READMEs). Do **not** commit unless asked.
- Smoke passed for all 3 types; sample answers look sane; build is 0-error.
- GGUF + config (+ head) built; upload card assembled; project READMEs updated.
- Some tooling loads skills/config only at startup — if a skill or config change
  is not taking effect, the environment may need a reload/restart (check the
  tool's docs for whether that applies here).
