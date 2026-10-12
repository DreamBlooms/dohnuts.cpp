// Shared llama.cpp runner for the side profiles. It owns the model/context and
// exposes the two readouts the profiles need: vocab logits at one token
// (decider) and hidden states at arbitrary tokens (kev).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "dohnuts/flash_attn.hpp"

struct llama_model;

namespace dohnuts::side {

struct runner_options {
    std::filesystem::path model;
    std::string device;
    int threads = 0;
    int n_batch = 512;
    int gpu_layers = 0;
    int max_length = 4096;
    bool embeddings = false;   // kev needs hidden states; decider does not
    bool diffusion = false;    // LLaDA-MoE: non-causal, mask-seeded readout
    int n_seq_max = 1;         // >1 enables decode_shared (one sequence per row)
    flash_attn_mode flash_attn = flash_attn_mode::automatic;
};

class runner {
public:
    explicit runner(const runner_options & options);
    ~runner();

    runner(const runner &) = delete;
    runner & operator=(const runner &) = delete;

    int hidden() const;
    int max_length() const;
    std::string backend_name() const;

    // Raw handles for the official decision interop.
    llama_model * model_handle() const;
    int n_embd_out() const;   // width of the embeddings output (2*pointer_dim for kev)

    // GGUF metadata accessors (official decision-model interop). Empty when the
    // key or template is missing.
    std::string architecture() const;
    std::string meta(const std::string & key) const;
    std::string chat_template(const std::string & name) const;
    // Every "key=value" whose key starts with `prefix`, in file order.
    std::vector<std::pair<std::string, std::string>> meta_prefix(const std::string & prefix) const;

    std::vector<int32_t> tokenize(const std::string & text, bool parse_special = true) const;
    int single_token(const std::string & text) const;

    // Decodes ids as one sequence. When embeddings are enabled every token is an
    // output; otherwise only logits_index is (pass -1 to skip). Positions are
    // the token index, matching the profiles' layouts.
    //
    // With `keep`, the first `prefix` tokens come from a cached state checkpoint
    // when one exists; otherwise they are decoded as their own batch and
    // checkpointed, so later rows and requests over the same state skip their
    // prefill. Either way the row scores the same.
    void decode(const std::vector<int32_t> & ids, int logits_index, size_t prefix = 0, bool keep = false);

    // Valid until the next decode. Indexed by batch token position.
    const float * logits_at(int index) const;
    const float * embeddings_at(int index) const;

    // The model's mask token id (LLaDA-MoE), or -1 when the model has none.
    int mask_id() const;

    // Decodes ids as one non-causal sequence and emits logits at `read_index`
    // only (the mask slot). For diffusion profiles; ignores the prefix cache.
    void decode_canvas(const std::vector<int32_t> & ids, int read_index);

    // Evaluates rows that share their first `prefix` tokens in one batch (each
    // row as its own sequence; needs n_seq_max >= rows.size()). `positions[r]`
    // are the absolute token positions to read for row r. Returns, per row, one
    // copied output vector per requested position: a logits row (n_vocab) when
    // `logits`, else an embeddings row (n_embd_out). Faster than decoding each
    // row, but on hybrid backbones it is not numerically identical to a full
    // per-row decode, so callers gate it behind an opt-in switch.
    std::vector<std::vector<std::vector<float>>> decode_shared(
        const std::vector<std::vector<int32_t>> & rows,
        size_t prefix,
        const std::vector<std::vector<int32_t>> & positions,
        bool logits);

private:
    struct impl;
    std::unique_ptr<impl> p;
};

} // namespace dohnuts::side
