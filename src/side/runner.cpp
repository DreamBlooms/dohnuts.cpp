#include "dohnuts/side/runner.hpp"

#include <list>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "dohnuts/side/common.hpp"
#include "llama.h"

namespace dohnuts::side {
namespace {

// Hybrid Qwen3.5 memory keeps recurrent state that cannot be truncated to a
// shorter prefix, so reuse goes through full sequence-state checkpoints (as in
// pcdServer). A checkpoint of the 0.8B model is about 22 MB; shorter prefixes
// are cheaper to decode again than to save.
constexpr size_t PREFIX_MIN_TOKENS = 16;
constexpr size_t PREFIX_CACHE_LIMIT = 256u * 1024 * 1024;

struct checkpoint {
    std::vector<int32_t> ids;
    std::vector<uint8_t> state;
};

// Decodes ids[begin, end) at their own positions onto the current memory.
void decode_range(llama_context * ctx, const std::vector<int32_t> & ids, size_t begin,
                  size_t end, int logits_index, bool embeddings) {
    auto batch = llama_batch_init((int32_t) (end - begin), 0, 1);
    batch.n_tokens = (int32_t) (end - begin);
    for (size_t i = begin; i < end; ++i) {
        const size_t j = i - begin;
        batch.token[j] = ids[i];
        batch.pos[j] = (llama_pos) i;
        batch.n_seq_id[j] = 1;
        batch.seq_id[j][0] = 0;
        batch.logits[j] = (!embeddings && (int) i == logits_index) ? 1 : 0;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("llama_decode failed");
}

} // namespace

struct runner::impl {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    std::vector<ggml_backend_dev_t> devices;
    int hidden = 0;
    int max_length = 4096;
    int gpu_layers = 0;
    bool embeddings = false;
    int offset = 0;   // leading tokens of the last decode that preceded its output batch

    // LRU of prefix checkpoints, front = most recently used.
    std::list<checkpoint> cache;
    size_t cache_bytes = 0;

    ~impl() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

runner::runner(const runner_options & options) : p(std::make_unique<impl>()) {
    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = options.gpu_layers;
    if (!options.device.empty()) {
        std::stringstream stream(options.device);
        std::string name;
        while (std::getline(stream, name, ',')) {
            name = trim(name);
            if (name.empty()) continue;
            ggml_backend_dev_t device = ggml_backend_dev_by_name(name.c_str());
            if (!device) throw std::runtime_error("Unknown device: " + name);
            p->devices.push_back(device);
        }
        p->devices.push_back(nullptr);
        mparams.devices = p->devices.data();
    }
    p->gpu_layers = options.gpu_layers;
    p->model = llama_model_load_from_file(to_utf8(options.model).c_str(), mparams);
    if (!p->model) throw std::runtime_error("Cannot load model: " + options.model.string());
    p->vocab = llama_model_get_vocab(p->model);
    p->hidden = llama_model_n_embd(p->model);
    p->max_length = options.max_length;
    p->embeddings = options.embeddings;

    auto cparams = llama_context_default_params();
    cparams.n_ctx = p->max_length;
    cparams.n_batch = std::max(options.n_batch, p->max_length);
    cparams.n_ubatch = std::min(std::max(options.n_batch, 1), p->max_length);
    cparams.n_seq_max = 1;
    cparams.embeddings = p->embeddings;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    const int threads = options.threads > 0 ? options.threads
                                            : (int) std::max(1u, std::thread::hardware_concurrency());
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    p->ctx = llama_init_from_model(p->model, cparams);
    if (!p->ctx) throw std::runtime_error("Cannot create context");
}

runner::~runner() = default;

int runner::hidden() const { return p->hidden; }
int runner::max_length() const { return p->max_length; }

std::vector<int32_t> runner::tokenize(const std::string & text, bool parse_special) const {
    int n = -llama_tokenize(p->vocab, text.data(), (int) text.size(), nullptr, 0, false, parse_special);
    std::vector<int32_t> tokens(n);
    const int written = llama_tokenize(p->vocab, text.data(), (int) text.size(), tokens.data(), n, false, parse_special);
    if (written < 0) throw std::runtime_error("Tokenization failed");
    tokens.resize(written);
    return tokens;
}

int runner::single_token(const std::string & text) const {
    const auto tokens = tokenize(text, true);
    if (tokens.size() != 1) throw std::runtime_error("Expected one token for: " + text);
    return tokens[0];
}

void runner::decode(const std::vector<int32_t> & ids, int logits_index, size_t prefix, bool keep) {
    if ((int) ids.size() > p->max_length) throw std::length_error("Input exceeds the token budget");
    llama_memory_t memory = llama_get_memory(p->ctx);
    llama_memory_clear(memory, true);
    p->offset = 0;
    // The checkpoint must leave at least one token to decode for the outputs.
    if (keep && prefix >= PREFIX_MIN_TOKENS && prefix < ids.size()) {
        const std::vector<int32_t> key(ids.begin(), ids.begin() + (std::ptrdiff_t) prefix);
        auto hit = std::find_if(p->cache.begin(), p->cache.end(),
                                [&](const checkpoint & c) { return c.ids == key; });
        if (hit != p->cache.end()) {
            p->cache.splice(p->cache.begin(), p->cache, hit);
            const auto & state = p->cache.front().state;
            if (llama_state_seq_set_data_ext(p->ctx, state.data(), state.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) == state.size()
                && llama_memory_seq_pos_max(memory, 0) == (llama_pos) prefix - 1) {
                p->offset = (int) prefix;
            } else {
                p->cache_bytes -= state.size();
                p->cache.pop_front();
                llama_memory_clear(memory, true);
            }
        }
        if (p->offset == 0) {
            decode_range(p->ctx, ids, 0, prefix, -1, p->embeddings);
            p->offset = (int) prefix;
            checkpoint entry{key, {}};
            entry.state.resize(llama_state_seq_get_size_ext(p->ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
            entry.state.resize(llama_state_seq_get_data_ext(p->ctx, entry.state.data(), entry.state.size(), 0,
                                                            LLAMA_STATE_SEQ_FLAGS_NONE));
            if (!entry.state.empty() && entry.state.size() <= PREFIX_CACHE_LIMIT) {
                p->cache_bytes += entry.state.size();
                p->cache.push_front(std::move(entry));
                while (p->cache_bytes > PREFIX_CACHE_LIMIT) {
                    p->cache_bytes -= p->cache.back().state.size();
                    p->cache.pop_back();
                }
            }
        }
    }
    decode_range(p->ctx, ids, (size_t) p->offset, ids.size(), logits_index, p->embeddings);
}

// Outputs of the last decode are indexed from its first decoded token, so a
// restored prefix shifts them.
const float * runner::logits_at(int index) const {
    if (index < p->offset) throw std::out_of_range("Logits requested inside the cached prefix");
    const float * logits = llama_get_logits_ith(p->ctx, index - p->offset);
    if (!logits) throw std::runtime_error("Logits unavailable");
    return logits;
}

const float * runner::embeddings_at(int index) const {
    if (index < p->offset) throw std::out_of_range("Embeddings requested inside the cached prefix");
    const float * embeddings = llama_get_embeddings_ith(p->ctx, index - p->offset);
    if (!embeddings) throw std::runtime_error("Embeddings unavailable");
    return embeddings;
}

std::string runner::backend_name() const {
    char buf[256] = {};
    llama_model_desc(p->model, buf, sizeof(buf));
    return buf;
}

} // namespace dohnuts::side
