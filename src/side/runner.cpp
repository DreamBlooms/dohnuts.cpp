#include "dohnuts/side/runner.hpp"

#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "dohnuts/prefix_cache.hpp"
#include "dohnuts/sys.hpp"
#include "dohnuts/decode.hpp"
#include "dohnuts/side/common.hpp"
#include "llama.h"

namespace dohnuts::side {
namespace {

// Decodes ids[begin, end) at their own positions onto the current memory.
void decode_range(llama_context * ctx, const std::vector<int32_t> & ids, size_t begin,
                  size_t end, int logits_index, bool embeddings) {
    const size_t n = end - begin;
    std::vector<llama_token> toks(n);
    std::vector<llama_pos> pos(n);
    std::vector<llama_seq_id> seq(n, 0);
    std::vector<int8_t> out(n);
    for (size_t i = begin; i < end; ++i) {
        const size_t j = i - begin;
        toks[j] = ids[i];
        pos[j] = (llama_pos) i;
        out[j] = (embeddings || (int) i == logits_index) ? 1 : 0;
    }
    if (decode_tokens(ctx, toks, pos, seq, out) != 0) throw std::runtime_error("llama_process failed");
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
    bool diffusion = false;
    int mask_id = -1;
    int n_seq_max = 1;
    int offset = 0;   // leading tokens of the last decode that preceded its output batch

    // Batch-branches state (see begin_batch).
    bool batch_active = false;
    size_t batch_row = 0;
    // per row: absolute position -> (is_logits, output ordinal)
    std::vector<std::map<int32_t, std::pair<int8_t, int32_t>>> batch_ord;
    const float * batch_logits = nullptr;
    const float * batch_embd = nullptr;
    int batch_n_vocab = 0;
    int batch_n_embd_out = 0;

    // Cross-request LRU of decoded prefixes, keyed by the exact prefix tokens.
    prefix_cache cache;

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
    p->diffusion = options.diffusion;
    p->mask_id = llama_vocab_mask(p->vocab);
    if (p->diffusion && p->mask_id < 0)
        throw std::runtime_error("Model has no mask token (not a masked-diffusion model)");

    auto cparams = llama_context_default_params();
    p->n_seq_max = std::max(1, options.n_seq_max);
    // Reserve a full token budget per sequence when rows are batched.
    cparams.n_ctx = p->max_length * p->n_seq_max;
    cparams.n_batch = std::max(options.n_batch, p->max_length);
    // A non-causal read attends over the whole sequence at once, so it must fit
    // in one ubatch; the causal profiles keep the smaller ubatch.
    cparams.n_ubatch = p->diffusion ? p->max_length
                                    : std::min(std::max(options.n_batch, 1), p->max_length);
    cparams.n_seq_max = p->n_seq_max;
    if (p->n_seq_max > 1) cparams.kv_unified = true;
    cparams.embeddings = p->embeddings;
    cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cparams.flash_attn_type = to_llama_flash_attn(options.flash_attn);
    const int threads = options.threads > 0 ? options.threads : physical_core_count();
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
    if (p->batch_active) return;   // rows were decoded together in begin_batch
    if ((int) ids.size() > p->max_length) throw std::length_error("Input exceeds the token budget");
    llama_memory_t memory = llama_get_memory(p->ctx);
    llama_memory_clear(memory, true);
    p->offset = 0;
    // The checkpoint must leave at least one token to decode for the outputs.
    if (keep && prefix >= PREFIX_MIN_TOKENS && prefix < ids.size()) {
        const std::vector<int32_t> key(ids.begin(), ids.begin() + (std::ptrdiff_t) prefix);
        const std::vector<uint8_t> * cached = p->cache.get(key);
        if (cached) {
            if (llama_state_seq_set_data_ext(p->ctx, cached->data(), cached->size(), 0,
                                             LLAMA_STATE_SEQ_FLAGS_NONE) == cached->size()
                && llama_memory_seq_pos_max(memory, 0) == (llama_pos) prefix - 1) {
                p->offset = (int) prefix;
            } else {
                p->cache.drop_front();
                llama_memory_clear(memory, true);
            }
        }
        if (p->offset == 0) {
            decode_range(p->ctx, ids, 0, prefix, -1, p->embeddings);
            p->offset = (int) prefix;
            std::vector<uint8_t> state(
                llama_state_seq_get_size_ext(p->ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE));
            state.resize(llama_state_seq_get_data_ext(p->ctx, state.data(), state.size(), 0,
                                                      LLAMA_STATE_SEQ_FLAGS_NONE));
            p->cache.put(key, std::move(state));
        }
    }
    decode_range(p->ctx, ids, (size_t) p->offset, ids.size(), logits_index, p->embeddings);
}

// Outputs of the last decode are indexed from its first decoded token, so a
// restored prefix shifts them.
const float * runner::logits_at(int index) const {
    if (p->batch_active) {
        const auto it = p->batch_ord[p->batch_row].find((int32_t) index);
        if (it == p->batch_ord[p->batch_row].end() || it->second.first == 0 || p->batch_logits == nullptr)
            throw std::out_of_range("Logits requested at an unread batch position");
        return p->batch_logits + (size_t) it->second.second * p->batch_n_vocab;
    }
    if (index < p->offset) throw std::out_of_range("Logits requested inside the cached prefix");
    const float * logits = llama_get_logits_ith(p->ctx, index - p->offset);
    if (!logits) throw std::runtime_error("Logits unavailable");
    return logits;
}

int runner::mask_id() const { return p->mask_id; }

void runner::decode_canvas(const std::vector<int32_t> & ids, int read_index) {
    if ((int) ids.size() > p->max_length) throw std::length_error("Input exceeds the token budget");
    llama_memory_t memory = llama_get_memory(p->ctx);
    llama_memory_clear(memory, true);
    p->offset = 0;

    llama_set_causal_attn(p->ctx, false);
    std::vector<llama_token> toks(ids.begin(), ids.end());
    std::vector<llama_pos> pos(ids.size());
    std::vector<llama_seq_id> seq(ids.size(), 0);
    std::vector<int8_t> out(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        pos[i] = (llama_pos) i;
        out[i] = ((int) i == read_index) ? 1 : 0;
    }
    const int rc = decode_tokens(p->ctx, toks, pos, seq, out);
    llama_set_causal_attn(p->ctx, true);
    if (rc != 0) throw std::runtime_error("llama_process failed");
}

const float * runner::embeddings_at(int index) const {
    if (p->batch_active) {
        const auto it = p->batch_ord[p->batch_row].find((int32_t) index);
        if (it == p->batch_ord[p->batch_row].end() || it->second.first != 0 || p->batch_embd == nullptr)
            throw std::out_of_range("Embeddings requested at an unread batch position");
        return p->batch_embd + (size_t) it->second.second * p->batch_n_embd_out;
    }
    if (index < p->offset) throw std::out_of_range("Embeddings requested inside the cached prefix");
    const float * embeddings = llama_get_embeddings_ith(p->ctx, index - p->offset);
    if (!embeddings) throw std::runtime_error("Embeddings unavailable");
    return embeddings;
}

void runner::begin_batch(const std::vector<std::vector<int32_t>> & rows,
                         const std::vector<std::vector<int32_t>> & reads,
                         const std::vector<int8_t> & logits,
                         size_t prefix) {
    if (rows.size() < 2) throw std::invalid_argument("begin_batch needs at least two rows");
    if ((int) rows.size() > p->n_seq_max) throw std::length_error("Too many rows for the context");
    if (prefix == 0) throw std::invalid_argument("begin_batch needs a non-empty prefix");
    for (const auto & row : rows) {
        if (prefix > row.size()) throw std::invalid_argument("shared prefix longer than a row");
    }

    llama_memory_clear(llama_get_memory(p->ctx), true);

    llama_batch_ext * batch = llama_batch_ext_init(p->ctx);
    auto add = [&](llama_seq_id seq, llama_token id, llama_pos pos) -> int32_t {
        const int32_t idx = llama_batch_ext_add_token(batch, seq, id);
        if (idx < 0) {
            llama_batch_ext_free(batch);
            throw std::runtime_error("llama_batch_ext_add_token failed");
        }
        llama_batch_ext_set_pos(batch, idx, &pos);
        return idx;
    };

    // The shared prefix is added once and attached to every sequence.
    std::vector<int32_t> prefix_idx(prefix);
    for (size_t i = 0; i < prefix; ++i) prefix_idx[i] = add(0, rows[0][i], (llama_pos) i);
    for (llama_seq_id s = 1; s < (llama_seq_id) rows.size(); ++s) {
        for (const int32_t idx : prefix_idx) llama_batch_ext_add_seq(batch, idx, s);
    }

    p->batch_ord.assign(rows.size(), {});
    int32_t ord_logits = 0, ord_embd = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        const bool want_logits = logits[r] != 0;
        for (size_t i = prefix; i < rows[r].size(); ++i) {
            const int32_t idx = add((llama_seq_id) r, rows[r][i], (llama_pos) i);
            for (const int32_t pos : reads[r]) {
                if ((int32_t) i != pos) continue;
                if (want_logits) {
                    llama_batch_ext_set_output_logits(batch, idx, true);
                    p->batch_ord[r][pos] = {1, ord_logits++};
                } else {
                    llama_batch_ext_set_output_embd(batch, idx, true);
                    p->batch_ord[r][pos] = {0, ord_embd++};
                }
            }
        }
    }

    const int rc = llama_process(p->ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    if (rc != 0) throw std::runtime_error("llama_process failed");

    p->batch_logits = llama_get_logits(p->ctx);
    p->batch_embd = p->embeddings ? llama_get_embeddings(p->ctx) : nullptr;
    p->batch_n_vocab = llama_vocab_n_tokens(p->vocab);
    p->batch_n_embd_out = llama_model_n_embd_out(p->model);
    p->batch_active = true;
    p->batch_row = 0;
}

void runner::select_row(size_t row) {
    if (!p->batch_active || row >= p->batch_ord.size()) throw std::out_of_range("select_row out of range");
    p->batch_row = row;
}

void runner::end_batch() {
    if (!p->batch_active) return;
    // Free the extra sequences created for the batch; seq 0 is cleared by the
    // next decode. Without this their copied prefix KV accumulates.
    llama_memory_t mem = llama_get_memory(p->ctx);
    for (llama_seq_id s = 1; s < (llama_seq_id) p->batch_ord.size(); ++s) {
        llama_memory_seq_rm(mem, s, -1, -1);
    }
    p->batch_active = false;
    p->batch_row = 0;
    p->batch_ord.clear();
    p->batch_logits = nullptr;
    p->batch_embd = nullptr;
}

bool runner::batching() const { return p->batch_active; }

std::vector<std::vector<std::vector<float>>> runner::decode_shared(
        const std::vector<std::vector<int32_t>> & rows,
        size_t prefix,
        const std::vector<std::vector<int32_t>> & positions,
        bool logits) {
    if (rows.empty()) return {};
    if ((int) rows.size() > p->n_seq_max) throw std::length_error("Too many rows for the context");
    if (prefix == 0) throw std::invalid_argument("decode_shared needs a non-empty prefix");
    for (const auto & row : rows) {
        if (prefix > row.size()) throw std::invalid_argument("shared prefix longer than a row");
    }

    llama_memory_clear(llama_get_memory(p->ctx), true);

    llama_batch_ext * batch = llama_batch_ext_init(p->ctx);
    auto add = [&](llama_seq_id seq, llama_token id, llama_pos pos) -> int32_t {
        const int32_t idx = llama_batch_ext_add_token(batch, seq, id);
        if (idx < 0) {
            llama_batch_ext_free(batch);
            throw std::runtime_error("llama_batch_ext_add_token failed");
        }
        llama_batch_ext_set_pos(batch, idx, &pos);
        return idx;
    };

    // The shared prefix is added once and attached to every sequence.
    std::vector<int32_t> prefix_idx;
    prefix_idx.reserve(prefix);
    for (size_t i = 0; i < prefix; ++i) prefix_idx.push_back(add(0, rows[0][i], (llama_pos) i));
    for (llama_seq_id seq = 1; seq < (llama_seq_id) rows.size(); ++seq) {
        for (const int32_t idx : prefix_idx) llama_batch_ext_add_seq(batch, idx, seq);
    }

    // Tails, remembering the order in which outputs are flagged.
    std::vector<std::pair<size_t, size_t>> order;   // (row, position index)
    for (size_t r = 0; r < rows.size(); ++r) {
        for (size_t i = prefix; i < rows[r].size(); ++i) {
            const int32_t idx = add((llama_seq_id) r, rows[r][i], (llama_pos) i);
            for (size_t k = 0; k < positions[r].size(); ++k) {
                if ((int32_t) i == positions[r][k]) {
                    if (logits) llama_batch_ext_set_output_logits(batch, idx, true);
                    else        llama_batch_ext_set_output_embd(batch, idx, true);
                    order.emplace_back(r, k);
                }
            }
        }
    }

    const int rc = llama_process(p->ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    if (rc != 0) throw std::runtime_error("llama_process failed");

    const size_t width = logits ? (size_t) llama_vocab_n_tokens(p->vocab)
                                : (size_t) llama_model_n_embd_out(p->model);
    const float * out = logits ? llama_get_logits(p->ctx) : llama_get_embeddings(p->ctx);
    if (out == nullptr) throw std::runtime_error("shared decode: outputs unavailable");

    std::vector<std::vector<std::vector<float>>> res(rows.size());
    for (size_t r = 0; r < rows.size(); ++r) res[r].resize(positions[r].size());
    for (size_t j = 0; j < order.size(); ++j) {
        const float * ptr = out + j * width;
        res[order[j].first][order[j].second].assign(ptr, ptr + width);
    }
    return res;
}

std::string runner::backend_name() const {
    char buf[256] = {};
    llama_model_desc(p->model, buf, sizeof(buf));
    return buf;
}

llama_model * runner::model_handle() const { return p->model; }

int runner::n_embd_out() const { return llama_model_n_embd_out(p->model); }

std::string runner::architecture() const {
    return meta("general.architecture");
}

std::string runner::meta(const std::string & key) const {
    char buf[512];
    const int32_t n = llama_model_meta_val_str(p->model, key.c_str(), buf, sizeof(buf));
    return n < 0 ? std::string() : std::string(buf);
}

std::string runner::chat_template(const std::string & name) const {
    const char * tmpl = llama_model_chat_template(p->model, name.c_str());
    return tmpl ? std::string(tmpl) : std::string();
}

std::vector<std::pair<std::string, std::string>> runner::meta_prefix(const std::string & prefix) const {
    std::vector<std::pair<std::string, std::string>> out;
    const int32_t count = llama_model_meta_count(p->model);
    for (int32_t i = 0; i < count; ++i) {
        char key[512];
        if (llama_model_meta_key_by_index(p->model, i, key, sizeof(key)) < 0) continue;
        if (std::string(key).rfind(prefix, 0) != 0) continue;
        char val[512];
        if (llama_model_meta_val_str_by_index(p->model, i, val, sizeof(val)) < 0) continue;
        out.emplace_back(key, val);
    }
    return out;
}

} // namespace dohnuts::side
