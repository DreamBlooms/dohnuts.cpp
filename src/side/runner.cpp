#include "dohnuts/side/runner.hpp"

#include <sstream>
#include <stdexcept>
#include <thread>

#include "dohnuts/side/common.hpp"
#include "llama.h"

namespace dohnuts::side {

struct runner::impl {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    std::vector<ggml_backend_dev_t> devices;
    int hidden = 0;
    int max_length = 4096;
    int gpu_layers = 0;
    bool embeddings = false;

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

void runner::decode(const std::vector<int32_t> & ids, int logits_index) {
    if ((int) ids.size() > p->max_length) throw std::length_error("Input exceeds the token budget");
    llama_memory_clear(llama_get_memory(p->ctx), true);
    auto batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    batch.n_tokens = (int32_t) ids.size();
    for (size_t i = 0; i < ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (!p->embeddings && (int) i == logits_index) ? 1 : 0;
    }
    const int rc = llama_decode(p->ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("llama_decode failed");
}

const float * runner::logits_at(int index) const {
    const float * logits = llama_get_logits_ith(p->ctx, index);
    if (!logits) throw std::runtime_error("Logits unavailable");
    return logits;
}

const float * runner::embeddings_at(int index) const {
    const float * embeddings = llama_get_embeddings_ith(p->ctx, index);
    if (!embeddings) throw std::runtime_error("Embeddings unavailable");
    return embeddings;
}

std::string runner::backend_name() const {
    char buf[256] = {};
    llama_model_desc(p->model, buf, sizeof(buf));
    return buf;
}

} // namespace dohnuts::side
