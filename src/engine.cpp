#include "dohnuts/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "llama.h"

namespace dohnuts {
namespace {

constexpr int MAX_LENGTH = 4096;   // Per-sequence token budget (dohnuts adapter limit).
constexpr int MAX_SEQS = 8;        // Sequences decoded in one batch.
constexpr const char * MARKER = "<|fim_suffix|>";

std::vector<float> read_head(const std::filesystem::path & path, int expected) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open head weights: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0);
    if (size != std::streamoff(expected) * 4)
        throw std::runtime_error("Head weight size does not match hidden size");
    std::vector<float> values(expected);
    file.read(reinterpret_cast<char *>(values.data()), size);
    if (!file) throw std::runtime_error("Truncated head weights");
    return values;
}

// llama.cpp takes UTF-8 paths; std::filesystem::path::c_str() is wchar_t on
// Windows, so convert explicitly.
std::string to_utf8(const std::filesystem::path & path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

std::string trim(const std::string & value) {
    const auto first = value.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t");
    return value.substr(first, last - first + 1);
}

} // namespace

std::vector<std::string> available_devices() {
    std::vector<std::string> names;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(device);
        const char * description = ggml_backend_dev_description(device);
        names.push_back(std::string(name ? name : "?") + " - " + (description ? description : ""));
    }
    return names;
}

struct engine::impl {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    int marker = -1;
    int hidden = 0;
    int gpu_layers = 0;
    std::vector<ggml_backend_dev_t> devices; // must outlive the model
    std::vector<float> head;

    ~impl() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    void load(const engine_options & options) {
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
                devices.push_back(device);
            }
            if (devices.empty()) throw std::runtime_error("No valid device in --device");
            devices.push_back(nullptr);
            mparams.devices = devices.data();
        }
        gpu_layers = options.gpu_layers;
        model = llama_model_load_from_file(to_utf8(options.model).c_str(), mparams);
        if (!model) throw std::runtime_error("Cannot load model: " + options.model.string());

        vocab = llama_model_get_vocab(model);
        hidden = llama_model_n_embd(model);
        head = read_head(options.head, hidden);

        auto cparams = llama_context_default_params();
        // n_ctx is the total cache; n_ctx_seq = n_ctx / n_seq_max. Reserve a full
        // token budget per sequence.
        cparams.n_ctx = MAX_LENGTH * MAX_SEQS;
        cparams.n_batch = std::max(options.n_batch, MAX_LENGTH);
        cparams.n_ubatch = std::max(options.n_batch, 512);
        cparams.n_seq_max = MAX_SEQS;
        // A unified cache keeps every sequence in one stream so partial-range
        // seq_cp (shared prefix) is supported.
        cparams.kv_unified = true;
        cparams.embeddings = true;
        cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        int threads = options.threads > 0 ? options.threads
                                          : (int) std::max(1u, std::thread::hardware_concurrency());
        cparams.n_threads = threads;
        cparams.n_threads_batch = threads;
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) throw std::runtime_error("Cannot create context");

        llama_token marker_tokens[8];
        const int marker_count = llama_tokenize(vocab, MARKER, (int) std::strlen(MARKER),
                                                marker_tokens, 8, false, true);
        if (marker_count != 1)
            throw std::runtime_error("Marker must tokenize to exactly one token");
        marker = marker_tokens[0];
    }

    std::vector<llama_token> tokenize(const std::string & text) {
        int n = -llama_tokenize(vocab, text.data(), (int) text.size(), nullptr, 0, false, true);
        std::vector<llama_token> tokens(n);
        const int written = llama_tokenize(vocab, text.data(), (int) text.size(),
                                           tokens.data(), n, false, true);
        if (written < 0) throw std::runtime_error("Tokenization failed");
        tokens.resize(written);
        return tokens;
    }
};

engine::engine(const engine_options & options) : p(std::make_unique<impl>()) {
    p->load(options);
}

engine::~engine() = default;

int engine::marker_id() const { return p->marker; }
int engine::max_length() const { return MAX_LENGTH; }

std::string engine::backend_name() const {
    char buf[256] = {};
    llama_model_desc(p->model, buf, sizeof(buf));
    return buf;
}

std::string engine::device_name() const {
    if (p->gpu_layers == 0) return "CPU";
    if (!p->devices.empty()) {
        std::string names;
        for (size_t i = 0; p->devices[i] != nullptr; ++i) {
            if (!names.empty()) names += ",";
            names += ggml_backend_dev_name(p->devices[i]);
        }
        return names;
    }
    return "GPU (n_gpu_layers=" + std::to_string(p->gpu_layers) + ")";
}

std::vector<engine::row_result> engine::score(
        const std::vector<std::string> & prompts,
        const std::vector<std::string> & types,
        const std::vector<std::vector<std::string>> & labels) {
    const size_t rows = prompts.size();
    if (rows == 0 || types.size() != rows || labels.size() != rows)
        throw std::invalid_argument("score requires matching nonempty prompt rows");

    std::vector<std::vector<llama_token>> token_rows(rows);
    std::vector<std::vector<int>> marker_pos(rows);
    for (size_t r = 0; r < rows; ++r) {
        token_rows[r] = p->tokenize(prompts[r]);
        if (token_rows[r].empty()) throw std::invalid_argument("Empty prompt");
        if (token_rows[r].size() > (size_t) MAX_LENGTH)
            throw std::length_error("Input exceeds the token budget");
        if (labels[r].size() < 2 || labels[r].size() > 128)
            throw std::invalid_argument("Each question requires 2-128 candidates");
        for (size_t i = 0; i < token_rows[r].size(); ++i)
            if (token_rows[r][i] == p->marker) marker_pos[r].push_back((int) i);
        if (marker_pos[r].size() != labels[r].size())
            throw std::runtime_error("Marker count does not match candidate count");
    }

    std::vector<row_result> results(rows);
    for (size_t r = 0; r < rows; ++r) {
        results[r].type = types[r];
        results[r].labels = labels[r];
        results[r].tokens = (int) token_rows[r].size();
        results[r].logits.resize(labels[r].size());
    }

    // Decode at most MAX_SEQS sequences per pass; each pass has its own cache.
    for (size_t start = 0; start < rows; start += MAX_SEQS) {
        const size_t count = std::min((size_t) MAX_SEQS, rows - start);
        decode_pass(token_rows, marker_pos, start, count, results);
    }
    return results;
}

void engine::decode_pass(const std::vector<std::vector<llama_token>> & token_rows,
                         const std::vector<std::vector<int>> & marker_pos,
                         size_t start, size_t count,
                         std::vector<row_result> & results) {
    llama_memory_t mem = llama_get_memory(p->ctx);
    llama_memory_clear(mem, true);

    // The prompts share a long `State: ...` prefix. Compute the longest common
    // prefix once and reuse it across sequences, mirroring dohnuts' plan_prefix:
    // cut before the first marker, leave a two-token margin, and align to 64.
    size_t shared = 0;
    if (count > 1) {
        size_t longest = 0;
        shared = token_rows[start].size();
        for (size_t r = start; r < start + count; ++r) {
            longest = std::max(longest, token_rows[r].size());
            if (r == start) continue;
            const size_t limit = std::min(shared, token_rows[r].size());
            size_t i = 0;
            while (i < limit && token_rows[start][i] == token_rows[r][i]) ++i;
            shared = i;
        }
        for (size_t r = start; r < start + count; ++r)
            if (!marker_pos[r].empty())
                shared = std::min(shared, (size_t) marker_pos[r].front());
        shared = std::min(shared, longest > 2 ? longest - 2 : 0);
        shared = shared / 64 * 64;
    }

    auto run = [&](const std::vector<llama_token> & toks,
                   const std::vector<llama_pos> & pos,
                   const std::vector<llama_seq_id> & seq,
                   const std::vector<int8_t> & out) {
        const size_t n = toks.size();
        auto batch = llama_batch_init((int32_t) n, 0, 1);
        batch.n_tokens = (int32_t) n;
        std::memcpy(batch.token, toks.data(), n * sizeof(llama_token));
        std::memcpy(batch.pos, pos.data(), n * sizeof(llama_pos));
        for (size_t i = 0; i < n; ++i) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq[i];
        }
        std::memcpy(batch.logits, out.data(), n * sizeof(int8_t));
        const int rc = llama_decode(p->ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("llama_decode failed");
    };

    // Suffix batch: rows [start, start+count), positions continue from `shared`.
    std::vector<llama_token> tokens;
    std::vector<llama_pos> positions;
    std::vector<llama_seq_id> seq_ids;
    std::vector<int8_t> output;
    std::vector<int> base(count);

    if (shared > 0) {
        std::vector<llama_token> pt(token_rows[start].begin(),
                                    token_rows[start].begin() + shared);
        std::vector<llama_pos> pp(shared);
        for (size_t i = 0; i < shared; ++i) pp[i] = (llama_pos) i;
        run(pt, pp, std::vector<llama_seq_id>(shared, 0), std::vector<int8_t>(shared, 0));
        // Share the prefix state with every other sequence.
        for (size_t r = 1; r < count; ++r)
            llama_memory_seq_cp(mem, 0, (llama_seq_id) r, 0, (llama_pos) shared);
    }

    for (size_t r = 0; r < count; ++r) {
        base[r] = (int) tokens.size();
        const size_t from = shared > 0 ? shared : 0;
        for (size_t i = from; i < token_rows[start + r].size(); ++i) {
            tokens.push_back(token_rows[start + r][i]);
            positions.push_back((llama_pos) i);
            seq_ids.push_back((llama_seq_id) r);
            const bool is_marker =
                std::find(marker_pos[start + r].begin(), marker_pos[start + r].end(), (int) i)
                != marker_pos[start + r].end();
            output.push_back(is_marker ? 1 : 0);
        }
    }

    if (!tokens.empty()) run(tokens, positions, seq_ids, output);

    for (size_t r = 0; r < count; ++r) {
        auto & out = results[start + r];
        const size_t from = shared > 0 ? shared : 0;
        for (size_t k = 0; k < out.labels.size(); ++k) {
            const int token_index = base[r] + marker_pos[start + r][k] - (int) from;
            const float * embd = llama_get_embeddings_ith(p->ctx, token_index);
            if (!embd) throw std::runtime_error("Marker hidden state unavailable");
            float acc = 0;
            for (int h = 0; h < p->hidden; ++h) acc += embd[h] * p->head[h];
            out.logits[k] = acc;
        }
    }
}

} // namespace dohnuts