#include "dohnuts/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace dohnuts {
namespace {

constexpr int MAX_LENGTH = 4096;   // Per-sequence token budget (dohnuts adapter limit).
constexpr int MAX_SEQS = 8;        // Sequences decoded in one batch.
constexpr const char * MARKER = "<|fim_suffix|>";
// Dohnuts preprocesses images to 512x512 (recipe.IMAGE_PIXELS), which the
// 16-pixel patch and 2x2 merge turn into 256 vision tokens. Pin mtmd to the
// same budget so resizing matches the Python reference.
constexpr int IMAGE_TOKENS = (512 / 16 / 2) * (512 / 16 / 2);

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

} // namespace

namespace {

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
    mtmd_context * mtmd = nullptr;
    int marker = -1;
    int hidden = 0;
    int gpu_layers = 0;
    int n_batch = 2048;
    std::string image_prefix;
    std::vector<ggml_backend_dev_t> devices; // must outlive the model
    std::vector<float> head;

    ~impl() {
        if (mtmd) mtmd_free(mtmd);
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
        n_batch = cparams.n_batch;

        if (!options.mmproj.empty()) {
            auto mparams_mtmd = mtmd_context_params_default();
            mparams_mtmd.use_gpu = options.gpu_layers != 0;
            mparams_mtmd.n_threads = threads;
            mparams_mtmd.image_min_tokens = IMAGE_TOKENS;
            mparams_mtmd.image_max_tokens = IMAGE_TOKENS;
            mtmd = mtmd_init_from_file(to_utf8(options.mmproj).c_str(), model, mparams_mtmd);
            if (!mtmd) throw std::runtime_error("Cannot load mmproj: " + options.mmproj.string());
            if (!mtmd_support_vision(mtmd))
                throw std::runtime_error("mmproj has no vision encoder: " + options.mmproj.string());
            image_prefix = std::string(mtmd_default_marker()) + "\n";
        }

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
        const std::vector<std::vector<std::string>> & labels,
        const std::vector<encoded_image> & images) {
    const size_t rows = prompts.size();
    if (rows == 0 || types.size() != rows || labels.size() != rows)
        throw std::invalid_argument("score requires matching nonempty prompt rows");
    if (!images.empty() && images.size() != rows)
        throw std::invalid_argument("images must match the prompt rows");

    const bool has_images = !images.empty() &&
                            std::any_of(images.begin(), images.end(),
                                        [](const auto & img) { return img != nullptr; });
    if (has_images && !p->mtmd)
        throw std::invalid_argument("Image input requires an mmproj (--mmproj)");

    std::vector<row_result> results(rows);
    for (size_t r = 0; r < rows; ++r) {
        results[r].type = types[r];
        results[r].labels = labels[r];
        results[r].logits.resize(labels[r].size());
        if (labels[r].size() < 2 || labels[r].size() > 128)
            throw std::invalid_argument("Each question requires 2-128 candidates");
    }

    // Text-only rows keep the fast batched path. Rows that share an image are
    // decoded together: the leading text and image run once and are copied to
    // the other sequences, so the vision tokens hit the language model once per
    // request instead of once per question.
    std::map<std::string, std::vector<float>> image_cache;
    size_t start = 0;
    while (start < rows) {
        const bool image_row = !images.empty() && images[start] != nullptr;
        if (image_row) {
            size_t count = 1;
            while (count < MAX_SEQS && start + count < rows && images[start + count] != nullptr &&
                   images[start + count].get() == images[start].get())
                ++count;
            score_image_group(prompts, labels, start, count, images, results, image_cache);
            start += count;
            continue;
        }
        size_t count = 0;
        while (count < MAX_SEQS && start + count < rows &&
               (images.empty() || images[start + count] == nullptr))
            ++count;
        score_text_rows(prompts, labels, start, count, results);
        start += count;
    }
    return results;
}

bool engine::supports_vision() const { return p->mtmd != nullptr; }

void engine::score_text_rows(const std::vector<std::string> & prompts,
                             const std::vector<std::vector<std::string>> & labels,
                             size_t start, size_t count,
                             std::vector<row_result> & results) {
    std::vector<std::vector<llama_token>> token_rows(count);
    std::vector<std::vector<int>> marker_pos(count);
    for (size_t r = 0; r < count; ++r) {
        token_rows[r] = p->tokenize(prompts[start + r]);
        if (token_rows[r].empty()) throw std::invalid_argument("Empty prompt");
        if (token_rows[r].size() > (size_t) MAX_LENGTH)
            throw std::length_error("Input exceeds the token budget");
        for (size_t i = 0; i < token_rows[r].size(); ++i)
            if (token_rows[r][i] == p->marker) marker_pos[r].push_back((int) i);
        if (marker_pos[r].size() != labels[start + r].size())
            throw std::runtime_error("Marker count does not match candidate count");
    }
    for (size_t r = 0; r < count; ++r)
        results[start + r].tokens = (int) token_rows[r].size();
    decode_pass(token_rows, marker_pos, 0, count, results, start);
}

void engine::score_image_group(const std::vector<std::string> & prompts,
                               const std::vector<std::vector<std::string>> & labels,
                               size_t start, size_t count,
                               const std::vector<encoded_image> & images,
                               std::vector<row_result> & results,
                               std::map<std::string, std::vector<float>> & image_cache) {
    if (!p->mtmd) throw std::invalid_argument("Image input requires an mmproj (--mmproj)");
    if (count == 0) return;

    std::vector<mtmd_input_chunks *> chunks(count, nullptr);
    std::vector<mtmd_bitmap *> bitmaps(count, nullptr);
    std::vector<std::vector<llama_token>> tails(count);
    std::vector<std::vector<int>> marker_pos(count);
    std::vector<size_t> image_index(count, 0);
    std::vector<size_t> total_tokens(count, 0);
    std::string key;

    auto cleanup = [&]() {
        for (size_t r = 0; r < count; ++r) {
            if (chunks[r]) { mtmd_input_chunks_free(chunks[r]); chunks[r] = nullptr; }
            if (bitmaps[r]) { mtmd_bitmap_free(bitmaps[r]); bitmaps[r] = nullptr; }
        }
    };
    auto fail = [&](const std::string & message) -> void {
        cleanup();
        throw std::runtime_error(message);
    };

    try {
        for (size_t r = 0; r < count; ++r) {
            const encoded_image & img = images[start + r];
            if (!img || img->empty()) throw std::invalid_argument("Image input is empty");

            mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(
                p->mtmd, img->data(), img->size(), /*placeholder=*/false,
                mtmd_helper_init_opt_default());
            bitmaps[r] = wrapper.bitmap;
            if (!bitmaps[r])
                throw std::runtime_error("Cannot decode image (unsupported or corrupt data)");
            const char * id = mtmd_bitmap_get_id(bitmaps[r]);
            const std::string row_key = id ? id : "";
            if (r == 0) key = row_key;
            else if (row_key != key) throw std::invalid_argument("Image group must share one image");

            // Prefix the image tokens the same way the Python adapter does, so
            // the text after it starts with the vision markers.
            const std::string text = p->image_prefix + prompts[start + r];
            mtmd_input_text input = {text.c_str(), text.size(), /*add_special=*/true,
                                     /*parse_special=*/true};
            const mtmd_bitmap * bmps[] = {bitmaps[r]};
            chunks[r] = mtmd_input_chunks_init();
            if (mtmd_tokenize(p->mtmd, chunks[r], &input, bmps, 1) != 0)
                throw std::runtime_error("Image tokenization failed");

            // Split the chunks into a leading part before the image and the
            // tail text after it. total_tokens counts real tokens (the 2D
            // M-RoPE image spans more cache rows than its grid positions).
            const size_t n_chunks = mtmd_input_chunks_size(chunks[r]);
            size_t image_chunk = n_chunks;
            int n_image_chunks = 0;
            for (size_t i = 0; i < n_chunks; ++i) {
                const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks[r], i);
                total_tokens[r] += mtmd_input_chunk_get_n_tokens(chunk);
                if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    image_chunk = i;
                    ++n_image_chunks;
                }
            }
            if (n_image_chunks != 1)
                throw std::invalid_argument("Exactly one image is supported per request");
            if (image_chunk == n_chunks) throw std::runtime_error("Prompt has no image chunk");
            if (total_tokens[r] > (size_t) MAX_LENGTH)
                throw std::length_error("Input exceeds the token budget");
            image_index[r] = image_chunk;

            // Every text chunk after the image forms the per-question tail.
            for (size_t i = image_chunk + 1; i < n_chunks; ++i) {
                const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks[r], i);
                size_t n = 0;
                const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
                tails[r].insert(tails[r].end(), toks, toks + n);
            }
            for (size_t i = 0; i < tails[r].size(); ++i)
                if (tails[r][i] == p->marker) marker_pos[r].push_back((int) i);
            if (marker_pos[r].size() != labels[start + r].size())
                throw std::runtime_error("Marker count does not match candidate count");
        }

        for (size_t r = 1; r < count; ++r)
            if (image_index[r] != image_index[0])
                throw std::runtime_error("Image group layout mismatch");

        // The shared tail prefix runs once and is copied to the other
        // sequences. Cut before the first marker, leave a two-token margin, and
        // align to 64 like the text path.
        size_t shared = 0;
        if (count > 1) {
            size_t longest = 0;
            shared = tails[0].size();
            for (size_t r = 0; r < count; ++r) {
                longest = std::max(longest, tails[r].size());
                if (r == 0) continue;
                const size_t limit = std::min(shared, tails[r].size());
                size_t i = 0;
                while (i < limit && tails[0][i] == tails[r][i]) ++i;
                shared = i;
            }
            for (size_t r = 0; r < count; ++r)
                if (!marker_pos[r].empty())
                    shared = std::min(shared, (size_t) marker_pos[r].front());
            shared = std::min(shared, longest > 2 ? longest - 2 : 0);
            shared = shared / 64 * 64;
        }

        llama_memory_t mem = llama_get_memory(p->ctx);
        llama_memory_clear(mem, true);

        // 1. Decode the leading text and the image into sequence 0.
        llama_pos n_past = 0;
        for (size_t i = 0; i < image_index[0]; ++i) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks[0], i);
            if (mtmd_helper_eval_chunk_single(p->mtmd, p->ctx, chunk, n_past, 0, p->n_batch,
                                              /*logits_last=*/false, &n_past) != 0)
                fail("Image decode failed");
        }
        {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks[0], image_index[0]);
            std::vector<float> * encoded = nullptr;
            auto cached = image_cache.find(key);
            if (cached != image_cache.end()) encoded = &cached->second;
            if (!encoded) {
                if (mtmd_encode_chunk(p->mtmd, chunk) != 0) fail("Image encoding failed");
                const int n_embd = llama_model_n_embd_inp(p->model);
                const size_t n = (size_t) n_embd * mtmd_input_chunk_get_n_tokens(chunk);
                const float * out = mtmd_get_output_embd(p->mtmd);
                if (!out) fail("Image embeddings unavailable");
                encoded = &image_cache.emplace(key, std::vector<float>(out, out + n)).first->second;
            }
            if (mtmd_helper_decode_image_chunk(p->mtmd, p->ctx, chunk, encoded->data(), n_past, 0,
                                               p->n_batch, &n_past, nullptr, nullptr) != 0)
                fail("Image decode failed");
        }

        // 2. Decode the shared tail prefix into sequence 0 as well.
        if (shared > 0) {
            auto batch = llama_batch_init((int32_t) shared, 0, 1);
            batch.n_tokens = (int32_t) shared;
            std::memcpy(batch.token, tails[0].data(), shared * sizeof(llama_token));
            for (size_t i = 0; i < shared; ++i) {
                batch.pos[i] = n_past + (llama_pos) i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = 0;
            }
            const int rc = llama_decode(p->ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) fail("llama_decode failed");
            n_past += (llama_pos) shared;
        }

        // 3. Copy the shared prefix to every other sequence.
        for (size_t r = 1; r < count; ++r)
            llama_memory_seq_cp(mem, 0, (llama_seq_id) r, 0, n_past);

        // 4. Decode each row's remaining tail and read its marker states.
        for (size_t r = 0; r < count; ++r) {
            results[start + r].tokens = (int) total_tokens[r];
            const size_t n = tails[r].size() - shared;
            if (n == 0) continue;
            const llama_token * toks = tails[r].data() + shared;
            auto batch = llama_batch_init((int32_t) n, 0, 1);
            batch.n_tokens = (int32_t) n;
            std::memcpy(batch.token, toks, n * sizeof(llama_token));
            std::vector<int> local_marker;
            for (size_t i = 0; i < n; ++i) {
                batch.pos[i] = n_past + (llama_pos) i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = (llama_seq_id) r;
                const bool is_marker = toks[i] == p->marker;
                batch.logits[i] = is_marker ? 1 : 0;
                if (is_marker) local_marker.push_back((int) i);
            }
            const int rc = llama_decode(p->ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) fail("llama_decode failed");
            if (local_marker.size() != labels[start + r].size())
                fail("Marker count does not match candidate count");
            for (size_t k = 0; k < local_marker.size(); ++k) {
                const float * embd = llama_get_embeddings_ith(p->ctx, local_marker[k]);
                if (!embd) fail("Marker hidden state unavailable");
                float acc = 0;
                for (int h = 0; h < p->hidden; ++h) acc += embd[h] * p->head[h];
                results[start + r].logits[k] = acc;
            }
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

void engine::decode_pass(const std::vector<std::vector<llama_token>> & token_rows,
                         const std::vector<std::vector<int>> & marker_pos,
                         size_t start, size_t count,
                         std::vector<row_result> & results, size_t result_offset) {
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
        auto & out = results[result_offset + r];
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