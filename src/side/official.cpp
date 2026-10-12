#include "dohnuts/side/official.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "dohnuts/side/runner.hpp"

#include "llama.h"
#include "gguf.h"            // gguf_* C API for the metadata-only probe
#include "json.h"            // common_json (llama-common)
#include "server-decision.h" // server_decision_context (llama.cpp, compiled in place)

namespace dohnuts::side {
namespace {

// server_tokens hides its vector; copy it out by index.
std::vector<int32_t> tokens_of(const server_tokens & t) {
    std::vector<int32_t> ids;
    ids.reserve(t.size());
    for (size_t i = 0; i < t.size(); ++i) ids.push_back((int32_t) t[i]);
    return ids;
}

} // namespace

struct official_engine::impl {
    runner & back;
    server_decision_context ctx;
    std::string name = "official";

    impl(runner & backend, const json & config) : back(backend) {
        ctx.init(back.model_handle());
        if (config.contains("version")) name += "-" + config.value("version", std::string());
    }

    // One raw score per option, read the same way server-context.cpp does.
    std::vector<float> readout(server_task & task) const {
        const auto & decision = task.decision;
        const std::vector<int32_t> ids = tokens_of(task.tokens);
        std::vector<float> scores;

        if (!decision.labels.empty()) {
            const int last = (int) ids.size() - 1;
            back.decode(ids, last, 0, false);
            const float * logits = back.logits_at(last);
            for (const llama_token label : decision.labels) scores.push_back(logits[label]);
            return scores;
        }

        if (!decision.order.empty()) {
            throw std::runtime_error("official: joint decision heads are not supported yet");
        }

        const int n_embd_out = back.n_embd_out();
        const int n_pointer  = n_embd_out / 2;
        back.decode(ids, -1, 0, false);
        const float * embd_q = decision.pointer >= 0 ? back.embeddings_at(decision.pointer) : nullptr;
        for (const int32_t marker : decision.markers) {
            const float * embd = back.embeddings_at(marker);
            if (decision.pointer < 0) {
                scores.push_back(embd[decision.column]);
                continue;
            }
            float dot = 0.0f;
            for (int i = 0; i < n_pointer; ++i) dot += embd_q[i] * embd[n_pointer + i];
            scores.push_back(dot / std::sqrt((float) n_pointer));
        }
        return scores;
    }

    json predict(const json & requests) const {
        if (!requests.is_array() || requests.empty())
            throw std::invalid_argument("requests must be a nonempty array");

        json output = json::array();
        for (const auto & request : requests) {
            if (!request.is_object()) throw std::invalid_argument("Request must be an object");

            const common_json body = common_json::parse(request.dump());
            std::vector<raw_buffer> files;
            const common_json state = ctx.parse_state(body, files);
            const std::vector<server_decision_question> questions = ctx.parse_questions(body);

            json answers = json::object();
            int input_tokens = 0;
            for (const auto & question : questions) {
                std::vector<std::vector<float>> scores;
                const size_t n_variants = ctx.n_variants(question);
                for (size_t variant = 0; variant < n_variants; ++variant) {
                    server_task task(SERVER_TASK_TYPE_DECISION);
                    ctx.fill_task(state, questions, question, variant, files, nullptr, {}, task);
                    if (task.tokens.has_mtmd)
                        throw std::runtime_error("official: media input is not supported");
                    input_tokens += (int) task.tokens.size();
                    scores.push_back(readout(task));
                }
                answers[question.id] = json::parse(ctx.format_answer(question, scores).dump());
            }
            output.push_back({{"model", name},
                              {"answers", answers},
                              {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}, {"images", (int) files.size()}}}});
        }
        return output;
    }
};

official_engine::official_engine(runner & backend, const json & config)
    : p(std::make_unique<impl>(backend, config)) {}

official_engine::~official_engine() = default;

json official_engine::predict(const json & requests) const { return p->predict(requests); }

std::string official_engine::model_name() const { return p->name; }

std::string gguf_decision_type(const std::string & model_path) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
    if (ctx == nullptr) {
        return "";
    }

    std::string type;
    const int64_t arch_id = gguf_find_key(ctx, "general.architecture");
    if (arch_id >= 0) {
        const char * arch = gguf_get_val_str(ctx, arch_id);
        if (arch != nullptr) {
            const std::string key = std::string(arch) + ".decision.type";
            const int64_t type_id = gguf_find_key(ctx, key.c_str());
            if (type_id >= 0) {
                const char * value = gguf_get_val_str(ctx, type_id);
                if (value != nullptr) {
                    type = value;
                }
            }
        }
    }

    gguf_free(ctx);
    return type;
}

} // namespace dohnuts::side
