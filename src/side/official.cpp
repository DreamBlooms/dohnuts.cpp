#include "dohnuts/side/official.hpp"

#include <algorithm>
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
    bool shared_prefix_on = false;
    int shared_prefix_max = 8;

    impl(runner & backend, const json & config, bool shared_prefix) : back(backend) {
        ctx.init(back.model_handle());
        shared_prefix_on = shared_prefix;
        if (config.contains("version")) name += "-" + config.value("version", std::string());
    }

    // Which token positions a task reads, and whether they are logits (label
    // readout) or embeddings (marker/pointer readout).
    struct read_plan {
        std::vector<int32_t> positions;
        bool logits = false;
    };

    static read_plan plan_read(const server_task & task) {
        const auto & d = task.decision;
        read_plan p;
        if (!d.labels.empty()) {
            p.logits = true;
            p.positions.push_back((int32_t) tokens_of(task.tokens).size() - 1);
            return p;
        }
        if (!d.order.empty()) {
            throw std::runtime_error("official: joint decision heads are not supported yet");
        }
        if (d.pointer >= 0) p.positions.push_back(d.pointer);
        for (const int32_t marker : d.markers) p.positions.push_back(marker);
        return p;
    }

    // Raw scores from the copied outputs, the same way server-context.cpp reads.
    static std::vector<float> scores_from(const server_task & task,
                                          const std::vector<std::vector<float>> & outs) {
        const auto & d = task.decision;
        std::vector<float> scores;
        if (!d.labels.empty()) {
            for (const llama_token label : d.labels) scores.push_back(outs[0][label]);
            return scores;
        }
        const int n_embd_out = (int) outs[0].size();
        const int n_pointer  = n_embd_out / 2;
        if (d.pointer < 0) {
            for (const auto & embd : outs) scores.push_back(embd[d.column]);
            return scores;
        }
        const auto & q = outs[0];
        for (size_t k = 1; k < outs.size(); ++k) {
            float dot = 0.0f;
            for (int i = 0; i < n_pointer; ++i) dot += q[i] * outs[k][n_pointer + i];
            scores.push_back(dot / std::sqrt((float) n_pointer));
        }
        return scores;
    }

    // Sequential fallback: one full decode per task (the default, exact path).
    std::vector<float> readout_sequential(server_task & task) const {
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

    // First token index a task reads (label slot, decide pointer or marker).
    static int first_read(const server_task & task) {
        const auto & d = task.decision;
        if (!d.labels.empty()) return (int) tokens_of(task.tokens).size() - 1;
        int pos = d.pointer;
        for (const int32_t marker : d.markers) pos = (pos < 0) ? marker : std::min(pos, marker);
        return pos;
    }

    // Longest token prefix shared by every task, stopped before every read
    // position so a read never lands inside the shared prefix.
    static size_t shared_prefix(const std::vector<server_task> & tasks) {
        if (tasks.size() < 2) return 0;
        const std::vector<int32_t> base = tokens_of(tasks[0].tokens);
        size_t shared = base.empty() ? 0 : base.size() - 1;
        for (const auto & task : tasks) {
            const std::vector<int32_t> ids = tokens_of(task.tokens);
            size_t n = std::min(base.size(), ids.size());
            if (n > 0) n -= 1;
            size_t k = 0;
            while (k < n && base[k] == ids[k]) ++k;
            shared = std::min(shared, k);
            const int read = first_read(task);
            if (read >= 0) shared = std::min(shared, (size_t) read);
        }
        return shared;
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

            // Fill every variant first. With --shared-prefix the common prompt
            // prefix is evaluated once in a single batch (like
            // server_decision_group_tasks); otherwise each task is decoded whole.
            std::vector<server_task> tasks;
            int input_tokens = 0;
            for (size_t qi = 0; qi < questions.size(); ++qi) {
                const size_t n_variants = ctx.n_variants(questions[qi]);
                for (size_t variant = 0; variant < n_variants; ++variant) {
                    server_task task(SERVER_TASK_TYPE_DECISION);
                    ctx.fill_task(state, questions, questions[qi], variant, files, nullptr, {}, task);
                    if (task.tokens.has_mtmd)
                        throw std::runtime_error("official: media input is not supported");
                    input_tokens += (int) task.tokens.size();
                    tasks.push_back(std::move(task));
                }
            }

            const size_t shared = (shared_prefix_on && ctx.can_share_prompt())
                                      ? shared_prefix(tasks) : 0;
            std::vector<std::vector<float>> task_scores(tasks.size());
            if (shared > 0 && tasks.size() > 1 && (int) tasks.size() <= shared_prefix_max) {
                const bool logits = plan_read(tasks[0]).logits;
                std::vector<std::vector<int32_t>> rows, positions;
                for (const auto & task : tasks) {
                    rows.push_back(tokens_of(task.tokens));
                    positions.push_back(plan_read(task).positions);
                }
                const auto outs = back.decode_shared(rows, shared, positions, logits);
                for (size_t r = 0; r < tasks.size(); ++r) task_scores[r] = scores_from(tasks[r], outs[r]);
            } else {
                for (size_t r = 0; r < tasks.size(); ++r) task_scores[r] = readout_sequential(tasks[r]);
            }

            json answers = json::object();
            size_t t = 0;
            for (const auto & question : questions) {
                const size_t n_variants = ctx.n_variants(question);
                std::vector<std::vector<float>> scores(task_scores.begin() + t,
                                                       task_scores.begin() + t + n_variants);
                answers[question.id] = json::parse(ctx.format_answer(question, scores).dump());
                t += n_variants;
            }
            output.push_back({{"model", name},
                              {"answers", answers},
                              {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}, {"images", (int) files.size()}}}});
        }
        return output;
    }
};

official_engine::official_engine(runner & backend, const json & config, bool shared_prefix)
    : p(std::make_unique<impl>(backend, config, shared_prefix)) {}

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
