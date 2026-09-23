#include "dohnuts/side.hpp"

#include <fstream>
#include <numeric>
#include <stdexcept>

#include "dohnuts/side/common.hpp"
#include "dohnuts/side/profile.hpp"
#include "dohnuts/side/runner.hpp"

namespace dohnuts {
namespace {

json load_config(const std::filesystem::path & path) {
    if (path.empty()) return json::object();
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open config: " + path.string());
    return json::parse(stream);
}

// Common (Dohnuts-shaped) answer fields; the profile adds its native block.
json common_answer(const side::planned_question & q, const std::vector<double> & p) {
    const size_t count = p.size();
    const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
    json answer = {{"type", q.type}, {"confidence", side::rounded(side::entropy_confidence(p))}};
    if (q.type == "noul") {
        answer["noul"] = side::rounded(p[1]);
    } else {
        json probabilities = json::object();
        for (size_t k = 0; k < count; ++k) probabilities[q.keys[k]] = side::rounded(p[k]);
        answer["probabilities"] = probabilities;
        if (q.type == "choice") {
            answer["choice"] = q.keys[best];
        } else {
            double score = 0;
            for (size_t k = 0; k < count; ++k) score += (double) k * p[k];
            answer["score"] = side::rounded(score);
            if (!q.legend.empty()) {
                json legend = json::object();
                for (size_t k = 0; k < q.legend.size(); ++k) legend[std::to_string(k)] = q.legend[k];
                answer["legend"] = legend;
            }
        }
    }
    return answer;
}

} // namespace

struct side_engine::impl {
    model_profile kind = model_profile::decider;
    int gpu_layers = 0;
    std::unique_ptr<side::runner> backend;
    std::unique_ptr<side::profile> handler;
};

side_engine::side_engine(const side_options & options) : p(std::make_unique<impl>()) {
    json config = load_config(options.config);
    p->kind = options.profile;
    p->gpu_layers = options.gpu_layers;

    side::runner_options ropts;
    ropts.model = options.model;
    ropts.device = options.device;
    ropts.threads = options.threads;
    ropts.n_batch = options.n_batch;
    ropts.gpu_layers = options.gpu_layers;
    ropts.max_length = options.max_length;
    ropts.embeddings = p->kind == model_profile::kev;
    p->backend = std::make_unique<side::runner>(ropts);

    if (p->kind == model_profile::kev) {
        if (options.head.empty()) throw std::runtime_error("Kev requires --head");
        p->handler = side::make_kev_profile(*p->backend, config, options.head);
    } else {
        p->handler = side::make_decider_profile(*p->backend, config);
    }
}

side_engine::~side_engine() = default;

model_profile side_engine::profile() const { return p->kind; }
int side_engine::max_length() const { return p->backend->max_length(); }
std::string side_engine::backend_name() const { return p->backend->backend_name(); }

std::string side_engine::device_name() const {
    return p->gpu_layers == 0 ? "CPU" : "GPU (n_gpu_layers=" + std::to_string(p->gpu_layers) + ")";
}

std::vector<int32_t> side_engine::tokenize(const std::string & text) const {
    return p->backend->tokenize(text, true);
}

json side_engine::predict(const json & requests, bool raw) const {
    (void) raw;
    if (!requests.is_array() || requests.empty())
        throw std::invalid_argument("requests must be a nonempty array");

    json output = json::array();
    for (const auto & request : requests) {
        if (!request.is_object()) throw std::invalid_argument("Request must be an object");
        const json & state = request.at("state");
        const json & questions = request.at("questions");
        if (!questions.is_object() || questions.empty())
            throw std::invalid_argument("questions must be a nonempty object");

        std::vector<side::planned_row> rows;
        std::vector<side::planned_question> plan;
        for (auto it = questions.begin(); it != questions.end(); ++it)
            p->handler->plan(it.key(), state, it.value(), rows, plan);

        std::vector<std::vector<double>> row_probs;
        row_probs.reserve(rows.size());
        int input_tokens = 0;
        for (const auto & row : rows) {
            input_tokens += (int) row.ids.size();
            row_probs.push_back(p->handler->score(row));
        }

        json answers = json::object();
        for (const auto & q : plan) {
            json answer;
            if (q.isolated) {
                std::vector<double> fit;
                fit.reserve(q.n_rows);
                for (size_t r = 0; r < q.n_rows; ++r) fit.push_back(row_probs[q.first_row + r][1]);
                const double mass = std::max(1e-9, std::accumulate(fit.begin(), fit.end(), 0.0));
                std::vector<double> dist = fit;
                for (double & value : dist) value /= mass;
                answer = common_answer(q, dist);
                json extra = p->handler->native(q, dist, fit, mass);
                if (!extra.empty()) answer["native"] = extra;
            } else {
                const std::vector<double> & probs = row_probs[q.first_row];
                answer = common_answer(q, probs);
                json extra = p->handler->native(q, probs, {}, 0.0);
                if (!extra.empty()) answer["native"] = extra;
            }
            answers[q.id] = answer;
        }
        output.push_back({{"model", p->handler->model_name()},
                          {"answers", answers},
                          {"usage", {{"input_tokens", input_tokens}, {"output_tokens", 0}, {"images", 0}}}});
    }
    return output;
}

} // namespace dohnuts