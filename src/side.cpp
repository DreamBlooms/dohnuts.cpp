#include "dohnuts/side.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "dohnuts/side/common.hpp"
#include "dohnuts/side/profile.hpp"
#include "dohnuts/side/runner.hpp"
#ifdef DOHNUTS_DECISION_INTEROP
#include "dohnuts/side/official.hpp"
#endif

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

// Longest token prefix shared by every row, stopped before each row's reads so
// a read never lands inside the shared prefix.
size_t common_prefix(const std::vector<side::planned_row> & rows) {
    if (rows.size() < 2) return 0;
    const std::vector<int32_t> & base = rows[0].ids;
    size_t shared = base.empty() ? 0 : base.size() - 1;
    for (const auto & r : rows) {
        size_t n = std::min(base.size(), r.ids.size());
        if (n > 0) n -= 1;
        size_t k = 0;
        while (k < n && base[k] == r.ids[k]) ++k;
        shared = std::min(shared, k);
        int first = r.slot_rel;
        for (int p : r.option_rel) first = (first < 0) ? p : std::min(first, p);
        if (first >= 0) shared = std::min(shared, (size_t) first);
    }
    return shared;
}

} // namespace

struct side_engine::impl {
    model_profile kind = model_profile::decider;
    int gpu_layers = 0;
    std::unique_ptr<side::runner> backend;
    std::unique_ptr<side::profile> handler;
    bool shared_prefix = false;
#ifdef DOHNUTS_DECISION_INTEROP
    std::unique_ptr<side::official_engine> official;
#endif
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
    ropts.max_length = options.max_length > 0 ? options.max_length
                                              : config.value("max_length", 4096);
    // The official readout needs hidden states only for the pointer/marker
    // types; the label types read logits at one token.
    std::string decision_type;
#ifdef DOHNUTS_DECISION_INTEROP
    if (p->kind == model_profile::official) {
        decision_type = config.contains("decision") && config.at("decision").contains("type")
            ? config.at("decision").at("type").get<std::string>()
            : side::gguf_decision_type(side::to_utf8(options.model));
    }
#endif
    ropts.embeddings = p->kind == model_profile::kev || p->kind == model_profile::neohorsejev
        || (p->kind == model_profile::official
            && (decision_type == "kev" || decision_type == "laya"));
    ropts.diffusion = p->kind == model_profile::jad;
    p->shared_prefix = options.shared_prefix;
    // --shared-prefix batches a request's rows into one decode; it needs room for
    // one sequence per row. The diffusion (jad) profile uses its own canvas path.
    if (options.shared_prefix && p->kind != model_profile::jad) ropts.n_seq_max = 8;
    ropts.flash_attn = options.flash_attn;
    p->backend = std::make_unique<side::runner>(ropts);

#ifdef DOHNUTS_DECISION_INTEROP
    if (p->kind == model_profile::official) {
        p->official = std::make_unique<side::official_engine>(*p->backend, config, options.shared_prefix);
        return;
    }
#else
    if (p->kind == model_profile::official) {
        throw std::runtime_error(
            "Official SystemOne Support is not built; configure with -DDOHNUTS_DECISION_INTEROP=ON");
    }
#endif

    if (p->kind == model_profile::jad) {
        p->handler = side::make_jad_profile(*p->backend, config);
    } else if (p->kind == model_profile::kev) {
        if (options.head.empty()) throw std::runtime_error("Kev requires --head");
        p->handler = side::make_kev_profile(*p->backend, config, options.head);
    } else if (p->kind == model_profile::neohorsejev) {
        if (options.head.empty()) throw std::runtime_error("NeoHorse requires --head");
        p->handler = side::make_neohorsejev_profile(*p->backend, config, options.head);
    } else if (p->kind == model_profile::jet) {
        p->handler = side::make_jet_profile(*p->backend, config);
    } else if (p->kind == model_profile::jpt) {
        p->handler = side::make_jpt_profile(*p->backend, config);
    } else if (p->kind == model_profile::tev1) {
        p->handler = side::make_tev1_profile(*p->backend, config);
    } else if (p->kind == model_profile::thisthat) {
        p->handler = side::make_thisthat_profile(*p->backend, config);
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

#ifdef DOHNUTS_DECISION_INTEROP
    if (p->official) return p->official->predict(requests);
#endif

    json output = json::array();
    for (const auto & request : requests) {
        if (!request.is_object()) throw std::invalid_argument("Request must be an object");
        const json & state = request.at("state");
        const json & questions = request.at("questions");
        if (!questions.is_object() || questions.empty())
            throw std::invalid_argument("questions must be a nonempty object");

        std::vector<side::planned_row> rows;
        std::vector<side::planned_question> plan;
        p->handler->plan_request(state, questions, rows, plan);

        std::vector<std::vector<double>> row_probs;
        row_probs.reserve(rows.size());
        int input_tokens = 0;

        // --shared-prefix: decode the request's rows in one multi-sequence batch
        // (any non-diffusion profile). Falls back to per-row decode otherwise.
        bool batching = p->shared_prefix && rows.size() > 1 && p->kind != model_profile::jad;
        if (batching) {
            const size_t shared = common_prefix(rows);
            size_t total = shared;
            for (const auto & r : rows) total += r.ids.size() - shared;
            if (shared == 0 || total > (size_t) p->backend->max_length()) {
                batching = false;
            } else {
                std::vector<std::vector<int32_t>> ids, reads;
                std::vector<int8_t> logits;
                ids.reserve(rows.size());
                reads.reserve(rows.size());
                logits.reserve(rows.size());
                for (const auto & r : rows) {
                    ids.push_back(r.ids);
                    std::vector<int32_t> rd;
                    rd.push_back(r.slot_rel);
                    if (r.letters.empty()) {
                        for (int p : r.option_rel) rd.push_back(p);   // pointer/marker readout
                        logits.push_back(0);
                    } else {
                        logits.push_back(1);                          // LM-head labels
                    }
                    reads.push_back(std::move(rd));
                }
                p->backend->begin_batch(ids, reads, logits, shared);
            }
        }

        try {
            for (size_t r = 0; r < rows.size(); ++r) {
                rows[r].keep = rows.size() > 1;   // every row of a request starts with its state
                input_tokens += (int) rows[r].ids.size();
                if (batching) p->backend->select_row(r);
                row_probs.push_back(p->handler->score(rows[r]));
            }
        } catch (...) {
            if (batching) p->backend->end_batch();
            throw;
        }
        if (batching) p->backend->end_batch();

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
                answer = p->handler->answer(q, dist);
                if (answer.is_null()) answer = common_answer(q, dist);
                json extra = p->handler->native(q, dist, fit, mass);
                if (!extra.empty()) answer["native"] = extra;
            } else {
                const std::vector<double> & probs = row_probs[q.first_row];
                answer = p->handler->answer(q, probs);
                if (answer.is_null()) answer = common_answer(q, probs);
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