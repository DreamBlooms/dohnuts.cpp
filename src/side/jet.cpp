#include "dohnuts/side/profile.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "dohnuts/side/common.hpp"

namespace dohnuts::side {
namespace {

// michaljach/jet is a text-only Qwen3.5 decision model. Every question is one
// forward pass whose next-token logits are restricted to the option label
// tokens (A..Z then single-token two-letter labels for choice; "0".."9" for
// score; "no"/"yes" for noul), exactly like decider/thisthat/tev1. What differs
// is the prompt and that the temperature is fit per question type.
constexpr size_t MAX_OPTIONS = 255;

constexpr const char * SYSTEM =
    "You are Jet, a decision model. Read the state and the question, then answer "
    "with exactly one label from the allowed labels.";

constexpr const char * ASSISTANT_SUFFIX = "<|im_assistant|>\n<|think|>\n\n<|/think|>\n\n";

// Label text for option index j: A..Z then AA, AB, ... (mirrors
// build_single_token_labels' ordering). Used for the prompt, not the readout.
std::string letter_label(size_t j) {
    static const std::string upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    if (j < 26) return std::string(1, upper[(size_t) j]);
    size_t n = j - 26;
    std::string s;
    s += upper[n / 26];
    s += upper[n % 26];
    return s;
}

std::string state_text(const json & state) {
    if (state.is_string()) return state.get<std::string>();
    if (state.is_array()) {
        bool all_strings = true;
        for (const auto & item : state)
            if (!item.is_string()) { all_strings = false; break; }
        if (all_strings) {
            std::string text;
            for (size_t i = 0; i < state.size(); ++i) {
                if (i) text += "\n\n";
                text += state[i].get<std::string>();
            }
            return text;
        }
    }
    return dump_python(state);
}

std::string chat_prefix(const std::string & user) {
    return std::string("<|im_start|>system\n") + SYSTEM + "<|im_end|>\n"
         + "<|im_start|>user\n" + user + "<|im_end|>\n" + ASSISTANT_SUFFIX;
}

std::string render_user(const std::string & state, const std::string & type,
                        const std::string & instructions,
                        const std::vector<std::string> & labels,
                        const std::vector<std::string> & keys,
                        const std::vector<std::string> & descs) {
    std::string out = "<state>\n" + state + "\n</state>\n";
    out += "Question: " + instructions + "\n";
    if (type == "choice") {
        out += "Options:\n";
        for (size_t j = 0; j < labels.size(); ++j)
            out += labels[j] + ": " + keys[j] + ": " + descs[j] + "\n";
        out += "Answer with the label of the best option (" + labels.front() + "-" + labels.back() + ").";
    } else if (type == "score") {
        out += "Scale (lowest to highest):\n";
        for (size_t j = 0; j < labels.size(); ++j)
            out += labels[j] + ": " + descs[j] + "\n";
        out += "Answer with the level number (0-" + std::to_string(labels.size() - 1) + ").";
    } else {
        out += "Answer yes or no.";
    }
    return out;
}

class jet_profile final : public profile {
public:
    jet_profile(runner & backend, const json & config)
        : back(backend),
          t_choice(config.value("t_choice", 1.0)),
          t_score(config.value("t_score", 1.0)),
          t_noul(config.value("t_noul", 1.0)),
          max_state_tokens((size_t) config.value("max_state_tokens", (long) 4096)) {
        labels = build_single_token_labels(back, MAX_OPTIONS, "jet");
        if (config.contains("version")) name = "jet-" + config.at("version").get<std::string>();
        else name = "jet";
    }

    void plan(const std::string & id, const json & state, const json & question,
              std::vector<planned_row> & rows,
              std::vector<planned_question> & questions) const override {
        const std::string type = question.value("type", "choice");
        const json criteria = question.contains("criteria") ? question.at("criteria")
                            : question.contains("options") ? question.at("options") : json();
        const std::string instructions = question.contains("instructions")
            ? render(question.at("instructions"))
            : question.contains("question") ? render(question.at("question")) : "";

        std::vector<std::string> keys, descs, legend;
        std::string qtype = (type == "bool") ? "noul" : type;
        if (type == "noul" || type == "bool") {
            keys = descs = {"no", "yes"};
        } else if (type == "choice") {
            if (!criteria.is_object()) throw std::invalid_argument("choice requires criteria");
            for (auto c = criteria.begin(); c != criteria.end(); ++c) {
                keys.push_back(c.key());
                descs.push_back(is_empty(c.value()) ? "" : render(c.value()));
            }
        } else if (type == "score") {
            if (!criteria.is_array()) throw std::invalid_argument("score requires criteria");
            for (size_t k = 0; k < criteria.size(); ++k) {
                const std::string level = render(criteria[k]);
                keys.push_back(std::to_string(k));
                descs.push_back(level);
                legend.push_back(level);
            }
        } else {
            throw std::invalid_argument("Unsupported decision type: " + type);
        }
        if (keys.size() < 2 || keys.size() > MAX_OPTIONS)
            throw std::invalid_argument("Each question requires 2-255 options");

        const size_t n = keys.size();
        // Prompt labels are the bare letter/number text (A..Z / 0..9 / no,yes);
        // `labels` holds their token ids and is only used for the readout.
        std::vector<std::string> lab;
        for (size_t j = 0; j < n; ++j)
            lab.push_back(qtype == "choice" ? letter_label(j)
                        : qtype == "score" ? std::to_string(j)
                        : std::string(j == 0 ? "no" : "yes"));
        const std::string user = render_user(state_text(state), qtype, instructions, lab, keys, descs);

        planned_row row;
        row.n_options = (int) n;
        row.ids = back.tokenize(chat_prefix(user), true);
        if (row.ids.size() > max_state_tokens) row.ids.resize(max_state_tokens);
        row.slot_rel = (int) row.ids.size() - 1;
        row.letters.assign(labels.begin(), labels.begin() + (std::ptrdiff_t) n);
        row.temperature = (qtype == "choice") ? t_choice
                        : (qtype == "score") ? t_score : t_noul;
        rows.push_back(std::move(row));

        planned_question q;
        q.id = id;
        q.type = qtype;
        q.keys = keys;
        q.legend = legend;
        q.first_row = rows.size() - 1;
        questions.push_back(std::move(q));
    }

    std::vector<double> score(const planned_row & row) const override {
        back.decode(row.ids, row.slot_rel, row.prefix, row.keep);
        const float * logits = back.logits_at(row.slot_rel);
        std::vector<double> raw(row.n_options);
        for (int j = 0; j < row.n_options; ++j) raw[j] = (double) logits[row.letters[j]];
        return softmax(raw, row.temperature > 0 ? row.temperature : temperature);
    }

    json native(const planned_question & q, const std::vector<double> & p,
                const std::vector<double> &, double) const override {
        const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        json out = json::object();
        out["confidence"] = rounded(entropy_confidence(p));
        if (q.type == "score" && !q.legend.empty()) {
            json legend = json::object();
            for (size_t k = 0; k < q.legend.size(); ++k) legend[std::to_string(k)] = q.legend[k];
            out["legend"] = legend;
        }
        return out;
    }

    std::string model_name() const override { return name; }

private:
    runner & back;
    double temperature = 1.0;
    double t_choice = 1.0, t_score = 1.0, t_noul = 1.0;
    std::string name = "jet";
    size_t max_state_tokens = 4096;
    std::vector<int32_t> labels;
};

} // namespace

std::unique_ptr<profile> make_jet_profile(runner & backend, const json & config) {
    return std::make_unique<jet_profile>(backend, config);
}

} // namespace dohnuts::side
