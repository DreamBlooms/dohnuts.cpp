#include "dohnuts/side/profile.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "dohnuts/side/common.hpp"

namespace dohnuts::side {
namespace {

// this-that-model reads one hidden state per answer slot and restricts the
// shared LM head to the option labels, exactly like decider. What differs is
// the prompt. One state and a request's questions share a layout in which every
// question carries its own answer slot:
//
//     Context: <state>
//     Question 1: <text>
//     Options:
//     - label: description
//     (A) option (B) option
//     Answer 1: (
//     Question 2: ...
//     Answer 2: (
//
// No answer letter is ever written into the prompt. The reference answers all
// questions in one causal pass; here each question is its own row whose tokens
// are the prefix of that pass up to its slot. Because attention is causal the
// slot's hidden state is the same either way, and the shared state prefix is
// cached across the rows.
constexpr int NARROW = 10;   // <= this many options uses the plain "(A) .. (J)" rendering
constexpr const char * NARROW_LETTERS = "ABCDEFGHIJ";
constexpr size_t MAX_OPTIONS = 255;
constexpr size_t DEFAULT_MAX_STATE_TOKENS = 1536;

using legend_line = std::pair<std::string, std::string>;

// systemone_protocol._legend: a heading, then one "- label: description" line
// per option (the description is dropped when the caller left it empty).
std::string legend_block(const std::string & heading, const std::vector<legend_line> & lines) {
    std::string out = heading + "\n";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) out += "\n";
        out += "- " + lines[i].first;
        if (!lines[i].second.empty()) out += ": " + lines[i].second;
    }
    return out;
}

std::string state_text(const json & state) {
    return state.is_string() ? state.get<std::string>() : dump_python(state);
}

// One typed question, rendered to the question text and its option strings.
struct question_plan {
    std::string id;
    std::string type;      // noul | choice | score
    std::string text;      // instructions, with the legend appended
    std::vector<std::string> keys;     // the labels the answer is keyed by
    std::vector<std::string> options;  // the option set, as bare labels
    std::vector<std::string> legend;   // per-level descriptions (score only)
};

// systemone_protocol._typed: the labels are read off the criteria, and the
// descriptions go into the question text as a legend under the instructions.
// The option set itself is always the bare labels; a description never appears
// in the "(A) .. (B) .." block.
question_plan build_question(const std::string & id, const json & question) {
    const std::string type = question.value("type", "choice");
    const json criteria = question.contains("criteria") ? question.at("criteria")
                        : question.contains("options") ? question.at("options") : json();
    const std::string instructions = question.contains("instructions")
        ? render(question.at("instructions"))
        : question.contains("question") ? render(question.at("question")) : "";

    question_plan q;
    q.id = id;
    q.type = (type == "bool") ? "noul" : type;
    q.text = instructions;
    if (type == "noul" || type == "bool") {
        q.keys = q.options = {"no", "yes"};
        const json c = criteria.is_null() ? json::object() : criteria;
        const json no = c.contains("false") ? c.at("false") : json();
        const json yes = c.contains("true") ? c.at("true") : json();
        if (!is_empty(no) || !is_empty(yes))
            q.text += "\n\n" + legend_block("Options:", {{"no", is_empty(no) ? "" : render(no)},
                                                          {"yes", is_empty(yes) ? "" : render(yes)}});
    } else if (type == "choice") {
        std::vector<legend_line> lines;
        if (criteria.is_object()) {
            for (auto c = criteria.begin(); c != criteria.end(); ++c) {
                const std::string desc = is_empty(c.value()) ? "" : render(c.value());
                q.keys.push_back(c.key());
                lines.emplace_back(c.key(), desc);
            }
        } else if (criteria.is_array()) {
            for (const auto & value : criteria) {
                const std::string label = render(value);
                q.keys.push_back(label);
                lines.emplace_back(label, "");
            }
        } else {
            throw std::invalid_argument("choice requires criteria");
        }
        q.options = q.keys;
        q.text += "\n\n" + legend_block("Options:", lines);
    } else if (type == "score") {
        if (!criteria.is_array()) throw std::invalid_argument("score requires criteria");
        std::vector<legend_line> lines;
        for (size_t k = 0; k < criteria.size(); ++k) {
            const std::string level = render(criteria[k]);
            q.keys.push_back(std::to_string(k));
            q.legend.push_back(level);
            lines.emplace_back(std::to_string(k), level);
        }
        q.options = q.keys;
        q.text += "\n\n" + legend_block("Levels:", lines);
    } else {
        throw std::invalid_argument("Unsupported decision type: " + type);
    }
    if (q.options.size() < 2 || q.options.size() > MAX_OPTIONS)
        throw std::invalid_argument("Each question requires 2-255 options");
    return q;
}

class thisthat_profile final : public profile {
public:
    thisthat_profile(runner & backend, const json & config)
        : back(backend),
          temperature(config.value("temperature", 1.0)),
          max_state_tokens((size_t) config.value("max_state_tokens",
                                                 (long) DEFAULT_MAX_STATE_TOKENS)) {
        labels = build_single_token_labels(back, MAX_OPTIONS, "this-that");
        if (config.contains("version")) name = "thisthat-" + config.at("version").get<std::string>();
        else name = "thisthat";
    }

    void plan(const std::string &, const json &, const json &,
              std::vector<planned_row> &, std::vector<planned_question> &) const override {
        throw std::logic_error("thisthat plans whole requests");
    }

    void plan_request(const json & state, const json & questions,
                      std::vector<planned_row> & rows,
                      std::vector<planned_question> & out) const override {
        std::vector<question_plan> plan;
        for (auto it = questions.begin(); it != questions.end(); ++it)
            plan.push_back(build_question(it.key(), it.value()));
        const bool multi = plan.size() > 1;

        // The shared context block. The reference truncates "Context:\n" + state
        // to max_state_tokens + 3 tokens (the "+3" being the context header).
        std::vector<int32_t> context = back.tokenize("Context:\n" + state_text(state), true);
        if (context.size() > max_state_tokens + 3) context.resize(max_state_tokens + 3);

        // Row k is the reference's single pass truncated at question k's answer
        // slot: the context, then questions 1..k each followed by its own slot.
        // A later question cannot affect an earlier slot, so the logits at the
        // last slot are the ones the reference reads there. The context prefix
        // is decoded once and cached, then reused by every row.
        for (size_t k = 0; k < plan.size(); ++k) {
            planned_row row;
            row.ids = context;
            row.prefix = context.size();
            for (size_t j = 0; j <= k; ++j) {
                append(row.ids, question_block(plan[j], j, multi));
                if (j == k)
                    row.letters.assign(labels.begin(),
                                       labels.begin() + (std::ptrdiff_t) plan[k].options.size());
                append(row.ids, back.tokenize(answer_slot(j, multi), true));
            }
            row.n_options = (int) plan[k].options.size();
            row.slot_rel = (int) row.ids.size() - 1;
            rows.push_back(std::move(row));

            planned_question q;
            q.id = plan[k].id;
            q.type = plan[k].type;
            q.keys = plan[k].keys;
            q.legend = plan[k].legend;
            q.first_row = k;
            out.push_back(std::move(q));
        }
    }

    std::vector<double> score(const planned_row & row) const override {
        back.decode(row.ids, row.slot_rel, row.prefix, row.keep);
        const float * logits = back.logits_at(row.slot_rel);
        std::vector<double> raw(row.n_options);
        for (int j = 0; j < row.n_options; ++j) raw[j] = (double) logits[row.letters[j]];
        return softmax(raw, temperature);
    }

    json native(const planned_question & q, const std::vector<double> & p,
                const std::vector<double> &, double) const override {
        const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        json out = {{"confidence", rounded(p[best])},
                    {"certainty", rounded(entropy_confidence(p))}};
        if (q.type == "score" && !q.legend.empty()) {
            json legend = json::object();
            for (size_t k = 0; k < q.legend.size(); ++k) legend[std::to_string(k)] = q.legend[k];
            out["legend"] = legend;
        }
        return out;
    }

    std::string model_name() const override { return name; }

private:
    static void append(std::vector<int32_t> & into, const std::vector<int32_t> & ids) {
        into.insert(into.end(), ids.begin(), ids.end());
    }

    // The reference encodes the header, the option block and the answer slot as
    // separate tokenizer calls, so BPE must not merge across those boundaries.
    std::vector<int32_t> question_block(const question_plan & q, size_t k, bool multi) const {
        const std::string head = "\n\nQuestion" + (multi ? " " + std::to_string(k + 1) : "")
                               + ": " + q.text + "\nOptions:";
        std::vector<int32_t> ids = back.tokenize(head, true);
        if (q.options.size() <= NARROW) {
            std::string block;
            for (size_t j = 0; j < q.options.size(); ++j)
                block += std::string("\n(") + NARROW_LETTERS[j] + ") " + q.options[j];
            append(ids, back.tokenize(block, true));
        } else {
            for (size_t j = 0; j < q.options.size(); ++j) {
                append(ids, back.tokenize("\n(", true));
                ids.push_back(labels[j]);
                append(ids, back.tokenize(") " + q.options[j], true));
            }
        }
        return ids;
    }

    static std::string answer_slot(size_t k, bool multi) {
        return "\nAnswer" + (multi ? " " + std::to_string(k + 1) : "") + ": (";
    }

    runner & back;
    double temperature = 1.0;
    std::string name = "thisthat";
    size_t max_state_tokens = DEFAULT_MAX_STATE_TOKENS;
    std::vector<int32_t> labels;
};

} // namespace

std::unique_ptr<profile> make_thisthat_profile(runner & backend, const json & config) {
    return std::make_unique<thisthat_profile>(backend, config);
}

} // namespace dohnuts::side
