#include "dohnuts/side/profile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

#include "dohnuts/side/common.hpp"

namespace dohnuts::side {
namespace {

constexpr size_t ANNOTATE_MIN = 8;
constexpr int NARROW = 10;   // <= this many options uses the "(A) .. (J)" rendering
constexpr const char * NARROW_LETTERS = "ABCDEFGHIJ";
constexpr const char * ISOLATED_TEMPLATE = "{q}\nProposed answer: {level}\nDoes the proposed answer fit?";

// decider/systemone.py annotate_indices: writes positions into long arrays.
json annotate_indices(const json & value, size_t min_len = ANNOTATE_MIN) {
    if (value.is_array()) {
        json out = json::array();
        if (value.size() >= min_len) {
            for (size_t i = 0; i < value.size(); ++i) {
                json entry = json::object();
                entry["_index"] = i;
                if (value[i].is_object()) {
                    json rest = annotate_indices(value[i], min_len);
                    for (auto it = rest.begin(); it != rest.end(); ++it) entry[it.key()] = it.value();
                } else {
                    entry["value"] = annotate_indices(value[i], min_len);
                }
                out.push_back(std::move(entry));
            }
            return out;
        }
        for (const auto & item : value) out.push_back(annotate_indices(item, min_len));
        return out;
    }
    if (value.is_object()) {
        json out = json::object();
        for (auto it = value.begin(); it != value.end(); ++it)
            out[it.key()] = annotate_indices(it.value(), min_len);
        return out;
    }
    return value;
}

std::string render_state(const json & state) {
    if (state.is_string()) return state.get<std::string>();
    return dump_python(annotate_indices(state));
}

std::string strip_level_number(const std::string & text) {
    size_t i = 0;
    while (i < text.size() && std::isspace((unsigned char) text[i])) ++i;
    if (i < text.size() && text[i] == '-') ++i;
    const size_t digits = i;
    while (i < text.size() && std::isdigit((unsigned char) text[i])) ++i;
    if (i == digits) return text;
    while (i < text.size() && std::isspace((unsigned char) text[i])) ++i;
    if (i + 1 < text.size() && text[i] == ':') {
        ++i;
        while (i < text.size() && std::isspace((unsigned char) text[i])) ++i;
        return text.substr(i);
    }
    return text;
}

class decider_profile final : public profile {
public:
    decider_profile(runner & backend, const json & config)
        : back(backend),
          temperature(config.value("temperature", 1.0)),
          isolated_levels(config.value("isolated_levels", true)) {
        build_letters();
        if (config.contains("version")) name = "decider-" + config.at("version").get<std::string>();
        else name = "decider";
    }

    void plan(const std::string & id, const json & state, const json & question,
              std::vector<planned_row> & rows,
              std::vector<planned_question> & questions) const override {
        const std::string type = question.value("type", "choice");
        json criteria = question.contains("criteria") ? question.at("criteria")
                       : question.contains("options") ? question.at("options") : json();
        const std::string instructions = question.contains("instructions")
            ? render(question.at("instructions"))
            : question.contains("question") ? render(question.at("question")) : "";

        std::vector<std::string> names, options, legend;
        json list = criteria;
        if (type == "noul" || type == "bool") {
            names = {"false", "true"};
            const json c = criteria.is_null() ? json::object() : criteria;
            const json no = c.contains("false") ? c.at("false") : json();
            const json yes = c.contains("true") ? c.at("true") : json();
            options = {is_empty(no) ? "no" : "no: " + render(no),
                       is_empty(yes) ? "yes" : "yes: " + render(yes)};
        } else if (type == "choice") {
            if (criteria.is_object()) {
                for (auto c = criteria.begin(); c != criteria.end(); ++c) {
                    names.push_back(c.key());
                    options.push_back(is_empty(c.value()) ? c.key() : c.key() + ": " + render(c.value()));
                }
            } else if (criteria.is_array()) {
                for (const auto & value : criteria) {
                    names.push_back(render(value));
                    options.push_back(render(value));
                }
            } else {
                throw std::invalid_argument("choice requires criteria");
            }
        } else if (type == "score") {
            if (criteria.is_object()) {
                std::vector<std::pair<double, json>> ordered;
                for (auto c = criteria.begin(); c != criteria.end(); ++c)
                    ordered.emplace_back(std::stod(c.key()), c.value());
                std::sort(ordered.begin(), ordered.end(),
                          [](const auto & a, const auto & b) { return a.first < b.first; });
                list = json::array();
                for (const auto & entry : ordered) list.push_back(entry.second);
            }
            if (!list.is_array()) throw std::invalid_argument("score requires criteria");
            for (size_t k = 0; k < list.size(); ++k) {
                names.push_back(std::to_string(k));
                options.push_back(std::to_string(k) + ": " + render(list[k]));
                legend.push_back(render(list[k]));
            }
        } else {
            throw std::invalid_argument("Unsupported decision type: " + type);
        }
        if (options.size() < 2 || options.size() > 255)
            throw std::invalid_argument("Each question requires 2-255 options");

        const std::string state_text = render_state(state);
        planned_question q;
        q.id = id;
        q.type = (type == "bool") ? "noul" : type;
        q.first_row = rows.size();
        q.keys = names;
        q.legend = legend;

        const bool spec_isolated = question.value("isolated", true);
        if (type == "score" && isolated_levels && spec_isolated) {
            q.isolated = true;
            q.n_rows = options.size();
            for (size_t k = 0; k < options.size(); ++k) {
                std::string text = replace_all(ISOLATED_TEMPLATE, "{q}", instructions);
                text = replace_all(text, "{level}", strip_level_number(legend[k]));
                rows.push_back(render_row(state_text, text, {"no", "yes"}));
            }
        } else {
            rows.push_back(render_row(state_text, instructions, options));
        }
        questions.push_back(std::move(q));
    }

    std::vector<double> score(const planned_row & row) const override {
        back.decode(row.ids, row.slot_rel, row.prefix, row.keep);
        const float * logits = back.logits_at(row.slot_rel);
        std::vector<double> raw(row.n_options);
        for (int j = 0; j < row.n_options; ++j) raw[j] = (double) logits[row.letters[j]];
        return softmax(raw, temperature);
    }

    json native(const planned_question & q, const std::vector<double> & p,
                const std::vector<double> & level_fit, double fit_mass) const override {
        const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        json out = {{"confidence", rounded(p[best])},
                    {"certainty", rounded(entropy_confidence(p))}};
        if (q.type == "score") {
            json legend = json::object();
            for (size_t k = 0; k < q.legend.size(); ++k) legend[std::to_string(k)] = q.legend[k];
            out["legend"] = legend;
            if (q.isolated) {
                json fits = json::object();
                for (size_t k = 0; k < level_fit.size(); ++k) fits[std::to_string(k)] = rounded(level_fit[k]);
                out["level_fit"] = fits;
                out["fit_mass"] = rounded(fit_mass);
            }
        }
        return out;
    }

    std::string model_name() const override { return name; }

private:
    planned_row render_row(const std::string & state_text, const std::string & question_text,
                           const std::vector<std::string> & options) const {
        planned_row row;
        row.n_options = (int) options.size();
        row.ids = back.tokenize("Context:\n" + state_text, true);
        row.prefix = row.ids.size();
        const std::string head = "\n\nQuestion: " + question_text + "\nOptions:";
        const std::string tail = "\nAnswer: (";
        if (row.n_options <= NARROW) {
            // Python tokenizes header, options and slot as one piece, so BPE
            // merges across those boundaries must match.
            std::string piece = head;
            for (int j = 0; j < row.n_options; ++j) {
                piece += std::string("\n(") + NARROW_LETTERS[j] + ") " + options[j];
                row.letters.push_back(letters[j]);
            }
            piece += tail;
            const std::vector<int32_t> ids = back.tokenize(piece, true);
            row.ids.insert(row.ids.end(), ids.begin(), ids.end());
        } else {
            std::vector<int32_t> block = back.tokenize(head, true);
            for (int j = 0; j < row.n_options; ++j) {
                block.insert(block.end(), open_paren.begin(), open_paren.end());
                block.push_back(letters[j]);
                const std::vector<int32_t> text = back.tokenize(") " + options[j], true);
                block.insert(block.end(), text.begin(), text.end());
                row.letters.push_back(letters[j]);
            }
            const std::vector<int32_t> t = back.tokenize(tail, true);
            block.insert(block.end(), t.begin(), t.end());
            row.ids.insert(row.ids.end(), block.begin(), block.end());
        }
        row.slot_rel = (int) row.ids.size() - 1;
        return row;
    }

    void build_letters() {
        const std::string U = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        std::vector<std::string> names;
        for (char c : U) names.push_back(std::string(1, c));
        for (char a : U)
            for (char b : U) names.push_back(std::string(1, a) + std::string(1, b));
        for (const auto & name : names) {
            if ((int) letters.size() >= 255) break;
            if (back.tokenize(name, false).size() == 1) letters.push_back(back.tokenize(name, false)[0]);
        }
        std::vector<int32_t> sorted = letters;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::runtime_error("Letter label tokens are not unique");
        open_paren = back.tokenize("\n(", false);
    }

    runner & back;
    double temperature = 1.0;
    bool isolated_levels = true;
    std::string name = "decider";
    std::vector<int32_t> letters;
    std::vector<int32_t> open_paren;
};

} // namespace

std::unique_ptr<profile> make_decider_profile(runner & backend, const json & config) {
    return std::make_unique<decider_profile>(backend, config);
}

} // namespace dohnuts::side
