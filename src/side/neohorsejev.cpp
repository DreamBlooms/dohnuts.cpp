#include "dohnuts/side/profile.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "dohnuts/side/common.hpp"

namespace dohnuts::side {
namespace {

// TokenRhythm/NeoHorse-Jev-4B is a Qwen3.5 hybrid backbone plus an independent
// bilinear pointer head (runtime_origin kev). Vision is ignored. The readout is
// kev's: one row per question over the FIM/box special-token layout, reading the
// hidden state at the decide marker and at each option end, then a bilinear form.
//
// The upstream prompt packs a shared state and per-question branches with a
// block-causal mask; on a hybrid (recurrent) backbone that is exactly a set of
// causal rows "state + one branch", which is what thisthat does. So plan_request
// builds one row per question whose ids are the shared state prefix followed by
// that question's branch, and the state prefix is decoded once and cached.
constexpr const char * FIM_PREFIX = "<|fim_prefix|>";
constexpr const char * FIM_MIDDLE = "<|fim_middle|>";
constexpr const char * BOX_START  = "<|box_start|>";
constexpr const char * BOX_END    = "<|box_end|>";
constexpr const char * FIM_SUFFIX = "<|fim_suffix|>";

constexpr size_t MAX_OPTIONS = 255;

// kev/api.py render: nested objects and arrays become indented text lines.
std::string neo_render(const json & value, int indent = 0) {
    const std::string pad(indent * 2, ' ');
    if (value.is_null()) return "";
    if (value.is_string()) return value.get<std::string>();
    if (value.is_boolean()) return value.get<bool>() ? "True" : "False";
    if (value.is_number_integer()) return std::to_string(value.get<long long>());
    if (value.is_number_unsigned()) return std::to_string(value.get<unsigned long long>());
    if (value.is_number_float()) {
        std::ostringstream out;
        out << value.get<double>();
        return out.str();
    }
    if (value.is_array()) {
        std::string text;
        for (size_t i = 0; i < value.size(); ++i) {
            if (i) text += "\n";
            std::string item = neo_render(value[i], indent + 1);
            const auto first = item.find_first_not_of(' ');
            item = first == std::string::npos ? "" : item.substr(first);
            text += pad + "- " + item;
        }
        return text;
    }
    std::string text;
    bool first = true;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!first) text += "\n";
        first = false;
        if (it.value().is_object() || it.value().is_array())
            text += pad + it.key() + ":\n" + neo_render(it.value(), indent + 1);
        else
            text += pad + it.key() + ": " + neo_render(it.value());
    }
    return text;
}

std::string neo_option_text(const std::string & name, const json & desc) {
    return is_empty(desc) ? name : name + ": " + neo_render(desc);
}

std::string neo_escape(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (text[i] == '<' && i + 1 < text.size() && text[i + 1] == '|') {
            size_t j = i + 2;
            while (j < text.size() && (std::isalnum((unsigned char) text[j]) || text[j] == '_')) ++j;
            if (j + 1 < text.size() && text[j] == '|' && text[j + 1] == '>') {
                out += "<\xC2\xA6";
                out.append(text, i + 2, j - (i + 2));
                out += "\xC2\xA6>";
                i = j + 2;
                continue;
            }
        }
        out += text[i++];
    }
    return out;
}

class neohorsejev_profile final : public profile {
public:
    neohorsejev_profile(runner & backend, const json & config, const std::filesystem::path & head_path)
        : back(backend),
          temperature(config.value("temperature", 1.0)),
          pointer_dim(config.at("pointer_dim").get<int>()),
          max_state_tokens((size_t) config.value("max_state_tokens", (long) 2048)),
          max_branch_tokens((size_t) config.value("max_branch_tokens", (long) 8192)) {
        id_prefix = back.single_token(FIM_PREFIX);
        id_middle = back.single_token(FIM_MIDDLE);
        id_box_start = back.single_token(BOX_START);
        id_box_end = back.single_token(BOX_END);
        id_suffix = back.single_token(FIM_SUFFIX);
        const size_t weights = (size_t) pointer_dim * (back.hidden() + 1);
        const std::vector<float> combined = read_floats(head_path, 2 * weights);
        head_q.assign(combined.begin(), combined.begin() + weights);
        head_k.assign(combined.begin() + weights, combined.end());
        if (config.contains("version")) name = "neohorsejev-" + config.at("version").get<std::string>();
        else name = "neohorsejev";
    }

    void plan(const std::string &, const json &, const json &,
              std::vector<planned_row> &, std::vector<planned_question> &) const override {
        throw std::logic_error("neohorsejev plans whole requests");
    }

    void plan_request(const json & state, const json & questions,
                      std::vector<planned_row> & rows,
                      std::vector<planned_question> & out) const override {
        // Shared state prefix: one FIM prefix token then the escaped state.
        std::vector<int32_t> prefix = back.tokenize(neo_escape(render(state)), true);
        prefix.insert(prefix.begin(), id_prefix);
        if (prefix.size() > max_state_tokens) prefix.resize(max_state_tokens);

        for (auto it = questions.begin(); it != questions.end(); ++it) {
            const json & question = it.value();
            const std::string type = question.value("type", "choice");
            const json criteria = question.contains("criteria") ? question.at("criteria")
                                : question.contains("options") ? question.at("options") : json();
            const std::string instructions = question.contains("instructions")
                ? render(question.at("instructions"))
                : question.contains("question") ? render(question.at("question")) : "";

            std::vector<std::string> names, options, legend;
            if (type == "noul" || type == "bool") {
                names = {"false", "true"};
                const json c = criteria.is_null() ? json::object() : criteria;
                const json no = c.contains("false") ? c.at("false") : json();
                const json yes = c.contains("true") ? c.at("true") : json();
                options = {is_empty(no) ? "no" : "no: " + neo_render(no),
                           is_empty(yes) ? "yes" : "yes: " + neo_render(yes)};
            } else if (type == "choice") {
                if (!criteria.is_object()) throw std::invalid_argument("choice requires criteria");
                for (auto c = criteria.begin(); c != criteria.end(); ++c) {
                    names.push_back(c.key());
                    options.push_back(neo_option_text(c.key(), c.value()));
                }
            } else if (type == "score") {
                json list = criteria;
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
                    options.push_back(neo_render(list[k]));
                    legend.push_back(neo_render(list[k]));
                }
            } else {
                throw std::invalid_argument("Unsupported decision type: " + type);
            }
            if (options.size() < 2 || options.size() > MAX_OPTIONS)
                throw std::invalid_argument("Each question requires 2-255 options");

            planned_row row = render_row(prefix, instructions, options);
            rows.push_back(std::move(row));

            planned_question q;
            q.id = it.key();
            q.type = (type == "bool") ? "noul" : type;
            q.keys = names;
            q.legend = legend;
            q.first_row = rows.size() - 1;
            out.push_back(std::move(q));
        }
    }

    std::vector<double> score(const planned_row & row) const override {
        back.decode(row.ids, -1, row.prefix, row.keep);
        const float * h_decide = back.embeddings_at(row.slot_rel);
        const std::vector<double> qv = project(head_q, h_decide);
        const double scale = 1.0 / std::sqrt((double) pointer_dim);
        std::vector<double> raw(row.n_options);
        for (int j = 0; j < row.n_options; ++j) {
            const float * h_opt = back.embeddings_at(row.option_rel[j]);
            const std::vector<double> kv = project(head_k, h_opt);
            double dot = 0;
            for (int d = 0; d < pointer_dim; ++d) dot += qv[d] * kv[d];
            raw[j] = dot * scale;
        }
        return softmax(raw, temperature);
    }

    json native(const planned_question & q, const std::vector<double> & p,
                const std::vector<double> &, double) const override {
        const size_t count = p.size();
        const size_t best = std::max_element(p.begin(), p.end()) - p.begin();
        json out = json::object();
        // Public confidence is the entropy (common_answer). NeoHorse's own
        // confidence is semantic and reported here, matching upstream schema.py.
        if (q.type == "score") {
            double spread = 0;
            for (size_t k = 0; k < count; ++k)
                spread += p[k] * std::abs((double) k - (double) best);
            out["confidence"] = count > 1 ? rounded(1.0 - spread / (double) (count - 1)) : 1.0;
            if (!q.legend.empty()) {
                json legend = json::object();
                for (size_t k = 0; k < q.legend.size(); ++k) legend[std::to_string(k)] = q.legend[k];
                out["legend"] = legend;
                out["level"] = q.legend[best];
            }
        } else {
            out["confidence"] = count > 1
                ? rounded((p[best] - 1.0 / (double) count) / (1.0 - 1.0 / (double) count)) : 1.0;
        }
        return out;
    }

    std::string model_name() const override { return name; }

private:
    planned_row render_row(const std::vector<int32_t> & prefix,
                           const std::string & instruction,
                           const std::vector<std::string> & options) const {
        planned_row row;
        row.n_options = (int) options.size();
        row.ids = prefix;                    // shared state prefix (causal, cached)
        row.prefix = prefix.size();

        row.ids.push_back(id_middle);
        const std::vector<int32_t> instr = back.tokenize(neo_escape(instruction), true);
        row.ids.insert(row.ids.end(), instr.begin(), instr.end());
        for (int j = 0; j < row.n_options; ++j) {
            row.ids.push_back(id_box_start);
            const std::vector<int32_t> text = back.tokenize(neo_escape(options[j]), true);
            row.ids.insert(row.ids.end(), text.begin(), text.end());
            row.ids.push_back(id_box_end);
            row.option_rel.push_back((int) row.ids.size() - 1);
        }
        row.ids.push_back(id_suffix);
        row.slot_rel = (int) row.ids.size() - 1;
        if (row.ids.size() > max_branch_tokens) row.ids.resize(max_branch_tokens);
        return row;
    }

    std::vector<double> project(const std::vector<float> & w, const float * h) const {
        const int hidden = back.hidden();
        std::vector<double> out(pointer_dim, 0.0);
        for (int d = 0; d < pointer_dim; ++d) {
            const float * wrow = w.data() + (size_t) d * (hidden + 1);
            double acc = wrow[hidden];
            for (int i = 0; i < hidden; ++i) acc += (double) wrow[i] * h[i];
            out[d] = acc;
        }
        return out;
    }

    runner & back;
    double temperature = 1.0;
    int pointer_dim = 0;
    size_t max_state_tokens = 2048;
    size_t max_branch_tokens = 8192;
    std::string name = "neohorsejev";
    std::vector<float> head_q;
    std::vector<float> head_k;
    int id_prefix = -1, id_middle = -1, id_box_start = -1, id_box_end = -1, id_suffix = -1;
};

} // namespace

std::unique_ptr<profile> make_neohorsejev_profile(runner & backend, const json & config,
                                                  const std::filesystem::path & head_path) {
    return std::make_unique<neohorsejev_profile>(backend, config, head_path);
}

} // namespace dohnuts::side
