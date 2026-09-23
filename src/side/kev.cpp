#include "dohnuts/side/profile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "dohnuts/side/common.hpp"

namespace dohnuts::side {
namespace {

constexpr const char * FIM_PREFIX = "<|fim_prefix|>";
constexpr const char * FIM_MIDDLE = "<|fim_middle|>";
constexpr const char * BOX_START  = "<|box_start|>";
constexpr const char * BOX_END    = "<|box_end|>";
constexpr const char * FIM_SUFFIX = "<|fim_suffix|>";

// kev/api.py render: nested objects and arrays become indented text lines.
std::string kev_render(const json & value, int indent = 0) {
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
            std::string item = kev_render(value[i], indent + 1);
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
            text += pad + it.key() + ":\n" + kev_render(it.value(), indent + 1);
        else
            text += pad + it.key() + ": " + kev_render(it.value());
    }
    return text;
}

std::string kev_option_text(const std::string & name, const json & desc) {
    return is_empty(desc) ? name : name + ": " + kev_render(desc);
}

// Rewrites "<|name|>" spans to "<U+00A6 name U+00A6>" so a tokenizer cannot
// forge control tokens, matching kev's user_tokens escaping.
std::string kev_escape(const std::string & text) {
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

class kev_profile final : public profile {
public:
    kev_profile(runner & backend, const json & config, const std::filesystem::path & head_path)
        : back(backend),
          temperature(config.value("temperature", 1.0)),
          pointer_dim(config.at("pointer_dim").get<int>()) {
        id_prefix = back.single_token(FIM_PREFIX);
        id_middle = back.single_token(FIM_MIDDLE);
        id_box_start = back.single_token(BOX_START);
        id_box_end = back.single_token(BOX_END);
        id_suffix = back.single_token(FIM_SUFFIX);
        // One file: q rows then k rows, each [pointer_dim, hidden + 1] with the
        // bias as the last column.
        const size_t weights = (size_t) pointer_dim * (back.hidden() + 1);
        const std::vector<float> combined = read_floats(head_path, 2 * weights);
        head_q.assign(combined.begin(), combined.begin() + weights);
        head_k.assign(combined.begin() + weights, combined.end());
        if (config.contains("version")) name = "kev-" + config.at("version").get<std::string>();
        else name = "kev";
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

        std::vector<std::string> names, options;
        if (type == "noul" || type == "bool") {
            names = {"false", "true"};
            const json c = criteria.is_null() ? json::object() : criteria;
            const json no = c.contains("false") ? c.at("false") : json();
            const json yes = c.contains("true") ? c.at("true") : json();
            options = {is_empty(no) ? "no" : "no: " + kev_render(no),
                       is_empty(yes) ? "yes" : "yes: " + kev_render(yes)};
        } else if (type == "choice") {
            if (!criteria.is_object()) throw std::invalid_argument("choice requires criteria");
            for (auto c = criteria.begin(); c != criteria.end(); ++c) {
                names.push_back(c.key());
                options.push_back(kev_option_text(c.key(), c.value()));
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
                options.push_back(kev_render(list[k]));
            }
        } else {
            throw std::invalid_argument("Unsupported decision type: " + type);
        }
        if (options.size() < 2 || options.size() > 255)
            throw std::invalid_argument("Each question requires 2-255 options");

        planned_question q;
        q.id = id;
        q.type = (type == "bool") ? "noul" : type;
        q.keys = names;
        q.first_row = rows.size();
        rows.push_back(render_row(state, instructions, options));
        questions.push_back(std::move(q));
    }

    std::vector<double> score(const planned_row & row) const override {
        back.decode(row.ids, -1);
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
        if (q.type == "choice") {
            out["confidence"] = count > 1
                ? rounded((p[best] - 1.0 / (double) count) / (1.0 - 1.0 / (double) count)) : 1.0;
        } else if (q.type == "score") {
            double spread = 0;
            for (size_t k = 0; k < count; ++k)
                spread += p[k] * std::abs((double) k - (double) best);
            out["confidence"] = count > 1 ? rounded(1.0 - spread / (double) (count - 1)) : 1.0;
        }
        return out;
    }

    std::string model_name() const override { return name; }

private:
    planned_row render_row(const json & state, const std::string & instruction,
                           const std::vector<std::string> & options) const {
        planned_row row;
        row.n_options = (int) options.size();
        row.ids = back.tokenize(kev_escape(render(state)), true);
        // The prefix is a single token prepended to the state.
        row.ids.insert(row.ids.begin(), id_prefix);
        const size_t limit = std::min(row.ids.size(), (size_t) std::max(0, back.max_length() - 1));
        row.ids.resize(limit);

        row.ids.push_back(id_middle);
        const std::vector<int32_t> instr = back.tokenize(kev_escape(instruction), true);
        row.ids.insert(row.ids.end(), instr.begin(), instr.end());
        for (int j = 0; j < row.n_options; ++j) {
            row.ids.push_back(id_box_start);
            const std::vector<int32_t> text = back.tokenize(kev_escape(options[j]), true);
            row.ids.insert(row.ids.end(), text.begin(), text.end());
            row.ids.push_back(id_box_end);
            row.option_rel.push_back((int) row.ids.size() - 1);
        }
        row.ids.push_back(id_suffix);
        row.slot_rel = (int) row.ids.size() - 1;
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
    std::string name = "kev";
    std::vector<float> head_q;
    std::vector<float> head_k;
    int id_prefix = -1, id_middle = -1, id_box_start = -1, id_box_end = -1, id_suffix = -1;
};

} // namespace

std::unique_ptr<profile> make_kev_profile(runner & backend, const json & config,
                                          const std::filesystem::path & head_path) {
    return std::make_unique<kev_profile>(backend, config, head_path);
}

} // namespace dohnuts::side
