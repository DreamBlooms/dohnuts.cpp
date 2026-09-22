#include "dohnuts/protocol.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace dohnuts {
namespace {

constexpr const char * MARKER = "<|fim_suffix|>";

int base64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

// Decodes standard base64 (optionally wrapped in a data: URL), ignoring
// whitespace. Throws on any invalid character.
std::vector<uint8_t> decode_base64(const std::string & input) {
    std::string text = input;
    const auto comma = text.find(',');
    if (text.compare(0, 5, "data:") == 0 && comma != std::string::npos)
        text = text.substr(comma + 1);

    std::vector<uint8_t> out;
    out.reserve(text.size() / 4 * 3);
    int buffer = 0;
    int bits = 0;
    for (unsigned char c : text) {
        if (c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        if (c == '=') break;
        const int value = base64_value(c);
        if (value < 0) throw std::invalid_argument("Invalid base64 image data");
        buffer = (buffer << 6) | value;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t) ((buffer >> bits) & 0xFF));
        }
    }
    return out;
}

std::vector<uint8_t> decode_image_field(const json & value) {
    if (!value.is_string()) throw std::invalid_argument("state.image must be a data URL string");
    auto bytes = decode_base64(value.get<std::string>());
    if (bytes.empty()) throw std::invalid_argument("state.image decoded to no bytes");
    return bytes;
}

std::string dump_python(const json & value) {
    if (!value.is_structured()) return value.dump();
    std::string text = value.is_object() ? "{" : "[";
    bool first = true;
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (!first) text += ", ";
        first = false;
        if (value.is_object()) text += json(it.key()).dump() + ": ";
        text += dump_python(it.value());
    }
    return text + (value.is_object() ? "}" : "]");
}

std::string render(const json & value) {
    return value.is_string() ? value.get<std::string>() : dump_python(value);
}

bool is_empty(const json & value) {
    return value.is_null() || (value.is_string() && value.get<std::string>().empty());
}

double rounded(double value) {
    return std::nearbyint(value * 10000.0) / 10000.0;
}

} // namespace

std::pair<std::string, std::vector<std::string>> render_question(
        const std::string & state_text, const json & question) {
    const std::string kind = question.at("type").get<std::string>();
    const json criteria = question.contains("criteria") ? question.at("criteria") : json();
    std::vector<std::string> labels;
    std::vector<std::string> options;

    if (kind == "noul") {
        json c = criteria.is_null() ? json::object() : criteria;
        if (!c.is_object()) throw std::invalid_argument("noul criteria must map false/true");
        const json no = c.contains("false") ? c.at("false") : json();
        const json yes = c.contains("true") ? c.at("true") : json();
        labels = {"false", "true"};
        options = {
            "false: " + (is_empty(no) ? std::string("no, the statement does not hold") : render(no)),
            "true: " + (is_empty(yes) ? std::string("yes, the statement holds") : render(yes)),
        };
    } else if (kind == "choice") {
        if (criteria.is_array()) {
            for (const auto & value : criteria) {
                labels.push_back(value.is_string() ? value.get<std::string>() : value.dump());
                options.push_back(render(value));
            }
        } else if (criteria.is_object()) {
            for (auto it = criteria.begin(); it != criteria.end(); ++it) {
                labels.push_back(it.key());
                options.push_back(is_empty(it.value()) ? it.key()
                                                       : it.key() + ": " + render(it.value()));
            }
        } else {
            throw std::invalid_argument("choice requires a candidate list or mapping");
        }
        auto sorted = labels;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
            throw std::invalid_argument("Candidate labels must be unique");
    } else if (kind == "score") {
        if (!criteria.is_array())
            throw std::invalid_argument("score requires an ordered list of levels");
        for (size_t i = 0; i < criteria.size(); ++i) {
            labels.push_back(std::to_string(i));
            options.push_back("level " + std::to_string(i) + ": " + render(criteria[i]));
        }
    } else {
        throw std::invalid_argument("Unsupported decision type: " + kind);
    }

    if (options.size() < 2 || options.size() > 128)
        throw std::invalid_argument("Each question requires 2-128 candidates");

    std::string content = "State: " + state_text + "\n" + kind + " question: "
                          + (question.contains("instructions")
                                     ? render(question.at("instructions"))
                                     : std::string())
                          + "\nOptions:\n";
    if (content.find(MARKER) != std::string::npos)
        throw std::invalid_argument("Input contains the reserved candidate marker");
    for (const auto & option : options) {
        if (option.find(MARKER) != std::string::npos)
            throw std::invalid_argument("Input contains the reserved candidate marker");
        content += "- " + option + MARKER;
    }
    return {content, labels};
}

json raw_answer(const std::string & type,
                const std::vector<std::string> & labels,
                const std::vector<float> & logits) {
    return {{"type", type}, {"labels", labels}, {"logits", logits}};
}

json calibrate_answer(const std::string & type,
                      const std::vector<std::string> & labels,
                      const std::vector<float> & logits,
                      double temperature) {
    if (labels.size() != logits.size() || labels.empty())
        throw std::invalid_argument("labels and logits must match");
    if (!std::isfinite(temperature) || temperature <= 0)
        throw std::invalid_argument("Calibration temperatures must be positive and finite");
    for (float value : logits)
        if (!std::isfinite(value))
            throw std::runtime_error("Model produced a non-finite logit");

    const size_t count = labels.size();
    const float maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(count);
    double total = 0;
    for (size_t k = 0; k < count; ++k) {
        probabilities[k] = std::exp((double(logits[k]) - maximum) / temperature);
        total += probabilities[k];
    }
    double entropy = 0;
    for (double & p : probabilities) {
        p /= total;
        entropy -= p * std::log(std::max(p, 1e-12));
    }

    json answer = {{"type", type}, {"confidence", 0.0}};
    if (type == "noul") {
        answer["noul"] = rounded(probabilities[1]);
        answer["confidence"] = rounded(std::max(probabilities[0], probabilities[1]));
    } else {
        json mapping = json::object();
        for (size_t k = 0; k < count; ++k) mapping[labels[k]] = rounded(probabilities[k]);
        answer["probabilities"] = mapping;
        if (type == "choice") {
            const size_t best = std::max_element(probabilities.begin(), probabilities.end())
                                - probabilities.begin();
            answer["choice"] = labels[best];
        } else {
            double score = 0;
            for (size_t k = 0; k < count; ++k) score += double(k) * probabilities[k];
            answer["score"] = rounded(score);
        }
        answer["confidence"] = std::clamp(1.0 - entropy / std::log(double(count)), 0.0, 1.0);
    }
    return answer;
}

predictor::predictor(engine & engine_ref, json temps)
    : eng(engine_ref), temperatures(std::move(temps)) {}

json predictor::predict(const json & requests, bool raw) {
    if (!requests.is_array() || requests.empty())
        throw std::invalid_argument("requests must be a nonempty array");

    std::vector<std::string> prompts, types;
    std::vector<std::vector<std::string>> labels;
    std::vector<encoded_image> images;
    // Per request: token estimate is filled after scoring.
    std::vector<int> image_flags;
    // (request index, question id).
    std::vector<std::pair<size_t, std::string>> slots;

    json output = json::array();
    for (const auto & request : requests) {
        if (!request.is_object()) throw std::invalid_argument("Request must be an object");
        const json state = request.at("state");

        // The image is a base64 data URL under state.image; it is excluded from
        // the text state the same way the Python predictor drops state["image"].
        bool has_image = false;
        encoded_image image;
        std::string state_text;
        if (state.is_object() && state.contains("image") && !state.at("image").is_null()) {
            if (state.contains("images"))
                throw std::invalid_argument("Pass one image via state.image");
            if (!eng.supports_vision())
                throw std::invalid_argument("Image input requires an mmproj (--mmproj)");
            auto bytes = decode_image_field(state.at("image"));
            image = std::make_shared<const std::vector<uint8_t>>(std::move(bytes));
            has_image = true;
            json rest = state;
            rest.erase("image");
            state_text = render(rest);
        } else {
            state_text = render(state);
        }

        const json questions = request.at("questions");
        if (!questions.is_object() || questions.empty())
            throw std::invalid_argument("questions must be a nonempty object");
        for (auto it = questions.begin(); it != questions.end(); ++it) {
            auto [content, option_labels] = render_question(state_text, it.value());
            prompts.push_back(std::move(content));
            types.push_back(it.value().at("type").get<std::string>());
            labels.push_back(std::move(option_labels));
            images.push_back(image);
            image_flags.push_back(has_image ? 1 : 0);
            slots.emplace_back(output.size(), it.key());
        }
        output.push_back({{"model", "dohnuts"},
                          {"answers", json::object()},
                          {"usage", {{"input_tokens", 0}, {"images", has_image ? 1 : 0}}}});
    }

    const bool any_image = std::any_of(image_flags.begin(), image_flags.end(),
                                       [](int flag) { return flag != 0; });
    auto results = eng.score(prompts, types, labels, any_image ? images : std::vector<encoded_image>{});

    for (size_t r = 0; r < results.size(); ++r) {
        const auto & row = results[r];
        json answer;
        if (raw) {
            answer = raw_answer(row.type, row.labels, row.logits);
        } else {
            const auto & temp = temperatures.at(row.type);
            answer = calibrate_answer(row.type, row.labels, row.logits, temp.get<double>());
        }
        const auto & [request_index, id] = slots[r];
        output[request_index]["answers"][id] = answer;
        output[request_index]["usage"]["input_tokens"] =
            output[request_index]["usage"]["input_tokens"].get<int>() + row.tokens;
    }
    return output;
}

} // namespace dohnuts