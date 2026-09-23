// Shared helpers for the side decision-model profiles. Internal to the
// dohnuts-side library.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace dohnuts::side {

using json = nlohmann::ordered_json;

inline std::string trim(const std::string & value) {
    const auto first = value.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t");
    return value.substr(first, last - first + 1);
}

inline std::string to_utf8(const std::filesystem::path & path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

// Insertion-order JSON with Python's default separators (", ", ": ").
inline std::string dump_python(const json & value) {
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

inline std::string render(const json & value) {
    return value.is_string() ? value.get<std::string>() : dump_python(value);
}

inline bool is_empty(const json & value) {
    return value.is_null() || (value.is_string() && value.get<std::string>().empty());
}

inline std::vector<float> read_floats(const std::filesystem::path & path, size_t expected) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open weights: " + path.string());
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0);
    if (size != (std::streamoff) expected * 4)
        throw std::runtime_error("Weight size mismatch: " + path.string());
    std::vector<float> values(expected);
    file.read(reinterpret_cast<char *>(values.data()), size);
    if (!file) throw std::runtime_error("Truncated weights: " + path.string());
    return values;
}

inline double rounded(double value) { return std::nearbyint(value * 10000.0) / 10000.0; }

inline std::vector<double> softmax(const std::vector<double> & logits, double temperature) {
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> p(logits.size());
    double total = 0;
    for (size_t k = 0; k < logits.size(); ++k) {
        p[k] = std::exp((logits[k] - maximum) / temperature);
        total += p[k];
    }
    for (double & value : p) value /= total;
    return p;
}

inline double entropy_confidence(const std::vector<double> & p) {
    if (p.size() <= 1) return 1.0;
    double entropy = 0;
    for (double value : p) entropy -= value * std::log(std::max(value, 1e-12));
    return std::clamp(1.0 - entropy / std::log((double) p.size()), 0.0, 1.0);
}

inline std::string replace_all(std::string text, const std::string & from, const std::string & to) {
    for (size_t pos = text.find(from); pos != std::string::npos; pos = text.find(from, pos + to.size()))
        text.replace(pos, from.size(), to);
    return text;
}

} // namespace dohnuts::side
