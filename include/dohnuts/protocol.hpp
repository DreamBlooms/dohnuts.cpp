#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "dohnuts/engine.hpp"

namespace dohnuts {

using dohnuts::json;

// Renders the shared prompt template and returns the candidate labels.
std::pair<std::string, std::vector<std::string>> render_question(
        const std::string & state_text, const json & question);

// Raw scorer logits and labels for one question, before calibration.
json raw_answer(const std::string & type,
                const std::vector<std::string> & labels,
                const std::vector<float> & logits);

// Turn raw scorer logits for one question into the public answer object.
json calibrate_answer(const std::string & type,
                      const std::vector<std::string> & labels,
                      const std::vector<float> & logits,
                      double temperature);

// Full predictor backed by a resident engine.
class predictor {
public:
    predictor(engine & engine_ref, json temperatures);

    // Accepts an array of requests and returns the native answer array.
    json predict(const json & requests, bool raw = false);

private:
    engine & eng;
    json temperatures;
};

} // namespace dohnuts