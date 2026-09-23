#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "dohnuts/engine.hpp"

namespace dohnuts {

using dohnuts::json;

// One rendered question: the prompt, its candidate labels, and (for score
// questions) the level index -> description legend.
struct rendered_question {
    std::string content;
    std::vector<std::string> labels;
    json legend;
};

// Renders the shared prompt template and returns the candidate labels.
rendered_question render_question(const std::string & state_text, const json & question);

// Raw scorer logits and labels for one question, before calibration.
json raw_answer(const std::string & type,
                const std::vector<std::string> & labels,
                const std::vector<float> & logits);

// Turn raw scorer logits for one question into the public answer object.
json calibrate_answer(const std::string & type,
                      const std::vector<std::string> & labels,
                      const std::vector<float> & logits,
                      double temperature,
                      const json & legend = json());

// Full predictor backed by a resident engine.
class predictor {
public:
    predictor(engine & engine_ref, json temperatures);

    // Accepts an array of requests and returns the native answer array.
    json predict(const json & requests, bool raw = false);

// True when the engine can score image inputs.
    bool supports_vision() const { return eng.supports_vision(); }

private:
    engine & eng;
    json temperatures;
};

} // namespace dohnuts