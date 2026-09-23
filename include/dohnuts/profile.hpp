// Side decision-model profiles. Dohnuts is the core model and the only one
// with vision; decider and kev share the Qwen3.5 backbone but use a different
// readout and prompt, so they are kept behind this switch instead of forking
// the core engine.
#pragma once

#include <string>

namespace dohnuts {

enum class model_profile {
    dohnuts,   // pointer head: logit = dot(head, hidden at the candidate marker)
    decider,   // LM head: option-letter logits read at the "Answer: (" slot
    kev,       // bilinear pointer head over the decide and option-end markers
};

// Parses "dohnuts", "decider" or "kev". Throws std::invalid_argument otherwise.
model_profile profile_from_string(const std::string & value);

// The canonical name used in metadata and diagnostics.
const char * profile_name(model_profile profile);

} // namespace dohnuts
