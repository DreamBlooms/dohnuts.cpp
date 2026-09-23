#include "dohnuts/profile.hpp"

#include <stdexcept>

namespace dohnuts {

model_profile profile_from_string(const std::string & value) {
    if (value == "dohnuts") return model_profile::dohnuts;
    if (value == "decider") return model_profile::decider;
    if (value == "kev") return model_profile::kev;
    throw std::invalid_argument("Unknown profile: " + value);
}

const char * profile_name(model_profile profile) {
    switch (profile) {
        case model_profile::decider: return "decider";
        case model_profile::kev: return "kev";
        default: return "dohnuts";
    }
}

} // namespace dohnuts
