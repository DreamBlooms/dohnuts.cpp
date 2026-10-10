#include "dohnuts/profile.hpp"

#include <stdexcept>

namespace dohnuts {

model_profile profile_from_string(const std::string & value) {
    if (value == "dohnuts") return model_profile::dohnuts;
    if (value == "decider") return model_profile::decider;
    if (value == "kev") return model_profile::kev;
    if (value == "tev1") return model_profile::tev1;
    if (value == "thisthat") return model_profile::thisthat;
    if (value == "jet") return model_profile::jet;
    if (value == "jpt") return model_profile::jpt;
    if (value == "neohorsejev") return model_profile::neohorsejev;
    if (value == "jad") return model_profile::jad;
    throw std::invalid_argument("Unknown profile: " + value);
}

const char * profile_name(model_profile profile) {
    switch (profile) {
        case model_profile::decider: return "decider";
        case model_profile::kev: return "kev";
        case model_profile::tev1: return "tev1";
        case model_profile::thisthat: return "thisthat";
        case model_profile::jet: return "jet";
        case model_profile::jpt: return "jpt";
        case model_profile::neohorsejev: return "neohorsejev";
        case model_profile::jad: return "jad";
        default: return "dohnuts";
    }
}

} // namespace dohnuts
