// Official SystemOne Support: run the decision models that official llama.cpp
// ships. This reuses the submodule's tools/server/server-decision.cpp verbatim
// (compiled in place), so a new decision type lands by bumping the submodule,
// not by adding a dohnuts profile. Only the generic readout (labels / markers /
// pointer) is local.
#pragma once

#include <memory>
#include <string>

#include "dohnuts/side.hpp"

namespace dohnuts::side {

class runner;

class official_engine {
public:
    official_engine(runner & backend, const json & config);
    ~official_engine();

    official_engine(const official_engine &) = delete;
    official_engine & operator=(const official_engine &) = delete;

    // Returns the official /v1/systemone output array, one object per request.
    json predict(const json & requests) const;

    std::string model_name() const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

// The "<arch>.decision.type" recorded in the GGUF at `model_path`, or "" when
// the file carries no decision metadata. Reads the header only, no model load.
std::string gguf_decision_type(const std::string & model_path);

} // namespace dohnuts::side
