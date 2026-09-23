// Side decision-model support (decider, kev). These profiles share the Qwen3.5
// backbone with Dohnuts but replace the readout and prompt entirely, so they
// live here rather than in the core engine. The core engine is untouched.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "dohnuts/profile.hpp"

namespace dohnuts {

using json = nlohmann::ordered_json;

struct side_options {
    model_profile profile = model_profile::decider;
    std::filesystem::path model;
    std::filesystem::path head;      // kev: q rows then k rows, [2 * pointer_dim, hidden + 1]
    std::filesystem::path config;    // decider.json or kev.json
    int threads = 0;        // 0 uses the llama.cpp default.
    int n_batch = 512;      // Side rows are short and decoded one at a time.
    int gpu_layers = 0;
    std::string device;
    int max_length = 4096;
};

class side_engine {
public:
    explicit side_engine(const side_options & options);
    ~side_engine();

    side_engine(const side_engine &) = delete;
    side_engine & operator=(const side_engine &) = delete;

    // Scores every request and returns the answer array in the Dohnuts shape,
    // with profile-native fields under "native".
    json predict(const json & requests, bool raw = false) const;

    model_profile profile() const;
    int max_length() const;
    std::string backend_name() const;
    std::string device_name() const;

    // Tokenizes with special-token parsing, as the prompt builders require.
    std::vector<int32_t> tokenize(const std::string & text) const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

} // namespace dohnuts
