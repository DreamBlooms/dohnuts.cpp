// Shared llama.cpp runner for the side profiles. It owns the model/context and
// exposes the two readouts the profiles need: vocab logits at one token
// (decider) and hidden states at arbitrary tokens (kev).
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace dohnuts::side {

struct runner_options {
    std::filesystem::path model;
    std::string device;
    int threads = 0;
    int n_batch = 512;
    int gpu_layers = 0;
    int max_length = 4096;
    bool embeddings = false;   // kev needs hidden states; decider does not
};

class runner {
public:
    explicit runner(const runner_options & options);
    ~runner();

    runner(const runner &) = delete;
    runner & operator=(const runner &) = delete;

    int hidden() const;
    int max_length() const;
    std::string backend_name() const;

    std::vector<int32_t> tokenize(const std::string & text, bool parse_special = true) const;
    int single_token(const std::string & text) const;

    // Decodes ids as one sequence. When embeddings are enabled every token is an
    // output; otherwise only logits_index is (pass -1 to skip). Positions are
    // the token index, matching the profiles' layouts.
    void decode(const std::vector<int32_t> & ids, int logits_index);

    // Valid until the next decode. Indexed by batch token position.
    const float * logits_at(int index) const;
    const float * embeddings_at(int index) const;

private:
    struct impl;
    std::unique_ptr<impl> p;
};

} // namespace dohnuts::side
