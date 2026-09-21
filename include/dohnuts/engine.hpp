#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace dohnuts {

using json = nlohmann::ordered_json;

struct engine_options {
    std::filesystem::path model;
    std::filesystem::path head;
    int threads = 0;      // 0 uses the llama.cpp default.
    int n_batch = 2048;   // Max tokens decoded in one pass.
};

class engine {
public:
    explicit engine(const engine_options & options);
    ~engine();

    engine(const engine &) = delete;
    engine & operator=(const engine &) = delete;

    // Scores every candidate marker in one batched forward pass.
    struct row_result {
        std::string type;
        std::vector<std::string> labels;
        std::vector<float> logits;
        int tokens = 0;
    };
    std::vector<row_result> score(const std::vector<std::string> & prompts,
                                  const std::vector<std::string> & types,
                                  const std::vector<std::vector<std::string>> & labels);

    int marker_id() const;
    int max_length() const;
    std::string backend_name() const;

private:
    struct impl;
    std::unique_ptr<impl> p;

    void decode_pass(const std::vector<std::vector<int32_t>> & token_rows,
                     const std::vector<std::vector<int>> & marker_pos,
                     size_t start, size_t count,
                     std::vector<row_result> & results);
};

} // namespace dohnuts