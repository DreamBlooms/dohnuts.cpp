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
    std::filesystem::path mmproj;   // Vision encoder; empty disables image input.
    int threads = 0;        // 0 uses the llama.cpp default.
    int n_batch = 2048;     // Max tokens decoded in one pass.
    int gpu_layers = 0;     // Layers kept in VRAM; 0 is CPU only, negative is all.
    std::string device;     // Comma-separated ggml device names; empty uses the default.
};

// Names of the compute devices compiled into this build, for --list-devices.
std::vector<std::string> available_devices();

// An encoded image (PNG/JPEG bytes) decoded by the vision encoder.
using encoded_image = std::shared_ptr<const std::vector<uint8_t>>;

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
    // images[r] is the encoded image shared by every question in row r, or
    // nullptr for text-only rows. Must be empty when the engine has no mmproj.
    std::vector<row_result> score(const std::vector<std::string> & prompts,
                                  const std::vector<std::string> & types,
                                  const std::vector<std::vector<std::string>> & labels,
                                  const std::vector<encoded_image> & images = {});

    // True when an mmproj was loaded and images can be scored.
    bool supports_vision() const;

    int marker_id() const;
    int max_length() const;
    std::string backend_name() const;
    std::string device_name() const; // "CPU" or the offloaded devices.

private:
    struct impl;
    std::unique_ptr<impl> p;

    void score_text_rows(const std::vector<std::string> & prompts,
                         const std::vector<std::vector<std::string>> & labels,
                         size_t start, size_t count,
                         std::vector<row_result> & results);
    // Scores rows [start, start+count) that share one image. The leading text
    // and image are decoded once into sequence 0 and copied to the other
    // sequences, mirroring the Python prefix sharing.
    void score_image_group(const std::vector<std::string> & prompts,
                           const std::vector<std::vector<std::string>> & labels,
                           size_t start, size_t count,
                           const std::vector<encoded_image> & images,
                           std::vector<row_result> & results);
    void decode_pass(const std::vector<std::vector<int32_t>> & token_rows,
                     const std::vector<std::vector<int>> & marker_pos,
                     size_t start, size_t count,
                     std::vector<row_result> & results, size_t result_offset);
};

} // namespace dohnuts