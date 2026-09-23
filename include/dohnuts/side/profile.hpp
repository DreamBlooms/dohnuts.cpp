// Profile interface for the side decision models. Each profile turns a request
// into scored rows and maps the row scores back to the Dohnuts answer shape.
#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "dohnuts/side/runner.hpp"

namespace dohnuts::side {

using json = nlohmann::ordered_json;

// One decoded row and the positions its readout needs.
struct planned_row {
    std::vector<int32_t> ids;
    std::vector<int> option_rel;   // per-option hidden-state positions (kev)
    int slot_rel = -1;             // logits position (decider) or decide position (kev)
    int n_options = 0;
    std::vector<int> letters;      // per-option label token ids (decider)
    size_t prefix = 0;             // leading state tokens, shared by rows of one request
    bool keep = false;             // reuse or checkpoint the prefix (multi-row requests)
};

// One request question; a Score question may expand to several rows.
struct planned_question {
    std::string id;
    std::string type;
    std::vector<std::string> keys;
    std::vector<std::string> legend;
    bool isolated = false;
    size_t first_row = 0;
    size_t n_rows = 1;
};

class profile {
public:
    virtual ~profile() = default;

    // Renders one question into one or more rows. `id` is the request key.
    virtual void plan(const std::string & id, const json & state, const json & question,
                      std::vector<planned_row> & rows,
                      std::vector<planned_question> & questions) const = 0;

    // Scores one row: a probability distribution over its options.
    virtual std::vector<double> score(const planned_row & row) const = 0;

    // Profile-native fields, added to the common answer under "native".
    virtual json native(const planned_question & question,
                        const std::vector<double> & p,
                        const std::vector<double> & level_fit,
                        double fit_mass) const = 0;

    virtual std::string model_name() const = 0;
};

// Builds the profile for a model, loading its config and weights.
std::unique_ptr<profile> make_decider_profile(runner & backend, const json & config);
std::unique_ptr<profile> make_kev_profile(runner & backend, const json & config,
                                          const std::filesystem::path & head_path);

} // namespace dohnuts::side
