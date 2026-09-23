#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "dohnuts/engine.hpp"
#include "dohnuts/http.hpp"
#include "dohnuts/profile.hpp"
#include "dohnuts/protocol.hpp"
#include "dohnuts/side.hpp"

namespace {

using json = dohnuts::json;

struct options {
    bool server = false;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string profile;    // empty = detect from --metadata (dohnuts, decider or kev)
    std::string model;
    std::string head;       // dohnuts: head.f32; kev: kev-head.f32
    std::string mmproj;     // vision encoder; enables image input
    std::string metadata;   // dohnuts.json (dohnuts) or decider.json / kev.json
    std::string input;      // JSONL file for CLI mode
    std::string api_key;
    std::string cors_origin = "*";
    std::string device;
    int threads = 0;
    int gpu_layers = 0;
    size_t max_questions = 8;
    bool batching = true;
    bool raw = false;
    bool list_devices = false;
};

json load_json(const std::string & path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open " + path);
    return json::parse(stream);
}

json temperatures_from(const json & metadata) {
    const json & temps = metadata.at("temperatures");
    return {{"choice", temps.at("choice")},
            {"score", temps.at("score")},
            {"noul", temps.at("noul")}};
}

void usage() {
    std::cerr << "usage: dohnuts-cli --model M.gguf --metadata M.json [--head H.f32] "
                 "[--mmproj MMPROJ.gguf] [--profile NAME] "
                 "[--gpu-layers N] [--device NAME[,NAME]] [--threads N] "
                 "[--server --host H --port P --api-key K --cors-origin ORIGIN "
                 "--max-questions N --no-batching | --input requests.jsonl [--raw]]\n"
                 "       dohnuts-cli --list-devices\n"
                 "\n"
                 "The metadata file names its profile: dohnuts (default), decider or kev.\n"
                 "dohnuts needs --head head.f32; kev needs --head kev-head.f32.\n";
}

} // namespace

int main(int argc, char ** argv) {
    options opts;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (++i >= argc) throw std::invalid_argument("Missing value for " + arg);
                return argv[i];
            };
            if (arg == "--server") opts.server = true;
            else if (arg == "--host") opts.host = next();
            else if (arg == "--port") opts.port = std::stoi(next());
            else if (arg == "--profile") opts.profile = next();
            else if (arg == "--model") opts.model = next();
            else if (arg == "--head") opts.head = next();
            else if (arg == "--mmproj") opts.mmproj = next();
            else if (arg == "--metadata") opts.metadata = next();
            else if (arg == "--input") opts.input = next();
            else if (arg == "--api-key") opts.api_key = next();
            else if (arg == "--cors-origin") opts.cors_origin = next();
            else if (arg == "--threads") opts.threads = std::stoi(next());
            else if (arg == "--gpu-layers") opts.gpu_layers = std::stoi(next());
            else if (arg == "--device") opts.device = next();
            else if (arg == "--max-questions") opts.max_questions = std::stoul(next());
            else if (arg == "--no-batching") opts.batching = false;
            else if (arg == "--raw") opts.raw = true;
            else if (arg == "--list-devices") opts.list_devices = true;
            else { usage(); return 2; }
        }
        if (opts.list_devices) {
            for (const auto & name : dohnuts::available_devices()) std::cout << name << '\n';
            return 0;
        }
        if (opts.model.empty() || opts.metadata.empty()) { usage(); return 2; }

        // The metadata file names its profile ("dohnuts", "decider" or "kev");
        // --profile overrides it. Unknown or missing defaults to dohnuts.
        const json metadata = load_json(opts.metadata);
        std::string profile_name = opts.profile;
        if (profile_name.empty()) profile_name = metadata.value("profile", "dohnuts");
        const dohnuts::model_profile profile = dohnuts::profile_from_string(profile_name);

        // The predictor callback is the same shape for every profile; only the
        // backend differs. Side profiles have no vision and require kev's head.
        dohnuts::http_options http;
        http.host = opts.host;
        http.port = opts.port;
        http.model = profile_name;
        http.api_key = opts.api_key;
        http.cors_origin = opts.cors_origin;
        http.max_questions = opts.max_questions;
        http.batching = opts.batching;

        if (profile != dohnuts::model_profile::dohnuts) {
            if (profile == dohnuts::model_profile::kev && opts.head.empty()) {
                std::cerr << "error: kev requires --head\n";
                return 2;
            }
            dohnuts::side_options side_opts;
            side_opts.profile = profile;
            side_opts.model = opts.model;
            side_opts.head = opts.head;
            side_opts.config = opts.metadata;
            side_opts.threads = opts.threads;
            side_opts.gpu_layers = opts.gpu_layers;
            side_opts.device = opts.device;
            auto eng = std::make_shared<dohnuts::side_engine>(side_opts);
            http.backend = eng->backend_name() + " on " + eng->device_name();
            auto predict = [eng, raw = opts.raw](const json & requests) {
                return eng->predict(requests, raw);
            };
            if (opts.server) return dohnuts::serve_http(http, predict);

            std::istream * stream = &std::cin;
            std::ifstream file;
            if (!opts.input.empty()) {
                file.open(opts.input);
                if (!file) throw std::runtime_error("Cannot open " + opts.input);
                stream = &file;
            }
            std::string line;
            while (std::getline(*stream, line)) {
                if (line.empty()) continue;
                try {
                    json value = json::parse(line);
                    json requests = value.is_array() ? value : json::array({value});
                    json results = predict(requests);
                    std::cout << (value.is_array() ? results : results.at(0)).dump() << '\n';
                } catch (const std::exception & e) {
                    std::cout << json({{"error", {{"message", e.what()}}}}).dump() << '\n';
                }
            }
            return 0;
        }

        if (opts.head.empty()) { usage(); return 2; }
        dohnuts::engine_options engine_opts;
        engine_opts.model = opts.model;
        engine_opts.head = opts.head;
        engine_opts.mmproj = opts.mmproj;
        engine_opts.threads = opts.threads;
        engine_opts.gpu_layers = opts.gpu_layers;
        engine_opts.device = opts.device;
        dohnuts::engine eng(engine_opts);

        dohnuts::predictor predict(eng, temperatures_from(metadata));

        if (opts.server) {
            http.model = "dohnuts";
            http.backend = eng.backend_name() + " on " + eng.device_name();
            return dohnuts::serve_http(http, [&](const json & requests) {
                return predict.predict(requests, opts.raw);
            });
        }

        std::istream * stream = &std::cin;
        std::ifstream file;
        if (!opts.input.empty()) {
            file.open(opts.input);
            if (!file) throw std::runtime_error("Cannot open " + opts.input);
            stream = &file;
        }
        std::string line;
        while (std::getline(*stream, line)) {
            if (line.empty()) continue;
            try {
                json value = json::parse(line);
                json requests = value.is_array() ? value : json::array({value});
                json results = predict.predict(requests, opts.raw);
                std::cout << (value.is_array() ? results : results.at(0)).dump() << '\n';
            } catch (const std::exception & e) {
                std::cout << json({{"error", {{"message", e.what()}}}}).dump() << '\n';
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
