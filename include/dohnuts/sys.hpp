// System introspection shared by the runtime: chiefly the physical-core count
// used as the default CPU thread budget (SMT/hyperthreading siblings excluded).
#pragma once

#include <filesystem>
#include <string>

namespace dohnuts {

// Number of physical CPU cores visible to this process (SMT siblings counted
// once), clamped to the logical CPUs the process may actually use. Falls back
// to std::thread::hardware_concurrency() when the topology cannot be read.
int physical_core_count();

// llama.cpp takes UTF-8 paths; std::filesystem::path::c_str() is wchar_t on
// Windows, so convert explicitly.
inline std::string to_utf8(const std::filesystem::path & path) {
    const auto utf8 = path.u8string();
    return std::string(utf8.begin(), utf8.end());
}

} // namespace dohnuts
