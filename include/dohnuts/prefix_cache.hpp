// LRU cache of decoded state prefixes, keyed by the exact prefix token ids.
//
// Hybrid (Qwen3.5) memory keeps recurrent state that cannot be truncated back
// to a shorter prefix, so reuse goes through whole-sequence state checkpoints
// rather than dropping KV cells. Shared by the core engine and the side runner.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <utility>
#include <vector>

namespace dohnuts {

// Prefixes shorter than this are cheaper to decode again than to checkpoint.
inline constexpr size_t PREFIX_MIN_TOKENS = 16;

// Total host RAM the cache may hold, in bytes.
inline constexpr size_t PREFIX_CACHE_LIMIT = 256u * 1024 * 1024;

class prefix_cache {
public:
    explicit prefix_cache(size_t limit = PREFIX_CACHE_LIMIT) : limit(limit) {}

    // Returns the checkpoint for the exact prefix tokens, or nullptr. A hit is
    // moved to the front of the LRU.
    const std::vector<uint8_t> * get(const std::vector<int32_t> & key) {
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->ids == key) {
                entries.splice(entries.begin(), entries, it);
                return &entries.front().state;
            }
        }
        return nullptr;
    }

    // Drops the front entry after a failed restore.
    void drop_front() {
        if (entries.empty()) return;
        bytes -= entries.front().state.size();
        entries.pop_front();
    }

    // Stores a checkpoint, evicting least-recently-used entries to stay under
    // the limit. Empty or oversized checkpoints are ignored.
    void put(std::vector<int32_t> key, std::vector<uint8_t> state) {
        if (state.empty() || state.size() > limit) return;
        bytes += state.size();
        entries.push_front({std::move(key), std::move(state)});
        while (bytes > limit && !entries.empty()) {
            bytes -= entries.back().state.size();
            entries.pop_back();
        }
    }

    size_t size_in_bytes() const { return bytes; }
    size_t count() const { return entries.size(); }

private:
    struct entry {
        std::vector<int32_t> ids;
        std::vector<uint8_t> state;
    };
    std::list<entry> entries;
    size_t bytes = 0;
    size_t limit;
};

} // namespace dohnuts
