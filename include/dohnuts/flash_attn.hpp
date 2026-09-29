// Attention dispatch mode, shared by the core engine and the side runner.
#pragma once

#include "llama.h"

namespace dohnuts {

// Mirrors llama.cpp's llama_flash_attn_type. `automatic` lets the backend
// decide (llama.cpp's resolve_fused_ops probe enables Flash Attention when the
// active device supports it); `enabled` and `disabled` force it either way.
enum class flash_attn_mode {
    automatic,
    enabled,
    disabled,
};

inline llama_flash_attn_type to_llama_flash_attn(flash_attn_mode mode) {
    switch (mode) {
    case flash_attn_mode::enabled:  return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    case flash_attn_mode::disabled: return LLAMA_FLASH_ATTN_TYPE_DISABLED;
    default:                        return LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
}

} // namespace dohnuts
