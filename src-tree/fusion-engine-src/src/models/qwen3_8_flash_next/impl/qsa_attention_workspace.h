#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

struct FlashNextQsaAttentionWorkspace {
    Tensor projected;
    Tensor query;
    Tensor gate;
    Tensor key;
    Tensor value;
    Tensor attended;
    Tensor gated;
};

template <class Arena>
FlashNextQsaAttentionWorkspace allocate_flash_next_qsa_attention_workspace(Arena& arena,
                                                                           std::int32_t tokens) {
    return {
        .projected = arena.alloc(DType::BF16, {13'312, tokens}, 256),
        .query     = arena.alloc(DType::BF16, {256, 24, tokens}, 256),
        .gate      = arena.alloc(DType::BF16, {6'144, tokens}, 256),
        .key       = arena.alloc(DType::BF16, {256, 2, tokens}, 256),
        .value     = arena.alloc(DType::BF16, {256, 2, tokens}, 256),
        .attended  = arena.alloc(DType::BF16, {256, 24, tokens}, 256),
        .gated     = arena.alloc(DType::BF16, {6'144, tokens}, 256),
    };
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
