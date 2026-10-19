#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr int kHyperDownSplitK = 4;

struct FlashNextHyperWorkspace {
    Tensor normalized;
    Tensor low_rank;
    Tensor injection;
    Tensor up_gemm;
    Tensor down_split;
};

template <class Arena>
FlashNextHyperWorkspace allocate_flash_next_hyper_workspace(Arena& arena, std::int32_t tokens) {
    return {
        .normalized = arena.alloc(DType::BF16, {10'240, tokens}, 256),
        .low_rank   = arena.alloc(DType::BF16, {320, tokens}, 256),
        .injection  = arena.alloc(DType::FP32, {4, tokens}, 16),
        .up_gemm    = arena.alloc(DType::BF16, {10'240, tokens}, 256),
        .down_split = arena.alloc(DType::FP32, {320, tokens * kHyperDownSplitK}, 256),
    };
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
