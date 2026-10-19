#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "models/qwen3_8_flash_next/impl/hyper_workspace.h"

namespace ninfer::models::qwen3_8_flash_next::detail {

struct FlashNextMtpStemWorkspace {
    Tensor embedding_norm;
    Tensor embedding_projection;
    Tensor hidden_norm;
    Tensor hidden_projection;
};

template <class Arena>
FlashNextMtpStemWorkspace allocate_flash_next_mtp_stem_workspace(Arena& arena, int tokens) {
    return {arena.alloc(DType::BF16, {2'560, tokens}, 256),
            arena.alloc(DType::BF16, {2'560, tokens}, 256),
            arena.alloc(DType::BF16, {10'240, tokens}, 256),
            arena.alloc(DType::BF16, {10'240, tokens}, 256)};
}

struct FlashNextMtpTeacherWorkspace {
    Tensor previous_hidden, indices, positions, hyper_hidden, attention_input;
    FlashNextHyperWorkspace hyper;
};

template <class Arena>
FlashNextMtpTeacherWorkspace allocate_flash_next_mtp_teacher_workspace(Arena& arena, int tokens) {
    return {arena.alloc(DType::BF16, {10'240, tokens}, 256),
            arena.alloc(DType::I32, {tokens}, 256),
            arena.alloc(DType::I32, {tokens, 3}, 256),
            arena.alloc(DType::BF16, {10'240, tokens}, 256),
            arena.alloc(DType::BF16, {2'560, tokens}, 256),
            allocate_flash_next_hyper_workspace(arena, tokens)};
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
