#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

struct FlashNextGdnWorkspace {
    Tensor projected;
    Tensor conv_in;
    Tensor conv_output;
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor z;
    Tensor recurrent_output;
    Tensor gated_output;
    Tensor g;
    Tensor beta;
};

template <class Arena>
FlashNextGdnWorkspace allocate_flash_next_gdn_workspace(Arena& arena, std::int32_t tokens) {
    return {
        .projected        = arena.alloc(DType::BF16, {16'384, tokens}, 256),
        .conv_in          = arena.alloc(DType::BF16, {10'240, tokens}, 256),
        .conv_output      = arena.alloc(DType::BF16, {10'240, tokens}, 256),
        .query            = arena.alloc(DType::BF16, {2'048, tokens}, 256),
        .key              = arena.alloc(DType::BF16, {2'048, tokens}, 256),
        .value            = arena.alloc(DType::BF16, {6'144, tokens}, 256),
        .z                = arena.alloc(DType::BF16, {6'144, tokens}, 256),
        .recurrent_output = arena.alloc(DType::BF16, {6'144, tokens}, 256),
        .gated_output     = arena.alloc(DType::BF16, {6'144, tokens}, 256),
        .g                = arena.alloc(DType::FP32, {48, tokens}, 16),
        .beta             = arena.alloc(DType::FP32, {48, tokens}, 16),
    };
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
