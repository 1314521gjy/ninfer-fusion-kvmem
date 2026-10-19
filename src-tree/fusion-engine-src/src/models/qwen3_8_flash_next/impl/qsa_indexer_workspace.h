#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

struct FlashNextQsaIndexerWorkspace {
    Tensor projected;
    Tensor query;
    Tensor scores;
    Tensor sorted_scores;
    Tensor ids;
    Tensor sorted_ids;
    Tensor packed_keys;
    Tensor packed_selected;
    Tensor topk_ids;
    Tensor offsets;
    DeviceSpan sort_temp;
};

constexpr std::size_t kMaxScoreWorkspaceBytes = 64ULL * 1024ULL * 1024ULL;

inline std::int32_t flash_next_qsa_indexer_tile_size(std::int32_t maximum_blocks,
                                                     std::int32_t tokens) {
    if (maximum_blocks <= 0 || tokens <= 0) { return 1; }
    const std::int32_t max_rows = static_cast<std::int32_t>(
        kMaxScoreWorkspaceBytes / (static_cast<std::size_t>(maximum_blocks) * sizeof(float)));
    return std::max<std::int32_t>(1, std::min<std::int32_t>(tokens, max_rows));
}

template <class Arena>
FlashNextQsaIndexerWorkspace
allocate_flash_next_qsa_indexer_workspace(Arena& arena, std::int32_t maximum_blocks,
                                          std::int32_t tokens, std::int32_t tile_size,
                                          std::size_t sort_temp_bytes) {
    return {
        .projected       = arena.alloc(DType::BF16, {640, tokens}, 256),
        .query           = arena.alloc(DType::BF16, {128, 4, tokens}, 256),
        .scores          = arena.alloc(DType::FP32, {maximum_blocks, tile_size}, 256),
        .sorted_scores   = arena.alloc(DType::FP32, {maximum_blocks, tile_size}, 256),
        .ids             = arena.alloc(DType::I32, {maximum_blocks, tile_size}, 256),
        .sorted_ids      = arena.alloc(DType::I32, {maximum_blocks, tile_size}, 256),
        .packed_keys     = arena.alloc(DType::I64, {maximum_blocks, tile_size}, 256),
        .packed_selected = arena.alloc(DType::I64, {512, tile_size}, 256),
        .topk_ids        = arena.alloc(DType::I32, {512, tile_size}, 256),
        .offsets         = arena.alloc(DType::I32, {tile_size + 1}, 16),
        .sort_temp       = arena.alloc_bytes(sort_temp_bytes, 256),
    };
}

template <class Arena>
FlashNextQsaIndexerWorkspace
allocate_flash_next_qsa_indexer_workspace(Arena& arena, std::int32_t maximum_blocks,
                                          std::int32_t batch, std::size_t sort_temp_bytes) {
    return allocate_flash_next_qsa_indexer_workspace(arena, maximum_blocks, batch, batch,
                                                     sort_temp_bytes);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
