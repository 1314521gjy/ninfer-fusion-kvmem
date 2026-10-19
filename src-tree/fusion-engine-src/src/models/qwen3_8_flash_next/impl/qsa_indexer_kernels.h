#pragma once

#include "models/qwen3_8_flash_next/impl/qsa_indexer.h"
#include "models/qwen3_8_flash_next/impl/qsa_indexer_workspace.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::models::qwen3_8_flash_next::detail {

void flash_next_qsa_indexer_store_launch(const Tensor& projected, const Tensor& token_indices,
    const Tensor& mrope_positions, const Tensor& table_rows, const Tensor& source_state_slots,
    const Tensor& destination_state_slots, const Tensor& key_norm, QsaIndexerCacheView cache,
    cudaStream_t stream, bool aliased_recurrent_scan = false);

void flash_next_qsa_indexer_store_prefill_launch(const Tensor& projected,
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    std::int32_t source_state_slot, std::int32_t destination_state_slot,
    const Tensor& key_norm, QsaIndexerCacheView cache, cudaStream_t stream);
[[nodiscard]] std::size_t flash_next_qsa_indexer_sort_temp_bytes(std::int32_t maximum_blocks,
                                                                 std::int32_t batch);

void flash_next_qsa_indexer_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                   const Tensor& table_rows, const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, const Tensor& query_norm,
                                   const Tensor& key_norm, QsaIndexerCacheView cache,
                                   FlashNextQsaIndexerWorkspace& scratch,
                                   std::int32_t active_blocks, Tensor& selected_blocks,
                                   Tensor& selected_counts, cudaStream_t stream,
                                   bool aliased_recurrent_scan = false);

void flash_next_qsa_indexer_prefill_launch(
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    std::int32_t source_state_slot, std::int32_t destination_state_slot, const Tensor& query_norm,
    const Tensor& key_norm, QsaIndexerCacheView cache, FlashNextQsaIndexerWorkspace& scratch,
    std::int32_t maximum_blocks, std::int32_t first_token_index, Tensor& selected_blocks,
    Tensor& selected_counts, cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
