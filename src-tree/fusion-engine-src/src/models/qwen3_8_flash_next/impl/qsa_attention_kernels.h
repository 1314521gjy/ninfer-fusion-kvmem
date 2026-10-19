#pragma once

#include "models/qwen3_8_flash_next/impl/qsa_attention.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention_workspace.h"

#include <cuda_runtime.h>

namespace ninfer::models::qwen3_8_flash_next::detail {

void flash_next_qsa_attention_store_launch(const Tensor& projected, const Tensor& token_indices,
    const Tensor& mrope_positions, const Tensor& table_rows, int table_row,
    const Tensor& key_norm, QsaAttentionCacheView cache, Tensor& key, Tensor& value,
    cudaStream_t stream);
void flash_next_qsa_attention_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                     const Tensor& table_rows, const Tensor& selected_blocks,
                                     const Tensor& selected_counts, const Tensor& query_norm,
                                     const Tensor& key_norm, QsaAttentionCacheView cache,
                                     FlashNextQsaAttentionWorkspace& scratch, WorkspaceArena& workspace,
                                     cudaStream_t stream);

void flash_next_qsa_attention_prefill_launch(
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    const Tensor& selected_blocks, const Tensor& selected_counts, const Tensor& query_norm,
    const Tensor& key_norm, QsaAttentionCacheView cache, FlashNextQsaAttentionWorkspace& scratch,
    cudaStream_t stream, bool use_mma = false);

} // namespace ninfer::models::qwen3_8_flash_next::detail
