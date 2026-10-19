#pragma once

#include "core/arena.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention.h"
#include "models/qwen3_8_flash_next/impl/text_decode.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Norm over all four streams, shared per-stream projection, embedding residual.
void flash_next_mtp_stem(const MtpModelView& mtp, const Tensor& embedding,
                         const Tensor& backbone_hidden, WorkspaceArena& workspace,
                         Tensor& hyper_hidden, cudaStream_t stream,
                         const FlashNextDecodeStateSink* sink = nullptr);

[[nodiscard]] std::size_t flash_next_mtp_teacher_workspace_capacity_bytes(int tokens);

// Extend the MTP KV/indexer cache with target-conditioned rows. The last target
// hidden is saved until the next input token is known; no MTP MoE/head is needed.
void flash_next_mtp_teacher_extend(const MtpModelView& mtp, const Tensor& embedding,
    const Tensor& target_hidden, const Tensor& token_indices, const Tensor& positions,
    const Tensor& table_rows, const Tensor& source_slots, const Tensor& destination_slots,
    int table_row, int source_slot, int destination_slot, int first_token_index,
    bool prefill, bool aliased_scan, FlashNextDecodeStateView state,
    WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t flash_next_mtp_workspace_capacity_bytes(std::int32_t maximum_blocks,
                                                                 std::int32_t batch);

void flash_next_mtp_step(const TextModelView& model, const Tensor& input_embedding,
                         const Tensor& backbone_hyper_hidden, const Tensor& token_indices,
                         const Tensor& mrope_positions, const Tensor& table_rows,
                         const Tensor& source_slots, const Tensor& destination_slots,
                         QsaIndexerCacheView indexer_cache, QsaAttentionCacheView mtp_cache,
                         std::int32_t maximum_blocks, std::int32_t active_blocks,
                         WorkspaceArena& workspace,
                         Tensor& draft_logits, Tensor& draft_tokens, cudaStream_t stream,
                         const FlashNextDecodeStateSink* sink = nullptr,
                         Tensor* out_hyper_hidden             = nullptr);

} // namespace ninfer::models::qwen3_8_flash_next::detail
