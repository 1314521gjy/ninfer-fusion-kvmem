#pragma once

#include "core/arena.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/text_decode_state.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>

namespace ninfer::models::qwen3_8_flash_next::detail {

struct FlashNextDecodeStateSink {
    std::function<void(std::string_view name, const Tensor& device_tensor)> on_state;
};

[[nodiscard]] std::size_t
flash_next_text_decode_workspace_capacity_bytes(std::int32_t maximum_blocks, std::int32_t batch, bool mtp = false);

[[nodiscard]] std::size_t
flash_next_text_prefill_workspace_capacity_bytes(std::int32_t maximum_blocks, std::int32_t tokens, bool mtp = false);

void flash_next_text_decode_core(const TextModelView& model, const Tensor& embedding,
                                 const Tensor& token_indices, const Tensor& mrope_positions,
                                 const Tensor& table_rows, const Tensor& source_slots,
                                 const Tensor& destination_slots,
                                 const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                 std::int32_t active_blocks, FlashNextDecodeStateView state,
                                 WorkspaceArena& workspace, Tensor& final_hidden, Tensor& logits,
                                 cudaStream_t stream,
                                 const FlashNextDecodeStateSink* sink = nullptr,
                                 Tensor* out_hyper_hidden             = nullptr,
                                 bool aliased_recurrent_scan          = false,
                                 const Tensor* mtp_token_ids          = nullptr);

void flash_next_text_decode(const TextModelView& model, const Tensor& token_ids,
                            const Tensor& token_indices, const Tensor& mrope_positions,
                            const Tensor& table_rows, const Tensor& source_slots,
                            const Tensor& destination_slots, const Tensor& gathered_ple_embedding,
                            std::int32_t maximum_blocks, std::int32_t active_blocks,
                            FlashNextDecodeStateView state, WorkspaceArena& workspace,
                            Tensor& final_hidden, Tensor& logits, cudaStream_t stream,
                            const FlashNextDecodeStateSink* sink = nullptr);

void flash_next_text_prefill_chunk(const TextModelView& model, const Tensor& embedding,
                                   const Tensor& token_indices, const Tensor& mrope_positions,
                                   std::int32_t table_row, std::int32_t source_slot,
                                   std::int32_t destination_slot,
                                   const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                   std::int32_t first_token_index, FlashNextDecodeStateView state,
                                   WorkspaceArena& workspace, Tensor& final_hidden, Tensor& logits,
                                   cudaStream_t stream,
                                   const FlashNextDecodeStateSink* sink = nullptr,
                                   bool use_qsa_prefill_mma            = false,
                                   Tensor* out_hyper_hidden            = nullptr,
                                   const Tensor* mtp_token_ids         = nullptr);

} // namespace ninfer::models::qwen3_8_flash_next::detail
