#pragma once

#include "core/arena.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t flash_next_ple_workspace_capacity_bytes(std::int32_t tokens);

void flash_next_ple_decode(const Tensor& hidden, const Tensor& gathered_embedding,
                           const PleWeights& weights, const Tensor& source_slots,
                           const Tensor& destination_slots, Tensor& convolution_states,
                           WorkspaceArena& workspace, Tensor& output, cudaStream_t stream,
                           bool aliased_recurrent_scan = false);

void flash_next_ple_prefill_chunk(const Tensor& hidden, const Tensor& gathered_embedding,
                                  const PleWeights& weights, std::int32_t source_slot,
                                  std::int32_t destination_slot, Tensor& convolution_states,
                                  WorkspaceArena& workspace, Tensor& output, cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
