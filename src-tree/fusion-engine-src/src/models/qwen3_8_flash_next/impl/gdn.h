#pragma once

#include "core/arena.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t flash_next_gdn_workspace_capacity_bytes(std::int32_t min_batch,
                                                                  std::int32_t max_batch);

// One compact decode round over B independent request slots. Convolution and FP32 DeltaNet state
// both transition from source slots to distinct destination slots transactionally.
void flash_next_gdn_decode(const Tensor& input, const GdnWeights& weights,
                           const Tensor& source_slots, const Tensor& destination_slots,
                           Tensor& convolution_states, Tensor& ssm_states,
                           WorkspaceArena& workspace, Tensor& output, cudaStream_t stream,
                           bool aliased_recurrent_scan = false);

// T-wide prefill chunk execution for one lane.
void flash_next_gdn_prefill_chunk(const Tensor& input, const GdnWeights& weights,
                                  std::int32_t source_slot, std::int32_t destination_slot,
                                  Tensor& convolution_states, Tensor& ssm_states,
                                  WorkspaceArena& workspace, Tensor& output, cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
