#pragma once

#include "core/arena.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

// Exact Qwen4-exp 512-expert/top-10 MoE leaf with top-10 renormalized probabilities
// (norm_topk_prob=true per transformers Qwen4ExpTextTopKRouter); the independent shared expert is sigmoid-gated.
void flash_next_moe(const Tensor& input, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream);

void flash_next_moe_bf16(const Tensor& input, const MoeBf16Weights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
