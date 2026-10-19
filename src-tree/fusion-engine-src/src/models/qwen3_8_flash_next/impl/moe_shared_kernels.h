#pragma once

#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/moe_workspace.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Prefill T>8 shared-expert / router / grouping MMA path.
// NINFER_FLASH_NEXT_MOE_SHARED_MMA=0 restores the scalar kernels. Unset or 1 enables MMA.
[[nodiscard]] bool flash_next_moe_shared_mma_enabled();

void flash_next_route_projection_mma(const Tensor& input, const Weight& router,
                                     const Weight& shared_gate, const Tensor& score_workspace,
                                     cudaStream_t stream);

void flash_next_moe_prefill_build_groups_mma(
    const Tensor& ids, const Tensor& expert_counts, const Tensor& expert_offsets,
    const Tensor& active_experts, const Tensor& active_count, const Tensor& grouped_tokens,
    const Tensor& grouped_paths, const Tensor& grouped_experts, const Tensor& token_to_pos,
    const Tensor& task_counter, int tokens, cudaStream_t stream);

void flash_next_moe_prefill_shared_gate_up_mma(const Tensor& input, const Weight& shared_gate,
                                               const Weight& shared_up, const Tensor& activations,
                                               const Tensor& shared_gemm, int tokens,
                                               cudaStream_t stream);

void flash_next_moe_prefill_shared_down_mma(const Weight& shared_down, const Tensor& shared_gemm,
                                            const Tensor& shared_scale, const Tensor& output,
                                            int tokens, cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
