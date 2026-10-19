#pragma once

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Decode-shape router: at most this many tokens take the single-launch fused path (one selection
// warp per token) or the legacy two-kernel path; larger T is prefill.
inline constexpr std::int32_t kFlashNextRouteDecodeMaxTokens = 8;

// Selects deterministic top-10 routes with top-10 renormalized probabilities (norm_topk_prob=true
// per transformers Qwen4ExpTextTopKRouter). scores carries the shared-expert gate in row 512.
void flash_next_route_scores(const Tensor& scores, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                             cudaStream_t stream);

// NINFER_FLASH_NEXT_ROUTE_LEGACY=1 routes decode shapes through the two-kernel path; read once
// on first use and cached. Unset or anything else selects the fused kernel.
[[nodiscard]] bool flash_next_route_legacy_enabled();

// Decode (T <= kFlashNextRouteDecodeMaxTokens) paths. Both write the FP32 score workspace, ids,
// alphas and the shared-expert gate bit-identically; the fused kernel is one launch instead of
// two and the legacy pair is the reference it is gated against.
void flash_next_route_decode_fused(const Tensor& input, const Weight& router,
                                   const Weight& shared_gate, Tensor& score_workspace,
                                   Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                                   cudaStream_t stream);
void flash_next_route_decode_legacy(const Tensor& input, const Weight& router,
                                    const Weight& shared_gate, Tensor& score_workspace,
                                    Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                                    cudaStream_t stream);

// Full router: projection (decode: fused or legacy per the env var; prefill: MMA or tiled) plus
// selection.
void flash_next_route(const Tensor& input, const Weight& router, const Weight& shared_gate,
                      Tensor& score_workspace, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                      cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
