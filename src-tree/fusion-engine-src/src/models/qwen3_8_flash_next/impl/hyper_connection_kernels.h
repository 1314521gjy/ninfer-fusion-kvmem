#pragma once

#include "core/tensor.h"
#include "models/qwen3_8_flash_next/impl/hyper_workspace.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Kernel selection for the decode route (T <= 8) of the norm + low-rank/injection stage.
//   Fused : one hyper_norm_low_rank_fused_kernel launch; every CTA recomputes the
//           group-norm statistics. Bitwise-identical to Legacy (gated by the hyper test).
//   Legacy: group_norm_vectorized_kernel followed by low_rank_and_injection_kernel.
// The prefill route (T > 8) ignores the selector.
enum class FlashNextHyperDecodeRoute : std::uint8_t { Fused, Legacy };

// Process-wide route, resolved once on first use and cached:
// NINFER_FLASH_NEXT_HYPER_LEGACY=1 selects Legacy, anything else selects Fused.
[[nodiscard]] FlashNextHyperDecodeRoute flash_next_hyper_decode_route();

// Production entry points: route = flash_next_hyper_decode_route().
void flash_next_hyper_prepare_launch(const Tensor& hidden, const HyperConnectionWeights& weights,
                                     FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                     cudaStream_t stream);
void flash_next_hyper_mix_launch(const Tensor& hidden, const HyperMixerWeights& weights,
                                 FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                 cudaStream_t stream);

// Route-explicit forms for the bit-exact gate, which runs both routes in one process.
void flash_next_hyper_prepare_route_launch(const Tensor& hidden,
                                           const HyperConnectionWeights& weights,
                                           FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                           cudaStream_t stream, FlashNextHyperDecodeRoute route);
void flash_next_hyper_mix_route_launch(const Tensor& hidden, const HyperMixerWeights& weights,
                                       FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                       cudaStream_t stream, FlashNextHyperDecodeRoute route);

void flash_next_hyper_inject_launch(const Tensor& block_output, const Tensor& injection,
                                    Tensor& hidden, cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
