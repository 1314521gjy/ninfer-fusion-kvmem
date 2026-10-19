#pragma once

#include "core/arena.h"
#include "models/qwen3_8_flash_next/impl/hyper_workspace.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t flash_next_hyper_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

// Normalizes the four streams, evaluates the low-rank input mixer, and retains the four
// block-output injection gates in scratch for the matching inject call.
void flash_next_hyper_prepare(const Tensor& hidden, const HyperConnectionWeights& weights,
                              FlashNextHyperWorkspace& scratch, Tensor& block_input,
                              cudaStream_t stream);

void flash_next_hyper_inject(const Tensor& block_output, const Tensor& injection, Tensor& hidden,
                             cudaStream_t stream);

// Final/MTP mixer form: identical normalized low-rank mixer without an injection projection.
void flash_next_hyper_mix(const Tensor& hidden, const HyperMixerWeights& weights,
                          FlashNextHyperWorkspace& scratch, Tensor& block_input,
                          cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
