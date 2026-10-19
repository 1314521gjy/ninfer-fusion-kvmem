#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

void flash_next_ple_launch(const Tensor& hidden, const Tensor& projected_key,
                           const Tensor& projected_value, const Tensor& query_norm,
                           const Tensor& key_norm, const Tensor& conv_norm,
                           const Tensor& convolution, const Tensor& source_slots,
                           const Tensor& destination_slots, Tensor& convolution_states,
                           Tensor& gated, Tensor& normalized_gated, Tensor& output, int state_slots,
                           int batch, cudaStream_t stream, bool aliased_recurrent_scan = false);

void flash_next_ple_chunk_launch(const Tensor& hidden, const Tensor& projected_key,
                                 const Tensor& projected_value, const Tensor& query_norm,
                                 const Tensor& key_norm, const Tensor& conv_norm,
                                 const Tensor& convolution, std::int32_t source_slot,
                                 std::int32_t destination_slot, Tensor& convolution_states,
                                 Tensor& gated, Tensor& normalized_gated, Tensor& output,
                                 int state_slots, int tokens, cudaStream_t stream);

void flash_next_ple_dequant_launch(const void* compressed, Tensor& output, int tokens,
                                   cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
