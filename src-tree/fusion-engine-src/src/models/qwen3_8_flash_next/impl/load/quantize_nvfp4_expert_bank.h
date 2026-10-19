#pragma once

#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

[[nodiscard]] std::size_t
flash_next_nvfp4_expert_bank_payload_bytes(std::int32_t experts, std::int32_t rows,
                                           std::int32_t columns);

void quantize_bf16_expert_bank_to_nvfp4(const void* bf16_data, void* nvfp4_payload,
                                        std::int32_t experts, std::int32_t rows,
                                        std::int32_t columns, cudaStream_t stream);

void populate_synthetic_bf16_bank(void* ptr, std::size_t elements, float scale,
                                  cudaStream_t stream);

void populate_constant_bf16_bank(void* ptr, std::size_t elements, float val,
                                 cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
