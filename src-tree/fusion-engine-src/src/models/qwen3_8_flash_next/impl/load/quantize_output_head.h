#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr std::int32_t kOutputHeadRows    = 248'320;
inline constexpr std::int32_t kOutputHeadColumns = 2'560;

// E4M3FN max finite is 448. Scale is per-row amax/448. Codes use round-to-nearest-even
// saturate-to-finite conversion (__nv_cvt_float2_to_fp8x2, __NV_SATFINITE, __NV_E4M3).
[[nodiscard]] std::size_t flash_next_fp8_output_head_payload_bytes();
[[nodiscard]] std::size_t flash_next_fp8_head_payload_bytes(std::int32_t rows,
                                                            std::int32_t cols = kOutputHeadColumns);

// Quantizes a [rows, cols] BF16 weight into `payload`. `bf16_rows` may point at device memory or
// at host memory (the artifact's file mapping): a host source is uploaded one bounded chunk at a
// time, so the BF16 copy never has to be resident next to the FP8 one. `cols` only has to be even
// - the kernel is a grid-stride over the row, valid for every Flash-Next weight width.
void quantize_bf16_rows_to_fp8_e4m3_row_f32s(const void* bf16_rows, DeviceBuffer& payload,
                                             Weight& fp8_out, std::int32_t rows, std::int32_t cols,
                                             cudaStream_t stream);

void quantize_bf16_output_head_to_fp8_e4m3_row_f32s(const Weight& bf16_head, DeviceBuffer& payload,
                                                    Weight& fp8_head, cudaStream_t stream);

void quantize_bf16_head_to_fp8_e4m3_row_f32s(const Weight& bf16_head, DeviceBuffer& payload,
                                             Weight& fp8_head, std::int32_t rows,
                                             std::int32_t cols, cudaStream_t stream);

void gather_head_rows_bf16(const Weight& src_head, const std::int32_t* token_ids,
                           std::int32_t rows, std::int32_t cols,
                           DeviceBuffer& dst_payload, Weight& dst_head, cudaStream_t stream);

// Host-source variant of gather_head_rows_bf16: `src_rows` is the mapped BF16 head and
// `token_ids` is a host array of `rows` row indices into it. Rows are gathered through a bounded
// host staging buffer, so the shortlist can be built without the full BF16 head on the device.
void gather_head_rows_bf16_from_host(std::span<const std::byte> src_rows,
                                     const std::int32_t* token_ids, std::int32_t rows,
                                     std::int32_t src_rows_count, std::int32_t cols,
                                     DeviceBuffer& dst_payload, Weight& dst_head,
                                     cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
