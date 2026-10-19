#include "models/qwen3_8_flash_next/impl/load/quantize_output_head.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

constexpr int kThreads = 256;

// One 64 MiB H2D chunk, the same working-set size the artifact materializer stages with
// (materializer.cpp kSlotBytes). Large enough that the copy runs at PCIe speed, small enough
// that quantizing a 1.27 GB weight from the file mapping costs 64 MiB of transient device memory.
constexpr std::size_t kStagingBudgetBytes = 64ULL << 20;

// One CTA per row. `columns` is a runtime argument and the row is walked with a grid-stride loop,
// so the same kernel is correct for every Flash-Next weight width (K in {320, 640, 2560, 10240};
// K=320 has 160 pairs, fewer than the 256 threads). The row is read twice - once for the amax,
// once for the encode - rather than cached in registers, because a register cache has to be sized
// at compile time; the second pass costs ~1 ms over the 1.27 GB head, once, at load.
__global__ __launch_bounds__(kThreads, 2) void quantize_row_kernel(
    const __nv_bfloat16* __restrict__ input, std::uint8_t* __restrict__ codes,
    float* __restrict__ scales, int columns) {
    constexpr int warps = kThreads / 32;
    __shared__ float warp_maxima[warps];
    __shared__ float row_scale;

    const int row   = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const int lane  = tid & 31;
    const int warp  = tid >> 5;
    const int pairs = columns >> 1;
    const auto* input_pairs =
        reinterpret_cast<const std::uint32_t*>(input + static_cast<std::int64_t>(row) * columns);
    auto* output_pairs =
        reinterpret_cast<std::uint16_t*>(codes + static_cast<std::int64_t>(row) * columns);

    float maximum = 0.0F;
    for (int pair = tid; pair < pairs; pair += kThreads) {
        const float2 value = ops::bf16x2_bits_to_float2(input_pairs[pair]);
        maximum            = fmaxf(maximum, fabsf(value.x));
        maximum            = fmaxf(maximum, fabsf(value.y));
    }
    maximum = ops::warp_max(maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < warps ? warp_maxima[lane] : 0.0F;
        maximum = ops::warp_max(maximum);
        if (lane == 0) { row_scale = maximum > 0.0F ? maximum / 448.0F : 0.0F; }
    }
    __syncthreads();

    const float scale   = row_scale;
    const float inverse = scale > 0.0F ? 1.0F / scale : 0.0F;
    for (int pair = tid; pair < pairs; pair += kThreads) {
        const float2 value  = ops::bf16x2_bits_to_float2(input_pairs[pair]);
        const float2 scaled = make_float2(value.x * inverse, value.y * inverse);
        output_pairs[pair]  = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) { scales[row] = scale; }
}

__global__ void gather_rows_kernel(
    const __nv_bfloat16* __restrict__ src,
    const std::int32_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ dst,
    int rows, int cols) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int src_token = token_ids[row];
    const auto* src_row = reinterpret_cast<const uint4*>(src + static_cast<std::int64_t>(src_token) * cols);
    auto* dst_row = reinterpret_cast<uint4*>(dst + static_cast<std::int64_t>(row) * cols);
    const int vectors = cols / 8; // 2560 / 8 = 320 uint4s
    for (int i = static_cast<int>(threadIdx.x); i < vectors; i += static_cast<int>(blockDim.x)) {
        dst_row[i] = src_row[i];
    }
}

Weight make_fp8_view_sized(void* payload, std::size_t payload_bytes, std::int32_t rows, std::int32_t cols) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * cols;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    auto* bytes                      = static_cast<std::byte*>(payload);
    const std::int64_t scale_stride  = static_cast<std::int64_t>(rows) * 4;
    Weight out{};
    out.payload         = payload;
    out.payload_bytes   = payload_bytes;
    out.qdata           = payload;
    out.scales          = bytes + scale_offset;
    out.qtype           = QType::FP8_E4M3FN_ROW_BF16;
    out.layout          = QuantLayout::RowScale;
    out.scale_dtype     = DType::FP32;
    out.n               = rows;
    out.k               = cols;
    out.group           = cols;
    out.group_size      = static_cast<std::uint32_t>(cols);
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = cols;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = cols;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = 4;
    out.scale_nb[1]     = scale_stride;
    out.scale_nb[2]     = scale_stride;
    out.scale_nb[3]     = scale_stride;
    return out;
}

Weight make_bf16_view(void* payload, std::size_t payload_bytes, std::int32_t rows,
                      std::int32_t cols) {
    Weight out{};
    out.payload         = payload;
    out.payload_bytes   = payload_bytes;
    out.qdata           = payload;
    out.qtype           = QType::BF16;
    out.layout          = QuantLayout::Contiguous;
    out.n               = rows;
    out.k               = cols;
    out.group           = cols;
    out.group_size      = static_cast<std::uint32_t>(cols);
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = cols;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = cols;
    return out;
}

// Rows per staged H2D chunk for a host source, clamped to at least one row.
std::int32_t staging_rows(std::int32_t rows, std::size_t row_bytes) {
    const std::size_t budget_rows = std::max<std::size_t>(1, kStagingBudgetBytes / row_bytes);
    return static_cast<std::int32_t>(
        std::min<std::size_t>(static_cast<std::size_t>(rows), budget_rows));
}

} // namespace

std::size_t flash_next_fp8_output_head_payload_bytes() {
    return flash_next_fp8_head_payload_bytes(kOutputHeadRows, kOutputHeadColumns);
}

std::size_t flash_next_fp8_head_payload_bytes(std::int32_t rows, std::int32_t cols) {
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * cols;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    return static_cast<std::size_t>(scale_offset + static_cast<std::uint64_t>(rows) * 4);
}

void quantize_bf16_output_head_to_fp8_e4m3_row_f32s(const Weight& bf16_head, DeviceBuffer& payload,
                                                    Weight& fp8_head, cudaStream_t stream) {
    quantize_bf16_head_to_fp8_e4m3_row_f32s(bf16_head, payload, fp8_head, kOutputHeadRows,
                                            kOutputHeadColumns, stream);
}

void quantize_bf16_head_to_fp8_e4m3_row_f32s(const Weight& bf16_head, DeviceBuffer& payload,
                                             Weight& fp8_head, std::int32_t rows,
                                             std::int32_t cols, cudaStream_t stream) {
    if (bf16_head.qtype != QType::BF16 || bf16_head.n != rows ||
        bf16_head.k != cols || bf16_head.qdata == nullptr) {
        throw std::invalid_argument("Flash-Next FP8 head quantize received an invalid BF16 view");
    }
    quantize_bf16_rows_to_fp8_e4m3_row_f32s(bf16_head.qdata, payload, fp8_head, rows, cols, stream);
}

void quantize_bf16_rows_to_fp8_e4m3_row_f32s(const void* bf16_rows, DeviceBuffer& payload,
                                             Weight& fp8_out, std::int32_t rows, std::int32_t cols,
                                             cudaStream_t stream) {
    if (bf16_rows == nullptr || rows <= 0 || cols <= 0 || (cols % 2) != 0) {
        throw std::invalid_argument("Flash-Next FP8 quantize received invalid dimensions");
    }
    const std::size_t bytes = flash_next_fp8_head_payload_bytes(rows, cols);
    if (payload.p == nullptr || payload.bytes < bytes) {
        throw std::invalid_argument("Flash-Next FP8 head payload is too small");
    }
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * cols;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    auto* code_ptr                   = static_cast<std::uint8_t*>(payload.p);
    auto* scale_ptr = reinterpret_cast<float*>(static_cast<std::byte*>(payload.p) + scale_offset);

    cudaPointerAttributes attributes{};
    const cudaError_t pointer_status = cudaPointerGetAttributes(&attributes, bf16_rows);
    const bool is_device_source =
        (pointer_status == cudaSuccess && attributes.type == cudaMemoryTypeDevice);
    if (pointer_status != cudaSuccess) {
        (void)cudaGetLastError(); // querying an unregistered host pointer must not poison the stream
    }

    if (is_device_source) {
        quantize_row_kernel<<<rows, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(bf16_rows), code_ptr, scale_ptr, cols);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Host source (the artifact's file mapping): stage bounded row chunks so the BF16 weight
        // never becomes device-resident. This is what makes the FP8 flags save VRAM - binding the
        // BF16 tensor on the device and quantizing from it costs the FP8 payload on top of it.
        const std::size_t row_bytes       = static_cast<std::size_t>(cols) * sizeof(std::uint16_t);
        const std::int32_t rows_per_chunk = staging_rows(rows, row_bytes);
        DeviceBuffer stage(static_cast<std::size_t>(rows_per_chunk) * row_bytes);
        const auto* host_rows = static_cast<const std::byte*>(bf16_rows);
        for (std::int32_t start = 0; start < rows; start += rows_per_chunk) {
            const std::int32_t count = std::min(rows_per_chunk, rows - start);
            // Same stream as the kernel: the next copy is ordered after the previous chunk's
            // kernel, so the staging buffer is never overwritten while it is still being read.
            CUDA_CHECK(cudaMemcpyAsync(stage.p,
                                       host_rows + static_cast<std::size_t>(start) * row_bytes,
                                       static_cast<std::size_t>(count) * row_bytes,
                                       cudaMemcpyHostToDevice, stream));
            quantize_row_kernel<<<count, kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(stage.p),
                code_ptr + static_cast<std::int64_t>(start) * cols, scale_ptr + start, cols);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream)); // `stage` is freed as this scope exits
    }
    fp8_out = make_fp8_view_sized(payload.p, payload.bytes, rows, cols);
}

void gather_head_rows_bf16(const Weight& src_head, const std::int32_t* token_ids,
                           std::int32_t rows, std::int32_t cols,
                           DeviceBuffer& dst_payload, Weight& dst_head, cudaStream_t stream) {
    if (src_head.qtype != QType::BF16 || src_head.qdata == nullptr || token_ids == nullptr) {
        throw std::invalid_argument("gather_head_rows_bf16 received invalid inputs");
    }
    const std::size_t required_bytes = static_cast<std::size_t>(rows) * cols * sizeof(std::uint16_t);
    if (dst_payload.p == nullptr || dst_payload.bytes < required_bytes) {
        throw std::invalid_argument("gather_head_rows_bf16 dst_payload is too small");
    }
    gather_rows_kernel<<<rows, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src_head.qdata), token_ids,
        static_cast<__nv_bfloat16*>(dst_payload.p), rows, cols);
    CUDA_CHECK(cudaGetLastError());

    dst_head = make_bf16_view(dst_payload.p, dst_payload.bytes, rows, cols);
}

void gather_head_rows_bf16_from_host(std::span<const std::byte> src_rows,
                                     const std::int32_t* token_ids, std::int32_t rows,
                                     std::int32_t src_rows_count, std::int32_t cols,
                                     DeviceBuffer& dst_payload, Weight& dst_head,
                                     cudaStream_t stream) {
    const std::size_t row_bytes = static_cast<std::size_t>(cols) * sizeof(std::uint16_t);
    if (token_ids == nullptr || rows <= 0 || cols <= 0 || src_rows_count <= 0 ||
        src_rows.size() < static_cast<std::size_t>(src_rows_count) * row_bytes) {
        throw std::invalid_argument("gather_head_rows_bf16_from_host received invalid inputs");
    }
    if (dst_payload.p == nullptr ||
        dst_payload.bytes < static_cast<std::size_t>(rows) * row_bytes) {
        throw std::invalid_argument("gather_head_rows_bf16_from_host dst_payload is too small");
    }

    const std::int32_t rows_per_chunk = staging_rows(rows, row_bytes);
    std::vector<std::byte> staging(static_cast<std::size_t>(rows_per_chunk) * row_bytes);
    for (std::int32_t start = 0; start < rows; start += rows_per_chunk) {
        const std::int32_t count = std::min(rows_per_chunk, rows - start);
        for (std::int32_t index = 0; index < count; ++index) {
            const std::int32_t token = token_ids[start + index];
            if (token < 0 || token >= src_rows_count) {
                throw std::out_of_range("gather_head_rows_bf16_from_host token id is out of range");
            }
            std::memcpy(staging.data() + static_cast<std::size_t>(index) * row_bytes,
                        src_rows.data() + static_cast<std::size_t>(token) * row_bytes, row_bytes);
        }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(dst_payload.p) +
                                       static_cast<std::size_t>(start) * row_bytes,
                                   staging.data(), static_cast<std::size_t>(count) * row_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); // `staging` is refilled by the next chunk
    }

    dst_head = make_bf16_view(dst_payload.p, dst_payload.bytes, rows, cols);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
