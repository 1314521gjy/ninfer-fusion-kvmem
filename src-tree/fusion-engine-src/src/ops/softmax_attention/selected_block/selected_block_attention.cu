#include "ninfer/ops/selected_block_attention.h"
#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops {
namespace {
__device__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ float to_float(__nv_fp8_e4m3 x) { return static_cast<float>(x); }
__device__ int selected_token(int ordinal, int count, int complete, const int* selected) {
    return ordinal < count * 4 ? selected[ordinal / 4] * 4 + ordinal % 4
                               : complete * 4 + ordinal - count * 4;
}
#include "prefill.cuh"
constexpr int kWarps = 4;
constexpr int kThreads = 32 * kWarps;
constexpr int kPartialStride = 288; // 256 values and two statistics; align rows to 128 bytes.

constexpr int kPartitions = 16;

__device__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, offset);
    }
    return value;
}

template <typename StorageT>
__global__ __launch_bounds__(kThreads) void selected_attention_split_kernel(
    const __nv_bfloat16* __restrict__ query, const int* __restrict__ positions,
    const int* __restrict__ table_rows, const int* __restrict__ selections,
    const int* __restrict__ counts, const int* __restrict__ tables, int logical_pages,
    const StorageT* __restrict__ keys, const StorageT* __restrict__ values,
    float* __restrict__ partial) {
    __shared__ float numerators[kWarps][256];
    __shared__ float maxima[kWarps];
    __shared__ float sums[kWarps];
    __shared__ float factors[kWarps];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int head = blockIdx.x;
    const int batch = blockIdx.y;
    const int split = blockIdx.z;
    const int splits = gridDim.z;
    const int complete = (positions[batch] + 1) / 4;
    const int count = counts[batch];
    const int total = count * 4 + ((positions[batch] + 1) & 3);
    const int begin = total * split / splits;
    const int end = total * (split + 1) / splits;
    const auto* selected = selections + batch * 512;
    const auto* table = tables + table_rows[batch] * logical_pages;
    const auto* q = query + (batch * 24 + head) * 256;
    float qvalues[8];
    float accumulator[8]{};
    #pragma unroll
    for (int i = 0; i < 8; ++i) { qvalues[i] = __bfloat162float(q[lane + 32 * i]); }
    float maximum = -CUDART_INF_F;
    float denominator = 0.0F;
    for (int ordinal = begin + warp; ordinal < end; ordinal += kWarps) {
        const int token = selected_token(ordinal, count, complete, selected);
        const int page = table[token / 64];
        const auto offset = ((static_cast<std::int64_t>(page) * 2 + head / 12) * 64 + token % 64) * 256;
        float score = 0.0F;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            score = fmaf(qvalues[i], to_float(keys[offset + lane + 32 * i]), score);
        }
        score = warp_sum(score) * 0.0625F;
        const float next = fmaxf(maximum, score);
        const float prior = expf(maximum - next);
        const float weight = expf(score - next);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            accumulator[i] = fmaf(weight, to_float(values[offset + lane + 32 * i]),
                                  accumulator[i] * prior);
        }
        denominator = denominator * prior + weight;
        maximum = next;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) { numerators[warp][lane + 32 * i] = accumulator[i]; }
    if (lane == 0) { maxima[warp] = maximum; sums[warp] = denominator; }
    __syncthreads();
    auto* result = partial + ((batch * 24 + head) * splits + split) * kPartialStride;
    if (threadIdx.x == 0) {
        float merged_max = -CUDART_INF_F;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) { merged_max = fmaxf(merged_max, maxima[w]); }
        float merged_sum = 0.0F;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            factors[w] = sums[w] > 0.0F ? expf(maxima[w] - merged_max) : 0.0F;
            merged_sum += factors[w] * sums[w];
        }
        result[256] = merged_max;
        result[257] = merged_sum;
    }
    __syncthreads();
    for (int dim = threadIdx.x; dim < 256; dim += kThreads) {
        float sum = 0.0F;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) { sum = fmaf(numerators[w][dim], factors[w], sum); }
        result[dim] = sum;
    }
}

__global__ void selected_attention_merge_kernel(const float* __restrict__ partial,
                                                 __nv_bfloat16* __restrict__ output, int splits) {
    __shared__ float factors[16];
    __shared__ float denominator;
    const int row = blockIdx.y * 24 + blockIdx.x;
    const auto* values = partial + row * splits * kPartialStride;
    if (threadIdx.x < 32) {
        const int lane = threadIdx.x;
        float maximum = lane < splits ? values[lane * kPartialStride + 256] : -CUDART_INF_F;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
        }
        const float sum = lane < splits ? values[lane * kPartialStride + 257] : 0.0F;
        const float factor = sum > 0.0F ? expf(values[lane * kPartialStride + 256] - maximum) : 0.0F;
        if (lane < splits) { factors[lane] = factor; }
        const float total = warp_sum(factor * sum);
        if (lane == 0) { denominator = total; }
    }
    __syncthreads();
    float value = 0.0F;
    for (int split = 0; split < splits; ++split) {
        value = fmaf(values[split * kPartialStride + threadIdx.x], factors[split], value);
    }
    output[row * 256 + threadIdx.x] = __float2bfloat16_rn(denominator > 0.0F ? value / denominator : 0.0F);
}


bool exact(const Tensor& tensor, DType type, int n0, int n1 = 1, int n2 = 1, int n3 = 1) {
    return tensor.data != nullptr && reinterpret_cast<std::uintptr_t>(tensor.data) % 16 == 0 &&
        tensor.dtype == type && tensor.ne[0] == n0 && tensor.ne[1] == n1 &&
        tensor.ne[2] == n2 && tensor.ne[3] == n3 && tensor.is_contiguous();
}
} // namespace

std::size_t selected_block_attention_workspace_capacity_bytes(int batch) {
    if (batch < 1 || batch > 8) { throw std::invalid_argument("selected_block_attention: invalid batch"); }
    return static_cast<std::size_t>(batch) * 24 * kPartitions * kPartialStride * sizeof(float);
}

void selected_block_attention(const Tensor& query, const Tensor& positions,
                              const Tensor& table_rows, const Tensor& selections,
                              const Tensor& counts, const SelectedBlockAttentionCache& cache,
                              WorkspaceArena& workspace, Tensor& output, cudaStream_t stream) {
    const int batch = query.ne[2];
    (void)selected_block_attention_workspace_capacity_bytes(batch);
    const auto type = cache.keys.dtype;
    const int pages = cache.keys.ne[3];
    const int logical = cache.block_tables.ne[0];
    const int rows = cache.block_tables.ne[1];
    if (!exact(query, DType::BF16, 256, 24, batch) ||
        !exact(output, DType::BF16, 256, 24, batch) || !exact(positions, DType::I32, batch) ||
        !exact(table_rows, DType::I32, batch) || !exact(selections, DType::I32, 512, batch) ||
        !exact(counts, DType::I32, batch) || (type != DType::BF16 && type != DType::FP8_E4M3FN) ||
        pages <= 0 || logical <= 0 || rows <= 0 ||
        !exact(cache.keys, type, 256, 64, 2, pages) ||
        !exact(cache.values, type, 256, 64, 2, pages) ||
        !exact(cache.block_tables, DType::I32, logical, rows) || stream == nullptr) {
        throw std::invalid_argument("selected_block_attention: invalid tensor profile");
    }
    const auto scope = workspace.scope();
    const int splits = kPartitions;
    const auto partial = workspace.alloc(DType::FP32, {kPartialStride, splits, 24, batch}, 256);
    const auto launch = [&]<class StorageT>() {
        selected_attention_split_kernel<StorageT><<<dim3(24, batch, splits), kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(query.data), static_cast<const int*>(positions.data),
            static_cast<const int*>(table_rows.data), static_cast<const int*>(selections.data),
            static_cast<const int*>(counts.data), static_cast<const int*>(cache.block_tables.data),
            logical, static_cast<const StorageT*>(cache.keys.data),
            static_cast<const StorageT*>(cache.values.data), static_cast<float*>(partial.data));
    };
    if (type == DType::BF16) { launch.template operator()<__nv_bfloat16>(); }
    else { launch.template operator()<__nv_fp8_e4m3>(); }
    CUDA_CHECK(cudaGetLastError());
    selected_attention_merge_kernel<<<dim3(24, batch), 256, 0, stream>>>(
        static_cast<const float*>(partial.data), static_cast<__nv_bfloat16*>(output.data), splits);
    CUDA_CHECK(cudaGetLastError());
}

void selected_block_attention(const Tensor& query, const Tensor& positions,
                                      int table_row, const Tensor& selections,
                                      const Tensor& counts, const SelectedBlockAttentionCache& cache,
                                      Tensor& output, cudaStream_t stream) {
    const int tokens = query.ne[2];
    const auto type = cache.keys.dtype;
    const int pages = cache.keys.ne[3];
    const int logical = cache.block_tables.ne[0];
    const int rows = cache.block_tables.ne[1];
    if (tokens < 1 || tokens > 262144 || table_row < 0 || table_row >= rows ||
        !exact(query, DType::BF16, 256, 24, tokens) ||
        !exact(output, DType::BF16, 256, 24, tokens) || !exact(positions, DType::I32, tokens) ||
        !exact(selections, DType::I32, 512, tokens) || !exact(counts, DType::I32, tokens) ||
        (type != DType::BF16 && type != DType::FP8_E4M3FN) || pages <= 0 || logical <= 0 ||
        !exact(cache.keys, type, 256, 64, 2, pages) ||
        !exact(cache.values, type, 256, 64, 2, pages) ||
        !exact(cache.block_tables, DType::I32, logical, rows) || stream == nullptr) {
        throw std::invalid_argument("selected_block_attention: invalid shared-row tensor profile");
    }
    const auto launch = [&]<class StorageT>() {
        selected_attention_prefill_kernel<StorageT><<<2 * tokens, kMmaThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(query.data), static_cast<const int*>(positions.data),
            table_row, static_cast<const int*>(selections.data), static_cast<const int*>(counts.data),
            static_cast<const int*>(cache.block_tables.data), logical,
            static_cast<const StorageT*>(cache.keys.data), static_cast<const StorageT*>(cache.values.data),
            static_cast<__nv_bfloat16*>(output.data));
    };
    if (type == DType::BF16) { launch.template operator()<__nv_bfloat16>(); }
    else { launch.template operator()<__nv_fp8_e4m3>(); }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops
