#include "targets/qwen3_8_flash_next/impl/qsa_attention_kernels.h"

#include "core/device.h"
#include "ninfer/ops/selected_block_attention.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace ninfer::targets::qwen3_8_flash_next::detail {
namespace {

template <typename StorageT>
__device__ __forceinline__ StorageT to_storage(__nv_bfloat16 x);

template <>
__device__ __forceinline__ __nv_bfloat16 to_storage<__nv_bfloat16>(__nv_bfloat16 x) {
    return x;
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 to_storage<__nv_fp8_e4m3>(__nv_bfloat16 x) {
    return __nv_fp8_e4m3(__bfloat162float(x));
}


template <typename StorageT>
struct VectorKV;

template <>
struct VectorKV<__nv_bfloat16> {
    __device__ __forceinline__ static void load_8(const __nv_bfloat16* ptr, int lane_id, float out[8]) {
        const float4 vec = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(ptr) + lane_id * 16);
        const auto* bf16 = reinterpret_cast<const __nv_bfloat16*>(&vec);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = __bfloat162float(bf16[i]);
        }
    }
};

template <>
struct VectorKV<__nv_fp8_e4m3> {
    __device__ __forceinline__ static void load_8(const __nv_fp8_e4m3* ptr, int lane_id, float out[8]) {
        const uint2 vec = *reinterpret_cast<const uint2*>(reinterpret_cast<const char*>(ptr) + lane_id * 8);
        const auto* fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(&vec);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = static_cast<float>(fp8[i]);
        }
    }
};

template <typename StorageT>
__device__ __forceinline__ void stage_kv_vector(
    __nv_bfloat16* dst, const StorageT* src, bool valid) {
    if constexpr (std::is_same_v<StorageT, __nv_bfloat16>) {
        ops::cp_async_zfill<16, ops::Cache::cg>(dst, src, valid ? 16 : 0);
    } else {
        if (valid) {
            const unsigned long long raw = *reinterpret_cast<const unsigned long long*>(src);
            const auto* fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
            __nv_bfloat16 bf16_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                bf16_vals[i] = __float2bfloat16_rn(static_cast<float>(fp8[i]));
            }
            *reinterpret_cast<float4*>(dst) = *reinterpret_cast<float4*>(bf16_vals);
        } else {
            float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(dst) = zero;
        }
    }
}

constexpr int kHeadDim         = 256;
constexpr int kQueryHeads      = 24;
constexpr int kKvHeads         = 2;
constexpr int kProjectedRows   = 13'312;
constexpr int kMainKeyOffset   = 12'288;
constexpr int kMainValueOffset = 12'800;
constexpr int kPageTokens      = 64;
constexpr int kSelectedBlocks  = 512;
constexpr float kRopeTheta     = 1.0e7F;
constexpr float kScale         = 0.0625F; // 1/sqrt(256)

__device__ float rope_frequency(int pair) {
    return expf((-2.0F * static_cast<float>(pair) / 64.0F) * logf(kRopeTheta));
}

__device__ void store_mrope(const __nv_bfloat16* normalized, const std::int32_t* positions,
                            __nv_bfloat16* output, int dim) {
    if (dim < 32) {
        const float angle = static_cast<float>(positions[dim % 3]) * rope_frequency(dim);
        float sine        = 0.0F;
        float cosine      = 0.0F;
        sincosf(angle, &sine, &cosine);
        const float first  = __bfloat162float(normalized[dim]);
        const float second = __bfloat162float(normalized[dim + 32]);
        output[dim]        = __float2bfloat16_rn(first * cosine - second * sine);
        output[dim + 32]   = __float2bfloat16_rn(second * cosine + first * sine);
    } else if (dim >= 64) {
        output[dim] = normalized[dim];
    }
}

__global__ void prepare_query_kernel(const __nv_bfloat16* __restrict__ projected,
                                     const __nv_bfloat16* __restrict__ norm,
                                     const std::int32_t* __restrict__ positions,
                                     __nv_bfloat16* __restrict__ query,
                                     __nv_bfloat16* __restrict__ gate, int batch_size) {
    __shared__ float warp_squares[8];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim   = static_cast<int>(threadIdx.x);
    const int head  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int warp  = dim >> 5;
    const int lane  = dim & 31;
    const std::int64_t source =
        static_cast<std::int64_t>(batch) * kProjectedRows + head * 2 * kHeadDim;
    const float x      = __bfloat162float(projected[source + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    gate[static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim + dim] =
        projected[source + kHeadDim + dim];
    __syncthreads();
    float sum = 0.0F;
    for (float value : warp_squares) { sum += value; }
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[batch], positions[batch_size + batch],
                                       positions[2 * batch_size + batch]};
    auto* destination =
        query + static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim;
    store_mrope(normalized, local_positions, destination, dim);
}

template <typename StorageT>
__global__ void prepare_append_kv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ norm,
    const std::int32_t* __restrict__ token_indices, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ block_tables,
    int logical_pages, StorageT* __restrict__ key_pages,
    StorageT* __restrict__ value_pages, __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value, int batch_size) {
    __shared__ float warp_squares[8];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim                     = static_cast<int>(threadIdx.x);
    const int head                    = static_cast<int>(blockIdx.x);
    const int batch                   = static_cast<int>(blockIdx.y);
    const int warp                    = dim >> 5;
    const int lane                    = dim & 31;
    const std::int64_t projected_base = static_cast<std::int64_t>(batch) * kProjectedRows;
    const float x =
        __bfloat162float(projected[projected_base + kMainKeyOffset + head * kHeadDim + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    const auto value_word = projected[projected_base + kMainValueOffset + head * kHeadDim + dim];
    value[static_cast<std::int64_t>(batch) * kKvHeads * kHeadDim + head * kHeadDim + dim] =
        value_word;
    __syncthreads();
    float sum = 0.0F;
    for (float entry : warp_squares) { sum += entry; }
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[batch], positions[batch_size + batch],
                                       positions[2 * batch_size + batch]};
    auto* key_destination =
        key + static_cast<std::int64_t>(batch) * kKvHeads * kHeadDim + head * kHeadDim;
    store_mrope(normalized, local_positions, key_destination, dim);
    __syncthreads();

    const int token         = token_indices[batch];
    if (token < 0) { return; }
    const int logical_page  = token / kPageTokens;
    const int page_offset   = token % kPageTokens;
    const int physical_page = block_tables[table_rows[batch] * logical_pages + logical_page];
    const std::int64_t page_index =
        ((static_cast<std::int64_t>(physical_page) * kKvHeads + head) * kPageTokens + page_offset) *
            kHeadDim +
        dim;
    key_pages[page_index]   = to_storage<StorageT>(key_destination[dim]);
    value_pages[page_index] = to_storage<StorageT>(value_word);
}

__device__ int selected_token(int ordinal, int selected_count, int complete_blocks,
                              const std::int32_t* selected) {
    const int selected_tokens = selected_count * 4;
    if (ordinal < selected_tokens) { return selected[ordinal / 4] * 4 + ordinal % 4; }
    return complete_blocks * 4 + ordinal - selected_tokens;
}

__global__ void gate_output_kernel(const __nv_bfloat16* __restrict__ attended,
                                   const __nv_bfloat16* __restrict__ gate,
                                   __nv_bfloat16* __restrict__ gated, int elements) {
    const int index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index < elements) {
        gated[index] = __float2bfloat16_rn(__bfloat162float(attended[index]) *
                                           ops::sigmoid(__bfloat162float(gate[index])));
    }
}

} // namespace

void flash_next_qsa_attention_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                     const Tensor& table_rows, const Tensor& selected_blocks,
                                     const Tensor& selected_counts, const Tensor& query_norm,
                                     const Tensor& key_norm, QsaAttentionCacheView cache,
                                     FlashNextQsaAttentionWorkspace& scratch, WorkspaceArena& workspace,
                                     cudaStream_t stream) {
    const int batch = token_indices.ne[0];
    prepare_query_kernel<<<dim3(kQueryHeads, batch), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data),
        static_cast<__nv_bfloat16*>(scratch.gate.data), batch);
    CUDA_CHECK(cudaGetLastError());
    flash_next_qsa_attention_store_launch(scratch.projected, token_indices, mrope_positions,
        table_rows, 0, key_norm, cache, scratch.key, scratch.value, stream);
    ops::selected_block_attention(scratch.query, token_indices, table_rows,
        selected_blocks, selected_counts, {cache.key_pages, cache.value_pages, cache.block_tables},
        workspace, scratch.attended, stream);
    const int elements    = kQueryHeads * kHeadDim * batch;
    constexpr int threads = 256;
    gate_output_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.attended.data),
        static_cast<const __nv_bfloat16*>(scratch.gate.data),
        static_cast<__nv_bfloat16*>(scratch.gated.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void qsa_prefill_prepare_query_kernel(const __nv_bfloat16* __restrict__ projected,
                                                 const __nv_bfloat16* __restrict__ norm,
                                                 const std::int32_t* __restrict__ positions,
                                                 __nv_bfloat16* __restrict__ query,
                                                 __nv_bfloat16* __restrict__ gate, int tokens) {
    __shared__ float warp_squares[8];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim   = static_cast<int>(threadIdx.x);
    const int head  = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = dim >> 5;
    const int lane  = dim & 31;
    const std::int64_t source =
        static_cast<std::int64_t>(token) * kProjectedRows + head * 2 * kHeadDim;
    const float x      = __bfloat162float(projected[source + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    gate[static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + head * kHeadDim + dim] =
        projected[source + kHeadDim + dim];
    __syncthreads();
    float sum = 0.0F;
    for (float value : warp_squares) { sum += value; }
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[token], positions[tokens + token],
                                       positions[2 * tokens + token]};
    auto* destination =
        query + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + head * kHeadDim;
    store_mrope(normalized, local_positions, destination, dim);
}

template <typename StorageT>
__global__ void qsa_prefill_prepare_append_kv_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ norm,
    const std::int32_t* __restrict__ token_indices, const std::int32_t* __restrict__ positions,
    int table_row, const std::int32_t* __restrict__ block_tables, int logical_pages,
    StorageT* __restrict__ key_pages, StorageT* __restrict__ value_pages,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value, int tokens) {
    __shared__ float warp_squares[8];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim   = static_cast<int>(threadIdx.x);
    const int head  = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = dim >> 5;
    const int lane  = dim & 31;
    const std::int64_t projected_base = static_cast<std::int64_t>(token) * kProjectedRows;
    const float x =
        __bfloat162float(projected[projected_base + kMainKeyOffset + head * kHeadDim + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    const auto value_word = projected[projected_base + kMainValueOffset + head * kHeadDim + dim];
    value[static_cast<std::int64_t>(token) * kKvHeads * kHeadDim + head * kHeadDim + dim] =
        value_word;
    __syncthreads();
    float sum = 0.0F;
    for (float entry : warp_squares) { sum += entry; }
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[token], positions[tokens + token],
                                       positions[2 * tokens + token]};
    auto* key_destination =
        key + static_cast<std::int64_t>(token) * kKvHeads * kHeadDim + head * kHeadDim;
    store_mrope(normalized, local_positions, key_destination, dim);
    __syncthreads();

    const int token_idx     = token_indices[token];
    if (token_idx < 0) { return; }
    const int logical_page  = token_idx / kPageTokens;
    const int page_offset   = token_idx % kPageTokens;
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    const std::int64_t page_index =
        ((static_cast<std::int64_t>(physical_page) * kKvHeads + head) * kPageTokens + page_offset) *
            kHeadDim +
        dim;
    key_pages[page_index]   = to_storage<StorageT>(key_destination[dim]);
    value_pages[page_index] = to_storage<StorageT>(value_word);
}

template <int NUM_WARPS, typename StorageT>
__global__ void qsa_prefill_sparse_attention_kernel(
    const __nv_bfloat16* __restrict__ query, const std::int32_t* __restrict__ token_indices,
    int table_row, const std::int32_t* __restrict__ selected_blocks,
    const std::int32_t* __restrict__ selected_counts, const std::int32_t* __restrict__ block_tables,
    int logical_pages, const StorageT* __restrict__ key_pages,
    const StorageT* __restrict__ value_pages, __nv_bfloat16* __restrict__ output) {
    
    constexpr int CHUNK_SIZE = 256;
    __shared__ float s_q[kHeadDim];
    __shared__ float s_scores[CHUNK_SIZE];
    __shared__ float s_reduction[CHUNK_SIZE];
    __shared__ float s_warp_acc[NUM_WARPS][kHeadDim];
    
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp_id  = tid >> 5;
    const int lane_id  = tid & 31;
    const int head     = static_cast<int>(blockIdx.x);
    const int token    = static_cast<int>(blockIdx.y);
    const int kv_head  = head / 12;

    const auto* q_ptr =
        query + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + head * kHeadDim;

    // Load Q into shared memory cooperatively (128-bit vector load)
    if (tid < 32) {
        const float4 q_vec = *reinterpret_cast<const float4*>(reinterpret_cast<const char*>(q_ptr) + lane_id * 16);
        const auto* bf16_q = reinterpret_cast<const __nv_bfloat16*>(&q_vec);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_q[lane_id * 8 + i] = __bfloat162float(bf16_q[i]);
        }
    }
    __syncthreads();

    // Register storage for local slice of Q (8 floats per lane)
    float reg_q[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        reg_q[i] = s_q[lane_id * 8 + i];
    }

    const int token_index     = token_indices[token];
    const int complete_blocks = (token_index + 1) / 4;
    const int tail_count      = (token_index + 1) & 3;
    const int selected_count  = selected_counts[token];
    const int total           = selected_count * 4 + tail_count;
    const auto* selected = selected_blocks + static_cast<std::int64_t>(token) * kSelectedBlocks;
    const bool is_dense  = (complete_blocks <= 512 && selected_count == complete_blocks);

    const auto* block_table_row = block_tables + table_row * logical_pages;

    // Global accumulator for this thread's 8 output dimensions
    float total_acc[8] = {0.0F};
    float running_max  = -__int_as_float(0x7F800000);
    float running_sum  = 0.0F;

    for (int start = 0; start < total; start += CHUNK_SIZE) {
        const int count = min(CHUNK_SIZE, total - start);

        // 1. Compute Q-K dot products cooperatively:
        // Unroll by 2 tokens per warp
        const int step = NUM_WARPS * 2;
        int c_base = warp_id * 2;
        for (; c_base + 1 < count; c_base += step) {
            int cand0, cand1;
            if (is_dense) {
                cand0 = start + c_base;
                cand1 = start + c_base + 1;
            } else {
                cand0 = selected_token(start + c_base, selected_count, complete_blocks, selected);
                cand1 = selected_token(start + c_base + 1, selected_count, complete_blocks, selected);
            }

            const int phys0 = block_table_row[cand0 / kPageTokens];
            const int phys1 = block_table_row[cand1 / kPageTokens];
            
            const auto* k_ptr0 = key_pages +
                ((static_cast<std::int64_t>(phys0) * kKvHeads + kv_head) * kPageTokens + (cand0 % kPageTokens)) * kHeadDim;
            const auto* k_ptr1 = key_pages +
                ((static_cast<std::int64_t>(phys1) * kKvHeads + kv_head) * kPageTokens + (cand1 % kPageTokens)) * kHeadDim;
            
            float k_val0[8], k_val1[8];
            VectorKV<StorageT>::load_8(k_ptr0, lane_id, k_val0);
            VectorKV<StorageT>::load_8(k_ptr1, lane_id, k_val1);

            float dot0 = 0.0F, dot1 = 0.0F;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dot0 = fmaf(reg_q[i], k_val0[i], dot0);
                dot1 = fmaf(reg_q[i], k_val1[i], dot1);
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot0 += __shfl_xor_sync(0xFFFFFFFF, dot0, offset);
                dot1 += __shfl_xor_sync(0xFFFFFFFF, dot1, offset);
            }

            if (lane_id == 0) {
                s_scores[c_base]     = dot0 * kScale;
                s_scores[c_base + 1] = dot1 * kScale;
            }
        }
        if (c_base < count) {
            const int cand0 = is_dense ? (start + c_base) : selected_token(start + c_base, selected_count, complete_blocks, selected);
            const int phys0 = block_table_row[cand0 / kPageTokens];
            const auto* k_ptr0 = key_pages +
                ((static_cast<std::int64_t>(phys0) * kKvHeads + kv_head) * kPageTokens + (cand0 % kPageTokens)) * kHeadDim;
            float k_val0[8];
            VectorKV<StorageT>::load_8(k_ptr0, lane_id, k_val0);

            float dot0 = 0.0F;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                dot0 = fmaf(reg_q[i], k_val0[i], dot0);
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                dot0 += __shfl_xor_sync(0xFFFFFFFF, dot0, offset);
            }
            if (lane_id == 0) {
                s_scores[c_base] = dot0 * kScale;
            }
        }
        __syncthreads();

        // Pad unused scores in chunk
        for (int i = tid; i < CHUNK_SIZE; i += blockDim.x) {
            s_reduction[i] = (i < count) ? s_scores[i] : -__int_as_float(0x7F800000);
        }
        __syncthreads();

        // Reduction for chunk_max
        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduction[tid] = fmaxf(s_reduction[tid], s_reduction[tid + stride]);
            }
            __syncthreads();
        }
        const float chunk_max = s_reduction[0];
        __syncthreads();

        // Softmax probabilities and reduction for chunk_sum
        for (int i = tid; i < CHUNK_SIZE; i += blockDim.x) {
            const float prob = (i < count) ? expf(s_scores[i] - chunk_max) : 0.0F;
            s_scores[i]    = prob;
            s_reduction[i] = prob;
        }
        __syncthreads();

        for (int stride = 128; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_reduction[tid] += s_reduction[tid + stride];
            }
            __syncthreads();
        }
        const float chunk_sum = s_reduction[0];

        // Rescale running accumulator
        const float next_max    = fmaxf(running_max, chunk_max);
        const float prior_scale = running_sum == 0.0F ? 0.0F : expf(running_max - next_max);
        const float chunk_scale = expf(chunk_max - next_max);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            total_acc[i] *= prior_scale;
        }
        running_sum = running_sum * prior_scale + chunk_sum * chunk_scale;
        running_max = next_max;

        // 2. Accumulate V in parallel across NUM_WARPS:
        float warp_acc[8] = {0.0F};
        int v_base = warp_id * 2;
        for (; v_base + 1 < count; v_base += step) {
            const float p0 = s_scores[v_base];
            const float p1 = s_scores[v_base + 1];
            if (p0 == 0.0F && p1 == 0.0F) continue;

            int cand0, cand1;
            if (is_dense) {
                cand0 = start + v_base;
                cand1 = start + v_base + 1;
            } else {
                cand0 = selected_token(start + v_base, selected_count, complete_blocks, selected);
                cand1 = selected_token(start + v_base + 1, selected_count, complete_blocks, selected);
            }

            const int phys0 = block_table_row[cand0 / kPageTokens];
            const int phys1 = block_table_row[cand1 / kPageTokens];
            
            const auto* v_ptr0 = value_pages +
                ((static_cast<std::int64_t>(phys0) * kKvHeads + kv_head) * kPageTokens + (cand0 % kPageTokens)) * kHeadDim;
            const auto* v_ptr1 = value_pages +
                ((static_cast<std::int64_t>(phys1) * kKvHeads + kv_head) * kPageTokens + (cand1 % kPageTokens)) * kHeadDim;
            
            float v_val0[8], v_val1[8];
            VectorKV<StorageT>::load_8(v_ptr0, lane_id, v_val0);
            VectorKV<StorageT>::load_8(v_ptr1, lane_id, v_val1);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                warp_acc[i] = fmaf(p0, v_val0[i], warp_acc[i]);
                warp_acc[i] = fmaf(p1, v_val1[i], warp_acc[i]);
            }
        }
        if (v_base < count) {
            const float p0 = s_scores[v_base];
            if (p0 != 0.0F) {
                const int cand0 = is_dense ? (start + v_base) : selected_token(start + v_base, selected_count, complete_blocks, selected);
                const int phys0 = block_table_row[cand0 / kPageTokens];
                const auto* v_ptr0 = value_pages +
                    ((static_cast<std::int64_t>(phys0) * kKvHeads + kv_head) * kPageTokens + (cand0 % kPageTokens)) * kHeadDim;
                float v_val0[8];
                VectorKV<StorageT>::load_8(v_ptr0, lane_id, v_val0);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    warp_acc[i] = fmaf(p0, v_val0[i], warp_acc[i]);
                }
            }
        }

        // Store warp accumulator to shared memory
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            s_warp_acc[warp_id][lane_id * 8 + i] = warp_acc[i];
        }
        __syncthreads();

        // Reduce across warps: each thread reduces its 8 elements across all NUM_WARPS
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float sum_v = 0.0F;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                sum_v += s_warp_acc[w][lane_id * 8 + i];
            }
            total_acc[i] += sum_v * chunk_scale;
        }
        __syncthreads();
    }

    // Write final output (tid 0..31 write all 256 output bf16s vectorized)
    if (warp_id == 0) {
        __nv_bfloat16 out_bf16[8];
        const float inv_sum = 1.0F / running_sum;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_bf16[i] = __float2bfloat16_rn(total_acc[i] * inv_sum);
        }
        auto* out_ptr = output + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + head * kHeadDim;
        *reinterpret_cast<float4*>(reinterpret_cast<char*>(out_ptr) + lane_id * 16) = *reinterpret_cast<float4*>(out_bf16);
    }
}

// GQA tiled MMA: one CTA per (kv_head, query token). The 12 query heads that share a KV head
// stage each K/V tile once and run m16n8k16 over it. Online softmax stays FP32.
constexpr int kMmaHeads     = 16; // 12 real heads, 4 padded
constexpr int kMmaBN        = 64;
constexpr int kMmaWarps     = 4;
constexpr int kMmaThreads   = 32 * kMmaWarps;
constexpr int kMmaNTiles    = kMmaBN / 8;
constexpr int kHeadsPerKv   = 12;

// G24: QK on all 4 warps (split 64-key N-tiles), PV as m16n8k16 over staged V.
// BN stays 64 so the online-softmax tile and FP32 order match the G17 kernel.
// P is rounded to BF16 after the FP32 rescale; that rounding is declared.
template <typename StorageT>
__global__ __launch_bounds__(kMmaThreads)
void qsa_prefill_sparse_attention_mma_sched_kernel(
    const __nv_bfloat16* __restrict__ query, const std::int32_t* __restrict__ token_indices,
    int table_row, const std::int32_t* __restrict__ selected_blocks,
    const std::int32_t* __restrict__ selected_counts, const std::int32_t* __restrict__ block_tables,
    int logical_pages, const StorageT* __restrict__ key_pages,
    const StorageT* __restrict__ value_pages, __nv_bfloat16* __restrict__ output) {
    using ops::cp_async_zfill;
    using ops::cp_commit;
    using ops::cp_wait;
    using ops::ldmatrix_x2;
    using ops::ldmatrix_x2_t;
    using ops::ldmatrix_x4;
    using ops::mma_bf16;
    using ops::smem_addr;
    using ops::detail::gemm_swz64;
    using Cache = ops::Cache;

    __shared__ __align__(16) __nv_bfloat16 Qs[kMmaHeads * kHeadDim];
    __shared__ __align__(16) __nv_bfloat16 Ks[kMmaBN * kHeadDim];
    __shared__ __align__(16) __nv_bfloat16 Vs[kMmaBN * kHeadDim];
    __shared__ float s_scores[kMmaHeads * kMmaBN];
    __shared__ __align__(16) __nv_bfloat16 Ps[kMmaHeads * kMmaBN];
    __shared__ float s_prior_scale[kMmaHeads];
    __shared__ float s_chunk_scale[kMmaHeads];
    __shared__ float s_inv_sum[kMmaHeads];

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int token   = static_cast<int>(blockIdx.y);
    const int q0      = kv_head * kHeadsPerKv;

    const int token_index     = token_indices[token];
    const int complete_blocks = (token_index + 1) / 4;
    const int tail_count      = (token_index + 1) & 3;
    const int selected_count  = selected_counts[token];
    const int total           = selected_count * 4 + tail_count;
    const auto* selected      = selected_blocks + static_cast<std::int64_t>(token) * kSelectedBlocks;
    const bool is_dense = (complete_blocks <= kSelectedBlocks && selected_count == complete_blocks);
    const auto* block_table_row = block_tables + table_row * logical_pages;

    const auto* q_base =
        query + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + q0 * kHeadDim;
    for (int item = tid; item < kMmaHeads * (kHeadDim / 8); item += kMmaThreads) {
        const int row = item / (kHeadDim / 8);
        const int k8  = item - row * (kHeadDim / 8);
        auto* dst     = &Qs[row * kHeadDim + gemm_swz64(row, k8 * 8)];
        const bool valid = row < kHeadsPerKv;
        const __nv_bfloat16* src = q_base + (valid ? row : 0) * kHeadDim + k8 * 8;
        cp_async_zfill<16, Cache::cg>(dst, src, valid ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    if (tid < kMmaHeads) {
        s_prior_scale[tid] = 1.0F;
        s_chunk_scale[tid] = 0.0F;
        s_inv_sum[tid]     = 0.0F;
    }
    float running_max[3] = {-__int_as_float(0x7F800000), -__int_as_float(0x7F800000),
                            -__int_as_float(0x7F800000)};
    float running_sum[3] = {};
    constexpr int kPvNTiles = 8;
    float oc[kPvNTiles][4]  = {};

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int arow     = a_rowoff;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int r0       = gid;
    const int r1       = r0 + 8;

    for (int start = 0; start < total; start += kMmaBN) {
        const int count = min(kMmaBN, total - start);
        for (int item = tid; item < kMmaBN * (kHeadDim / 8); item += kMmaThreads) {
            const int row = item / (kHeadDim / 8);
            const int k8  = item - row * (kHeadDim / 8);
            auto* kdst    = &Ks[row * kHeadDim + gemm_swz64(row, k8 * 8)];
            auto* vdst    = &Vs[row * kHeadDim + gemm_swz64(row, k8 * 8)]; // same K swizzle; PV uses ldmatrix.trans
            const bool valid = row < count;
            int cand = 0;
            if (valid) {
                cand = is_dense ? (start + row)
                                : selected_token(start + row, selected_count, complete_blocks,
                                                 selected);
            }
            const int phys = valid ? block_table_row[cand / kPageTokens] : 0;
            const std::int64_t kv_off =
                ((static_cast<std::int64_t>(phys) * kKvHeads + kv_head) * kPageTokens +
                 (valid ? (cand % kPageTokens) : 0)) *
                    kHeadDim +
                k8 * 8;
            stage_kv_vector(kdst, key_pages + kv_off, valid);
            stage_kv_vector(vdst, value_pages + kv_off, valid);
        }
        if constexpr (std::is_same_v<StorageT, __nv_bfloat16>) {
            cp_commit();
            cp_wait<0>();
        }
        __syncthreads();

        float tile_score[2][4] = {};
        const int nt0          = warp * 2;
#pragma unroll
        for (int ki = 0; ki < kHeadDim / 16; ++ki) {
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Qs[arow * kHeadDim + gemm_swz64(arow, ki * 16 + a_coloff)]));
#pragma unroll
            for (int nt_i = 0; nt_i < 2; ++nt_i) {
                unsigned bf[2];
                const int nt   = nt0 + nt_i;
                const int brow = nt * 8 + b_rin;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&Ks[brow * kHeadDim + gemm_swz64(brow, ki * 16 + b_koff)]));
                mma_bf16(tile_score[nt_i][0], tile_score[nt_i][1], tile_score[nt_i][2],
                         tile_score[nt_i][3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int nt_i = 0; nt_i < 2; ++nt_i) {
            const int nt = nt0 + nt_i;
            const int c0 = nt * 8 + 2 * lid;
            const int c1 = c0 + 1;
            auto store = [&](int head, int col, float value) {
                const float ninf = -__int_as_float(0x7F800000);
                s_scores[head * kMmaBN + col] =
                    (head < kHeadsPerKv && col < count) ? value * kScale : ninf;
            };
            store(r0, c0, tile_score[nt_i][0]);
            store(r0, c1, tile_score[nt_i][1]);
            store(r1, c0, tile_score[nt_i][2]);
            store(r1, c1, tile_score[nt_i][3]);
        }
        __syncthreads();

#pragma unroll
        for (int hi = 0; hi < 3; ++hi) {
            const int head = warp + hi * kMmaWarps;
            float hmax     = -__int_as_float(0x7F800000);
            for (int col = lane; col < kMmaBN; col += 32) {
                hmax = fmaxf(hmax, s_scores[head * kMmaBN + col]);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                hmax = fmaxf(hmax, __shfl_xor_sync(0xFFFFFFFF, hmax, off));
            }
            float hsum = 0.0F;
            for (int col = lane; col < kMmaBN; col += 32) {
                const float p = expf(s_scores[head * kMmaBN + col] - hmax);
                s_scores[head * kMmaBN + col] = p;
                hsum += p;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                hsum += __shfl_xor_sync(0xFFFFFFFF, hsum, off);
            }
            const float next_max    = fmaxf(running_max[hi], hmax);
            const float prior_scale = running_sum[hi] == 0.0F ? 0.0F : expf(running_max[hi] - next_max);
            const float chunk_scale = expf(hmax - next_max);
            running_sum[hi]         = running_sum[hi] * prior_scale + hsum * chunk_scale;
            running_max[hi]         = next_max;
            if (lane == 0) {
                s_prior_scale[head] = prior_scale;
                s_chunk_scale[head] = chunk_scale;
                s_inv_sum[head]     = 1.0F / running_sum[hi];
            }
        }
        __syncthreads();

        // P as A: gemm_swz64 over row stride kMmaBN. V stays [key][dim] swizzled like K;
        // B is ldmatrix_x2_t with krow covering both 8-key groups of the 16-key K-tile.
        for (int item = tid; item < kMmaHeads * kMmaBN; item += kMmaThreads) {
            const int h = item / kMmaBN;
            const int c = item - h * kMmaBN;
            const float p =
                (h < kHeadsPerKv && c < count) ? s_scores[h * kMmaBN + c] * s_chunk_scale[h] : 0.0F;
            Ps[h * kMmaBN + gemm_swz64(h, c)] = __float2bfloat16_rn(p);
        }
        __syncthreads();

#pragma unroll
        for (int nt = 0; nt < kPvNTiles; ++nt) {
            oc[nt][0] *= s_prior_scale[r0];
            oc[nt][1] *= s_prior_scale[r0];
            oc[nt][2] *= s_prior_scale[r1];
            oc[nt][3] *= s_prior_scale[r1];
        }
#pragma unroll
        for (int ki = 0; ki < kMmaBN / 16; ++ki) {
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ps[arow * kMmaBN + gemm_swz64(arow, ki * 16 + a_coloff)]));
#pragma unroll
            for (int nt = 0; nt < kPvNTiles; ++nt) {
                unsigned bf[2];
                const int dim0 = (warp * kPvNTiles + nt) * 8;
                const int krow = ki * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
                ldmatrix_x2_t(bf[0], bf[1],
                              smem_addr(&Vs[krow * kHeadDim + gemm_swz64(krow, dim0)]));
                mma_bf16(oc[nt][0], oc[nt][1], oc[nt][2], oc[nt][3], af[0], af[1], af[2], af[3],
                         bf[0], bf[1]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int nt = 0; nt < kPvNTiles; ++nt) {
        const int d0 = (warp * kPvNTiles + nt) * 8 + 2 * lid;
        const int d1 = d0 + 1;
        auto store = [&](int head, int dim, float value) {
            if (head >= kHeadsPerKv) { return; }
            auto* out_ptr = output + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim +
                            (q0 + head) * kHeadDim + dim;
            *out_ptr = __float2bfloat16_rn(value * s_inv_sum[head]);
        };
        store(r0, d0, oc[nt][0]);
        store(r0, d1, oc[nt][1]);
        store(r1, d0, oc[nt][2]);
        store(r1, d1, oc[nt][3]);
    }
}

bool flash_next_qsa_mma_sched_new() {
    const char* env = std::getenv("NINFER_FLASH_NEXT_QSA_MMA_SCHED");
    if (env == nullptr || env[0] == '\0') { return false; }
    return std::strcmp(env, "new") == 0 || (env[0] == '1' && env[1] == '\0');
}

void flash_next_qsa_attention_store_launch(const Tensor& projected, const Tensor& token_indices,
                                           const Tensor& mrope_positions, const Tensor& table_rows,
                                           int table_row, const Tensor& key_norm,
                                           QsaAttentionCacheView cache, Tensor& key, Tensor& value,
                                           cudaStream_t stream) {
    const int tokens = token_indices.ne[0];
    const int batch  = tokens;
    if (table_rows.data != nullptr) {
        if (cache.key_pages.dtype == DType::FP8_E4M3FN) {
            prepare_append_kv_kernel<__nv_fp8_e4m3><<<dim3(kKvHeads, batch), kHeadDim, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(projected.data),
                static_cast<const __nv_bfloat16*>(key_norm.data),
                static_cast<const std::int32_t*>(token_indices.data),
                static_cast<const std::int32_t*>(mrope_positions.data),
                static_cast<const std::int32_t*>(table_rows.data),
                static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
                static_cast<__nv_fp8_e4m3*>(cache.key_pages.data),
                static_cast<__nv_fp8_e4m3*>(cache.value_pages.data),
                static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(value.data),
                batch);
            CUDA_CHECK(cudaGetLastError());
        } else {
            prepare_append_kv_kernel<__nv_bfloat16><<<dim3(kKvHeads, batch), kHeadDim, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(projected.data),
                static_cast<const __nv_bfloat16*>(key_norm.data),
                static_cast<const std::int32_t*>(token_indices.data),
                static_cast<const std::int32_t*>(mrope_positions.data),
                static_cast<const std::int32_t*>(table_rows.data),
                static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
                static_cast<__nv_bfloat16*>(cache.key_pages.data),
                static_cast<__nv_bfloat16*>(cache.value_pages.data),
                static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(value.data),
                batch);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        if (cache.key_pages.dtype == DType::FP8_E4M3FN) {
            qsa_prefill_prepare_append_kv_kernel<__nv_fp8_e4m3>
                <<<dim3(kKvHeads, tokens), kHeadDim, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(projected.data),
                    static_cast<const __nv_bfloat16*>(key_norm.data),
                    static_cast<const std::int32_t*>(token_indices.data),
                    static_cast<const std::int32_t*>(mrope_positions.data), table_row,
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    cache.block_tables.ne[0], static_cast<__nv_fp8_e4m3*>(cache.key_pages.data),
                    static_cast<__nv_fp8_e4m3*>(cache.value_pages.data),
                    static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(value.data),
                    tokens);
            CUDA_CHECK(cudaGetLastError());
        } else {
            qsa_prefill_prepare_append_kv_kernel<__nv_bfloat16>
                <<<dim3(kKvHeads, tokens), kHeadDim, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(projected.data),
                    static_cast<const __nv_bfloat16*>(key_norm.data),
                    static_cast<const std::int32_t*>(token_indices.data),
                    static_cast<const std::int32_t*>(mrope_positions.data), table_row,
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    cache.block_tables.ne[0], static_cast<__nv_bfloat16*>(cache.key_pages.data),
                    static_cast<__nv_bfloat16*>(cache.value_pages.data),
                    static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(value.data),
                    tokens);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

void flash_next_qsa_attention_prefill_launch(
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    const Tensor& selected_blocks, const Tensor& selected_counts, const Tensor& query_norm,
    const Tensor& key_norm, QsaAttentionCacheView cache, FlashNextQsaAttentionWorkspace& scratch,
    cudaStream_t stream, bool use_mma) {
    const int tokens = token_indices.ne[0];
    qsa_prefill_prepare_query_kernel<<<dim3(kQueryHeads, tokens), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data),
        static_cast<__nv_bfloat16*>(scratch.gate.data), tokens);
    CUDA_CHECK(cudaGetLastError());
    flash_next_qsa_attention_store_launch(scratch.projected, token_indices, mrope_positions,
        Tensor{}, table_row, key_norm, cache, scratch.key, scratch.value, stream);
    const bool is_fp8 = (cache.key_pages.dtype == DType::FP8_E4M3FN);
    if (is_fp8) {
        if (use_mma) {
            const auto* q_ptr  = static_cast<const __nv_bfloat16*>(scratch.query.data);
            const auto* ti_ptr = static_cast<const std::int32_t*>(token_indices.data);
            const auto* sb_ptr = static_cast<const std::int32_t*>(selected_blocks.data);
            const auto* sc_ptr = static_cast<const std::int32_t*>(selected_counts.data);
            const auto* bt_ptr = static_cast<const std::int32_t*>(cache.block_tables.data);
            const auto* k_ptr  = static_cast<const __nv_fp8_e4m3*>(cache.key_pages.data);
            const auto* v_ptr  = static_cast<const __nv_fp8_e4m3*>(cache.value_pages.data);
            auto* att_ptr      = static_cast<__nv_bfloat16*>(scratch.attended.data);
            if (flash_next_qsa_mma_sched_new()) {
                qsa_prefill_sparse_attention_mma_sched_kernel<__nv_fp8_e4m3><<<dim3(kKvHeads, tokens), kMmaThreads, 0,
                                                                stream>>>(
                    q_ptr, ti_ptr, table_row, sb_ptr, sc_ptr, bt_ptr, cache.block_tables.ne[0], k_ptr,
                    v_ptr, att_ptr);
            } else {
                ops::selected_block_attention(
                    scratch.query, token_indices, table_row, selected_blocks, selected_counts,
                    {cache.key_pages, cache.value_pages, cache.block_tables}, scratch.attended, stream);
            }
        } else {
            constexpr int kPrefillWarps   = 4;
            constexpr int kPrefillThreads = kPrefillWarps * 32;
            qsa_prefill_sparse_attention_kernel<kPrefillWarps, __nv_fp8_e4m3>
                <<<dim3(kQueryHeads, tokens), kPrefillThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(scratch.query.data),
                    static_cast<const std::int32_t*>(token_indices.data), table_row,
                    static_cast<const std::int32_t*>(selected_blocks.data),
                    static_cast<const std::int32_t*>(selected_counts.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
                    static_cast<const __nv_fp8_e4m3*>(cache.key_pages.data),
                    static_cast<const __nv_fp8_e4m3*>(cache.value_pages.data),
                    static_cast<__nv_bfloat16*>(scratch.attended.data));
        }
        CUDA_CHECK(cudaGetLastError());
    } else {
        if (use_mma) {
            const auto* q_ptr  = static_cast<const __nv_bfloat16*>(scratch.query.data);
            const auto* ti_ptr = static_cast<const std::int32_t*>(token_indices.data);
            const auto* sb_ptr = static_cast<const std::int32_t*>(selected_blocks.data);
            const auto* sc_ptr = static_cast<const std::int32_t*>(selected_counts.data);
            const auto* bt_ptr = static_cast<const std::int32_t*>(cache.block_tables.data);
            const auto* k_ptr  = static_cast<const __nv_bfloat16*>(cache.key_pages.data);
            const auto* v_ptr  = static_cast<const __nv_bfloat16*>(cache.value_pages.data);
            auto* att_ptr      = static_cast<__nv_bfloat16*>(scratch.attended.data);
            if (flash_next_qsa_mma_sched_new()) {
                qsa_prefill_sparse_attention_mma_sched_kernel<__nv_bfloat16><<<dim3(kKvHeads, tokens), kMmaThreads, 0,
                                                                stream>>>(
                    q_ptr, ti_ptr, table_row, sb_ptr, sc_ptr, bt_ptr, cache.block_tables.ne[0], k_ptr,
                    v_ptr, att_ptr);
            } else {
                ops::selected_block_attention(
                    scratch.query, token_indices, table_row, selected_blocks, selected_counts,
                    {cache.key_pages, cache.value_pages, cache.block_tables}, scratch.attended, stream);
            }
        } else {
            constexpr int kPrefillWarps   = 4;
            constexpr int kPrefillThreads = kPrefillWarps * 32;
            qsa_prefill_sparse_attention_kernel<kPrefillWarps, __nv_bfloat16>
                <<<dim3(kQueryHeads, tokens), kPrefillThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(scratch.query.data),
                    static_cast<const std::int32_t*>(token_indices.data), table_row,
                    static_cast<const std::int32_t*>(selected_blocks.data),
                    static_cast<const std::int32_t*>(selected_counts.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
                    static_cast<const __nv_bfloat16*>(cache.key_pages.data),
                    static_cast<const __nv_bfloat16*>(cache.value_pages.data),
                    static_cast<__nv_bfloat16*>(scratch.attended.data));
        }
        CUDA_CHECK(cudaGetLastError());
    }
    const int elements    = kQueryHeads * kHeadDim * tokens;
    constexpr int threads = 256;
    gate_output_kernel<<<(elements + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.attended.data),
        static_cast<const __nv_bfloat16*>(scratch.gate.data),
        static_cast<__nv_bfloat16*>(scratch.gated.data), elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_8_flash_next::detail
