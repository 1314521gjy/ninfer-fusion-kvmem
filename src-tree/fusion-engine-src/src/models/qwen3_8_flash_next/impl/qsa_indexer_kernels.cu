#include "models/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include "ops/common/warp.cuh"

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_topk.cuh>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

constexpr int kHeadDim          = 128;
constexpr int kQueryHeads       = 4;
constexpr int kProjectionRows   = 640;
constexpr int kRawKeyOffset     = 512;
constexpr int kCompressedPage   = 64;
constexpr int kSelectedBlocks   = 512;
constexpr float kRopeTheta      = 1.0e7F;
constexpr float kIndexerScaling = 0.08838834764831845F; // 1/sqrt(128)

__device__ float rope_frequency(int pair) {
    return expf((-2.0F * static_cast<float>(pair) / 64.0F) * logf(kRopeTheta));
}

__device__ void store_rotated(const __nv_bfloat16* normalized, const std::int32_t* positions,
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
                                     __nv_bfloat16* __restrict__ query, int batch_size) {
    __shared__ float warp_squares[4];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim   = static_cast<int>(threadIdx.x);
    const int head  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int warp  = dim >> 5;
    const int lane  = dim & 31;
    const float x   = __bfloat162float(
        projected[static_cast<std::int64_t>(batch) * kProjectionRows + head * kHeadDim + dim]);
    const float square = ops::warp_reduce_sum(x * x);
    if (lane == 0) { warp_squares[warp] = square; }
    __syncthreads();
    const float sum = warp_squares[0] + warp_squares[1] + warp_squares[2] + warp_squares[3];
    normalized[dim] = __float2bfloat16_rn(x * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                                          (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();
    std::int32_t local_positions[3] = {positions[batch], positions[batch_size + batch],
                                       positions[2 * batch_size + batch]};
    auto* destination =
        query + static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim;
    store_rotated(normalized, local_positions, destination, dim);
}

__global__ void update_key_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ norm,
    const std::int32_t* __restrict__ token_indices, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ table_rows, const std::int32_t* __restrict__ source_slots,
    const std::int32_t* __restrict__ destination_slots, __nv_bfloat16* __restrict__ block_keys,
    const std::int32_t* __restrict__ block_tables, int logical_pages,
    __nv_bfloat16* __restrict__ raw_keys, std::int32_t* __restrict__ raw_positions,
    int batch_size, int batch_offset = 0) {
    __shared__ float warp_squares[4];
    __shared__ __nv_bfloat16 pooled[kHeadDim];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    const int dim                       = static_cast<int>(threadIdx.x);
    const int batch                     = static_cast<int>(blockIdx.x) + batch_offset;
    const int warp                      = dim >> 5;
    const int lane                      = dim & 31;
    const int source                    = source_slots[batch];
    const int destination               = destination_slots[batch];
    const int token_index               = token_indices[batch];
    const int tail                      = token_index & 3;
    const std::int64_t source_base      = static_cast<std::int64_t>(source) * 4 * kHeadDim;
    const std::int64_t destination_base = static_cast<std::int64_t>(destination) * 4 * kHeadDim;
    for (int slot = 0; slot < 4; ++slot) {
        raw_keys[destination_base + slot * kHeadDim + dim] =
            raw_keys[source_base + slot * kHeadDim + dim];
    }
    if (dim < 12) {
        raw_positions[static_cast<std::int64_t>(destination) * 12 + dim] =
            raw_positions[static_cast<std::int64_t>(source) * 12 + dim];
    }
    __syncthreads();
    if (token_index < 0) { return; }
    raw_keys[destination_base + tail * kHeadDim + dim] =
        projected[static_cast<std::int64_t>(batch) * kProjectionRows + kRawKeyOffset + dim];
    if (dim < 3) {
        raw_positions[static_cast<std::int64_t>(destination) * 12 + tail * 3 + dim] =
            positions[dim * batch_size + batch];
    }
    __syncthreads();
    if (tail != 3) { return; }

    float mean = 0.0F;
    for (int slot = 0; slot < 4; ++slot) {
        mean += __bfloat162float(raw_keys[destination_base + slot * kHeadDim + dim]);
    }
    pooled[dim]              = __float2bfloat16_rn(mean * 0.25F);
    const float pooled_float = __bfloat162float(pooled[dim]);
    const float square       = ops::warp_reduce_sum(pooled_float * pooled_float);
    if (lane == 0) { warp_squares[warp] = square; }
    __syncthreads();
    const float sum = warp_squares[0] + warp_squares[1] + warp_squares[2] + warp_squares[3];
    normalized[dim] =
        __float2bfloat16_rn(pooled_float * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                            (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();

    const int block         = token_index / 4;
    const int logical_page  = block / kCompressedPage;
    const int page_offset   = block % kCompressedPage;
    const int table_row     = table_rows[batch];
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    auto* destination_key   = block_keys +
                            static_cast<std::int64_t>(physical_page) * kCompressedPage * kHeadDim +
                            page_offset * kHeadDim;
    const std::int32_t* block_positions =
        raw_positions + static_cast<std::int64_t>(destination) * 12;
    store_rotated(normalized, block_positions, destination_key, dim);
}

__global__ void initialize_sort_kernel(std::int32_t* ids, std::int32_t* offsets, int active_blocks,
                                       int batch_size) {
    const int index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int items = active_blocks * batch_size;
    if (index < items) { ids[index] = index % active_blocks; }
    if (index <= batch_size) { offsets[index] = index * active_blocks; }
}

__global__ void score_blocks_kernel(const __nv_bfloat16* __restrict__ query,
                                    const __nv_bfloat16* __restrict__ block_keys,
                                    const std::int32_t* __restrict__ block_tables,
                                    const std::int32_t* __restrict__ table_rows,
                                    const std::int32_t* __restrict__ token_indices,
                                    int logical_pages, int active_blocks,
                                    float* __restrict__ scores) {
    __shared__ float head_scores[kQueryHeads];
    const int block                = static_cast<int>(blockIdx.x);
    const int batch                = static_cast<int>(blockIdx.y);
    const int tid                  = static_cast<int>(threadIdx.x);
    const int head                 = tid >> 5;
    const int lane                 = tid & 31;
    const int complete_blocks      = (token_indices[batch] + 1) / 4;
    const std::int64_t score_index = static_cast<std::int64_t>(batch) * active_blocks + block;
    // Padded slots in [complete_blocks, active_blocks) must be written, not skipped.
    // Stale NaN ranks highest under the G5 packed-key order and would enter the TopK set.
    if (block >= complete_blocks) {
        if (tid == 0) { scores[score_index] = -__int_as_float(0x7F800000); }
        return;
    }
    const int logical_page  = block / kCompressedPage;
    const int page_offset   = block % kCompressedPage;
    const int table_row     = table_rows[batch];
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    const auto* key         = block_keys +
                      static_cast<std::int64_t>(physical_page) * kCompressedPage * kHeadDim +
                      page_offset * kHeadDim;
    const auto* q =
        query + static_cast<std::int64_t>(batch) * kQueryHeads * kHeadDim + head * kHeadDim;
    float dot = 0.0F;
    for (int dim = lane; dim < kHeadDim; dim += 32) {
        dot = fmaf(__bfloat162float(q[dim]), __bfloat162float(key[dim]), dot);
    }
    dot = ops::warp_reduce_sum(dot);
    if (lane == 0) { head_scores[head] = dot; }
    __syncthreads();
    if (tid == 0) {
        float score = 0.0F;
        for (float value : head_scores) { score += fmaxf(value, 0.0F); }
        scores[score_index] = score * kIndexerScaling;
    }
}

// Exact fully-selected fast path: every complete block is kept, so the selection is the identity.
// Launched only when the host-known complete-block envelope is <= kSelectedBlocks.
__global__ void publish_identity_selection_kernel(const std::int32_t* __restrict__ token_indices,
                                                  std::int32_t* __restrict__ selected_blocks,
                                                  std::int32_t* __restrict__ selected_counts) {
    const int row             = static_cast<int>(blockIdx.x);
    const int tid             = static_cast<int>(threadIdx.x);
    const int complete_blocks = (token_indices[row] + 1) / 4;
    const int count           = min(complete_blocks, kSelectedBlocks);
    if (tid == 0) { selected_counts[row] = count; }
    for (int index = tid; index < kSelectedBlocks; index += static_cast<int>(blockDim.x)) {
        selected_blocks[static_cast<std::int64_t>(row) * kSelectedBlocks + index] =
            index < count ? index : -1;
    }
}

cudaError_t sort_pairs_descending(void* temp, std::size_t& temp_bytes, const float* keys_in,
                                  float* keys_out, const std::int32_t* values_in,
                                  std::int32_t* values_out, int items, int segments,
                                  const std::int32_t* offsets, cudaStream_t stream) {
    return cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp, temp_bytes, keys_in, keys_out, values_in, values_out, items, segments, offsets,
        offsets + 1, 0, static_cast<int>(sizeof(float) * 8), stream);
}

__device__ __forceinline__ std::uint32_t float_ascending_bits(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    const std::uint32_t mask =
        (static_cast<std::int32_t>(bits) < 0) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

__global__ void pack_topk_keys_kernel(const float* __restrict__ scores,
                                      const std::int32_t* __restrict__ ids,
                                      std::uint64_t* __restrict__ packed, int items) {
    const int index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (index >= items) { return; }
    const std::uint32_t rank = float_ascending_bits(scores[index]);
    const std::uint32_t tie  = ~static_cast<std::uint32_t>(ids[index]);
    packed[index]            = (static_cast<std::uint64_t>(rank) << 32) | tie;
}

__global__ void bitonic_sort_512_descending(std::uint64_t* keys, std::int32_t* ids) {
    __shared__ std::uint64_t skey[kSelectedBlocks];
    __shared__ std::int32_t sid[kSelectedBlocks];
    const int row = static_cast<int>(blockIdx.x);
    const int t   = static_cast<int>(threadIdx.x);
    keys += static_cast<std::int64_t>(row) * kSelectedBlocks;
    ids += static_cast<std::int64_t>(row) * kSelectedBlocks;
    skey[t] = keys[t];
    sid[t]  = ids[t];
    __syncthreads();
    for (int k = 2; k <= kSelectedBlocks; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = t ^ j;
            if (ixj > t) {
                const bool descending_half = (t & k) == 0;
                const bool out_of_order =
                    descending_half ? (skey[t] < skey[ixj]) : (skey[t] > skey[ixj]);
                if (out_of_order) {
                    const std::uint64_t tk = skey[t];
                    skey[t]                = skey[ixj];
                    skey[ixj]              = tk;
                    const std::int32_t ti  = sid[t];
                    sid[t]                 = sid[ixj];
                    sid[ixj]               = ti;
                }
            }
            __syncthreads();
        }
    }
    keys[t] = skey[t];
    ids[t]  = sid[t];
}

__global__ void publish_compact_selection_kernel(const std::int32_t* __restrict__ selected_ids,
                                                 const std::int32_t* __restrict__ token_indices,
                                                 std::int32_t* __restrict__ selected_blocks,
                                                 std::int32_t* __restrict__ selected_counts,
                                                 int id_stride) {
    const int row             = static_cast<int>(blockIdx.x);
    const int tid             = static_cast<int>(threadIdx.x);
    const int complete_blocks = (token_indices[row] + 1) / 4;
    const int count           = min(complete_blocks, kSelectedBlocks);
    if (tid == 0) { selected_counts[row] = count; }
    for (int index = tid; index < kSelectedBlocks; index += static_cast<int>(blockDim.x)) {
        selected_blocks[static_cast<std::int64_t>(row) * kSelectedBlocks + index] =
            index < count ? selected_ids[static_cast<std::int64_t>(row) * id_stride + index] : -1;
    }
}

auto topk_env(cudaStream_t stream) {
    return cuda::std::execution::env(
        cuda::stream_ref{stream},
        cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                 cuda::execution::output_ordering::unsorted));
}

cudaError_t topk_max_pairs(void* temp, std::size_t& temp_bytes, const std::uint64_t* keys_in,
                           std::uint64_t* keys_out, const std::int32_t* values_in,
                           std::int32_t* values_out, int num_items, int k, cudaStream_t stream) {
    return cub::DeviceTopK::MaxPairs(temp, temp_bytes, keys_in, keys_out, values_in, values_out,
                                     num_items, k, topk_env(stream));
}

std::size_t topk_temp_bytes(int num_items, int k) {
    std::size_t bytes = 0;
    const auto* keys_in =
        reinterpret_cast<const std::uint64_t*>(std::uintptr_t{0x1000});
    auto* keys_out = reinterpret_cast<std::uint64_t*>(std::uintptr_t{0x2000});
    const auto* values_in =
        reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x3000});
    auto* values_out = reinterpret_cast<std::int32_t*>(std::uintptr_t{0x4000});
    CUDA_CHECK(topk_max_pairs(nullptr, bytes, keys_in, keys_out, values_in, values_out, num_items,
                              k, nullptr));
    return bytes;
}

void select_top512_topk(const float* scores, const std::int32_t* ids, std::uint64_t* packed_keys,
                        std::uint64_t* packed_selected, std::int32_t* topk_ids, void* temp,
                        std::size_t temp_capacity, int active_blocks, int batch,
                        cudaStream_t stream) {
    constexpr int threads = 256;
    const int items       = active_blocks * batch;
    pack_topk_keys_kernel<<<(items + threads - 1) / threads, threads, 0, stream>>>(
        scores, ids, packed_keys, items);
    CUDA_CHECK(cudaGetLastError());
    for (int row = 0; row < batch; ++row) {
        const std::int64_t packed_row = static_cast<std::int64_t>(row) * active_blocks;
        const std::int64_t topk_row   = static_cast<std::int64_t>(row) * kSelectedBlocks;
        std::size_t temp_bytes        = temp_capacity;
        CUDA_CHECK(topk_max_pairs(temp, temp_bytes, packed_keys + packed_row,
                                  packed_selected + topk_row, ids + packed_row, topk_ids + topk_row,
                                  active_blocks, kSelectedBlocks, stream));
    }
    bitonic_sort_512_descending<<<batch, kSelectedBlocks, 0, stream>>>(packed_selected, topk_ids);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t flash_next_qsa_indexer_sort_temp_bytes(std::int32_t maximum_blocks,
                                                   std::int32_t batch) {
    std::size_t sort_bytes = 0;
    const auto* keys_in    = reinterpret_cast<const float*>(std::uintptr_t{0x1000});
    auto* keys_out         = reinterpret_cast<float*>(std::uintptr_t{0x2000});
    const auto* values_in  = reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x3000});
    auto* values_out       = reinterpret_cast<std::int32_t*>(std::uintptr_t{0x4000});
    const auto* offsets    = reinterpret_cast<const std::int32_t*>(std::uintptr_t{0x5000});
    CUDA_CHECK(sort_pairs_descending(nullptr, sort_bytes, keys_in, keys_out, values_in, values_out,
                                     maximum_blocks * batch, batch, offsets, nullptr));
    const std::size_t topk_bytes = topk_temp_bytes(maximum_blocks, kSelectedBlocks);
    return std::max(sort_bytes, topk_bytes);
}

void flash_next_qsa_indexer_store_launch(const Tensor& projected, const Tensor& token_indices,
    const Tensor& mrope_positions, const Tensor& table_rows, const Tensor& source_state_slots,
    const Tensor& destination_state_slots, const Tensor& key_norm, QsaIndexerCacheView cache,
    cudaStream_t stream, bool aliased_recurrent_scan) {
    const int batch = token_indices.ne[0];
    if (aliased_recurrent_scan && batch > 1) {
        for (int r = 0; r < batch; ++r) {
            update_key_kernel<<<1, kHeadDim, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(projected.data),
                static_cast<const __nv_bfloat16*>(key_norm.data),
                static_cast<const std::int32_t*>(token_indices.data),
                static_cast<const std::int32_t*>(mrope_positions.data),
                static_cast<const std::int32_t*>(table_rows.data),
                static_cast<const std::int32_t*>(source_state_slots.data),
                static_cast<const std::int32_t*>(destination_state_slots.data),
                static_cast<__nv_bfloat16*>(cache.block_keys.data),
                static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
                static_cast<__nv_bfloat16*>(cache.raw_keys.data),
                static_cast<std::int32_t*>(cache.raw_positions.data), batch, r);
        }
    } else {
        update_key_kernel<<<batch, kHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(key_norm.data),
            static_cast<const std::int32_t*>(token_indices.data),
            static_cast<const std::int32_t*>(mrope_positions.data),
            static_cast<const std::int32_t*>(table_rows.data),
            static_cast<const std::int32_t*>(source_state_slots.data),
            static_cast<const std::int32_t*>(destination_state_slots.data),
            static_cast<__nv_bfloat16*>(cache.block_keys.data),
            static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
            static_cast<__nv_bfloat16*>(cache.raw_keys.data),
            static_cast<std::int32_t*>(cache.raw_positions.data), batch, 0);
    }
    CUDA_CHECK(cudaGetLastError());

}

void flash_next_qsa_indexer_launch(const Tensor& token_indices, const Tensor& mrope_positions,
                                   const Tensor& table_rows, const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, const Tensor& query_norm,
                                   const Tensor& key_norm, QsaIndexerCacheView cache,
                                   FlashNextQsaIndexerWorkspace& scratch, int active_blocks,
                                   Tensor& selected_blocks, Tensor& selected_counts,
                                   cudaStream_t stream, bool aliased_recurrent_scan) {
    const int batch = token_indices.ne[0];
    flash_next_qsa_indexer_store_launch(scratch.projected, token_indices, mrope_positions,
        table_rows, source_state_slots, destination_state_slots, key_norm, cache, stream,
        aliased_recurrent_scan);
    if (active_blocks == 0) {
        CUDA_CHECK(cudaMemsetAsync(selected_counts.data, 0,
                                   static_cast<std::size_t>(batch) * sizeof(std::int32_t), stream));
        return;
    }

    if (active_blocks <= kSelectedBlocks) {
        publish_identity_selection_kernel<<<batch, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(token_indices.data),
            static_cast<std::int32_t*>(selected_blocks.data),
            static_cast<std::int32_t*>(selected_counts.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    prepare_query_kernel<<<dim3(kQueryHeads, batch), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data), batch);
    CUDA_CHECK(cudaGetLastError());
    constexpr int threads = 256;
    const int items       = active_blocks * batch;
    initialize_sort_kernel<<<(items + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<std::int32_t*>(scratch.ids.data),
        static_cast<std::int32_t*>(scratch.offsets.data), active_blocks, batch);
    CUDA_CHECK(cudaGetLastError());
    score_blocks_kernel<<<dim3(active_blocks, batch), kQueryHeads * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.query.data),
        static_cast<const __nv_bfloat16*>(cache.block_keys.data),
        static_cast<const std::int32_t*>(cache.block_tables.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<const std::int32_t*>(token_indices.data), cache.block_tables.ne[0],
        active_blocks, static_cast<float*>(scratch.scores.data));
    CUDA_CHECK(cudaGetLastError());
    select_top512_topk(
        static_cast<const float*>(scratch.scores.data),
        static_cast<const std::int32_t*>(scratch.ids.data),
        static_cast<std::uint64_t*>(scratch.packed_keys.data),
        static_cast<std::uint64_t*>(scratch.packed_selected.data),
        static_cast<std::int32_t*>(scratch.topk_ids.data), scratch.sort_temp.data,
        scratch.sort_temp.bytes, active_blocks, batch, stream);
    publish_compact_selection_kernel<<<batch, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(scratch.topk_ids.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<std::int32_t*>(selected_blocks.data),
        static_cast<std::int32_t*>(selected_counts.data), kSelectedBlocks);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void indexer_publish_complete_blocks_chunk_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ norm,
    const std::int32_t* __restrict__ token_indices, const std::int32_t* __restrict__ positions,
    int source_slot, int destination_slot, int table_row,
    const std::int32_t* __restrict__ block_tables, int logical_pages,
    __nv_bfloat16* __restrict__ block_keys, const __nv_bfloat16* __restrict__ raw_keys,
    const std::int32_t* __restrict__ raw_positions, int tokens) {
    __shared__ float warp_squares[4];
    __shared__ __nv_bfloat16 pooled[kHeadDim];
    __shared__ __nv_bfloat16 normalized[kHeadDim];
    __shared__ std::int32_t first_token_pos[3];

    const int dim               = static_cast<int>(threadIdx.x);
    const int b                 = static_cast<int>(blockIdx.x);
    const int warp              = dim >> 5;
    const int lane              = dim & 31;
    const int first_token_index = token_indices[0];
    const int leftover_in       = first_token_index & 3;
    const int complete_blocks   = (leftover_in + tokens) / 4;
    if (b >= complete_blocks) { return; }

    const std::int64_t source_base = static_cast<std::int64_t>(source_slot) * 4 * kHeadDim;

    // First token of block b: s0 = 4 * b + 0
    if (dim < 3) {
        const int s0 = 4 * b;
        if (s0 < leftover_in) {
            first_token_pos[dim] =
                raw_positions[static_cast<std::int64_t>(source_slot) * 12 + s0 * 3 + dim];
        } else {
            const int t0         = s0 - leftover_in;
            first_token_pos[dim] = positions[dim * tokens + t0];
        }
    }
    __syncthreads();

    // 4 tokens of block b: s = 4 * b + slot (slot = 0, 1, 2, 3)
    float sum_raw = 0.0F;
    for (int slot = 0; slot < 4; ++slot) {
        const int s = 4 * b + slot;
        float val   = 0.0F;
        if (s < leftover_in) {
            val = __bfloat162float(raw_keys[source_base + slot * kHeadDim + dim]);
        } else {
            const int t = s - leftover_in;
            val         = __bfloat162float(
                projected[static_cast<std::int64_t>(t) * kProjectionRows + kRawKeyOffset + dim]);
        }
        sum_raw += val;
    }

    pooled[dim]              = __float2bfloat16_rn(sum_raw * 0.25F);
    const float pooled_float = __bfloat162float(pooled[dim]);
    const float square       = ops::warp_reduce_sum(pooled_float * pooled_float);
    if (lane == 0) { warp_squares[warp] = square; }
    __syncthreads();
    const float sum = warp_squares[0] + warp_squares[1] + warp_squares[2] + warp_squares[3];
    normalized[dim] =
        __float2bfloat16_rn(pooled_float * rsqrtf(sum / static_cast<float>(kHeadDim) + 1.0e-6F) *
                            (1.0F + __bfloat162float(norm[dim])));
    __syncthreads();

    const int global_block  = (first_token_index - leftover_in) / 4 + b;
    const int logical_page  = global_block / kCompressedPage;
    const int page_offset   = global_block % kCompressedPage;
    const int physical_page = block_tables[table_row * logical_pages + logical_page];
    auto* destination_key   = block_keys +
                            static_cast<std::int64_t>(physical_page) * kCompressedPage * kHeadDim +
                            page_offset * kHeadDim;
    store_rotated(normalized, first_token_pos, destination_key, dim);
}

__global__ void indexer_update_leftover_chunk_kernel(
    const __nv_bfloat16* __restrict__ projected, const std::int32_t* __restrict__ token_indices,
    const std::int32_t* __restrict__ positions, int source_slot, int destination_slot,
    __nv_bfloat16* __restrict__ raw_keys, std::int32_t* __restrict__ raw_positions, int tokens) {
    const int dim               = static_cast<int>(threadIdx.x);
    const int first_token_index = token_indices[0];
    const int leftover_in       = first_token_index & 3;
    const int complete_blocks   = (leftover_in + tokens) / 4;
    const int leftover_out      = (leftover_in + tokens) & 3;

    const std::int64_t source_base      = static_cast<std::int64_t>(source_slot) * 4 * kHeadDim;
    const std::int64_t destination_base = static_cast<std::int64_t>(destination_slot) * 4 * kHeadDim;

    for (int slot = 0; slot < leftover_out; ++slot) {
        const int s = complete_blocks * 4 + slot;
        if (s < leftover_in) {
            raw_keys[destination_base + slot * kHeadDim + dim] =
                raw_keys[source_base + s * kHeadDim + dim];
            if (dim < 3) {
                raw_positions[static_cast<std::int64_t>(destination_slot) * 12 + slot * 3 + dim] =
                    raw_positions[static_cast<std::int64_t>(source_slot) * 12 + s * 3 + dim];
            }
        } else {
            const int t = s - leftover_in;
            raw_keys[destination_base + slot * kHeadDim + dim] =
                projected[static_cast<std::int64_t>(t) * kProjectionRows + kRawKeyOffset + dim];
            if (dim < 3) {
                raw_positions[static_cast<std::int64_t>(destination_slot) * 12 + slot * 3 + dim] =
                    positions[dim * tokens + t];
            }
        }
    }
}

// Prefill scoring GEMM. Reuses sparse_moe_prefill_router_mma_kernel mechanics
// (gemm_swz64, ldmatrix, mma_bf16 m16n8k16, cp_async_zfill). A CTA owns a
// (BM x BN) region of scores, stages BM paged block-keys once, and reuses them
// across BN tokens and 4 query heads. Decode keeps the per-(block,token) kernel.
constexpr int kScoreBM      = 64;
constexpr int kScoreBN      = 64;
constexpr int kScoreWarps   = 4;
constexpr int kScoreThreads = 32 * kScoreWarps;
constexpr int kScoreNTiles  = kScoreBN / 8;

__global__ __launch_bounds__(kScoreThreads)
void score_blocks_chunk_kernel(const __nv_bfloat16* __restrict__ query,
                               const __nv_bfloat16* __restrict__ block_keys,
                               const std::int32_t* __restrict__ block_tables,
                               int table_row,
                               const std::int32_t* __restrict__ token_indices,
                               int logical_pages, int active_blocks, int tokens,
                               float* __restrict__ scores) {
    using ops::cp_async_zfill;
    using ops::cp_commit;
    using ops::cp_wait;
    using ops::ldmatrix_x2;
    using ops::ldmatrix_x4;
    using ops::mma_bf16;
    using ops::smem_addr;
    using ops::detail::gemm_swz64;
    using Cache = ops::Cache;

    __shared__ __align__(16) __nv_bfloat16 Ks[kScoreBM * kHeadDim];
    __shared__ __align__(16) __nv_bfloat16 Qs[kScoreBN * kHeadDim];
    __shared__ std::int32_t complete_s[kScoreBN];

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int block0  = static_cast<int>(blockIdx.x) * kScoreBM;
    const int token0  = static_cast<int>(blockIdx.y) * kScoreBN;

    if (tid < kScoreBN) {
        const int t = token0 + tid;
        complete_s[tid] =
            (t < tokens) ? (token_indices[t] + 1) / 4 : 0;
    }

    for (int item = tid; item < kScoreBM * (kHeadDim / 8); item += kScoreThreads) {
        const int row = item / (kHeadDim / 8);
        const int k8  = item - row * (kHeadDim / 8);
        const int blk = block0 + row;
        auto* dst     = &Ks[row * kHeadDim + gemm_swz64(row, k8 * 8)];
        const bool valid = blk >= 0 && blk < active_blocks;
        const __nv_bfloat16* src = block_keys;
        if (valid) {
            const int logical_page  = blk / kCompressedPage;
            const int page_offset   = blk % kCompressedPage;
            const int physical_page = block_tables[table_row * logical_pages + logical_page];
            src = block_keys +
                  static_cast<std::int64_t>(physical_page) * kCompressedPage * kHeadDim +
                  page_offset * kHeadDim + k8 * 8;
        }
        cp_async_zfill<16, Cache::cg>(dst, src, valid ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    float score[kScoreNTiles][4] = {};
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int arow     = warp * 16 + a_rowoff;

    for (int head = 0; head < kQueryHeads; ++head) {
        for (int item = tid; item < kScoreBN * (kHeadDim / 8); item += kScoreThreads) {
            const int col = item / (kHeadDim / 8);
            const int k8  = item - col * (kHeadDim / 8);
            const int t   = token0 + col;
            auto* dst     = &Qs[col * kHeadDim + gemm_swz64(col, k8 * 8)];
            const bool valid = t >= 0 && t < tokens;
            const __nv_bfloat16* src =
                query + static_cast<std::int64_t>(valid ? t : 0) * kQueryHeads * kHeadDim +
                head * kHeadDim + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, valid ? 16 : 0);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();

        float acc[kScoreNTiles][4] = {};
#pragma unroll
        for (int ki = 0; ki < kHeadDim / 16; ++ki) {
            unsigned af[4];
            ldmatrix_x4(af[0], af[1], af[2], af[3],
                        smem_addr(&Ks[arow * kHeadDim + gemm_swz64(arow, ki * 16 + a_coloff)]));
#pragma unroll
            for (int nt = 0; nt < kScoreNTiles; ++nt) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&Qs[brow * kHeadDim + gemm_swz64(brow, ki * 16 + b_koff)]));
                mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], af[0], af[1], af[2],
                         af[3], bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < kScoreNTiles; ++nt) {
            score[nt][0] += fmaxf(acc[nt][0], 0.0F);
            score[nt][1] += fmaxf(acc[nt][1], 0.0F);
            score[nt][2] += fmaxf(acc[nt][2], 0.0F);
            score[nt][3] += fmaxf(acc[nt][3], 0.0F);
        }
        __syncthreads();
    }

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int r0  = block0 + warp * 16 + gid;
    const int r1  = r0 + 8;
    const float ninf = -__int_as_float(0x7F800000);
#pragma unroll
    for (int nt = 0; nt < kScoreNTiles; ++nt) {
        const int c0 = token0 + nt * 8 + 2 * lid;
        const int c1 = c0 + 1;
        auto store = [&](int blk, int tok, float value) {
            if (tok < 0 || tok >= tokens || blk < 0 || blk >= active_blocks) { return; }
            const int complete = complete_s[tok - token0];
            scores[static_cast<std::int64_t>(tok) * active_blocks + blk] =
                (blk < complete) ? value * kIndexerScaling : ninf;
        };
        store(r0, c0, score[nt][0]);
        store(r0, c1, score[nt][1]);
        store(r1, c0, score[nt][2]);
        store(r1, c1, score[nt][3]);
    }
}

void flash_next_qsa_indexer_store_prefill_launch(const Tensor& projected,
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    std::int32_t source_state_slot, std::int32_t destination_state_slot,
    const Tensor& key_norm, QsaIndexerCacheView cache, cudaStream_t stream) {
    const int tokens = token_indices.ne[0];
    const int max_complete_blocks = (tokens + 3) / 4;
    if (max_complete_blocks > 0) {
        indexer_publish_complete_blocks_chunk_kernel<<<max_complete_blocks, kHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(key_norm.data),
            static_cast<const std::int32_t*>(token_indices.data),
            static_cast<const std::int32_t*>(mrope_positions.data), source_state_slot,
            destination_state_slot, table_row,
            static_cast<const std::int32_t*>(cache.block_tables.data), cache.block_tables.ne[0],
            static_cast<__nv_bfloat16*>(cache.block_keys.data),
            static_cast<const __nv_bfloat16*>(cache.raw_keys.data),
            static_cast<const std::int32_t*>(cache.raw_positions.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }

    indexer_update_leftover_chunk_kernel<<<1, kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<const std::int32_t*>(mrope_positions.data), source_state_slot,
        destination_state_slot, static_cast<__nv_bfloat16*>(cache.raw_keys.data),
        static_cast<std::int32_t*>(cache.raw_positions.data), tokens);
    CUDA_CHECK(cudaGetLastError());

}

void flash_next_qsa_indexer_prefill_launch(
    const Tensor& token_indices, const Tensor& mrope_positions, std::int32_t table_row,
    std::int32_t source_state_slot, std::int32_t destination_state_slot, const Tensor& query_norm,
    const Tensor& key_norm, QsaIndexerCacheView cache, FlashNextQsaIndexerWorkspace& scratch,
    std::int32_t maximum_blocks, std::int32_t first_token_index, Tensor& selected_blocks,
    Tensor& selected_counts, cudaStream_t stream) {
    const int tokens = token_indices.ne[0];
    flash_next_qsa_indexer_store_prefill_launch(scratch.projected, token_indices,
        mrope_positions, table_row, source_state_slot, destination_state_slot, key_norm,
        cache, stream);
    std::int32_t resolved_first = first_token_index;
    const char* restore         = std::getenv("NINFER_FLASH_NEXT_PREFILL_HOST_SYNC");
    const bool host_sync = restore != nullptr && restore[0] == '1' && restore[1] == '\0';
    if (host_sync) {
        CUDA_CHECK(cudaMemcpyAsync(&resolved_first, token_indices.data, sizeof(resolved_first),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    const int chunk_complete_blocks = (resolved_first + tokens) / 4;
    if (maximum_blocks <= kSelectedBlocks || chunk_complete_blocks <= kSelectedBlocks) {
        publish_identity_selection_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(token_indices.data),
            static_cast<std::int32_t*>(selected_blocks.data),
            static_cast<std::int32_t*>(selected_counts.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    prepare_query_kernel<<<dim3(kQueryHeads, tokens), kHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const std::int32_t*>(mrope_positions.data),
        static_cast<__nv_bfloat16*>(scratch.query.data), tokens);
    CUDA_CHECK(cudaGetLastError());

    const int tile_size = flash_next_qsa_indexer_tile_size(maximum_blocks, tokens);

    for (int t_start = 0; t_start < tokens; t_start += tile_size) {
        const int current_tile = min(tile_size, tokens - t_start);
        const int tile_last    = first_token_index + t_start + current_tile - 1;
        const int tile_complete = (tile_last + 1) / 4;
        auto* tile_token_indices =
            static_cast<const std::int32_t*>(token_indices.data) + t_start;
        auto* tile_selected = static_cast<std::int32_t*>(selected_blocks.data) +
                              static_cast<std::int64_t>(t_start) * kSelectedBlocks;
        auto* tile_counts = static_cast<std::int32_t*>(selected_counts.data) + t_start;
        if (tile_complete <= kSelectedBlocks) {
            publish_identity_selection_kernel<<<current_tile, 256, 0, stream>>>(
                tile_token_indices, tile_selected, tile_counts);
            CUDA_CHECK(cudaGetLastError());
            continue;
        }

        // Eager prefill is not captured: score/sort the live complete-block frontier, not the
        // plan envelope. Workspace remains sized to maximum_blocks.
        const int active_blocks = min(maximum_blocks, tile_complete);
        constexpr int threads   = 256;
        const int items         = active_blocks * current_tile;

        initialize_sort_kernel<<<(items + threads - 1) / threads, threads, 0, stream>>>(
            static_cast<std::int32_t*>(scratch.ids.data),
            static_cast<std::int32_t*>(scratch.offsets.data), active_blocks, current_tile);
        CUDA_CHECK(cudaGetLastError());

        const dim3 score_grid((active_blocks + kScoreBM - 1) / kScoreBM,
                              (current_tile + kScoreBN - 1) / kScoreBN);
        score_blocks_chunk_kernel<<<score_grid, kScoreThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.query.data) +
                static_cast<std::int64_t>(t_start) * kQueryHeads * kHeadDim,
            static_cast<const __nv_bfloat16*>(cache.block_keys.data),
            static_cast<const std::int32_t*>(cache.block_tables.data), table_row,
            tile_token_indices, cache.block_tables.ne[0], active_blocks, current_tile,
            static_cast<float*>(scratch.scores.data));
        CUDA_CHECK(cudaGetLastError());

        std::size_t temp_bytes = scratch.sort_temp.bytes;
        CUDA_CHECK(sort_pairs_descending(
            scratch.sort_temp.data, temp_bytes, static_cast<const float*>(scratch.scores.data),
            static_cast<float*>(scratch.sorted_scores.data),
            static_cast<const std::int32_t*>(scratch.ids.data),
            static_cast<std::int32_t*>(scratch.sorted_ids.data), items, current_tile,
            static_cast<const std::int32_t*>(scratch.offsets.data), stream));

        publish_compact_selection_kernel<<<current_tile, 256, 0, stream>>>(
            static_cast<const std::int32_t*>(scratch.sorted_ids.data), tile_token_indices,
            tile_selected, tile_counts, active_blocks);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
