#include "models/qwen3_8_flash_next/impl/mtp_forward_kernels.h"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

__global__ void mtp_stem_add_embedding_kernel(
    const __nv_bfloat16* __restrict__ emb_proj,
    const __nv_bfloat16* __restrict__ hid_proj,
    __nv_bfloat16* __restrict__ mtp_hyper_hidden,
    int batch) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = 10240 * batch;
    if (idx >= total) { return; }
    const int token = idx / 10240;
    const int col = idx % 2560;

    const float e = __bfloat162float(emb_proj[token * 2560 + col]);
    const float h = __bfloat162float(hid_proj[idx]);
    const float sum = e + h;
    const __nv_bfloat16 sum_bf16 = __float2bfloat16_rn(sum);

    mtp_hyper_hidden[idx] = sum_bf16;
}

} // namespace

__global__ void mtp_shift_inputs_kernel(const __nv_bfloat16* hidden,
    const std::int32_t* indices, const std::int32_t* positions,
    const __nv_bfloat16* saved_hidden, const std::int32_t* saved_positions,
    const std::int32_t* source_slots, int source_slot, bool chain, int offset,
    __nv_bfloat16* shifted_hidden, std::int32_t* shifted_indices,
    std::int32_t* shifted_positions, int tokens, int rows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 10240 * rows) { return; }
    const int row = idx / 10240;
    const int dim = idx % 10240;
    const int current = row + offset;
    const int previous = chain ? current - 1 : -1;
    const int slot = source_slots ? source_slots[current] : source_slot;
    shifted_hidden[idx] = previous >= 0 ? hidden[previous * 10240 + dim]
                                         : saved_hidden[slot * 10240 + dim];
    if (dim == 0) { shifted_indices[row] = indices[current] - 1; }
    if (dim < 3) {
        shifted_positions[dim * rows + row] = previous >= 0
            ? positions[dim * tokens + previous] : saved_positions[slot * 3 + dim];
    }
}

__global__ void mtp_save_target_kernel(const __nv_bfloat16* hidden,
    const std::int32_t* positions, const std::int32_t* destination_slots, int destination_slot,
    __nv_bfloat16* saved_hidden, std::int32_t* saved_positions, int tokens, int rows) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 10240 * rows) { return; }
    const int row = destination_slots ? idx / 10240 : tokens - 1;
    const int dim = idx % 10240;
    const int slot = destination_slots ? destination_slots[row] : destination_slot;
    saved_hidden[slot * 10240 + dim] = hidden[row * 10240 + dim];
    if (dim < 3) { saved_positions[slot * 3 + dim] = positions[dim * tokens + row]; }
}

void flash_next_mtp_shift_inputs_launch(const Tensor& target_hidden,
    const Tensor& token_indices, const Tensor& positions, const Tensor& saved_hidden,
    const Tensor& saved_positions, const Tensor& source_slots, int source_slot,
    bool chain, int offset, Tensor& shifted_hidden, Tensor& shifted_indices,
    Tensor& shifted_positions, cudaStream_t stream) {
    const int rows = shifted_hidden.ne[1];
    mtp_shift_inputs_kernel<<<(10240 * rows + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(target_hidden.data),
        static_cast<const std::int32_t*>(token_indices.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const __nv_bfloat16*>(saved_hidden.data),
        static_cast<const std::int32_t*>(saved_positions.data),
        static_cast<const std::int32_t*>(source_slots.data), source_slot, chain, offset,
        static_cast<__nv_bfloat16*>(shifted_hidden.data),
        static_cast<std::int32_t*>(shifted_indices.data),
        static_cast<std::int32_t*>(shifted_positions.data), target_hidden.ne[1], rows);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_mtp_save_target_launch(const Tensor& target_hidden, const Tensor& positions,
    const Tensor& destination_slots, int destination_slot, Tensor& saved_hidden,
    Tensor& saved_positions, cudaStream_t stream) {
    const int rows = destination_slots.data ? target_hidden.ne[1] : 1;
    mtp_save_target_kernel<<<(10240 * rows + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(target_hidden.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(destination_slots.data), destination_slot,
        static_cast<__nv_bfloat16*>(saved_hidden.data),
        static_cast<std::int32_t*>(saved_positions.data), target_hidden.ne[1], rows);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_mtp_stem_add_embedding_launch(const Tensor& emb_proj, const Tensor& hid_proj,
                                             Tensor& mtp_hyper_hidden, cudaStream_t stream) {
    const int batch = static_cast<int>(emb_proj.ne[1]);
    const int total = 10240 * batch;
    const int block = 256;
    const int grid = (total + block - 1) / block;

    mtp_stem_add_embedding_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(emb_proj.data),
        static_cast<const __nv_bfloat16*>(hid_proj.data),
        static_cast<__nv_bfloat16*>(mtp_hyper_hidden.data),
        batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
