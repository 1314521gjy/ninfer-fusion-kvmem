#include "models/qwen3_8_flash_next/impl/moe_shared_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

constexpr int kExperts      = 512;
constexpr int kTopK         = 10;
constexpr int kPaths        = 11;
constexpr int kHidden       = 2'560;
constexpr int kIntermediate = 640;
constexpr int kScoreRows    = kExperts + 1;

using RouterGeom  = ops::detail::Bf16GemvGeometry<kExperts, kHidden>;
using GateGeom    = ops::detail::Bf16GemvGeometry<kIntermediate, kHidden>;
using DownGeom    = ops::detail::Bf16GemvGeometry<kHidden, kIntermediate>;
using RouterSched = ops::detail::Bf16MmaProductionSchedule<RouterGeom>;
using GateSched   = ops::detail::Bf16MmaProductionSchedule<GateGeom>;
using DownSched   = ops::detail::Bf16MmaProductionSchedule<DownGeom>;

struct Fp32ScoreTile {
    float* data;
    std::int32_t leading_dim;
    std::int32_t parent_row_begin;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * leading_dim + parent_row] = value;
        (void)parent_row_begin;
    }
};

struct Fp32ScoreOutput {
    float* data;
    std::int32_t leading_dim;

    __device__ __forceinline__ Fp32ScoreTile tile(std::int32_t) const {
        return {data, leading_dim, 0};
    }
};

struct ScaledBf16Tile {
    __nv_bfloat16* data;
    const float* scale;
    std::int32_t leading_dim;
    std::int32_t parent_row_begin;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const float s = scale[token];
        data[static_cast<std::int64_t>(token) * leading_dim + parent_row] =
            __float2bfloat16_rn(value * s);
        (void)parent_row_begin;
    }
};

struct ScaledBf16Output {
    __nv_bfloat16* data;
    const float* scale;
    std::int32_t leading_dim;

    __device__ __forceinline__ ScaledBf16Tile tile(std::int32_t) const {
        return {data, scale, leading_dim, 0};
    }
};

__global__ void shared_gate_score_row_kernel(const __nv_bfloat16* __restrict__ input,
                                             const __nv_bfloat16* __restrict__ shared_gate,
                                             float* __restrict__ scores, int tokens) {
    __shared__ float partial[8];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    if (token >= tokens) { return; }
    const __nv_bfloat16* in_tok = input + static_cast<std::int64_t>(token) * kHidden;
    float acc                   = 0.0F;
    for (int column = tid; column < kHidden; column += static_cast<int>(blockDim.x)) {
        acc = fmaf(__bfloat162float(shared_gate[column]), __bfloat162float(in_tok[column]), acc);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFF'FFFFU, acc, offset);
    }
    if (lane == 0) { partial[warp] = acc; }
    __syncthreads();
    if (warp == 0) {
        float warp_sum = (lane < 8) ? partial[lane] : 0.0F;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xFFFF'FFFFU, warp_sum, offset);
        }
        if (lane == 0) {
            scores[static_cast<std::int64_t>(token) * kScoreRows + kExperts] = warp_sum;
        }
    }
}

__global__ void silu_mul_shared_path_kernel(const __nv_bfloat16* __restrict__ gate,
                                            __nv_bfloat16* __restrict__ packed_up,
                                            __nv_bfloat16* __restrict__ strided_path10,
                                            int tokens) {
    const int n   = tokens * kIntermediate;
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (tid >= n) { return; }
    const int token = tid / kIntermediate;
    const int pair  = tid - token * kIntermediate;
    const float g   = __bfloat162float(gate[tid]);
    const float u   = __bfloat162float(packed_up[tid]);
    const auto out  = __float2bfloat16_rn(ops::silu(g) * u);
    packed_up[tid]  = out;
    strided_path10[static_cast<std::int64_t>(token) * kPaths * kIntermediate + pair] = out;
}

__global__ void moe_zero_i32_kernel(std::int32_t* __restrict__ data, int n) {
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (tid < n) { data[tid] = 0; }
}

__global__ void moe_histogram_kernel(const std::int32_t* __restrict__ ids,
                                     std::int32_t* __restrict__ counts, int total_items) {
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (tid >= total_items) { return; }
    const int expert = ids[tid];
    if (expert >= 0 && expert < kExperts) { atomicAdd(counts + expert, 1); }
}

__global__ void moe_prefix_compact_kernel(const std::int32_t* __restrict__ counts,
                                          std::int32_t* __restrict__ offsets,
                                          std::int32_t* __restrict__ active_experts,
                                          std::int32_t* __restrict__ active_count_ptr,
                                          std::int32_t* __restrict__ task_counter) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid >= kExperts) { return; }
    const int count = counts[tid];
    const int lane  = tid & 31;
    const int warp  = tid >> 5;

    int warp_sum = count;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int val = __shfl_up_sync(0xffffffff, warp_sum, offset);
        if (lane >= offset) { warp_sum += val; }
    }
    __shared__ int s_warp_totals[16];
    if (lane == 31) { s_warp_totals[warp] = warp_sum; }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        int w_sum = s_warp_totals[lane];
#pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const int val = __shfl_up_sync(0xffff, w_sum, offset);
            if (lane >= offset) { w_sum += val; }
        }
        s_warp_totals[lane] = w_sum;
    }
    __syncthreads();
    const int base_offset = (warp > 0) ? s_warp_totals[warp - 1] : 0;
    const int exc_offset  = base_offset + warp_sum - count;
    offsets[tid]          = exc_offset;
    if (tid == 511) { offsets[512] = exc_offset + count; }

    const int is_active = (count > 0) ? 1 : 0;
    int act_warp_sum    = is_active;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int val = __shfl_up_sync(0xffffffff, act_warp_sum, offset);
        if (lane >= offset) { act_warp_sum += val; }
    }
    __shared__ int s_act_totals[16];
    if (lane == 31) { s_act_totals[warp] = act_warp_sum; }
    __syncthreads();
    if (warp == 0 && lane < 16) {
        int w_sum = s_act_totals[lane];
#pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const int val = __shfl_up_sync(0xffff, w_sum, offset);
            if (lane >= offset) { w_sum += val; }
        }
        s_act_totals[lane] = w_sum;
    }
    __syncthreads();
    const int act_base = (warp > 0) ? s_act_totals[warp - 1] : 0;
    const int act_exc  = act_base + act_warp_sum - is_active;
    if (is_active) { active_experts[act_exc] = tid; }
    if (tid == 511) {
        active_count_ptr[0] = act_exc + is_active;
        task_counter[0]     = 0;
        task_counter[1]     = 0;
        task_counter[2]     = 0;
        task_counter[3]     = 0;
    }
}

__global__ void moe_scatter_groups_kernel(const std::int32_t* __restrict__ ids,
                                          const std::int32_t* __restrict__ offsets,
                                          std::int32_t* __restrict__ grouped_tokens,
                                          std::int32_t* __restrict__ grouped_paths,
                                          std::int32_t* __restrict__ grouped_experts,
                                          std::int32_t* __restrict__ token_to_pos, int total_items) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid >= kExperts) { return; }
    __shared__ int s_heads[512];
    s_heads[tid] = offsets[tid];
    __syncthreads();
    for (int i = tid; i < total_items; i += kExperts) {
        const int expert         = ids[i];
        const int pos            = atomicAdd(&s_heads[expert], 1);
        grouped_tokens[pos]      = i / kTopK;
        grouped_paths[pos]       = i % kTopK;
        grouped_experts[pos]     = expert;
        if (token_to_pos != nullptr) { token_to_pos[i] = pos; }
    }
}

template <class Geometry, class Schedule, class Output, bool FullTokens>
void launch_bf16_mma(const __nv_bfloat16* x, const __nv_bfloat16* weight, Output output, int tokens,
                     cudaStream_t stream) {
    constexpr int tiles_m = Geometry::kOutputRows / Schedule::kBlockRows;
    const int tiles_n     = (tokens + Schedule::kBlockCols - 1) / Schedule::kBlockCols;
    const int blocks      = tiles_m * tiles_n;
    static const cudaError_t attr = cudaFuncSetAttribute(
        ops::detail::bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens, Output>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
    CUDA_CHECK(attr);
    ops::detail::bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens, Output>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(x, weight, output, tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool flash_next_moe_shared_mma_enabled() {
    const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_SHARED_MMA");
    if (env == nullptr || env[0] == '\0') { return true; }
    return env[0] == '1' && env[1] == '\0';
}

void flash_next_route_projection_mma(const Tensor& input, const Weight& router,
                                     const Weight& shared_gate, const Tensor& score_workspace,
                                     cudaStream_t stream) {
    const int tokens = static_cast<int>(input.ne[1]);
    Fp32ScoreOutput scores{static_cast<float*>(score_workspace.data), kScoreRows};
    const auto* x = static_cast<const __nv_bfloat16*>(input.data);
    const auto* w = static_cast<const __nv_bfloat16*>(router.qdata);
    if ((tokens % RouterSched::kBlockCols) == 0) {
        launch_bf16_mma<RouterGeom, RouterSched, Fp32ScoreOutput, true>(x, w, scores, tokens,
                                                                        stream);
    } else {
        launch_bf16_mma<RouterGeom, RouterSched, Fp32ScoreOutput, false>(x, w, scores, tokens,
                                                                         stream);
    }
    shared_gate_score_row_kernel<<<static_cast<unsigned>(tokens), 256, 0, stream>>>(
        x, static_cast<const __nv_bfloat16*>(shared_gate.qdata),
        static_cast<float*>(score_workspace.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_moe_prefill_build_groups_mma(
    const Tensor& ids, const Tensor& expert_counts, const Tensor& expert_offsets,
    const Tensor& active_experts, const Tensor& active_count, const Tensor& grouped_tokens,
    const Tensor& grouped_paths, const Tensor& grouped_experts, const Tensor& token_to_pos,
    const Tensor& task_counter, int tokens, cudaStream_t stream) {
    const int total_items = tokens * kTopK;
    auto* counts          = static_cast<std::int32_t*>(expert_counts.data);
    auto* offsets         = static_cast<std::int32_t*>(expert_offsets.data);
    moe_zero_i32_kernel<<<(kExperts + 255) / 256, 256, 0, stream>>>(counts, kExperts);
    CUDA_CHECK(cudaGetLastError());
    const int hist_blocks = (total_items + 255) / 256;
    moe_histogram_kernel<<<hist_blocks, 256, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), counts, total_items);
    CUDA_CHECK(cudaGetLastError());
    moe_prefix_compact_kernel<<<1, 512, 0, stream>>>(
        counts, offsets, static_cast<std::int32_t*>(active_experts.data),
        static_cast<std::int32_t*>(active_count.data),
        static_cast<std::int32_t*>(task_counter.data));
    CUDA_CHECK(cudaGetLastError());
    moe_scatter_groups_kernel<<<1, 512, 0, stream>>>(
        static_cast<const std::int32_t*>(ids.data), offsets,
        static_cast<std::int32_t*>(grouped_tokens.data),
        static_cast<std::int32_t*>(grouped_paths.data),
        static_cast<std::int32_t*>(grouped_experts.data),
        static_cast<std::int32_t*>(token_to_pos.data), total_items);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_moe_prefill_shared_gate_up_mma(const Tensor& input, const Weight& shared_gate,
                                               const Weight& shared_up, const Tensor& activations,
                                               const Tensor& shared_gemm, int tokens,
                                               cudaStream_t stream) {
    const auto* x      = static_cast<const __nv_bfloat16*>(input.data);
    auto* packed       = static_cast<__nv_bfloat16*>(shared_gemm.data);
    auto* gate_out     = packed;
    auto* up_out       = packed + static_cast<std::int64_t>(kIntermediate) * tokens;
    auto* strided_path = static_cast<__nv_bfloat16*>(activations.data) +
                         static_cast<std::int64_t>(kTopK) * kIntermediate;
    ops::detail::Bf16MmaContiguousOutput gate_output{gate_out, kIntermediate};
    ops::detail::Bf16MmaContiguousOutput up_output{up_out, kIntermediate};
    const auto* gw = static_cast<const __nv_bfloat16*>(shared_gate.qdata);
    const auto* uw = static_cast<const __nv_bfloat16*>(shared_up.qdata);
    if ((tokens % GateSched::kBlockCols) == 0) {
        launch_bf16_mma<GateGeom, GateSched, ops::detail::Bf16MmaContiguousOutput, true>(
            x, gw, gate_output, tokens, stream);
        launch_bf16_mma<GateGeom, GateSched, ops::detail::Bf16MmaContiguousOutput, true>(
            x, uw, up_output, tokens, stream);
    } else {
        launch_bf16_mma<GateGeom, GateSched, ops::detail::Bf16MmaContiguousOutput, false>(
            x, gw, gate_output, tokens, stream);
        launch_bf16_mma<GateGeom, GateSched, ops::detail::Bf16MmaContiguousOutput, false>(
            x, uw, up_output, tokens, stream);
    }
    const int n     = tokens * kIntermediate;
    const int block = 256;
    silu_mul_shared_path_kernel<<<(n + block - 1) / block, block, 0, stream>>>(
        gate_out, up_out, strided_path, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_moe_prefill_shared_down_mma(const Weight& shared_down, const Tensor& shared_gemm,
                                            const Tensor& shared_scale, const Tensor& output,
                                            int tokens, cudaStream_t stream) {
    auto* packed_up = static_cast<__nv_bfloat16*>(shared_gemm.data) +
                      static_cast<std::int64_t>(kIntermediate) * tokens;
    ScaledBf16Output out{static_cast<__nv_bfloat16*>(output.data),
                         static_cast<const float*>(shared_scale.data), kHidden};
    const auto* w = static_cast<const __nv_bfloat16*>(shared_down.qdata);
    if ((tokens % DownSched::kBlockCols) == 0) {
        launch_bf16_mma<DownGeom, DownSched, ScaledBf16Output, true>(packed_up, w, out, tokens,
                                                                     stream);
    } else {
        launch_bf16_mma<DownGeom, DownSched, ScaledBf16Output, false>(packed_up, w, out, tokens,
                                                                      stream);
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
