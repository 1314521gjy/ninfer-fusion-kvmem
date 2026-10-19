#include "models/qwen3_8_flash_next/impl/hyper_connection_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

constexpr int kHidden      = 2'560;
constexpr int kStreams     = 4;
constexpr int kConcat      = kHidden * kStreams; // 10,240
constexpr int kLowRank     = 320;
constexpr int kNormThreads = 256;

// The fused decode kernel recomputes the group-norm statistics with the legacy norm
// kernel's thread->chunk mapping, so its block geometry is pinned to the norm kernel's.
constexpr int kPrepThreads         = kNormThreads;
constexpr int kPrepChunksPerThread = (kConcat / 8) / kPrepThreads; // 1280 / 256 = 5
static_assert((kConcat / 8) % kPrepThreads == 0,
              "low-rank rows must split into whole 16 B chunks per thread");

// =========================================================================
// Shared per-element arithmetic for the decode route.
//
// The fused kernel below recomputes what group_norm_vectorized_kernel stores and
// consumes it in place of low_rank_and_injection_kernel's global read. Bit-exactness
// between the two routes rests on both evaluating the same float expressions in the
// same order under the same nvcc contraction decisions; keeping a single copy of each
// expression in these __forceinline__ helpers (same TU, same flags) removes the way
// two hand-copied bodies could drift apart. The test pins legacy == fused bitwise.
// =========================================================================

// Per-thread partial sum of squares over one 2,560-wide stream. The chunk order
// (tid, tid + 256; threads 0..63 take two chunks) and the fmaf chain feed a
// block_reduce_sum<256> tree; any other split changes the rounding of the sum.
__device__ __forceinline__ float hyper_stream_sum_sq_partial(
    const __nv_bfloat16* __restrict__ in_stream, int tid) {
    // 2560 elements = 320 ulonglong2 (8 BF16) chunks
    float sum_sq = 0.0F;
    for (int chunk = tid; chunk < (kHidden / 8); chunk += kNormThreads) {
        const auto raw_in = *reinterpret_cast<const ulonglong2*>(in_stream + chunk * 8);
        const auto* bf_in = reinterpret_cast<const __nv_bfloat16*>(&raw_in);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = __bfloat162float(bf_in[i]);
            sum_sq        = fmaf(v, v, sum_sq);
        }
    }
    return sum_sq;
}

__device__ __forceinline__ float hyper_inv_rms(float sum_sq) {
    return rsqrtf(sum_sq / static_cast<float>(kHidden) + 1.0e-6F);
}

// One 8-element chunk of the normalized stream: (v * inv_rms) * (1 + n), one BF16
// rounding. The legacy route stores this to `normalized` and re-reads it; the fused
// route feeds it straight into hyper_dot_chunk, which sees the identical BF16 value.
__device__ __forceinline__ ulonglong2 hyper_normalize_chunk(ulonglong2 raw_in,
                                                            ulonglong2 raw_norm,
                                                            float inv_rms) {
    const auto* bf_in   = reinterpret_cast<const __nv_bfloat16*>(&raw_in);
    const auto* bf_norm = reinterpret_cast<const __nv_bfloat16*>(&raw_norm);
    ulonglong2 raw_out;
    auto* bf_out = reinterpret_cast<__nv_bfloat16*>(&raw_out);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float v = __bfloat162float(bf_in[i]);
        const float n = __bfloat162float(bf_norm[i]);
        bf_out[i]     = __float2bfloat16_rn(v * inv_rms * (1.0F + n));
    }
    return raw_out;
}

// Materializes one stream of `normalized` for the consumers that read it from global
// memory (mix_up_and_reduce_kernel, the stage sink, the tests).
__device__ __forceinline__ void hyper_store_normalized_stream(
    const __nv_bfloat16* __restrict__ in_stream, const __nv_bfloat16* __restrict__ norm_stream,
    __nv_bfloat16* __restrict__ out_stream, float inv_rms, int tid) {
    for (int chunk = tid; chunk < (kHidden / 8); chunk += kNormThreads) {
        const int col_base  = chunk * 8;
        const auto raw_in   = *reinterpret_cast<const ulonglong2*>(in_stream + col_base);
        const auto raw_norm = *reinterpret_cast<const ulonglong2*>(norm_stream + col_base);
        *reinterpret_cast<ulonglong2*>(out_stream + col_base) =
            hyper_normalize_chunk(raw_in, raw_norm, inv_rms);
    }
}

// Eight fmaf steps of the row dot product, in element order, into the running sum.
__device__ __forceinline__ float hyper_dot_chunk(ulonglong2 w_raw, ulonglong2 x_raw, float sum) {
    const auto* w_bf = reinterpret_cast<const __nv_bfloat16*>(&w_raw);
    const auto* x_bf = reinterpret_cast<const __nv_bfloat16*>(&x_raw);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), sum);
    }
    return sum;
}

// Row epilogue after the block reduction: rows < 320 are the SiLU'd low-rank mixer
// input, rows 320..323 the injection gates (absent on the final/MTP mixer form).
__device__ __forceinline__ void hyper_low_rank_epilogue(float sum, int row, int token,
                                                        __nv_bfloat16* __restrict__ low_rank,
                                                        float* __restrict__ injection) {
    if (row < kLowRank) {
        low_rank[static_cast<std::int64_t>(token) * kLowRank + row] =
            __float2bfloat16_rn(ops::silu(sum * 0.25F));
    } else if (injection != nullptr) {
        injection[static_cast<std::int64_t>(token) * kStreams + (row - kLowRank)] =
            2.0F * ops::sigmoid(sum * 0.25F);
    }
}

// =========================================================================
// Kernel 1 (legacy decode route): Vectorized 4-Stream Group RMSNorm
// Grid: dim3(4, tokens), Block: 256 threads
// =========================================================================
__global__ void __launch_bounds__(kNormThreads)
group_norm_vectorized_kernel(const __nv_bfloat16* __restrict__ hidden,
                             const __nv_bfloat16* __restrict__ norm,
                             __nv_bfloat16* __restrict__ normalized,
                             int tokens) {
    __shared__ float s_warp_sums[kNormThreads / 32];
    __shared__ float s_inv_rms;

    const int stream = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);

    if (token >= tokens || stream >= kStreams) { return; }

    const int stream_offset = stream * kHidden;
    const auto* in_stream   = hidden + static_cast<std::int64_t>(token) * kConcat + stream_offset;

    float sum_sq = hyper_stream_sum_sq_partial(in_stream, tid);
    sum_sq       = ops::block_reduce_sum<kNormThreads>(sum_sq, s_warp_sums);
    if (tid == 0) {
        s_inv_rms = hyper_inv_rms(sum_sq);
    }
    __syncthreads();

    hyper_store_normalized_stream(
        in_stream, norm + stream_offset,
        normalized + static_cast<std::int64_t>(token) * kConcat + stream_offset, s_inv_rms, tid);
}

// =========================================================================
// Kernel 2 (legacy decode route): Fused Down Projection & Injection Gates
// Grid: dim3(total_rows, tokens) where total_rows = 324 (or 320)
// Block: 256 threads (8 warps)
// 1 CTA per row -> 324 CTAs at T=1 saturate all SMs.
// Vectorized 128-bit memory loads across 10,240 elements (40 steps x 8 elements).
// =========================================================================
__global__ void __launch_bounds__(256)
low_rank_and_injection_kernel(const __nv_bfloat16* __restrict__ normalized,
                              const __nv_bfloat16* __restrict__ down_weight,
                              const __nv_bfloat16* __restrict__ inject_weight,
                              __nv_bfloat16* __restrict__ low_rank,
                              float* __restrict__ injection,
                              int tokens, int total_rows) {
    __shared__ float s_warp_sums[8]; // 256 / 32

    const int row   = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);

    if (token >= tokens || row >= total_rows) { return; }

    const auto* x_token = normalized + static_cast<std::int64_t>(token) * kConcat;
    const __nv_bfloat16* w_row = (row < kLowRank)
        ? (down_weight + static_cast<std::int64_t>(row) * kConcat)
        : (inject_weight + static_cast<std::int64_t>(row - kLowRank) * kConcat);

    // 10,240 elements = 1,280 ulonglong2 chunks.
    // With 256 threads, each thread loads 1280 / 256 = 5 chunks (40 BF16 elements).
    float sum = 0.0F;
    #pragma unroll
    for (int chunk = tid; chunk < (kConcat / 8); chunk += 256) {
        const int col_base = chunk * 8;
        const auto w_raw   = *reinterpret_cast<const ulonglong2*>(w_row + col_base);
        const auto x_raw   = *reinterpret_cast<const ulonglong2*>(x_token + col_base);
        sum                = hyper_dot_chunk(w_raw, x_raw, sum);
    }

    sum = ops::block_reduce_sum<256>(sum, s_warp_sums);

    if (tid == 0) {
        hyper_low_rank_epilogue(sum, row, token, low_rank, injection);
    }
}

// =========================================================================
// Kernel 1+2 (default decode route): Group RMSNorm fused into the row dot products
// Grid: dim3(total_rows, tokens), Block: 256 threads, 48 B static smem
//
// group_norm_vectorized_kernel is a 4-CTA launch whose whole duration is latency
// (two block reductions, 13% warps active, ~3.5 us) and it runs 97 times per token.
// Every CTA here recomputes the four stream statistics itself: 20 KB of L2-resident
// `hidden` per CTA plus four block reductions, overlapped with the CTA's own 20 KB
// DRAM weight-row fetch, which is issued first and held in registers. Rows 0..3 also
// store their stream of `normalized` because mix_up_and_reduce_kernel still reads it
// from global memory; no CTA of this kernel reads `normalized`, so there is no race.
//
// The row->CTA mapping is deliberately the legacy one (one 256-thread CTA per row,
// chunks tid + 256k, block_reduce_sum<256>): splitting a row across CTAs or threads
// would re-associate the fmaf chain and break bitwise identity with the legacy route.
// =========================================================================
__global__ void __launch_bounds__(kPrepThreads)
hyper_norm_low_rank_fused_kernel(const __nv_bfloat16* __restrict__ hidden,        // [10240, T]
                                 const __nv_bfloat16* __restrict__ norm,          // [10240]
                                 const __nv_bfloat16* __restrict__ down_weight,   // [320, 10240]
                                 const __nv_bfloat16* __restrict__ inject_weight, // [4, 10240] or nullptr
                                 __nv_bfloat16* __restrict__ normalized,          // [10240, T], rows 0..3 write
                                 __nv_bfloat16* __restrict__ low_rank,            // [320, T]
                                 float* __restrict__ injection,                   // [4, T] or nullptr
                                 int tokens, int total_rows) {
    __shared__ float s_warp_sums[kPrepThreads / 32];
    __shared__ float s_inv_rms[kStreams];

    const int row   = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);

    if (token >= tokens || row >= total_rows) { return; }

    const __nv_bfloat16* w_row = (row < kLowRank)
        ? (down_weight + static_cast<std::int64_t>(row) * kConcat)
        : (inject_weight + static_cast<std::int64_t>(row - kLowRank) * kConcat);

    // (1) Weight-row prefetch: the only DRAM traffic of this CTA, independent of the
    //     activations, so it is in flight while the statistics pass runs from L2.
    ulonglong2 w_raw[kPrepChunksPerThread];
    #pragma unroll
    for (int k = 0; k < kPrepChunksPerThread; ++k) {
        w_raw[k] = *reinterpret_cast<const ulonglong2*>(w_row + (tid + kPrepThreads * k) * 8);
    }

    const auto* hid_token = hidden + static_cast<std::int64_t>(token) * kConcat;

    // (2) Statistics, one stream at a time through the legacy reduction tree.
    //     block_reduce_sum ends with __syncthreads, so s_warp_sums is free for the next
    //     stream and s_inv_rms[s] is only read after the barrier below.
    #pragma unroll 1
    for (int s = 0; s < kStreams; ++s) {
        float sum_sq = hyper_stream_sum_sq_partial(hid_token + s * kHidden, tid);
        sum_sq       = ops::block_reduce_sum<kPrepThreads>(sum_sq, s_warp_sums);
        if (tid == 0) {
            s_inv_rms[s] = hyper_inv_rms(sum_sq);
        }
    }
    __syncthreads();

    // (3) Rows 0..3 materialize `normalized` for stream == row (legacy K1 second loop).
    if (row < kStreams) {
        const int stream_offset = row * kHidden;
        hyper_store_normalized_stream(
            hid_token + stream_offset, norm + stream_offset,
            normalized + static_cast<std::int64_t>(token) * kConcat + stream_offset,
            s_inv_rms[row], tid);
    }

    // (4) Row dot product with the normalized value recomputed per chunk. Chunk order
    //     tid + 256k is the legacy K2 loop order; the fmaf chain is unchanged.
    float sum = 0.0F;
    #pragma unroll
    for (int k = 0; k < kPrepChunksPerThread; ++k) {
        const int chunk     = tid + kPrepThreads * k;
        const int col_base  = chunk * 8;
        const float inv_rms = s_inv_rms[chunk / (kHidden / 8)];
        const auto h_raw    = *reinterpret_cast<const ulonglong2*>(hid_token + col_base);
        const auto n_raw    = *reinterpret_cast<const ulonglong2*>(norm + col_base);
        sum = hyper_dot_chunk(w_raw[k], hyper_normalize_chunk(h_raw, n_raw, inv_rms), sum);
    }

    sum = ops::block_reduce_sum<kPrepThreads>(sum, s_warp_sums);

    if (tid == 0) {
        hyper_low_rank_epilogue(sum, row, token, low_rank, injection);
    }
}

// =========================================================================
// Kernel 3: Mix Up Projection & Stream Reduction
// Grid: dim3(kHidden = 2560, tokens), Block: 128 threads (4 warps = 1 warp per stream)
// 2,560 CTAs at T=1 saturate all SMs.
// Warp s computes stream s dot-product (320 elements), applies sigmoid * normalized,
// thread 0 averages the 4 streams into block_input.
// =========================================================================
__global__ void __launch_bounds__(128)
mix_up_and_reduce_kernel(const __nv_bfloat16* __restrict__ normalized,
                         const __nv_bfloat16* __restrict__ low_rank,
                         const __nv_bfloat16* __restrict__ up_weight,
                         __nv_bfloat16* __restrict__ block_input,
                         int tokens) {
    __shared__ float s_contrib[kStreams];

    const int hidden  = static_cast<int>(blockIdx.x);
    const int token   = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int stream  = tid >> 5; // 0, 1, 2, 3
    const int lane_id = tid & 31; // 0..31

    if (token >= tokens || hidden >= kHidden) { return; }

    const int row              = stream * kHidden + hidden;
    const __nv_bfloat16* w_row = up_weight + static_cast<std::int64_t>(row) * kLowRank;
    const auto* lr_token       = low_rank + static_cast<std::int64_t>(token) * kLowRank;

    // 320 elements:
    // Step 0: 32 lanes x 8 elements = 256 elements (columns 0..255)
    float sum = 0.0F;
    {
        const int col_base = lane_id * 8;
        const auto w_raw   = *reinterpret_cast<const ulonglong2*>(w_row + col_base);
        const auto* w_bf   = reinterpret_cast<const __nv_bfloat16*>(&w_raw);
        const auto x_raw   = *reinterpret_cast<const ulonglong2*>(lr_token + col_base);
        const auto* x_bf   = reinterpret_cast<const __nv_bfloat16*>(&x_raw);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), sum);
        }
    }
    // Step 1: 32 lanes x 2 elements = 64 elements (columns 256..319)
    {
        const int col_base = 256 + lane_id * 2;
        const auto w_raw   = *reinterpret_cast<const std::uint32_t*>(w_row + col_base);
        const auto* w_bf   = reinterpret_cast<const __nv_bfloat16*>(&w_raw);
        const auto x_raw   = *reinterpret_cast<const std::uint32_t*>(lr_token + col_base);
        const auto* x_bf   = reinterpret_cast<const __nv_bfloat16*>(&x_raw);

        sum = fmaf(__bfloat162float(w_bf[0]), __bfloat162float(x_bf[0]), sum);
        sum = fmaf(__bfloat162float(w_bf[1]), __bfloat162float(x_bf[1]), sum);
    }

    sum = ops::warp_reduce_sum(sum);

    if (lane_id == 0) {
        const float mix_gate = ops::sigmoid(sum);
        const float norm_val =
            __bfloat162float(normalized[static_cast<std::int64_t>(token) * kConcat + row]);
        s_contrib[stream]    = mix_gate * norm_val;
    }
    __syncthreads();

    if (tid == 0) {
        const float mean =
            (s_contrib[0] + s_contrib[1] + s_contrib[2] + s_contrib[3]) * 0.25F;
        block_input[static_cast<std::int64_t>(token) * kHidden + hidden] =
            __float2bfloat16_rn(mean);
    }
}

// =========================================================================
// Kernel 4: Vectorized In-Place Injection (hidden += block_output * injection)
// Grid: dim3((10240 / 8 + 255) / 256, tokens), Block: 256 threads
// =========================================================================
__global__ void __launch_bounds__(256)
hyper_inject_vectorized_kernel(const __nv_bfloat16* __restrict__ block_output,
                               const float* __restrict__ injection,
                               __nv_bfloat16* __restrict__ hidden,
                               int tokens) {
    const int tid   = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.y);

    if (token >= tokens || tid >= (kConcat / 8)) { return; }

    const int col_base   = tid * 8;
    const int stream     = col_base / kHidden;
    const int hidden_col = col_base - stream * kHidden;

    const float inj_scale = injection[static_cast<std::int64_t>(token) * kStreams + stream];
    const auto* out_ptr   = block_output + static_cast<std::int64_t>(token) * kHidden + hidden_col;
    auto* hid_ptr         = hidden + static_cast<std::int64_t>(token) * kConcat + col_base;

    const auto out_raw = *reinterpret_cast<const ulonglong2*>(out_ptr);
    const auto* out_bf = reinterpret_cast<const __nv_bfloat16*>(&out_raw);
    const auto hid_raw = *reinterpret_cast<const ulonglong2*>(hid_ptr);
    const auto* hid_bf = reinterpret_cast<const __nv_bfloat16*>(&hid_raw);

    ulonglong2 res_raw;
    auto* res_bf = reinterpret_cast<__nv_bfloat16*>(&res_raw);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float out_f = __bfloat162float(out_bf[i]);
        const float hid_f = __bfloat162float(hid_bf[i]);
        const float upd   = fmaf(out_f, inj_scale, hid_f);
        res_bf[i]         = __float2bfloat16_rn(upd);
    }

    *reinterpret_cast<ulonglong2*>(hid_ptr) = res_raw;
}

// =========================================================================
// 2. PREFILL KERNELS (T >= 16) - Weight-Stationary & Tensor Core Accelerated
// =========================================================================

// Streamlined 4-Stream Group RMSNorm: 1 CTA per token (64 threads per stream)
__global__ void __launch_bounds__(256)
group_norm_prefill_kernel(
    const __nv_bfloat16* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ norm,
    __nv_bfloat16* __restrict__ normalized,
    int tokens) {
    
    __shared__ float s_warp_sums[kStreams][2];
    __shared__ float s_inv_rms[kStreams];

    const int token   = static_cast<int>(blockIdx.x);
    const int tid     = static_cast<int>(threadIdx.x);
    const int stream  = tid / 64;
    const int stream_tid = tid % 64;
    const int warp_in_stream = stream_tid / 32;
    const int lane_id = stream_tid % 32;

    if (token >= tokens) { return; }

    const int stream_offset = stream * kHidden;
    const auto* in_stream   = hidden + static_cast<std::int64_t>(token) * kConcat + stream_offset;

    float sum_sq = 0.0F;
    #pragma unroll
    for (int step = 0; step < 5; ++step) {
        const int chunk = stream_tid + step * 64;
        const auto raw_in = *reinterpret_cast<const ulonglong2*>(in_stream + chunk * 8);
        const auto* bf_in = reinterpret_cast<const __nv_bfloat16*>(&raw_in);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = __bfloat162float(bf_in[i]);
            sum_sq = fmaf(v, v, sum_sq);
        }
    }

    sum_sq = ops::warp_reduce_sum(sum_sq);
    if (lane_id == 0) {
        s_warp_sums[stream][warp_in_stream] = sum_sq;
    }
    __syncthreads();

    if (stream_tid == 0) {
        float total_sq = s_warp_sums[stream][0] + s_warp_sums[stream][1];
        s_inv_rms[stream] = rsqrtf(total_sq / static_cast<float>(kHidden) + 1.0e-6F);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms[stream];
    const auto* norm_stream = norm + stream_offset;
    auto* out_stream        = normalized + static_cast<std::int64_t>(token) * kConcat + stream_offset;

    #pragma unroll
    for (int step = 0; step < 5; ++step) {
        const int chunk     = stream_tid + step * 64;
        const int col_base  = chunk * 8;
        const auto raw_in   = *reinterpret_cast<const ulonglong2*>(in_stream + col_base);
        const auto* bf_in   = reinterpret_cast<const __nv_bfloat16*>(&raw_in);
        const auto raw_norm = *reinterpret_cast<const ulonglong2*>(norm_stream + col_base);
        const auto* bf_norm = reinterpret_cast<const __nv_bfloat16*>(&raw_norm);

        ulonglong2 raw_out;
        auto* bf_out = reinterpret_cast<__nv_bfloat16*>(&raw_out);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float v = __bfloat162float(bf_in[i]);
            const float n = __bfloat162float(bf_norm[i]);
            bf_out[i]     = __float2bfloat16_rn(v * inv_rms * (1.0F + n));
        }
        *reinterpret_cast<ulonglong2*>(out_stream + col_base) = raw_out;
    }
}

// Split-K MMA writes per-split FP32 partials. SiLU is applied after a fixed-order
// reduction over splits (no atomics).
struct Bf16HyperDownSplitOutputTile {
    float* partials;
    std::int32_t tokens;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                          float accumulator) const {
        if (row < kLowRank) {
            const int split = static_cast<int>(blockIdx.y);
            partials[(static_cast<std::int64_t>(split) * tokens + token) * kLowRank + row] =
                accumulator;
        }
    }
};

struct Bf16HyperDownSplitOutput {
    float* partials;
    std::int32_t tokens;

    __device__ __forceinline__ Bf16HyperDownSplitOutputTile tile(std::int32_t) const {
        return {partials, tokens};
    }
};

template <int SplitK>
__global__ __launch_bounds__(256) void down_splitk_reduce_kernel(const float* __restrict__ partials,
                                                                __nv_bfloat16* __restrict__ low_rank,
                                                                int tokens) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    const int n = tokens * kLowRank;
    if (idx >= n) { return; }
    const int token = idx / kLowRank;
    const int row   = idx - token * kLowRank;
    float sum       = 0.0F;
#pragma unroll
    for (int split = 0; split < SplitK; ++split) {
        sum += partials[(static_cast<std::int64_t>(split) * tokens + token) * kLowRank + row];
    }
    low_rank[idx] = __float2bfloat16_rn(ops::silu(sum * 0.25F));
}

// Streamlined Injection Gate: 1 CTA per token (4 warps = 4 streams, zero inter-warp sync)
__global__ void __launch_bounds__(128)
injection_gate_prefill_kernel(
    const __nv_bfloat16* __restrict__ normalized,
    const __nv_bfloat16* __restrict__ inject_weight,
    float* __restrict__ injection,
    int tokens) {
    
    const int token   = static_cast<int>(blockIdx.x);
    const int tid     = static_cast<int>(threadIdx.x);
    const int stream  = tid >> 5; // 0..3
    const int lane_id = tid & 31; // 0..31

    if (token >= tokens || stream >= kStreams) { return; }

    const auto* x_token = normalized + static_cast<std::int64_t>(token) * kConcat;
    const auto* w_row   = inject_weight + static_cast<std::int64_t>(stream) * kConcat;

    float sum = 0.0F;
    #pragma unroll 4
    for (int step = 0; step < 40; ++step) {
        const int chunk    = lane_id + step * 32;
        const int col_base = chunk * 8;
        const auto w_raw   = *reinterpret_cast<const ulonglong2*>(w_row + col_base);
        const auto* w_bf   = reinterpret_cast<const __nv_bfloat16*>(&w_raw);
        const auto x_raw   = *reinterpret_cast<const ulonglong2*>(x_token + col_base);
        const auto* x_bf   = reinterpret_cast<const __nv_bfloat16*>(&x_raw);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), sum);
        }
    }

    sum = ops::warp_reduce_sum(sum);

    if (lane_id == 0) {
        injection[static_cast<std::int64_t>(token) * kStreams + stream] =
            2.0F * ops::sigmoid(sum * 0.25F);
    }
}

// Vectorized Up Reduction: 1 CTA per token (256 threads process 320 ulonglong2 chunks)
__global__ void __launch_bounds__(256)
up_reduction_vectorized_kernel(
    const __nv_bfloat16* __restrict__ gemm_out,
    const __nv_bfloat16* __restrict__ normalized,
    __nv_bfloat16* __restrict__ block_input,
    int tokens) {
    
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);

    if (token >= tokens) { return; }

    const std::int64_t tok_offset = static_cast<std::int64_t>(token) * kConcat;

    for (int chunk = tid; chunk < (kHidden / 8); chunk += 256) {
        const int h_base = chunk * 8;

        float mean[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            mean[i] = 0.0F;
        }

        #pragma unroll
        for (int s = 0; s < kStreams; ++s) {
            const int row_base = s * kHidden + h_base;
            const auto z_raw   = *reinterpret_cast<const ulonglong2*>(gemm_out + tok_offset + row_base);
            const auto* z_bf   = reinterpret_cast<const __nv_bfloat16*>(&z_raw);
            const auto n_raw   = *reinterpret_cast<const ulonglong2*>(normalized + tok_offset + row_base);
            const auto* n_bf   = reinterpret_cast<const __nv_bfloat16*>(&n_raw);

            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float z = __bfloat162float(z_bf[i]);
                const float n = __bfloat162float(n_bf[i]);
                mean[i] += ops::sigmoid(z) * n;
            }
        }

        ulonglong2 out_raw;
        auto* out_bf = reinterpret_cast<__nv_bfloat16*>(&out_raw);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_bf[i] = __float2bfloat16_rn(mean[i] * 0.25F);
        }

        *reinterpret_cast<ulonglong2*>(block_input + static_cast<std::int64_t>(token) * kHidden + h_base) = out_raw;
    }
}

// Down & Up MMA Geometries and Schedules
// Prefill down-projection is N=320 x T. A 32x32 tile yields only 40 CTAs at T=128 on 188 SMs.
// Split-K=4 keeps that tile and launches 160 CTAs. Decode T<=8 stays on the fused path.
using DownGeom    = ops::detail::Bf16GemvGeometry<320, 10240>;
using DownSched   = ops::detail::Bf16MmaSchedule<32, 32, 256, 16, 16, 2, 2, ops::Cache::cg, ops::Cache::cg,
                                                 ops::detail::Bf16MmaFragmentPipeline::PingPong,
                                                 ops::detail::Bf16MmaRaster::TokenFast>;

using UpGeom      = ops::detail::Bf16GemvGeometry<10240, 320>;
using UpSched     = ops::detail::Bf16MmaSchedule<64, 64, 64, 32, 32, 2, 2, ops::Cache::cg, ops::Cache::cg,
                                                 ops::detail::Bf16MmaFragmentPipeline::PingPong,
                                                 ops::detail::Bf16MmaRaster::TokenFast>;

template <bool FullTokens>
void launch_down_prefill(const __nv_bfloat16* x, const __nv_bfloat16* weight, float* partials,
                         __nv_bfloat16* low_rank, int tokens, cudaStream_t stream) {
    Bf16HyperDownSplitOutput out_down{partials, tokens};
    const int tiles_m_down = 320 / DownSched::kBlockRows;
    const int tiles_n_down = (tokens + DownSched::kBlockCols - 1) / DownSched::kBlockCols;
    const int blocks_down  = tiles_m_down * tiles_n_down;
    static const cudaError_t attr_down = cudaFuncSetAttribute(
        ops::detail::bf16_gemm_mma_kernel<DownGeom, DownSched, FullTokens, Bf16HyperDownSplitOutput,
                                          kHyperDownSplitK>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, DownSched::kSharedBytes);
    CUDA_CHECK(attr_down);
    ops::detail::bf16_gemm_mma_kernel<DownGeom, DownSched, FullTokens, Bf16HyperDownSplitOutput,
                                      kHyperDownSplitK>
        <<<dim3(blocks_down, kHyperDownSplitK), DownSched::kThreads, DownSched::kSharedBytes,
           stream>>>(x, weight, out_down, tokens);
    CUDA_CHECK(cudaGetLastError());
    constexpr int kReduceThreads = 256;
    const int reduce_blocks      = (tokens * kLowRank + kReduceThreads - 1) / kReduceThreads;
    down_splitk_reduce_kernel<kHyperDownSplitK>
        <<<reduce_blocks, kReduceThreads, 0, stream>>>(partials, low_rank, tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_down_prefill_dispatch(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                  float* partials, __nv_bfloat16* low_rank, int tokens,
                                  cudaStream_t stream) {
    if ((tokens % DownSched::kBlockCols) == 0) {
        launch_down_prefill<true>(x, weight, partials, low_rank, tokens, stream);
    } else {
        launch_down_prefill<false>(x, weight, partials, low_rank, tokens, stream);
    }
}

template <bool FullTokens>
void launch_up_prefill(const __nv_bfloat16* low_rank, const __nv_bfloat16* weight,
                       __nv_bfloat16* up_gemm, int tokens, cudaStream_t stream) {
    ops::detail::Bf16MmaContiguousOutput out_up{up_gemm, 10240};
    const int tiles_m_up = 10240 / UpSched::kBlockRows;
    const int tiles_n_up = (tokens + UpSched::kBlockCols - 1) / UpSched::kBlockCols;
    const int blocks_up  = tiles_m_up * tiles_n_up;
    static const cudaError_t attr_up = cudaFuncSetAttribute(
        ops::detail::bf16_gemm_mma_kernel<UpGeom, UpSched, FullTokens,
                                          ops::detail::Bf16MmaContiguousOutput>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, UpSched::kSharedBytes);
    CUDA_CHECK(attr_up);
    ops::detail::bf16_gemm_mma_kernel<UpGeom, UpSched, FullTokens>
        <<<blocks_up, UpSched::kThreads, UpSched::kSharedBytes, stream>>>(low_rank, weight, out_up,
                                                                         tokens);
    CUDA_CHECK(cudaGetLastError());
}

void launch_up_prefill_dispatch(const __nv_bfloat16* low_rank, const __nv_bfloat16* weight,
                                __nv_bfloat16* up_gemm, int tokens, cudaStream_t stream) {
    if ((tokens % UpSched::kBlockCols) == 0) {
        launch_up_prefill<true>(low_rank, weight, up_gemm, tokens, stream);
    } else {
        launch_up_prefill<false>(low_rank, weight, up_gemm, tokens, stream);
    }
}

// Decode-route stage 1: `normalized` (all four streams), `low_rank`, and (when
// inject_weight != nullptr) `injection`. Both routes leave identical bits in all three.
void launch_decode_norm_low_rank(FlashNextHyperDecodeRoute route, const __nv_bfloat16* hidden,
                                 const __nv_bfloat16* norm, const __nv_bfloat16* down_weight,
                                 const __nv_bfloat16* inject_weight, __nv_bfloat16* normalized,
                                 __nv_bfloat16* low_rank, float* injection, int tokens,
                                 int total_rows, cudaStream_t stream) {
    if (route == FlashNextHyperDecodeRoute::Legacy) {
        group_norm_vectorized_kernel<<<dim3(kStreams, tokens), kNormThreads, 0, stream>>>(
            hidden, norm, normalized, tokens);
        CUDA_CHECK(cudaGetLastError());

        low_rank_and_injection_kernel<<<dim3(total_rows, tokens), 256, 0, stream>>>(
            normalized, down_weight, inject_weight, low_rank, injection, tokens, total_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    hyper_norm_low_rank_fused_kernel<<<dim3(total_rows, tokens), kPrepThreads, 0, stream>>>(
        hidden, norm, down_weight, inject_weight, normalized, low_rank, injection, tokens,
        total_rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

FlashNextHyperDecodeRoute flash_next_hyper_decode_route() {
    // Read once per process: decode graphs are captured with whichever route is active,
    // and a mid-run flip would make the eager and replayed paths disagree on topology.
    static const FlashNextHyperDecodeRoute route = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_HYPER_LEGACY");
        const bool legacy = env != nullptr && env[0] == '1' && env[1] == '\0';
        return legacy ? FlashNextHyperDecodeRoute::Legacy : FlashNextHyperDecodeRoute::Fused;
    }();
    return route;
}

void flash_next_hyper_prepare_launch(const Tensor& hidden, const HyperConnectionWeights& weights,
                                     FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                     cudaStream_t stream) {
    flash_next_hyper_prepare_route_launch(hidden, weights, scratch, block_input, stream,
                                          flash_next_hyper_decode_route());
}

void flash_next_hyper_prepare_route_launch(const Tensor& hidden,
                                           const HyperConnectionWeights& weights,
                                           FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                           cudaStream_t stream, FlashNextHyperDecodeRoute route) {
    const int tokens = static_cast<int>(hidden.ne[1]);

    if (tokens <= 8) {
        // Decode route (T <= 8)
        constexpr int kTotalRows = kLowRank + kStreams; // 324
        launch_decode_norm_low_rank(
            route, static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const __nv_bfloat16*>(weights.norm.data),
            static_cast<const __nv_bfloat16*>(weights.input_mix_down.qdata),
            static_cast<const __nv_bfloat16*>(weights.block_inject.qdata),
            static_cast<__nv_bfloat16*>(scratch.normalized.data),
            static_cast<__nv_bfloat16*>(scratch.low_rank.data),
            static_cast<float*>(scratch.injection.data), tokens, kTotalRows, stream);

        mix_up_and_reduce_kernel<<<dim3(kHidden, tokens), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.normalized.data),
            static_cast<const __nv_bfloat16*>(scratch.low_rank.data),
            static_cast<const __nv_bfloat16*>(weights.input_mix_up.qdata),
            static_cast<__nv_bfloat16*>(block_input.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Prefill route (T >= 16) - Weight-Stationary & Tensor Core Accelerated
        // 1. Group Norm
        group_norm_prefill_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const __nv_bfloat16*>(weights.norm.data),
            static_cast<__nv_bfloat16*>(scratch.normalized.data), tokens);
        CUDA_CHECK(cudaGetLastError());

        // 2. Down-Projection via Tensor Core MMA, Split-K=4, then SiLU reduce
        launch_down_prefill_dispatch(
            static_cast<const __nv_bfloat16*>(scratch.normalized.data),
            static_cast<const __nv_bfloat16*>(weights.input_mix_down.qdata),
            static_cast<float*>(scratch.down_split.data),
            static_cast<__nv_bfloat16*>(scratch.low_rank.data), tokens, stream);

        // 3. Injection Gates
        injection_gate_prefill_kernel<<<tokens, 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.normalized.data),
            static_cast<const __nv_bfloat16*>(weights.block_inject.qdata),
            static_cast<float*>(scratch.injection.data), tokens);
        CUDA_CHECK(cudaGetLastError());

        // 4. Up-Projection via Tensor Core MMA. FullTokens=true is exact only when
        // T % 64 == 0; otherwise the 64-col tile would store past up_gemm.
        launch_up_prefill_dispatch(
            static_cast<const __nv_bfloat16*>(scratch.low_rank.data),
            static_cast<const __nv_bfloat16*>(weights.input_mix_up.qdata),
            static_cast<__nv_bfloat16*>(scratch.up_gemm.data), tokens, stream);

        // 5. Up Reduction & 4-Stream Average
        up_reduction_vectorized_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.up_gemm.data),
            static_cast<const __nv_bfloat16*>(scratch.normalized.data),
            static_cast<__nv_bfloat16*>(block_input.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

void flash_next_hyper_mix_launch(const Tensor& hidden, const HyperMixerWeights& weights,
                                 FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                 cudaStream_t stream) {
    flash_next_hyper_mix_route_launch(hidden, weights, scratch, block_input, stream,
                                      flash_next_hyper_decode_route());
}

void flash_next_hyper_mix_route_launch(const Tensor& hidden, const HyperMixerWeights& weights,
                                       FlashNextHyperWorkspace& scratch, Tensor& block_input,
                                       cudaStream_t stream, FlashNextHyperDecodeRoute route) {
    const int tokens = static_cast<int>(hidden.ne[1]);

    if (tokens <= 8) {
        // Decode route (T <= 8): 320 rows, no injection gates on the mixer form.
        launch_decode_norm_low_rank(
            route, static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const __nv_bfloat16*>(weights.norm.data),
            static_cast<const __nv_bfloat16*>(weights.input_mix_down.qdata), nullptr,
            static_cast<__nv_bfloat16*>(scratch.normalized.data),
            static_cast<__nv_bfloat16*>(scratch.low_rank.data), nullptr, tokens, kLowRank,
            stream);

        mix_up_and_reduce_kernel<<<dim3(kHidden, tokens), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.normalized.data),
            static_cast<const __nv_bfloat16*>(scratch.low_rank.data),
            static_cast<const __nv_bfloat16*>(weights.input_mix_up.qdata),
            static_cast<__nv_bfloat16*>(block_input.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Prefill route (T >= 16)
        // 1. Group Norm
        group_norm_prefill_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const __nv_bfloat16*>(weights.norm.data),
            static_cast<__nv_bfloat16*>(scratch.normalized.data), tokens);
        CUDA_CHECK(cudaGetLastError());

        // 2. Down-Projection via Tensor Core MMA, Split-K=4, then SiLU reduce
        launch_down_prefill_dispatch(
            static_cast<const __nv_bfloat16*>(scratch.normalized.data),
            static_cast<const __nv_bfloat16*>(weights.input_mix_down.qdata),
            static_cast<float*>(scratch.down_split.data),
            static_cast<__nv_bfloat16*>(scratch.low_rank.data), tokens, stream);

        // 3. Up-Projection via Tensor Core MMA (same FullTokens rule as prepare)
        launch_up_prefill_dispatch(
            static_cast<const __nv_bfloat16*>(scratch.low_rank.data),
            static_cast<const __nv_bfloat16*>(weights.input_mix_up.qdata),
            static_cast<__nv_bfloat16*>(scratch.up_gemm.data), tokens, stream);

        // 4. Up Reduction & 4-Stream Average
        up_reduction_vectorized_kernel<<<tokens, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.up_gemm.data),
            static_cast<const __nv_bfloat16*>(scratch.normalized.data),
            static_cast<__nv_bfloat16*>(block_input.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

void flash_next_hyper_inject_launch(const Tensor& block_output, const Tensor& injection,
                                    Tensor& hidden, cudaStream_t stream) {
    const int tokens = static_cast<int>(hidden.ne[1]);
    dim3 block(256);
    dim3 grid((kConcat / 8 + block.x - 1) / block.x, tokens);
    hyper_inject_vectorized_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(block_output.data),
        static_cast<const float*>(injection.data),
        static_cast<__nv_bfloat16*>(hidden.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
