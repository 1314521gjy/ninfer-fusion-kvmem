#include "models/qwen3_8_flash_next/impl/moe_route.h"

#include "core/device.h"
#include "models/qwen3_8_flash_next/impl/moe_shared_kernels.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Arrival counter for the fused decode router (see route_fused_decode_kernel). Every launch adds
// exactly kFusedRowBlocks arrivals, so the counter is a multiple of kFusedRowBlocks at every
// launch boundary and never needs a reset: the CTA whose ticket completes a group is the last one
// of its launch. This relies on route launches being serialized, which holds because the engine
// owns a single compute stream (src/core/device.cu) and CUDA-graph replays serialize on it. It is
// 64-bit so that wrap-around (which would break the grouping once, 2^32 is not a multiple of 129)
// never happens in practice.
__device__ unsigned long long flash_next_route_fused_arrivals = 0ULL;

namespace {

constexpr int kExperts = 512;
constexpr int kTopK    = 10;
constexpr int kHidden  = 2'560;

constexpr unsigned kFullMask = 0xFFFF'FFFFU;

struct RankedValue {
    float value;
    int id;
    int origin;
};

// The selection order: higher score first, equal scores (float ==, so -0.0 ties +0.0) broken by
// the lower expert id. Any NaN operand makes this false in both directions.
__device__ __forceinline__ bool better(float left_value, int left_id, float right_value,
                                       int right_id) {
    return left_value > right_value || (left_value == right_value && left_id < right_id);
}

__device__ __forceinline__ bool better(const RankedValue& left, const RankedValue& right) {
    return better(left.value, left.id, right.value, right.id);
}

__device__ __forceinline__ RankedValue warp_best(RankedValue value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        RankedValue other;
        other.value  = __shfl_down_sync(0xFFFF'FFFFU, value.value, offset);
        other.id     = __shfl_down_sync(0xFFFF'FFFFU, value.id, offset);
        other.origin = __shfl_down_sync(0xFFFF'FFFFU, value.origin, offset);
        if (better(other, value)) { value = other; }
    }
    value.value  = __shfl_sync(0xFFFF'FFFFU, value.value, 0);
    value.id     = __shfl_sync(0xFFFF'FFFFU, value.id, 0);
    value.origin = __shfl_sync(0xFFFF'FFFFU, value.origin, 0);
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    return __shfl_sync(0xFFFF'FFFFU, value, 0);
}

// Legacy selection kernel (one warp per token over a [513, T] FP32 score workspace). This is the
// bit-exact reference for the fused decode kernel below and the selection stage of the prefill
// path; do not change its arithmetic.
__global__ void route_kernel(const float* __restrict__ scores, std::int32_t* __restrict__ ids,
                             float* __restrict__ alpha, float* __restrict__ shared_scale,
                             int tokens) {
    __shared__ float selected_logits[kTopK];
    const int token           = static_cast<int>(blockIdx.x);
    const int lane            = static_cast<int>(threadIdx.x) & 31;
    if (token >= tokens) { return; }
    const float* token_scores = scores + static_cast<std::int64_t>(token) * (kExperts + 1);

    RankedValue local[16];
#pragma unroll
    for (int item = 0; item < 16; ++item) {
        const int id = lane + item * 32;
        local[item]  = {token_scores[id], id, lane};
    }
#pragma unroll
    for (int item = 1; item < 16; ++item) {
        const RankedValue value = local[item];
        int position            = item;
        while (position > 0 && better(value, local[position - 1])) {
            local[position] = local[position - 1];
            --position;
        }
        local[position] = value;
    }

    int cursor = 0;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        RankedValue candidate =
            cursor < 16 ? local[cursor] : RankedValue{-CUDART_INF_F, INT_MAX, lane};
        const RankedValue winner = warp_best(candidate);
        if (lane == 0) {
            ids[token * kTopK + rank] = winner.id;
            selected_logits[rank]     = winner.value;
        }
        if (lane == winner.origin) { ++cursor; }
        __syncwarp();
    }
    const float top_0 = selected_logits[0];
    const float weight =
        lane < kTopK ? expf(selected_logits[lane] - top_0) : 0.0F;
    const float denominator = warp_sum(weight);
    if (lane < kTopK) {
        alpha[token * kTopK + lane] = weight / denominator;
    }
    if (lane == 0) {
        const float gate    = token_scores[kExperts];
        shared_scale[token] = 1.0F / (1.0F + expf(-gate));
    }
}

__device__ __forceinline__ float warp_sum_lane0(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFF'FFFFU, value, offset);
    }
    return value;
}

// Legacy decode projection: one 256-thread CTA per (row, token). Thread tid accumulates columns
// tid + 256*j (j = 0..9) with acc = fmaf(w, x, acc) from 0, each warp reduces with the shfl_down
// tree (offsets 16, 8, 4, 2, 1), and warp 0 reduces the eight warp sums (lanes 0..7, zeros
// elsewhere) with the same tree. The FP32 score is what the selection compares, so the fused
// kernel reproduces this exact sequence.
__global__ void route_projection_kernel(const __nv_bfloat16* __restrict__ input,
                                        const __nv_bfloat16* __restrict__ router,
                                        const __nv_bfloat16* __restrict__ shared_gate,
                                        float* __restrict__ scores, int tokens) {
    __shared__ float partial[8];
    const int row   = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    if (row >= kExperts + 1 || token >= tokens) { return; }

    const __nv_bfloat16* weight =
        row < kExperts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate;
    const __nv_bfloat16* in_tok = input + static_cast<std::int64_t>(token) * kHidden;

    float acc = 0.0F;
    for (int column = tid; column < kHidden; column += static_cast<int>(blockDim.x)) {
        acc = fmaf(__bfloat162float(weight[column]), __bfloat162float(in_tok[column]), acc);
    }
    const float sum = warp_sum_lane0(acc);
    if (lane == 0) { partial[warp] = sum; }
    __syncthreads();
    if (warp == 0) {
        float warp_sum = (lane < 8) ? partial[lane] : 0.0F;
        warp_sum       = warp_sum_lane0(warp_sum);
        if (lane == 0) {
            scores[static_cast<std::int64_t>(token) * (kExperts + 1) + row] = warp_sum;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Fused decode router (T <= 8): projection + top-10 selection + renormalisation + shared gate in
// ONE launch, bit-identical to route_projection_kernel followed by route_kernel.
//
// Measured before this kernel (nsys, per token, 48 layers, T = 1): route_projection_kernel
// 0.174 ms + route_kernel 0.501 ms = 0.675 ms and 96 launches for ~2.6 MB of router weights
// whose transfer floor is ~1.6 us/layer. The route_kernel cost was the data-dependent insertion
// loop, which indexes local[] at runtime and therefore lives in local memory.
//
// Geometry: kFusedRowBlocks CTAs of 8 warps. Warps 0..3 each own one router row (four rows per
// CTA, 20 KB of weights, so 129 SMs pull the matrix in a single wave with every lane's ten
// 16-byte loads in flight at once); the weights stay in registers while the warp loops over the
// T tokens (weight-stationary: the matrix is read once per launch, not once per token). Every
// CTA then announces its arrival on a device counter, and the last CTA of the launch runs the
// selection with one warp per token (8 warps, so T <= 8), reading the scores back through L2.
// ---------------------------------------------------------------------------------------------

constexpr int kFusedWarps        = 8;                              // selection: one warp per token
constexpr int kFusedThreads      = kFusedWarps * 32;
constexpr int kFusedRowsPerBlock = 4;                              // projection: one row per warp
constexpr int kFusedRowBlocks    = (kExperts + 1 + kFusedRowsPerBlock - 1) / kFusedRowsPerBlock;
constexpr int kFusedMaxTokens    = kFlashNextRouteDecodeMaxTokens;
constexpr int kLegacyThreads     = 256;                            // route_projection_kernel CTA
constexpr int kStepsPerThread    = kHidden / kLegacyThreads;       // columns per legacy thread

static_assert(kHidden % kLegacyThreads == 0, "legacy projection covers the row exactly");
static_assert(kLegacyThreads == 8 * 32, "each lane owns 8 legacy threads (one 16-byte chunk)");
static_assert(kFusedRowBlocks == 129, "64 CTAs of expert rows plus one CTA for the gate row");
static_assert(kFusedRowBlocks * kFusedRowsPerBlock >= kExperts + 1, "every row has a warp");
static_assert(kFusedRowsPerBlock <= kFusedWarps, "projection rows fit the CTA");
static_assert(kFusedMaxTokens <= kFusedWarps, "selection has one warp per token");
static_assert(kTopK <= 16, "a lane's cursor never runs past its 16 sorted items");
static_assert(kExperts == 16 * 32, "each lane sorts exactly 16 expert scores");

// Score of one router row for one token, bit-identical to route_projection_kernel.
//
// Legacy thread tid (0..255) accumulates columns tid + 256*j, j = 0..9. Lane `lane` here owns
// legacy threads 8*lane + i (i = 0..7), so its columns for step j are the contiguous run
// 256*j + 8*lane .. +7: one 16-byte load, and acc[i] runs exactly the legacy fmaf chain of thread
// 8*lane + i. Legacy warp w (threads 32w..32w+31) is therefore held by physical lanes 4w..4w+3,
// with lane 4w+q holding legacy lanes 8q+i. The legacy tree adds lane vl with vl+16, then vl+8,
// then vl+4, vl+2, vl+1 (only lane 0's chain matters): vl+16 sits two physical lanes up and vl+8
// one lane up, and the last three levels stay inside lane 4w's own eight registers. The eight
// warp sums then go through the legacy second level unchanged (lanes 0..7 loaded, zeros
// elsewhere, same tree, including the exact +0.0F additions).
__device__ __forceinline__ float fused_row_score(const uint4 (&w)[kStepsPerThread],
                                                 const uint4* __restrict__ x_chunks, int lane) {
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] = 0.0F; }
#pragma unroll
    for (int j = 0; j < kStepsPerThread; ++j) {
        const uint4 x_raw = __ldg(x_chunks + j * 32 + lane);
        const auto* w_bf  = reinterpret_cast<const __nv_bfloat16*>(&w[j]);
        const auto* x_bf  = reinterpret_cast<const __nv_bfloat16*>(&x_raw);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[i] = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), acc[i]);
        }
    }
    // Legacy offset 16: legacy lane vl pairs with vl+16 = physical lane + 2.
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] += __shfl_down_sync(kFullMask, acc[i], 2); }
    // Legacy offset 8: vl pairs with vl+8 = physical lane + 1.
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc[i] += __shfl_down_sync(kFullMask, acc[i], 1); }
    // Legacy offsets 4, 2, 1 inside lane 4w's registers (legacy lanes 0..7 of warp w).
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc[i] += acc[i + 4]; }
    acc[0] += acc[2];
    acc[1] += acc[3];
    const float warp_partial = acc[0] + acc[1]; // legacy partial[w] on physical lane 4w
    // Legacy second level: partial[lane] for lane < 8, 0.0F elsewhere, same tree.
    const float gathered = __shfl_sync(kFullMask, warp_partial, lane < 8 ? lane * 4 : lane);
    return warp_sum_lane0(lane < 8 ? gathered : 0.0F);
}

struct RankedPair {
    float value;
    int id;
};

__device__ __forceinline__ bool better(const RankedPair& left, const RankedPair& right) {
    return better(left.value, left.id, right.value, right.id);
}

// Same shfl_down tree and predicate as warp_best; the origin lane is recovered from the id
// instead of shuffled (see fused_select_token).
__device__ __forceinline__ RankedPair warp_best(RankedPair value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        RankedPair other;
        other.value = __shfl_down_sync(kFullMask, value.value, offset);
        other.id    = __shfl_down_sync(kFullMask, value.id, offset);
        if (better(other, value)) { value = other; }
    }
    value.value = __shfl_sync(kFullMask, value.value, 0);
    value.id    = __shfl_sync(kFullMask, value.id, 0);
    return value;
}

// One warp selects a token's top-10, bit-identical to route_kernel for every input (ties, signed
// zeros, infinities and NaNs included), because it is the same algorithm with static register
// indexing:
//  * Lane l loads items id = l + 32*item, item = 0..15, in the same order.
//  * The legacy per-lane insertion sort moves the item at position `item` up while
//    better(item, above) holds. Written as the fixed compare-exchange sequence
//    j = item, item-1, ..., 1 it is identical: the moving item stops at the same place, and the
//    remaining compare-exchanges look at adjacent pairs of the already-inserted prefix, which
//    never swap because every adjacent prefix pair (a above b) has better(b, a) == false (b was
//    placed below a either because better(b, a) failed or because a passed b, and a passing b
//    means both are non-NaN with a strictly ahead of b). That holds for NaN too, since better is
//    false in both directions whenever a NaN is involved.
//  * Each round shuffles the lanes' heads through the same tree. The legacy cursor is a
//    register shift of the winning lane's list (kTopK <= 16, so the cursor never runs out and
//    the -inf/INT_MAX sentinel branch is dead). The winning lane is id & 31 because every
//    candidate carries id = lane + 32*item and no sentinel is ever offered, which is exactly the
//    origin field route_kernel shuffles along.
//  * Renormalisation and the gate are the same expressions in the same order; the selected
//    logits live in lane `rank` instead of shared memory.
__device__ __forceinline__ void fused_select_token(const float* __restrict__ token_scores,
                                                   std::int32_t* __restrict__ ids,
                                                   float* __restrict__ alpha,
                                                   float* __restrict__ shared_scale, int lane) {
    RankedPair local[16];
#pragma unroll
    for (int item = 0; item < 16; ++item) {
        const int id = lane + item * 32;
        local[item]  = {__ldcg(token_scores + id), id};
    }
#pragma unroll
    for (int item = 1; item < 16; ++item) {
#pragma unroll
        for (int j = item; j > 0; --j) {
            if (better(local[j], local[j - 1])) {
                const RankedPair moving = local[j];
                local[j]                = local[j - 1];
                local[j - 1]            = moving;
            }
        }
    }

    float my_logit = 0.0F; // lane r keeps the r-th selected logit (route_kernel: selected_logits[r])
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        const RankedPair winner = warp_best(local[0]);
        if (lane == 0) { ids[rank] = winner.id; }
        if (lane == rank) { my_logit = winner.value; }
        if (lane == (winner.id & 31)) {
#pragma unroll
            for (int item = 0; item < 15; ++item) { local[item] = local[item + 1]; }
        }
    }
    const float top_0       = __shfl_sync(kFullMask, my_logit, 0);
    const float weight      = lane < kTopK ? expf(my_logit - top_0) : 0.0F;
    const float denominator = warp_sum(weight);
    if (lane < kTopK) { alpha[lane] = weight / denominator; }
    if (lane == 0) {
        const float gate = __ldcg(token_scores + kExperts);
        *shared_scale    = 1.0F / (1.0F + expf(-gate));
    }
}

__global__ void __launch_bounds__(kFusedThreads)
route_fused_decode_kernel(const __nv_bfloat16* __restrict__ input,
                          const __nv_bfloat16* __restrict__ router,
                          const __nv_bfloat16* __restrict__ shared_gate,
                          float* __restrict__ scores, std::int32_t* __restrict__ ids,
                          float* __restrict__ alpha, float* __restrict__ shared_scale,
                          int tokens) {
    __shared__ bool last_arrival;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row  = static_cast<int>(blockIdx.x) * kFusedRowsPerBlock + warp;

    // Projection: warps 0..3 own rows 4b..4b+3 (CTA 128 owns only the gate row, 512). Warps
    // without a row skip straight to the arrival barrier; nobody returns before it.
    if (warp < kFusedRowsPerBlock && row <= kExperts) {
        const __nv_bfloat16* weight =
            row < kExperts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate;
        const auto* w_chunks = reinterpret_cast<const uint4*>(weight);
        uint4 w[kStepsPerThread];
#pragma unroll
        for (int j = 0; j < kStepsPerThread; ++j) { w[j] = __ldg(w_chunks + j * 32 + lane); }
        for (int token = 0; token < tokens; ++token) {
            const auto* x_chunks = reinterpret_cast<const uint4*>(
                input + static_cast<std::int64_t>(token) * kHidden);
            const float score = fused_row_score(w, x_chunks, lane);
            if (lane == 0) {
                scores[static_cast<std::int64_t>(token) * (kExperts + 1) + row] = score;
            }
        }
    }

    // Arrival (CUDA programming guide "memory fence" pattern): every writer fences its scores to
    // device scope, the CTA barrier makes them precede thread 0's ticket, and the CTA that
    // completes a group of kFusedRowBlocks tickets is the last of this launch, so every other
    // CTA's scores are published by the time it reads them.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence(); // cumulative: this CTA's stores precede its ticket
        const unsigned long long ticket = atomicAdd(&flash_next_route_fused_arrivals, 1ULL);
        last_arrival = ((ticket + 1ULL) % static_cast<unsigned long long>(kFusedRowBlocks)) == 0ULL;
        __threadfence(); // the ticket precedes this CTA's reads of the others' scores
    }
    __syncthreads();
    if (!last_arrival) { return; }

    // Selection: warp t handles token t. The scores are read with __ldcg (L2) so that no L1 line
    // from before the fence is consulted. No block-level barrier follows, so warps may leave.
    if (warp >= tokens) { return; }
    fused_select_token(scores + static_cast<std::int64_t>(warp) * (kExperts + 1),
                       ids + warp * kTopK, alpha + warp * kTopK, shared_scale + warp, lane);
}

// Prefill router projection: Tiled GEMM amortizing router weights over tokens
// Grid.x = (513 + 31) / 32 = 17 blocks, Grid.y = (tokens + 15) / 16 blocks.
// Inside each block, weights are loaded ONCE and multiplied with 16 tokens in registers.
__global__ void __launch_bounds__(256)
route_prefill_projection_kernel(const __nv_bfloat16* __restrict__ input,
                                const __nv_bfloat16* __restrict__ router,
                                const __nv_bfloat16* __restrict__ shared_gate,
                                float* __restrict__ scores, int tokens) {
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int row_base     = (static_cast<int>(blockIdx.x) * 8 + warp) * 4;
    const int token_base   = static_cast<int>(blockIdx.y) * 16;
    const int batch_tokens = min(16, tokens - token_base);

    if (batch_tokens <= 0) { return; }

    float acc[4][16] = {};

    for (int chunk = lane; chunk < (kHidden / 8); chunk += 32) {
        const int col_base = chunk * 8;

        // 1. Load 16 token inputs for this chunk
        ulonglong2 x_raw[16];
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            if (b < batch_tokens) {
                const int t_idx = token_base + b;
                x_raw[b] = *reinterpret_cast<const ulonglong2*>(
                    input + static_cast<std::int64_t>(t_idx) * kHidden + col_base);
            }
        }

        // 2. Load router weight rows ONCE and multiply across all 16 tokens
        #pragma unroll
        for (int r_idx = 0; r_idx < 4; ++r_idx) {
            const int row = row_base + r_idx;
            if (row < kExperts + 1) {
                const __nv_bfloat16* weight =
                    row < kExperts ? router + static_cast<std::int64_t>(row) * kHidden : shared_gate;
                const auto w_raw = *reinterpret_cast<const ulonglong2*>(weight + col_base);
                const auto* w_bf = reinterpret_cast<const __nv_bfloat16*>(&w_raw);

                #pragma unroll
                for (int b = 0; b < 16; ++b) {
                    if (b < batch_tokens) {
                        const auto* x_bf = reinterpret_cast<const __nv_bfloat16*>(&x_raw[b]);
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            acc[r_idx][b] = fmaf(__bfloat162float(w_bf[i]), __bfloat162float(x_bf[i]), acc[r_idx][b]);
                        }
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int r_idx = 0; r_idx < 4; ++r_idx) {
        const int row = row_base + r_idx;
        if (row < kExperts + 1) {
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                if (b < batch_tokens) {
                    const float sum = warp_sum_lane0(acc[r_idx][b]);
                    if (lane == 0) {
                        const int t_idx = token_base + b;
                        scores[static_cast<std::int64_t>(t_idx) * (kExperts + 1) + row] = sum;
                    }
                }
            }
        }
    }
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool exact_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    return weight.qtype == QType::BF16 && weight.layout == QuantLayout::Contiguous &&
           weight.n == rows && weight.k == columns && weight.ndim == 2 && weight.shape[0] == rows &&
           weight.shape[1] == columns && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.qdata == weight.payload &&
           weight.payload_bytes >= static_cast<std::uint64_t>(rows) * columns * 2 &&
           aligned_to(weight.qdata, 16);
}

void validate_route_outputs(const Tensor& scores, const Tensor& ids, const Tensor& alpha,
                            const Tensor& shared_scale, cudaStream_t stream) {
    const std::int32_t tokens = scores.ne[1];
    if (scores.dtype != DType::FP32 || ids.dtype != DType::I32 || alpha.dtype != DType::FP32 ||
        shared_scale.dtype != DType::FP32 || scores.ne[0] != kExperts + 1 || scores.ne[2] != 1 ||
        scores.ne[3] != 1 || ids.ne[0] != kTopK || ids.ne[1] != tokens || ids.ne[2] != 1 ||
        ids.ne[3] != 1 || alpha.ne[0] != kTopK || alpha.ne[1] != tokens || alpha.ne[2] != 1 ||
        alpha.ne[3] != 1 || shared_scale.ne[0] != tokens || shared_scale.ne[1] != 1 ||
        shared_scale.ne[2] != 1 || shared_scale.ne[3] != 1 || !scores.is_contiguous() ||
        !ids.is_contiguous() || !alpha.is_contiguous() || !shared_scale.is_contiguous() ||
        !aligned_to(scores.data, 16) || !aligned_to(ids.data, 16) || !aligned_to(alpha.data, 16) ||
        !aligned_to(shared_scale.data, 16) || stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next route requires aligned exact [513,T] top-10 tensors");
    }
}

void validate_route_inputs(const Tensor& input, const Weight& router, const Weight& shared_gate,
                           const Tensor& score_workspace, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || input.ne[0] != kHidden || input.ne[2] != 1 ||
        input.ne[3] != 1 || tokens < 1 || !input.is_contiguous() ||
        !aligned_to(input.data, 16) || !exact_bf16_weight(router, kExperts, kHidden) ||
        !exact_bf16_weight(shared_gate, 1, kHidden) || score_workspace.dtype != DType::FP32 ||
        score_workspace.ne[0] != kExperts + 1 || score_workspace.ne[1] != tokens ||
        score_workspace.ne[2] != 1 || score_workspace.ne[3] != 1 ||
        !score_workspace.is_contiguous() || !aligned_to(score_workspace.data, 16) ||
        stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next router requires exact aligned BF16 [2560,T] inputs");
    }
}

void validate_decode_shape(const Tensor& input) {
    if (input.ne[1] > kFusedMaxTokens) {
        throw std::invalid_argument("Flash-Next decode router paths take at most 8 tokens");
    }
}

} // namespace

bool flash_next_route_legacy_enabled() {
    static const bool legacy = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_ROUTE_LEGACY");
        return env != nullptr && env[0] == '1' && env[1] == '\0';
    }();
    return legacy;
}

void flash_next_route_scores(const Tensor& scores, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                             cudaStream_t stream) {
    validate_route_outputs(scores, ids, alpha, shared_scale, stream);
    const std::int32_t tokens = scores.ne[1];
    dim3 grid(static_cast<unsigned>(tokens), 1);
    route_kernel<<<grid, 32, 0, stream>>>(
        static_cast<const float*>(scores.data), static_cast<std::int32_t*>(ids.data),
        static_cast<float*>(alpha.data), static_cast<float*>(shared_scale.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_route_decode_legacy(const Tensor& input, const Weight& router,
                                    const Weight& shared_gate, Tensor& score_workspace,
                                    Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                                    cudaStream_t stream) {
    validate_route_inputs(input, router, shared_gate, score_workspace, stream);
    validate_decode_shape(input);
    const std::int32_t tokens = input.ne[1];
    dim3 grid(static_cast<unsigned>(kExperts + 1), static_cast<unsigned>(tokens));
    route_projection_kernel<<<grid, kLegacyThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(router.qdata),
        static_cast<const __nv_bfloat16*>(shared_gate.qdata),
        static_cast<float*>(score_workspace.data), tokens);
    CUDA_CHECK(cudaGetLastError());
    flash_next_route_scores(score_workspace, ids, alpha, shared_scale, stream);
}

void flash_next_route_decode_fused(const Tensor& input, const Weight& router,
                                   const Weight& shared_gate, Tensor& score_workspace,
                                   Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                                   cudaStream_t stream) {
    validate_route_inputs(input, router, shared_gate, score_workspace, stream);
    validate_decode_shape(input);
    validate_route_outputs(score_workspace, ids, alpha, shared_scale, stream);
    const std::int32_t tokens = input.ne[1];
    route_fused_decode_kernel<<<kFusedRowBlocks, kFusedThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(router.qdata),
        static_cast<const __nv_bfloat16*>(shared_gate.qdata),
        static_cast<float*>(score_workspace.data), static_cast<std::int32_t*>(ids.data),
        static_cast<float*>(alpha.data), static_cast<float*>(shared_scale.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_route(const Tensor& input, const Weight& router, const Weight& shared_gate,
                      Tensor& score_workspace, Tensor& ids, Tensor& alpha, Tensor& shared_scale,
                      cudaStream_t stream) {
    validate_route_inputs(input, router, shared_gate, score_workspace, stream);
    const std::int32_t tokens = input.ne[1];
    if (tokens <= kFusedMaxTokens) {
        if (flash_next_route_legacy_enabled()) {
            flash_next_route_decode_legacy(input, router, shared_gate, score_workspace, ids, alpha,
                                           shared_scale, stream);
        } else {
            flash_next_route_decode_fused(input, router, shared_gate, score_workspace, ids, alpha,
                                          shared_scale, stream);
        }
        return;
    }
    if (flash_next_moe_shared_mma_enabled()) {
        flash_next_route_projection_mma(input, router, shared_gate, score_workspace, stream);
    } else {
        const dim3 grid((kExperts + 1 + 31) / 32, (static_cast<unsigned>(tokens) + 15) / 16);
        route_prefill_projection_kernel<<<grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(input.data),
            static_cast<const __nv_bfloat16*>(router.qdata),
            static_cast<const __nv_bfloat16*>(shared_gate.qdata),
            static_cast<float*>(score_workspace.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }

    flash_next_route_scores(score_workspace, ids, alpha, shared_scale, stream);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
