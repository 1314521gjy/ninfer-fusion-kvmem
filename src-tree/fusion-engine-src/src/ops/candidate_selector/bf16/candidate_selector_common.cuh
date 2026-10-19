#pragma once
// Shared DFlash2 proposal machinery: the predecessor-conditional edge lattice and the one draw.
//
// WHY THIS FILE EXISTS
//   `candidate_selector_path` draws the chain proposal. A tree proposal
//   (docs/maintainer/tree-verification.md of the Infernix line) must draw that same chain BIT FOR
//   BIT -- the tree document makes it the builder's first acceptance criterion ("Builder: main
//   chain bit-identical to `candidate_selector_path`"). The only way to hold that by construction
//   is for both to call the SAME draw: a second copy would be free to drift, and this tree has
//   already recorded the "two copies of one routine" failure mode more than once.
//
// THE DRAW'S COUNTER KEY
//   (seed, position, kSamplePurposeDFlash2Proposal, index). The chain passes index 0. A tree's
//   side column passes ((column + 1) << 5) | sibling, which is the encoding the tree document
//   specifies for sibling j of a node v. A tree's MAIN-chain column keeps index 0 -- that is
//   exactly what makes it bit-identical to the chain.
//
// MOVED, NOT REWRITTEN
//   Everything below was moved verbatim out of candidate_selector_path.cu. The bodies are
//   unchanged; the only additions are `__forceinline__` (so the header is safe to include from
//   more than one translation unit) and the `index` argument of selector_draw_rank.
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail {

inline constexpr int kSelectorCandidates = 16;
inline constexpr int kSelectorRank       = 256;

struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    const __nv_bfloat16* predecessor;
    const __nv_bfloat16* successor;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    int steps;
};

struct alignas(16) SelectorShared {
    __nv_bfloat16 successors[kSelectorCandidates * kSelectorRank];
    float product[kSelectorRank];
    float edge[kSelectorCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

// The probabilities written here are the same FP32 values consumed by the draw.
// `index` is the sibling slot of the draw: 0 for the chain (and for a tree's main chain),
// ((column + 1) << 5) | sibling for a tree's side column.
__device__ __forceinline__ int selector_draw_rank(float edge, float temperature,
                                                  unsigned long long seed, int position,
                                                  unsigned int index, float* q) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kSelectorCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kSelectorCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight =
        lane < kSelectorCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kSelectorCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, index) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kSelectorCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits =
        __ballot_sync(kFullWarpMask, lane < kSelectorCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kSelectorCandidates - 1;
}

__device__ __forceinline__ void selector_score_row(const DeviceArgs& a, int column,
                                                   SelectorShared& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int token = lane == 0 ? a.ids[column * kSelectorCandidates + warp] : 0;
    token     = __shfl_sync(kFullWarpMask, token, 0);
    cp_async<16, Cache::cg>(&shared.successors[warp * kSelectorRank + lane * 8],
                            a.successor + static_cast<std::int64_t>(token) * kSelectorRank +
                                lane * 8);
    cp_commit();
    // Publish the preceding draw before reading its token. Successor prefetch is independent.
    __syncthreads();
    if (tid < kSelectorRank)
        shared.product[tid] =
            __bfloat162float(
                a.predecessor[static_cast<std::int64_t>(shared.predecessor) * kSelectorRank +
                              tid]) *
            __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kSelectorRank + tid]);
    cp_wait<0>();
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
#pragma unroll
        for (int r = lane; r < kSelectorRank; r += 32)
            sum = fmaf(shared.product[r],
                       __bfloat162float(shared.successors[c * kSelectorRank + r]), sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = a.unary[column * kSelectorCandidates + c] + sum;
    }
    __syncthreads();
}

} // namespace ninfer::ops::detail
