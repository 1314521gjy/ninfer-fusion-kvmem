#include "ops/candidate_selector/bf16/candidate_selector_tree.h"
#include "ops/candidate_selector/bf16/candidate_selector_common.cuh"
#include "core/device.h"
#include "core/pdl.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

// The chain-shaped tree (nodes == steps + 1). Column 0 is the anchor; column c >= 1 is main-chain
// column c, whose token is drawn EXACTLY as the chain round draws it -- same selector_draw_rank,
// same counter index 0, same position (base_position + step) -- which is what makes
// tree-verification.md:264 ("main chain bit-identical to candidate_selector_path") hold by
// construction. Node-at-a-time on purpose: the side columns of the document's builder append to
// this loop, and they are the only thing missing here.
//
// Ancestor mask semantics are the document's: bit a of column c is set iff a == c or a is an
// ancestor of c (tree-verification.md:45-46). On the chain that is the causal prefix
// (1 << (c + 1)) - 1, and the main-chain column's ancestor set is exactly the preceding columns --
// which is also why every main-chain quantity has the chain definition (:143-144).
__global__ __launch_bounds__(512, 1) void selector_tree_chain_kernel(SelectorTreeArgs a) {
    pdl::enter();
    __shared__ SelectorShared shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    const int base = batch * a.nodes;
    if (tid == 0) {
        shared.predecessor   = a.base.anchors[batch];
        shared.base_position = a.base.positions[batch];
        shared.temperature   = a.base.configs[batch].temperature;
        shared.seed          = a.base.configs[batch].seed;
        // Column 0: the anchor. It has no parent, depth 0 and an empty ancestor set.
        a.tokens[base]  = a.base.anchors[batch];
        a.parents[base] = -1;
        a.depths[base]  = 0;
        a.masks[base]   = 0;
        a.live[batch]   = a.nodes;
    }
    __syncthreads();
#pragma unroll 1
    for (int step = 0; step < a.base.steps; ++step) {
        const int in_column  = batch * a.base.steps + step;  // input plane column
        const int out_column = step + 1;                     // tree column
        selector_score_row(a.base, in_column, shared);
        if (warp == 0) {
            const float edge = lane < kSelectorCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = selector_draw_rank(edge, shared.temperature, shared.seed,
                                                    shared.base_position + step,
                                                    static_cast<unsigned int>(a.draw_index),
                                                    a.base.q + in_column * kSelectorCandidates);
            if (lane == 0) {
                const int token  = a.base.ids[in_column * kSelectorCandidates + selected];
                shared.predecessor = token;
                a.base.drafts[in_column]                 = token;  // same place the chain writes
                a.tokens[base + out_column]              = token;
                a.parents[base + out_column]             = out_column - 1;
                a.depths[base + out_column]              = out_column;
                a.masks[base + out_column]               = (1 << (out_column + 1)) - 1;
            }
        }
        __syncthreads();
    }
}

void candidate_selector_tree_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                    const Tensor& projected_hidden, const Tensor& anchors,
                                    const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook,
                                    const Tensor& base_positions, const SamplingConfig* configs,
                                    Tensor& drafts, Tensor& proposal_q, const Tensor& tokens,
                                    const Tensor& parents, const Tensor& depths, const Tensor& masks,
                                    const Tensor& live, std::int32_t nodes, bool sabotage,
                                    cudaStream_t stream) {
    const std::int32_t steps = candidate_ids.ne[1];
    const std::int32_t batch = candidate_ids.ne[2];
    if (nodes != steps + 1) {
        // The side columns are the next block of this port. Refusing beats silently building a
        // chain and calling it a tree (tree-verification.md:114-120 describes what is missing).
        throw std::invalid_argument("candidate_selector_tree: only the chain-shaped tree is built");
    }
    const SelectorTreeArgs args{
        DeviceArgs{static_cast<const std::int32_t*>(candidate_ids.data),
                   static_cast<const float*>(unary_scores.data),
                   static_cast<const __nv_bfloat16*>(projected_hidden.data),
                   static_cast<const std::int32_t*>(anchors.data),
                   static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
                   static_cast<const __nv_bfloat16*>(successor_codebook.data),
                   static_cast<const std::int32_t*>(base_positions.data),
                   configs,
                   static_cast<std::int32_t*>(drafts.data),
                   static_cast<float*>(proposal_q.data),
                   steps},
        nodes,
        sabotage ? 1 : 0,
        static_cast<std::int32_t*>(tokens.data),
        static_cast<std::int32_t*>(parents.data),
        static_cast<std::int32_t*>(depths.data),
        static_cast<std::int32_t*>(masks.data),
        static_cast<std::int32_t*>(live.data)};
    CUDA_CHECK(pdl::launch_consumer({dim3(batch), dim3(512), 0, stream}, selector_tree_chain_kernel,
                                    args));
    CUDA_CHECK(cudaGetLastError());
}

// Host entry point used by candidate_selector_path_dispatch (a host-only .cpp). It lives here
// because the workspace members and the CUDA headers are only visible to an nvcc translation unit.
void candidate_selector_tree_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Tensor& predecessor_codebook,
                                      const Tensor& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q,
                                      const SelectorWorkspace& workspace, bool sabotage,
                                      cudaStream_t stream) {
    candidate_selector_tree_launch(candidate_ids, unary_scores, projected_hidden, anchors,
                                   predecessor_codebook, successor_codebook, base_positions,
                                   configs, drafts, proposal_q, workspace.tree_tokens,
                                   workspace.tree_parents, workspace.tree_depths,
                                   workspace.tree_masks, workspace.tree_live,
                                   candidate_ids.ne[1] + 1, sabotage, stream);
}

} // namespace ninfer::ops::detail
