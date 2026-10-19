#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "ops/candidate_selector/bf16/candidate_selector_common.cuh"
#include "core/device.h"
#include "core/pdl.cuh"
#include <stdexcept>

namespace ninfer::ops::detail {

__global__ __launch_bounds__(512, 1) void selector_walk_kernel(DeviceArgs a) {
    pdl::enter();
    __shared__ SelectorShared shared;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor   = a.anchors[batch];
        shared.base_position = a.positions[batch];
        shared.temperature   = a.configs[batch].temperature;
        shared.seed          = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        selector_score_row(a, column, shared);
        if (warp == 0) {
            const float edge =
                lane < kSelectorCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = selector_draw_rank(edge, shared.temperature, shared.seed,
                                                    shared.base_position + step, 0U,
                                                    a.q + column * kSelectorCandidates);
            if (lane == 0) {
                shared.predecessor = a.ids[column * kSelectorCandidates + selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

__global__ __launch_bounds__(512, 2) void selector_lattice_kernel(DeviceArgs a, float* edges) {
    pdl::enter();
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if (threadIdx.x == 0)
        shared.predecessor =
            step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kSelectorCandidates + p];
    selector_score_row(a, column, shared);
    if (threadIdx.x < kSelectorCandidates)
        edges[(static_cast<std::int64_t>(column) * kSelectorCandidates + p) *
                  kSelectorCandidates +
              threadIdx.x] = shared.edge[threadIdx.x];
}

__global__ __launch_bounds__(32) void selector_lattice_walk_kernel(DeviceArgs a,
                                                                   const float* edges) {
    pdl::enter();
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kSelectorCandidates
                ? edges[(static_cast<std::int64_t>(column) * kSelectorCandidates +
                         predecessor_rank) *
                            kSelectorCandidates +
                        lane]
                : -CUDART_INF_F;
        int selected = selector_draw_rank(edge, temperature, seed, position + step, 0U,
                                          a.q + column * kSelectorCandidates);
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kSelectorCandidates + selected];
    }
}

void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream) {
    const DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                          static_cast<const float*>(unary_scores.data),
                          static_cast<const __nv_bfloat16*>(projected_hidden.data),
                          static_cast<const std::int32_t*>(anchors.data),
                          static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
                          static_cast<const __nv_bfloat16*>(successor_codebook.data),
                          static_cast<const std::int32_t*>(base_positions.data),
                          configs,
                          static_cast<std::int32_t*>(drafts.data),
                          static_cast<float*>(proposal_q.data),
                          candidate_ids.ne[1]};
    if (route == SelectorRoute::Direct) {
        CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(512), 0, stream},
                                        selector_walk_kernel, args));
    } else {
        auto* edges = static_cast<float*>(workspace.edges.data);
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(args.steps * candidate_ids.ne[2], kSelectorCandidates), dim3(512), 0, stream},
            selector_lattice_kernel, args, edges));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(pdl::launch_consumer({dim3(candidate_ids.ne[2]), dim3(32), 0, stream},
                                        selector_lattice_walk_kernel, args, edges));
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
