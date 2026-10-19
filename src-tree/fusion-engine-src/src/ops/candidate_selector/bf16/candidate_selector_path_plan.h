#pragma once
#include "core/tensor.h"
#include "core/layout.h"
#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
enum class SelectorRoute { Direct, Lattice };

struct SelectorWorkspace {
    Tensor edges;
    // S-A tree builder buffers. Present whether or not the tree switch is on, so that the caller's
    // workspace reservation (candidate_selector_path_workspace_capacity_bytes, used by
    // planning/startup.cpp) does NOT depend on an environment variable -- a reservation that
    // changes with the environment is how a caller ends up short of scratch.
    Tensor tree_tokens;   // [batch, nodes] I32
    Tensor tree_parents;  // [batch, nodes] I32
    Tensor tree_depths;   // [batch, nodes] I32
    Tensor tree_masks;    // [batch, nodes] I32
    Tensor tree_live;     // [batch] I32
};

SelectorRoute candidate_selector_path_route(int steps, int batch);
const char* candidate_selector_path_route_name(int steps, int batch);
const char* selector_route_name(SelectorRoute route);

// True when the DFlash2 tree builder replaces the chain draw. Read ONCE per process; unset and "0"
// both mean off (the shipped behaviour), which is what makes the OFF arm the natural negative
// control. The form (static const bool + immediately invoked lambda) follows this tree's op-layer
// convention, e.g. ops/linear/t2/t2_dispatch.cpp:17-25.
[[nodiscard]] bool selector_tree_enabled();
// Deliberate breakage for the red arm: the tree draws with counter index 1 instead of 0, so the
// chain it builds must STOP matching candidate_selector_path. A judgement nobody has seen fail is
// not a judgement (expert_staging.h:35-37).
[[nodiscard]] bool selector_tree_sabotage();

template <class Allocator>
SelectorWorkspace allocate_selector_workspace(Allocator& allocator, SelectorRoute route, int steps,
                                              int batch, bool with_tree = false) {
    SelectorWorkspace out;
    if (route == SelectorRoute::Lattice)
        out.edges = allocator.alloc(DType::FP32, {16, 16, steps, batch}, 256);
    if (with_tree) {
        const int nodes = steps + 1;  // column 0 is the anchor (tree-verification.md:31)
        out.tree_tokens  = allocator.alloc(DType::I32, {batch, nodes}, 256);
        out.tree_parents = allocator.alloc(DType::I32, {batch, nodes}, 256);
        out.tree_depths  = allocator.alloc(DType::I32, {batch, nodes}, 256);
        out.tree_masks   = allocator.alloc(DType::I32, {batch, nodes}, 256);
        out.tree_live    = allocator.alloc(DType::I32, {batch}, 256);
    }
    return out;
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Tensor& predecessor_codebook,
                                      const Tensor& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                                      cudaStream_t stream);

// S-A tree builder, implemented in candidate_selector_tree.cu (nvcc). Declared HERE, in the
// host-safe header, and NOT in candidate_selector_tree.h: that header pulls in the CUDA device
// headers (ops/common/memory.cuh, ops/kernel/sampling_device.cuh), which the MSVC translation unit
// that calls this (candidate_selector_path_plan.cpp) cannot parse -- it fails inside
// cuda_pipeline_helpers.h with "unexpected volatile". Host code declares, device code defines.
void candidate_selector_tree_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Tensor& predecessor_codebook,
                                      const Tensor& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q,
                                      const SelectorWorkspace& workspace, bool sabotage,
                                      cudaStream_t stream);
} // namespace ninfer::ops::detail
