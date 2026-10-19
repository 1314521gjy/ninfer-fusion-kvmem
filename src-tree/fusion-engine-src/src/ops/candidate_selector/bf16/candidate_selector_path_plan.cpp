#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"
#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
// NOTE: candidate_selector_tree.h is deliberately NOT included here. This is a host-only .cpp
// compiled by MSVC; that header reaches the CUDA device headers, which cl cannot parse. The tree's
// host entry point is declared host-safely in candidate_selector_path_plan.h instead.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::ops::detail {
SelectorRoute candidate_selector_path_route(int steps, int batch) {
    if (steps < 1 || steps > 15 || batch < 1 || batch > 8)
        throw std::invalid_argument("invalid selector K/B");
    return steps <= 4 ? SelectorRoute::Direct : SelectorRoute::Lattice;
}

const char* selector_route_name(SelectorRoute route) {
    return route == SelectorRoute::Direct ? "direct.t512" : "lattice.t512";
}

const char* candidate_selector_path_route_name(int steps, int batch) {
    return selector_route_name(candidate_selector_path_route(steps, batch));
}

// Both readers use the flag semantics of expert_staging.cpp's read_flag_once (env non-null, non-empty
// and not "0"), read once per process. Unset and "0" therefore BOTH mean off, so there is no
// "0 means unset" fallback for a negative control to fall into (the kvmem_score.h:100 accident).
bool selector_tree_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_DRAFT_TREE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool selector_tree_sabotage() {
    static const bool sabotage = [] {
        const char* value = std::getenv("NINFER_DRAFT_TREE_SABOTAGE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return sabotage;
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Tensor& predecessor_codebook,
                                      const Tensor& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                                      cudaStream_t stream) {
    const auto route = candidate_selector_path_route(candidate_ids.ne[1], candidate_ids.ne[2]);
    auto scope       = workspace.scope();
    // with_tree = true always: the buffers are allocated whether or not the switch is on, so the
    // caller's reservation (candidate_selector_path_workspace_capacity_bytes) does not depend on the
    // environment. ~2 KB at K=15, B=8.
    const auto scratch = allocate_selector_workspace(workspace, route, candidate_ids.ne[1],
                                                     candidate_ids.ne[2], /*with_tree=*/true);
    const Tensor* live[]{
        &candidate_ids,      &unary_scores,   &projected_hidden, &anchors,   &predecessor_codebook,
        &successor_codebook, &base_positions, &drafts,           &proposal_q};
    for (const auto* work : {&scratch.edges, &scratch.tree_tokens, &scratch.tree_parents,
                             &scratch.tree_depths, &scratch.tree_masks, &scratch.tree_live}) {
        if (!work->data) continue;
        const auto begin = reinterpret_cast<std::uintptr_t>(work->data),
                   end   = begin + work->bytes();
        for (const auto* tensor : live) {
            const auto tb = reinterpret_cast<std::uintptr_t>(tensor->data);
            if (begin < tb + tensor->bytes() && tb < end)
                throw std::invalid_argument("selector workspace overlaps operand");
        }
        const auto cb = reinterpret_cast<std::uintptr_t>(configs);
        if (begin < cb + candidate_ids.ne[2] * sizeof(SamplingConfig) && cb < end)
            throw std::invalid_argument("selector workspace overlaps configs");
    }
    if (selector_tree_enabled()) {
        // ONE witness line per process: a switch whose effect cannot be seen in the log is
        // indistinguishable from a switch that never fired (expert_staging.cpp's three recorded
        // silent-no-op accidents all looked like this).
        static bool witnessed = false;
        if (!witnessed) {
            witnessed = true;
            std::fprintf(stderr, "[selector_tree] enabled nodes=%d sabotage=%d route=%s\n",
                         candidate_ids.ne[1] + 1, selector_tree_sabotage() ? 1 : 0,
                         selector_route_name(route));
        }
        candidate_selector_tree_dispatch(candidate_ids, unary_scores, projected_hidden, anchors,
                                         predecessor_codebook, successor_codebook, base_positions,
                                         configs, drafts, proposal_q, scratch,
                                         selector_tree_sabotage(), stream);
        return;
    }
    candidate_selector_path_launch(route, candidate_ids, unary_scores, projected_hidden, anchors,
                                   predecessor_codebook, successor_codebook, base_positions,
                                   configs, drafts, proposal_q, scratch, stream);
}
} // namespace ninfer::ops::detail
