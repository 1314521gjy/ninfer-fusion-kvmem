#pragma once
// S-A step ①: the tree builder of the DFlash2 tree round (Infernix line, docs/maintainer/
// tree-verification.md sections "Topology" and "Tree builder"). Detail level only -- the public
// op lives in include/ninfer/ops/candidate_selector.h and is added with the wiring step.
//
// WHAT THIS FIRST BLOCK DOES AND DOES NOT DO
//   It builds the CHAIN-SHAPED tree: `nodes == steps + 1`, i.e. column 0 is the anchor and every
//   other column is a main-chain column. The document's side columns (the priority-queue walk that
//   gives the mechanism its gain) are NOT built yet -- this block exists so that the document's
//   verification item 1 has something to be measured against:
//
//     "Builder: main chain bit-identical to `candidate_selector_path`"   (tree-verification.md:264)
//
//   The main chain is drawn with the SAME selector_draw_rank, the same counter index 0 and the same
//   position, so that identity holds by construction rather than by coincidence; the topology this
//   block adds (parents, depths, ancestor masks, live columns) is what the side columns will be
//   appended to.
//
// LAYOUT: DECLARED AND INDEXED THE SAME WAY
//   Every array here is declared [batch, nodes] and indexed `batch * nodes + column`. Note that the
//   existing `drafts` declares [k, batch] while its kernel indexes `batch * steps + step` -- the
//   declared shape there is nominal only. That mismatch is NOT copied here: a shape that disagrees
//   with the index arithmetic is how a column ends up in the wrong row.
#include "ops/candidate_selector/bf16/candidate_selector_common.cuh"
#include "ops/candidate_selector/bf16/candidate_selector_path_plan.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

struct SelectorTreeArgs {
    DeviceArgs base;  // same inputs the chain op takes
    std::int32_t nodes;
    std::int32_t draw_index;  // counter index of the main-chain draw: 0 normally, 1 in the red arm
    std::int32_t* tokens;     // [batch, nodes]
    std::int32_t* parents;    // [batch, nodes]; -1 for the anchor
    std::int32_t* depths;     // [batch, nodes]
    std::int32_t* masks;      // [batch, nodes]; bit a set iff a == c or a is an ancestor of c
    std::int32_t* live;       // [batch]
};

void candidate_selector_tree_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                    const Tensor& projected_hidden, const Tensor& anchors,
                                    const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook,
                                    const Tensor& base_positions, const SamplingConfig* configs,
                                    Tensor& drafts, Tensor& proposal_q, const Tensor& tokens,
                                    const Tensor& parents, const Tensor& depths, const Tensor& masks,
                                    const Tensor& live, std::int32_t nodes, bool sabotage,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
