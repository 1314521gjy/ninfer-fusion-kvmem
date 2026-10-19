#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace ninfer::ops {

// Read-only paged K/V storage: [256 features, 64 tokens, 2 heads, physical pages].
// Both tensors have the same dtype, BF16 or unscaled FP8 E4M3FN. Page tables are
// contiguous I32 [logical pages, table rows]; their entries name physical pages.
struct SelectedBlockAttentionCache {
    Tensor keys;
    Tensor values;
    Tensor block_tables;
};

/**
 * Softmax attention over selected complete four-token blocks and the causal tail.
 *
 * Fixed profile: head dimension 256, 24 query heads, 2 KV heads, scale 1/16,
 * and batch 1..8. Query/output are contiguous BF16 [256,24,batch]. Positions and
 * table_rows are device I32 [batch], selections I32 [512,batch], counts I32 [batch].
 * Query head h reads KV head floor(h/12).
 *
 * For row b at inclusive position p, complete=floor((p+1)/4). The visible set is
 * {4*selections[j,b]+r : j<count[b], r=0..3} union [4*complete,p]. Counts are in
 * [0,512]; selections are unique complete blocks in [0,complete), in any order.
 * The caller guarantees nonnegative positions, valid table rows/page mappings,
 * and initialized K/V for every visible token. No unselected cache value is read.
 *
 * output[d,h,b] = sum_t softmax_t(dot(query[:,h,b], key[:,h/12,t])/16)
 *                              * value[d,h/12,t].
 * An empty visible set produces exact zero. FP8 inputs are decoded using their
 * represented E4M3FN values; BF16 output is the sole semantic rounding boundary.
 * The oracle evaluates this complete formula independently in FP64. Production
 * reduction order and intermediate precision are implementation choices.
 *
 * All inputs, output, and live workspace allocations are non-overlapping, remain
 * valid until stream completion, and use contiguous storage aligned to 16 bytes.
 * Only output is modified. Workspace is transient, caller-owned, and graph-safe;
 * this operation performs no device allocation or host/device synchronization.
 */
void selected_block_attention(const Tensor& query, const Tensor& positions,
                              const Tensor& table_rows, const Tensor& selections,
                              const Tensor& counts, const SelectedBlockAttentionCache& cache,
                              WorkspaceArena& workspace, Tensor& output, cudaStream_t stream);

[[nodiscard]] std::size_t selected_block_attention_workspace_capacity_bytes(int batch);

// Same formula and storage contract, for 1..262144 query tokens sharing one page-table
// row. Positions/counts are I32 [tokens], selections I32 [512,tokens], and query/output
// BF16 [256,24,tokens]. No transient device workspace is required.
void selected_block_attention(const Tensor& query, const Tensor& positions,
                                      int table_row, const Tensor& selections,
                                      const Tensor& counts, const SelectedBlockAttentionCache& cache,
                                      Tensor& output, cudaStream_t stream);

} // namespace ninfer::ops
