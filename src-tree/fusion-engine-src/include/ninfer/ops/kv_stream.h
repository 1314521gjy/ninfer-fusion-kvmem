#pragma once
//
// ninfer/ops/kv_stream.h — KV streaming residency map for paged QSA attention.
//
// P2 shape of docs: the AUTHORITATIVE K/V of every streamed layer lives in a pinned host copy laid
// out exactly as a fully resident pool would be (identity page table). VRAM holds a pool of
// `n_slots` pages; the page table the reader already uses (`block_tables`) becomes a RESIDENCY MAP:
// `page_table[row][logical_page]` is the slot holding that page, or -1.
//
// Every reader is unchanged: it goes through the page table as before and only ever does
// `phys * stride + ...` arithmetic. What changes is that after the selection and before the
// attention, `kv_stream_resolve_host` makes every page the selection names resident.
//
// Differences from the reference design this borrows its shape from (documented, not copied):
//   * one map carries a ROW per lane (`block_tables` is [logical_pages, concurrency] and every QSA
//     layer shares that one table), so a slot records (block, row) and eviction unlinks through
//     `slot_row` -- forgetting it leaves dangling table entries;
//   * selections are in ANY order, so de-duplication is the atomic claim only (no ascending scan);
//   * the residency granule is the reader's 64-token page, NOT the indexer's 4-token block;
//   * a page pool slot spans every streamed layer (24 plane runs), because the table is shared.
//
// P2 is host-driven on purpose: `kv_stream_resolve_host` synchronizes and raises on overflow, which
// is legal only outside a CUDA graph. The capturable variant is P3.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKvStreamCtlInts = 16;
// [0] epoch  [1] hand  [2] misses placed in the last call  [3] overflow
// [4..5] misses u64  [6..7] lookups u64  [8..9] calls u64  [10] host copy missing a page
inline constexpr int kKvStreamCtlEpoch       = 0;
inline constexpr int kKvStreamCtlHand        = 1;
inline constexpr int kKvStreamCtlPlaced      = 2;
inline constexpr int kKvStreamCtlOverflow    = 3;
inline constexpr int kKvStreamCtlMisses      = 4;
inline constexpr int kKvStreamCtlLookups     = 6;
inline constexpr int kKvStreamCtlCalls       = 8;
inline constexpr int kKvStreamCtlHostMissing = 10;
// Distinct pages the LAST call demanded (M-6 diagnostics: overflow reports "needs N, have S").
inline constexpr int kKvStreamCtlDemanded    = 11;

// Upper bound of one row's selection (qsa_attention_kernels.cu:92) plus the causal tail page.
inline constexpr int kKvStreamSelectedBlocks = 512;
inline constexpr int kKvStreamMissPerRow     = kKvStreamSelectedBlocks + 2;

// All device memory at fixed addresses. `page_table` is BORROWED from the layer's block table.
struct KvStreamMap {
    std::int32_t* page_table = nullptr;  // [rows][logical_pages] slot / -1 / -2 (claim in flight)
    std::int32_t* slot_block = nullptr;  // [n_slots] slot -> logical page, or -1
    std::int32_t* slot_row   = nullptr;  // [n_slots] slot -> lane row, or -1
    std::int32_t* slot_stamp = nullptr;  // [n_slots] epoch that last used the slot
    std::int32_t* slot_ref   = nullptr;  // [n_slots] clock reference bit
    std::int32_t* miss_block = nullptr;  // [rows * kKvStreamMissPerRow]
    std::int32_t* miss_row   = nullptr;  // [rows * kKvStreamMissPerRow]
    std::int32_t* miss_slot  = nullptr;  // [rows * kKvStreamMissPerRow] victim picked by the sweep
    std::int32_t* host_written = nullptr;  // [rows][logical_pages] 1 = the host copy has this page
    std::int32_t* ctl          = nullptr;  // [kKvStreamCtlInts]
    std::int64_t rows          = 0;
    std::int64_t logical_pages = 0;
    std::int64_t n_slots       = 0;
};

// One byte run per plane: 2 per layer (K then V). The host side is a pinned, page-record layout
// [page][K 64 KiB][V 64 KiB]; the device side is a separate [page][plane] pool per layer.
inline constexpr int kKvStreamMaxRuns = 24;
struct KvStreamRuns {
    const void* host[kKvStreamMaxRuns] = {};
    void* device[kKvStreamMaxRuns]      = {};
    std::int64_t host_stride[kKvStreamMaxRuns] = {};  // bytes between consecutive host pages
    std::int32_t host_offset[kKvStreamMaxRuns] = {};  // plane offset inside one host page record
    std::int32_t len[kKvStreamMaxRuns]         = {};  // bytes of one plane of one page
    std::int32_t count                         = 0;
};

// Slices of one contiguous device allocation, so the map can be carved without duplicated math.
struct KvStreamMapLayout {
    std::size_t slot_block = 0, slot_row = 0, slot_stamp = 0, slot_ref = 0;
    std::size_t miss_block = 0, miss_row = 0, miss_slot = 0;
    std::size_t host_written = 0, ctl = 0, total = 0;
};
[[nodiscard]] KvStreamMapLayout kv_stream_map_layout(std::int64_t rows, std::int64_t logical_pages,
                                                     std::int64_t n_slots);
[[nodiscard]] std::size_t kv_stream_map_bytes(std::int64_t rows, std::int64_t logical_pages,
                                              std::int64_t n_slots);
// Wires `map` onto `base` (a device allocation of at least kv_stream_map_bytes). Leaves contents
// untouched; call kv_stream_reset before the first resolve.
void kv_stream_attach(KvStreamMap& map, void* base, std::int64_t rows, std::int64_t logical_pages,
                      std::int64_t n_slots);

// Page table all -1, slots free, stamps/refs zero, counters zero.
// `clear_host_written` = false keeps the "host copy has this page" flags: the prefill phase fills
// them before the residency map exists, and clearing them would make every historical page look
// missing on the first decode (third-knife M-2).
void kv_stream_reset(const KvStreamMap& map, bool clear_host_written, cudaStream_t stream);

// P2 entry point. Makes every page named by the selection (and the causal tail page of every row)
// resident, copies what was missing from the host copy, then synchronizes and throws on
// overflow / missing host page. `drop_one_miss` is a negative-control hook (J1): it deliberately
// skips the last copy so a bit-exact comparison must fail.
void kv_stream_resolve_host(const KvStreamMap& map, const KvStreamRuns& runs,
                            const std::int32_t* selected_blocks, std::int32_t selected_stride,
                            const std::int32_t* selected_counts, const std::int32_t* token_indices,
                            std::int32_t rows, std::int32_t page_tokens, std::int32_t block_tokens,
                            bool drop_one_miss, cudaStream_t stream);

struct KvStreamCounters {
    std::uint64_t misses = 0, lookups = 0, calls = 0;
    // Pages the last call actually placed (== misses of that call, unless it overflowed).
    std::uint64_t placed = 0;
    // Distinct pages the last call demanded (== placed when it fit; > placed when it overflowed).
    std::uint64_t demanded = 0;
    bool overflow = false, host_missing = false;
};
// Synchronous read (debug / end-of-request summary). Never called from inside a capture.
[[nodiscard]] KvStreamCounters kv_stream_counters(const KvStreamMap& map);

// Frees every slot that belongs to `row` (lane teardown). Orphan slots would also be reclaimed by
// the sweep, but this keeps the table<->slot inversion exact for the parity check.
void kv_stream_evict_row(const KvStreamMap& map, std::int32_t row, cudaStream_t stream);

}  // namespace ninfer::ops
