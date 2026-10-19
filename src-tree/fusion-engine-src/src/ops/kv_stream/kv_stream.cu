// src/ops/kv_stream/kv_stream.cu — residency map for paged QSA attention (P2, see the header).
//
// The plan this implements is E:\视频\_flashnext\experiments\20261018-kv-streaming\
// 施工图-KV流式-20261018.md (v2.1). Deliberate P2 shape decisions, all documented in the receipt:
//   * one block of 1024 threads resolves the whole call; the CLOCK sweep itself is serial by
//     thread 0 (n_slots is small and this runs once per call) -- correctness first, the P3 variant
//     parallelizes the sweep;
//   * every launch is sized from a capacity (1 block / n_slots blocks), never from a live count;
//   * counters live in device `ctl`; nothing is mirrored to the host except the explicit read in
//     kv_stream_counters() / the end of kv_stream_resolve_host().

#include "ninfer/ops/kv_stream.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr int kResolveThreads = 1024;
constexpr int kCopyThreads    = 256;

void check(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) { return; }
    throw std::runtime_error(std::string("kv_stream: ") + what + ": " + cudaGetErrorString(error));
}

__device__ __forceinline__ void add_u64(std::int32_t* ctl, int index, unsigned long long value) {
    atomicAdd(reinterpret_cast<unsigned long long*>(ctl + index), value);
}

__global__ void kv_stream_decide_kernel(KvStreamMap map, const std::int32_t* __restrict__ ids,
                                        std::int32_t selected_stride,
                                        const std::int32_t* __restrict__ counts,
                                        const std::int32_t* __restrict__ tokens, std::int32_t rows,
                                        std::int32_t page_tokens, std::int32_t block_tokens) {
    __shared__ int s_misses;
    __shared__ int s_lookups;
    const std::int32_t epoch = map.ctl[kKvStreamCtlEpoch] + 1;
    if (threadIdx.x == 0) {
        s_misses  = 0;
        s_lookups = 0;
        // Per-call diagnostics: a hard error is reported by the host right after this call, so the
        // flags start clean instead of latching for the life of the map.
        map.ctl[kKvStreamCtlOverflow]    = 0;
        map.ctl[kKvStreamCtlHostMissing] = 0;
        map.ctl[kKvStreamCtlPlaced]      = 0;
        map.ctl[kKvStreamCtlDemanded]    = 0;
    }
    __syncthreads();

    // Phase 1: every selected 4-token block names a 64-token logical page; claim it exactly once.
    // Selections are in ANY order, so the atomic claim is the only de-duplication -- there is no
    // ascending-neighbour shortcut to take.
    for (int item = threadIdx.x; item < rows * kKvStreamSelectedBlocks; item += blockDim.x) {
        const int row = item / kKvStreamSelectedBlocks;
        const int j   = item % kKvStreamSelectedBlocks;
        if (j >= counts[row]) { continue; }
        const std::int32_t block = ids[static_cast<std::int64_t>(row) * selected_stride + j];
        if (block < 0) { continue; }
        const std::int32_t page = block * block_tokens / page_tokens;
        std::int32_t* cell = map.page_table + static_cast<std::int64_t>(row) * map.logical_pages + page;
        const std::int32_t slot = *cell;
        if (slot >= 0) {
            map.slot_stamp[slot] = epoch;
            map.slot_ref[slot]   = 1;
            atomicAdd(&s_lookups, 1);
        } else if (slot == -1) {
            // A selected page must already be in the host copy: the writer double-writes every
            // appended token. A zero here means the host copy lost a page (J8).
            if (map.host_written[static_cast<std::int64_t>(row) * map.logical_pages + page] == 0) {
                map.ctl[kKvStreamCtlHostMissing] = 1;
                continue;
            }
            if (atomicCAS(cell, -1, -2) == -1) {
                const int k     = atomicAdd(&s_misses, 1);
                map.miss_row[k] = row;
                map.miss_block[k] = page;
                atomicAdd(&s_lookups, 1);
            }
        }
        // slot == -2: another thread of this call claimed it; nothing to do.
    }
    __syncthreads();

    // Phase 1b: the reader's visible set is the selected blocks UNION the causal tail, and the tail
    // page never appears in the selection. It is exempt from the host-written check: the current
    // token's own append happens after this resolve (the writer then fills slot and host copy).
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const std::int32_t page = tokens[row] / page_tokens;
        std::int32_t* cell =
            map.page_table + static_cast<std::int64_t>(row) * map.logical_pages + page;
        if (*cell != -1) { continue; }
        if (atomicCAS(cell, -1, -2) == -1) {
            const int k       = atomicAdd(&s_misses, 1);
            map.miss_row[k]   = row;
            map.miss_block[k] = page;
        }
    }
    __syncthreads();

    const int need = s_misses;
    if (threadIdx.x == 0) {
        map.ctl[kKvStreamCtlDemanded] = need;
        if (need > map.n_slots) {
            map.ctl[kKvStreamCtlOverflow] = 1;  // hard failure: the caller throws, nothing is placed
            map.ctl[kKvStreamCtlPlaced]   = 0;
        } else {
            // Phase 2 (P2: serial by thread 0): CLOCK second chance. A slot used by THIS call is
            // never taken; a referenced slot loses its bit as the hand passes it.
            int hand = map.ctl[kKvStreamCtlHand];
            for (int k = 0; k < need; ++k) {
                int victim  = -1;
                const int limit = 3 * static_cast<int>(map.n_slots);
                for (int scanned = 0; scanned < limit; ++scanned) {
                    const int j = static_cast<int>((hand + scanned) % map.n_slots);
                    if (map.slot_stamp[j] == epoch) { continue; }
                    if (map.slot_block[j] < 0 || map.slot_ref[j] == 0) {
                        victim = j;
                        break;
                    }
                    map.slot_ref[j] = 0;
                }
                if (victim < 0) { victim = static_cast<int>(hand % map.n_slots); }
                hand                  = (victim + 1) % static_cast<int>(map.n_slots);
                map.miss_slot[k]      = victim;
                map.slot_stamp[victim] = epoch;  // taken: the sweep must not pick it twice
            }
            map.ctl[kKvStreamCtlHand]   = hand;
            map.ctl[kKvStreamCtlPlaced] = need;
        }
    }
    __syncthreads();

    if (map.ctl[kKvStreamCtlOverflow] != 0) {
        // Roll the claims back so the next call starts from a consistent map.
        for (int k = threadIdx.x; k < need; k += blockDim.x) {
            map.page_table[static_cast<std::int64_t>(map.miss_row[k]) * map.logical_pages +
                           map.miss_block[k]] = -1;
        }
        if (threadIdx.x == 0) { map.ctl[kKvStreamCtlEpoch] = epoch; }
        return;
    }

    // Phase 3: re-point the table. Unlinking goes through slot_row: a slot may belong to any row,
    // and clearing it through the *current* row would leave a dangling entry behind.
    for (int k = threadIdx.x; k < need; k += blockDim.x) {
        const int row = map.miss_row[k];
        const int page = map.miss_block[k];
        const int slot = map.miss_slot[k];
        const int old  = map.slot_block[slot];
        if (old >= 0) {
            const int old_row = map.slot_row[slot];
            map.page_table[static_cast<std::int64_t>(old_row) * map.logical_pages + old] = -1;
        }
        map.slot_block[slot] = page;
        map.slot_row[slot]   = row;
        map.slot_stamp[slot] = epoch;
        map.slot_ref[slot]   = 1;
        map.page_table[static_cast<std::int64_t>(row) * map.logical_pages + page] = slot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        map.ctl[kKvStreamCtlEpoch] = epoch;
        add_u64(map.ctl, kKvStreamCtlMisses, static_cast<unsigned long long>(need));
        add_u64(map.ctl, kKvStreamCtlLookups, static_cast<unsigned long long>(s_lookups));
        add_u64(map.ctl, kKvStreamCtlCalls, 1ULL);
    }
}

// One block per miss (grid-stride): copy every plane run of the missed page into its slot, 16 B per
// thread. 65,536 B per run / 16 B = 4,096 uint4, 256 threads => 16 iterations.
__global__ void kv_stream_copy_kernel(KvStreamMap map, KvStreamRuns runs, int drop_one_miss) {
    int need = map.ctl[kKvStreamCtlPlaced];
    if (drop_one_miss != 0) { need -= 1; }  // J1 negative control: one page stays stale on purpose
    for (int item = blockIdx.x; item < need; item += gridDim.x) {
        const std::int64_t slot = map.miss_slot[item];
        const std::int64_t page = map.miss_block[item];
        for (int a = 0; a < runs.count; ++a) {
            const auto* source = reinterpret_cast<const uint4*>(
                static_cast<const std::uint8_t*>(runs.host[a]) +
                page * runs.host_stride[a] + runs.host_offset[a]);
            auto* destination = reinterpret_cast<uint4*>(static_cast<std::uint8_t*>(runs.device[a]) +
                                                         slot * runs.len[a]);
            const int words = runs.len[a] / 16;
            for (int i = threadIdx.x; i < words; i += blockDim.x) { destination[i] = source[i]; }
        }
    }
}

__global__ void kv_stream_reset_kernel(KvStreamMap map, int clear_host_written) {
    const std::int64_t stride = (std::int64_t) gridDim.x * blockDim.x;
    for (std::int64_t i = (std::int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < map.rows * map.logical_pages; i += stride) {
        map.page_table[i] = -1;
    }
    for (std::int64_t i = (std::int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < map.n_slots;
         i += stride) {
        map.slot_block[i] = -1;
        map.slot_row[i]   = -1;
        map.slot_stamp[i] = -1;
        map.slot_ref[i]   = 0;
    }
    if (clear_host_written != 0) {
        for (std::int64_t i = (std::int64_t) blockIdx.x * blockDim.x + threadIdx.x;
             i < map.rows * map.logical_pages; i += stride) {
            map.host_written[i] = 0;
        }
    }
    const std::int64_t begin = (std::int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (begin < kKvStreamCtlInts) { map.ctl[begin] = 0; }
}

__global__ void kv_stream_evict_row_kernel(KvStreamMap map, std::int32_t row) {
    const std::int64_t stride = (std::int64_t) gridDim.x * blockDim.x;
    for (std::int64_t i = (std::int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < map.rows * map.logical_pages; i += stride) {
        if (i / map.logical_pages != row) { continue; }
        const std::int32_t page = static_cast<std::int32_t>(i % map.logical_pages);
        const std::int32_t slot = map.page_table[i];
        map.page_table[i]       = -1;
        if (slot >= 0 && map.slot_block[slot] == page && map.slot_row[slot] == row) {
            map.slot_block[slot] = -1;
            map.slot_row[slot]   = -1;
            map.slot_ref[slot]   = 0;
        }
    }
}

std::size_t align_pages(std::size_t bytes) { return (bytes + 255U) & ~std::size_t(255U); }

}  // namespace

KvStreamMapLayout kv_stream_map_layout(std::int64_t rows, std::int64_t logical_pages,
                                       std::int64_t n_slots) {
    KvStreamMapLayout layout;
    std::size_t cursor = 0;
    const auto take    = [&cursor](std::size_t bytes) {
        const std::size_t at = cursor;
        cursor += align_pages(bytes);
        return at;
    };
    layout.slot_block   = take(static_cast<std::size_t>(n_slots) * 4U);
    layout.slot_row     = take(static_cast<std::size_t>(n_slots) * 4U);
    layout.slot_stamp   = take(static_cast<std::size_t>(n_slots) * 4U);
    layout.slot_ref     = take(static_cast<std::size_t>(n_slots) * 4U);
    const std::size_t miss_items =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(kKvStreamMissPerRow);
    layout.miss_block  = take(miss_items * 4U);
    layout.miss_row    = take(miss_items * 4U);
    layout.miss_slot   = take(miss_items * 4U);
    layout.host_written =
        take(static_cast<std::size_t>(rows) * static_cast<std::size_t>(logical_pages) * 4U);
    layout.ctl   = take(static_cast<std::size_t>(kKvStreamCtlInts) * 4U);
    layout.total = cursor;
    return layout;
}

std::size_t kv_stream_map_bytes(std::int64_t rows, std::int64_t logical_pages,
                                std::int64_t n_slots) {
    return kv_stream_map_layout(rows, logical_pages, n_slots).total;
}

void kv_stream_attach(KvStreamMap& map, void* base, std::int64_t rows, std::int64_t logical_pages,
                      std::int64_t n_slots) {
    if (base == nullptr) { throw std::invalid_argument("kv_stream: null map storage"); }
    if (rows <= 0 || logical_pages <= 0 || n_slots <= 0) {
        throw std::invalid_argument("kv_stream: map geometry must be positive");
    }
    const KvStreamMapLayout layout = kv_stream_map_layout(rows, logical_pages, n_slots);
    auto* bytes                    = static_cast<std::uint8_t*>(base);
    map.slot_block   = reinterpret_cast<std::int32_t*>(bytes + layout.slot_block);
    map.slot_row     = reinterpret_cast<std::int32_t*>(bytes + layout.slot_row);
    map.slot_stamp   = reinterpret_cast<std::int32_t*>(bytes + layout.slot_stamp);
    map.slot_ref     = reinterpret_cast<std::int32_t*>(bytes + layout.slot_ref);
    map.miss_block   = reinterpret_cast<std::int32_t*>(bytes + layout.miss_block);
    map.miss_row     = reinterpret_cast<std::int32_t*>(bytes + layout.miss_row);
    map.miss_slot    = reinterpret_cast<std::int32_t*>(bytes + layout.miss_slot);
    map.host_written = reinterpret_cast<std::int32_t*>(bytes + layout.host_written);
    map.ctl          = reinterpret_cast<std::int32_t*>(bytes + layout.ctl);
    map.rows         = rows;
    map.logical_pages = logical_pages;
    map.n_slots       = n_slots;
}

void kv_stream_reset(const KvStreamMap& map, bool clear_host_written, cudaStream_t stream) {
    if (map.page_table == nullptr || map.ctl == nullptr) {
        throw std::invalid_argument("kv_stream_reset: map is not attached");
    }
    kv_stream_reset_kernel<<<64, 256, 0, stream>>>(map, clear_host_written ? 1 : 0);
    check("reset");
}

void kv_stream_resolve_host(const KvStreamMap& map, const KvStreamRuns& runs,
                            const std::int32_t* selected_blocks, std::int32_t selected_stride,
                            const std::int32_t* selected_counts, const std::int32_t* token_indices,
                            std::int32_t rows, std::int32_t page_tokens, std::int32_t block_tokens,
                            bool drop_one_miss, cudaStream_t stream) {
    if (rows <= 0) { return; }
    if (rows > 8) { throw std::invalid_argument("kv_stream: rows must be 1..8"); }
    if (page_tokens <= 0 || block_tokens <= 0 || page_tokens % block_tokens != 0) {
        throw std::invalid_argument("kv_stream: page/block geometry is invalid");
    }
    if (selected_stride < kKvStreamSelectedBlocks) {
        throw std::invalid_argument("kv_stream: selection stride is below kSelectedBlocks");
    }
    if (selected_blocks == nullptr || selected_counts == nullptr || token_indices == nullptr) {
        throw std::invalid_argument("kv_stream: null selection inputs");
    }
    if (runs.count <= 0) { throw std::invalid_argument("kv_stream: no plane runs registered"); }
    if (map.page_table == nullptr || map.ctl == nullptr) {
        throw std::invalid_argument("kv_stream: map is not attached");
    }

    kv_stream_decide_kernel<<<1, kResolveThreads, 0, stream>>>(
        map, selected_blocks, selected_stride, selected_counts, token_indices, rows, page_tokens,
        block_tokens);
    check("decide");
    kv_stream_copy_kernel<<<static_cast<unsigned>(map.n_slots), kCopyThreads, 0, stream>>>(
        map, runs, drop_one_miss ? 1 : 0);
    check("copy");

    // P2: host-side resolve. Legal only outside a capture (asserted by A4 on the model side).
    if (stream != nullptr) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) { check("resolve sync"); }
    } else if (cudaDeviceSynchronize() != cudaSuccess) {
        check("resolve sync");
    }

    const KvStreamCounters counters = kv_stream_counters(map);
    if (counters.overflow) {
        char message[320];
        std::snprintf(message, sizeof(message),
                      "kv_stream: overflow -- this call demanded %lld distinct pages but the map "
                      "has %lld slots (%s)",
                      static_cast<long long>(counters.demanded),
                      static_cast<long long>(map.n_slots),
                      drop_one_miss ? "drop_one_miss control armed" : "raise NINFER_FLASH_NEXT_KV_STREAM_SLOTS");
        throw std::runtime_error(message);
    }
    if (counters.host_missing) {
        throw std::runtime_error(
            "kv_stream: the authoritative host copy is missing a page a selection named (J8)");
    }
    if (drop_one_miss) {
        // The control is only meaningful if it actually skipped a copy.
        if (counters.misses == 0) {
            throw std::runtime_error("kv_stream: drop_one_miss armed but the call had no misses");
        }
    }
}

KvStreamCounters kv_stream_counters(const KvStreamMap& map) {
    KvStreamCounters out;
    if (map.ctl == nullptr) { return out; }
    std::int32_t host[kKvStreamCtlInts] = {};
    if (cudaMemcpy(host, map.ctl, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) {
        check("counters");
        return out;
    }
    const auto* words = reinterpret_cast<const unsigned long long*>(host + 4);
    out.misses       = words[0];
    out.lookups      = words[1];
    out.calls        = words[2];
    out.placed       = static_cast<std::uint64_t>(host[kKvStreamCtlPlaced]);
    out.demanded     = static_cast<std::uint64_t>(host[kKvStreamCtlDemanded]);
    out.overflow     = host[kKvStreamCtlOverflow] != 0;
    out.host_missing = host[kKvStreamCtlHostMissing] != 0;
    return out;
}

void kv_stream_evict_row(const KvStreamMap& map, std::int32_t row, cudaStream_t stream) {
    if (row < 0 || row >= map.rows) { return; }
    kv_stream_evict_row_kernel<<<16, 128, 0, stream>>>(map, row);
    check("evict row");
}

}  // namespace ninfer::ops
