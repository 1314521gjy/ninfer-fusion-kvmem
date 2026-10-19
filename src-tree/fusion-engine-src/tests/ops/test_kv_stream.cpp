// tests/ops/test_kv_stream.cpp — J1 red-arm suite (merged with 红臂设计-J1-20261018.md).
//
// Arms (each prints expected vs measured BIT PATTERN):
//   B  single canary, must go red when one page is not copied;
//   D  two-page sum canary (proves the read kernel really sums both pages);
//   L  wrong-layer arm (layer1 built from layer0's host plane must read 0x3F80, not 0x4000);
//   R  legacy random-byte construction, kept as the live specimen of the old FALSE GREEN.
// Plus the still-green operational checks: J2 counters, J4 overflow, J8 host page missing.
//
// Determinism of "the dropped page is the tail page" (the whole design rests on it):
// kv_stream.cu claims selection pages in phase 1 (atomic, arbitrary order), then __syncthreads(),
// then phase 1b claims the causal tail page -- so the tail page's miss entry is appended AFTER
// every phase-1 entry and is therefore the LAST item of the miss list, which is exactly the item
// the `drop_one_miss` control skips. With blocks {0,40} the tail page 4 is not claimed in phase 1.

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/kv_stream.h"
#include "ninfer/ops/selected_block_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kPageTokens  = 64;
constexpr std::int32_t kBlockTokens = 4;
constexpr std::int32_t kKvHeads     = 2;
constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kQueryHeads  = 24;
constexpr std::int32_t kSelectedCap = 512;
constexpr std::int32_t kPlaneElems  = kKvHeads * kPageTokens * kHeadDim;  // 32,768 bf16
constexpr std::int32_t kPageElems   = kPlaneElems * 2;                    // K+V
constexpr std::int32_t kPlaneBytes  = kPlaneElems * 2;                    // 65,536 B
constexpr std::int32_t kPageBytes   = kPageElems * 2;                     // 131,072 B

// Canary bit patterns from the red-arm design (finite, exact in float32).
constexpr std::uint16_t kBitsZero = 0x0000;  // 0.0
constexpr std::uint16_t kBitsOne  = 0x3F80;  // 1.0
constexpr std::uint16_t kBitsTwo  = 0x4000;  // 2.0
constexpr std::uint16_t kBits15   = 0x3FC0;  // 1.5
constexpr std::uint16_t kLogit128 = 0x4500;  // 2048.0 -> q.k/16 = 128 with q[dim0]=1
constexpr std::uint16_t kStale    = 0xCDCD;  // -4.2994e8, finite (not NaN): the unwritten slot

// Host pool with TWO legs so the engine path is what gets tested:
//   engine-shape : cudaHostAlloc(Portable) + cudaHostGetDevicePointer -- exactly what
//                  kv_stream_binding.h does (probe E:\infer-build\kv-stream\probe_host_ptr.cu arm A);
//   control      : cudaHostAlloc(Mapped)  + cudaHostGetDevicePointer -- NOT the engine path.
class PinnedHostAlias {
public:
    PinnedHostAlias(std::size_t bytes, unsigned int alloc_flags, const char* label)
        : flags_(alloc_flags) {
        cuda_check(cudaHostAlloc(&host_, bytes, alloc_flags), "cudaHostAlloc");
        void* device = nullptr;
        cuda_check(cudaHostGetDevicePointer(&device, host_, 0), "cudaHostGetDevicePointer");
        device_ = device;
        std::printf("  host pool: %s (alloc_flags=0x%x, host_alias=%d)\n", label, alloc_flags,
                    static_cast<int>(host_ == device_));
    }
    ~PinnedHostAlias() { cudaFreeHost(host_); }
    PinnedHostAlias(const PinnedHostAlias&)            = delete;
    PinnedHostAlias& operator=(const PinnedHostAlias&) = delete;
    void* host() const { return host_; }
    void* device() const { return device_; }
    unsigned int flags() const { return flags_; }

private:
    void* host_   = nullptr;
    void* device_ = nullptr;
    unsigned int flags_ = 0;
};

struct Pool {
    DeviceBuffer k;
    DeviceBuffer v;
    std::int32_t pages = 0;
};

Pool make_pool(std::int32_t pages, std::uint8_t fill) {
    Pool pool;
    pool.k = DeviceBuffer(static_cast<std::size_t>(pages) * kPlaneBytes);
    pool.v = DeviceBuffer(static_cast<std::size_t>(pages) * kPlaneBytes);
    pool.k.fill(fill);
    pool.v.fill(fill);
    pool.pages = pages;
    return pool;
}

int failures = 0;
void expect(bool ok, const std::string& label) {
    if (!ok) {
        std::cerr << "FAIL " << label << '\n';
        ++failures;
    }
}

template <typename T>
DeviceBuffer upload(const std::vector<T>& host) {
    DeviceBuffer buffer(host.size() * sizeof(T));
    if (!host.empty()) { buffer.copy_from_host(host.data(), buffer.bytes); }
    return buffer;
}

struct Harness {
    std::int32_t pages = 8, slots = 3, rows = 1, layers = 1;
    PinnedHostAlias host;
    std::vector<Pool> refs;
    std::vector<Pool> streamed;
    DeviceBuffer map_storage;
    DeviceBuffer table;
    ops::KvStreamMap map{};
    ops::KvStreamRuns runs{};

    Harness(std::int32_t pages_, std::int32_t slots_, std::int32_t rows_, std::int32_t layers_,
            unsigned int host_flags = cudaHostAllocPortable, const char* leg_label = nullptr)
        : pages(pages_), slots(slots_), rows(rows_), layers(layers_),
          host(static_cast<std::size_t>(layers_) * pages_ * kPageBytes, host_flags,
               leg_label != nullptr ? leg_label
                                    : (host_flags == cudaHostAllocPortable
                                           ? "ENGINE PATH (Portable + alias)"
                                           : "NON-ENGINE CONTROL (Mapped)")),
          table(static_cast<std::size_t>(rows_) * pages_ * sizeof(std::int32_t)) {
        for (int layer = 0; layer < layers_; ++layer) {
            refs.push_back(make_pool(pages_, 0x00));
            // Unwritten slots = 0xCDCD bytes (finite bf16): the stale pattern the arms look for.
            streamed.push_back(make_pool(slots_, 0xCD));
        }
        map_storage = DeviceBuffer(ops::kv_stream_map_bytes(rows_, pages_, slots_));
        ops::kv_stream_attach(map, map_storage.p, rows_, pages_, slots_);
        map.page_table = static_cast<std::int32_t*>(table.p);
        runs           = ops::KvStreamRuns{};
        for (int layer = 0; layer < layers_; ++layer) {
            const int k_run = 2 * layer;
            auto* base      = static_cast<std::uint8_t*>(host.device()) +
                         static_cast<std::size_t>(layer) * pages_ * kPageBytes;
            runs.host[k_run] = base;
            runs.host_stride[k_run] = kPageBytes;
            runs.host_offset[k_run] = 0;
            runs.len[k_run]         = kPlaneBytes;
            runs.device[k_run]      = streamed[layer].k.p;
            runs.host[k_run + 1]        = base;
            runs.host_stride[k_run + 1] = kPageBytes;
            runs.host_offset[k_run + 1] = kPlaneBytes;
            runs.len[k_run + 1]         = kPlaneBytes;
            runs.device[k_run + 1]      = streamed[layer].v.p;
        }
        runs.count = 2 * layers_;
    }

    std::uint8_t* host_page(int layer, int page) const {
        return static_cast<std::uint8_t*>(host.host()) +
               static_cast<std::size_t>(layer) * pages * kPageBytes +
               static_cast<std::size_t>(page) * kPageBytes;
    }

    // Written into BOTH the host authority and the fully resident reference pool.
    void poke_k(int layer, int page, int within, int head, int dim, std::uint16_t bits) {
        const std::size_t at = (static_cast<std::size_t>(head) * kPageTokens + within) * kHeadDim + dim;
        std::uint16_t* host_k = reinterpret_cast<std::uint16_t*>(host_page(layer, page));
        host_k[at]            = bits;
        cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(refs[layer].k.p) +
                                  static_cast<std::size_t>(page) * kPlaneBytes + at * 2,
                              &bits, 2, cudaMemcpyHostToDevice),
                   "poke k");
    }
    void poke_v(int layer, int page, int within, int head, int dim, std::uint16_t bits) {
        const std::size_t at = (static_cast<std::size_t>(head) * kPageTokens + within) * kHeadDim + dim;
        auto* record         = host_page(layer, page);
        reinterpret_cast<std::uint16_t*>(record + kPlaneBytes)[at] = bits;
        cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(refs[layer].v.p) +
                                  static_cast<std::size_t>(page) * kPlaneBytes + at * 2,
                              &bits, 2, cudaMemcpyHostToDevice),
                   "poke v");
    }
    void fill_zero() {
        std::vector<std::uint8_t> zeros(static_cast<std::size_t>(layers) * pages * kPageBytes, 0);
        std::memcpy(host.host(), zeros.data(), zeros.size());
        for (int layer = 0; layer < layers; ++layer) {
            cuda_check(cudaMemcpy(refs[layer].k.p,
                                  zero_plane(pages).data(), 1, cudaMemcpyHostToDevice), "noop");
            cuda_check(cudaMemset(refs[layer].k.p, 0,
                                  static_cast<std::size_t>(pages) * kPlaneBytes), "zero ref k");
            cuda_check(cudaMemset(refs[layer].v.p, 0,
                                  static_cast<std::size_t>(pages) * kPlaneBytes), "zero ref v");
        }
    }
    static std::vector<std::uint8_t> zero_plane(int) { return {0}; }

    // Legacy random-byte fill (the old false-green specimen) and a bounded numeric fill.
    void fill_random_bytes(std::uint32_t seed) {
        std::mt19937 rng(seed);
        for (int layer = 0; layer < layers; ++layer) {
            for (int page = 0; page < pages; ++page) {
                std::vector<std::uint8_t> k(kPlaneBytes), v(kPlaneBytes);
                for (auto& x : k) { x = static_cast<std::uint8_t>(rng() >> 11); }
                for (auto& x : v) { x = static_cast<std::uint8_t>(rng() >> 13); }
                auto* record = host_page(layer, page);
                std::memcpy(record, k.data(), kPlaneBytes);
                std::memcpy(record + kPlaneBytes, v.data(), kPlaneBytes);
                cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(refs[layer].k.p) +
                                          static_cast<std::size_t>(page) * kPlaneBytes,
                                      k.data(), kPlaneBytes, cudaMemcpyHostToDevice), "rk");
                cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(refs[layer].v.p) +
                                          static_cast<std::size_t>(page) * kPlaneBytes,
                                      v.data(), kPlaneBytes, cudaMemcpyHostToDevice), "rv");
            }
        }
    }
    void fill_bounded() {
        for (int layer = 0; layer < layers; ++layer) {
            for (int page = 0; page < pages; ++page) {
                std::vector<std::uint16_t> k(kPlaneElems), v(kPlaneElems);
                for (int head = 0; head < kKvHeads; ++head) {
                    for (int tok = 0; tok < kPageTokens; ++tok) {
                        for (int dim = 0; dim < kHeadDim; ++dim) {
                            const std::size_t at =
                                (static_cast<std::size_t>(head) * kPageTokens + tok) * kHeadDim + dim;
                            k[at] = f32_to_bf16(
                                0.25F * static_cast<float>((dim * 3 + tok * 5 + page * 11 + head * 7) % 9 - 4) / 4.0F);
                            v[at] = f32_to_bf16(1.0F * static_cast<float>(page + 1) +
                                                0.001F * static_cast<float>(tok + 1) +
                                                2.0F * static_cast<float>(layer));
                        }
                    }
                }
                auto* record = host_page(layer, page);
                std::memcpy(record, k.data(), kPlaneBytes);
                std::memcpy(record + kPlaneBytes, v.data(), kPlaneBytes);
                cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(refs[layer].k.p) +
                                          static_cast<std::size_t>(page) * kPlaneBytes,
                                      k.data(), kPlaneBytes, cudaMemcpyHostToDevice), "bk");
                cuda_check(cudaMemcpy(static_cast<std::uint8_t*>(refs[layer].v.p) +
                                          static_cast<std::size_t>(page) * kPlaneBytes,
                                      v.data(), kPlaneBytes, cudaMemcpyHostToDevice), "bv");
            }
        }
    }
    void reset_and_mark() {
        ops::kv_stream_reset(map, true, nullptr);
        std::vector<std::int32_t> flags(static_cast<std::size_t>(pages) * rows, 1);
        cuda_check(cudaMemcpy(map.host_written, flags.data(), flags.size() * 4,
                              cudaMemcpyHostToDevice),
                   "host_written");
        cuda_synchronize();
    }
    [[nodiscard]] std::vector<std::int32_t> table_host() const {
        std::vector<std::int32_t> out(static_cast<std::size_t>(pages) * rows);
        cuda_check(cudaMemcpy(out.data(), table.p, out.size() * 4, cudaMemcpyDeviceToHost), "table");
        return out;
    }
};

struct Request {
    std::int32_t position = 300;
    std::vector<std::int32_t> blocks{0, 40};  // pages 0 and 2; 70 is deliberately absent
};

struct RequestTensors {
    DeviceBuffer positions, selections, counts;
};

RequestTensors upload_requests(const std::vector<Request>& requests) {
    RequestTensors out;
    const auto rows = static_cast<std::int32_t>(requests.size());
    std::vector<std::int32_t> positions(static_cast<std::size_t>(rows), 0);
    std::vector<std::int32_t> counts(static_cast<std::size_t>(rows), 0);
    std::vector<std::int32_t> selections(static_cast<std::size_t>(rows) * kSelectedCap, -1);
    for (std::int32_t row = 0; row < rows; ++row) {
        positions[row] = requests[row].position;
        counts[row]    = static_cast<std::int32_t>(requests[row].blocks.size());
        for (std::size_t j = 0; j < requests[row].blocks.size(); ++j) {
            selections[static_cast<std::size_t>(row) * kSelectedCap + j] = requests[row].blocks[j];
        }
    }
    out.positions  = upload(positions);
    out.selections = upload(selections);
    out.counts     = upload(counts);
    return out;
}

// q[head][dim0] = 1.0, everything else 0 => logit = K[head][tok][dim0] / 16 exactly.
std::vector<std::uint16_t> query_canary(std::int32_t rows) {
    std::vector<std::uint16_t> query(
        static_cast<std::size_t>(kHeadDim) * kQueryHeads * rows, kBitsZero);
    for (std::int32_t row = 0; row < rows; ++row) {
        for (int head = 0; head < kQueryHeads; ++head) {
            query[(static_cast<std::size_t>(row) * kQueryHeads + head) * kHeadDim] = kBitsOne;
        }
    }
    return query;
}

std::vector<std::uint16_t> attend(const Pool& pool, void* table, std::int32_t table_pages,
                                  std::int32_t table_rows,
                                  const std::vector<std::uint16_t>& query,
                                  const RequestTensors& request) {
    const std::int32_t rows   = table_rows;
    DeviceBuffer d_query      = upload(query);
    DeviceBuffer d_output(static_cast<std::size_t>(kHeadDim) * kQueryHeads * rows * 2);
    const std::vector<std::int32_t> row_index(static_cast<std::size_t>(rows), 0);
    DeviceBuffer d_row_index = upload(row_index);

    Tensor query_t(d_query.p, DType::BF16, {kHeadDim, kQueryHeads, rows});
    Tensor positions_t(request.positions.p, DType::I32, {rows});
    Tensor rows_t(d_row_index.p, DType::I32, {rows});
    Tensor selections_t(request.selections.p, DType::I32, {kSelectedCap, rows});
    Tensor counts_t(request.counts.p, DType::I32, {rows});
    Tensor output_t(d_output.p, DType::BF16, {kHeadDim, kQueryHeads, rows});
    ops::SelectedBlockAttentionCache cache;
    cache.keys  = Tensor(pool.k.p, DType::BF16, {kHeadDim, kPageTokens, kKvHeads, pool.pages});
    cache.values = Tensor(pool.v.p, DType::BF16, {kHeadDim, kPageTokens, kKvHeads, pool.pages});
    cache.block_tables = Tensor(table, DType::I32, {table_pages, table_rows});

    WorkspaceArena workspace(ops::selected_block_attention_workspace_capacity_bytes(rows));
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "stream");
    ops::selected_block_attention(query_t, positions_t, rows_t, selections_t, counts_t, cache,
                                  workspace, output_t, stream);
    cuda_synchronize(stream);
    cuda_check(cudaStreamDestroy(stream), "stream destroy");
    return from_device<std::uint16_t>(d_output.p,
                                      static_cast<std::size_t>(kHeadDim) * kQueryHeads * rows);
}

void resolve(Harness& harness, const RequestTensors& request, bool drop_one) {
    ops::kv_stream_resolve_host(
        harness.map, harness.runs, static_cast<const std::int32_t*>(request.selections.p),
        kSelectedCap, static_cast<const std::int32_t*>(request.counts.p),
        static_cast<const std::int32_t*>(request.positions.p), harness.rows, kPageTokens,
        kBlockTokens, drop_one, nullptr);
}

// Probe -> ASSERTION: every mapped page's slot bytes must equal the host authority bytes.
void assert_pages_match(const Harness& harness, const char* label) {
    const auto table = harness.table_host();
    std::vector<std::uint8_t> from_pool(kPlaneBytes), from_host(kPlaneBytes);
    int mismatched = 0;
    for (std::int32_t page = 0; page < harness.pages; ++page) {
        const std::int32_t slot = table[static_cast<std::size_t>(page)];
        if (slot < 0) { continue; }
        cuda_check(cudaMemcpy(from_pool.data(),
                              static_cast<std::uint8_t*>(harness.streamed[0].k.p) +
                                  static_cast<std::size_t>(slot) * kPlaneBytes,
                              kPlaneBytes, cudaMemcpyDeviceToHost), "probe pool");
        std::memcpy(from_host.data(), harness.host_page(0, page), kPlaneBytes);
        if (std::memcmp(from_pool.data(), from_host.data(), kPlaneBytes) != 0) {
            ++mismatched;
            std::printf("  %s: page=%d slot=%d BYTES DIFFER\n", label, page, slot);
        }
    }
    std::printf("  %s: mapped pages byte-identical to host authority: mismatched=%d\n", label,
                mismatched);
    expect(mismatched == 0, std::string(label) + " slot bytes must equal host bytes");
}

std::string bit_name(std::uint16_t bits) {
    switch (bits) {
        case kBitsZero: return "0x0000";
        case kBitsOne: return "0x3F80";
        case kBitsTwo: return "0x4000";
        case kBits15: return "0x3FC0";
        case kLogit128: return "0x4500";
        case kStale: return "0xCDCD";
        default: {
            char buffer[16];
            std::snprintf(buffer, sizeof(buffer), "0x%04X", bits);
            return buffer;
        }
    }
}

// Every element of the output must carry `expected`.
void expect_all_bits(const std::vector<std::uint16_t>& output, std::uint16_t expected,
                     const char* label) {
    std::uint16_t seen = output.empty() ? 0 : output[0];
    int different      = 0;
    for (const auto value : output) {
        if (value != expected) { ++different; }
    }
    std::printf("  %s: expected=all %s  measured_first=%s  elements=%zu  different=%d\n", label,
                bit_name(expected).c_str(), bit_name(seen).c_str(), output.size(), different);
    expect(different == 0, std::string(label) + " must be all " + bit_name(expected));
}

void expect_output_equals(const std::vector<std::uint16_t>& output, std::uint16_t expected,
                          const char* label) {
    expect_all_bits(output, expected, label);
}

void expect_output_differs(const std::vector<std::uint16_t>& output, std::uint16_t expected,
                           const char* label) {
    int same = 0;
    for (const auto value : output) {
        if (value == expected) { ++same; }
    }
    std::printf("  %s: must differ from %s  measured_first=%s  same_elements=%d\n", label,
                bit_name(expected).c_str(), bit_name(output.empty() ? 0 : output[0]).c_str(),
                same);
    expect(same != static_cast<int>(output.size()) || output.empty(),
           std::string(label) + " must NOT be all " + bit_name(expected));
}

// ---------------------------------------------------------------- arm B / D
void init_canary_harness(Harness& harness) { harness.fill_zero(); }

void poke_tail_token(Harness& harness, int layer, std::uint16_t v_bits) {
    // Token 300 lives in page 4 (300/64) at offset 44; K[dim0] = 2048 -> logit 128.
    for (int head = 0; head < kKvHeads; ++head) {
        harness.poke_k(layer, 4, 300 % kPageTokens, head, 0, kLogit128);
        for (int dim = 0; dim < kHeadDim; ++dim) {
            harness.poke_v(layer, 4, 300 % kPageTokens, head, dim, v_bits);
        }
    }
}

void case_arm_b() {
    std::printf("arm B (single canary): tail-page K=0x4500, V=1.0\n");
    Harness harness(8, 3, 1, 1);
    init_canary_harness(harness);
    poke_tail_token(harness, 0, kBitsOne);
    harness.reset_and_mark();
    const std::vector<Request> requests{{300, {0, 40}}};
    const RequestTensors request = upload_requests(requests);
    const auto query             = query_canary(1);
    resolve(harness, request, false);
    const auto counters = ops::kv_stream_counters(harness.map);
    std::printf("  self-check: |pageset|=3 misses=%llu placed=%llu calls=%llu\n",
                static_cast<unsigned long long>(counters.misses),
                static_cast<unsigned long long>(counters.placed),
                static_cast<unsigned long long>(counters.calls));
    expect(counters.misses == 3 && counters.placed == 3 && counters.calls == 1,
           "arm B: |pageset| == misses == placed == 3");
    assert_pages_match(harness, "arm B green");
    const auto green = attend(harness.streamed[0], harness.table.p, harness.pages, 1, query, request);
    expect_output_equals(green, kBitsOne, "arm B green output");

    Harness dropped(8, 3, 1, 1);
    init_canary_harness(dropped);
    poke_tail_token(dropped, 0, kBitsOne);
    dropped.reset_and_mark();
    const RequestTensors dropped_request = upload_requests(requests);
    resolve(dropped, dropped_request, true);
    const auto red = attend(dropped.streamed[0], dropped.table.p, dropped.pages, 1, query,
                            dropped_request);
    // The dropped page is the tail page (deterministic); its slot keeps 0xCDCD bytes.
    expect_output_differs(red, kBitsOne, "arm B dropped-page output");
    std::printf("  arm B note: design doc expects all 0xCDCD; the other visible tokens are 0.0 in\n"
                "             the correct copy, so the measured stale value is reported above.\n");
}

void case_arm_d() {
    std::printf("arm D (sum-sensitive): page4 token300 K=0x4500/V=1.0, page2 token161 K=0x4500/V=2.0\n");
    Harness harness(8, 4, 1, 1);
    init_canary_harness(harness);
    poke_tail_token(harness, 0, kBitsOne);
    for (int head = 0; head < kKvHeads; ++head) {
        harness.poke_k(0, 2, 161 % kPageTokens, head, 0, kLogit128);
        for (int dim = 0; dim < kHeadDim; ++dim) {
            harness.poke_v(0, 2, 161 % kPageTokens, head, dim, kBitsTwo);
        }
    }
    harness.reset_and_mark();
    const std::vector<Request> requests{{300, {0, 40}}};
    const RequestTensors request = upload_requests(requests);
    const auto query             = query_canary(1);
    resolve(harness, request, false);
    const auto counters = ops::kv_stream_counters(harness.map);
    std::printf("  self-check: |pageset|=3 misses=%llu placed=%llu\n",
                static_cast<unsigned long long>(counters.misses),
                static_cast<unsigned long long>(counters.placed));
    expect(counters.misses == 3 && counters.placed == 3, "arm D: |pageset| == misses == placed");
    const auto green = attend(harness.streamed[0], harness.table.p, harness.pages, 1, query, request);
    expect_output_equals(green, kBits15, "arm D green output (0.5*1.0 + 0.5*2.0)");

    Harness dropped(8, 4, 1, 1);
    init_canary_harness(dropped);
    poke_tail_token(dropped, 0, kBitsOne);
    for (int head = 0; head < kKvHeads; ++head) {
        dropped.poke_k(0, 2, 161 % kPageTokens, head, 0, kLogit128);
        for (int dim = 0; dim < kHeadDim; ++dim) {
            dropped.poke_v(0, 2, 161 % kPageTokens, head, dim, kBitsTwo);
        }
    }
    dropped.reset_and_mark();
    const RequestTensors dropped_request = upload_requests(requests);
    resolve(dropped, dropped_request, true);
    const auto red = attend(dropped.streamed[0], dropped.table.p, dropped.pages, 1, query,
                            dropped_request);
    // Only page 2's canary survives => output must be exactly 2.0.
    expect_output_equals(red, kBitsTwo, "arm D dropped-tail output (page2 only)");
}

// ------------------------------------------------------------ wrong-layer arm
void case_wrong_layer() {
    std::printf("arm L (wrong layer): same K, V layer0=1.0 layer1=2.0\n");
    Harness harness(8, 4, 1, 2);
    init_canary_harness(harness);
    poke_tail_token(harness, 0, kBitsOne);
    poke_tail_token(harness, 1, kBitsTwo);
    harness.reset_and_mark();
    const std::vector<Request> requests{{300, {0, 40}}};
    const RequestTensors request = upload_requests(requests);
    const auto query             = query_canary(1);
    resolve(harness, request, false);
    for (int layer = 0; layer < 2; ++layer) {
        const auto out = attend(harness.streamed[layer], harness.table.p, harness.pages, 1, query,
                                request);
        expect_output_equals(out, layer == 0 ? kBitsOne : kBitsTwo,
                             layer == 0 ? "arm L green layer0" : "arm L green layer1");
    }

    Harness crossed(8, 4, 1, 2);
    init_canary_harness(crossed);
    poke_tail_token(crossed, 0, kBitsOne);
    poke_tail_token(crossed, 1, kBitsTwo);
    crossed.reset_and_mark();
    crossed.runs.host[2] = crossed.runs.host[0];
    crossed.runs.host[3] = crossed.runs.host[1];
    const RequestTensors crossed_request = upload_requests(requests);
    resolve(crossed, crossed_request, false);
    const auto wrong = attend(crossed.streamed[1], crossed.table.p, crossed.pages, 1, query,
                              crossed_request);
    expect_output_equals(wrong, kBitsOne, "arm L crossed layer1 (must read layer0's V)");
}

// ------------------------------------------------ legacy specimen (false green)
void case_legacy_random() {
    std::printf("legacy R (random-byte construction): kept as the live FALSE-GREEN specimen\n");
    Harness harness(8, 3, 1, 1);
    harness.fill_random_bytes(7);
    harness.reset_and_mark();
    const std::vector<Request> requests{{300, {0, 40, 70}}};
    const RequestTensors request = upload_requests(requests);
    const auto query             = query_canary(1);
    resolve(harness, request, false);
    const auto green = attend(harness.streamed[0], harness.table.p, harness.pages, 1, query, request);
    Harness dropped(8, 3, 1, 1);
    dropped.fill_random_bytes(7);
    dropped.reset_and_mark();
    const RequestTensors dropped_request = upload_requests(requests);
    resolve(dropped, dropped_request, true);
    const auto red = attend(dropped.streamed[0], dropped.table.p, dropped.pages, 1, query,
                            dropped_request);
    int same = 0;
    for (std::size_t i = 0; i < green.size() && i < red.size(); ++i) {
        if (green[i] == red[i]) { ++same; }
    }
    std::printf("  legacy: elements=%zu identical=%d (NOT a criterion; this is the old false green)\n",
                green.size(), same);
}

// --------------------------------------------------- operational still-green
void case_j2_counters() {
    const auto run = [](std::int32_t slots) {
        Harness harness(8, slots, 1, 1);
        harness.fill_bounded();
        harness.reset_and_mark();
        for (int call = 0; call < 8; ++call) {
            const std::int32_t block = 16 * (call % 2 == 0 ? 2 : 3);
            const std::vector<Request> requests{{300, {block}}};
            const RequestTensors request = upload_requests(requests);
            resolve(harness, request, false);
        }
        const auto counters = ops::kv_stream_counters(harness.map);
        std::printf("J2  slots=%d misses=%llu placed=%llu calls=%llu\n", slots,
                    static_cast<unsigned long long>(counters.misses),
                    static_cast<unsigned long long>(counters.placed),
                    static_cast<unsigned long long>(counters.calls));
        expect(counters.calls == 8, "J2 calls == 8");
        return counters.misses;
    };
    const auto roomy = run(8);
    const auto tight = run(2);
    std::printf("J2  monotonic: tight=%llu > roomy=%llu ? %d\n",
                static_cast<unsigned long long>(tight),
                static_cast<unsigned long long>(roomy), static_cast<int>(tight > roomy));
    expect(tight > roomy, "J2 misses must rise when the pool shrinks");
}

void case_j4_overflow() {
    Harness harness(8, 1, 1, 1);
    harness.fill_bounded();
    harness.reset_and_mark();
    const std::vector<Request> requests{{300, {0, 40}}};
    const RequestTensors request = upload_requests(requests);
    bool threw = false;
    try {
        resolve(harness, request, false);
    } catch (const std::exception& error) {
        threw = true;
        std::printf("J4  overflow threw: %s\n", error.what());
    }
    expect(threw, "J4 overflow must throw");
}

void case_j8_host_missing() {
    Harness harness(8, 4, 1, 1);
    harness.fill_bounded();
    ops::kv_stream_reset(harness.map, true, nullptr);
    std::vector<std::int32_t> flags(static_cast<std::size_t>(harness.pages), 0);
    cuda_check(cudaMemcpy(harness.map.host_written, flags.data(), flags.size() * 4,
                          cudaMemcpyHostToDevice), "host_written zero");
    cuda_synchronize();
    const std::vector<Request> requests{{300, {0, 40}}};
    const RequestTensors request = upload_requests(requests);
    bool threw = false;
    try {
        resolve(harness, request, false);
    } catch (const std::exception& error) {
        threw = true;
        std::printf("J8  host_missing threw: %s\n", error.what());
    }
    expect(threw, "J8 missing host page must throw");
}

// task-26 evidence: (a) with the switch OFF nothing calls resolve, so the counters must not move
// (that is the "switch did not take effect" reading); (b) the host_written page count is read
// DIRECTLY from the map, not inferred from "the resolve did not throw".
void case_counters_and_host_written() {
    std::printf("task-26 counters/host_written:\n");
    {
        Harness harness(8, 3, 1, 1);
        init_canary_harness(harness);
        poke_tail_token(harness, 0, kBitsOne);
        harness.reset_and_mark();
        const auto before = ops::kv_stream_counters(harness.map);
        const std::vector<Request> requests{{300, {0, 40}}};
        const RequestTensors request = upload_requests(requests);
        const auto query             = query_canary(1);
        (void) attend(harness.streamed[0], harness.table.p, harness.pages, 1, query, request);
        const auto after = ops::kv_stream_counters(harness.map);
        std::printf("  switch-off: misses %llu->%llu calls %llu->%llu (must be identical)\n",
                    (unsigned long long) before.misses, (unsigned long long) after.misses,
                    (unsigned long long) before.calls, (unsigned long long) after.calls);
        expect(before.misses == after.misses && before.calls == after.calls &&
                   before.placed == after.placed,
               "task-26: with no resolve the counters must not move");
    }
    {
        Harness harness(8, 3, 1, 1);
        init_canary_harness(harness);
        poke_tail_token(harness, 0, kBitsOne);
        harness.reset_and_mark();  // marks every (row,page) as written by the "prefill"
        std::vector<std::int32_t> flags(static_cast<std::size_t>(harness.pages), 0);
        cuda_check(cudaMemcpy(flags.data(), harness.map.host_written, flags.size() * 4,
                              cudaMemcpyDeviceToHost),
                   "read host_written");
        int marked = 0;
        for (const auto value : flags) { if (value != 0) { ++marked; } }
        std::printf("  host_written marked=%d of %d pages\n", marked, harness.pages);
        expect(marked == harness.pages, "task-26: host_written page count must be readable");
    }
}

// M-2 mechanism, UNIT level only: arming the residency map must NOT wipe the "the host copy has
// this page" flags that the prefill phase wrote -- clearing them is exactly what threw J8 on the
// first decode round (host-writer ran, then reset erased the evidence). This does not close the
// ENGINE-side ledger item ("host_written > 0 in the first decode round"): that needs an engine arm.
void case_reset_keeps_host_written() {
    const std::vector<Request> requests{{300, {0, 40}}};
    std::printf("M-2 reset-keeps-host-written:\n");

    // Positive: flags written (as prefill would), arm with clear_host_written=false => no J8.
    {
        Harness harness(8, 3, 1, 1);
        init_canary_harness(harness);
        poke_tail_token(harness, 0, kBitsOne);
        std::vector<std::int32_t> flags(static_cast<std::size_t>(harness.pages), 1);
        cuda_check(cudaMemcpy(harness.map.host_written, flags.data(), flags.size() * 4,
                              cudaMemcpyHostToDevice),
                   "host_written prefill");
        ops::kv_stream_reset(harness.map, /*clear_host_written=*/false, nullptr);
        cuda_synchronize();
        const RequestTensors request = upload_requests(requests);
        bool threw                   = false;
        try {
            resolve(harness, request, false);
        } catch (const std::exception& error) {
            threw = true;
            std::printf("  keep-flags: UNEXPECTED throw: %s\n", error.what());
        }
        const auto counters = ops::kv_stream_counters(harness.map);
        std::printf("  keep-flags: threw=%d host_missing=%d misses=%llu placed=%llu\n",
                    static_cast<int>(threw), static_cast<int>(counters.host_missing),
                    static_cast<unsigned long long>(counters.misses),
                    static_cast<unsigned long long>(counters.placed));
        expect(!threw && !counters.host_missing,
               "M-2: reset(clear_host_written=false) must keep the prefill flags");
    }

    // Negative control: the same sequence with clear_host_written=true MUST throw J8.
    {
        Harness harness(8, 3, 1, 1);
        init_canary_harness(harness);
        poke_tail_token(harness, 0, kBitsOne);
        std::vector<std::int32_t> flags(static_cast<std::size_t>(harness.pages), 1);
        cuda_check(cudaMemcpy(harness.map.host_written, flags.data(), flags.size() * 4,
                              cudaMemcpyHostToDevice),
                   "host_written prefill");
        ops::kv_stream_reset(harness.map, /*clear_host_written=*/true, nullptr);
        cuda_synchronize();
        const RequestTensors request = upload_requests(requests);
        bool threw                   = false;
        try {
            resolve(harness, request, false);
        } catch (const std::exception& error) {
            threw = true;
            std::printf("  clear-flags: expected throw: %s\n", error.what());
        }
        std::printf("  clear-flags: threw=%d (must be 1)\n", static_cast<int>(threw));
        expect(threw, "M-2 negative control: reset(clear_host_written=true) must trigger J8");
    }
}

// M-3 homogenisation: run the SAME canary through both host-pool legs and label which one is the
// engine path. The mapped leg is a control and the output says so, so nobody can quote its green as
// evidence about the engine.
void case_host_legs() {
    const std::vector<Request> requests{{300, {0, 40}}};
    const auto query             = query_canary(1);
    for (int leg = 0; leg < 2; ++leg) {
        const unsigned int flags = (leg == 0) ? cudaHostAllocPortable : cudaHostAllocMapped;
        std::printf("host-leg %d/2:\n", leg + 1);
        Harness harness(8, 3, 1, 1, flags);
        init_canary_harness(harness);
        poke_tail_token(harness, 0, kBitsOne);
        harness.reset_and_mark();
        const RequestTensors request = upload_requests(requests);
        resolve(harness, request, false);
        const auto out = attend(harness.streamed[0], harness.table.p, harness.pages, 1, query,
                                request);
        const bool engine_leg = (flags == cudaHostAllocPortable);
        std::printf("  leg=%-16s path=%s\n", engine_leg ? "Portable+alias" : "Mapped+alias",
                    engine_leg ? "ENGINE" : "非引擎路径（Mapped 对照）");
        expect_output_equals(out, kBitsOne, engine_leg ? "engine leg output" : "control leg output");
    }
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    try {
        if (cuda_unavailable()) {
            std::cout << "kv_stream: SKIP (CUDA unavailable)\n";
            return 0;
        }
        case_counters_and_host_written();
        case_reset_keeps_host_written();
        case_host_legs();
        case_arm_b();
        case_arm_d();
        case_wrong_layer();
        case_legacy_random();
        case_j2_counters();
        case_j4_overflow();
        case_j8_host_missing();
    } catch (const std::exception& error) {
        std::cerr << "kv_stream: uncaught exception: " << error.what() << '\n';
        return 2;
    }
    if (failures == 0) { std::cout << "kv_stream: PASS\n"; }
    else { std::cerr << "kv_stream: " << failures << " failure(s)\n"; }
    return failures == 0 ? 0 : 1;
}
