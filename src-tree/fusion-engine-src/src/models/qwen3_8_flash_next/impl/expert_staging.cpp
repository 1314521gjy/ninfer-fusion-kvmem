#include "models/qwen3_8_flash_next/impl/expert_staging.h"
#include "models/qwen3_8_flash_next/impl/cpu_expert_nvfp4.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/weight_view.h"
#include "ninfer/ops/residual_add.h" // block-6 W2: the apply side's residual add
#include "models/qwen3_5/load/vision_cpu.h"
#include "models/qwen3_8_flash_next/impl/expert_stage_kernels.h"
// Instrumentation only (see the ledger record at the end of the staging call): reuses an existing
// stage id rather than adding an enum value, so no header changes and no rebuild exposure.
#include "models/qwen3_8_flash_next/impl/stage_ledger.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

// ---- S5: the four metric pairs, per layer, reported once per token ----------------------------
//
// WHY THIS EXISTS. The community's per-step ledger (ROCm, llama.cpp PR #27861) found host time in
// stream synchronisation at 28-50% of the GPU span, and 65 us PER SMALL H2D OPERATION -- i.e. the
// OPERATION COUNT is decoupled from the BYTES. This tree has never measured either. Every claim we
// have made about "how much of the 2.1 ms/layer staging is DMA and how much is host" has been an
// INFERENCE from total bytes divided by line rate, which silently assumes one clean burst.
//
// What is measured here (no CUPTI needed):
//   syncs + host milliseconds blocked inside cudaStreamSynchronize
//   copy OPERATIONS and bytes, split H2D / D2H            <-- the pair we never had
//   host_total   = the whole host-side span of one staging call
//   host_decision = host_total - sync_host_ms             (LRU/victim choice, drop logic, slot deltas)
//
// What is NOT measured and must not be claimed: KERNEL LAUNCH count and host launch time. That needs
// CUPTI/nsys. Do not read a number for it out of this block.
// ---------------------------------------------------------------------------------------------
// P1-1: the DISK layer of "how many bytes did this window really move".
//
// `moved_mib` / `h2d_mib` count bytes STAGED host->device. They say nothing about whether the host
// side came out of the OS page cache or off the disk, and this tree has never measured that layer.
// The audit that asked for it is explicit: "moved_mib must be split into layers (H2D / RAM hit /
// disk read)". GetProcessIoCounters(ReadTransferCount) is the one ready-made counter that can tell
// the two apart: it counts bytes this process actually pulled from storage, so
//     disk = min(window disk reads, staged bytes)   and   ram = staged - disk.
//
// Declared by hand instead of including <windows.h>: this TU also pulls CUDA-facing headers and this
// tree has been bitten by host/device header collisions more than once. IO_COUNTERS is six ULONGLONGs
// in a fixed order (ReadOperation, WriteOperation, OtherOperation, ReadTransfer, WriteTransfer,
// OtherTransfer). If the call fails we return 0 and the printed residual says so out loud, instead of
// silently reporting "no disk reads".
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess();
extern "C" __declspec(dllimport) int __stdcall GetProcessIoCounters(void* process, void* counters);

std::uint64_t process_read_transfer_bytes() {
    struct IoCounters {
        unsigned long long read_operation_count;
        unsigned long long write_operation_count;
        unsigned long long other_operation_count;
        unsigned long long read_transfer_count;
        unsigned long long write_transfer_count;
        unsigned long long other_transfer_count;
    } counters{};
    if (GetProcessIoCounters(GetCurrentProcess(), &counters) == 0) { return 0; }
    return static_cast<std::uint64_t>(counters.read_transfer_count);
}

// R6 of block 1: the first run of the line below printed disk_mib=0.000 AND io_read_mib=0.000 in all
// 404 windows, and those two zeros are indistinguishable -- "this window really read no disk" and
// "the counter never worked" look exactly the same. So the line now also reports whether the query
// itself succeeded, and the counter's value at the first window, which is what a positive control
// needs to compare against (a run that is KNOWN to read storage, e.g. PIN=1 loading 63.3 GiB, must
// move that baseline). Two zeros with an unproven instrument are not a reading.
bool process_io_query_ok() {
    struct IoCounters {
        unsigned long long read_operation_count;
        unsigned long long write_operation_count;
        unsigned long long other_operation_count;
        unsigned long long read_transfer_count;
        unsigned long long write_transfer_count;
        unsigned long long other_transfer_count;
    } counters{};
    return GetProcessIoCounters(GetCurrentProcess(), &counters) != 0;
}

struct StageProfile {
    unsigned long long layers = 0;
    unsigned long long syncs  = 0;
    double host_ms = 0.0;
    double sync_ms = 0.0;
    unsigned long long h2d_ops = 0;
    unsigned long long d2h_ops = 0;
    unsigned long long h2d_bytes = 0;
    unsigned long long d2h_bytes = 0;
    // The one number that decides whether a call fits its destination at all: how many DIFFERENT
    // experts this call asked for. Reported as a per-window maximum, because the median is what the
    // wave width was chosen from and the maximum is what the pool has to survive.
    std::int32_t max_distinct = 0;

    // P3-1 self-证. `h2d_ops` already counts copy operations; these two split that count into "how
    // many batch calls" and "how many individual operations they replaced", so a knob that is on but
    // doing nothing is distinguishable from one that is working.
    unsigned long long dma_batches      = 0;
    unsigned long long dma_ops_replaced = 0;

    // P1-13: the three named spans inside host_decision_ms (host_total - sync_host). The point of the
    // split is the three ABSOLUTE numbers -- where the per-token host time actually goes -- and not
    // the closure: `other_ms` is a RESIDUAL computed as total - (drop+decide+stage), so it closes by
    // construction and closure is NOT evidence. What can turn red is the sanity bound
    // (drop+decide+stage must not exceed the total) and the sense of the comparison against the
    // pinned tier's 0.819 ms/token.
    double drop_ms   = 0.0; // the R1 drop gate: alpha readback + threshold scan
    double decide_ms = 0.0; // per-path hit/miss + victim choice (the three-pass O(slots) scan)
    double stage_ms  = 0.0; // issuing the copies for everything that is missing

    void note_h2d(std::size_t bytes) noexcept { ++h2d_ops; h2d_bytes += bytes; }
    void note_d2h(std::size_t bytes) noexcept { ++d2h_ops; d2h_bytes += bytes; }
    void note_distinct(std::int32_t count) noexcept {
        if (count > max_distinct) { max_distinct = count; }
    }

    void report_and_reset() {
        // P1-1: sample the process IO counter BEFORE anything is reset, so the split below belongs to
        // exactly the same window as the counters it splits. The first call primes the mark (delta 0).
        const std::uint64_t io_now  = process_read_transfer_bytes();
        static std::uint64_t io_mark = io_now;
        static const std::uint64_t io_first_window = io_now; // the value at the first window: the
                                                            // baseline a positive control compares to
        const std::uint64_t io_read = io_now >= io_mark ? io_now - io_mark : 0;
        io_mark                     = io_now;
        const std::uint64_t disk_bytes =
            io_read < h2d_bytes ? io_read : static_cast<std::uint64_t>(h2d_bytes);
        const std::uint64_t ram_bytes = static_cast<std::uint64_t>(h2d_bytes) - disk_bytes;
        std::fprintf(stderr,
                     "flash_next_stage_profile layers=%llu syncs=%llu sync_host_ms=%.3f "
                     "host_total_ms=%.3f host_decision_ms=%.3f "
                     "h2d_ops=%llu h2d_mib=%.3f d2h_ops=%llu d2h_kib=%.1f h2d_bytes_per_op=%llu "
                     "dma_batches=%llu dma_ops_replaced=%llu max_distinct=%d\n",
                     layers, syncs, sync_ms, host_ms, host_ms - sync_ms,
                     h2d_ops, static_cast<double>(h2d_bytes) / (1024.0 * 1024.0), d2h_ops,
                     static_cast<double>(d2h_bytes) / 1024.0,
                     h2d_ops ? (h2d_bytes / h2d_ops) : 0ULL,
                     dma_batches, dma_ops_replaced,
                     static_cast<int>(max_distinct));
        // P1-1 self-证 line. The conservation it can actually turn red on is NOT ram+disk == h2d
        // (that holds by construction); it is `residual`, i.e. how far the raw process disk reads are
        // from the disk bytes we attributed to staging. A residual that is not small next to the
        // staged bytes means the attribution is wrong (other reads in the window, or a counter that
        // never moved), and the column must not be used until that is explained.
        std::fprintf(stderr,
                     "flash_next_moved_split h2d_mib=%.3f ram_mib=%.3f disk_mib=%.3f "
                     "io_read_mib=%.3f residual_mib=%.3f io_query_ok=%d io_first_win_mib=%.3f\n",
                     static_cast<double>(h2d_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(ram_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(disk_bytes) / (1024.0 * 1024.0),
                     static_cast<double>(io_read) / (1024.0 * 1024.0),
                     (static_cast<double>(io_read) - static_cast<double>(disk_bytes)) /
                         (1024.0 * 1024.0),
                     process_io_query_ok() ? 1 : 0,
                     static_cast<double>(io_first_window) / (1024.0 * 1024.0));
        // P1-13 self-证 line. `other_ms` is a residual by construction (see the field comment): the
        // value of this line is the three absolute spans, and the only thing that can turn red here
        // is the sanity bound named+other == total with named <= total.
        {
            const double total = host_ms - sync_ms;
            const double named = drop_ms + decide_ms + stage_ms;
            std::fprintf(stderr,
                         "flash_next_host_decision_split total_ms=%.3f drop_ms=%.3f decide_ms=%.3f "
                         "stage_ms=%.3f other_ms=%.3f\n",
                         total, drop_ms, decide_ms, stage_ms, total - named);
        }
        *this = StageProfile{};
    }
};

StageProfile& stage_profile() {
    static StageProfile profile;
    return profile;
}

// Monotonic milliseconds. steady_clock, not system_clock: this is a duration, never a timestamp.
double host_now_ms() noexcept {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

// ---- Pinned control-path staging --------------------------------------------------------------
//
// NINFER_FLASH_NEXT_EXPERTS_PIN=1 makes the BULK expert planes pinned, and that part works: the byte
// ledger reports line rate (22.55 GB/s measured against a 23.97 GB/s one-copy ceiling). But every
// CONTROL copy in this file used to read or write an automatic (stack) array, and a
// cudaMemcpyAsync whose host side is PAGEABLE is not asynchronous at all:
//   * H2D from pageable memory: the driver performs a stream synchronisation BEFORE the copy is
//     initiated (it packs the payload into its own internal pinned staging buffer first);
//   * D2H to pageable memory: the call returns only once the copy has completed.
// On the miss path the two divisor copies sat INSIDE the per-miss loop, between the pinned plane
// copies, so each one drained the DMA queue the pinned copies had just filled -- exactly the
// signature of "the transfer is at line rate but the layer still stalls". Staging the control path
// in pinned memory removes the mechanism without changing one byte that reaches the device, so the
// counters (calls/hits/misses/moved_mib/dropped) MUST come out bit-identical.
template <typename T>
struct PinnedArray {
    explicit PinnedArray(std::size_t count) : buffer_(count * sizeof(T)) {}

    [[nodiscard]] T* data() const noexcept { return static_cast<T*>(buffer_.data()); }
    T& operator[](std::size_t index) const noexcept { return data()[index]; }
    [[nodiscard]] std::size_t size() const noexcept { return buffer_.size() / sizeof(T); }

    PinnedArray(const PinnedArray&)            = delete;
    PinnedArray& operator=(const PinnedArray&) = delete;

private:
    PinnedHostBuffer buffer_;
};

struct StagingControlBuffers {
    StagingControlBuffers() {
        // Block-8 item 3a: zero the stream-wait completion words before anything can wait on them.
        std::memset(serve_done_flag.data(), 0,
                    cpu_expert::kCpuServeRingSlots * sizeof(std::uint32_t));
    }

    // Indexed by ROUTED PATH, not by slot: one entry per (token, top-k) pair, so the wide arm's
    // 32-token wave needs 320 entries where the decode path needs at most 80.
    PinnedArray<std::int32_t> ids{kMaxStagedPaths};
    PinnedArray<float> alpha{kMaxStagedPaths};
    PinnedArray<std::int32_t> slots{kMaxStagedPaths};

    // Block-6 W1/W2: the forward face's pinned host buffers, sized for the decode width the face
    // arms at (cpu_expert::kCpuServeForwardCols columns). x mirrors this call's MoE input
    // [kHidden, cols] BF16; partial_f32 is the accumulator the serve team's ordered reduce fills;
    // partial_bf16 is the converted copy the apply side hands to the device.
    //
    // Block-8 item 7 (ring): each of the three buffers is a kCpuServeRingSlots-slot arena, so an
    // in-flight round's slabs stay untouched until its answer and N rounds can be in flight. Index
    // within a slab: [col * kHidden + h] (the input/output tensor layout); slab stride:
    // kCpuServeForwardCols * kHidden elements. `serve_x_stage` is the single D2H landing area (the
    // slot is not known when that copy is issued); the slab copy happens once begin_request has
    // picked the slot.
    PinnedArray<std::uint16_t> serve_x_stage{cpu_expert::kCpuServeForwardCols * 2'560};
    PinnedArray<std::uint16_t> serve_x{cpu_expert::kCpuServeRingSlots *
                                       cpu_expert::kCpuServeForwardCols * 2'560};
    PinnedArray<float> serve_partial_f32{cpu_expert::kCpuServeRingSlots *
                                         cpu_expert::kCpuServeForwardCols * 2'560};
    PinnedArray<std::uint16_t> serve_partial_bf16{cpu_expert::kCpuServeRingSlots *
                                                  cpu_expert::kCpuServeForwardCols * 2'560};
    // Block-8 item 3a: the per-slot completion words the stream waits on (pinned host memory; the
    // serve thread release-stores the round's sequence, the apply side's stream waits for it).
    // Zeroed once in the constructor -- every later store happens before any waiter exists.
    PinnedArray<std::uint32_t> serve_done_flag{cpu_expert::kCpuServeRingSlots};

    // Block-7 fix (deterministic reduce): per-job y slots -- `kCpuServeForwardCols * kHidden` BF16
    // cells per job, written only by the worker that picked the job. Grows to the widest call's
    // job count (jobs <= paths <= kMaxStagedPaths) on first use, like `divisors` below; the serve
    // thread folds the slots into partial_f32 in job order after the round, so the served sum is
    // bit-stable run to run (the parallel `+=` into partial_f32 this replaces was a data race on
    // shared token columns and let the float sum order follow completion timing).
    std::unique_ptr<PinnedArray<std::uint16_t>> serve_job_y;

    [[nodiscard]] std::uint16_t* serve_job_y_plane() {
        const auto needed = static_cast<std::size_t>(kMaxStagedPaths) *
                            static_cast<std::size_t>(cpu_expert::kCpuServeForwardCols) * 2'560;
        if (serve_job_y == nullptr || serve_job_y->size() < needed) {
            serve_job_y = std::make_unique<PinnedArray<std::uint16_t>>(needed);
        }
        return serve_job_y->data();
    }

    // The divisors are indexed by SLOT, not by path, and the slot count is a runtime knob, so this
    // one grows to the live pool size on first use (both planes in one allocation).
    std::unique_ptr<PinnedArray<float>> divisors;

    [[nodiscard]] float* divisor_plane(std::int32_t slot_count) {
        const auto needed = 2 * static_cast<std::size_t>(slot_count);
        if (divisors == nullptr || divisors->size() < needed) {
            divisors = std::make_unique<PinnedArray<float>>(needed);
        }
        return divisors->data();
    }
};

StagingControlBuffers& staging_control_buffers() {
    static StagingControlBuffers buffers;
    return buffers;
}

bool read_flag_once(const char* name) {
    const char* env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool staging_verify_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_EXPERTS_VERIFY");
    return enabled;
}

bool staging_sabotage_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_EXPERTS_SABOTAGE");
    return enabled;
}

// NINFER_FLASH_NEXT_EXPERTS_DEVICE_STAGE=1 moves the whole staging step onto the device: a decide
// kernel reads the router ids in place, resolves expert -> slot through a device slot table and
// rewrites ids to slot indices; a copy kernel then pulls the misses straight out of the pinned host
// bank. No ids readback and no per-layer cudaStreamSynchronize, so the 48-per-token host round trip
// disappears. Measured premise (lever_probe.exe): a kernel copying from the pinned bank reaches
// 22.58 GB/s for the cached miss volume, versus 22.57 GB/s for the driver DMA path -- same bandwidth,
// none of the latency.
bool device_stage_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_EXPERTS_DEVICE_STAGE");
    return enabled;
}

// NINFER_FLASH_NEXT_HOT_FIX=1 replaces the `++calls % 48 == 0` hot/pass gate with layer-transition
// counting. See the long note at the gate itself. Default 0 = off = the shipped behaviour unchanged,
// because the OLD gate also drives the decode regime we have already optimised and verified, and a
// cache-policy change must not be assumed harmless there.
bool hot_fix_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_HOT_FIX");
    return enabled;
}

// ------------------------------------------------------------------------------------------------
// P2a/P2b, first cut. All three are DEFAULT OFF, and the OFF arm is the shipped behaviour not by
// intention but by construction: the branches below are new code beside the old code, never a
// rewrite of it. (`F-76` discipline D-1: "0 = off = the shipped behaviour unchanged".)
//
// ROUND: the old `++calls % 48 == 0` pass gate is wrong under prefill (the executor is layer-major,
// so 48 calls is only THREE layers -- see the long note at the gate). This flag makes the round an
// explicit fact -- a completed sweep over the layer tables -- instead of an inference from the call
// count. HOT_FIX already counts layer transitions for `pass`; ROUND additionally records the sweep
// boundary so the route log can be closed and the next round's decisions can be prepared.
//
// GLOBAL_POOL: one pool over every layer instead of one pool per layer. The allocation is ALREADY
// one block (`ensure_slot_pool` multiplies by kMaxLayerTables and does a single cudaMalloc); only
// the indexing and the victim scan are per-layer. So this flag changes *ownership*, not memory:
// a slot is no longer owned by a layer, and a resident key becomes (layer, expert) rather than
// expert. The MoE kernels are untouched: they read ids only as a bank index and the bank simply
// gets bigger (slots -> slots * kMaxLayerTables), which is what `.experts` below reports.
//
// LFRU: victim ranking f / (now - last + 1) over a frequency that SURVIVES eviction. The existing
// `slot_uses` plane cannot serve: it is indexed by slot and reset to 1 when an expert moves in
// (expert_staging.cpp:290 and :880), so it measures "since the slot was last filled", not routing
// frequency. `evict_freq` below is indexed by the *resident key* and never reset.
// Scores are binary64 on purpose: the reference measured that integer halving is unstable over
// short horizons. LFRU is a ranking only -- `hot` still wins (F-77d: hot > LFRU > empty).
// ------------------------------------------------------------------------------------------------
bool cpu_expert_check_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_CHECK");
    return enabled;
}

// The red arm for the above: corrupts one expected spec value, so the judgement can be SEEN to fail.
bool cpu_expert_sabotage_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SABOTAGE");
    return enabled;
}

// Phase-II red-arm switch: carries a MODE NUMBER, on purpose. The phase-I readers above are pure
// booleans, so `=2` would read the same as `=1` and every phase-II red arm would silently collapse
// onto the phase-I one -- a false green of exactly the family consensus 4.3 records (QA v3 sec-1/4.2).
// The phase-I name above stays frozen; unset or "0" reads as 0 = no sabotage.
std::int32_t cpu_expert_fwd_sabotage_mode() {
    static const std::int32_t mode = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_CPU_EXPERT_FWD_SABOTAGE");
        if (env == nullptr || env[0] == '\0') { return 0; }
        return static_cast<std::int32_t>(std::strtol(env, nullptr, 10));
    }();
    return mode;
}

// ------------------------------------------------------------------------------------------------
// Phase IV (block 3, stage IV): the CPU-serve switch and its share derivation. DEFAULT OFF; the
// OFF arm is the shipped behaviour by construction (every branch below is new code beside the old
// code, never a rewrite of it -- F-76 discipline D-1). The env names, for the all-off list:
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE            0/1  master switch
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_DIVISOR    0=derive d, >0 fixed d (cpu_pcie_divisor shape)
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PCIE_FRAC  table value: PCIe share of misses (0<x<=1) -> d =
//                                                 lround(1/x); 0.35 -> d=3, i.e. the same divisor the
//                                                 derivation gives on this machine. Highest priority
//                                                 is still DIVISOR; unset keeps `derived` untouched.
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_LEGACY     0/1  pre-patch floor quota (0 links when misses<d)
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_C          experts/s override for the derivation
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_T_US       link us/record override for the derivation
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PIN        0/1  optional worker pinning (default off)
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PIN_FIRST  first CPU of the pin map (even stride; default 2)
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PREFILL    0/1  serve PREFILL calls too (default 0 = decode-only, aligned)
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_ADAPTIVE   0/1  adaptive d from live service throughput (default 0 = frozen)
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PROF       0/1  segmented serve timing line (default 0 = no new line)
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_STREAMWAIT 0/1  slot stream wait: GPU waits for the CPU round (default 0)
//   NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_POKE       0/1  driver poke after the stream-wait enqueue (default 0)
//
// d = max(1, floor(1.65 + C*t)) is derived at startup from the block-3 calibers: C = 7,390.7
// experts/s (stage III loaded closeout, 1col-cold best) and t = 292 us/record (the live mapped
// tier; the pinned tier would be 122.4) => floor(1.65 + 2.158) = 3, i.e. the frozen d0 = 3. The
// derivation is printed once (see below) so the number can be recomputed from its inputs.
struct CpuServeShare {
    bool        enabled   = false;
    bool        legacy    = false;
    bool        pin       = false;
    bool        adaptive  = false; // Block-8 item 8: live-C adaptive d (default frozen)
    int         pin_first = 2;
    int         d         = 3;
    double      c         = 7390.7; // experts/s (stage III, loaded, 1col-cold best of 3 runs)
    double      t_us      = 292.0;  // us/record  (mapped tier 263-321 us; pinned tier = 122.4)
    // ③ 按表定比例 (supervisor G2): the reference's own knob is a SHARE (`--pcie-frac 0.35`), ours is
    // a divisor, and the two are one physical quantity -- a miss takes the PCIe link exactly when
    // `miss_ordinal % d == 0` (see the serve decision in the staging body), so the link share is 1/d
    // and the CPU share is (d-1)/d. 0.35 maps to d = lround(1/0.35) = 3 = 0.3333: the same divisor
    // this machine's derivation already gives, which is why taking the table value costs no
    // behaviour change. DIVISOR keeps the highest priority; unset keeps `derived` exactly as today.
    double      frac      = 0.0;    // requested PCIe share (frac_set says whether one was given)
    bool        frac_set  = false;
    const char* mode      = "derived";
};

double read_double_once(const char* name, double fallback) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') { return fallback; }
    return std::strtod(env, nullptr);
}

std::int32_t read_int_once(const char* name, std::int32_t fallback) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') { return fallback; }
    return static_cast<std::int32_t>(std::strtol(env, nullptr, 10));
}

const CpuServeShare& cpu_serve_share() {
    static const CpuServeShare share = [] {
        CpuServeShare s;
        s.enabled   = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE");
        s.legacy    = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_LEGACY");
        s.pin       = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PIN");
        s.pin_first = read_int_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PIN_FIRST", 2);
        s.c         = read_double_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_C", 7390.7);
        s.t_us      = read_double_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_T_US", 292.0);
        const std::int32_t fixed = read_int_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_DIVISOR", 0);
        // ③ 按表定比例: a share from the table, if one was given. Out-of-range is refused rather than
        // clamped -- a silently clamped fraction makes "I set 0.9 and it ran like 0.5" undetectable,
        // and that is the same class of accident this tree already paid for twice.
        if (const char* fx = std::getenv("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PCIE_FRAC");
            fx != nullptr && fx[0] != '\0') {
            char* end      = nullptr;
            const double v = std::strtod(fx, &end);
            if (end == fx || !(v > 0.0) || !(v <= 1.0)) {
                throw std::runtime_error(
                    std::string("Flash-Next CPU-serve split: ") +
                    "NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PCIE_FRAC='" + fx +
                    "' is not a fraction in (0,1]; the link share is 1/d, so a divisor has to be "
                    "derivable from it. Unset the variable, or set the divisor directly with "
                    "NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_DIVISOR.");
            }
            s.frac     = v;
            s.frac_set = true;
        }
        // Block-8 item 8: adaptive d from the live serve throughput. The fixed divisor keeps the
        // highest priority (an explicit override beats adaptive); the derived seed stays the value
        // in force until enough service time has accumulated (cpu_serve_c_live() == 0.0).
        s.adaptive = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_ADAPTIVE") && (fixed <= 0);
        if (fixed > 0) {
            s.d    = fixed;
            s.mode = "fixed";
        } else if (s.frac_set) {
            s.d    = std::max(1, static_cast<int>(std::lround(1.0 / s.frac)));
            s.mode = "table";
        } else {
            s.d    = std::max(1, static_cast<int>(std::floor(1.65 + s.c * s.t_us * 1.0e-6)));
            s.mode = s.adaptive ? "adaptive" : "derived";
        }
        if (s.enabled) {
            // Startup self-证 line: every input of the derivation is on it, so d can be recomputed.
            std::fprintf(stderr,
                         "flash_next_cpu_expert_share d=%d mode=%s C=%.1f experts_per_s t_us=%.1f "
                         "c_t=%.3f at_least_one=1 legacy=%d workers=%d max_jobs=%d cap_records=%zu\n",
                         s.d, s.mode, s.c, s.t_us, s.c * s.t_us * 1.0e-6, s.legacy ? 1 : 0,
                         cpu_expert::kCpuServeWorkers, cpu_expert::kCpuServeMaxJobs,
                         cpu_expert::kCpuServeBiasCap);
        }
        return s;
    }();
    return share;
}

// Block-8 alignment item 9 (prefill no-serve): serving is a decode-phase mechanism -- the
// reference's split applies to verify windows, not prefill (Strata generate.cpp:242-243), and on
// B1 prefill rounds were 5,990 of all 7,830 rounds (76.5%). The master switch alone now arms
// DECODE-width calls only; set this to 1 to restore the pre-alignment behaviour where every call
// width <= kCpuServeForwardCols could serve (the A/B arm for the aligned round). Name is new in
// this tree (checked zero matches before adding, CR-8).
bool cpu_serve_prefill_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PREFILL");
    return enabled;
}

// Block-8 item 7 (segmented timing): the run/reduce/apply-wait accumulators surface on their own
// line (`flash_next_cpu_expert_serve_prof`), printed only when this is on -- the frozen serve
// line's field set is not touched. Default 0: the OFF arm's line set is unchanged.
bool cpu_serve_prof_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_PROF");
    return enabled;
}

// Block-8 item 3a: the stream-wait channel -- the GPU waits (in stream order) for the CPU round's
// completion word, so the host no longer blocks in the apply step. Default 0 = the blocking await,
// bit for bit.
bool cpu_serve_streamwait_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_STREAMWAIT");
    return enabled;
}

// Block-8 item 3b: force a driver submission right after the stream-wait enqueue (the WDDM
// "poke" candidate). Deliberately separate from 3a: the concentrated test needs the marginal of
// the poke alone. Default 0.
bool cpu_serve_poke_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_SERVE_POKE");
    return enabled;
}

// ---- Block-8 item 3a: the stream-wait channel's host plumbing -----------------------------------
// The driver entry is fetched once through the runtime (no cuda.lib link -- the tree has no other
// driver-API dependency). The flag value CU_STREAM_WAIT_VALUE_EQ (0x1, cuda.h:534) is spelled
// locally on purpose: pulling cuda.h next to the runtime headers in this TU is a header-mixing
// hazard this tree has already been burned by.
using CuStreamWaitValue32Fn = int (*)(void* stream, std::uintptr_t addr, std::uint32_t value,
                                      unsigned int flags);
constexpr unsigned int kCuStreamWaitValueEq = 0x1;

CuStreamWaitValue32Fn cu_serve_stream_wait_value32() {
    static const CuStreamWaitValue32Fn fn = [] {
        void* fp = nullptr;
        const cudaError_t err =
            cudaGetDriverEntryPointByVersion("cuStreamWaitValue32", &fp, 12000, 0);
        if (err != cudaSuccess || fp == nullptr) {
            std::fprintf(stderr,
                         "flash_next_cpu_expert_serve streamwait: driver entry "
                         "cuStreamWaitValue32 unavailable (err=%d); applying the blocking await\n",
                         static_cast<int>(err));
            return static_cast<CuStreamWaitValue32Fn>(nullptr);
        }
        return reinterpret_cast<CuStreamWaitValue32Fn>(fp);
    }();
    return fn;
}

// Device-side address of slot `slot`'s completion word; 0 when the mapping is unavailable (the
// caller then falls back to the blocking await; the reason is printed once).
std::uintptr_t serve_done_flag_device_ptr(std::size_t slot) {
    struct DevicePtrs {
        std::uintptr_t ptr[cpu_expert::kCpuServeRingSlots] = {};
        bool resolved                                      = false;
    };
    static DevicePtrs state; // process-lifetime; the buffer is a singleton too
    if (!state.resolved) {
        state.resolved = true;
        auto& ctl      = staging_control_buffers();
        for (std::uint32_t i = 0; i < cpu_expert::kCpuServeRingSlots; ++i) {
            void* dp = nullptr;
            const cudaError_t err =
                cudaHostGetDevicePointer(&dp, ctl.serve_done_flag.data() + i, 0);
            if (err != cudaSuccess || dp == nullptr) {
                std::fprintf(stderr,
                             "flash_next_cpu_expert_serve streamwait: cudaHostGetDevicePointer "
                             "failed (err=%d); applying the blocking await\n",
                             static_cast<int>(err));
                return 0;
            }
            state.ptr[i] = reinterpret_cast<std::uintptr_t>(dp);
        }
    }
    return state.ptr[slot % cpu_expert::kCpuServeRingSlots];
}

// Block-4 P1: the port probe switch and the red-arm sabotage mode (both default off; diagnostic
// only). The probe measures the ported kernels' read throughput on this layer's real records and
// checks that the three sabotage modes change the output (V2 arm feasibility). The sabotage mode
// feeds the ported forward when the serve path is switched over (block-4 wiring, later round).
bool cpu_expert_port_probe_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_CPU_EXPERT_PORT_PROBE");
    return enabled;
}

std::int32_t cpu_expert_port_sabotage_mode() {
    static const std::int32_t mode = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_CPU_EXPERT_PORT_SABOTAGE");
        if (env == nullptr || env[0] == '\0') { return 0; }
        return static_cast<std::int32_t>(std::strtol(env, nullptr, 10));
    }();
    return mode;
}

bool round_tracking_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_EXPERTS_ROUND");
    return enabled;
}

bool global_pool_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_EXPERTS_GLOBAL_POOL");
    return enabled;
}

bool lfru_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_EXPERTS_LFRU");
    return enabled;
}

// P3-1: one cudaMemcpyBatchAsync per call instead of six cudaMemcpyAsync per miss. Default 0 = the
// shipped per-copy loop, which stays in the tree verbatim as the OFF arm and as the fallback.
// The contract was read out of the CUDA 13.3 headers BEFORE this was written (see the note at the
// call site): numAttrs must not be 0, copies within a batch must not depend on each other, and the
// stream must not be the legacy NULL stream.
bool dma_batch_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_EXPERTS_DMA_BATCH");
    return enabled;
}

// ------------------------------------------------------------------------------------------------
// P5-1: experts used only by REJECTED draft columns must not count as uses.
//
// The reference states the requirement (design §11.3:1762-1763): "Experts used only by rejected
// draft columns are not credited as uses in the LFRU state, so a rejected path cannot pollute the
// cache." Note what it does NOT say: it does not say the experts are not FETCHED. They cannot be --
// the acceptance does not exist yet while the verification forward is running, which is when the
// staging happens. This is a BOOKKEEPING change, and that is the whole reason it is implementable
// at all.
//
// So the credit is deferred rather than retracted. With the switch on, a hit records the pair
// (resident key, column) instead of incrementing the frequency, and flash_next_note_live_columns()
// commits only the pairs whose column is below the accepted prefix length. Deferring beats
// retracting because a retraction has to be exactly the inverse of an increment that already
// happened, and an inexact inverse is a silent wrong count.
//
// A scalar is enough because acceptance is a PREFIX: speculative_round.cuh:164-165 "the first
// failed event determines the prefix", and :193 spells a live column as `column < accepted_count`.
//
// If the count never arrives, the round boundary commits EVERYTHING instead (and counts a stale).
// Crediting too much costs hit-rate quality; carrying a pending table into the next round would
// attribute that round's rejection to this round's uses, which is a different and worse error.
// ------------------------------------------------------------------------------------------------
bool live_columns_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_MOE_LIVE_COLUMNS");
    return enabled;
}

constexpr std::int32_t kRoutedPathsPerToken = 10; // enforced at the entry point below

struct LiveColumnRef {
    std::uint32_t key    = 0; // layer * kExpertsPerBank + expert
    std::uint16_t column = 0; // which draft column routed it
};

struct LiveColumns {
    std::vector<LiveColumnRef> refs;
    bool pending = false;
    // Self-证. The first two counters exist because the 2026-10-14 four-arm run could not be read:
    // `commits == 0` did not distinguish "the hook was never called" from "it was called and there
    // was no pool to commit into", and those need opposite fixes. `calls` moves whenever the switch
    // is on; `no_pool` moves when it is on but the global pool was never sized (P5-1 is a companion
    // of the global pool: `evict_freq` does not exist without it). A switch that is on with
    // calls == 0 is a switch that never reached the code path.
    std::uint64_t calls    = 0;
    std::uint64_t no_pool  = 0;
    std::uint64_t commits  = 0;
    std::uint64_t stales   = 0;
    std::uint64_t credited = 0;
    std::uint64_t skipped  = 0;
};

LiveColumns& live_columns() {
    static LiveColumns state;
    return state;
}

// commit_live_columns() is defined further down, immediately before halve_evict_freq(): it touches
// SlotPool, which is not declared until the slot-pool section. Defining it here would be a reference
// to an incomplete type, and the compiler says so rather than guessing.

// Distinct expert banks: the 48 text layers, plus the MTP layer if it is ever host-tiered. The pool
// is sized for this many per-layer tables; a 50th bank is a loud error, never a silent overwrite.
constexpr int kMaxLayerTables = 49;

// Experts per bank. `require_matching_geometry` below enforces it, and the global pool indexes
// keys as layer * kExpertsPerBank + expert, so the two must agree.
constexpr std::int32_t kExpertsPerBank = 512;

// ONE POOL OVER THE TEXT LAYERS ONLY. The reference is explicit (design §9.3): "Victim ranking:
// LFRU, one global pool over the 24,576 Text experts. The MTP pool runs the same policy
// separately." 24,576 = 48 * 512, i.e. the 48 text layers -- the MTP layer keeps its own pool.
//
// This is a SEPARATE constant from kMaxLayerTables on purpose, and the split matters: those two are
// one number with two meanings, which is exactly the shape of the accident recorded in
// expert_stage_kernels.cu:296-299 ("the slot count is a single constant in four places ... changing
// only some of them makes the decide kernel write out of bounds"). kMaxLayerTables is the pool's
// PHYSICAL capacity (49 allocations, 49 layer tables, the route log's dimension); this one is how
// many of them share the global pool.
constexpr std::int32_t kGlobalPoolLayers = 48;

// Resident-key planes for the global pool. A key is `layer * kExpertsPerBank + expert` because the
// same expert id is a DIFFERENT weight in every layer: a global pool must not let layer 7's expert
// 311 answer for layer 22's expert 311.
constexpr std::int32_t kMaxResidentKeys = kGlobalPoolLayers * kExpertsPerBank;

// LFRU's frequency decay. The reference measured this rather than guessed it (design §9.3, table at
// :1372-1384): with the frequency never decayed, LFRU LOSES to plain LRU by 4-5 %, "because global
// counts stay stuck on the previous topic's experts"; halving every 8,192 ticks beats never halving
// by 9.2-11.7 % on misses per token, and integer halving is unstable at short periods (+12 to +45 %
// at 1,024 ticks, because single uses truncate to zero) while fractional halving is smooth.
// So: f is a binary64 fraction, halved exactly every kLfruHalvingPeriod ticks, and a halved count
// below kLfruFloor becomes zero.
constexpr std::int64_t kLfruHalvingPeriod = 8192;
constexpr double kLfruFloor               = 1.0 / 1024.0;

// The route log keeps one ring of recent rounds so a decision can be made from the PREVIOUS round's
// routing instead of from this round's, which is the whole point of moving the decision off the
// critical path. Shape follows the reference's `route_log[round][L][T*10]` (u16 ids).
constexpr std::int32_t kRouteLogRounds = 4;

struct RouteLogRound {
    std::int32_t table_index = -1;                  // which layer table this record belongs to
    std::int32_t count       = 0;                   // number of routed paths recorded
    std::uint16_t ids[kMaxStagedPaths] = {};        // RAW expert ids, before the slot rewrite
};

struct RouteLog {
    RouteLogRound rounds[kRouteLogRounds][kMaxLayerTables];
    std::int32_t head  = 0;     // the round currently being written
    std::uint64_t closed = 0;   // completed rounds, for the counter line
};

struct StagingPlane {
    const void* staged;      // device base of the plane (slot 0)
    std::uint64_t slot_stride;
    const std::byte* source; // host bank base of the same plane
};

// ------------------------------------------------------------------------------------------------
// Judgement: what landed in the staging planes must equal the source expert of that slot.
//
// This is what makes tiering placement-only -- the MoE kernels are pure functions of the expert
// bytes they read, so identical bytes mean identical arithmetic. Prints exactly one machine-readable
// line (prose lines are never judgement input) and throws on any mismatch.
//
// The divisor check is deliberately NOT "the plane is a straight copy". The kernels index the
// divisor plane with the same id they index the codes plane with (moe_kernels.cu:99 and :203), and
// after the rewrite that id is a slot, so the only correct value is the divisor of the expert that
// slot holds. Comparing planes instead of gathered values is exactly the blind assertion that once
// let a mismatched divisor through while every byte still "verified": the experts' 32-bit weight
// scales were off by the ratio of two unrelated experts' divisors and the model answered garbage.
// Hence per-slot comparison, plus a reorder counter so the check cannot go vacuous.
void verify_staged_bytes(const Nvfp4ExpertBankView& gate_up, const Nvfp4ExpertBankView& down,
                         const std::array<StagingPlane, 4>& planes,
                         const std::array<float*, 2>& divisor_planes,
                         const std::array<std::int32_t, 512>& slot_expert,
                         const std::array<std::int32_t, kMaxStagedPaths>& used_slots,
                         int used_count, int sabotage, cudaStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<std::uint8_t> landed;
    std::size_t bytes_checked     = 0;
    std::size_t mismatched_planes = 0;
    std::size_t reordered_slots   = 0;

    for (int i = 0; i < used_count; ++i) {
        const int slot = used_slots[static_cast<std::size_t>(i)];
        if (slot < 0 || slot >= 512) {
            throw std::logic_error("Flash-Next staging verification saw an out-of-range slot");
        }
        const int expert_id = slot_expert[static_cast<std::size_t>(slot)];
        if (expert_id < 0 || expert_id >= gate_up.experts) {
            throw std::logic_error("Flash-Next staging verification saw an empty slot");
        }
        const auto expert = static_cast<std::uint64_t>(expert_id);
        if (expert != static_cast<std::uint64_t>(slot)) { ++reordered_slots; }

        for (const StagingPlane& plane : planes) {
            landed.resize(static_cast<std::size_t>(plane.slot_stride));
            const auto* staged = static_cast<const std::uint8_t*>(plane.staged) +
                                 static_cast<std::uint64_t>(slot) * plane.slot_stride;
            CUDA_CHECK(cudaMemcpy(landed.data(), staged, plane.slot_stride, cudaMemcpyDeviceToHost));
            if (std::memcmp(landed.data(), plane.source + expert * plane.slot_stride,
                            plane.slot_stride) != 0) {
                ++mismatched_planes;
            }
            bytes_checked += plane.slot_stride;
        }

        if (divisor_planes[0] != nullptr) {
            float staged_divisor = 0.0F;
            CUDA_CHECK(cudaMemcpy(&staged_divisor, divisor_planes[0] + slot, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            if (staged_divisor != gate_up.weight_scale_divisors[expert]) { ++mismatched_planes; }
            bytes_checked += sizeof(float);
        }
        if (divisor_planes[1] != nullptr) {
            float staged_divisor = 0.0F;
            CUDA_CHECK(cudaMemcpy(&staged_divisor, divisor_planes[1] + slot, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            if (staged_divisor != down.weight_scale_divisors[expert]) { ++mismatched_planes; }
            bytes_checked += sizeof(float);
        }
    }

    std::fprintf(stderr,
                 "flash_next_experts_verify slots=%d bytes=%zu mismatched_planes=%zu "
                 "slots_reordered=%zu sabotage=%d\n",
                 used_count, bytes_checked, mismatched_planes, reordered_slots, sabotage);
    if (mismatched_planes != 0) {
        throw std::runtime_error(
            "Flash-Next expert staging verification failed: staged bytes differ from the artifact "
            "mapping (NINFER_FLASH_NEXT_EXPERTS_VERIFY)");
    }
}

void require_matching_geometry(const Nvfp4ExpertBankView& bank, std::int32_t code_bytes,
                               std::int32_t scale_bytes, const char* which) {
    if (bank.code_bytes_per_expert != static_cast<std::uint64_t>(code_bytes) ||
        bank.scale_bytes_per_expert != static_cast<std::uint64_t>(scale_bytes)) {
        throw std::logic_error(std::string("Flash-Next host expert bank geometry does not match the "
                                           "staging planes for ") +
                               which);
    }
    if (bank.experts != 512) {
        throw std::logic_error(std::string("Flash-Next host expert bank has an unexpected expert "
                                           "count for ") +
                               which);
    }
}

// ------------------------------------------------------------------------------------------------
// Persistent per-layer LRU slot pool (the community's --moe-expert-cache-size, 56-80 slots).
// One table per distinct expert bank, keyed by that bank's host pointer: each layer owns a separate
// bank allocation, so the pointer identifies the layer and the MoE leaf needs no layer index.
// ------------------------------------------------------------------------------------------------
struct LayerSlots {
    std::array<std::int32_t, 512> of_expert{};   // expert id -> slot, -1 = absent
    std::array<std::int32_t, 512> slot_expert{}; // slot -> expert id, -1 = empty
    std::array<std::int32_t, 512> slot_age{};    // slot -> last-use tick
    std::array<std::int32_t, 512> slot_uses{};   // slot -> times this slot's expert was routed
    std::array<std::int32_t, 512> slot_pass{};   // slot -> pass (token step) of last use
    std::array<std::uint8_t, 512> slot_hot{};    // 1 = hot-expert residency: never evicted
};

// Hot-expert residency: the top-N most-routed resident slots of a layer are protected from eviction,
// so the experts this session keeps asking for stop moving. Recomputed once per token (48 calls)
// from the live routing histogram -- no offline profile. The community pins a learned hot store
// (Colibri's .coli_usage) and this is the same idea fed by the counters we already keep.
// NINFER_FLASH_NEXT_EXPERTS_HOT sets the size; default is a quarter of the slots, 0 disables.
std::int32_t hot_slot_count(std::int32_t slots) {
    static const std::int32_t configured = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_EXPERTS_HOT");
        if (env == nullptr || env[0] == '\0') { return -1; } // -1 = default (a quarter)
        const long parsed = std::strtol(env, nullptr, 10);
        return static_cast<std::int32_t>(parsed < 0 ? 0 : std::min<long>(parsed, 512));
    }();
    if (configured < 0) { return slots / 4; }
    return std::min(configured, slots);
}

// R1 knob: miss-gated expert dropping, F in units of the uniform routing share 1/k (k = 10 paths).
// F <= 1.0 makes the top expert provably undroppable, because the argmax weight cannot sit below the
// uniform share. Default 0 = off = bit-identical to the previous behaviour.
float moe_drop_fraction() {
    static const float configured = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_DROP_F");
        if (env == nullptr || env[0] == '\0') { return 0.0F; }
        const double parsed = std::strtod(env, nullptr);
        if (!(parsed > 0.0)) { return 0.0F; }
        return static_cast<float>(parsed > 2.0 ? 2.0 : parsed);
    }();
    return configured;
}

// R1 is a DECODE-ONLY knob, and this is the fix for a measured disaster rather than a guess.
//
// The community recipe this comes from excludes prefill BY DEFAULT, with a stated reason: with a cold
// cache almost every expert is a miss, so the same threshold discards ~42% of the routing mass
// instead of ~9%, and "prefill is compute-bound anyway, so there is little to win".
//
// Our first run armed dropping on every call, prefill included. Our prefill arrives at this function
// as 8-token slices (kFlashNextRouteDecodeMaxTokens = 8), so it is indistinguishable from a decode
// call except by width. The result: the prompt was encoded with a butchered routing, and the model
// degenerated into emitting one repeated token from the very first step. Meanwhile the counters
// looked excellent -- moved_mib -69%, decode 24.33 -> 65.85 tok/s, acceptance 55.8% -> 94.3% -- because
// a degenerate target is trivially predictable. That is exactly the failure the external report warned
// about: judge this knob on --ppl-step or on checkable answers, never by reading the output.
//
// NINFER_FLASH_NEXT_MOE_DROP_MAX_TOKENS (default 3) gates it: dropping is armed only when the call is
// at most this wide. Our MTP verify call is 3 tokens, a prefill slice is 8.
std::int32_t moe_drop_max_tokens() {
    static const std::int32_t configured = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_DROP_MAX_TOKENS");
        if (env == nullptr || env[0] == '\0') { return 3; }
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed < 1) { return 1; }
        return static_cast<std::int32_t>(parsed > 8 ? 8 : parsed);
    }();
    return configured;
}

// R2b knob: a SEPARATE drop fraction for the wide (prefill) calls.
//
// Why this exists -- measured, and it is the whole story of the prefill result:
//   gate<=3 (decode only) : dropped= 3,168   moved_mib=208,971
//   gate<=8 (prefill too) : dropped=118,381   moved_mib= 54,541
// => 97.3% of ALL substitutions happen during prefill, because a fresh prompt arrives with a COLD
// cache, so at threshold F/k most routed paths qualify as "low-score misses" there.
//
// That concentration is also where the only quality cost we have found lives: with substitution the
// decode-only setting is knowledge-clean (6/6 MMLU) while arming prefill deterministically loses one
// item (5/6, reproduced bit-for-bit). Plain dropping armed in prefill is far worse -- it collapses
// the model outright (0/15 on GSM8K, every reply 1-10 tokens of repeated control text).
//
// So the prefill phase needs its OWN threshold. Default 0 = prefill NOT armed = exactly today's
// behaviour, which keeps the stack safe by construction.
float moe_drop_fraction_prefill() {
    static const float configured = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_DROP_F_PREFILL");
        if (env == nullptr || env[0] == '\0') { return 0.0F; }
        const double parsed = std::strtod(env, nullptr);
        if (!(parsed > 0.0)) { return 0.0F; }
        return static_cast<float>(parsed > 2.0 ? 2.0 : parsed);
    }();
    return configured;
}

// R2c knob: the damping factor lambda that joins the two rungs into ONE continuous axis.
//
// The fixup below computes, for the set of replaced paths with total weight d = sum(w_dropped):
//     survivor scale = (1 - lambda * d) / kept_sum          substituted weight = lambda * w
// which keeps the routed sum at exactly 1 for every lambda in [0, 1] and reproduces both ends:
//     lambda = 0  -> survivors renormalised to 1, dropped weight 0   == the pure DROP variant
//     lambda = 1  -> survivors untouched, dropped keep their weight  == the pure SUBSTITUTION variant
// lambda = 1 is the only value with NO survivor amplification, and amplification is what correlates
// with the prefill collapse, so 1 is the default and the interesting direction to explore is NOT
// downward -- it is whatever lets us cut prefill distortion while keeping the bytes.
// Default 1.0 = the substitution end = today's behaviour.
float moe_subst_damp() {
    static const float configured = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_SUBST_DAMP");
        if (env == nullptr || env[0] == '\0') { return 1.0F; }
        const double parsed = std::strtod(env, nullptr);
        if (!(parsed >= 0.0)) { return 1.0F; }
        return static_cast<float>(parsed > 1.0 ? 1.0 : parsed);
    }();
    return configured;
}

// R2 knob: SUBSTITUTION instead of dropping -- the second rung of the same ladder.
//
// The drop variant (SUBST=0, the default) sets a dropped path's weight to ZERO and renormalises the
// survivors, so that expert's contribution is erased outright; only the PCIe copy is saved. The
// substitution variant keeps the path's ORIGINAL weight and merely redirects its slot at a resident
// expert (the top expert is resident by construction, and pointing at a committed slot is what keeps
// the matmul off uncommitted memory). The router's top-k weights already sum to 1, so the routed sum
// stays exactly 1 and NO renormalisation is needed -- which also means scratch.alpha is left exactly
// as the router wrote it, so the H2D copy of the weights can be skipped entirely: one less
// synchronisation point per layer per forward.
//
// The BYTE saving is identical to the drop variant (the copy is skipped either way). Only the
// numerics differ, and that is the whole point of the switch: it lets us ask whether keeping the
// routing mass buys the right to arm the policy somewhere we currently dare not arm it -- prefill.
//
// Why that question matters (measured on this machine, GSM8K 15 questions, deployed MTP regime):
//   off    22.293 MiB/call  12/15  13.46 tok/s   TTFT 2.4-3.0 s  (~40-50 tok/s effective prefill)
//   F=1.0  13.841 MiB/call  12/15  17.02 tok/s   <- drop, in the stack; wrong SET identical to off
//   F=2.0  12.192 MiB/call  12/15  17.09 tok/s   <- only -11.9% more bytes for +0.4% tok/s => the
//                                                   "unlock F>1.0" rationale is DEAD, and the wrong
//                                                   set changed (q13 fixed, q8 broke) => real damage
// So prefill is now the target: it is roughly HALF of end-to-end latency on question-answering
// workloads, and we deliberately disarm dropping there (MOE_DROP_MAX_TOKENS=3) because plain
// dropping butchered the prompt encoding on 09-27. Substitution may be the safe way to arm it.
//
// Default 0 = off = bit-identical to the previous behaviour.
bool moe_subst_enabled() {
    static const bool configured = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_SUBST");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return configured;
}

void refresh_hot_slots(LayerSlots& table, std::int32_t slots, std::int32_t hot_count) {
    table.slot_hot.fill(0);
    if (hot_count <= 0) { return; }
    for (std::int32_t rank = 0; rank < hot_count; ++rank) {
        std::int32_t best_slot = -1;
        std::int32_t best_uses = 0;
        for (std::int32_t s = 0; s < slots; ++s) {
            const auto slot = static_cast<std::size_t>(s);
            if (table.slot_hot[slot] != 0) { continue; }
            if (table.slot_expert[slot] < 0) { continue; } // empty slot cannot be hot
            const std::int32_t uses = table.slot_uses[slot];
            if (best_slot < 0 || uses > best_uses) {
                best_slot = s;
                best_uses = uses;
            }
        }
        if (best_slot < 0) { break; }
        table.slot_hot[static_cast<std::size_t>(best_slot)] = 1;
    }
}

struct SlotPool {
    bool ready        = false;
    std::int32_t slots = 0;
    std::uint8_t* gate_up_codes  = nullptr;
    std::uint8_t* gate_up_scales = nullptr;
    std::uint8_t* down_codes     = nullptr;
    std::uint8_t* down_scales    = nullptr;
    float* gate_up_divisors = nullptr;
    float* down_divisors    = nullptr;
    std::vector<const void*> layer_keys;
    std::vector<LayerSlots> layers;
    std::int32_t tick = 0;  // per-use counter, drives LRU
    std::int32_t pass = 0;  // per-token counter, drives Least-Stale (see the eviction rule)
    // Community-shaped counters (they report hits / misses / evictions the same way).
    std::uint64_t hits       = 0;
    std::uint64_t misses     = 0;
    std::uint64_t evictions  = 0;
    std::uint64_t dropped    = 0;   // R1: routed paths replaced by a resident expert (see below)
    // Phase IV: routed MISS paths handed to the CPU serve engine instead of the PCIe copy. A
    // served path is counted as a miss AND here -- never as a hit (the recorded failure this
    // pair exists to prevent: "CPU-served although the policy counts them as hits").
    std::uint64_t cpu_served = 0;
    // ③ 按表定比例: the OTHER half of the split. A miss either goes to the CPU engine (`cpu_served`)
    // or keeps the PCIe link and gets a copy -- counting only the first would make the measured link
    // share a subtraction away from a name (`misses` also carries drops and the served paths).
    std::uint64_t serve_link_paths = 0;
    // Phase IV self-证: served paths whose write-back found no committed-slot template (0 is the
    // guarded fallback; a non-zero value here is a red instrument, never silent).
    std::uint64_t serve_fallback = 0;
    std::uint64_t calls      = 0;
    std::uint64_t bytes_moved = 0;

    // ---- P2a: explicit rounds -------------------------------------------------------------------
    // `last_table_index` is the layer the previous call staged. The round closes when the layer
    // order wraps (the table index goes backwards), which needs no knowledge of how many layer
    // tables exist and cannot drift when that count changes.
    std::int32_t last_table_index = -1;
    std::int64_t round            = 0;
    RouteLog route_log;

    // ---- P2b: global pool + LFRU ----------------------------------------------------------------
    // Only touched when the corresponding flag is on; when it is off these stay untouched and the
    // per-layer `lanes` path runs exactly as shipped.
    bool global_ready = false;
    // Every plane is HOST memory and a std::vector: nothing here is copied to the device, so a
    // pointer/owner split would buy nothing and cost a leak path.
    std::vector<std::int32_t> key_of_slot;    // [kMaxLayerTables * slots]  resident key, -1 = empty
    std::vector<std::int32_t> slot_of_key;    // [kMaxResidentKeys]         key -> global slot, -1 = absent
    std::vector<std::int32_t> g_slot_age;     // [kMaxLayerTables * slots]
    std::vector<std::int32_t> g_slot_pass;    // [kMaxLayerTables * slots]
    std::vector<std::uint8_t> g_slot_hot;     // [kGlobalPoolLayers * slots]
    // binary64, per the reference's measured decay (see kLfruHalvingPeriod): a fraction, not a count,
    // and never reset -- a frequency that dies with its slot cannot rank anything.
    std::vector<double> evict_freq;           // [kMaxResidentKeys] SURVIVES eviction -- LFRU's f
    std::uint64_t lfru_rankings = 0;          // self-证: how many victim choices LFRU actually ranked
    std::uint64_t lfru_fallbacks = 0;         // self-证: how many fell back to Least-Stale instead
    std::uint64_t lfru_halvings = 0;          // self-证: how many times f was decayed
};

SlotPool& slot_pool() {
    static SlotPool pool;
    return pool;
}

void ensure_slot_pool(SlotPool& pool, std::int32_t slots) {
    if (pool.ready) { return; }

    const auto per_layer_codes   = static_cast<std::size_t>(slots) * kStagedGateUpCodeBytes;
    const auto per_layer_scales  = static_cast<std::size_t>(slots) * kStagedGateUpScaleBytes;
    const auto per_layer_dcodes  = static_cast<std::size_t>(slots) * kStagedDownCodeBytes;
    const auto per_layer_dscales = static_cast<std::size_t>(slots) * kStagedDownScaleBytes;
    const auto per_layer_divs    = static_cast<std::size_t>(slots) * sizeof(float);

    const std::size_t codes_bytes   = per_layer_codes * kMaxLayerTables;
    const std::size_t scales_bytes  = per_layer_scales * kMaxLayerTables;
    const std::size_t dcodes_bytes  = per_layer_dcodes * kMaxLayerTables;
    const std::size_t dscales_bytes = per_layer_dscales * kMaxLayerTables;
    const std::size_t div_bytes     = per_layer_divs * kMaxLayerTables;
    const std::size_t need = codes_bytes + scales_bytes + dcodes_bytes + dscales_bytes + 2 * div_bytes;

    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (need > free_bytes) {
        throw std::runtime_error("Flash-Next expert slot pool needs " +
                                 std::to_string(need >> 20) + " MiB but only " +
                                 std::to_string(free_bytes >> 20) +
                                 " MiB of device memory is free; lower "
                                 "NINFER_FLASH_NEXT_EXPERTS_SLOTS (currently " +
                                 std::to_string(slots) + ")");
    }

    CUDA_CHECK(cudaMalloc(&pool.gate_up_codes, codes_bytes));
    CUDA_CHECK(cudaMalloc(&pool.gate_up_scales, scales_bytes));
    CUDA_CHECK(cudaMalloc(&pool.down_codes, dcodes_bytes));
    CUDA_CHECK(cudaMalloc(&pool.down_scales, dscales_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pool.gate_up_divisors), div_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pool.down_divisors), div_bytes));

    pool.layers.assign(kMaxLayerTables, LayerSlots{});
    pool.layer_keys.assign(kMaxLayerTables, nullptr);
    for (LayerSlots& table : pool.layers) {
        table.of_expert.fill(-1);
        table.slot_expert.fill(-1);
        table.slot_age.fill(0);
        table.slot_uses.fill(0);
        table.slot_pass.fill(-1);
        table.slot_hot.fill(0);
    }
    // ---- P2b: the global pool's own planes --------------------------------------------------------
    // Host memory only, and only when asked for. `need` above is unchanged: the device allocation is
    // already ONE block over kMaxLayerTables (see the note on the flags), so a global pool adds
    // bookkeeping, not bytes. That is also why this flag cannot make the device run out of room.
    if (global_pool_enabled() || lfru_enabled()) {
        const std::size_t total_slots =
            static_cast<std::size_t>(kGlobalPoolLayers) * static_cast<std::size_t>(slots);
        pool.key_of_slot.assign(total_slots, -1);
        pool.slot_of_key.assign(static_cast<std::size_t>(kMaxResidentKeys), -1);
        pool.g_slot_age.assign(total_slots, 0);
        pool.g_slot_pass.assign(total_slots, -1);
        pool.g_slot_hot.assign(total_slots, 0);
        // Frequencies are keyed by the resident key and are NEVER reset on eviction. That is the
        // whole difference from `slot_uses`, which lives and dies with a slot.
        pool.evict_freq.assign(static_cast<std::size_t>(kMaxResidentKeys), 0);
        pool.global_ready = true;
        std::fprintf(stderr,
                     "flash_next_experts_cache global_pool=%d lfru=%d slots_total=%zu "
                     "keys=%d planes_kib=%.0f\n",
                     global_pool_enabled() ? 1 : 0, lfru_enabled() ? 1 : 0, total_slots,
                     kMaxResidentKeys,
                     static_cast<double>(total_slots * (4 + 4 + 1) +
                                         static_cast<std::size_t>(kMaxResidentKeys) * (8 + 4)) /
                         1024.0);
    }

    pool.slots = slots;
    pool.ready = true;

    std::fprintf(stderr,
                 "flash_next_experts_cache slots=%d layer_tables=%d pool_mib=%.0f "
                 "free_mib_before=%.0f free_after_pool_mib=%.0f\n",
                 slots, kMaxLayerTables, static_cast<double>(need) / (1024.0 * 1024.0),
                 static_cast<double>(free_bytes) / (1024.0 * 1024.0),
                 static_cast<double>(free_bytes - need) / (1024.0 * 1024.0));
}

int layer_table_for(SlotPool& pool, const void* key) {
    for (std::size_t i = 0; i < pool.layer_keys.size(); ++i) {
        if (pool.layer_keys[i] == key) { return static_cast<int>(i); }
    }
    for (std::size_t i = 0; i < pool.layer_keys.size(); ++i) {
        if (pool.layer_keys[i] == nullptr) {
            pool.layer_keys[i] = key;
            return static_cast<int>(i);
        }
    }
    throw std::runtime_error(
        "Flash-Next expert slot pool ran out of per-layer tables (more than 64 expert banks)");
}

// ------------------------------------------------------------------------------------------------
// P2b: victim choice and hot re-election over ONE pool instead of 49 per-layer pools.
//
// Why this is affordable on the host, and was not affordable on the device: the scan is
// O(kMaxLayerTables * slots) -- 49 * 128 = 6,272 reads of planes totalling ~60 KiB, i.e. L1/L2
// resident -- and it runs once per MISS (~140 per token by the measured moved_mib), not once per
// routed path. The device path paid the same scan from a `<<<1, 1>>>` kernel with no cache to hide
// it, and that is why it is refused up front rather than ported.
//
// `hot` still wins: a hot slot is never a victim until nothing else is left (F-77d). That is also
// what makes the two protections compose -- `hot` is an unconditional exclusion, LFRU is only a
// RANKING among the remaining candidates. Conflating them would make a hit-rate change impossible
// to attribute.
// ------------------------------------------------------------------------------------------------

// The candidate set is walked twice: once over slots last used in an EARLIER pass ("stale", the
// SpecMD finding: an entry evicted inside the same pass and needed again immediately after is what
// plain LRU gets wrong, measured 4.5-12.6% of accesses under LRU against 1.6-1.9% with the split),
// then once over anything not hot. Only if both are empty does anything, hot included, become a
// candidate -- a fully hot pool must still make room rather than fail.
// LFRU's decay. Called on the tick boundary: O(kMaxResidentKeys) = 24,576 doubles once every
// kLfruHalvingPeriod ticks (~170 rounds), i.e. amortised to nothing against 48 calls per round.
// NOT applied on eviction -- a frequency outliving its slot is the whole point. A halved fraction
// below kLfruFloor becomes exactly zero so the plane does not silt up with values that can never
// win a ranking.
// P5-1. Commit the uses recorded by the hits of the round just staged. `accepted_count < 0` means
// "no count ever arrived" and commits everything -- see the long note beside LiveColumns. `stale`
// only chooses which self-证 counter moves, never what is credited.
void commit_live_columns(SlotPool& pool, std::int32_t accepted_count, bool stale) {
    LiveColumns& lc = live_columns();
    // The frequency plane must be the same length as the key index, because a ref key is an index
    // into BOTH. If it is not (a pool that was never sized, i.e. the no-cache path), drop the refs
    // instead of writing: past the end of that vector is heap corruption, not a wrong counter.
    const bool plane_ok = !pool.evict_freq.empty() &&
                          pool.evict_freq.size() == pool.slot_of_key.size();
    if (lc.pending) {
        if (plane_ok) {
            for (const LiveColumnRef& ref : lc.refs) {
                const bool live = (accepted_count < 0) ||
                                  (static_cast<std::int32_t>(ref.column) < accepted_count);
                if (live) {
                    pool.evict_freq[static_cast<std::size_t>(ref.key)] += 1.0;
                    ++lc.credited;
                } else {
                    ++lc.skipped;
                }
            }
        } else {
            lc.skipped += lc.refs.size();
        }
        lc.refs.clear();
        lc.pending = false;
    }
    if (stale) {
        ++lc.stales;
    } else {
        ++lc.commits;
    }
}

void halve_evict_freq(SlotPool& p) {
    for (double& f : p.evict_freq) {
        if (f == 0.0) { continue; }
        f *= 0.5;
        if (f < kLfruFloor) { f = 0.0; }
    }
    ++p.lfru_halvings;
}

std::int32_t global_pick_victim(SlotPool& p) {    const std::size_t total = p.key_of_slot.size();

    // 1. a never-filled slot costs nothing to take and cannot lose anything.
    for (std::size_t s = 0; s < total; ++s) {
        if (p.key_of_slot[s] < 0) { return static_cast<std::int32_t>(s); }
    }

    const bool use_lfru = lfru_enabled();
    for (int stage = 0; stage < 3; ++stage) {
        const bool stale_only = (stage == 0);
        const bool skip_hot   = (stage < 2);
        std::int32_t victim    = -1;
        std::int32_t best_rank = -1;
        double best_score      = 0.0;
        for (std::size_t s = 0; s < total; ++s) {
            const std::size_t slot = s;
            if (skip_hot && p.g_slot_hot[slot] != 0) { continue; }
            if (stale_only && p.g_slot_pass[slot] == p.pass) { continue; }
            if (use_lfru) {
                // A filled slot must carry a valid key. If it does not, the two planes disagree --
                // skip it and count the disagreement rather than score a key that is not resident.
                const std::int32_t rk = p.key_of_slot[slot];
                if (rk < 0) {
                    ++p.lfru_fallbacks;
                    continue;
                }
                const std::int64_t age =
                    static_cast<std::int64_t>(p.tick) - static_cast<std::int64_t>(p.g_slot_age[slot]) + 1;
                // Already a binary64 fraction (kLfruHalvingPeriod's note): no cast, so the division
                // below is the same arithmetic the reference's replay uses.
                const double score = p.evict_freq[static_cast<std::size_t>(rk)] /
                                     static_cast<double>(age > 0 ? age : 1);
                // Ties evict the LOWER KEY, not the lower slot. The reference is explicit (§9.3:
                // "ties evict the lower expert id, so host and replay decisions are identical step
                // for step") -- and slot order is allocation order, which is not reproducible.
                if (victim < 0 || score < best_score ||
                    (score == best_score && rk < best_rank)) {
                    victim     = static_cast<std::int32_t>(slot);
                    best_rank  = rk;
                    best_score = score;
                }
            } else {
                const std::int32_t age = p.g_slot_age[slot];
                if (victim < 0 || age < p.g_slot_age[static_cast<std::size_t>(victim)]) {
                    victim = static_cast<std::int32_t>(slot);
                }
            }
        }
        if (victim >= 0) {
            if (use_lfru) { ++p.lfru_rankings; }
            return victim;
        }
    }
    return -1;
}

// Hot re-election for the global pool: the top-N most-routed RESIDENT slots of THIS layer are
// protected. Ranked by `evict_freq`, which is keyed by the resident key and never reset -- the
// per-layer path's `slot_uses` is indexed by slot and reset to 1 on every fill, so it cannot rank
// anything across an eviction. Clearing is confined to this layer's keys: with one pool the layers
// interleave, so a blanket clear would drop every other layer's hot set.
void global_refresh_hot(SlotPool& p, std::int32_t table_index, std::int32_t hot_count) {
    if (hot_count <= 0) { return; }
    const std::int32_t base = table_index * kExpertsPerBank;
    const std::int32_t top  = base + kExpertsPerBank;
    for (std::size_t s = 0; s < p.key_of_slot.size(); ++s) {
        const std::int32_t rk = p.key_of_slot[s];
        if (rk >= base && rk < top) { p.g_slot_hot[s] = 0; }
    }
    for (std::int32_t rank = 0; rank < hot_count; ++rank) {
        std::int32_t best_slot = -1;
        std::int32_t best_rank = -1;
        double best_f          = 0.0;
        for (std::size_t s = 0; s < p.key_of_slot.size(); ++s) {
            if (p.g_slot_hot[s] != 0) { continue; }
            const std::int32_t rk = p.key_of_slot[s];
            if (rk < base || rk >= top) { continue; }
            const double f = p.evict_freq[static_cast<std::size_t>(rk)];
            if (best_slot < 0 || f > best_f || (f == best_f && rk < best_rank)) {
                best_slot = static_cast<std::int32_t>(s);
                best_rank = rk;
                best_f    = f;
            }
        }
        if (best_slot < 0) { break; }
        p.g_slot_hot[static_cast<std::size_t>(best_slot)] = 1;
    }
}

// One call site for both pools. Deliberately a single place: the two victim implementations already
// drifted apart silently (the device's three passes carry no hot protection, the host's four do),
// and a second such drift in the hot rule is exactly what must not happen again.
void refresh_hot_any(SlotPool& p, LayerSlots& table, std::int32_t table_index, std::int32_t slots) {
    // Per TABLE, not per pool: a table outside the global pool (the MTP layer) must re-elect from its
    // own LayerSlots, because its slots do not live in the global planes at all.
    const bool pooled = p.global_ready && table_index >= 0 && table_index < kGlobalPoolLayers;
    if (pooled) {
        global_refresh_hot(p, table_index, hot_slot_count(slots));
    } else {
        refresh_hot_slots(table, slots, hot_slot_count(slots));
    }
}

} // namespace

// Block-6 W1/W2: the CPU-serve forward face rides the serve master switch (no new env name, D-1);
// this predicate exists so the MoE workspace allocation can size the partial plane exactly when the
// face can run. Read once per process through cpu_serve_share, like every other switch here.
bool flash_next_cpu_serve_forward_enabled() { return cpu_serve_share().enabled; }

// ================================================================================================
// Block-8 alignment item 1 (INSTRUMENT side) -- the prefill prefetch ring.
//
// Contract (joint interface piece, ①-chunk波宽接口小件-联合-20261015.md): wave calls (tokens > 8,
// the prefill waves) get a per-layer, FIXED-ORDER fill of the whole 512-expert bank into a
// ping-pong pair of device rings, issued on a private copy stream so the copy engine works while
// the layer's attention runs. Consumption -- id -> ring-slot direct mapping (expert id IS the
// slot), skipping the per-miss copies -- is the OPERATOR side's seam (`prefetch_ring_engaged` /
// `prefetch_ring_planes` below are the interface). This side owns: the rings, the fill, the event
// ordering, the trigger and the self-proof line.
//
// ORDER OF THE GUARD CHAIN (why the ping-pong is safe):
//   fill(l) writes ring (l % 2); that ring was last READ by kernels(l-2). The private copy stream
//   waits `guard`, an event that is re-recorded on the ENGINE stream at every fill point -- the
//   guard recorded at layer l-1 covers everything enqueued up to staging(l-1), which includes
//   kernels(l-2). After the copies, `fill_done` is recorded on the copy stream and the engine
//   stream waits it, so kernels(l) can never read a half-filled ring.
//
// DEFAULT OFF (env below): one static read, no allocation, no stream. The OFF arm is untouched.
// ================================================================================================
bool ring_prefill_enabled() {
    static const bool enabled = read_flag_once("NINFER_FLASH_NEXT_EXPERTS_RING_PREFILL");
    return enabled;
}

struct PrefetchRingBuffer {
    std::uint8_t* gu_codes    = nullptr;
    std::uint8_t* gu_scales   = nullptr;
    std::uint8_t* dn_codes    = nullptr;
    std::uint8_t* dn_scales   = nullptr;
    float*        gu_divisors = nullptr;
    float*        dn_divisors = nullptr;
    std::size_t   bytes       = 0;
};

struct PrefetchRingState {
    bool tried               = false;
    bool usable              = false;
    PrefetchRingBuffer ring[2];
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t fill_done    = nullptr;
    cudaEvent_t guard        = nullptr;
    bool guard_valid         = false;
    std::int32_t fills       = 0;   // ring for the next fill = fills % 2
    std::int32_t last_layer  = -1;  // raw table_index layer-change detection
    std::vector<void*> batch_dst;
    std::vector<void*> batch_src;
    std::vector<std::size_t> batch_sizes;
};

PrefetchRingState& prefetch_ring_state() {
    static PrefetchRingState s;
    return s;
}

void prefetch_ring_init(PrefetchRingState& st) noexcept {
    st.tried = true;
    const std::size_t gu_codes_seg =
        static_cast<std::size_t>(kExpertsPerBank) * static_cast<std::size_t>(kStagedGateUpCodeBytes);
    const std::size_t gu_scales_seg =
        static_cast<std::size_t>(kExpertsPerBank) * static_cast<std::size_t>(kStagedGateUpScaleBytes);
    const std::size_t dn_codes_seg =
        static_cast<std::size_t>(kExpertsPerBank) * static_cast<std::size_t>(kStagedDownCodeBytes);
    const std::size_t dn_scales_seg =
        static_cast<std::size_t>(kExpertsPerBank) * static_cast<std::size_t>(kStagedDownScaleBytes);
    const std::size_t div_bytes = static_cast<std::size_t>(kExpertsPerBank) * sizeof(float);
    const std::size_t ring_bytes = gu_codes_seg + gu_scales_seg + dn_codes_seg + dn_scales_seg +
                                   2 * div_bytes;
    const std::size_t total = 2 * ring_bytes;
    void* raw               = nullptr;
    if (cudaMalloc(&raw, total) != cudaSuccess || raw == nullptr) {
        std::fprintf(stderr,
                     "flash_next_experts_ring_prefill: cudaMalloc of %.2f GiB failed; ring "
                     "prefetch stays off\n",
                     static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0));
        return;
    }
    auto* base = static_cast<std::uint8_t*>(raw);
    for (int r = 0; r < 2; ++r) {
        std::uint8_t* p            = base + static_cast<std::size_t>(r) * ring_bytes;
        st.ring[r].gu_codes        = p;
        st.ring[r].gu_scales       = p + gu_codes_seg;
        st.ring[r].dn_codes        = st.ring[r].gu_scales + gu_scales_seg;
        st.ring[r].dn_scales       = st.ring[r].dn_codes + dn_codes_seg;
        st.ring[r].gu_divisors     = reinterpret_cast<float*>(st.ring[r].dn_scales + dn_scales_seg);
        st.ring[r].dn_divisors     = st.ring[r].gu_divisors + kExpertsPerBank;
        st.ring[r].bytes           = ring_bytes;
    }
    if (cudaStreamCreateWithFlags(&st.copy_stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&st.fill_done, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&st.guard, cudaEventDisableTiming) != cudaSuccess) {
        std::fprintf(stderr,
                     "flash_next_experts_ring_prefill: copy stream / events could not be created; "
                     "ring prefetch stays off\n");
        return;
    }
    st.batch_dst.reserve(static_cast<std::size_t>(kExpertsPerBank) * 4 + 2);
    st.batch_src.reserve(static_cast<std::size_t>(kExpertsPerBank) * 4 + 2);
    st.batch_sizes.reserve(static_cast<std::size_t>(kExpertsPerBank) * 4 + 2);
    st.usable = true;
}

// Issue the fixed-order fill of the current layer's whole bank into the next ring. Called from the
// staging path on the FIRST WAVE of each layer (see ring_prefill_maybe_fill, below).
void prefetch_ring_fill(std::int32_t table_index, const Nvfp4ExpertBankView& gate_up,
                        const Nvfp4ExpertBankView& down, cudaStream_t engine_stream) {
    PrefetchRingState& st = prefetch_ring_state();
    if (!st.tried) { prefetch_ring_init(st); }
    if (!st.usable) { return; }

    const std::int32_t r          = st.fills % 2;
    const PrefetchRingBuffer& dst = st.ring[r];
    const std::size_t experts     = static_cast<std::size_t>(gate_up.experts);
    const double t0               = host_now_ms();

    if (st.guard_valid) { CUDA_CHECK(cudaStreamWaitEvent(st.copy_stream, st.guard, 0)); }

    st.batch_dst.clear();
    st.batch_src.clear();
    st.batch_sizes.clear();
    const auto* const gu_codes_src =
        reinterpret_cast<const std::uint8_t*>(gate_up.codes);
    const auto* const gu_scales_src =
        reinterpret_cast<const std::uint8_t*>(gate_up.scales);
    const auto* const dn_codes_src = reinterpret_cast<const std::uint8_t*>(down.codes);
    const auto* const dn_scales_src =
        reinterpret_cast<const std::uint8_t*>(down.scales);
    for (std::size_t e = 0; e < experts; ++e) {
        st.batch_dst.push_back(dst.gu_codes + e * kStagedGateUpCodeBytes);
        st.batch_src.push_back(const_cast<std::uint8_t*>(
            gu_codes_src + e * static_cast<std::size_t>(gate_up.code_bytes_per_expert)));
        st.batch_sizes.push_back(static_cast<std::size_t>(kStagedGateUpCodeBytes));

        st.batch_dst.push_back(dst.gu_scales + e * kStagedGateUpScaleBytes);
        st.batch_src.push_back(const_cast<std::uint8_t*>(
            gu_scales_src + e * static_cast<std::size_t>(gate_up.scale_bytes_per_expert)));
        st.batch_sizes.push_back(static_cast<std::size_t>(kStagedGateUpScaleBytes));

        st.batch_dst.push_back(dst.dn_codes + e * kStagedDownCodeBytes);
        st.batch_src.push_back(const_cast<std::uint8_t*>(
            dn_codes_src + e * static_cast<std::size_t>(down.code_bytes_per_expert)));
        st.batch_sizes.push_back(static_cast<std::size_t>(kStagedDownCodeBytes));

        st.batch_dst.push_back(dst.dn_scales + e * kStagedDownScaleBytes);
        st.batch_src.push_back(const_cast<std::uint8_t*>(
            dn_scales_src + e * static_cast<std::size_t>(down.scale_bytes_per_expert)));
        st.batch_sizes.push_back(static_cast<std::size_t>(kStagedDownScaleBytes));
    }
    st.batch_dst.push_back(dst.gu_divisors);
    st.batch_src.push_back(const_cast<float*>(gate_up.weight_scale_divisors));
    st.batch_sizes.push_back(experts * sizeof(float));
    st.batch_dst.push_back(dst.dn_divisors);
    st.batch_src.push_back(const_cast<float*>(down.weight_scale_divisors));
    st.batch_sizes.push_back(experts * sizeof(float));

    for (std::size_t i = 0; i < st.batch_dst.size(); ++i) {
        CUDA_CHECK(cudaMemcpyAsync(st.batch_dst[i], st.batch_src[i], st.batch_sizes[i],
                                   cudaMemcpyHostToDevice, st.copy_stream));
    }
    CUDA_CHECK(cudaEventRecord(st.fill_done, st.copy_stream));
    CUDA_CHECK(cudaStreamWaitEvent(engine_stream, st.fill_done, 0));
    CUDA_CHECK(cudaEventRecord(st.guard, engine_stream));
    st.guard_valid = true;
    ++st.fills;

    std::fprintf(stderr,
                 "flash_next_experts_ring_prefill ring=%d layer=%d experts=%zu bytes=%llu "
                 "host_ms=%.2f\n",
                 r, table_index, experts,
                 static_cast<unsigned long long>(dst.bytes), host_now_ms() - t0);
}

// Trigger: once per layer, on its first wave. `table_index` is the raw first-touch layer-table id
// (the same identity the hot-fix block uses): consecutive waves of one layer share it, the next
// layer changes it -- and the order wrap does too, so every layer visit in every chunk pass gets
// exactly one fill (the aligned target has one pass per layer; extra passes simply refill, which
// is safe and is the "8x on chunk=128" note in the interface piece).
void ring_prefill_maybe_fill(std::int32_t table_index, int tokens,
                             const Nvfp4ExpertBankView& gate_up, const Nvfp4ExpertBankView& down,
                             cudaStream_t engine_stream) {
    if (!ring_prefill_enabled()) { return; }
    if (tokens <= kFlashNextRouteDecodeMaxTokens) { return; }
    if (table_index < 0) { return; } // no pool, no layer identity: the ring path needs the pool setup
    PrefetchRingState& st = prefetch_ring_state();
    if (table_index == st.last_layer) { return; }
    st.last_layer = table_index;
    if (gate_up.experts != kExpertsPerBank || down.experts != kExpertsPerBank) {
        static std::atomic<bool> warned{false};
        bool expected = false;
        if (warned.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr,
                         "flash_next_experts_ring_prefill: bank expert count != %d; ring prefetch "
                         "stays off\n",
                         static_cast<int>(kExpertsPerBank));
        }
        return;
    }
    prefetch_ring_fill(table_index, gate_up, down, engine_stream);
}

// ---- the consumption seam (OPERATOR side) -------------------------------------------------------
// `prefetch_ring_engaged` says the ring path is live for THIS call: the operator's edits gate the
// copy-skip / direct id mapping on it. `prefetch_ring_planes` returns the ring the latest fill
// completed for -- the ring this call's kernels must read (expert id == slot). Both are inert when
// the env is off (engaged() is then false and planes() is all-null).
struct PrefetchRingPlanes {
    const std::byte* gu_codes    = nullptr;
    const std::byte* gu_scales   = nullptr;
    const std::byte* dn_codes    = nullptr;
    const std::byte* dn_scales   = nullptr;
    const float*     gu_divisors = nullptr;
    const float*     dn_divisors = nullptr;
    std::int32_t     experts     = 0;
};

bool prefetch_ring_engaged(int tokens_in) {
    return tokens_in > kFlashNextRouteDecodeMaxTokens && ring_prefill_enabled() &&
           prefetch_ring_state().usable;
}

PrefetchRingPlanes prefetch_ring_planes() {
    PrefetchRingPlanes v;
    PrefetchRingState& st = prefetch_ring_state();
    if (!st.usable || st.fills == 0) { return v; }
    const std::int32_t r = (st.fills - 1) % 2;
    v.gu_codes    = reinterpret_cast<const std::byte*>(st.ring[r].gu_codes);
    v.gu_scales   = reinterpret_cast<const std::byte*>(st.ring[r].gu_scales);
    v.dn_codes    = reinterpret_cast<const std::byte*>(st.ring[r].dn_codes);
    v.dn_scales   = reinterpret_cast<const std::byte*>(st.ring[r].dn_scales);
    v.gu_divisors = st.ring[r].gu_divisors;
    v.dn_divisors = st.ring[r].dn_divisors;
    v.experts     = kExpertsPerBank;
    return v;
}

// ================================================================================================
// Block-8 alignment item 1b -- the expert NAME-LIST preload (① 名单预载; supervisor G1 批准 2026-10-16).
//
// WHY: Strata fills its VRAM tier from an offline routing profile once, at startup, and then never
// evicts (`generate.cpp:4486-4545`; `expert_cache.hpp:40-55` -- "the profile is the policy"). Our
// pool keeps its online LFRU/HOT adaptation, so the aligned form here is: on a layer's FIRST routed
// call, claim that layer's pool slots for the profile's hottest experts and copy their records for
// real. For real matters: `global_refresh_hot` only ranks RESIDENT slots (:1267-1300), so a seed
// that moves no bytes protects nothing -- which is the constraint the 9/27 handoff wrote as
// "必须真搬进 + 低初值". The frequency plane is the pooled form's live counter (the per-layer
// `slot_uses` belongs to the non-pooled tables), so the low initial value goes there.
//
// FILE FORMAT -- read off Strata's own reader/writer (`expert_cache.cpp:72-137` / `:170-208`),
// little-endian (every machine this runs on):
//   "STRP", u32 version(=1), u32 n_layers, u32 n_expert, u32 slots_built_for, u32 n_ranked   (24 B)
//   n_ranked x (u16 layer, u16 expert)          -- GLOBAL, DESCENDING ROUTING FREQUENCY, so the
//                                                 layers INTERLEAVE: bucket by layer and take the
//                                                 first K of each. Cutting K*n_layers pairs off the
//                                                 front would hand most slots to the first layers.
//   n_layers*n_expert x i32 rank table          -- the profile's own working; Strata does not read
//                                                 it, we still require the exact total length so a
//                                                 half-written artifact cannot pass silently.
// The file carries a RANKING only, no counts, so nothing here measures a hit rate: coverage has to
// be predicted (see the ledger piece), and the self-证 lines below report only what was claimed.
//
// DEFAULT OFF: the switch IS the path. Unset/empty => one static read, no file, no allocation, no
// copy, zero new stderr lines (F-76 D-1). The ring consume path never reads these key planes, so
// the ring arm stays byte-identical as well. Everything below lives behind `expert_profile_armed()`.
// ================================================================================================

struct ExpertProfileConfig {
    const char*  path = nullptr;
    std::int32_t k    = 0;   // 0 = the whole pool width (P1: 装满)
};

const ExpertProfileConfig& expert_profile_config() {
    static const ExpertProfileConfig cfg = [] {
        ExpertProfileConfig c;
        if (const char* p = std::getenv("NINFER_FLASH_NEXT_EXPERTS_PROFILE"); p != nullptr && p[0] != '\0') {
            c.path = p;
            const std::int32_t want = read_int_once("NINFER_FLASH_NEXT_EXPERTS_PROFILE_K", 0);
            c.k = (want > 0) ? want : 0;   // negative/absent = default (whole width)
        }
        return c;
    }();
    return cfg;
}

bool expert_profile_armed() { return expert_profile_config().path != nullptr; }

// Seed band for the frequency plane: monotone by rank so HOT election protects the list's hottest,
// not its lowest-numbered ids (ties evict/pick the LOWER key -- see `global_refresh_hot`). The band
// is deliberately low: two real routed uses (+1.0 each) already outrank every seed, so online
// learning takes over within a round or two instead of freezing on a possibly load-shifted list.
constexpr double kProfileSeedHi = 8.0;
constexpr double kProfileSeedLo = 1.0;

struct ExpertProfilePreload {
    bool   parsed = false;
    std::int32_t version = 0, layers = 0, experts = 0, slots_field = 0, n_ranked = 0;
    std::uint64_t file_bytes = 0, fnv64 = 0, dup_pairs = 0;
    std::vector<std::int32_t> ids;                                  // layer * kExpertsPerBank + rank
    std::array<std::int32_t, kGlobalPoolLayers> entries{};          // ranked ids available per layer
    std::array<bool, kGlobalPoolLayers> claimed{};
    std::int32_t layers_done = 0, covered = 0, min_entries = 0;
    std::uint64_t records = 0, bytes = 0, occupied = 0, resident = 0, skipped_warming = 0;
    std::uint64_t verify_windows = 0;
    double ms_total = 0.0;
    std::unique_ptr<PinnedArray<float>> divisors;                   // 2 * kExpertsPerBank, pinned
};

ExpertProfilePreload& expert_profile_state() {
    static ExpertProfilePreload state;
    return state;
}

[[noreturn]] void expert_profile_refuse(const std::string& why) {
    throw std::runtime_error("Flash-Next expert profile preload refused: " + why + " (file=" +
                             expert_profile_config().path +
                             "). The switch is on, so ignoring this would be a silent no-op -- "
                             "the false green this tree keeps re-learning -- instead of an answer.");
}

// FNV-1a/64 over the file bytes: an artifact identity that is recomputable from the shipped bytes
// without pulling in the qwen3_5 frontend's sha256 (cross-model include is not this piece's call).
std::uint64_t fnv64_of(const std::vector<std::uint8_t>& data) {
    std::uint64_t h = 1469598103934665603ULL;
    for (const std::uint8_t b : data) {
        h ^= static_cast<std::uint64_t>(b);
        h *= 1099511628211ULL;
    }
    return h;
}

void expert_profile_parse(ExpertProfilePreload& st, int bank_experts) {
    const char* path = expert_profile_config().path;
    std::FILE*  f    = std::fopen(path, "rb");
    if (f == nullptr) { expert_profile_refuse("cannot open"); }
    std::fseek(f, 0, SEEK_END);
    const long long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 24) {
        std::fclose(f);
        expert_profile_refuse("file is shorter than a header");
    }
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(size));
    const std::size_t got = std::fread(raw.data(), 1, raw.size(), f);
    std::fclose(f);
    if (got != raw.size()) { expert_profile_refuse("short read"); }
    st.file_bytes = static_cast<std::uint64_t>(got);
    st.fnv64      = fnv64_of(raw);

    if (std::memcmp(raw.data(), "STRP", 4) != 0) { expert_profile_refuse("magic is not STRP"); }
    auto u32_at = [&](std::size_t at) {
        std::uint32_t v = 0;
        std::memcpy(&v, raw.data() + at, sizeof(v));   // little-endian by format
        return static_cast<std::int64_t>(v);
    };
    st.version     = static_cast<std::int32_t>(u32_at(4));
    st.layers      = static_cast<std::int32_t>(u32_at(8));
    st.experts     = static_cast<std::int32_t>(u32_at(12));
    st.slots_field = static_cast<std::int32_t>(u32_at(16));
    st.n_ranked    = static_cast<std::int32_t>(u32_at(20));
    if (st.version != 1) {
        expert_profile_refuse("version " + std::to_string(st.version) + " (this engine reads v1)");
    }
    if (st.layers != kGlobalPoolLayers) {
        expert_profile_refuse("the profile ranks " + std::to_string(st.layers) +
                              " layers, this engine's pool has " +
                              std::to_string(kGlobalPoolLayers) + " routed layer tables");
    }
    if (st.experts != kExpertsPerBank || st.experts != bank_experts) {
        expert_profile_refuse("the profile's expert count is " + std::to_string(st.experts) +
                              ", the live bank is " + std::to_string(bank_experts) +
                              " and the pool geometry is " + std::to_string(kExpertsPerBank));
    }
    if (st.n_ranked < 0 ||
        st.n_ranked > st.layers * st.experts) {
        expert_profile_refuse("n_ranked " + std::to_string(st.n_ranked) + " is out of range");
    }
    const long long expect = 24LL + 4LL * st.n_ranked +
                             4LL * static_cast<long long>(st.layers) * st.experts;
    if (static_cast<long long>(got) != expect) {
        expert_profile_refuse("length is " + std::to_string(got) + " B, the format requires " +
                              std::to_string(expect) + " B (24 + 4*ranked + 4*layers*experts)");
    }

    st.ids.assign(static_cast<std::size_t>(kGlobalPoolLayers) * kExpertsPerBank, -1);
    std::vector<std::uint64_t> seen(static_cast<std::size_t>(kGlobalPoolLayers) * 8, 0); // 512 bits
    const std::uint8_t* pairs = raw.data() + 24;
    for (std::int32_t i = 0; i < st.n_ranked; ++i) {
        std::uint16_t l = 0, e = 0;
        std::memcpy(&l, pairs + static_cast<std::size_t>(i) * 4, sizeof(l));
        std::memcpy(&e, pairs + static_cast<std::size_t>(i) * 4 + 2, sizeof(e));
        if (l >= st.layers || e >= st.experts) {
            expert_profile_refuse("pair " + std::to_string(i) + " is (layer " + std::to_string(l) +
                                  ", expert " + std::to_string(e) + "), out of range");
        }
        std::uint64_t& word = seen[static_cast<std::size_t>(l) * 8 + (e >> 6)];
        const std::uint64_t bit = 1ULL << (e & 63);
        if ((word & bit) != 0) { ++st.dup_pairs; continue; }   // keep the first (hottest) rank
        word |= bit;
        if (st.entries[l] < kExpertsPerBank) {
            st.ids[static_cast<std::size_t>(l) * kExpertsPerBank +
                   static_cast<std::size_t>(st.entries[l])] = static_cast<std::int32_t>(e);
            ++st.entries[l];
        }
    }
    st.min_entries = kExpertsPerBank;
    for (std::int32_t l = 0; l < kGlobalPoolLayers; ++l) {
        if (st.entries[l] > 0) { ++st.covered; }
        st.min_entries = std::min(st.min_entries, st.entries[l]);
    }
    if (st.covered == 0) { expert_profile_refuse("no layer has a single ranked pair"); }
    st.parsed = true;

    std::fprintf(stderr,
                 "flash_next_experts_profile file=%s bytes=%llu version=%d layers=%d experts=%d "
                 "slots_field=%d ranked=%d dup_pairs=%llu covered=%d/%d min_layer_entries=%d "
                 "fnv64=%016llx seed=[%.2f,%.2f]\n",
                 path, static_cast<unsigned long long>(st.file_bytes), st.version, st.layers,
                 st.experts, st.slots_field, st.n_ranked,
                 static_cast<unsigned long long>(st.dup_pairs), st.covered, kGlobalPoolLayers,
                 st.min_entries, static_cast<unsigned long long>(st.fnv64),
                 kProfileSeedLo, kProfileSeedHi);
}

// One layer, once per process: claim [table_index*slots, table_index*slots + K) for the profile's
// hottest K of that layer, seed the frequency plane by rank, and copy the records on the ENGINE
// stream. The engine stream is the ordering proof -- this same call may map a routed path onto a
// slot that was just claimed, and a stream-ordered copy needs no event and adds no synchronisation
// (a separate copy stream would have to be waited, which is how the ring's guard came to exist).
void expert_profile_preload_maybe_fill(SlotPool& p, bool pooled, std::int32_t table_index,
                                       const Nvfp4ExpertBankView& gate_up,
                                       const Nvfp4ExpertBankView& down, cudaStream_t engine_stream) {
    if (!expert_profile_armed()) { return; }
    if (table_index < 0) { return; }                       // no layer identity yet
    if (table_index >= kGlobalPoolLayers) { return; }      // the MTP table: not pooled, not ranked
    if (!pooled) {
        expert_profile_refuse(
            "the profile writes the global pool's key planes; with the per-layer tables the same "
            "residency is a different data structure. Set "
            "NINFER_FLASH_NEXT_EXPERTS_GLOBAL_POOL=1 or unset NINFER_FLASH_NEXT_EXPERTS_PROFILE");
    }
    ExpertProfilePreload& st = expert_profile_state();
    if (!st.parsed) { expert_profile_parse(st, gate_up.experts); }
    if (st.claimed[table_index]) { return; }
    st.claimed[table_index] = true;

    const std::int32_t slots   = p.slots;
    const std::int32_t want_k  = expert_profile_config().k;
    std::int32_t k = (want_k > 0) ? std::min(want_k, slots) : slots;
    k = std::min(k, st.entries[table_index]);               // a partially-ranked layer claims less
    const std::size_t base = static_cast<std::size_t>(table_index) * static_cast<std::size_t>(slots);

    const double t0 = host_now_ms();
    std::int32_t done = 0;
    std::size_t first_slot = 0, last_slot = 0;
    std::int32_t first_expert = -1, last_expert = -1;
    if (st.divisors == nullptr) {
        st.divisors = std::make_unique<PinnedArray<float>>(2 * static_cast<std::size_t>(kExpertsPerBank));
    }
    float* const div_gu = st.divisors->data();
    float* const div_dn = div_gu + kExpertsPerBank;

    for (std::int32_t r = 0; r < k; ++r) {
        const std::int32_t expert = st.ids[static_cast<std::size_t>(table_index) * kExpertsPerBank +
                                           static_cast<std::size_t>(r)];
        const std::size_t s   = base + static_cast<std::size_t>(r);
        const std::int32_t key = table_index * kExpertsPerBank + expert;
        if (p.slot_of_key[static_cast<std::size_t>(key)] >= 0) {
            // The key is already resident somewhere else (a real routing hit filled it first):
            // keep the online answer, leave this slot alone, and COUNT the gap. Its divisor stays
            // untouched because only claimed records are copied below.
            ++st.resident;
            continue;
        }
        if (p.key_of_slot[s] >= 0) {
            // Some OTHER layer's key holds this slot -- the flat pool hands out any free slot. The
            // preload must not evict a live entry, so the record is dropped and COUNTED, never
            // silent. (Re-stating the occupant's divisor is not possible here: the divisors belong
            // to the occupant's OWN layer bank, and this call only holds the current layer's.)
            ++st.occupied;
            continue;
        }
        // Exactly what a real fill writes (see the pooled branch of the mapping loop), plus the
        // rank-ordered seed. `hot` stays 0: the next `global_refresh_hot` elects from the seeds.
        p.key_of_slot[s]   = key;
        p.slot_of_key[key] = static_cast<std::int32_t>(s);
        p.g_slot_age[s]    = p.tick;
        p.g_slot_pass[s]   = p.pass;
        p.g_slot_hot[s]    = 0;
        p.evict_freq[static_cast<std::size_t>(key)] +=
            (k > 1) ? (kProfileSeedHi - (kProfileSeedHi - kProfileSeedLo) *
                        static_cast<double>(r) / static_cast<double>(k - 1))
                    : kProfileSeedHi;

        CUDA_CHECK(cudaMemcpyAsync(p.gate_up_codes + s * kStagedGateUpCodeBytes,
                                   gate_up.codes + static_cast<std::size_t>(expert) *
                                       gate_up.code_bytes_per_expert,
                                   kStagedGateUpCodeBytes, cudaMemcpyHostToDevice, engine_stream));
        CUDA_CHECK(cudaMemcpyAsync(p.gate_up_scales + s * kStagedGateUpScaleBytes,
                                   gate_up.scales + static_cast<std::size_t>(expert) *
                                       gate_up.scale_bytes_per_expert,
                                   kStagedGateUpScaleBytes, cudaMemcpyHostToDevice, engine_stream));
        CUDA_CHECK(cudaMemcpyAsync(p.down_codes + s * kStagedDownCodeBytes,
                                   down.codes + static_cast<std::size_t>(expert) *
                                       down.code_bytes_per_expert,
                                   kStagedDownCodeBytes, cudaMemcpyHostToDevice, engine_stream));
        CUDA_CHECK(cudaMemcpyAsync(p.down_scales + s * kStagedDownScaleBytes,
                                   down.scales + static_cast<std::size_t>(expert) *
                                       down.scale_bytes_per_expert,
                                   kStagedDownScaleBytes, cudaMemcpyHostToDevice, engine_stream));
        // The two 4 B copies come from PINNED memory on purpose (the note above the emitter says why:
        // a pageable 4 B copy inside a loop drains the DMA queue the plane copies just filled). The
        // temp is reused across layers, which is safe only because every staging call starts with the
        // one per-layer host round trip (`cudaStreamSynchronize`, see the S5 note in the staging body)
        // -- so a later layer cannot write an address an earlier layer's copy has not read yet. If
        // that per-layer sync is ever removed (split-window work), this must move to a per-layer
        // plane before it can be trusted again.
        div_gu[r] = gate_up.weight_scale_divisors[static_cast<std::size_t>(expert)];
        div_dn[r] = down.weight_scale_divisors[static_cast<std::size_t>(expert)];
        CUDA_CHECK(cudaMemcpyAsync(p.gate_up_divisors + s, div_gu + r, sizeof(float),
                                   cudaMemcpyHostToDevice, engine_stream));
        CUDA_CHECK(cudaMemcpyAsync(p.down_divisors + s, div_dn + r, sizeof(float),
                                   cudaMemcpyHostToDevice, engine_stream));
        st.bytes += kStagedGateUpCodeBytes + kStagedGateUpScaleBytes + kStagedDownCodeBytes +
                    kStagedDownScaleBytes;
        if (done == 0) { first_slot = s; first_expert = expert; }
        last_slot = s; last_expert = expert;
        ++done;
        ++st.records;
    }

    st.ms_total += host_now_ms() - t0;
    if (done == 0 && k > 0) { ++st.skipped_warming; }       // the layer was fully displaced already

    // **AND ONE RECORD IS READ BACK AND COMPARED** (Strata's own discipline, generate.cpp:4536-4541).
    // Once per process, on layer 0, at both ends of the claimed range: an index table that is right
    // about ids and wrong about BYTES produces a plausible token, and the pooled path has no byte
    // verification at all (VERIFY + pooled is refused above). Skipped while the sabotage switch is
    // armed, which stages a neighbour expert on purpose.
    if (table_index == 0 && done > 0 && !staging_sabotage_enabled()) {
        CUDA_CHECK(cudaStreamSynchronize(engine_stream));
        bool ok = true;
        std::uint8_t got[64];
        auto check = [&](std::uint8_t* dev, const std::byte* host, std::size_t rec_bytes,
                         std::size_t slot, std::int32_t expert) {
            const std::size_t offsets[3] = {0, rec_bytes / 2, rec_bytes - sizeof(got)};
            for (const std::size_t o : offsets) {
                CUDA_CHECK(cudaMemcpy(got, dev + slot * rec_bytes + o, sizeof(got),
                                      cudaMemcpyDeviceToHost));
                if (std::memcmp(got, host + static_cast<std::size_t>(expert) * rec_bytes + o,
                                 sizeof(got)) != 0) {
                    ok = false;
                    std::fprintf(stderr,
                                 "flash_next_experts_profile_verify MISMATCH slot=%zu expert=%d "
                                 "offset=%zu\n", slot, expert, o);
                }
                ++st.verify_windows;
            }
        };
        check(p.gate_up_codes, gate_up.codes, kStagedGateUpCodeBytes, first_slot, first_expert);
        check(p.down_codes, down.codes, kStagedDownCodeBytes, first_slot, first_expert);
        check(p.gate_up_codes, gate_up.codes, kStagedGateUpCodeBytes, last_slot, last_expert);
        check(p.down_codes, down.codes, kStagedDownCodeBytes, last_slot, last_expert);
        if (!ok) {
            expert_profile_refuse("the read-back of a preloaded record does not match its source "
                                  "bytes: the slot table and the arena disagree, which is the one "
                                  "failure a plausible token would hide");
        }
    }

    std::fprintf(stderr,
                 "flash_next_experts_profile_fill layer=%d records=%d seeded=%d bytes=%llu ms=%.1f "
                 "k=%d occupied=%llu resident=%llu skipped_warming=%d pool_slots=%d\n",
                 table_index, done, done, static_cast<unsigned long long>(st.bytes),
                 host_now_ms() - t0, k, static_cast<unsigned long long>(st.occupied),
                 static_cast<unsigned long long>(st.resident),
                 (done == 0 && k > 0) ? 1 : 0, slots);

    ++st.layers_done;
    if (st.layers_done == kGlobalPoolLayers) {
        std::fprintf(stderr,
                     "flash_next_experts_profile_done layers=%d records=%llu bytes_moved=%.2f_gib "
                     "ms_total=%.1f verify=%d occupied=%llu resident=%llu skipped_warming=%llu\n",
                     st.layers_done, static_cast<unsigned long long>(st.records),
                     static_cast<double>(st.bytes) / 1073741824.0, st.ms_total,
                     st.verify_windows > 0 ? 1 : 0,
                     static_cast<unsigned long long>(st.occupied),
                     static_cast<unsigned long long>(st.resident),
                     static_cast<unsigned long long>(st.skipped_warming));
    }
}

MoeWeights stage_flash_next_moe_experts(const MoeWeights& weights,
                                        const FlashNextMoeWorkspace& scratch, int tokens,
                                        cudaStream_t stream, const Tensor* serve_input) {
    const Nvfp4ExpertBankView& gate_up = weights.expert_gate_up;
    const Nvfp4ExpertBankView& down    = weights.expert_down;

    if (!gate_up.on_host || !down.on_host) {
        throw std::invalid_argument("Flash-Next expert staging requires host-tiered expert banks");
    }
    if (tokens < 1 || tokens > kMaxStagedTokens) {
        throw std::invalid_argument("Flash-Next expert staging requires 1 <= tokens <= 32");
    }
    require_matching_geometry(gate_up, kStagedGateUpCodeBytes, kStagedGateUpScaleBytes, "gate_up");
    require_matching_geometry(down, kStagedDownCodeBytes, kStagedDownScaleBytes, "down");
    if (scratch.ids.ne[0] != 10) {
        throw std::logic_error("Flash-Next expert staging expects ten routed paths per token");
    }

    // Capture guard: staging does host round trips and pageable host->device copies, neither of which
    // is legal inside a stream capture (cudaStreamSynchronize and pageable cudaMemcpyAsync both fail
    // during capture). The flash_next decode round is captured by FlashNextTextExecutor, so say so
    // explicitly instead of surfacing a bare CUDA error from inside the capture body.
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        throw std::runtime_error(
            "Flash-Next host-tiered expert staging cannot run inside a CUDA graph capture: run with "
            "--no-cuda-graph, or keep the expert banks resident on the device");
    }

    const double t_enter_ms = host_now_ms();
    const std::int32_t paths = scratch.ids.ne[0] * tokens;
    if (paths > kMaxStagedPaths) {
        throw std::runtime_error("Flash-Next expert staging: one call asked for more routed paths "
                                 "than the staging bookkeeping holds");
    }

    // ---- the R1 gate, hoisted ABOVE the two paths that both need it ------------------------------
    // The gate splits the two phases, and each phase has its OWN threshold: a wide call (prefill
    // slice) uses MOE_DROP_F_PREFILL, which defaults to 0 = not armed. 97.3% of all replacements
    // happen in that phase, because a fresh prompt arrives with a cold cache.
    // The gate means "the WIDEST call that uses the DECODE threshold". Anything wider uses the
    // prefill threshold (default 0 = not armed). This split is deliberate and was arrived at the hard
    // way: our prefill arrives as 8-token slices, so a naive "tokens <= gate ? decode : prefill"
    // makes gate=8 put those slices in the DECODE branch and MOE_DROP_F_PREFILL is never called at
    // all -- a silent no-op that three bit-identical counter sets exposed (dropped=118381 for
    // fpre absent / 0.50 / 0.25 alike). Calls wider than 8 are not expected and stay unarmed.
    const std::int32_t gate = moe_drop_max_tokens();
    const float drop_f      = (tokens <= gate) ? moe_drop_fraction()
                              : (tokens <= 8)  ? moe_drop_fraction_prefill()
                                               : 0.0F;
    const bool drop_armed   = drop_f > 0.0F;
    if (drop_armed && moe_subst_enabled()) {
        throw std::runtime_error(
            "Flash-Next device staging does not implement MOE_SUBST. Running it would silently "
            "behave like pure dropping and diverge from the host path. Turn MOE_SUBST off, or run "
            "the host path (NINFER_FLASH_NEXT_EXPERTS_DEVICE_STAGE=0).");
    }

    // ---- the device path is DISABLED (D1, decided 2026-10-14) ------------------------------------
    // This path exists to remove the host readback of the router ids, and it measured 0 gain over the
    // host path (31.52 vs 33.05 tok/s, R1 armed -- see the note it used to carry below). Its victim
    // selection is also a second, NON-EQUIVALENT implementation of Least-Stale (three passes, no hot
    // protection) against the host path's four passes with hot protection, so the two cannot be
    // compared on one counter. Moving every decision to the host (the next step) leaves the device
    // with only a residency lookup, which is what the reference does too.
    //
    // Refused LOUDLY rather than ignored: a silently ignored switch makes "I set it and nothing
    // changed" indistinguishable from a real measurement -- the false green this tree keeps
    // re-learning (see the note about `dropped=0` in expert_stage_kernels.cu:296-299).
    if (device_stage_enabled()) {
        throw std::runtime_error(
            "Flash-Next device-side expert staging is disabled: all cache decisions now run on the "
            "host, so the device no longer needs a slot table. Unset "
            "NINFER_FLASH_NEXT_EXPERTS_DEVICE_STAGE (it measured zero gain over the host path "
            "anyway: 31.52 vs 33.05 tok/s).");
    }

    // ---- device-side path: decide + copy on the device, no readback, no per-layer sync ----------
    if (const std::int32_t resolved = flash_next_experts_slot_count();
        resolved > 0 && device_stage_enabled()) {
        // The wide arm is refused HERE, before anything is launched, and for a structural reason: this
        // path exists to avoid the host readback of the router ids, and that readback is exactly what
        // lets the host path prove a call's DISTINCT expert count fits the pool (see the check below).
        // Without it a wave could make the device decide kernel re-use a slot that earlier paths of the
        // same call still point at -- silent corruption, not a crash. This path was also measured at 0
        // gain over the host path (31.52 vs 33.05 tok/s, R1 armed), so it is not worth loosening.
        if (tokens > kFlashNextRouteDecodeMaxTokens) {
            throw std::runtime_error(
                "Flash-Next device staging does not support the wide MoE arm (tokens > 8): without the "
                "host readback it cannot verify that the wave fits the slot pool. Run the host path "
                "(NINFER_FLASH_NEXT_EXPERTS_DEVICE_STAGE=0) or turn off NINFER_FLASH_NEXT_MOE_WIDE_ARM.");
        }
        SlotPool& p           = slot_pool();
        ensure_slot_pool(p, resolved);
        const auto index      = static_cast<std::size_t>(layer_table_for(p, gate_up.codes));
        const auto slot_count = static_cast<std::size_t>(p.slots);
        auto* device_gate_up_codes = p.gate_up_codes + index * slot_count * kStagedGateUpCodeBytes;
        auto* device_gate_up_scales =
            p.gate_up_scales + index * slot_count * kStagedGateUpScaleBytes;
        auto* device_down_codes = p.down_codes + index * slot_count * kStagedDownCodeBytes;
        auto* device_down_scales = p.down_scales + index * slot_count * kStagedDownScaleBytes;
        auto* device_gate_up_divisors = p.gate_up_divisors + index * slot_count;
        auto* device_down_divisors    = p.down_divisors + index * slot_count;

        flash_next_expert_device_stage_launch(
            gate_up.codes, static_cast<std::int32_t*>(scratch.ids.data), paths, p.slots,
            static_cast<float*>(scratch.alpha.data), drop_armed ? (drop_f / 10.0F) : 0.0F,
            reinterpret_cast<const std::uint8_t*>(gate_up.codes), gate_up.code_bytes_per_expert,
            reinterpret_cast<const std::uint8_t*>(gate_up.scales), gate_up.scale_bytes_per_expert,
            reinterpret_cast<const std::uint8_t*>(down.codes), down.code_bytes_per_expert,
            reinterpret_cast<const std::uint8_t*>(down.scales), down.scale_bytes_per_expert,
            gate_up.weight_scale_divisors, down.weight_scale_divisors, device_gate_up_codes,
            device_gate_up_scales, device_down_codes, device_down_scales, device_gate_up_divisors,
            device_down_divisors, static_cast<std::int32_t>(p.calls / 48), stream);

        ++p.calls;
        if (p.calls % 48 == 0) {
            // Pinned mirror written without a sync; values lag the device by one call.
            const unsigned long long* counters = flash_next_expert_device_stage_counters();
            const double total = static_cast<double>(counters[0] + counters[1]);
            std::fprintf(stderr,
                         "flash_next_experts_cache device calls=%llu hits=%llu misses=%llu "
                         "evictions=%llu hit_rate=%.1f%% dropped=%llu F=%.2f gate<=%d fpre=%.2f "
                         "slots=%d hot=%d\n",
                         static_cast<unsigned long long>(p.calls), counters[0], counters[1],
                         counters[2], total > 0.0 ? 100.0 * static_cast<double>(counters[0]) / total : 0.0,
                         counters[3], static_cast<double>(moe_drop_fraction()), moe_drop_max_tokens(),
                         static_cast<double>(moe_drop_fraction_prefill()), p.slots,
                         hot_slot_count(p.slots));
        }

        MoeWeights staged     = weights;
        staged.expert_gate_up = Nvfp4ExpertBankView{
            .codes                  = reinterpret_cast<const std::byte*>(device_gate_up_codes),
            .scales                 = reinterpret_cast<const std::byte*>(device_gate_up_scales),
            .weight_scale_divisors  = device_gate_up_divisors,
            .experts                = p.slots,
            .rows                   = gate_up.rows,
            .columns                = gate_up.columns,
            .code_bytes_per_expert  = gate_up.code_bytes_per_expert,
            .scale_bytes_per_expert = gate_up.scale_bytes_per_expert,
            .on_host                = false,
        };
        staged.expert_down = Nvfp4ExpertBankView{
            .codes                  = reinterpret_cast<const std::byte*>(device_down_codes),
            .scales                 = reinterpret_cast<const std::byte*>(device_down_scales),
            .weight_scale_divisors  = device_down_divisors,
            .experts                = p.slots,
            .rows                   = down.rows,
            .columns                = down.columns,
            .code_bytes_per_expert  = down.code_bytes_per_expert,
            .scale_bytes_per_expert = down.scale_bytes_per_expert,
            .on_host                = false,
        };
        return staged;
    }

    // Router output -> host. One host round trip per layer; it is what makes the copy list knowable,
    // and it is the first thing to remove in a later stage (device-side slot table).
    PinnedArray<std::int32_t>& host_ids = staging_control_buffers().ids;
    std::memset(host_ids.data(), 0, static_cast<std::size_t>(paths) * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpyAsync(host_ids.data(), scratch.ids.data,
                               static_cast<std::size_t>(paths) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream));
    stage_profile().note_d2h(static_cast<std::size_t>(paths) * sizeof(std::int32_t));

    // R1 (miss-gated dropping) needs each path's routing weight beside its id. 40 bytes, and it
    // rides the SAME synchronize the ids already pay for, so there is no extra round trip.
    PinnedArray<float>& host_alpha = staging_control_buffers().alpha;
    // The buffers are reused across calls now, so keep the old `{}` zero-initialisation exactly: the
    // only paths that read alpha when dropping is disarmed are the ones that must see zero.
    std::memset(host_alpha.data(), 0, static_cast<std::size_t>(paths) * sizeof(float));
    // The gate splits the two phases, and each phase has its OWN threshold: a wide call (prefill
    // slice) uses MOE_DROP_F_PREFILL, which defaults to 0 = not armed. 97.3% of all replacements
    // happen in that phase, because a fresh prompt arrives with a cold cache.
    // The gate means "the WIDEST call that uses the DECODE threshold". Anything wider uses the
    // prefill threshold (default 0 = not armed). This split is deliberate and was arrived at the hard
    // way: our prefill arrives as 8-token slices, so a naive "tokens <= gate ? decode : prefill"
    // makes gate=8 put those slices in the DECODE branch and MOE_DROP_F_PREFILL is never called at
    // all -- a silent no-op that three bit-identical counter sets exposed (dropped=118381 for
    // fpre absent / 0.50 / 0.25 alike). Calls wider than 8 are not expected and stay unarmed.
    // gate / drop_f / drop_armed are computed once, above, because the device path needs them too.
    (void)gate;
    (void)drop_armed;
    // Block-6 W1: the CPU serve forward face needs (a) the REAL routing weights on the host and
    // (b) the MoE input mirrored to the host. Both ride the same per-layer synchronise the ids
    // already pay for. The face arms only when the whole chain can carry it: master switch on,
    // decode width, this target's geometry (the ported forward's invariants), and an input tensor.
    // Anything wider keeps the landing-only round -- additively, the block-3 behaviour untouched.
    const bool serve_forward =
        cpu_serve_share().enabled && serve_input != nullptr &&
        (tokens == 1 || cpu_serve_prefill_enabled()) &&
        tokens <= cpu_expert::kCpuServeForwardCols && gate_up.columns == 2'560 &&
        down.columns == 640 && gate_up.rows == 1'280 && down.rows == 2'560;
    if (drop_armed || serve_forward) {
        CUDA_CHECK(cudaMemcpyAsync(host_alpha.data(), scratch.alpha.data,
                                   static_cast<std::size_t>(paths) * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
        stage_profile().note_d2h(static_cast<std::size_t>(paths) * sizeof(float));
    }
    if (serve_forward) {
        // One x D2H per call, into the single staging area -- the ring slot is not known yet (the
        // channel is taken lazily on the first serve candidate below), and a slot slab must not be
        // written while an in-flight round on that slot is still un-answered. The slab copy and the
        // partial zero happen in the fill block once the slot is known.
        auto& ctl               = staging_control_buffers();
        const std::size_t x_elems = static_cast<std::size_t>(2'560) * static_cast<std::size_t>(tokens);
        CUDA_CHECK(cudaMemcpyAsync(ctl.serve_x_stage.data(), serve_input->data,
                                   x_elems * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, stream));
        stage_profile().note_d2h(x_elems * sizeof(std::uint16_t));
    }
    {
        // The one per-layer host round trip. Its host-blocked time is half the S5 ledger: if this is
        // small, the staging cost is DMA; if it is large, S1/S2 (kill the round trip) is the lever.
        const double t0 = host_now_ms();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto& prof = stage_profile();
        prof.sync_ms += host_now_ms() - t0;
        ++prof.syncs;
    }

    const int sabotage       = staging_sabotage_enabled() ? 1 : 0;
    const std::int32_t slots = flash_next_experts_slot_count();
    if (slots == 0 && expert_profile_armed()) {
        // Anti-false-green: with no slot pool the name list has nowhere to live, and a run that set
        // the profile and quietly kept zero slots would read exactly like one that preloaded.
        throw std::runtime_error(
            "Flash-Next expert profile preload needs a slot pool: unset "
            "NINFER_FLASH_NEXT_EXPERTS_PROFILE or set NINFER_FLASH_NEXT_EXPERTS_SLOTS (>0).");
    }

    // ---- does this call FIT its destination? ------------------------------------------------------
    // The bookkeeping above is sized by PATHS; the destination is sized by DISTINCT EXPERTS, and for a
    // wide call the two differ enormously: 32 tokens x top-10 is up to 320 paths against a pool of
    // 128-172 slots. Overrunning the pool is NOT an out-of-bounds write -- the LRU would evict and
    // re-fill a slot that EARLIER PATHS IN THE SAME CALL still point at, so those paths would silently
    // read another expert's weights. On the decode path the question cannot arise (paths <= 80 and
    // slots >= 128), so it arrived with the wide arm; it is answered here, loudly, rather than left to
    // the median of a routing trace. host_ids is already on the host at this point, so the count is a
    // 512-bit pass over data we have already paid to transfer.
    std::int32_t distinct = 0;
    {
        std::uint64_t seen[8] = {}; // one bit per expert id, 512 experts
        for (std::int32_t i = 0; i < paths; ++i) {
            const std::int32_t expert = host_ids[static_cast<std::size_t>(i)];
            if (expert < 0 || expert >= gate_up.experts) { continue; } // reported in the mapping loop
            const std::uint64_t bit = 1ULL << (expert & 63);
            std::uint64_t& word     = seen[expert >> 6];
            if ((word & bit) == 0) {
                word |= bit;
                ++distinct;
            }
        }
    }
    stage_profile().note_distinct(distinct);

    PinnedArray<std::int32_t>& host_slots = staging_control_buffers().slots; // per routed path -> slot
    std::memset(host_slots.data(), 0, static_cast<std::size_t>(paths) * sizeof(std::int32_t));
    std::array<std::int32_t, kMaxStagedPaths> used_slots{}; // distinct slots touched, in order
    std::array<std::int32_t, 512> layer_slot_expert{};      // slot -> expert (this layer)
    std::vector<std::int32_t> pending_copy;                 // slots that need a copy (misses)
    int used_count = 0;

    // The no-cache layout (slots == 0) really is bounded by its planes, so it gets the tighter limit.
    // The CACHED case is checked further down, once `pooled` is known: with one pool the destination
    // holds slots * kGlobalPoolLayers slots, not slots, and checking the wrong bound here would refuse
    // legal wide calls (and, worse, would have to be changed in two places at once).
    if (slots == 0 && distinct > kStagedExpertSlots) {
        throw std::runtime_error(
            "Flash-Next expert staging: this call needs " + std::to_string(distinct) +
            " distinct experts but its destination holds only " +
            std::to_string(kStagedExpertSlots) + " (tokens=" + std::to_string(tokens) +
            ", paths=" + std::to_string(paths) +
            "). Lower the wave width (kMaxStagedTokens); refusing rather than re-using a slot that "
            "earlier paths in this same call already point at, which would silently corrupt the "
            "output.");
    }

    SlotPool* pool    = nullptr;
    LayerSlots* table = nullptr;
    int table_index   = -1;
    // `global_ready` says the pool EXISTS; `pooled` says whether THIS table is in it. The MTP layer
    // (the 49th table, when the run has one) is deliberately not pooled, and a per-layer slot number
    // used as a global one would index another layer's weights -- silent, and invisible in every
    // aggregate counter.
    bool pooled = false;

    // Block-8 alignment item 1 (CONSUME side, operator): function scope because the SHARED tail
    // below (plane selection, copy section, id rewrite, staged build) reads both. Set inside the
    // pool branch, next to the ring fill; stays false everywhere else -- the no-cache path never
    // reaches the fill, and every other combination falls out of the gate there.
    bool ring_engaged = false;
    PrefetchRingPlanes ring_planes{};

    if (slots > 0) {
        SlotPool& p = slot_pool();
        ensure_slot_pool(p, slots);
        table_index = layer_table_for(p, gate_up.codes);
        pooled      = p.global_ready && table_index < kGlobalPoolLayers;
        if (p.global_ready && !pooled) {
            // Known weak point, said out loud rather than hidden: `table_index` is assigned in
            // FIRST-TOUCH order, not by layer number, so "index >= 48" is only certainly the MTP
            // layer if the MTP layer is the last table touched. This line is the evidence for
            // deciding that: with MTP enabled the route analysis should report 49 layers, and this
            // notice should fire exactly once.
            static bool s_notice = false;
            if (!s_notice) {
                s_notice = true;
                std::fprintf(stderr,
                             "flash_next_experts_cache table=%d is NOT in the global pool "
                             "(per-layer path); kGlobalPoolLayers=%d\n",
                             table_index, kGlobalPoolLayers);
            }
        }

        // Capacity, checked where `pooled` is known: how many distinct experts this call may hold AT
        // ONCE. Overrunning it is not an out-of-bounds write -- the policy would evict and refill a
        // slot that earlier paths of the SAME call still point at, so those paths would silently read
        // another expert's weights. With one pool the destination is slots * kGlobalPoolLayers.
        const std::int32_t live_capacity = pooled ? slots * kGlobalPoolLayers : slots;
        if (distinct > live_capacity) {
            throw std::runtime_error(
                "Flash-Next expert staging: this call needs " + std::to_string(distinct) +
                " distinct experts but its destination holds only " +
                std::to_string(live_capacity) + " (tokens=" + std::to_string(tokens) +
                ", paths=" + std::to_string(paths) + ", pooled=" +
                std::to_string(pooled ? 1 : 0) +
                "). Lower the wave width (kMaxStagedTokens) or raise "
                "NINFER_FLASH_NEXT_EXPERTS_SLOTS; refusing rather than re-using a slot that earlier "
                "paths in this same call already point at, which would silently corrupt the output.");
        }
        table       = &p.layers[static_cast<std::size_t>(table_index)];
        pool        = &p;
        layer_slot_expert = table->slot_expert;

        // ---- P2a: close the previous round BEFORE writing into the log ---------------------------
        // Order matters and was wrong at first: this detection used to sit at the END of the call,
        // while the log is written at the START. A sweep's first call (table 0) therefore wrote its
        // record, and then the end-of-call roll cleared the ring it had just written into -- losing
        // exactly one layer per round (measured: logged_layers=47, not 48).
        // The wrap is the layer order going backwards, which needs no knowledge of how many layer
        // tables exist and cannot drift when that count changes. The round counter is maintained
        // unconditionally so the self-证 line is always available; `round_tracking_enabled()` gates
        // only the USE of the round, which keeps the OFF arm's decisions byte-identical.
        if (p.last_table_index >= 0 && table_index < p.last_table_index) {
            p.round += 1;
            p.route_log.head = static_cast<std::int32_t>((p.route_log.head + 1) % kRouteLogRounds);
            p.route_log.closed += 1;
            for (std::int32_t t = 0; t < kMaxLayerTables; ++t) {
                p.route_log.rounds[p.route_log.head][t].table_index = -1;
                p.route_log.rounds[p.route_log.head][t].count       = 0;
            }
            // P5-1 timeout. A round ended and no acceptance count ever arrived, so the deferred refs
            // are committed IN FULL (and counted as stale). Refusing to credit them would be a silent
            // freeze of the frequency plane -- and crediting them is the honest reading: those uses
            // really did happen. `pooled` is required because a non-pooled pool has no frequency
            // plane to write into.
            if (pooled && live_columns_enabled()) {
                commit_live_columns(p, /*accepted_count=*/-1, /*stale=*/true);
            }
        }
        p.last_table_index = table_index;

        // ---- Block-8 alignment item 1 (instrument side): the per-layer prefetch-ring fill -------
        // The first wave of a layer (a layer change) issues the whole-bank fill on the private copy
        // stream; see the block above the staging function. Inert unless the env is on.
        ring_prefill_maybe_fill(table_index, tokens, gate_up, down, stream);

        // ---- Block-8 alignment item 1b (名单预载): the same seam, one step later ------------------
        // Inert unless NINFER_FLASH_NEXT_EXPERTS_PROFILE names a file. Fires once per layer on its
        // FIRST call (prefill in the aligned shape, so the DMA rides the attention cover instead of
        // the decode critical path). Kept strictly after the ring fill: the ring writes its own
        // planes and never reads the key planes this writes, so the two cannot disagree, and the
        // order here is the order the approved design recorded.
        expert_profile_preload_maybe_fill(p, pooled, table_index, gate_up, down, stream);

        // ---- Block-8 alignment item 1 (CONSUME side, operator) -----------------------------------
        // tokens > 8 (a prefill wave) with the ring armed: this layer's whole bank is resident in
        // the ring the fill above just issued -- and the engine stream already waits on fill_done,
        // so the kernels this call returns read a complete ring. Expert id == ring slot, so the
        // whole per-miss copy / slot-pool / id-rewrite machinery below is bypassed for this call:
        // the decision loop is cut short (nothing there may touch the pool), nothing is copied,
        // the ids stay the router's raw ids, and the staged views at the bottom point at the ring
        // planes. COMPOSITION: prefill drop and prefill CPU-serve both need the machinery this path
        // skips, so the ring stays out of those combinations (each keeps the normal path, which
        // stays byte-identical; the fill above is still issued, which is benign).
        ring_planes  = prefetch_ring_planes();
        ring_engaged = prefetch_ring_engaged(tokens) && ring_planes.experts == kExpertsPerBank &&
                       ring_planes.gu_codes != nullptr && !drop_armed &&
                       !(cpu_serve_share().enabled && cpu_serve_prefill_enabled());
        if (ring_engaged) {
            static std::atomic<bool> s_consume_reported{false};
            bool expected = false;
            if (s_consume_reported.compare_exchange_strong(expected, true)) {
                std::fprintf(stderr,
                             "flash_next_experts_ring_consume layer=%d experts=%d tokens=%d\n",
                             table_index, kExpertsPerBank, tokens);
                std::fflush(stderr);
            }
        }

        // ---- P2a: record this call's routing, so a LATER round can decide from it -----------------
        // `host_ids` is already on the host here (that D2H is what makes the copy list knowable at
        // all), so the log costs a copy into a ring and no round trip. It records RAW expert ids:
        // the slot rewrite below happens to `host_slots`, never to `host_ids`.
        // Out-of-range ids are stored as 0xFFFF rather than dropped, so the record keeps its length
        // and a reader can see the anomaly instead of silently reading a shorter list.
        if (round_tracking_enabled()) {
            RouteLogRound& rec =
                p.route_log.rounds[p.route_log.head][static_cast<std::size_t>(table_index)];
            rec.table_index = table_index;
            rec.count       = paths;
            for (std::int32_t i = 0; i < paths; ++i) {
                const std::int32_t expert = host_ids[static_cast<std::size_t>(i)];
                rec.ids[static_cast<std::size_t>(i)] =
                    (expert >= 0 && expert < kExpertsPerBank)
                        ? static_cast<std::uint16_t>(expert)
                        : static_cast<std::uint16_t>(0xFFFF);
            }
        }

        ++p.tick;
        // LFRU's clock: one tick per routed layer call (48 per round), so this fires every
        // kLfruHalvingPeriod calls (~170 rounds) -- the same decay the reference measured.
        if (pooled && p.tick % static_cast<std::int32_t>(kLfruHalvingPeriod) == 0) {
            halve_evict_freq(p);
        }

        // ---- R1: miss-gated expert dropping (BigMoeOnEdge expert-dropping.md, F <= 1.0) ----------
        // Drop a routed path ONLY when BOTH hold:
        //   (a) its expert is NOT resident right now -> dropping it saves a whole PCIe copy, and
        //   (b) its renormalized weight is below F * (uniform share 1/k) -> it contributes little.
        // This is deliberately narrower than "lower top-k": a resident expert, however low its
        // weight, is FREE, so it is never dropped, and the argmax path is never dropped either.
        bool drop[kMaxStagedPaths];
        const double t_drop0 = host_now_ms();
        for (std::int32_t i = 0; i < paths; ++i) { drop[i] = false; }
        std::int32_t top_path = 0;
        if (drop_armed) {
            for (std::int32_t i = 1; i < paths; ++i) {
                if (host_alpha[static_cast<std::size_t>(i)] >
                    host_alpha[static_cast<std::size_t>(top_path)]) {
                    top_path = i;
                }
            }
            const float threshold = drop_f / 10.0F;
            for (std::int32_t i = 0; i < paths; ++i) {
                if (i == top_path) { continue; }
                const std::int32_t candidate = host_ids[static_cast<std::size_t>(i)];
                if (candidate < 0 || candidate >= gate_up.experts) { continue; } // raised below
                // "Resident right now" must be asked of the SAME table the lookup uses. With one pool
                // the per-layer table is never filled, so asking it would report every expert as
                // non-resident and arm the drop for paths that are actually free -- and guard (a) of
                // the drop is exactly "not resident right now".
                const bool resident_now =
                    pooled
                        ? (p.slot_of_key[static_cast<std::size_t>(table_index * kExpertsPerBank +
                                                                  candidate)] >= 0)
                        : (table->of_expert[static_cast<std::size_t>(candidate)] >= 0);
                if (resident_now) { continue; }
                if (host_alpha[static_cast<std::size_t>(i)] < threshold) { drop[i] = true; }
            }
        }

        stage_profile().drop_ms += host_now_ms() - t_drop0;

        // ---- Phase IV W3: per-call CPU-serve preparation (default off) ---------------------------
        // serve_active gates every branch below; when off, this block costs one static read and the
        // call is byte-identical to the shipped one. The rule (see cpu_expert_nvfp4.h, Phase IV):
        // of this call's miss sequence, every d-th miss counting from 0 takes the LINK (so the
        // first miss of a call always does -- "at least one link", which keeps a 1-miss layer from
        // being served 100%); the rest are served. LEGACY keeps the pre-patch floor quota instead:
        // link_quota = floor(distinct_misses / d), i.e. 0 links when misses < d.
        // Block-8 item 9: decode-only serving (see cpu_serve_prefill_enabled above). The gate is on
        // the call width, not a phase flag: decode arrives as 1-token calls (a future MTP window of
        // 2..8 is a follow-up widening, noted in the alignment map), prefill as slices of up to 8.
        const CpuServeShare& serve_share = cpu_serve_share();
        bool serve_active = serve_share.enabled && (tokens == 1 || cpu_serve_prefill_enabled());
        const bool serve_legacy         = serve_share.legacy;
        // Block-8 item 8: with adaptive on, d is re-derived per call from the live serve
        // throughput (seed/fallback = the startup derivation). Default OFF: exactly serve_share.d.
        const std::int32_t serve_d = serve_share.adaptive
                                         ? cpu_expert::adaptive_serve_d(serve_share.d, serve_share.t_us)
                                         : serve_share.d;
        cpu_expert::CpuServeRequest* serve_req = nullptr;
        std::int32_t serve_miss_ordinal = 0;
        std::int32_t serve_link_used    = 0;
        std::int32_t serve_link_quota   = 0;
        std::array<bool, kMaxStagedPaths> cpu_serve_path{};
        if (serve_active && serve_legacy) {
            std::array<std::uint32_t, kMaxStagedPaths> seen_keys{};
            std::int32_t seen_count      = 0;
            std::int32_t distinct_misses = 0;
            for (std::int32_t i = 0; i < paths; ++i) {
                if (drop[i]) { continue; }
                const std::int32_t expert = host_ids[static_cast<std::size_t>(i)];
                if (expert < 0 || expert >= gate_up.experts) { continue; }
                const std::int32_t key = table_index * kExpertsPerBank + expert;
                const std::int32_t slot =
                    pooled ? p.slot_of_key[static_cast<std::size_t>(key)]
                           : table->of_expert[static_cast<std::size_t>(expert)];
                if (slot >= 0) { continue; }
                bool duplicate = false;
                for (std::int32_t q = 0; q < seen_count; ++q) {
                    if (seen_keys[static_cast<std::size_t>(q)] ==
                        static_cast<std::uint32_t>(key)) {
                        duplicate = true;
                        break;
                    }
                }
                if (duplicate) { continue; }
                if (seen_count < kMaxStagedPaths) {
                    seen_keys[static_cast<std::size_t>(seen_count++)] =
                        static_cast<std::uint32_t>(key);
                }
                ++distinct_misses;
            }
            serve_link_quota = distinct_misses / serve_d; // floor: misses < d -> 0 links (old shape)
        }

        const double t_decide0 = host_now_ms();
        for (std::int32_t i = 0; i < paths; ++i) {
            if (ring_engaged) {
                break; // ring path: the bank is resident; the loop must not touch pool state
            }
            if (drop[i]) { continue; } // never enters pending_copy: this is where the bytes are saved
            const std::int32_t expert = host_ids[static_cast<std::size_t>(i)];
            if (expert < 0 || expert >= gate_up.experts) {
                throw std::out_of_range("Flash-Next router produced an expert id outside the bank");
            }
            // The resident key. With one pool the same expert id in two layers is two different
            // weights, so a lookup that ignored the layer would let layer 7's expert 311 answer for
            // layer 22's -- silent, wrong, and invisible in any aggregate counter.
            const std::int32_t key = table_index * kExpertsPerBank + expert;
            std::int32_t slot =
                pooled ? p.slot_of_key[static_cast<std::size_t>(key)]
                       : table->of_expert[static_cast<std::size_t>(expert)];
            if (slot >= 0) {
                ++p.hits;
                if (pooled) {
                    p.g_slot_age[static_cast<std::size_t>(slot)]  = p.tick;
                    p.g_slot_pass[static_cast<std::size_t>(slot)] = p.pass;
                    // binary64, so there is nothing to saturate and no wrap to guard against; the
                    // decay below is what keeps the plane from silting up.
                    //
                    // P5-1: with live columns on, the credit is DEFERRED -- this path records which
                    // column asked for the key and the frequency moves only when the acceptance count
                    // arrives (commit_live_columns). Off, this is the original single increment, and
                    // that is the arm the OFF-arm hash pins.
                    if (live_columns_enabled()) {
                        LiveColumns& lc = live_columns();
                        lc.refs.push_back(LiveColumnRef{
                            static_cast<std::uint32_t>(key),
                            static_cast<std::uint16_t>(i / kRoutedPathsPerToken)});
                        lc.pending = true;
                    } else {
                        p.evict_freq[static_cast<std::size_t>(key)] += 1.0;
                    }
                } else {
                    table->slot_age[static_cast<std::size_t>(slot)]  = p.tick;
                    table->slot_pass[static_cast<std::size_t>(slot)] = p.pass;
                    ++table->slot_uses[static_cast<std::size_t>(slot)];
                }
            } else {
                ++p.misses;
                // ---- Phase IV W3: the CPU-serve decision (default off; rule in the per-call
                // block above). A served path takes NEITHER a slot nor a copy: it is marked here
                // and written back after the loop in the R1-drop form (legal slot + alpha = 0).
                if (serve_active) {
                    const std::int32_t miss_ordinal = serve_miss_ordinal++;
                    const bool serve_this = serve_legacy ? (serve_link_used >= serve_link_quota)
                                                         : (miss_ordinal % serve_d) != 0;
                    if (!serve_this) {
                        ++serve_link_used; // legacy quota accounting; harmless otherwise
                        ++p.serve_link_paths; // ③ measured link share (reachable only when serving is on)
                    } else {
                        if (serve_req == nullptr) {
                            // Lazy first use: the channel is taken only when a call actually has
                            // a serve candidate. A refused channel (previous request still
                            // unanswered after the bounded wait) turns serving off for this call
                            // -- every path then keeps the PCIe route, the safe direction.
                            serve_req = cpu_expert::CpuServeService::instance(serve_share.pin,
                                                                               serve_share.pin_first)
                                            .begin_request();
                            if (serve_req == nullptr) {
                                serve_active = false;
                            } else {
                                serve_req->sab_mode   = cpu_expert_fwd_sabotage_mode();
                                serve_req->layer      = table_index;
                                serve_req->rows_gu    = static_cast<std::uint64_t>(gate_up.rows);
                                serve_req->columns_gu = static_cast<std::uint64_t>(gate_up.columns);
                                serve_req->rows_dn    = static_cast<std::uint64_t>(down.rows);
                                serve_req->columns_dn = static_cast<std::uint64_t>(down.columns);
                                // Block-6 W1: arm the forward face for this round (or not -- then
                                // the round stays the block-3 landing+unpack shape).
                                auto& ctl          = staging_control_buffers();
                                serve_req->forward = serve_forward;
                                serve_req->tokens  = serve_forward ? tokens : 0;
                                if (serve_forward) {
                                    // Block-8 item 7 (ring): this round's slabs, picked by the slot
                                    // begin_request just assigned. x moves from the staging area into
                                    // its slab now (the D2H above ran before the slot existed); the
                                    // partial slab starts at zero because the team accumulates into
                                    // it and the apply side adds the SUM of the served columns.
                                    // job_y stays the SHARED per-job plane on purpose: rounds are
                                    // served strictly serially (run+reduce complete before the
                                    // answer), so only one job generation ever writes it at a time.
                                    const std::size_t slot    = static_cast<std::size_t>(serve_req->slot);
                                    const std::size_t slab    =
                                        static_cast<std::size_t>(cpu_expert::kCpuServeForwardCols) * 2'560;
                                    const std::size_t x_bytes =
                                        static_cast<std::size_t>(2'560) *
                                        static_cast<std::size_t>(tokens) * sizeof(std::uint16_t);
                                    const std::size_t p_elems =
                                        static_cast<std::size_t>(2'560) * static_cast<std::size_t>(tokens);
                                    std::uint16_t* const x_slab = ctl.serve_x.data() + slot * slab;
                                    std::memcpy(x_slab, ctl.serve_x_stage.data(), x_bytes);
                                    std::memset(ctl.serve_partial_f32.data() + slot * slab, 0,
                                                p_elems * sizeof(float));
                                    serve_req->x       = x_slab;
                                    serve_req->partial = ctl.serve_partial_f32.data() + slot * slab;
                                    serve_req->job_y   = ctl.serve_job_y_plane();
                                    // Block-8 item 3a: the bf16 slab both consumption paths read
                                    // (the serve thread converts into it before the answer), and
                                    // the completion word -- wired only when the stream-wait path
                                    // is on (the blocking path needs no flag).
                                    serve_req->partial_bf16 =
                                        ctl.serve_partial_bf16.data() + slot * slab;
                                    serve_req->done_flag = cpu_serve_streamwait_enabled()
                                                               ? (ctl.serve_done_flag.data() + slot)
                                                               : nullptr;
                                } else {
                                    serve_req->x            = nullptr;
                                    serve_req->partial      = nullptr;
                                    serve_req->job_y        = nullptr;
                                    serve_req->partial_bf16 = nullptr;
                                    serve_req->done_flag    = nullptr;
                                }
                            }
                        }
                        if (serve_active &&
                            serve_req->jobs < cpu_expert::kCpuServeMaxJobs) {
                            // One job per identity: a repeat within the call shares the first
                            // job's landing (the key, not the address, is the identity). Block-6
                            // W1: a repeat also shares the forward job, and its column+weight is
                            // appended to that job's list below.
                            std::int32_t q = -1;
                            for (std::int32_t qq = 0; qq < serve_req->jobs; ++qq) {
                                if (serve_req->key[qq] == static_cast<std::uint32_t>(key)) {
                                    q = qq;
                                    break;
                                }
                            }
                            if (q < 0) {
                                q = serve_req->jobs;
                                const auto source = static_cast<std::uint64_t>(
                                    (expert + sabotage) % gate_up.experts);
                                serve_req->key[q] = static_cast<std::uint32_t>(key);
                                serve_req->codes_gu[q] =
                                    reinterpret_cast<const std::uint8_t*>(gate_up.codes) +
                                    source *
                                        static_cast<std::uint64_t>(gate_up.code_bytes_per_expert);
                                serve_req->scales_gu[q] =
                                    reinterpret_cast<const std::uint8_t*>(gate_up.scales) +
                                    source *
                                        static_cast<std::uint64_t>(gate_up.scale_bytes_per_expert);
                                serve_req->codes_dn[q] =
                                    reinterpret_cast<const std::uint8_t*>(down.codes) +
                                    source * static_cast<std::uint64_t>(down.code_bytes_per_expert);
                                serve_req->scales_dn[q] =
                                    reinterpret_cast<const std::uint8_t*>(down.scales) +
                                    source * static_cast<std::uint64_t>(down.scale_bytes_per_expert);
                                if (serve_req->forward) {
                                    // The divisors the ported forward divides by, from the same
                                    // (sabotage-adjusted) expert the four pointers above name.
                                    serve_req->divisor_gu[q] =
                                        gate_up.weight_scale_divisors[source];
                                    serve_req->divisor_dn[q] =
                                        down.weight_scale_divisors[source];
                                    serve_req->jn[q] = 0; // fresh column list
                                }
                                serve_req->jobs = q + 1;
                            }
                            if (serve_req->forward) {
                                // The column (token) this path belongs to and its routing weight.
                                // host_alpha is the ROUTER's value here: the R1-form write-back
                                // below zeroes the served path's alpha only after this point, and
                                // that zero is for the GPU kernel -- the real weight is what the
                                // CPU contribution must carry.
                                const std::int32_t col = i / kRoutedPathsPerToken;
                                std::int32_t c         = 0;
                                while (c < serve_req->jn[q] && serve_req->jt[q][c] != col) { ++c; }
                                if (c == serve_req->jn[q]) {
                                    if (c < cpu_expert::kCpuServeForwardCols) {
                                        serve_req->jt[q][c] = col;
                                        serve_req->ja[q][c] = 0.0F;
                                        serve_req->jn[q]   = c + 1;
                                    } else {
                                        c = -1; // defensive: columns <= tokens <= the face's cap
                                    }
                                }
                                if (c >= 0) {
                                    serve_req->ja[q][c] +=
                                        host_alpha[static_cast<std::size_t>(i)];
                                }
                            }
                            cpu_serve_path[static_cast<std::size_t>(i)] = true;
                            ++p.cpu_served;
                            continue; // no victim, no copy: the PCIe bytes are the thing we buy
                        }
                    }
                }
                // ---- block-2 phase I: CPU NVFP4 decoder self-check + real-record probe (default off,
                // once per process). The self-check judges the DECODERS against the format spec's
                // literal values (see impl/cpu_expert_nvfp4.h), and with SABOTAGE=1 it must go red --
                // that is the whole point: this line is the first evidence that the CPU path reads the
                // record the way this tree says the record is laid out.
                if (cpu_expert_check_enabled()) {
                    static bool reported = false;
                    if (!reported) {
                        reported             = true;
                        const bool sab       = cpu_expert_sabotage_enabled();
                        const auto checked   = cpu_expert::self_check(sab);
                        std::fprintf(stderr,
                                     "flash_next_cpu_expert_check elements=%u mismatches=%u "
                                     "first_bits=0x%02X first_expect=%.7g first_got=%.7g sabotage=%d "
                                     "verdict=%s\n",
                                     checked.elements, checked.mismatches, checked.first_bits,
                                     static_cast<double>(checked.first_expect),
                                     static_cast<double>(checked.first_got), sab ? 1 : 0,
                                     checked.ok() ? "OK" : "FAIL");
                        // ---- phase I-b: the real-record walk, judged element by element against decode_tensor -- the
                        // tree's independent "any stored representation -> FP32" reader. One contiguous record per
                        // matrix [codes | pad256 | scales | divisor], laid out exactly as this tree's own
                        // weight_geometry() computes (core/weight_view.cpp:87). The CPU side transcribes the
                        // device scale formula BY HAND (nvfp4_codec.cuh:31-38) on purpose: a transcription
                        // error in our own code is exactly what this comparison must catch. Reference only --
                        // decode_tensor never appears on the hot path (block-2 plan sec-0).
                        const auto* gu_codes = reinterpret_cast<const std::uint8_t*>(gate_up.codes);
                        const auto* gu_scale = reinterpret_cast<const std::uint8_t*>(gate_up.scales);
                        const auto* dn_codes = reinterpret_cast<const std::uint8_t*>(down.codes);
                        const auto* dn_scale = reinterpret_cast<const std::uint8_t*>(down.scales);
                        struct IbMatrix {
                            const char* name;
                            const std::uint8_t* codes;
                            const std::uint8_t* scales;
                            std::uint64_t rows;
                            std::uint64_t columns;
                            std::uint64_t code_bytes;
                            std::uint64_t scale_bytes;
                            float divisor;
                        };
                        const IbMatrix ib_mats[2] = {
                            {"gate_up", gu_codes, gu_scale,
                             static_cast<std::uint64_t>(gate_up.rows),
                             static_cast<std::uint64_t>(gate_up.columns), gate_up.code_bytes_per_expert,
                             gate_up.scale_bytes_per_expert,
                             gate_up.weight_scale_divisors[static_cast<std::size_t>(expert)]},
                            {"down", dn_codes, dn_scale,
                             static_cast<std::uint64_t>(down.rows),
                             static_cast<std::uint64_t>(down.columns), down.code_bytes_per_expert,
                             down.scale_bytes_per_expert,
                             down.weight_scale_divisors[static_cast<std::size_t>(expert)]}};
                        // Phase II reuses these decoded fp32 matrices as reference A's weights, so
                        // the decode_tensor call site below stays exactly one (no second decode).
                        std::array<std::vector<float>, 2> ref_mats;
                        for (std::size_t mi = 0; mi < 2; ++mi) {
                            const IbMatrix& m = ib_mats[mi];
                            const bool instrument_ok =
                                (m.code_bytes == m.rows * m.columns / 2) &&
                                (m.scale_bytes == m.rows * m.columns / 16) && (m.divisor > 0.0F) &&
                                (m.rows % 128 == 0) && (m.columns % 64 == 0);
                            if (!instrument_ok) {
                                std::fprintf(stderr,
                                             "flash_next_cpu_expert_ib matrix=%s expert=%d rows=%llu "
                                             "cols=%llu code_bytes=%llu scale_bytes=%llu divisor=%.6g "
                                             "instrument_ok=0 verdict=FAIL\n",
                                             m.name, static_cast<int>(expert),
                                             static_cast<unsigned long long>(m.rows),
                                             static_cast<unsigned long long>(m.columns),
                                             static_cast<unsigned long long>(m.code_bytes),
                                             static_cast<unsigned long long>(m.scale_bytes),
                                             static_cast<double>(m.divisor));
                                continue;
                            }
                            try {
                                const std::array<std::uint64_t, 2> shape = {m.rows, m.columns};
                                const auto geom = ::ninfer::weight_geometry(
                                    ::ninfer::QType::NVFP4,
                                    ::ninfer::QuantLayout::BlockScaleK16M128x4,
                                    std::span<const std::uint64_t>(shape.data(), shape.size()), 1);
                                std::vector<std::byte> record(geom.bytes, std::byte{0});
                                std::memcpy(record.data(), m.codes, geom.code_bytes);
                                std::memcpy(record.data() + geom.scale_offset, m.scales, geom.scale_bytes);
                                std::memcpy(record.data() + geom.divisor_offset, &m.divisor,
                                            sizeof(float));
                                const std::vector<float> reference = ::ninfer::models::qwen3_5::decode_tensor(
                                    geom, std::span<const std::byte>(record.data(), record.size()));
                                ref_mats[mi] = reference;
                                const auto cmp = cpu_expert::compare_matrix(
                                    m.codes, m.scales, m.rows, m.columns, m.divisor, sab,
                                    reference.data());
                                const char* verdict = cmp.ok(0.99, 0.04) ? "OK" : "FAIL";
                                std::fprintf(stderr,
                                             "flash_next_cpu_expert_ib matrix=%s expert=%d rows=%llu "
                                             "cols=%llu elements=%llu cos=%.9f rel_mae=%.6g "
                                             "mismatches=%llu first_expect=%.7g first_got=%.7g "
                                             "max_abs_diff=%.6g divisor=%.6g sabotage=%d "
                                             "instrument_ok=1 verdict=%s\n",
                                             m.name, static_cast<int>(expert),
                                             static_cast<unsigned long long>(m.rows),
                                             static_cast<unsigned long long>(m.columns),
                                             static_cast<unsigned long long>(cmp.elements), cmp.cos,
                                             cmp.rel_mae, static_cast<unsigned long long>(cmp.mismatches),
                                             static_cast<double>(cmp.first_expect),
                                             static_cast<double>(cmp.first_got), cmp.max_abs_diff,
                                             static_cast<double>(m.divisor), sab ? 1 : 0, verdict);
                            } catch (const std::exception& ib_error) {
                                std::fprintf(stderr,
                                             "flash_next_cpu_expert_ib matrix=%s expert=%d error=%.200s "
                                             "verdict=FAIL\n",
                                             m.name, static_cast<int>(expert), ib_error.what());
                            }
                        }
                        // ---- phase II: single-expert CPU W4A8 forward, offline comparison. The
                        // frozen judgment is QA v3 (shared block-3 phase II file); the kernel and
                        // every reference run once per process, here only, off the hot path. The
                        // 8 + 1 lines below are re-judged from the raw fields by the harness.
                        {
                            const std::int32_t sab_ii = cpu_expert_fwd_sabotage_mode();
                            const std::size_t gu_elems =
                                static_cast<std::size_t>(ib_mats[0].rows) *
                                static_cast<std::size_t>(ib_mats[0].columns);
                            const std::size_t dn_elems =
                                static_cast<std::size_t>(ib_mats[1].rows) *
                                static_cast<std::size_t>(ib_mats[1].columns);
                            const bool refs_ready =
                                ref_mats[0].size() == gu_elems && ref_mats[1].size() == dn_elems;
                            cpu_expert::Phase2RunResult p2;
                            if (refs_ready) {
                                const cpu_expert::Phase2Matrix gu_view{
                                    ib_mats[0].codes, ib_mats[0].scales, ib_mats[0].rows,
                                    ib_mats[0].columns, ib_mats[0].divisor, ib_mats[0].code_bytes,
                                    ib_mats[0].scale_bytes};
                                const cpu_expert::Phase2Matrix dn_view{
                                    ib_mats[1].codes, ib_mats[1].scales, ib_mats[1].rows,
                                    ib_mats[1].columns, ib_mats[1].divisor, ib_mats[1].code_bytes,
                                    ib_mats[1].scale_bytes};
                                p2 = cpu_expert::run_phase2(gu_view, dn_view, ref_mats[0].data(),
                                                            ref_mats[1].data(), sab_ii);
                            }
                            const char* const probe_names[2] = {"P1", "P2"};
                            for (int pi = 0; pi < 2; ++pi) {
                                const cpu_expert::Phase2ProbeResult& pr = p2.probes[pi];
                                const bool gu_ok =
                                    pr.gu.computed && pr.gu.elements == 1280ULL &&
                                    pr.gu.codes_mismatches == 0 && pr.gu.int_mismatches == 0 &&
                                    pr.gu.cos > 0.99 && pr.gu.rel_mae < 0.04 &&
                                    pr.gu.rel_mae_vs_ideal <= 1.0e-4;
                                const bool act_ok = pr.act_computed && pr.act_elements == 640ULL &&
                                                    pr.act_bf16_mismatches == 0;
                                const bool dn_ok =
                                    pr.dn.computed && pr.dn.elements == 2560ULL &&
                                    pr.dn.codes_mismatches == 0 && pr.dn.int_mismatches == 0 &&
                                    pr.dn.cos > 0.99 && pr.dn.rel_mae < 0.04 &&
                                    pr.dn.rel_mae_vs_ideal <= 1.0e-4;
                                const bool whole_ok =
                                    pr.whole.computed && pr.whole.elements == 2560ULL &&
                                    pr.whole.cos > 0.99 && pr.whole.rel_mae < 0.04 &&
                                    pr.whole.rel_mae_vs_ideal <= 1.0e-4;  // alpha_on is 0 by design
                                std::fprintf(stderr,
                                             "flash_next_cpu_expert_ii stage=gate_up probe=%s "
                                             "expert=%d elements=%llu codes_mismatches=%llu "
                                             "int_mismatches=%llu max_abs_diff_fixed=%lld cos=%.9f "
                                             "rel_mae=%.6g floor=%.6g rel_mae_vs_ideal=%.6g "
                                             "sab_mode=%d verdict=%s\n",
                                             probe_names[pi], static_cast<int>(expert),
                                             static_cast<unsigned long long>(pr.gu.elements),
                                             static_cast<unsigned long long>(pr.gu.codes_mismatches),
                                             static_cast<unsigned long long>(pr.gu.int_mismatches),
                                             static_cast<long long>(pr.gu.max_abs_diff_fixed),
                                             pr.gu.cos, pr.gu.rel_mae, pr.gu.floor_v,
                                             pr.gu.rel_mae_vs_ideal, static_cast<int>(sab_ii),
                                             gu_ok ? "OK" : "FAIL");
                                std::fprintf(stderr,
                                             "flash_next_cpu_expert_ii stage=act probe=%s expert=%d "
                                             "elements=%llu bf16_mismatches=%llu bf16_round=1 "
                                             "sab_mode=%d verdict=%s\n",
                                             probe_names[pi], static_cast<int>(expert),
                                             static_cast<unsigned long long>(pr.act_elements),
                                             static_cast<unsigned long long>(pr.act_bf16_mismatches),
                                             static_cast<int>(sab_ii), act_ok ? "OK" : "FAIL");
                                std::fprintf(stderr,
                                             "flash_next_cpu_expert_ii stage=down probe=%s expert=%d "
                                             "elements=%llu codes_mismatches=%llu int_mismatches=%llu "
                                             "max_abs_diff_fixed=%lld cos=%.9f rel_mae=%.6g floor=%.6g "
                                             "rel_mae_vs_ideal=%.6g sab_mode=%d verdict=%s\n",
                                             probe_names[pi], static_cast<int>(expert),
                                             static_cast<unsigned long long>(pr.dn.elements),
                                             static_cast<unsigned long long>(pr.dn.codes_mismatches),
                                             static_cast<unsigned long long>(pr.dn.int_mismatches),
                                             static_cast<long long>(pr.dn.max_abs_diff_fixed),
                                             pr.dn.cos, pr.dn.rel_mae, pr.dn.floor_v,
                                             pr.dn.rel_mae_vs_ideal, static_cast<int>(sab_ii),
                                             dn_ok ? "OK" : "FAIL");
                                std::fprintf(stderr,
                                             "flash_next_cpu_expert_ii stage=whole probe=%s expert=%d "
                                             "elements=%llu cos=%.9f rel_mae=%.6g floor=%.6g "
                                             "rel_mae_vs_ideal=%.6g alpha_on=0 bf16_round=1 sab_mode=%d "
                                             "verdict=%s\n",
                                             probe_names[pi], static_cast<int>(expert),
                                             static_cast<unsigned long long>(pr.whole.elements),
                                             pr.whole.cos, pr.whole.rel_mae, pr.whole.floor_v,
                                             pr.whole.rel_mae_vs_ideal, static_cast<int>(sab_ii),
                                             whole_ok ? "OK" : "FAIL");
                            }
                            const bool recompute_ok =
                                p2.probes[0].gu.codes_mismatches == 0 &&
                                p2.probes[0].dn.codes_mismatches == 0 &&
                                p2.probes[1].gu.codes_mismatches == 0 &&
                                p2.probes[1].dn.codes_mismatches == 0;
                            const cpu_expert::IsaFlags isa = cpu_expert::isa_flags();
                            const bool instr_ok =
                                p2.ran && p2.instr.expand_ok && p2.instr.q9_finite &&
                                p2.instr.quant_lossless_p2 && p2.instr.zero_x_ok &&
                                p2.instr.probe_p1_ok && p2.instr.probe_p2_ok && recompute_ok &&
                                p2.instr.oracle_elements_gu == 1280ULL &&
                                p2.instr.oracle_elements_dn == 2560ULL;
                            std::fprintf(stderr,
                                         "flash_next_cpu_expert_ii_instr expert=%d fixed_shift=%d "
                                         "divisor_gu=%.6g divisor_dn=%.6g quant_lossless=%d "
                                         "codes_recompute_match=%d expand_scalar_match=%d zero_x_ok=%d "
                                         "q9_finite=%d probe_p1_ok=%d probe_p2_ok=%d isa_avx2=%d "
                                         "isa_avxvnni=%d emitted512=0 oracle_elements_gu=%llu "
                                         "oracle_elements_dn=%llu sab_mode=%d verdict=%s\n",
                                         static_cast<int>(expert), cpu_expert::kFixedShift,
                                         static_cast<double>(ib_mats[0].divisor),
                                         static_cast<double>(ib_mats[1].divisor),
                                         p2.instr.quant_lossless_p2 ? 1 : 0, recompute_ok ? 1 : 0,
                                         p2.instr.expand_ok ? 1 : 0, p2.instr.zero_x_ok ? 1 : 0,
                                         p2.instr.q9_finite ? 1 : 0, p2.instr.probe_p1_ok ? 1 : 0,
                                         p2.instr.probe_p2_ok ? 1 : 0, isa.avx2, isa.avx_vnni,
                                         static_cast<unsigned long long>(p2.instr.oracle_elements_gu),
                                         static_cast<unsigned long long>(p2.instr.oracle_elements_dn),
                                         static_cast<int>(sab_ii), instr_ok ? "OK" : "FAIL");
                        }
                        // ---- block-6 W1: the serve forward face's own correctness line. The
                        // ported W4A16 forward on this expert's real faces, judged element by
                        // element against the same reference A the phase-II line uses
                        // (decode_tensor fp32 weights x the exact bf16 activation vector the
                        // kernel sees), one deterministic column. This is the line J1 re-judges:
                        // cos>0.99 and rel_MAE<0.04 are the frozen gates, and the sabotage env
                        // turns it red on purpose -- a judgement nobody has seen fail is not a
                        // judgement.
                        {
                            const std::int32_t pf_sab = cpu_expert_fwd_sabotage_mode();
                            const bool pf_geom =
                                ib_mats[0].rows == 1'280 && ib_mats[0].columns == 2'560 &&
                                ib_mats[1].rows == 2'560 && ib_mats[1].columns == 640 &&
                                ref_mats[0].size() ==
                                    static_cast<std::size_t>(1'280) * static_cast<std::size_t>(2'560) &&
                                ref_mats[1].size() ==
                                    static_cast<std::size_t>(2'560) * static_cast<std::size_t>(640);
                            std::vector<std::uint16_t> pf_x(2'560, 0);
                            std::vector<std::uint16_t> pf_y(2'560, 0);
                            bool pf_x_ok = false;
                            bool pf_ran  = false;
                            cpu_expert::VecCompare pf_cmp{};
                            if (pf_geom) {
                                std::vector<float> xf(2'560, 0.0F);
                                cpu_expert::fill_probe_p1(xf, pf_x_ok);
                                for (std::size_t k = 0; k < xf.size(); ++k) {
                                    pf_x[k] = cpu_expert::ported::f32_to_bf16_rn(xf[k]);
                                }
                                const std::uint16_t* pf_xp[1] = {pf_x.data()};
                                std::uint16_t* pf_yp[1]       = {pf_y.data()};
                                pf_ran = cpu_expert::ported::ported_forward_a16(
                                    cpu_expert::ported::best_isa(), ib_mats[0].codes,
                                    static_cast<std::size_t>(ib_mats[0].columns / 2),
                                    ib_mats[0].scales, ib_mats[0].divisor, ib_mats[1].codes,
                                    static_cast<std::size_t>(ib_mats[1].columns / 2),
                                    ib_mats[1].scales, ib_mats[1].divisor, 1, pf_xp, pf_yp, pf_sab,
                                    2);
                            }
                            if (pf_ran) {
                                // Reference A over the EXACT bf16 activation vector: the oracle
                                // isolates the weight pipeline (record walk, K16M128x4 scale
                                // tiles, divisor-as-divide, A16 math) from the x-bf16 rounding
                                // both sides share.
                                std::vector<double> xd(2'560, 0.0);
                                for (std::size_t k = 0; k < pf_x.size(); ++k) {
                                    xd[k] =
                                        static_cast<double>(cpu_expert::ported::bf16_to_f32(pf_x[k]));
                                }
                                const std::vector<float>& gu_w = ref_mats[0];
                                const std::vector<float>& dn_w = ref_mats[1];
                                std::vector<double> h_ref(640, 0.0);
                                for (int u = 0; u < 640; ++u) {
                                    double g_v = 0.0, u_v = 0.0;
                                    for (int k = 0; k < 2'560; ++k) {
                                        g_v += static_cast<double>(
                                                   gu_w[static_cast<std::size_t>(u) * 2'560 + k]) *
                                               xd[k];
                                        u_v +=
                                            static_cast<double>(gu_w[(static_cast<std::size_t>(640) +
                                                                      static_cast<std::size_t>(u)) *
                                                                         2'560 +
                                                                     k]) *
                                            xd[k];
                                    }
                                    h_ref[u] = cpu_expert::silu_d64(g_v) * u_v;
                                }
                                std::vector<double> y_ref(2'560, 0.0);
                                for (int r = 0; r < 2'560; ++r) {
                                    double dn_acc = 0.0;
                                    for (int u = 0; u < 640; ++u) {
                                        dn_acc += static_cast<double>(
                                                      dn_w[static_cast<std::size_t>(r) * 640 + u]) *
                                                  h_ref[u];
                                    }
                                    y_ref[r] = dn_acc;
                                }
                                std::vector<double> y_got(2'560, 0.0);
                                for (std::size_t k = 0; k < pf_y.size(); ++k) {
                                    y_got[k] =
                                        static_cast<double>(cpu_expert::ported::bf16_to_f32(pf_y[k]));
                                }
                                pf_cmp = cpu_expert::compare_vec(y_got.data(), y_ref.data(),
                                                                 y_got.size());
                            }
                            const bool pf_ok = pf_ran && pf_cmp.cos > 0.99 && pf_cmp.rel_mae < 0.04;
                            std::fprintf(stderr,
                                         "flash_next_cpu_expert_portfwd stage=whole probe=P1 "
                                         "expert=%d elements=%llu cos=%.9f rel_mae=%.6g x_ok=%d "
                                         "sab_mode=%d verdict=%s\n",
                                         static_cast<int>(expert),
                                         static_cast<unsigned long long>(pf_cmp.elements),
                                         pf_cmp.cos, pf_cmp.rel_mae, pf_x_ok ? 1 : 0, pf_sab,
                                         pf_ok ? "OK" : "FAIL");
                        }
                    }
                }
                // ---- block-4 P1: port probe (default off, once per process) ---------------------
                // Throughput of the ported kernels (this tree's layout arm vs the reference access
                // arm) on this layer's real GU-face records, plus the one-shot sabotage feasibility
                // check. Zero GPU; probe grade only (V4 caliber -- never quoted as arm tok/s).
                if (cpu_expert_port_probe_enabled()) {
                    static bool port_reported = false;
                    if (!port_reported) {
                        port_reported = true;
                        const std::int32_t port_sab = cpu_expert_port_sabotage_mode();
                        const cpu_expert::ported::CpuIsa port_isa = cpu_expert::ported::best_isa();
                        constexpr int kPortProbeExperts = 16;
                        int port_experts[kPortProbeExperts];
                        for (int pqi = 0; pqi < kPortProbeExperts; ++pqi) {
                            port_experts[pqi] = (expert + pqi) % gate_up.experts;
                        }
                        const auto tp = cpu_expert::ported::port_probe_throughput(
                            port_isa, /*gu=*/true,
                            reinterpret_cast<const std::uint8_t*>(gate_up.codes),
                            static_cast<std::size_t>(gate_up.code_bytes_per_expert),
                            reinterpret_cast<const std::uint8_t*>(gate_up.scales),
                            static_cast<std::size_t>(gate_up.code_bytes_per_expert),
                            static_cast<std::size_t>(gate_up.scale_bytes_per_expert), port_experts,
                            kPortProbeExperts, /*ncols=*/1, /*rounds=*/3);
                        const auto sc = cpu_expert::ported::port_probe_sab_check(
                            port_isa, /*gu=*/true,
                            reinterpret_cast<const std::uint8_t*>(gate_up.codes),
                            static_cast<std::size_t>(gate_up.code_bytes_per_expert),
                            reinterpret_cast<const std::uint8_t*>(gate_up.scales),
                            static_cast<std::size_t>(gate_up.code_bytes_per_expert),
                            static_cast<std::size_t>(gate_up.scale_bytes_per_expert), expert,
                            /*ncols=*/1);
                        const char* port_isa_name =
                            (port_isa == cpu_expert::ported::CpuIsa::kAvxVnni)  ? "avx-vnni"
                            : (port_isa == cpu_expert::ported::CpuIsa::kAvx2) ? "avx2"
                                                                              : "scalar";
                        std::fprintf(stderr,
                                     "flash_next_cpu_expert_port_probe isa=%s experts=%d ncols=%d "
                                     "ours_best_us=%.1f ours_worst_us=%.1f orig_best_us=%.1f "
                                     "orig_worst_us=%.1f ratio_best=%.3f elements=%llu "
                                     "sab_diffs=%llu/%llu/%llu sab_modes_red=%d sab_env=%d "
                                     "probe_grade=1\n",
                                     port_isa_name, tp.experts, tp.ncols, tp.ours_best_us,
                                     tp.ours_worst_us, tp.orig_best_us, tp.orig_worst_us,
                                     tp.ratio_best, static_cast<unsigned long long>(sc.elements),
                                     static_cast<unsigned long long>(sc.diff1),
                                     static_cast<unsigned long long>(sc.diff2),
                                     static_cast<unsigned long long>(sc.diff3), sc.ok ? 1 : 0,
                                     static_cast<int>(port_sab));
                    }
                }
                // Victim choice, in order. Least-Stale ahead of plain LRU is SpecMD's finding that
                // an entry evicted inside the same forward pass and needed again immediately after
                // (a "collision miss") is what plain LRU gets wrong: measured 4.5-12.6% of accesses
                // under LRU versus 1.6-1.9% with a stale/current split.
                //   1. a never-filled slot
                //   2. STALE and not hot: last used in an earlier pass, least recently used first
                //   3. current-pass and not hot, least recently used first
                //   4. anything (a fully hot layer still has to make room rather than fail)
                const auto oldest_where = [&](bool stale_only, bool skip_hot) {
                    std::int32_t victim_slot = -1;
                    std::int32_t oldest_age  = 0x7FFFFFFF;
                    for (std::int32_t s = 0; s < slots; ++s) {
                        const auto candidate = static_cast<std::size_t>(s);
                        if (skip_hot && table->slot_hot[candidate] != 0) { continue; }
                        if (stale_only && table->slot_pass[candidate] == p.pass) { continue; }
                        const std::int32_t age = table->slot_age[candidate];
                        if (age < oldest_age) {
                            oldest_age  = age;
                            victim_slot = s;
                        }
                    }
                    return victim_slot;
                };
                std::int32_t victim = pooled ? global_pick_victim(p) : -1;
                for (std::int32_t s = 0; !pooled && s < slots; ++s) {
                    if (table->slot_expert[static_cast<std::size_t>(s)] < 0) {
                        victim = s;
                        break;
                    }
                }
                if (!pooled && victim < 0) { victim = oldest_where(/*stale_only=*/true, /*skip_hot=*/true); }
                if (!pooled && victim < 0) { victim = oldest_where(/*stale_only=*/false, /*skip_hot=*/true); }
                if (!pooled && victim < 0) { victim = oldest_where(/*stale_only=*/false, /*skip_hot=*/false); }
                if (victim < 0) {
                    throw std::logic_error("Flash-Next slot pool could not pick an eviction victim");
                }
                if (pooled) {
                    // Global bookkeeping: every plane is keyed by the RESIDENT KEY, not by the slot,
                    // so an eviction removes a key and the same key may come back into any slot.
                    const std::int32_t evicted = p.key_of_slot[static_cast<std::size_t>(victim)];
                    if (evicted >= 0) {
                        ++p.evictions;
                        p.slot_of_key[static_cast<std::size_t>(evicted)] = -1;
                        // `evict_freq[evicted]` is deliberately NOT reset: a frequency that survives
                        // eviction is the only reason LFRU can rank anything.
                    }
                    p.key_of_slot[static_cast<std::size_t>(victim)] = key;
                    p.slot_of_key[static_cast<std::size_t>(key)]    = victim;
                    p.g_slot_age[static_cast<std::size_t>(victim)]  = p.tick;
                    p.g_slot_pass[static_cast<std::size_t>(victim)] = p.pass;
                    p.g_slot_hot[static_cast<std::size_t>(victim)]  = 0;
                    p.evict_freq[static_cast<std::size_t>(key)] += 1.0;
                } else {
                if (table->slot_expert[static_cast<std::size_t>(victim)] >= 0) {
                    ++p.evictions;
                    const std::int32_t evicted = table->slot_expert[static_cast<std::size_t>(victim)];
                    table->of_expert[static_cast<std::size_t>(evicted)] = -1;
                }
                table->slot_expert[static_cast<std::size_t>(victim)] = expert;
                table->slot_age[static_cast<std::size_t>(victim)]    = p.tick;
                table->slot_pass[static_cast<std::size_t>(victim)]   = p.pass;
                table->slot_uses[static_cast<std::size_t>(victim)]   = 1;
                table->slot_hot[static_cast<std::size_t>(victim)]    = 0;
                table->of_expert[static_cast<std::size_t>(expert)]   = victim;
                layer_slot_expert[static_cast<std::size_t>(victim)]  = expert;
                }
                slot = victim;
                pending_copy.push_back(slot);
                used_slots[static_cast<std::size_t>(used_count)] = slot;
                ++used_count;
            }
            host_slots[static_cast<std::size_t>(i)] = slot;
        }
        stage_profile().decide_ms += host_now_ms() - t_decide0;

        // ---- Phase IV W3: publish, then write back the served paths in the R1-drop form ----------
        // Served paths still point at a LEGAL slot and carry alpha = 0 (the R1 drop form), so no
        // kernel can read an uncommitted slot and the routed sum stays well-formed. Publishing
        // before the write-back lets the serve engine start while this call finishes staging.
        //
        // BLOCK-6 W2 -- WHERE THE "REAL VALUE" GOES, AND WHY IT IS NOT THIS ALPHA. The served
        // path's contribution comes from the CPU forward result, and it is carried by the apply
        // side (flash_next_cpu_serve_apply), which adds the team's partial into `output` AFTER the
        // MoE kernels. A non-zero alpha HERE would make the kernel add alpha*E(x, legal_slot) --
        // the WRONG expert's weights (the slot belongs to another path's expert) AND a double
        // count once the partial lands. The R1 form (legal slot, alpha = 0) is what keeps the GPU
        // side of the sum exactly zero for served paths; the CPU partial is the other side.
        if (serve_req != nullptr && serve_req->jobs > 0) {
            cpu_expert::CpuServeService::instance().commit_request();
            // The committed-slot template: any path that is neither served nor dropped has one.
            // (The first miss of a call always takes the link and the argmax path is never
            // dropped, so at least one such path exists; the fallback is counted, never silent.)
            std::int32_t legal = -1;
            for (std::int32_t i = 0; i < paths; ++i) {
                if (cpu_serve_path[static_cast<std::size_t>(i)] || drop[i]) { continue; }
                legal = host_slots[static_cast<std::size_t>(i)];
                break;
            }
            if (legal < 0) {
                legal = 0;
                ++p.serve_fallback;
            }
            for (std::int32_t i = 0; i < paths; ++i) {
                if (!cpu_serve_path[static_cast<std::size_t>(i)]) { continue; }
                host_slots[static_cast<std::size_t>(i)] = legal;
                host_alpha[static_cast<std::size_t>(i)] = 0.0F;
            }
            CUDA_CHECK(cudaMemcpyAsync(scratch.alpha.data, host_alpha.data(),
                                       static_cast<std::size_t>(paths) * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
            stage_profile().note_h2d(static_cast<std::size_t>(paths) * sizeof(float));
        }

        // R1 fixup. Survivors are renormalized back to a unit routed sum (the router's top-10 weights
        // are already renormalized, so kept_sum = 1 - sum(dropped) and the scale is 1/kept_sum). Every
        // dropped path has its slot index REWRITTEN TO THE TOP EXPERT'S SLOT, and its weight set to 0:
        // the kernel therefore reads a COMMITTED slot and multiplies it by zero, so no matmul can ever
        // touch an uncommitted slot (the failure mode the community recipe warns about). The GEMV work
        // is NOT saved -- only the PCIe bytes are, which is exactly the thing we are buying.
        if (drop_armed) {
            float kept_sum          = 0.0F;
            float drop_sum          = 0.0F;
            std::int32_t dropped_here = 0;
            for (std::int32_t i = 0; i < paths; ++i) {
                if (drop[i]) {
                    ++dropped_here;
                    drop_sum += host_alpha[static_cast<std::size_t>(i)];
                    continue;
                }
                kept_sum += host_alpha[static_cast<std::size_t>(i)];
            }
            if (dropped_here > 0) {
                p.dropped += static_cast<std::uint64_t>(dropped_here);
                // lambda joins the two rungs into one axis; see moe_subst_damp() for the derivation.
                const float damp = moe_subst_enabled() ? moe_subst_damp() : 0.0F;
                if (damp >= 1.0F) {
                    // R2 SUBSTITUTION: keep the weight, only move the slot. The routed sum is already
                    // 1 (the router renormalises top-k), so there is nothing to renormalise, and
                    // scratch.alpha stays EXACTLY as the router wrote it -- so the H2D copy is skipped
                    // entirely, i.e. one less synchronisation point per layer per forward. (Handled as
                    // a separate branch, rather than as damp==1 arithmetic, so the last bit of every
                    // weight is provably untouched.)
                    for (std::int32_t i = 0; i < paths; ++i) {
                        if (drop[i]) {
                            host_slots[static_cast<std::size_t>(i)] =
                                host_slots[static_cast<std::size_t>(top_path)];
                        }
                    }
                } else {
                    // Damped: survivors are amplified by (1 - lambda*d)/kept_sum and the replaced
                    // paths keep lambda of their weight, so the routed total stays exactly 1.
                    // lambda == 0 reproduces the pure drop variant bit-for-bit.
                    const float scale = (kept_sum > 1e-6F)
                                            ? ((1.0F - damp * drop_sum) / kept_sum) : 0.0F;
                    for (std::int32_t i = 0; i < paths; ++i) {
                        if (drop[i]) {
                            host_alpha[static_cast<std::size_t>(i)] = damp * host_alpha[static_cast<std::size_t>(i)];
                            host_slots[static_cast<std::size_t>(i)] =
                                host_slots[static_cast<std::size_t>(top_path)];
                        } else {
                            host_alpha[static_cast<std::size_t>(i)] *= scale;
                        }
                    }
                    CUDA_CHECK(cudaMemcpyAsync(scratch.alpha.data, host_alpha.data(),
                                               static_cast<std::size_t>(paths) * sizeof(float),
                                               cudaMemcpyHostToDevice, stream));
                    stage_profile().note_h2d(static_cast<std::size_t>(paths) * sizeof(float));
                }
            }
        }
    } else {
        // No cache: first-seen order, one slot per distinct expert, and every slot is a miss.
        std::array<std::int32_t, 512> slot_of_expert{};
        slot_of_expert.fill(-1);
        for (std::int32_t i = 0; i < paths; ++i) {
            const std::int32_t expert = host_ids[static_cast<std::size_t>(i)];
            if (expert < 0 || expert >= gate_up.experts) {
                throw std::out_of_range("Flash-Next router produced an expert id outside the bank");
            }
            if (slot_of_expert[static_cast<std::size_t>(expert)] < 0) {
                slot_of_expert[static_cast<std::size_t>(expert)] = used_count;
                layer_slot_expert[static_cast<std::size_t>(used_count)] = expert;
                pending_copy.push_back(used_count);
                used_slots[static_cast<std::size_t>(used_count)] = used_count;
                ++used_count;
            }
            host_slots[static_cast<std::size_t>(i)] =
                slot_of_expert[static_cast<std::size_t>(expert)];
        }
    }

    // ---- destination planes: this layer's sub-array of the pool, or the per-call planes ----------
    std::uint8_t* gate_up_codes  = nullptr;
    std::uint8_t* gate_up_scales = nullptr;
    std::uint8_t* down_codes     = nullptr;
    std::uint8_t* down_scales    = nullptr;
    float* gate_up_divisors      = nullptr;
    float* down_divisors         = nullptr;

    if (slots > 0) {
        SlotPool& p      = *pool;
        const auto index = static_cast<std::size_t>(table_index);
        const auto slot_count = static_cast<std::size_t>(p.slots);
        // With one pool there is no per-layer sub-array: slot indices are global, so the plane base
        // is the allocation itself. The stride per slot is unchanged, which is why the copy kernel
        // and its 16-byte-unit walk need nothing (every stride is already a multiple of 256).
        const auto base = pooled ? std::size_t{0} : index * slot_count;
        gate_up_codes  = p.gate_up_codes + base * kStagedGateUpCodeBytes;
        gate_up_scales = p.gate_up_scales + base * kStagedGateUpScaleBytes;
        down_codes     = p.down_codes + base * kStagedDownCodeBytes;
        down_scales    = p.down_scales + base * kStagedDownScaleBytes;
        gate_up_divisors = p.gate_up_divisors + base;
        down_divisors    = p.down_divisors + base;
    } else {
        if (scratch.staged_gate_up_codes.data == nullptr ||
            scratch.staged_gate_up_scales.data == nullptr ||
            scratch.staged_down_codes.data == nullptr ||
            scratch.staged_down_scales.data == nullptr ||
            scratch.staged_gate_up_divisors.data == nullptr ||
            scratch.staged_down_divisors.data == nullptr) {
            throw std::logic_error("Flash-Next expert staging planes are missing: build the MoE "
                                   "workspace with NINFER_FLASH_NEXT_EXPERTS_HOST set");
        }
        gate_up_codes    = static_cast<std::uint8_t*>(scratch.staged_gate_up_codes.data);
        gate_up_scales   = static_cast<std::uint8_t*>(scratch.staged_gate_up_scales.data);
        down_codes       = static_cast<std::uint8_t*>(scratch.staged_down_codes.data);
        down_scales      = static_cast<std::uint8_t*>(scratch.staged_down_scales.data);
        gate_up_divisors = static_cast<float*>(scratch.staged_gate_up_divisors.data);
        down_divisors    = static_cast<float*>(scratch.staged_down_divisors.data);
    }

    // ---- copy only what is missing ----------------------------------------------------------------
    const double t_stage0 = host_now_ms();
    // Pinned staging for the per-slot divisors. Indexed by SLOT, so it is sized by the pool (and by
    // the path count in the no-cache branch, where slots == 0 but offsets still count up to paths).
    //
    // P0 (2026-10-16, alignment round 2): with ONE POOL the slot index is GLOBAL (see the plane-base
    // note above the destination planes: `base = pooled ? 0 : index * slot_count`), and
    // `global_pick_victim` hands out slots across all kGlobalPoolLayers * slots of them -- so the
    // old `max(slots, kStagedExpertSlots)` bound was indexed by a number up to 48 times larger.
    // Two distinct failures followed, both silent because the pooled path has no byte verification
    // (it refuses VERIFY: "not implemented for the global pool yet"):
    //   * offsets in [slots, 2*slots) wrote the gate-up value INTO the down plane's own entries
    //     (same address), so a down divisor still queued for an earlier slot could be overwritten
    //     before its asynchronous copy read it -- a wrong weight scale, i.e. exactly the failure
    //     mode the verify comment at the top of this file says this project has paid for before;
    //   * offsets >= 2*slots ran past the pinned allocation itself (up to ~28 KiB at slots=144).
    // Non-pooled (and the OFF arms, which never reach a pooled slot) compute the identical value,
    // so the shipped behaviour is byte-for-byte unchanged; only the pooled index bound is corrected.
    const std::int32_t divisor_slots =
        pooled ? kGlobalPoolLayers * slots : std::max<std::int32_t>(slots, kStagedExpertSlots);
    float* const staged_divisor_planes = staging_control_buffers().divisor_plane(divisor_slots);
    float* const staged_gate_up_divisors = staged_divisor_planes;
    float* const staged_down_divisors    = staged_divisor_planes + divisor_slots;

    std::uint64_t bytes_moved = 0;

    // ---- P3-1: one batched call instead of six per miss -------------------------------------------
    // WHY: every miss issues SIX separate H2D operations -- four planes plus two FOUR-BYTE divisors --
    // and the divisor pair is the worst bytes-per-op ratio in the whole staging path. The community's
    // per-step ledger is explicit that the operation COUNT is decoupled from the BYTES (65 us per
    // small H2D op), so an operation that moves 4 bytes costs the same as one that moves 1.6 MB.
    // Batched, those six become one. The BYTES are unchanged, so `h2d_mib`/`moved_mib` MUST come out
    // bit-identical and only `h2d_ops` is allowed to move -- that equality is the regression
    // invariant, not an approximate target.
    //
    // The contract was read out of the CUDA 13.3 headers before this was written:
    //   * cuda_runtime_api.h:6472-6473  "no ordering guarantees for copies within a batch, specifying
    //     any dependent copies within a batch will result in undefined behavior" -- a miss's six
    //     copies are independent, so they may share a batch.
    //   * cuda_runtime_api.h:6495-6497  "Each memcpy operation in the batch must have a valid
    //     cudaMemcpyAttributes corresponding to it ... otherwise the API will return
    //     cudaErrorInvalidValue" -- so numAttrs CANNOT be 0.
    //   * cuda_runtime_api.h:6520  the stream must not be the legacy NULL stream.
    // srcAccessOrder is the conservative Stream (driver_types.h:2449): our sources are the artifact
    // mapping and a REUSED pinned buffer, not the ephemeral stack variables DuringApiCall is for.
    //
    // There is deliberately NO silent fallback on a runtime failure: it is CUDA_CHECKed, so a batch
    // that cannot be issued fails loudly. The fallback is the switch itself -- OFF is the shipped
    // per-copy loop, which stays in the tree verbatim below.
#if CUDART_VERSION >= 12080
    const bool dma_batch = dma_batch_enabled();
#else
    // The batch API needs 12.08+. This tree builds with 13.3, but saying so explicitly beats letting
    // "it compiled" imply the requirement.
    const bool dma_batch = false;
#endif

    if (dma_batch) {
#if CUDART_VERSION >= 12080
        std::vector<void*> batch_dst;
        std::vector<const void*> batch_src;
        std::vector<std::size_t> batch_sizes;
        const std::size_t per_miss = 6; // 4 planes + 2 divisors
        batch_dst.reserve(pending_copy.size() * per_miss);
        batch_src.reserve(pending_copy.size() * per_miss);
        batch_sizes.reserve(pending_copy.size() * per_miss);

        for (const std::int32_t slot : pending_copy) {
            const int expert = pooled
                                   ? static_cast<int>(pool->key_of_slot[static_cast<std::size_t>(slot)] %
                                                      kExpertsPerBank)
                                   : layer_slot_expert[static_cast<std::size_t>(slot)];
            if (expert < 0 || expert >= 512) {
                throw std::logic_error("Flash-Next staging tried to fill a slot with no expert");
            }
            const auto source = static_cast<std::uint64_t>((expert + sabotage) % gate_up.experts);
            const auto offset = static_cast<std::uint64_t>(slot);

            batch_dst.push_back(gate_up_codes + offset * kStagedGateUpCodeBytes);
            batch_src.push_back(gate_up.codes + source * gate_up.code_bytes_per_expert);
            batch_sizes.push_back(static_cast<std::size_t>(kStagedGateUpCodeBytes));
            batch_dst.push_back(gate_up_scales + offset * kStagedGateUpScaleBytes);
            batch_src.push_back(gate_up.scales + source * gate_up.scale_bytes_per_expert);
            batch_sizes.push_back(static_cast<std::size_t>(kStagedGateUpScaleBytes));
            batch_dst.push_back(down_codes + offset * kStagedDownCodeBytes);
            batch_src.push_back(down.codes + source * down.code_bytes_per_expert);
            batch_sizes.push_back(static_cast<std::size_t>(kStagedDownCodeBytes));
            batch_dst.push_back(down_scales + offset * kStagedDownScaleBytes);
            batch_src.push_back(down.scales + source * down.scale_bytes_per_expert);
            batch_sizes.push_back(static_cast<std::size_t>(kStagedDownScaleBytes));
            bytes_moved += kStagedGateUpCodeBytes + kStagedGateUpScaleBytes + kStagedDownCodeBytes +
                           kStagedDownScaleBytes;

            // The divisor staging MUST be finished before the batch is issued, because the batch
            // reads these addresses. That is why it sits here and not after the call.
            staged_gate_up_divisors[offset] =
                gate_up.weight_scale_divisors[static_cast<std::size_t>(expert)];
            staged_down_divisors[offset] =
                down.weight_scale_divisors[static_cast<std::size_t>(expert)];
            batch_dst.push_back(gate_up_divisors + offset);
            batch_src.push_back(staged_gate_up_divisors + offset);
            batch_sizes.push_back(sizeof(float));
            batch_dst.push_back(down_divisors + offset);
            batch_src.push_back(staged_down_divisors + offset);
            batch_sizes.push_back(sizeof(float));
        }

        const std::size_t batch_count = batch_dst.size();
        if (batch_count > 0) {
            // ONE attribute applied to every copy in the batch, which is exactly what the 6-argument
            // overload at cuda_runtime.h:985 does; written out longhand so the requirement that
            // numAttrs is 1 (not 0) is visible in the source.
            cudaMemcpyAttributes batch_attr{};
            batch_attr.srcAccessOrder     = cudaMemcpySrcAccessOrderStream;
            std::size_t batch_attr_idx[1] = {0};
            CUDA_CHECK(cudaMemcpyBatchAsync(batch_dst.data(), batch_src.data(), batch_sizes.data(),
                                            batch_count, &batch_attr, batch_attr_idx, 1, stream));
            auto& prof = stage_profile();
            // ONE operation. The byte total is computed by the same expression as the unbatched
            // path, so `h2d_mib` can be compared digit for digit between the two arms.
            prof.note_h2d(static_cast<std::size_t>(bytes_moved));
            ++prof.dma_batches;
            prof.dma_ops_replaced += static_cast<unsigned long long>(batch_count);
        }
#endif
    } else {
    for (const std::int32_t slot : pending_copy) {
        // With one pool `layer_slot_expert` (sized 512, indexed by a per-layer slot) cannot carry the
        // answer, so the expert is recovered from the key plane instead. The key's low bits ARE the
        // expert id, which is the same convention the lookup uses.
        const int expert = pooled
                               ? static_cast<int>(pool->key_of_slot[static_cast<std::size_t>(slot)] %
                                                  kExpertsPerBank)
                               : layer_slot_expert[static_cast<std::size_t>(slot)];
        if (expert < 0 || expert >= 512) {
            throw std::logic_error("Flash-Next staging tried to fill a slot with no expert");
        }
        // NINFER_FLASH_NEXT_EXPERTS_SABOTAGE stages the neighbouring expert to prove the verifier
        // below can go red.
        const auto source = static_cast<std::uint64_t>((expert + sabotage) % gate_up.experts);
        const auto offset = static_cast<std::uint64_t>(slot);

        CUDA_CHECK(cudaMemcpyAsync(gate_up_codes + offset * kStagedGateUpCodeBytes,
                                   gate_up.codes + source * gate_up.code_bytes_per_expert,
                                   kStagedGateUpCodeBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(gate_up_scales + offset * kStagedGateUpScaleBytes,
                                   gate_up.scales + source * gate_up.scale_bytes_per_expert,
                                   kStagedGateUpScaleBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(down_codes + offset * kStagedDownCodeBytes,
                                   down.codes + source * down.code_bytes_per_expert,
                                   kStagedDownCodeBytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(down_scales + offset * kStagedDownScaleBytes,
                                   down.scales + source * down.scale_bytes_per_expert,
                                   kStagedDownScaleBytes, cudaMemcpyHostToDevice, stream));
        bytes_moved += kStagedGateUpCodeBytes + kStagedGateUpScaleBytes + kStagedDownCodeBytes +
                       kStagedDownScaleBytes;
        // FOUR separate H2D operations per expert. The community's per-step ledger is explicit that
        // the operation COUNT is decoupled from the bytes (65 us per small H2D op), so count ops.
        {
            auto& prof = stage_profile();
            prof.note_h2d(kStagedGateUpCodeBytes);
            prof.note_h2d(kStagedGateUpScaleBytes);
            prof.note_h2d(kStagedDownCodeBytes);
            prof.note_h2d(kStagedDownScaleBytes);
        }

        // Divisors: per slot, the divisor of THAT slot's expert. The kernels index the divisor plane
        // with the same rewritten id they index the codes with, so a straight plane copy would hand
        // slot s the divisor of whichever expert happens to carry the id s.
        //
        // Staged through pinned memory on purpose. The previous form took the address of a stack local
        // (&gate_up_divisor), i.e. a PAGEABLE source, and a pageable H2D copy makes the driver
        // synchronise the stream before it is initiated -- inside this loop that drained the pinned
        // plane copies issued immediately above, once per miss. Same bytes to the same destinations.
        staged_gate_up_divisors[offset] =
            gate_up.weight_scale_divisors[static_cast<std::size_t>(expert)];
        staged_down_divisors[offset] = down.weight_scale_divisors[static_cast<std::size_t>(expert)];
        CUDA_CHECK(cudaMemcpyAsync(gate_up_divisors + offset, staged_gate_up_divisors + offset,
                                   sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(down_divisors + offset, staged_down_divisors + offset,
                                   sizeof(float), cudaMemcpyHostToDevice, stream));
        // Two more ops per miss, 4 bytes each. This is the worst bytes-per-op ratio in the whole
        // staging path, and exactly the case the 65 us/op figure is about.
        {
            auto& prof = stage_profile();
            prof.note_h2d(sizeof(float));
            prof.note_h2d(sizeof(float));
        }
    }
    } // end of the unbatched arm (the shipped per-copy loop, kept verbatim as the OFF arm)
    stage_profile().stage_ms += host_now_ms() - t_stage0;

    // Rewrite the router output in place: expert id -> slot. The decode kernels read ids only as a
    // bank index (moe_kernels.cu:96 and :312) and alpha carries the weights, so this is the whole of
    // the indirection. SKIPPED on the ring path: there the raw expert id IS the ring slot, and
    // scratch.ids still holds exactly what the router wrote -- copying anything here would break it.
    if (!ring_engaged) {
        CUDA_CHECK(cudaMemcpyAsync(scratch.ids.data, host_slots.data(),
                                   static_cast<std::size_t>(paths) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        stage_profile().note_h2d(static_cast<std::size_t>(paths) * sizeof(std::int32_t));
    }

    // Ledger probe (diagnostic only, no behavior change): this record closes the span that started at
    // the previous ledger event, so the table's "MoE: grouping" row reports the HOST-SIDE cost of
    // one layer's expert staging -- router-ids D2H, LRU/Least-Stale slot choice, and issuing the
    // copies. The copies themselves stay asynchronous, so their transfer time shows up in the span
    // AFTER this record (i.e. in whatever kernel has to wait for them). Splitting host cost from
    // transfer cost is the whole point: the two need opposite fixes.
    stage_ledger_record(stream, FlashNextStageId::MoE_Grouping);

    if (staging_verify_enabled() && pooled && !ring_engaged) {
        throw std::runtime_error(
            "Flash-Next staging verification is not implemented for the global pool yet: it compares "
            "each layer's own slot table, and with one pool a slot belongs to no single layer. Run "
            "with NINFER_FLASH_NEXT_EXPERTS_GLOBAL_POOL unset, or turn "
            "NINFER_FLASH_NEXT_EXPERTS_VERIFY off. Refused loudly rather than verifying a plane that "
            "no longer means what the check assumes.");
    }

    if (staging_verify_enabled() && !ring_engaged) {
        const std::array<StagingPlane, 4> planes = {{
            {gate_up_codes, gate_up.code_bytes_per_expert, gate_up.codes},
            {gate_up_scales, gate_up.scale_bytes_per_expert, gate_up.scales},
            {down_codes, down.code_bytes_per_expert, down.codes},
            {down_scales, down.scale_bytes_per_expert, down.scales},
        }};
        const std::array<float*, 2> divisor_planes = {{gate_up_divisors, down_divisors}};
        verify_staged_bytes(gate_up, down, planes, divisor_planes, layer_slot_expert, used_slots,
                            used_count, sabotage, stream);
    }

    {
        auto& prof = stage_profile();
        ++prof.layers;
        prof.host_ms += host_now_ms() - t_enter_ms;
    }

    if (pool != nullptr) {
        pool->bytes_moved += bytes_moved;
        const bool report_now = (++pool->calls % 48 == 0);

        // ---- why the `calls % 48` gate is wrong under prefill, and what it silently disabled --------
        // The original intent, per its own comment, was "once per token: start a new pass (what
        // Least-Stale compares against), re-elect this layer's hot (never-evicted) slots from the live
        // counters". In DECODE that intent holds by accident: 48 calls really is one token, one call
        // per layer, so each firing lands on a different layer and every layer gets its hot set
        // re-elected once per token.
        //
        // It does NOT hold under prefill, and the failure is silent. Measured from the route dump
        // (scripts/wave_distinct.py --order): the executor is LAYER-MAJOR, layer L running its 16
        // eight-token slices before layer L+1, so 48 calls is only THREE layers. The gate therefore
        // touches exactly the 16 layers with index 2 mod 3 and never the other 32 -- whose slot_hot
        // stays 0 for the entire run, because ensure_slot_pool paints every table's slot_hot with 0
        // and nothing else ever sets it. HOT is then a no-op for two thirds of the model.
        //
        // That is the explanation for a measurement we had recorded as a falsification: the notes
        // predicted EXPERTS_HOT would buy -25..-40%, and EXPERTS_HOT=0 measured only +1.8% WORSE. With
        // two thirds of the layers unable to respond at all, a knob cannot show more than a third of
        // its effect -- the prediction was not falsified, it was never fully tested.
        //
        // The fix counts LAYER TRANSITIONS and splits the two actions the old gate conflated:
        //   * a new layer visit -> re-elect THAT layer's hot set (once per layer per sweep, both modes)
        //   * a WRAP of the layer order -> advance `pass`, i.e. a completed sweep over the layers.
        // A wrap is detected by the table index going backwards, so it needs no knowledge of how many
        // layer tables exist (48 text layers + 1 MTP = 49) and cannot drift when that count changes.
        if (hot_fix_enabled()) {
            static std::int32_t s_last_table = -1;
            if (table_index != s_last_table) {
                refresh_hot_any(*pool, *table, table_index, slots);
                if (s_last_table >= 0 && table_index < s_last_table) { ++pool->pass; }
                s_last_table = table_index;
            }
        } else if (report_now) {
            ++pool->pass;
            refresh_hot_any(*pool, *table, table_index, slots);
        }

        if (report_now) {
            // One line per token, community-shaped counters. Phase IV adds the two service-side
            // fields: gpu_served = hits + (misses - cpu_served) -- the paths the device actually
            // computed -- and cpu_served = the miss paths handed to the CPU engine. They sum to
            // hits + misses by construction; a future change that recorded a served path as a hit
            // would break that identity HERE instead of inflating hit_rate quietly.
            std::fprintf(stderr,
                         "flash_next_experts_cache calls=%llu hits=%llu misses=%llu evictions=%llu "
                         "hit_rate=%.1f%% moved_mib=%.0f slots=%d hot=%d dropped=%llu F=%.2f gate<=%d "
                         "fpre=%.2f subst=%d damp=%.2f gpu_served=%llu cpu_served=%llu\n",
                         static_cast<unsigned long long>(pool->calls),
                         static_cast<unsigned long long>(pool->hits),
                         static_cast<unsigned long long>(pool->misses),
                         static_cast<unsigned long long>(pool->evictions),
                         100.0 * static_cast<double>(pool->hits) /
                             static_cast<double>(pool->hits + pool->misses),
                         static_cast<double>(pool->bytes_moved) / (1024.0 * 1024.0), pool->slots,
                         hot_slot_count(slots),
                         static_cast<unsigned long long>(pool->dropped),
                         static_cast<double>(moe_drop_fraction()), moe_drop_max_tokens(),
                         static_cast<double>(moe_drop_fraction_prefill()),
                         moe_subst_enabled() ? 1 : 0,
                         static_cast<double>(moe_subst_damp()),
                         static_cast<unsigned long long>(pool->hits +
                                                         (pool->misses - pool->cpu_served)),
                         static_cast<unsigned long long>(pool->cpu_served));
            // P2a/P2b self-证 line: the round boundary, the route log, and how much work LFRU
            // actually did. A knob that produces no line is a knob nobody can tell from an unused
            // one -- the failure this tree has recorded three times.
            {
                std::int32_t logged = 0;
                for (std::int32_t t = 0; t < kMaxLayerTables; ++t) {
                    if (pool->route_log.rounds[pool->route_log.head][t].table_index >= 0) { ++logged; }
                }
                std::fprintf(stderr,
                             "flash_next_experts_round round=%lld closed=%llu head=%d "
                             "logged_layers=%d round_flag=%d global_pool=%d pooled=%d lfru=%d "
                             "lfru_rankings=%llu lfru_fallbacks=%llu lfru_halvings=%llu pass=%d "
                             "live_cols=%d live_calls=%llu live_no_pool=%llu "
                             "live_commits=%llu live_stales=%llu "
                             "live_credited=%llu live_skipped=%llu\n",
                             static_cast<long long>(pool->round),
                             static_cast<unsigned long long>(pool->route_log.closed),
                             pool->route_log.head, logged, round_tracking_enabled() ? 1 : 0,
                             pool->global_ready ? 1 : 0, pooled ? 1 : 0, lfru_enabled() ? 1 : 0,
                             static_cast<unsigned long long>(pool->lfru_rankings),
                             static_cast<unsigned long long>(pool->lfru_fallbacks),
                             static_cast<unsigned long long>(pool->lfru_halvings), pool->pass,
                             live_columns_enabled() ? 1 : 0,
                             static_cast<unsigned long long>(live_columns().calls),
                             static_cast<unsigned long long>(live_columns().no_pool),
                             static_cast<unsigned long long>(live_columns().commits),
                             static_cast<unsigned long long>(live_columns().stales),
                             static_cast<unsigned long long>(live_columns().credited),
                             static_cast<unsigned long long>(live_columns().skipped));
            }
            // S5: one profile line per token, right after the cache counters, same cadence.
            stage_profile().report_and_reset();
            // Phase IV W3 self-证 line: same cadence as the counters above, and ONLY when the
            // switch is on -- the OFF arm must print zero new lines. `r` = jobs / landed_records:
            // how many times, on average, one landed record was served (the reuse the W0-b(A)
            // cache converts into saved scale/decode work). Block-6 W1/W2: `scope` now names what
            // the LAST committed round served -- `forward` when the decode-width compute face ran
            // (fwd_* counters carry its evidence; fwd_applied is the apply side's acceptance of
            // the partial), `landing_unpack` otherwise.
            if (cpu_serve_share().enabled) {
                // Pin arguments are passed here too: whichever call builds the singleton first
                // (a report point may precede the first serve candidate) must carry the config.
                const cpu_expert::CpuServeService::Snapshot s =
                    cpu_expert::CpuServeService::instance(cpu_serve_share().pin,
                                                          cpu_serve_share().pin_first)
                        .snapshot();
                const double reuse =
                    s.landed_records > 0
                        ? static_cast<double>(s.jobs) / static_cast<double>(s.landed_records)
                        : 0.0;
                const char* const serve_scope = s.scope_forward ? "forward" : "landing_unpack";
                std::fprintf(stderr,
                             "flash_next_cpu_expert_serve requests=%llu answered=%llu pending=%llu "
                             "jobs=%llu landed=%llu landed_mib=%.1f r=%.2f cache_hits=%llu "
                             "cache_misses=%llu cache_evictions=%llu failed=%llu slot_fallback=%llu "
                             "stalls=%llu errors=%llu last_status=%u d=%d mode=%s workers=%d "
                             "max_jobs=%d cap_records=%zu scope=%s fwd_jobs=%llu fwd_elements=%llu "
                             "fwd_applied=%llu fwd_stalls=%llu\n",
                             static_cast<unsigned long long>(s.requests),
                             static_cast<unsigned long long>(s.answered),
                             static_cast<unsigned long long>(s.published - s.answered),
                             static_cast<unsigned long long>(s.jobs),
                             static_cast<unsigned long long>(s.landed_records),
                             static_cast<double>(s.landed_bytes) / (1024.0 * 1024.0), reuse,
                             static_cast<unsigned long long>(s.cache.hits),
                             static_cast<unsigned long long>(s.cache.misses),
                             static_cast<unsigned long long>(s.cache.evictions),
                             static_cast<unsigned long long>(s.jobs_failed),
                             static_cast<unsigned long long>(pool->serve_fallback),
                             static_cast<unsigned long long>(s.stalls),
                             static_cast<unsigned long long>(s.errors), s.last_status,
                             (cpu_serve_share().adaptive
                                  ? cpu_expert::adaptive_serve_d(cpu_serve_share().d,
                                                                 cpu_serve_share().t_us)
                                  : cpu_serve_share().d),
                             cpu_serve_share().mode,
                             cpu_expert::kCpuServeWorkers, cpu_expert::kCpuServeMaxJobs,
                             cpu_expert::kCpuServeBiasCap, serve_scope,
                             static_cast<unsigned long long>(s.fwd_jobs),
                             static_cast<unsigned long long>(s.fwd_elements),
                             static_cast<unsigned long long>(s.fwd_applied),
                             static_cast<unsigned long long>(s.fwd_stalls));
                // ③ 按表定比例 (supervisor G2/T3): the MEASURED split, next to the requested table
                // value. A NEW line name, because the frozen serve line above keeps its field set --
                // the precedent this file already follows for `serve_prof`. Reachable only while the
                // master switch is on, so the OFF arm prints nothing new. `frac_link_actual` is what
                // actually took the PCIe copy, not the nominal 1/d: with `d=3` a call whose distinct
                // misses are fewer than d keeps the first miss on the link and can serve nothing, so
                // small-width calls drift toward the link, and only this line can show by how much.
                {
                    const std::int32_t d_now = (cpu_serve_share().adaptive
                                                    ? cpu_expert::adaptive_serve_d(
                                                          cpu_serve_share().d, cpu_serve_share().t_us)
                                                    : cpu_serve_share().d);
                    const std::uint64_t link = pool->serve_link_paths;
                    const std::uint64_t cpu  = pool->cpu_served;
                    const std::uint64_t tot  = link + cpu;
                    const double nominal = (d_now > 0) ? (1.0 / static_cast<double>(d_now)) : 1.0;
                    const double actual  = (tot > 0)
                                               ? (static_cast<double>(link) / static_cast<double>(tot))
                                               : 0.0;
                    std::fprintf(stderr,
                                 "flash_next_cpu_expert_split d=%d mode=%s frac_req=%.4f "
                                 "frac_req_set=%d frac_nominal=%.4f frac_link_actual=%.4f "
                                 "frac_cpu_actual=%.4f link_paths=%llu cpu_paths=%llu split_paths=%llu "
                                 "diverged=%d\n",
                                 d_now, cpu_serve_share().mode,
                                 cpu_serve_share().frac_set ? cpu_serve_share().frac : 0.0,
                                 cpu_serve_share().frac_set ? 1 : 0, nominal, actual, 1.0 - actual,
                                 static_cast<unsigned long long>(link),
                                 static_cast<unsigned long long>(cpu),
                                 static_cast<unsigned long long>(tot),
                                 (tot > 0 && std::fabs(actual - nominal) > 0.02) ? 1 : 0);
                }
                // Block-8 item 7: segmented timing, env-gated -- a NEW line name; the frozen
                // serve line above keeps its exact field set.
                if (cpu_serve_prof_enabled()) {
                    std::fprintf(stderr,
                                 "flash_next_cpu_expert_serve_prof rounds=%llu run_ms=%.1f "
                                 "reduce_ms=%.1f apply_wait_ms=%.1f fwd_fused=%llu sw=%d poke=%d\n",
                                 static_cast<unsigned long long>(s.requests),
                                 static_cast<double>(s.prof_run_ns) / 1.0e6,
                                 static_cast<double>(s.prof_reduce_ns) / 1.0e6,
                                 static_cast<double>(s.prof_wait_ns) / 1.0e6,
                                 static_cast<unsigned long long>(s.fwd_fused),
                                 cpu_serve_streamwait_enabled() ? 1 : 0,
                                 cpu_serve_poke_enabled() ? 1 : 0);
                }
            }
        }
    }

    MoeWeights staged = weights;
    if (ring_engaged) {
        // CONSUME side: the staged views ARE the ring planes. `experts` is the full bank, so every
        // raw expert id is a valid index into the ring; the divisor planes travel with the ring
        // (the fill copied them whole), so the kernels' divisor[slot] reads land on the right
        // expert's divisor with no per-slot gather -- which is exactly why the id rewrite can be
        // skipped above. on_host=false: a device kernel must never dereference these.
        staged.expert_gate_up = Nvfp4ExpertBankView{
            .codes                  = ring_planes.gu_codes,
            .scales                 = ring_planes.gu_scales,
            .weight_scale_divisors  = ring_planes.gu_divisors,
            .experts                = ring_planes.experts,
            .rows                   = gate_up.rows,
            .columns                = gate_up.columns,
            .code_bytes_per_expert  = gate_up.code_bytes_per_expert,
            .scale_bytes_per_expert = gate_up.scale_bytes_per_expert,
            .on_host                = false,
        };
        staged.expert_down = Nvfp4ExpertBankView{
            .codes                  = ring_planes.dn_codes,
            .scales                 = ring_planes.dn_scales,
            .weight_scale_divisors  = ring_planes.dn_divisors,
            .experts                = ring_planes.experts,
            .rows                   = down.rows,
            .columns                = down.columns,
            .code_bytes_per_expert  = down.code_bytes_per_expert,
            .scale_bytes_per_expert = down.scale_bytes_per_expert,
            .on_host                = false,
        };
        return staged;
    }
    staged.expert_gate_up = Nvfp4ExpertBankView{
        .codes                  = reinterpret_cast<const std::byte*>(gate_up_codes),
        .scales                 = reinterpret_cast<const std::byte*>(gate_up_scales),
        .weight_scale_divisors  = gate_up_divisors,
        .experts                = slots > 0
                                      ? (pooled ? slots * kGlobalPoolLayers : slots)
                                      : used_count,
        .rows                   = gate_up.rows,
        .columns                = gate_up.columns,
        .code_bytes_per_expert  = gate_up.code_bytes_per_expert,
        .scale_bytes_per_expert = gate_up.scale_bytes_per_expert,
        .on_host                = false,
    };
    staged.expert_down = Nvfp4ExpertBankView{
        .codes                  = reinterpret_cast<const std::byte*>(down_codes),
        .scales                 = reinterpret_cast<const std::byte*>(down_scales),
        .weight_scale_divisors  = down_divisors,
        .experts                = slots > 0
                                      ? (pooled ? slots * kGlobalPoolLayers : slots)
                                      : used_count,
        .rows                   = down.rows,
        .columns                = down.columns,
        .code_bytes_per_expert  = down.code_bytes_per_expert,
        .scale_bytes_per_expert = down.scale_bytes_per_expert,
        .on_host                = false,
    };
    return staged;
}

// ================================================================================================
// Block-6 W2 -- the write-back's second step: after the MoE kernels have written `output`, add the
// CPU-served experts' REAL contribution (the serve team's partial) into it.
//
// WHY THE CONTRIBUTION IS NOT CARRIED BY `host_alpha` AT THE R1-FORM SITE: the GPU kernel computes
// alpha * E(x, slot[path]) and a served path's slot is a LEGAL slot holding ANOTHER expert's
// weights -- a non-zero alpha there would add the wrong expert's output, and the CPU partial would
// then double-count it. So the GPU side keeps the R1-drop form (legal slot, alpha = 0) and the
// real contribution arrives HERE, as the sum over served paths of alpha * E_served(x) per token --
// exactly the shape and the position the Infernix reference feeds through its residual add
// (hybrid_experts.cpp: wait_completed -> cpu_sum H2D -> residual_add).
//
// BOUNDED: a partial that is not fully answered is NOT added (reading the team's accumulator while
// it is still being written is the one failure this must never have) and is counted loudly in
// fwd_stalls. No-op -- one static read, zero copies -- unless the forward face was armed for the
// round the staging just committed.
// ================================================================================================
void flash_next_cpu_serve_apply(Tensor& output, const FlashNextMoeWorkspace& scratch, int tokens,
                                cudaStream_t stream) {
    if (!cpu_serve_share().enabled) { return; }
    // Same pin config discipline as every other construction point: first call wins.
    auto& service = cpu_expert::CpuServeService::instance(cpu_serve_share().pin,
                                                          cpu_serve_share().pin_first);
    if (!service.take_forward_round()) { return; }
    if (scratch.cpu_serve_partial.data == nullptr || tokens < 1 ||
        tokens > cpu_expert::kCpuServeForwardCols) {
        service.note_forward_stall();
        return;
    }
    // fp32 partial -> bf16: DONE by the serve thread before it answered (block-8 item 3a), so both
    // paths below H2D a finished slab. Block-8 item 7: the slab is per ring slot -- the H2D reads
    // it while the host moves on, and a later round must not rewrite an in-flight copy's source.
    const std::size_t elements =
        static_cast<std::size_t>(2'560) * static_cast<std::size_t>(tokens);
    auto& ctl = staging_control_buffers();
    const std::size_t bf16_slab_stride =
        static_cast<std::size_t>(cpu_expert::kCpuServeForwardCols) * 2'560;

    // ---- Block-8 item 3a: the stream-wait path (env-gated; every failure falls back LOUDLY to
    // the blocking path below -- the safe direction) ---------------------------------------------
    if (cpu_serve_streamwait_enabled()) {
        service.arm_stream_watchdog(); // once; the monitor only exists where this channel does
        const std::uint32_t seq = service.last_committed_seq();
        const std::size_t slot =
            static_cast<std::size_t>((seq - 1) % cpu_expert::kCpuServeRingSlots);
        const CuStreamWaitValue32Fn wait_fn = cu_serve_stream_wait_value32();
        const std::uintptr_t flag_dev = seq > 0 ? serve_done_flag_device_ptr(slot) : 0;
        if (seq > 0 && wait_fn != nullptr && flag_dev != 0) {
            // The stream waits for the completion word; the H2D then reads the slab the serve
            // thread wrote before storing that word (release). The host does NOT block here.
            if (wait_fn(reinterpret_cast<void*>(stream), flag_dev, seq, kCuStreamWaitValueEq) ==
                0) {
                const std::uint16_t* const bf16_slab =
                    ctl.serve_partial_bf16.data() + slot * bf16_slab_stride;
                CUDA_CHECK(cudaMemcpyAsync(scratch.cpu_serve_partial.data, bf16_slab,
                                           elements * sizeof(std::uint16_t),
                                           cudaMemcpyHostToDevice, stream));
                stage_profile().note_h2d(elements * sizeof(std::uint16_t));
                ops::residual_add(scratch.cpu_serve_partial, output, stream);
                service.note_forward_applied();
                // Block-8 item 3b: the optional driver poke (a WDDM submission trigger candidate;
                // its marginal is measured in the concentrated test).
                if (cpu_serve_poke_enabled()) { (void)cudaStreamQuery(stream); }
                return;
            }
            static std::atomic<bool> wait_err_warned{false};
            bool expected = false;
            if (wait_err_warned.compare_exchange_strong(expected, true)) {
                std::fprintf(stderr,
                             "flash_next_cpu_expert_serve streamwait: cuStreamWaitValue32 "
                             "refused (seq=%u); applying the blocking await\n",
                             seq);
            }
        }
        // The entry / device-pointer reason lines already fired once inside their helpers.
    }

    // ---- the blocking path (block-8 item 7 semantics, unchanged wire shape) ---------------------
    if (!service.await_last_round(cpu_expert::kCpuServeApplyWaitMs)) {
        service.note_forward_stall();
        return;
    }
    if (service.forward_partial() == nullptr) { // pre-3a integrity check, kept
        service.note_forward_stall();
        return;
    }
    const std::uint16_t* const bf16_slab =
        ctl.serve_partial_bf16.data() +
        static_cast<std::size_t>(service.last_applied_slot()) * bf16_slab_stride;
    CUDA_CHECK(cudaMemcpyAsync(scratch.cpu_serve_partial.data, bf16_slab,
                               elements * sizeof(std::uint16_t), cudaMemcpyHostToDevice, stream));
    stage_profile().note_h2d(elements * sizeof(std::uint16_t));
    ops::residual_add(scratch.cpu_serve_partial, output, stream);
    service.note_forward_applied();
}

// P5-1 entry point. Declared in expert_staging.h; see the long note beside LiveColumns above for
// why the credit is deferred instead of retracted.
void flash_next_note_live_columns(std::int32_t accepted_count) {
    if (!live_columns_enabled()) {
        return;
    }
    LiveColumns& lc = live_columns();
    // Counted BEFORE any early return: a hook that returns silently because the pool is absent must
    // still be distinguishable from a hook that was never reached.
    ++lc.calls;
    SlotPool& p = slot_pool();
    // A count can only mean anything against a resident frequency plane; anything else is a caller
    // that got the order wrong, so it is ignored rather than written to. `global_ready` and the plane
    // size are both checked by commit_live_columns as well -- belt and braces, because a write past
    // the end of that vector would be heap corruption, not a wrong number.
    if (!p.global_ready) {
        ++lc.no_pool;
        return;
    }
    commit_live_columns(p, accepted_count, /*stale=*/false);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
