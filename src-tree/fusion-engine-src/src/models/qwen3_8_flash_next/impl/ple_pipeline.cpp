#include "models/qwen3_8_flash_next/impl/ple_pipeline.h"

#include "artifact/file_io.h" // block-7 alignment ④: the direct-read handle type
#include "core/arena.h"
#include "models/qwen3_8_flash_next/impl/ple_decode_kernels.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

constexpr std::size_t kPleOutputWidth          = 16 * kPleRowWidth;
constexpr std::size_t kPleTokenBytes           = kPleOutputWidth * sizeof(std::uint16_t);
constexpr std::size_t kPleCodesPerRow          = 80;
constexpr std::size_t kPleScalesPerRow         = 20;
constexpr std::size_t kPleCompressedRowBytes   = kPleCodesPerRow + kPleScalesPerRow; // 100
constexpr std::size_t kPleCompressedTokenBytes = 16 * kPleCompressedRowBytes;       // 1600

enum class SlotState : std::uint8_t { Idle, Gathering, Copying };

// Block-7 alignment ④: read once per process, like every other switch in this line.
bool ple_io_direct_requested() {
    static const bool requested = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_PLE_IO_DIRECT");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return requested;
}

// Block-7 alignment ④: one direct handle serves ONE outstanding read at a time. The direct handle
// is opened FILE_FLAG_OVERLAPPED with hEvent = NULL, and concurrent outstanding reads through such
// a handle make ReadFile/GetOverlappedResult misreport completion (measured by _ple-dirprobe:
// 7 of 8 concurrent 4 MiB reads through one shared handle returned 0 bytes with NO error; the same
// reads with a per-request event, or through separate handles, all succeeded). So every gather
// worker thread opens its OWN InputFile, lazily, on first use -- one outstanding read per handle,
// the same usage the loader has always exercised. Workers are long-lived pool threads, so the open
// cost lands once per worker.
const artifact::InputFile* worker_direct_file(const std::string& path) {
    struct Bound {
        std::unique_ptr<artifact::InputFile> file;
        std::string path;
    };
    static thread_local Bound bound;
    if (bound.file == nullptr || bound.path != path) {
        bound.file = std::make_unique<artifact::InputFile>(std::filesystem::path(path));
        bound.path = path;
    }
    return bound.file.get();
}

} // namespace

struct PleGatherPipeline::Slot {
    Slot(DeviceContext& device, std::size_t host_bytes, std::size_t dev_bytes)
        : buffer(host_bytes), device_compressed(dev_bytes), completion(device), ordering(device) {}

    PinnedHostBuffer buffer;
    DeviceBuffer device_compressed;
    CudaCompletionEvent completion;
    // Orders the H2D copy after everything already enqueued on the compute stream: the startup
    // zero-fill of the round tensors and the previous round's consumers of `output`. Without it
    // the transfer-stream copy can land before an in-flight memset (WAW) or before the previous
    // PLE layer has read the buffer (WAR).
    CudaCompletionEvent ordering;
    std::vector<std::future<void>> work;
    std::uint64_t generation = 0;
    SlotState state          = SlotState::Idle;
};

PleGatherPipeline::Ticket::Ticket(PleGatherPipeline* owner, std::size_t slot,
                                  std::uint64_t generation, std::size_t tokens) noexcept
    : owner_(owner), slot_(slot), generation_(generation), tokens_(tokens) {}

PleGatherPipeline::PleGatherPipeline(PleTableView table, DeviceContext& device,
                                     std::size_t max_tokens, std::size_t slot_count,
                                     std::uint32_t worker_threads)
    : table_(std::move(table)), device_(device), max_tokens_(max_tokens),
      workers_(worker_threads, slot_count * max_tokens),
      fixed_host_buffer_(max_tokens * kPleTokenBytes),
      direct_enabled_(ple_io_direct_requested()) {
    if (max_tokens == 0 || slot_count == 0) {
        throw std::invalid_argument("PLE gather pipeline capacity must be nonzero");
    }
    slots_.reserve(slot_count);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
        slots_.push_back(std::make_unique<Slot>(
            device_, max_tokens_ * kPleCompressedTokenBytes,
            max_tokens_ * kPleCompressedTokenBytes));
    }
}

PleGatherPipeline::~PleGatherPipeline() {
    for (const std::unique_ptr<Slot>& slot : slots_) {
        for (std::future<void>& work : slot->work) {
            if (work.valid()) {
                try {
                    work.get();
                } catch (...) {}
            }
        }
        if (slot->state == SlotState::Copying) {
            try {
                slot->completion.synchronize();
            } catch (...) {}
        }
    }
}

bool PleGatherPipeline::ensure_direct() {
    if (!direct_enabled_) { return false; } // env unset: nothing opens, nothing prints
    if (direct_ready_) { return true; }
    // Called only from the calling thread (prepare/gather_pinned), before any worker dispatch, so
    // the lazy open below runs exactly once and cannot race with itself.
    const char* path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT_PATH");
    const bool path_set = path != nullptr && path[0] != '\0';
    bool shards_ok      = path_set;
    for (const PleShardView& shard : table_.shards) {
        if (shard.file_offset == 0) {
            shards_ok = false;
            break;
        }
    }
    if (!shards_ok) {
        // Refused, loudly and exactly once -- a shard without a recorded file offset must never be
        // read from a guessed base. The mapped path keeps serving every caller.
        static bool refused_reported = false;
        if (!refused_reported) {
            refused_reported = true;
            std::fprintf(stderr,
                         "flash_next ple_io mode=refused path_set=%d shards_ok=%d window=4096 "
                         "buf=8192\n",
                         path_set ? 1 : 0, shards_ok ? 1 : 0);
            std::fflush(stderr);
        }
        return false;
    }
    try {
        direct_file_ = std::make_unique<artifact::InputFile>(std::filesystem::path(path));
    } catch (...) {
        direct_file_.reset();
        static bool open_reported = false;
        if (!open_reported) {
            open_reported = true;
            std::fprintf(stderr, "flash_next ple_io mode=refused path_set=1 reason=open_failed\n");
            std::fflush(stderr);
        }
        return false;
    }
    direct_ready_ = true;
    // One-line self-证: on/off, the read window, and the shard count the direct face will serve.
    std::fprintf(stderr, "flash_next ple_io mode=direct path_set=1 shards=%zu window=4096 buf=8192\n",
                 table_.shards.size());
    std::fflush(stderr);
    return true;
}

std::size_t PleGatherPipeline::acquire_slot() {
    for (std::size_t offset = 0; offset < slots_.size(); ++offset) {
        const std::size_t index = (next_slot_ + offset) % slots_.size();
        Slot& slot              = *slots_[index];
        if (slot.state == SlotState::Copying && slot.completion.ready()) {
            slot.state = SlotState::Idle;
        }
        if (slot.state == SlotState::Idle) {
            next_slot_ = (index + 1) % slots_.size();
            return index;
        }
    }
    throw std::runtime_error("PLE gather pipeline has no reusable slot");
}

PleGatherPipeline::Ticket
PleGatherPipeline::prepare(std::span<const std::array<std::int64_t, 16>> global_rows) {
    if (global_rows.empty() || global_rows.size() > max_tokens_) {
        throw std::invalid_argument("PLE gather batch is outside the startup-fixed capacity");
    }
    const std::size_t slot_index = acquire_slot();
    Slot& slot                   = *slots_[slot_index];
    slot.state                   = SlotState::Gathering;
    ++slot.generation;
    slot.work.clear();

    const std::size_t tokens       = global_rows.size();
    auto* base                     = static_cast<std::byte*>(slot.buffer.data());
    const std::size_t codes_bytes  = tokens * 16 * kPleCodesPerRow;
    const std::size_t scales_bytes = tokens * 16 * kPleScalesPerRow;

    // Block-7 alignment ④: the direct face produces the identical byte layout (see ple_table.h);
    // the mapped call below stays the OFF-arm path verbatim.
    if (ensure_direct()) {
        alignas(4096) std::byte window[8192]; // two 4096 blocks: a row window can span the boundary
        gather_ple_rows_compressed_direct(
            table_, PleDirectReadIo{direct_file_.get(), window, sizeof(window)}, global_rows,
            std::span<std::byte>(base, codes_bytes),
            std::span<std::byte>(base + codes_bytes, scales_bytes));
    } else {
        gather_ple_rows_compressed(
            table_, global_rows,
            std::span<std::byte>(base, codes_bytes),
            std::span<std::byte>(base + codes_bytes, scales_bytes));
    }

    return Ticket(this, slot_index, slot.generation, tokens);
}

void PleGatherPipeline::enqueue_copy(Ticket&& ticket, Tensor& output) {
    if (ticket.owner_ != this || ticket.slot_ >= slots_.size()) {
        throw std::invalid_argument("PLE gather ticket belongs to another pipeline");
    }
    Slot& slot = *slots_[ticket.slot_];
    if (slot.state != SlotState::Gathering || slot.generation != ticket.generation_) {
        throw std::invalid_argument("PLE gather ticket is stale");
    }
    if (output.dtype != DType::BF16 || output.ne[0] != static_cast<std::int32_t>(kPleOutputWidth) ||
        output.ne[1] != static_cast<std::int32_t>(ticket.tokens_) || output.ne[2] != 1 ||
        output.ne[3] != 1 || !output.is_contiguous() || output.data == nullptr) {
        throw std::invalid_argument("PLE gather output must be contiguous BF16 [2560,tokens]");
    }

    try {
        for (std::future<void>& work : slot.work) { work.get(); }
    } catch (...) {
        slot.work.clear();
        slot.state    = SlotState::Idle;
        ticket.owner_ = nullptr;
        throw;
    }
    slot.work.clear();
    const std::size_t bytes = ticket.tokens_ * kPleCompressedTokenBytes;
    // Payload H2D of gathered PLE rows; not a host control-flow round-trip.
    slot.ordering.record(device_.stream);
    slot.ordering.wait(device_.transfer_stream);
    CUDA_CHECK(cudaMemcpyAsync(slot.device_compressed.p, slot.buffer.data(), bytes,
                               cudaMemcpyHostToDevice, device_.transfer_stream));
    slot.completion.record(device_.transfer_stream);
    slot.completion.wait(device_.stream);
    flash_next_ple_dequant_launch(slot.device_compressed.p, output,
                                  static_cast<int>(ticket.tokens_), device_.stream);
    slot.state    = SlotState::Copying;
    ticket.owner_ = nullptr;
}

void PleGatherPipeline::gather_pinned(
    std::span<const std::array<std::int64_t, 16>> global_rows) {
    if (global_rows.empty() || global_rows.size() > max_tokens_) {
        throw std::invalid_argument("PLE gather batch is outside the startup-fixed capacity");
    }
    auto* output = static_cast<std::uint16_t*>(fixed_host_buffer_.data());

    // Block-7 alignment ④: same dispatch rule as prepare(). The calling thread's handle serves the
    // inline branch below; worker threads open their own (worker_direct_file) so no handle ever
    // carries two outstanding direct reads at once.
    const bool direct              = ensure_direct();
    const artifact::InputFile* file = direct ? direct_file_.get() : nullptr;

    // Served decode runs at B=1, and this gather sits in the inter-round window that nsys
    // measured at 0.99 ms/token of GPU idle (~1.2 host gaps >100 us per token). At that batch the
    // HostWorkerPool round-trip is two Windows condvar wake-ups (submit notify, then the future
    // wait) to buy ~2560 scalar dequants of parallelism that has nowhere to go. Below the
    // threshold the calling thread does the work itself.
    // Bitwise identity: same gather_ple_rows_bf16 over the same rows into the same
    // token * kPleOutputWidth offsets, so the pinned bytes are unchanged either way. Only the
    // thread that produces them differs.
    constexpr std::size_t kInlineGatherTokens = 2;
    if (global_rows.size() <= kInlineGatherTokens) {
        for (std::size_t token = 0; token < global_rows.size(); ++token) {
            if (direct) {
                alignas(4096) std::byte window[8192]; // two 4096 blocks: a row window can span the
                                                      // boundary
                gather_ple_rows_bf16_direct(
                    table_, PleDirectReadIo{file, window, sizeof(window)}, global_rows[token],
                    std::span<std::uint16_t>(output + token * kPleOutputWidth, kPleOutputWidth));
            } else {
                gather_ple_rows_bf16(
                    table_, global_rows[token],
                    std::span<std::uint16_t>(output + token * kPleOutputWidth, kPleOutputWidth));
            }
        }
        return;
    }

    std::vector<std::future<void>> work;
    work.reserve(global_rows.size());
    // Worker threads each open their own handle (see worker_direct_file): one outstanding direct
    // read per handle, so concurrent workers never share an OVERLAPPED handle. The path is the
    // same snapshot ensure_direct() validated above.
    std::string direct_path;
    if (direct) {
        const char* path = std::getenv("NINFER_FLASH_NEXT_ARTIFACT_PATH");
        if (path != nullptr) { direct_path.assign(path); }
    }
    for (std::size_t token = 0; token < global_rows.size(); ++token) {
        const std::array<std::int64_t, 16> rows = global_rows[token];
        work.push_back(workers_.submit([this, output, rows, token, direct, direct_path] {
            if (direct) {
                // Per-invocation stack window; 8192 bytes because a row window can span two
                // 4096-byte blocks when the row starts near a boundary.
                alignas(4096) std::byte window[8192];
                const artifact::InputFile* worker_file = worker_direct_file(direct_path);
                gather_ple_rows_bf16_direct(
                    table_, PleDirectReadIo{worker_file, window, sizeof(window)}, rows,
                    std::span<std::uint16_t>(output + token * kPleOutputWidth, kPleOutputWidth));
            } else {
                gather_ple_rows_bf16(
                    table_, rows,
                    std::span<std::uint16_t>(output + token * kPleOutputWidth, kPleOutputWidth));
            }
        }));
    }
    for (std::future<void>& w : work) {
        w.get();
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
