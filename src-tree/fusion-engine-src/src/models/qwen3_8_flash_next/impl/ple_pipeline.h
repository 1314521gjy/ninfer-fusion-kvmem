#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/host_worker_pool.h"
#include "core/tensor.h"
#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

class PleGatherPipeline {
public:
    class Ticket {
    public:
        Ticket() noexcept = default;

    private:
        friend class PleGatherPipeline;
        Ticket(PleGatherPipeline* owner, std::size_t slot, std::uint64_t generation,
               std::size_t tokens) noexcept;

        PleGatherPipeline* owner_ = nullptr;
        std::size_t slot_         = 0;
        std::uint64_t generation_ = 0;
        std::size_t tokens_       = 0;
    };

    PleGatherPipeline(PleTableView table, DeviceContext& device, std::size_t max_tokens,
                      std::size_t slot_count = 2, std::uint32_t worker_threads = 2);
    ~PleGatherPipeline();

    PleGatherPipeline(const PleGatherPipeline&)            = delete;
    PleGatherPipeline& operator=(const PleGatherPipeline&) = delete;

    [[nodiscard]] Ticket prepare(std::span<const std::array<std::int64_t, 16>> global_rows);

    // Waits only for the host gather, enqueues pinned BF16 H2D on the transfer stream, and inserts
    // a dependency into the main stream. The main stream itself is never host-synchronized.
    void enqueue_copy(Ticket&& ticket, Tensor& output);

    // Decode graph support: synchronous host gather into fixed pinned buffer before graph launch.
    void gather_pinned(std::span<const std::array<std::int64_t, 16>> global_rows);
    [[nodiscard]] const void* fixed_host_buffer() const noexcept { return fixed_host_buffer_.data(); }

private:
    struct Slot;

    [[nodiscard]] std::size_t acquire_slot();

    PleTableView table_;
    DeviceContext& device_;
    std::size_t max_tokens_ = 0;
    std::vector<std::unique_ptr<Slot>> slots_;
    HostWorkerPool workers_;
    std::size_t next_slot_ = 0;
    PinnedHostBuffer fixed_host_buffer_;
    // Block-7 alignment ④: unbuffered direct PLE reads (NINFER_FLASH_NEXT_PLE_IO_DIRECT; default
    // off -- an OFF process never opens the file and prints nothing). The handle opens lazily on
    // the calling thread at the first gather, before any worker dispatch, so InputFile's own lazy
    // open cannot race; after that it is only read positionally (OVERLAPPED) from any thread.
    bool direct_enabled_ = false;
    bool direct_ready_   = false;
    std::unique_ptr<artifact::InputFile> direct_file_;
    // Opens the direct handle once; refuses (and says so on stderr exactly once) when the artifact
    // path is unset or any shard is missing its file offset -- never guesses an offset.
    [[nodiscard]] bool ensure_direct();
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
