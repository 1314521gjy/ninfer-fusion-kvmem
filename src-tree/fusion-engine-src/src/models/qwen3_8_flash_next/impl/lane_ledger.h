#pragma once

#include "ninfer/ops/sampling.h"
#include "models/qwen3_8_flash_next/impl/ple_index.h"
#include "models/qwen3_8_flash_next/impl/runtime_plan.h"
#include "models/qwen3_8_flash_next/impl/runtime_state.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

class FlashNextLaneLedger;
class FlashNextTextExecutor;

class LaneHandle {
public:
    LaneHandle() noexcept = default;

    [[nodiscard]] std::uint32_t lane_index() const noexcept { return lane_index_; }

    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

    [[nodiscard]] const void* owner() const noexcept { return owner_; }

    [[nodiscard]] friend constexpr bool operator==(LaneHandle, LaneHandle) noexcept = default;

private:
    friend class FlashNextLaneLedger;
    friend class FlashNextTextExecutor;

    LaneHandle(const void* owner, std::uint32_t lane_index, std::uint64_t epoch) noexcept
        : owner_(owner), lane_index_(lane_index), epoch_(epoch) {}

    const void* owner_        = nullptr;
    std::uint32_t lane_index_ = 0;
    std::uint64_t epoch_      = 0;
};

struct LaneStepRequest {
    LaneHandle handle;
    std::int32_t token_id                       = 0;
    std::int32_t token_index                    = 0;
    std::array<std::int32_t, 3> mrope_positions = {0, 0, 0};
    ops::SamplingConfig sampling                = {};
    const Tensor* custom_embedding              = nullptr;
};

struct LaneCommitDecision {
    bool accept = true;
};

class FlashNextLaneLedger {
public:
    explicit FlashNextLaneLedger(const FlashNextRuntimePlan& plan);

    ~FlashNextLaneLedger() = default;

    FlashNextLaneLedger(const FlashNextLaneLedger&)            = delete;
    FlashNextLaneLedger& operator=(const FlashNextLaneLedger&) = delete;
    FlashNextLaneLedger(FlashNextLaneLedger&&)                 = delete;
    FlashNextLaneLedger& operator=(FlashNextLaneLedger&&)      = delete;

    [[nodiscard]] LaneHandle allocate_lane(const void* owner = nullptr);
    void release_lane(LaneHandle handle);

    [[nodiscard]] std::int32_t committed_frontier(LaneHandle handle) const;
    [[nodiscard]] const PleTokenHistory& lane_history(LaneHandle handle) const;

    // Physical group and capacity introspection
    [[nodiscard]] std::size_t available_physical_groups() const noexcept {
        return free_physical_groups_.size();
    }
    [[nodiscard]] std::span<const std::uint32_t> lane_physical_groups(LaneHandle handle) const;
    std::vector<std::uint32_t> take_lane_physical_groups(LaneHandle handle);
    void acquire_physical_groups(std::span<const std::uint32_t> groups);
    void release_physical_groups(std::span<const std::uint32_t> groups);
    void attach_physical_groups(LaneHandle handle, std::span<const std::uint32_t> groups,
                                std::int32_t committed_frontier, const PleTokenHistory& history);
    [[nodiscard]] std::uint32_t group_refcount(std::uint32_t group) const noexcept {
        return group < physical_group_refcounts_.size() ? physical_group_refcounts_[group] : 0U;
    }

    [[nodiscard]] std::size_t active_lanes_count() const noexcept;

    [[nodiscard]] bool has_pending_transaction() const noexcept { return has_pending_batch_; }

    [[nodiscard]] std::uint32_t pending_batch_size() const noexcept {
        return static_cast<std::uint32_t>(pending_requests_.size());
    }

    [[nodiscard]] std::uint64_t current_transaction_id() const noexcept {
        return current_transaction_id_;
    }

    struct PreparedRound {
        std::uint64_t transaction_id   = 0;
        std::int32_t max_active_blocks = 0;
        std::vector<std::array<std::int64_t, 16>> ple_indices;
    };

    // Validates batch, computes PLE indices, reserves physical groups, and updates shadow block
    // tables. Throws with strong exception guarantee (zero mutation) on dry-run failure.
    [[nodiscard]] PreparedRound begin_round(std::span<const LaneStepRequest> requests,
                                            const PleIndexMetadata& ple_meta);

    [[nodiscard]] PreparedRound begin_speculative_round(LaneHandle handle,
                                                        std::int32_t anchor_token_id,
                                                        std::span<const std::int32_t> draft_tokens,
                                                        std::int32_t first_token_index,
                                                        const PleIndexMetadata& ple_meta);

    [[nodiscard]] PreparedRound begin_prefill_chunk(LaneHandle handle,
                                                    std::span<const std::int32_t> token_ids,
                                                    std::int32_t first_token_index,
                                                    const PleIndexMetadata& ple_meta);

    // Rollback if subsequent launch staging fails before completion.
    void rollback_prepared_round(std::uint64_t tx_id);
    void rollback_prepared_prefill_chunk(std::uint64_t tx_id);

    // Commits per-row decisions, swaps accepted recurrent slots in alloc, and updates committed
    // frontier/history.
    void commit_round(std::uint64_t tx_id, std::span<const LaneCommitDecision> decisions,
                      FlashNextRuntimeAllocation& alloc, cudaStream_t stream);
    void commit_speculative_round(std::uint64_t tx_id,
                                  std::span<const std::int32_t> accepted_tokens,
                                  FlashNextRuntimeAllocation& alloc, cudaStream_t stream);
    void commit_prefill_chunk(std::uint64_t tx_id, FlashNextRuntimeAllocation& alloc,
                              cudaStream_t stream);

    // Aborts transaction: prevents frontier advancement, history commit, and slot publication.
    // Decode may have already written standby recurrent state; abort does not undo those writes.
    // Assigned page groups are retained by their lanes (capacity is not reclaimed).
    void abort_round(std::uint64_t tx_id) noexcept;
    void abort_prefill_chunk(std::uint64_t tx_id) noexcept;

    // Synchronizes dirty host block tables to device memory.
    void sync_tables_if_dirty(FlashNextRuntimeAllocation& alloc, cudaStream_t stream);

    // Reserve and publish the page mapping before an MTP proposal can write KV.
    void reserve_mtp_pages(LaneHandle handle, std::int32_t last_token_index);

    // Table introspection for verification
    [[nodiscard]] std::span<const std::int32_t> host_attention_table() const noexcept {
        return host_attention_table_;
    }

    [[nodiscard]] std::span<const std::int32_t> host_indexer_table() const noexcept {
        return host_indexer_table_;
    }

private:
    enum class LaneState : std::uint8_t {
        Free,
        Active,
        Pending,
    };

    struct LaneInfo {
        const void* owner               = nullptr;
        LaneState state                 = LaneState::Free;
        std::uint64_t epoch             = 0;
        std::int32_t committed_frontier = 0;
        PleTokenHistory history{};
    };

    FlashNextRuntimePlan plan_;

    std::vector<LaneInfo> lanes_;
    std::vector<std::vector<std::uint32_t>> lane_physical_groups_;
    std::vector<std::uint32_t> free_physical_groups_;
    std::vector<std::uint32_t> physical_group_refcounts_;

    // Host block tables (stride = logical_pages)
    // attention index: lane * attention_logical_pages + page
    // indexer index: lane * indexer_logical_pages + page
    std::vector<std::int32_t> host_attention_table_;
    std::vector<std::int32_t> host_indexer_table_;
    bool block_tables_dirty_ = false;

    // Transaction state
    std::uint64_t current_transaction_id_ = 0;
    bool has_pending_batch_               = false;
    bool is_pending_prefill_chunk_        = false;
    std::vector<LaneStepRequest> pending_requests_;
    std::vector<std::uint32_t> pending_lane_indices_;
    std::vector<std::size_t> previous_group_counts_;

    // Pending prefill chunk state
    std::uint32_t pending_prefill_lane_             = 0;
    std::vector<std::int32_t> pending_prefill_tokens_;
    std::int32_t pending_prefill_first_token_index_ = 0;

    void validate_handle(LaneHandle handle, LaneState expected_state) const;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
