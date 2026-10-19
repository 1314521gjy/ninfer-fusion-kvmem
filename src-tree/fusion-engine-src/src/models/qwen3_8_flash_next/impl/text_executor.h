#pragma once

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/tensor.h"
#include "models/qwen3_8_flash_next/impl/lane_ledger.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/ple_index.h"
#include "models/qwen3_8_flash_next/impl/ple_pipeline.h"
#include "models/qwen3_8_flash_next/impl/runtime_plan.h"
#include "models/qwen3_8_flash_next/impl/runtime_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

struct FlashNextDecodeStateSink;
class FlashNextTextExecutor;

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t bucket_index           = 0; // explicit integer; not derived from frontiers
    std::uint32_t bucket_blocks          = 0;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0; // (bucket_index << 8) | batch_size
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0; // (bucket_index << 8) | batch_size
    std::uint32_t batch_size     = 0;
    std::uint32_t bucket_index   = 0; // explicit integer field
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    FlashNextDecodeGraphBuckets buckets;
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

struct MtpDraftGraph {
    std::uint32_t step_index = 0;
    std::uint32_t bucket_index = 0;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
};

class PendingRound {
public:
    PendingRound() noexcept = default;

    ~PendingRound() noexcept {
        if (valid()) { abort(); }
    }

    PendingRound(const PendingRound&)            = delete;
    PendingRound& operator=(const PendingRound&) = delete;
    PendingRound(PendingRound&& other) noexcept;
    PendingRound& operator=(PendingRound&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t batch_size() const;
    [[nodiscard]] Tensor logits() const;
    [[nodiscard]] Tensor final_hidden() const;
    [[nodiscard]] Tensor hyper_hidden() const;
    [[nodiscard]] std::span<const std::int32_t> sampled_tokens() const;

    void commit(std::span<const LaneCommitDecision> decisions);
    void commit_speculative(std::uint32_t lane_index,
                            std::span<const std::int32_t> accepted_tokens);
    void abort() noexcept;

private:
    friend class FlashNextTextExecutor;
    PendingRound(FlashNextTextExecutor* owner, std::uint64_t transaction_id,
                 std::uint32_t batch_size, Tensor logits, Tensor final_hidden,
                 Tensor hyper_hidden,
                 std::span<const std::int32_t> sampled_tokens = {}) noexcept;

    FlashNextTextExecutor* owner_ = nullptr;
    std::uint64_t transaction_id_ = 0;
    std::uint32_t batch_size_     = 0;
    Tensor logits_{};
    Tensor final_hidden_{};
    Tensor hyper_hidden_{};
    std::array<std::int32_t, 8> sampled_tokens_{};
};

class FlashNextTextExecutor {
public:
    FlashNextTextExecutor(const TextModelView& model, PleIndexMetadata ple_metadata,
                          DeviceContext& device, FlashNextRuntimeAllocation& allocation);

    ~FlashNextTextExecutor();

    FlashNextTextExecutor(const FlashNextTextExecutor&)            = delete;
    FlashNextTextExecutor& operator=(const FlashNextTextExecutor&) = delete;
    FlashNextTextExecutor(FlashNextTextExecutor&&)                 = delete;
    FlashNextTextExecutor& operator=(FlashNextTextExecutor&&)      = delete;

    [[nodiscard]] LaneHandle allocate_lane();
    void release_lane(LaneHandle handle);

    [[nodiscard]] std::int32_t committed_frontier(LaneHandle handle) const;
    [[nodiscard]] const PleTokenHistory& lane_history(LaneHandle handle) const {
        return ledger_.lane_history(handle);
    }

    [[nodiscard]] std::size_t active_lanes_count() const noexcept;

    [[nodiscard]] std::size_t available_physical_groups() const noexcept {
        return ledger_.available_physical_groups();
    }

    [[nodiscard]] std::span<const std::uint32_t> lane_physical_groups(LaneHandle handle) const {
        return ledger_.lane_physical_groups(handle);
    }

    std::vector<std::uint32_t> take_lane_physical_groups(LaneHandle handle) {
        return ledger_.take_lane_physical_groups(handle);
    }

    void acquire_physical_groups(std::span<const std::uint32_t> groups) {
        ledger_.acquire_physical_groups(groups);
    }

    void release_physical_groups(std::span<const std::uint32_t> groups) {
        ledger_.release_physical_groups(groups);
    }

    [[nodiscard]] std::uint32_t group_refcount(std::uint32_t group) const noexcept {
        return ledger_.group_refcount(group);
    }

    void attach_physical_groups(LaneHandle handle, std::span<const std::uint32_t> groups,
                                std::int32_t committed_frontier, const PleTokenHistory& history) {
        ledger_.attach_physical_groups(handle, groups, committed_frontier, history);
    }

    void copy_state_slot(std::uint32_t src_slot, std::uint32_t dst_slot) {
        alloc_.copy_state_slot(src_slot, dst_slot, device_.stream);
    }

    [[nodiscard]] const FlashNextRuntimeAllocation& allocation() const noexcept { return alloc_; }
    [[nodiscard]] FlashNextRuntimeAllocation& allocation() noexcept { return alloc_; }

    [[nodiscard]] bool has_pending_round() const noexcept {
        return ledger_.has_pending_transaction();
    }

    [[nodiscard]] std::uint32_t pending_batch_size() const noexcept {
        return ledger_.pending_batch_size();
    }

    // Executes one compact decode round for a batch of 1..max_concurrency requests on
    // device_.stream. Returns an owner-bound PendingRound transaction.
    [[nodiscard]] PendingRound execute_round(std::span<const LaneStepRequest> requests,
                                             const FlashNextDecodeStateSink* sink = nullptr);

    // Consumes the accumulated blocking-wait time from the last round, in nanoseconds.
    [[nodiscard]] std::uint64_t take_round_device_wait_ns() noexcept {
        const std::uint64_t value = round_device_wait_ns_;
        round_device_wait_ns_     = 0;
        return value;
    }

    [[nodiscard]] PendingRound
    execute_speculative_verify_round(LaneHandle handle, std::int32_t anchor_token_id,
                                     std::span<const std::int32_t> draft_tokens,
                                     std::int32_t first_token_index,
                                     std::array<std::int32_t, 3> first_mrope_position,
                                     const ops::SamplingConfig& sampling,
                                     std::span<const ops::SamplingConfig> column_sampling = {});

    void draft_mtp_tokens(LaneHandle handle, std::int32_t token_id, std::int32_t token_index,
                          std::array<std::int32_t, 3> mrope_pos, const Tensor& backbone_hidden,
                          std::uint32_t draft_count, std::span<std::int32_t> out_draft_tokens);

    [[nodiscard]] PendingRound
    execute_prefill_chunk(LaneHandle handle, std::span<const std::int32_t> token_ids,
                          std::span<const std::array<std::int32_t, 3>> positions,
                          std::int32_t first_token_index,
                          const FlashNextDecodeStateSink* sink                     = nullptr,
                          const Tensor* visual_embeddings                          = nullptr,
                          std::span<const std::int32_t> chunk_local_scatter_indices = {});

    [[nodiscard]] const FlashNextLaneLedger& ledger() const noexcept { return ledger_; }

    [[nodiscard]] FlashNextLaneLedger& ledger() noexcept { return ledger_; }

    [[nodiscard]] bool use_cuda_graph() const noexcept { return use_cuda_graph_; }
    void set_use_cuda_graph(bool enable) noexcept { use_cuda_graph_ = enable; }
    [[nodiscard]] const DecodeGraphFamily& decode_graphs() const noexcept { return decode_graphs_; }
    // Retained for interface compatibility: lazy capture is deprecated in Sequence D23 (graphs are captured eagerly at startup).
    [[nodiscard]] std::optional<double> last_lazy_capture_milliseconds() const noexcept {
        return last_lazy_capture_ms_;
    }
    // Returns true if decode graph for (batch_size, bucket_index) failed capture at startup
    // and was pinned to eager execution.
    [[nodiscard]] bool decode_graph_pinned_eager(std::uint32_t batch_size,
                                                 std::uint32_t bucket_index) const noexcept;

    void instantiate_graphs();
    void execute_round_body(std::uint32_t batch_size, std::int32_t active_blocks,
                            const FlashNextDecodeStateSink* sink, bool aliased_recurrent_scan = false);

    // Production execute_round always uses the selected bucket envelope. Tests use this to
    // run eager decode with an explicit indexer envelope (live frontier).
    [[nodiscard]] PendingRound execute_round_eager(std::span<const LaneStepRequest> requests,
                                                   std::int32_t active_blocks,
                                                   const FlashNextDecodeStateSink* sink = nullptr);

private:
    friend class PendingRound;

    const TextModelView& model_;
    PleIndexMetadata ple_metadata_;
    DeviceContext& device_;
    FlashNextRuntimeAllocation& alloc_;

    PleGatherPipeline ple_pipeline_;
    FlashNextLaneLedger ledger_;

    bool use_cuda_graph_ = true;
    DecodeGraphFamily decode_graphs_;
    DecodeGraphFamily speculative_graphs_;
    std::vector<MtpDraftGraph> mtp_draft_graphs_;
    void execute_mtp_draft_step(std::uint32_t step_index, std::int32_t active_blocks);
    WorkspaceArena sampling_workspace_;
    CudaCompletionEvent round_completion_;
    // Nanoseconds blocked on target or draft completion since the last take.
    // Program reclassifies this out of the host bucket; see ExecutionTimingRecorder.
    std::uint64_t round_device_wait_ns_ = 0;
    bool round_in_flight_ = false;
    // Eager-only: custom embedding columns for the round being built (never captured).
    std::vector<const Tensor*> pending_custom_embeddings_;

    bool pending_is_prefill_chunk_                     = false;
    std::uint32_t pending_prefill_lane_                = 0;
    std::int32_t pending_prefill_initial_active_slot_  = 0;
    std::int32_t pending_prefill_initial_standby_slot_ = 0;

    std::optional<double> last_lazy_capture_ms_;
    std::uint8_t graph_capture_failures_[8][kFlashNextDecodeGraphMaxBuckets]{};
    bool graph_pinned_eager_[8][kFlashNextDecodeGraphMaxBuckets]{};



    [[nodiscard]] DecodeGraphTopology* find_topology(std::uint32_t batch_size,
                                                     std::uint32_t bucket_index, bool speculative = false) noexcept;
    [[nodiscard]] const DecodeGraphTopology* find_topology(std::uint32_t batch_size,
                                                           std::uint32_t bucket_index, bool speculative = false) const noexcept;
    bool install_captured_graph(std::uint32_t batch_size, std::uint32_t bucket_index,
                                std::int32_t bucket_blocks, bool speculative = false);
    [[nodiscard]] PendingRound
    finish_prepared_round(std::span<const LaneStepRequest> requests,
                          FlashNextLaneLedger::PreparedRound prepared,
                          const FlashNextDecodeStateSink* sink, bool force_eager,
                          std::int32_t active_blocks);

    void commit_transaction(std::uint64_t tx_id, std::span<const LaneCommitDecision> decisions);
    void commit_speculative_transaction(std::uint64_t tx_id, std::uint32_t lane_index,
                                        std::span<const std::int32_t> accepted_tokens);
    void abort_transaction(std::uint64_t tx_id) noexcept;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
