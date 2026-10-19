#pragma once

#include <ninfer/models/qwen3_8_flash_next/runtime.h>
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "runtime/contract/structured_output.h"
#include "models/qwen3_8_flash_next/impl/load/materialized.h"
#include "models/qwen3_8_flash_next/impl/runtime_plan.h"
#include "models/qwen3_8_flash_next/impl/runtime_state.h"
#include "models/qwen3_8_flash_next/impl/text_executor.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include "models/qwen3_5/program/prefix_identity.h"
#include "models/qwen3_8_flash_next/impl/vision_execute.h"

namespace ninfer::models::qwen3_8_flash_next::detail {

class RequestBasePlanImpl {
public:
    runtime::RequestPlanSummary summary;
    qwen3_5::PreparedContextCache context_cache;
    ops::SamplingConfig sampling_config;
    std::shared_ptr<const runtime::CompiledOutputConstraint> output_constraint;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    bool allow_prefix_reuse               = false;
    ninfer::ThinkingControlOptions thinking;
    std::optional<qwen3_5::VisionControlPlan> vision_control_plan;
    std::optional<qwen3_5::VisionControl> vision_control;
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::uint32_t prefix_identity_tag = 0;
    std::uint32_t prefill_chunk       = 0;
    std::uint32_t checkpoint_slots_required = 0;
};

class AdmissionCandidateImpl {
public:
    runtime::RequestPlanSummary summary;
    runtime::IdentityMaterializationAssessment assessment;
    std::unique_ptr<RequestBasePlanImpl> base_plan;
    std::uint32_t reusable_tokens = 0;
    std::uint32_t source_continuation_index = 0;
    std::uint64_t source_continuation_generation = 0;
    // The Engine owner is the endpoint; the selected checkpoint may be its rewrite.
    std::uint32_t source_checkpoint_index = 0;
    std::uint64_t source_checkpoint_generation = 0;
    bool has_source = false;
    std::uint32_t required_page_groups = 0;
    std::uint64_t planning_revision = 0;

    std::vector<std::uint32_t> pressure_evicted_slots;
    std::vector<std::uint32_t> pressure_evicted_ordinals;
};

struct PressurePlanningOwner {
    const ContinuationHandle* handle = nullptr;
    runtime::PlanningOwnerId owner_id{};
    std::uint32_t continuation_index = 0;
    std::uint64_t generation = 0;
};

struct PressurePlanningTargetNode {
    std::uint32_t candidate_index = 0;
    std::vector<std::uint8_t> owner_evicted;
    std::uint32_t stable_ordinal = 0;
    bool root_maximal = false;
};

class PressurePlanningSessionImpl {
public:
    PressurePlanningSessionImpl(
        ProgramImpl& program,
        std::span<const AdmissionCandidate* const> candidates,
        std::span<const runtime::PlanningCandidateId> candidate_ids,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids);

    ~PressurePlanningSessionImpl() noexcept;

    [[nodiscard]] bool valid(PressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(runtime::PlanningCandidateId candidate) const;

    [[nodiscard]] PressureTargetHandle identity_target(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] PressureTargetHandle root_maximal_target(runtime::PlanningCandidateId root_candidate);
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target);
    void retain_assessment(PressureTargetHandle target);
    [[nodiscard]] std::optional<PressureTargetHandle>
    guided_closure_target(runtime::PlanningCandidateId candidate,
                          std::span<const runtime::PlanningOwnerId> preferred_owner_ids);
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle parent);
    // How many further single-owner sacrifices this session could still enumerate. This model has no
    // arena bound: `prepare_expansion` builds children by flipping `owner_evicted[i]`, one owner per
    // expansion, and the child set is a subset of `owners_`; so `owners_.size()` is the length of the
    // sacrifice ladder -- the honest answer to the same question fusion answers with
    // `target_arena_maximum() - targets.size()` (`planning/pressure_planner.cpp:809-816`) against its
    // own 4096-slot arena. See `optional_targets_remaining` in `impl/program.cpp` for the full note.
    [[nodiscard]] std::uint32_t optional_targets_remaining() const noexcept;
    [[nodiscard]] PressureExpansionView commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] std::optional<ResourcePlan>
    seal(AssessedPressureTarget&& assessed, const qwen3_5::PreparedPrompt& prompt,
         runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<CapturePressurePlan>
    seal_capture(AssessedPressureTarget&& assessed);
    // The seal window, one exclusive claim at a time. Same shape as fusion's
    // (`models/qwen3_5/program/planning/pressure.cpp:2410-2417`): a compare-exchange claim and an
    // idempotent release, with the state in `seal_window_claimed_` below. The caller
    // (`SealWindowClaim`) backs off on refusal and takes its own fallback, so `false` is a normal
    // answer.
    [[nodiscard]] bool try_claim_seal_window() noexcept;
    void release_seal_window() noexcept;

    ProgramImpl* program_ = nullptr;
    std::uint64_t resource_revision_ = 0;
    std::uint32_t session_generation_ = 0;

    std::vector<const AdmissionCandidate*> candidates_;
    std::vector<runtime::PlanningCandidateId> candidate_ids_;
    std::vector<PressurePlanningOwner> owners_;
    std::vector<PressurePlanningTargetNode> targets_;

    std::vector<PressurePlanningTargetNode> expansion_scratch_;
    std::vector<PressureTargetHandle> committed_children_;
    std::vector<runtime::PressureOwnerOutcome> assessment_outcomes_;
    std::vector<runtime::PressureOwnerOutcome> guidance_outcomes_;
    std::vector<runtime::PressureCheckpointRecoveryImpact> assessment_impacts_;
    std::vector<runtime::CheckpointRecoveryAlternativeWork> assessment_recovery_work_;
    std::uint32_t retained_target_idx_ = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t scratch_generation_ = 0;
    std::uint32_t scratch_parent_index_ = 0;
    std::uint32_t scratch_new_count_ = 0;
    bool scratch_live_ = false;
};

class SequencePlanImpl {
public:
    explicit SequencePlanImpl(FlashNextRuntimePlan target_plan) : plan(std::move(target_plan)) {}
    FlashNextRuntimePlan plan;
};

class SequencePlannerImpl {
public:
    explicit SequencePlannerImpl(FlashNextRuntimeConfig cfg) : config(cfg) {}
    FlashNextRuntimeConfig config;
};

enum class ContinuationSlotRole : std::uint8_t {
    Vacant,
    ReservedEndpoint,
    ReservedTurnClosure,
    Catalogued,
};

struct ContinuationSlot {
    ContinuationSlotRole role = ContinuationSlotRole::Vacant;
    std::uint32_t cache_slot = 0;
    std::uint64_t generation = 1;
    std::vector<TokenId> committed_tokens;
    std::vector<std::uint32_t> physical_groups;
    std::int32_t committed_frontier = 0;
    std::array<std::uint64_t, 2> prefix_digests{};
    PleTokenHistory history{};
    std::uint64_t last_used_epoch = 0;
    runtime::CheckpointKind kind = runtime::CheckpointKind::SessionEndpoint;
    // What the Engine's catalog entry for this owner counts (endpoint + rewrite): the Engine
    // validates an eviction's dropped_checkpoints against its own summary, not ours.
    std::uint32_t published_checkpoints = 1;
    // The TurnClosure slot published as this endpoint's rewrite; it belongs to the same
    // Engine owner and goes away with it.
    std::optional<std::uint32_t> paired_rewrite_slot;
    std::uint64_t paired_rewrite_generation = 0;
};

struct LaneState {
    bool active = false;
    std::uint64_t epoch = 0;
    LaneHandle lane_handle{};
    std::vector<TokenId> prompt_tokens;
    std::vector<std::int32_t> mrope_pos0;
    std::vector<std::int32_t> mrope_pos1;
    std::vector<std::int32_t> mrope_pos2;
    std::vector<std::shared_ptr<const qwen3_5::PreparedMediaPayload>> media_payloads;
    std::vector<qwen3_5::VisionItem> vision_items;
    std::optional<qwen3_5::VisionControl> vision_control;
    std::optional<std::size_t> encoded_item_index;
    std::uint32_t prompt_tokens_processed = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    std::int32_t committed_frontier       = 0;
    std::int32_t last_token_id            = 0;
    std::int32_t last_token_pos           = 0;
    std::int32_t last_token_index         = 0;
    std::uint32_t total_generated_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    ops::SamplingConfig sampling_config{};
    std::unique_ptr<runtime::OutputConstraintState> output_constraint;
    bool publish_continuation = false;
    bool prefill_completed = false;
    bool finished          = false;
    std::vector<std::int32_t> draft_tokens;
    std::vector<std::int32_t> pending_accepted_tokens;
    SpeculativeStats speculative_stats{};
    qwen3_5::detail::PrefixShortlistDigests prefix_digests;
    std::optional<std::uint32_t> turn_closure_continuation_index;
    // Owned only while this lane is admitted; optional captures cannot consume it.
    std::optional<std::uint32_t> reserved_endpoint_index;
    std::optional<std::uint32_t> reserved_turn_closure_index;
    std::uint64_t pending_capture_offer = 0;
    std::optional<std::uint32_t> capture_frontier;
    bool capture_offered = false;
    bool reused_from_turn_closure = false;
    // Page groups this lane may still take (from the admitted plan); admission subtracts the
    // part not yet owned from the free list so concurrent lanes cannot oversubscribe the pool.
    std::uint32_t planned_groups = 0;
    std::optional<std::uint32_t> reused_from_continuation_index;
    std::optional<std::uint64_t> reused_from_continuation_generation;
};

class ProgramImpl {
public:
    ProgramImpl(const LoadedModelData* model_data, FlashNextRuntimePlan plan_in, DeviceContext& dev,
                TextModelView text_override = {},
                std::optional<VisionModelView> vision_override = std::nullopt,
                PleIndexMetadata ple_override = kPleIndexMetadata);
    ~ProgramImpl() = default;

    ProgramImpl(const ProgramImpl&)            = delete;
    ProgramImpl& operator=(const ProgramImpl&) = delete;
    ProgramImpl(ProgramImpl&&)                 = delete;
    ProgramImpl& operator=(ProgramImpl&&)      = delete;

    void sample_tokens(const Tensor& logits,
                       std::span<const std::uint32_t> lane_indices,
                       std::span<std::int32_t> out_tokens);
    const std::int32_t* upload_constraint_mask(std::uint32_t lane, std::uint32_t column,
                                              runtime::OutputConstraintState& constraint);

    [[nodiscard]] bool is_slot_protected(std::size_t slot_idx) const;
    [[nodiscard]] std::uint32_t count_freeable_physical_groups(
        std::optional<std::size_t> current_matching_slot = std::nullopt) const;
    void evict_continuation_slot(std::size_t slot_idx);
    bool ensure_physical_groups_available(std::size_t needed);
    std::int32_t allocate_vacant_continuation_slot();
    void vacate_slot(std::size_t slot_idx);
    void vacate_owner(std::size_t slot_idx);
    [[nodiscard]] std::uint32_t published_checkpoints_of(std::size_t slot_idx) const noexcept;
    [[nodiscard]] std::uint32_t reserved_unowned_groups() const noexcept;
    void drop_unpublished_checkpoints(LaneState& st);
    void release_checkpoint_reservations(LaneState& st);
    [[nodiscard]] std::uint32_t vacant_checkpoint_slots() const noexcept;
    [[nodiscard]] std::uint32_t owner_checkpoint_slots(std::size_t owner) const;
    [[nodiscard]] bool has_mtp() const noexcept {
        return (model_data_ != nullptr ? model_data_->text : text_override_).mtp.has_value();
    }

    // The seal window, one exclusive claim at a time. Same shape as fusion's
    // (`models/qwen3_5/program/planning/pressure.cpp:2410-2417`): a compare-exchange claim and an
    // idempotent release, state in `seal_window_claimed_` below. `SealWindowClaim` backs off on
    // refusal and takes its own fallback, so a `false` answer is normal, not an error.
    [[nodiscard]] bool try_claim_seal_window() noexcept;
    void release_seal_window() noexcept;

    std::uint32_t pressure_planning_generation_ = 0;
    bool pressure_planning_active_ = false;
    // The seal window, copied in shape from the fusion Program
    // (`models/qwen3_5/program/planning/pressure.cpp:2410-2417` and
    // `models/qwen3_5/program/program_impl.h:971`): one exclusive claim, held across a seal so a
    // concurrent demote cannot bump a victim's slot generation between the final assess and the seal.
    // `SealWindowClaim` (`runtime/engine/context_cache/seal_window_claim.h`) is the only user and it
    // backs off and falls back when the claim is refused, so refusing is a correct answer, not an
    // error path.
    std::atomic<bool> seal_window_claimed_{false};
    std::vector<qwen3_5::MaterializationVictimResult> transaction_victims_;

    const LoadedModelData* model_data_ = nullptr;
    TextModelView text_override_{};
    std::optional<VisionModelView> vision_override_;
    FlashNextRuntimePlan plan_;
    DeviceContext& device_;
    FlashNextRuntimeAllocation allocation_;
    FlashNextTextExecutor executor_;
    std::optional<FlashNextVisionSession> vision_session_;

    std::uint64_t resource_revision_ = 1;

    std::vector<LaneState> lane_states_;
    std::vector<ContinuationSlot> continuation_slots_;
    std::uint64_t continuation_epoch_ = 0;
    std::uint64_t capture_counter_ = 0;

    bool has_context_transaction_ = false;
    std::optional<runtime::LaneId> transaction_lane_;
    std::optional<std::uint64_t> transaction_epoch_;
    bool transaction_has_source_ = false;
    runtime::PrivateSourceMode transaction_source_mode_ = runtime::PrivateSourceMode::ConsumeToActive;

    bool is_capture_transaction_ = false;
    ActiveCaptureResult pending_capture_result_;

    PendingRound pending_round_;
    std::vector<TokenId> pending_batch_tokens_;
    std::vector<std::int32_t> pending_batch_row_counts_;

    WorkspaceArena sampling_workspace_;
    DeviceBuffer device_sampling_configs_;
    DeviceBuffer device_sampling_positions_;
    DeviceBuffer device_sampled_tokens_;
    std::vector<std::int32_t> host_sampled_tokens_;
    std::vector<ops::SamplingConfig> host_sampling_configs_;
    std::vector<std::int32_t> host_sampling_positions_;
    static constexpr std::size_t kConstraintMaskWords = (248077 + 31) / 32;
    static constexpr std::size_t kConstraintColumns = 5;
    DeviceBuffer device_constraint_masks_;
    std::vector<std::int32_t> host_constraint_masks_;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
