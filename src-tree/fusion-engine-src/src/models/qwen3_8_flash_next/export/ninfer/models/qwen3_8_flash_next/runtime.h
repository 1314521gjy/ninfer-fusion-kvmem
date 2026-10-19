#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <models/qwen3_5/frontend/prepared_prompt.h>
#include <models/qwen3_5/program/program.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::models::qwen3_8_flash_next {

namespace detail {
class RequestBasePlanImpl;
class AdmissionCandidateImpl;
class PressurePlanningSessionImpl;
class ProgramImpl;
class SequencePlanImpl;
class SequencePlannerImpl;
} // namespace detail

using PrefixShortlistKey    = qwen3_5::PrefixShortlistKey;
using TargetKVRequirement   = qwen3_5::TargetKVRequirement;
using CheckpointSummary     = qwen3_5::CheckpointSummary;
using ContinuationSummary   = qwen3_5::ContinuationSummary;
using SharedPrefixSummary   = qwen3_5::SharedPrefixSummary;
using CaptureAssessment     = qwen3_5::CaptureAssessment;
using PressureTargetHandle  = qwen3_5::PressureTargetHandle;
using PhysicalUsageSnapshot = qwen3_5::PhysicalUsageSnapshot;
using CommitRowResult       = qwen3_5::CommitRowResult;
// Two names the fusion engine's ModelContract needs and upstream's target had no reason to
// export: the engine builds the n-gram archive itself (`engine_core.h` says
// `make_unique<NgramArchive>`) and quotes hybrid-cache admission through this struct. Both are
// the engine's own types, so these are aliases, not new definitions.
using NgramArchive          = qwen3_5::NgramArchive;
using HybridAdmissionQuote = qwen3_5::HybridAdmissionQuote;
using DeviceKVPages        = qwen3_5::DeviceKVPages;
using DeviceKVLeaseShortfall = qwen3_5::DeviceKVLeaseShortfall;

class SequenceHandle {
public:
    SequenceHandle() noexcept                                 = default;
    SequenceHandle(const SequenceHandle&) noexcept            = default;
    SequenceHandle& operator=(const SequenceHandle&) noexcept = default;

    explicit SequenceHandle(const void* owner, runtime::LaneId lane, std::uint64_t epoch) noexcept
        : owner_(owner), lane_(lane), epoch_(epoch) {}

    [[nodiscard]] const void* owner() const noexcept { return owner_; }
    [[nodiscard]] runtime::LaneId lane() const noexcept { return lane_; }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
};

class ContinuationHandle {
public:
    ContinuationHandle() noexcept = default;
    ~ContinuationHandle()         = default;

    ContinuationHandle(ContinuationHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    ContinuationHandle& operator=(ContinuationHandle&& other) noexcept {
        if (this != &other) {
            owner_      = std::exchange(other.owner_, nullptr);
            index_      = other.index_;
            generation_ = std::exchange(other.generation_, 0);
        }
        return *this;
    }

    ContinuationHandle(const ContinuationHandle&)            = delete;
    ContinuationHandle& operator=(const ContinuationHandle&) = delete;

    explicit ContinuationHandle(const void* owner, std::uint32_t index,
                                std::uint64_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    [[nodiscard]] const void* owner() const noexcept { return owner_; }
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;
};

class SharedPrefixHandle {
public:
    SharedPrefixHandle() noexcept = default;
    ~SharedPrefixHandle()         = default;

    SharedPrefixHandle(SharedPrefixHandle&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
          generation_(std::exchange(other.generation_, 0)) {}

    SharedPrefixHandle& operator=(SharedPrefixHandle&& other) noexcept {
        if (this != &other) {
            owner_      = std::exchange(other.owner_, nullptr);
            index_      = other.index_;
            generation_ = std::exchange(other.generation_, 0);
        }
        return *this;
    }

    SharedPrefixHandle(const SharedPrefixHandle&)            = delete;
    SharedPrefixHandle& operator=(const SharedPrefixHandle&) = delete;

    explicit SharedPrefixHandle(const void* owner, std::uint32_t index,
                                std::uint64_t generation) noexcept
        : owner_(owner), index_(index), generation_(generation) {}

    [[nodiscard]] const void* owner() const noexcept { return owner_; }
    [[nodiscard]] std::uint32_t index() const noexcept { return index_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }

private:
    const void* owner_        = nullptr;
    std::uint32_t index_      = 0;
    std::uint64_t generation_ = 0;
};

class CaptureOffer {
public:
    CaptureOffer() noexcept = default;
    ~CaptureOffer()         = default;

    CaptureOffer(CaptureOffer&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), lane_(other.lane_), epoch_(other.epoch_),
          id_(std::exchange(other.id_, 0)) {}

    CaptureOffer& operator=(CaptureOffer&& other) noexcept {
        if (this != &other) {
            owner_ = std::exchange(other.owner_, nullptr);
            lane_  = other.lane_;
            epoch_ = other.epoch_;
            id_    = std::exchange(other.id_, 0);
        }
        return *this;
    }

    CaptureOffer(const CaptureOffer&)            = delete;
    CaptureOffer& operator=(const CaptureOffer&) = delete;

    explicit CaptureOffer(const void* owner, runtime::LaneId lane, std::uint64_t epoch,
                          std::uint64_t id) noexcept
        : owner_(owner), lane_(lane), epoch_(epoch), id_(id) {}

    [[nodiscard]] const void* owner() const noexcept { return owner_; }
    [[nodiscard]] runtime::LaneId lane() const noexcept { return lane_; }
    [[nodiscard]] std::uint64_t epoch() const noexcept { return epoch_; }
    friend class ContractAccess;

private:
    const void* owner_ = nullptr;
    runtime::LaneId lane_{};
    std::uint64_t epoch_ = 0;
    std::uint64_t id_    = 0;
};

class PendingBatch {
public:
    PendingBatch() noexcept = default;
    ~PendingBatch()         = default;

    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          transaction_(std::exchange(other.transaction_, 0)), rows_(other.rows_),
          row_count_(std::exchange(other.row_count_, 0)), tokens_(other.tokens_),
          row_counts_(other.row_counts_), row_stride_(other.row_stride_), timing_(other.timing_) {
        other.tokens_     = {};
        other.row_counts_ = {};
        other.row_stride_ = 0;
        other.timing_     = {};
    }

    PendingBatch& operator=(PendingBatch&& other) noexcept {
        if (this != &other) {
            owner_       = std::exchange(other.owner_, nullptr);
            transaction_ = std::exchange(other.transaction_, 0);
            rows_        = other.rows_;
            row_count_   = std::exchange(other.row_count_, 0);
            tokens_      = other.tokens_;
            row_counts_  = other.row_counts_;
            row_stride_  = other.row_stride_;
            timing_      = other.timing_;
            other.tokens_     = {};
            other.row_counts_ = {};
            other.row_stride_ = 0;
            other.timing_     = {};
        }
        return *this;
    }

    PendingBatch(const PendingBatch&)            = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;

    [[nodiscard]] std::size_t row_count() const noexcept { return row_count_; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return row_counts_; }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return row_stride_; }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }

    friend class ContractAccess;

private:
    const void* owner_         = nullptr;
    std::uint64_t transaction_ = 0;
    std::array<SequenceHandle, kMaximumConcurrency> rows_{};
    std::size_t row_count_ = 0;
    std::span<const TokenId> tokens_;
    std::span<const std::int32_t> row_counts_;
    std::uint32_t row_stride_ = 0;
    runtime::ExecutionTiming timing_;
};

class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const qwen3_5::PreparedContextCache& context_cache() const noexcept;
    [[nodiscard]] std::optional<PrefixShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept;
    // Read at `context_cache/resource_manager.h:356/380/387`, all three inside debug-logging
    // arguments. Fusion declares it next to `prefix_shortlist_key`
    // (`models/qwen3_5/program/program.h:306`, implementation at `program.cpp:101-103`); the ported
    // header had only the key accessor, so `ResourceManager<TargetContract>` failed to instantiate.
    // Same meaning and same answer as fusion: the number of frontiers the key accessor can key,
    // which is the prompt's digest count.
    [[nodiscard]] std::size_t prefix_shortlist_size() const noexcept;
    [[nodiscard]] std::optional<runtime::PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl> impl_;
};

class AdmissionCandidate {
public:
    AdmissionCandidate(AdmissionCandidate&&) noexcept;
    AdmissionCandidate& operator=(AdmissionCandidate&&) noexcept;
    ~AdmissionCandidate();

    AdmissionCandidate(const AdmissionCandidate&)            = delete;
    AdmissionCandidate& operator=(const AdmissionCandidate&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] const runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept;

public:
    explicit AdmissionCandidate(std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept;
    std::unique_ptr<detail::AdmissionCandidateImpl> impl_;
};

class ResourcePlan {
public:
    ResourcePlan(ResourcePlan&&) noexcept;
    ResourcePlan& operator=(ResourcePlan&&) noexcept;
    ~ResourcePlan();

    ResourcePlan(const ResourcePlan&)            = delete;
    ResourcePlan& operator=(const ResourcePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    [[nodiscard]] bool needs_transfer() const noexcept { return needs_transfer_; }
    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept { return revision_; }

public:
    ResourcePlan(AdmissionCandidate&& admission, runtime::ProgramResourceRevision revision,
                 bool needs_transfer) noexcept;

    friend class ContractAccess;

private:
    AdmissionCandidate admission_;
    runtime::ProgramResourceRevision revision_{};
    bool needs_transfer_    = false;
};

class PersistentBackfillProof {
public:
    PersistentBackfillProof(PersistentBackfillProof&&) noexcept            = default;
    PersistentBackfillProof& operator=(PersistentBackfillProof&&) noexcept = default;

    PersistentBackfillProof(const PersistentBackfillProof&)            = delete;
    PersistentBackfillProof& operator=(const PersistentBackfillProof&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    explicit PersistentBackfillProof(runtime::ProgramResourceRevision revision) noexcept
        : revision_(revision) {}

    runtime::ProgramResourceRevision revision_;

    friend class Program;
};

class PreparedPressureExpansion {
public:
    PreparedPressureExpansion(PreparedPressureExpansion&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          scratch_generation_(std::exchange(other.scratch_generation_, 0)),
          parent_index_(other.parent_index_), new_canonical_count_(other.new_canonical_count_) {}

    PreparedPressureExpansion& operator=(PreparedPressureExpansion&&)      = delete;
    PreparedPressureExpansion(const PreparedPressureExpansion&)            = delete;
    PreparedPressureExpansion& operator=(const PreparedPressureExpansion&) = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept {
        return new_canonical_count_;
    }

public:
    PreparedPressureExpansion(const void* session, std::uint32_t session_generation,
                              std::uint32_t scratch_generation, std::uint32_t parent_index,
                              std::uint32_t new_canonical_count) noexcept
        : session_(session), session_generation_(session_generation),
          scratch_generation_(scratch_generation), parent_index_(parent_index),
          new_canonical_count_(new_canonical_count) {}

private:
    const void* session_               = nullptr;
    std::uint32_t session_generation_  = 0;
    std::uint32_t scratch_generation_  = 0;
    std::uint32_t parent_index_        = 0;
    std::uint32_t new_canonical_count_ = 0;
};

struct PressureExpansionView {
    std::span<const PressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
};

class CapturePressurePlan {
public:
    CapturePressurePlan(CapturePressurePlan&&) noexcept            = default;
    CapturePressurePlan& operator=(CapturePressurePlan&&) noexcept = default;
    ~CapturePressurePlan()                                         = default;

    CapturePressurePlan(const CapturePressurePlan&)            = delete;
    CapturePressurePlan& operator=(const CapturePressurePlan&) = delete;

    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept {
        return revision_;
    }

private:
    CapturePressurePlan(AdmissionCandidate&& pressure,
                        runtime::ProgramResourceRevision revision) noexcept
        : pressure_(std::move(pressure)), revision_(revision) {}

    AdmissionCandidate pressure_;
    runtime::ProgramResourceRevision revision_;

    friend class Program;
    friend class PressurePlanningSession;
};

class AssessedPressureTarget {
public:
    AssessedPressureTarget(AssessedPressureTarget&& other) noexcept
        : session_(std::exchange(other.session_, nullptr)),
          session_generation_(std::exchange(other.session_generation_, 0)),
          target_index_(other.target_index_), assessment_(other.assessment_),
          assessment_slot_(std::exchange(other.assessment_slot_, 0)),
          assessment_slot_generation_(std::exchange(other.assessment_slot_generation_, 0)),
          release_slot_(std::exchange(other.release_slot_, nullptr)),
          executable_(std::move(other.executable_)),
          capture_executable_(std::move(other.capture_executable_)) {}

    ~AssessedPressureTarget() { reset(); }

    AssessedPressureTarget& operator=(AssessedPressureTarget&& other) noexcept {
        if (this == &other) { return *this; }
        reset();
        session_                    = std::exchange(other.session_, nullptr);
        session_generation_         = std::exchange(other.session_generation_, 0);
        target_index_               = other.target_index_;
        assessment_                 = other.assessment_;
        assessment_slot_            = std::exchange(other.assessment_slot_, 0);
        assessment_slot_generation_ = std::exchange(other.assessment_slot_generation_, 0);
        release_slot_               = std::exchange(other.release_slot_, nullptr);
        executable_                 = std::move(other.executable_);
        capture_executable_         = std::move(other.capture_executable_);
        return *this;
    }

    AssessedPressureTarget(const AssessedPressureTarget&)            = delete;
    AssessedPressureTarget& operator=(const AssessedPressureTarget&) = delete;

    [[nodiscard]] const runtime::PressureTargetAssessment& assessment() const noexcept {
        return assessment_;
    }

public:
    AssessedPressureTarget(
        const void* session, std::uint32_t session_generation, std::uint32_t target_index,
        runtime::PressureTargetAssessment assessment, std::uint32_t assessment_slot,
        std::uint32_t assessment_slot_generation,
        void (*release_slot)(const void*, std::uint32_t, std::uint32_t) noexcept,
        std::optional<AdmissionCandidate>&& executable,
        std::optional<CapturePressurePlan>&& capture_executable) noexcept
        : session_(session), session_generation_(session_generation), target_index_(target_index),
          assessment_(assessment), assessment_slot_(assessment_slot),
          assessment_slot_generation_(assessment_slot_generation), release_slot_(release_slot),
          executable_(std::move(executable)), capture_executable_(std::move(capture_executable)) {}

    void reset() noexcept {
        if (session_ != nullptr && release_slot_ != nullptr) {
            release_slot_(session_, assessment_slot_, assessment_slot_generation_);
        }
        session_                    = nullptr;
        session_generation_         = 0;
        assessment_slot_            = 0;
        assessment_slot_generation_ = 0;
        release_slot_               = nullptr;
    }

    const void* session_              = nullptr;
    std::uint32_t session_generation_ = 0;
    std::uint32_t target_index_       = 0;
    runtime::PressureTargetAssessment assessment_;
    std::uint32_t assessment_slot_                                            = 0;
    std::uint32_t assessment_slot_generation_                                 = 0;
    void (*release_slot_)(const void*, std::uint32_t, std::uint32_t) noexcept = nullptr;
    std::optional<AdmissionCandidate> executable_;
    std::optional<CapturePressurePlan> capture_executable_;

    friend class PressurePlanningSession;
    friend class detail::PressurePlanningSessionImpl;
};

class PressurePlanningSession {
public:
    PressurePlanningSession(PressurePlanningSession&&) noexcept;
    PressurePlanningSession& operator=(PressurePlanningSession&&) noexcept;
    ~PressurePlanningSession();

    PressurePlanningSession(const PressurePlanningSession&)            = delete;
    PressurePlanningSession& operator=(const PressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle
    identity_target(runtime::PlanningCandidateId candidate) const;
    [[nodiscard]] PressureTargetHandle
    root_maximal_target(runtime::PlanningCandidateId root_candidate);
    [[nodiscard]] std::optional<PressureTargetHandle>
    guided_closure_target(runtime::PlanningCandidateId candidate,
                          std::span<const runtime::PlanningOwnerId> preferred_owner_ids);
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target);
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target);
    void retain_assessment(PressureTargetHandle target);
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle parent);
    // Read by `SharedCapturePlanner` (`context_cache/shared_capture_planner.h:172-173`) to bound a
    // batch commit: `min(remaining, session.optional_targets_remaining())`. Fusion answers "canonical
    // target slots the arena can still hold" against its own arena constant
    // (`models/qwen3_5/program/planning/pressure_planner.cpp:809-816`). This model has no arena
    // bound at all -- its session enumerates expansions by flipping `owner_evicted` bits
    // (`impl/program.cpp:1096-1124`) -- so the binding quantity here is how many owners are left to
    // evict. Borrowing fusion's 4096 constant would be a second home for a fact this model does not
    // have; see the implementation for the derivation.
    [[nodiscard]] std::uint32_t optional_targets_remaining() const noexcept;
    // The seal window, forwarded to the Program. Read by `SealWindowClaim`
    // (`context_cache/seal_window_claim.h:20/37`), which claims it before sealing its selected
    // target so a concurrent demote cannot bump a victim's slot generation between the final assess
    // and the seal. Fusion exposes the same pair at the same level
    // (`models/qwen3_5/program/program.h:733-734`); here it is a thin forward to the Program's own
    // flag, so there is one owner of the claim for the whole Program rather than one per session.
    [[nodiscard]] bool try_claim_seal_window() noexcept;
    void release_seal_window() noexcept;
    [[nodiscard]] PressureExpansionView
    commit_expansion(PreparedPressureExpansion&& prepared);
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AssessedPressureTarget& assessed,
                                      const qwen3_5::PreparedPrompt& prompt,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<ResourcePlan>
    seal(AssessedPressureTarget&& assessed, const qwen3_5::PreparedPrompt& prompt,
         runtime::FinalScheduleIntent intent = {});
    [[nodiscard]] std::optional<CapturePressurePlan>
    seal_capture(AssessedPressureTarget&& assessed);

public:
    explicit PressurePlanningSession(
        std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept;
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl_;
};

class CapturePressurePlanningSession {
public:
    CapturePressurePlanningSession(CapturePressurePlanningSession&&) noexcept            = default;
    CapturePressurePlanningSession& operator=(CapturePressurePlanningSession&&) noexcept = default;
    ~CapturePressurePlanningSession()                                                    = default;

    CapturePressurePlanningSession(const CapturePressurePlanningSession&)            = delete;
    CapturePressurePlanningSession& operator=(const CapturePressurePlanningSession&) = delete;

    [[nodiscard]] PressureTargetHandle identity_target() const {
        return session_.identity_target(candidate_id());
    }
    [[nodiscard]] runtime::PressureTargetGuidance guidance(PressureTargetHandle target) {
        return session_.guidance(target);
    }
    [[nodiscard]] AssessedPressureTarget assess(PressureTargetHandle target) {
        return session_.assess(target);
    }
    [[nodiscard]] PreparedPressureExpansion prepare_expansion(PressureTargetHandle parent) {
        return session_.prepare_expansion(parent);
    }
    // Forwarded like every other operation of this session (`runtime.h` above). The bound is the
    // inner session's; see its own declaration for the derivation.
    [[nodiscard]] std::uint32_t optional_targets_remaining() const noexcept {
        return session_.optional_targets_remaining();
    }
    // Forwarded for the same reason. `SealWindowClaim` is templated on whatever session type it is
    // given (`seal_window_claim.h:15-16`), so the capture path needs the pair too.
    [[nodiscard]] bool try_claim_seal_window() noexcept {
        return session_.try_claim_seal_window();
    }
    void release_seal_window() noexcept { session_.release_seal_window(); }
    [[nodiscard]] PressureExpansionView
    commit_expansion(PreparedPressureExpansion&& prepared) {
        return session_.commit_expansion(std::move(prepared));
    }
    void discard_expansion(PreparedPressureExpansion&& prepared) noexcept {
        session_.discard_expansion(std::move(prepared));
    }
    [[nodiscard]] std::optional<CapturePressurePlan>
    seal(AssessedPressureTarget&& assessed) {
        return session_.seal_capture(std::move(assessed));
    }

    [[nodiscard]] static constexpr runtime::PlanningCandidateId candidate_id() noexcept {
        return runtime::PlanningCandidateId{.value = 0};
    }

private:
    explicit CapturePressurePlanningSession(PressurePlanningSession&& session) noexcept
        : session_(std::move(session)) {}

    PressurePlanningSession session_;

    friend class Program;
};

struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<CaptureOffer> capture;
    // Read at `engine_core.h:1524` (`request->first_token_logprobs = std::move(progress.
    // first_token_logprobs)`). Fusion has had it since `models/qwen3_5/program/program.h:858`;
    // the ported Program does not fill it, so it stays empty and the engine stores nothing.
    // Type comes from `ninfer/types.h`, already included at the top of this header.
    std::optional<FirstTokenLogprobs> first_token_logprobs;
};

struct SharedPrefixPublication {
    SharedPrefixHandle handle;
    SharedPrefixSummary summary;
};

struct ActiveCaptureResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed      = false;
    ContinuationSummary active_summary;
    std::optional<SharedPrefixPublication> shared;
    std::vector<qwen3_5::MaterializationVictimResult> victims;
    std::vector<qwen3_5::MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
};

struct StartResult {
    SequenceHandle sequence;
};

struct MaterializationResult {
    runtime::ContextTransactionStatus status = runtime::ContextTransactionStatus::Aborted;
    std::optional<StartResult> published;
    std::optional<qwen3_5::MaterializationSourceResult> source;
    std::optional<qwen3_5::MaterializationSharedSourceResult> shared_source;
    std::vector<qwen3_5::MaterializationVictimResult> victims;
    std::vector<qwen3_5::MaterializationSharedVictimResult> shared_victims;
    std::vector<runtime::ContextTransferObservation> transfer_observations;
    runtime::ContextOperationCounts operations;
    // Read at `engine_core.h:1680/1687` and `context_cache/hybrid_resource_manager.h:196-197`;
    // fusion carries both since `models/qwen3_5/program/program.h:941-945`. Empty/zero here means
    // "the ported path reported nothing", which is what the engine then passes through -- it is not
    // a fabricated success. Types come from `ninfer/types.h`, included at the top of this header.
    std::string failure;
    MaterializationDiagnostics diagnostics;
};

using ContextTransactionProgress =
    std::variant<runtime::ContextTransactionInProgress, MaterializationResult, ActiveCaptureResult>;

struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::array<std::optional<CaptureOffer>, kMaximumConcurrency> captures{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};

struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count         = 0;
};

struct FinishResult {
    runtime::ConsumeStatus status          = runtime::ConsumeStatus::InvariantMismatch;
    runtime::FinishDisposition disposition = runtime::FinishDisposition::Released;
    GenerationTimings timings;
    SpeculativeStats speculative;
    // Read at `engine_core.h:1101`, and placed where the fusion struct places it
    // (`models/qwen3_5/program/program.h:978`). Default and never set, for the same reason as in
    // `AbortResult` below: the ported Program does not publish live state as a salvaged continuation
    // endpoint, so `false` is the truthful value.
    bool salvaged = false;
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    // This engine's `AbortResult` carries three fields this facade does not have
    // (`models/qwen3_5/program/program.h:983-992`). All three are required by name for
    // `ResourceManager<TargetContract>` to instantiate, and all three are correct here as empty:
    //
    //   `salvaged`     read at `engine_core.h:1132` and
    //                  `context_cache/resource_manager.h:1267/1351`; it gates everything else.
    //   `summary`      read at `resource_manager.h:1224/1235/1241`.
    //   `continuation` read at `resource_manager.h:1224/1227-1229/1236-1237`.
    //
    // The reader of summary/continuation is `publish_salvaged_continuation`
    // (`resource_manager.h:1221`), which its own call site reaches only under
    // `if (discarded.salvaged)`. The ported Program's `abort` discards the lane outright
    // (`impl/program.cpp:2941` -> `drop_unpublished_checkpoints` + `release_lane`) and never
    // publishes a salvaged endpoint, so `salvaged` stays false, that function never runs, and
    // these two are never read. Note `FinishResult` above does carry a real summary/continuation,
    // because finishing does publish them.
    bool salvaged = false;
    ContinuationSummary summary;
    std::optional<ContinuationHandle> continuation;
};

struct ReleaseResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
};

class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::uint32_t continuation_capacity() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;

public:
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl> impl_;
};

class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] runtime::SequenceCapacityCurve capacity_curve() const;
    [[nodiscard]] SequencePlan finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl> impl_;
};

class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    [[nodiscard]] RequestBasePlan
    plan_request(const qwen3_5::PreparedPrompt& prompt,
                 const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] std::optional<AdmissionCandidate>
    inspect_admission(const qwen3_5::PreparedPrompt& prompt, const RequestBasePlan& base,
                      runtime::LaneId destination, const ContinuationHandle* source,
                      const SharedPrefixHandle* shared_source,
                      std::optional<runtime::CheckpointRef> checkpoint,
                      bool must_retain_private_source);
    [[nodiscard]] std::optional<ResourcePlan>
    seal_identity(const AdmissionCandidate& candidate, const qwen3_5::PreparedPrompt& prompt,
                  runtime::FinalScheduleIntent intent = {});
    // `recency_order` is the fusion engine's recency ladder input (owners ranked by latest hit or
    // publication, most recent first). Fusion's engine passes it at
    // `context_cache/materialization_planner.h:294-296`; this model's session does its own selection
    // through `guided_closure_target` / `guidance` / `assess` and has no such parameter -- upstream
    // declares six of them too (`src/targets/qwen3_8_flash_next/export/.../runtime.h:743` has no
    // recency input anywhere). The parameter is accepted so the engine's call matches, and is NOT
    // read: see `impl/program.cpp` for how the session ranks instead.
    [[nodiscard]] PressurePlanningSession
    begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                            std::span<const runtime::PlanningCandidateId> candidate_ids,
                            std::span<const ContinuationHandle* const> private_owners,
                            std::span<const runtime::PlanningOwnerId> private_owner_ids,
                            std::span<const SharedPrefixHandle* const> shared_owners,
                            std::span<const runtime::PlanningOwnerId> shared_owner_ids,
                            std::span<const runtime::PlanningOwnerId> recency_order);
    [[nodiscard]] runtime::PrefillWork
    shared_capture_split_prefill_work(const AdmissionCandidate& candidate,
                                      const qwen3_5::PreparedPrompt& prompt,
                                      std::span<const std::uint32_t> frontiers);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    start_resource_transaction(ResourcePlan&& plan, qwen3_5::PreparedPrompt&& prompt,
                               runtime::CancellationFlagView cancellation);
    [[nodiscard]] std::optional<PersistentBackfillProof>
    prove_persistent_backfill(const RequestBasePlan& blocked_head,
                              const ResourcePlan& candidate,
                              std::span<const SequenceHandle> persistent_borrowers) const;
    [[nodiscard]] ContextTransactionProgress
    progress_context_transaction(runtime::CancellationFlagView cancellation);
    void finalize_context_transaction() noexcept;
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] PrefillProgress
    advance_prefill(SequenceHandle sequence,
                    runtime::ExecutionTiming* failed_timing = nullptr);
    void select_shared_captures(ResourcePlan& /*plan*/, const qwen3_5::PreparedPrompt& /*prompt*/,
                                std::span<const std::uint32_t> /*frontiers*/) {}
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const ContinuationHandle& /*owner*/,
                             runtime::CheckpointRef /*checkpoint*/) const {
        return {};
    }
    [[nodiscard]] std::vector<runtime::CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const SharedPrefixHandle& /*owner*/,
                             runtime::CheckpointRef /*checkpoint*/) const {
        return {};
    }
    [[nodiscard]] CaptureAssessment
    inspect_capture(const CaptureOffer& offer,
                    const SharedPrefixHandle* exact_shared = nullptr,
                    const SharedPrefixHandle* replacement = nullptr,
                    std::optional<runtime::CheckpointRef> private_replacement = std::nullopt,
                    bool permit_shared_publication = false) const;
    [[nodiscard]] CapturePressurePlanningSession begin_capture_pressure_planning(
        const CaptureAssessment& assessment,
        std::span<const ContinuationHandle* const> private_owners,
        std::span<const runtime::PlanningOwnerId> private_owner_ids,
        std::span<const SharedPrefixHandle* const> shared_owners,
        std::span<const runtime::PlanningOwnerId> shared_owner_ids,
        std::span<const runtime::PlanningOwnerId> recency_order);
    [[nodiscard]] bool shared_capture_matches(const CaptureOffer& offer,
                                              const SharedPrefixHandle& shared) const;
    void skip_capture(CaptureOffer&& offer);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer,
                           const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           bool permit_shared_publication,
                           runtime::CancellationFlagView cancellation);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    reserve_active_capture(CaptureOffer&& offer,
                           const SharedPrefixHandle* exact_shared,
                           const SharedPrefixHandle* replacement,
                           std::optional<runtime::CheckpointRef> private_replacement,
                           runtime::CancellationFlagView cancellation) {
        return reserve_active_capture(std::move(offer), exact_shared, replacement,
                                      private_replacement, false, cancellation);
    }
    [[nodiscard]] runtime::ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
        const SharedPrefixHandle* replacement,
        std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
        CapturePressurePlan&& pressure,
        runtime::CancellationFlagView cancellation);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle> sequences,
                                      std::span<const runtime::RoundBudget> budgets,
                                      runtime::ExecutionTiming* failed_timing = nullptr);
    // `prefix_execution_splits` is one entry per sequence: the token count at which the row's
    // execution is split. It exists in this engine's fusion Program but NOT in either reference
    // target (upstream `qwen3_6` `runtime.h:926` and `qwen3_8_flash_next` `runtime.h:743` both take
    // four arguments) -- this engine's `EngineCore` passes five
    // (`engine_core.h:2157-2161`, from `CommitDecision::prefix_execution_split_after`). The ported
    // implementation cannot honour a non-empty split, so it validates the span and refuses a
    // non-zero entry rather than silently running the row whole (see `impl/program.cpp`).
    [[nodiscard]] runtime::ExecutionTiming
    append_forced_tokens(std::span<const SequenceHandle> sequences,
                         std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                         std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                         runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] CommitResult
    commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
           runtime::CommitObservation observation  = runtime::CommitObservation::AllRows,
           runtime::ExecutionTiming* failed_timing = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&& pending) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle sequence) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle sequence) noexcept;

    // The engine writes `decltype(instance_.program->fail_all_cleanup()) leaked;`
    // (`engine_core.h:2260`) and then picks between it and `shutdown_cleanup` by cleanup kind, so the
    // pair must hand back the same type. Upstream's target returned void because the engine that
    // shipped it had no leak report; this one does, so `fail_all_cleanup` is widened below to match
    // `shutdown_cleanup` and `nullopt` means "cleanup left nothing behind".
    [[nodiscard]] std::optional<PhysicalUsageSnapshot> shutdown_cleanup() noexcept;

    // Vision, sampling and the KV-lease controls below are engine-side controls the ported core does
    // not implement. Each one is answered in the direction that makes the engine's own guard skip the
    // step, so a call is inert rather than wrong; the guard that makes each answer unreachable is
    // named at its definition in `program.cpp`. They are declared here because the engine's
    // ModelContract must be satisfiable at compile time whether or not a feature is switched on.
    [[nodiscard]] bool vision_pending(SequenceHandle sequence) const noexcept;
    void update_sampling(SequenceHandle sequence, const ResolvedSamplingParameters& sampling);
    [[nodiscard]] bool wait_context_transfer() noexcept;
    [[nodiscard]] std::optional<DeviceKVLeaseShortfall>
    device_kv_lease_shortfall(SequenceHandle sequence) const noexcept;
    [[nodiscard]] bool resume_device_kv_lease(SequenceHandle sequence) noexcept;
    [[nodiscard]] std::optional<std::uint32_t>
    device_kv_lease_settlement_tokens(SequenceHandle sequence,
                                      std::uint32_t forced_span_tokens) const noexcept;
    [[nodiscard]] DeviceKVPages
    retained_device_kv_pages(const ContinuationHandle& continuation) const noexcept;
    // Engine-side `ResourceManager` calls this for BOTH kinds of catalog entry -- private ones carry
    // a `ContinuationHandle` and shared ones a `SharedPrefixHandle`
    // (`context_cache/resource_manager.h:1194` and `:1200`). Fusion declares both overloads
    // (`models/qwen3_5/program/program.h:1123,1125`); the ported header had only the first, so
    // instantiating `ResourceManager<TargetContract>` failed on the shared branch. Added here as the
    // exact mirror: same body, same answer.
    [[nodiscard]] DeviceKVPages
    retained_device_kv_pages(const SharedPrefixHandle& shared) const noexcept;

    // Causal scoring drives the perplexity tool, not generation. The engine keeps a scoring core for
    // it; a target without the path reports an empty distribution instead of a wrong one.
    [[nodiscard]] std::vector<float> causal_score(qwen3_5::PreparedPrompt&& prompt,
                                                  std::uint32_t first_target);

    // Hybrid prefix cache. The engine reaches these only through `HybridResourceManager`, which is
    // selected when `context_cache.enabled`. A target with no hybrid tier must answer
    // `PermanentlyInfeasible` rather than `TemporarilyBlocked`: the latter would park every request
    // in the admission queue forever instead of declining the mode. `main_page_groups` is the
    // resolved KV size the engine already computed, and the engine only ever writes it back.
    [[nodiscard]] HybridAdmissionQuote hybrid_quote(const qwen3_5::PreparedPrompt& prompt,
                                                    const RequestBasePlan& base,
                                                    runtime::LaneId destination);
    [[nodiscard]] runtime::ContextTransactionReserveStatus
    hybrid_reserve_materialization(HybridAdmissionQuote&& quote, qwen3_5::PreparedPrompt&& prompt,
                                   runtime::CancellationFlagView cancellation);
    void set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& cost);
    void set_hybrid_coalesce_wait_limit(double seconds);
    [[nodiscard]] qwen3_5::HybridCachePersistence
    attach_hybrid_cache_file(const std::filesystem::path& path, std::string fingerprint);
    [[nodiscard]] std::optional<qwen3_5::HybridCachePersistence> hybrid_shutdown_save() const;
    [[nodiscard]] std::uint32_t hybrid_reclaim_device_kv(std::uint32_t main_pages,
                                                         std::uint32_t backend_pages);
    [[nodiscard]] qwen3_5::HybridPrefixCacheStats hybrid_stats() const noexcept;
    [[nodiscard]] ReleaseResult
    release_continuation(ContinuationHandle&& continuation) noexcept;
    [[nodiscard]] ReleaseResult
    release_shared_prefix(SharedPrefixHandle&& shared) noexcept;
    [[nodiscard]] std::optional<PhysicalUsageSnapshot> fail_all_cleanup() noexcept;

    [[nodiscard]] bool
    isolated_request_feasible(const RequestBasePlan& base) const noexcept;
    [[nodiscard]] runtime::ProgramResourceRevision resource_revision() const noexcept;
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

public:
    explicit Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl> impl_;
};

class ContractAccess {
public:
    [[nodiscard]] static PendingBatch
    make_pending(const void* owner, std::uint64_t transaction,
                 std::span<const SequenceHandle> rows, std::span<const TokenId> tokens,
                 std::span<const std::int32_t> row_counts, std::uint32_t row_stride,
                 runtime::ExecutionTiming timing) {
        PendingBatch out;
        out.owner_       = owner;
        out.transaction_ = transaction;
        out.row_count_   = rows.size();
        for (std::size_t i = 0; i < rows.size(); ++i) { out.rows_[i] = rows[i]; }
        out.tokens_     = tokens;
        out.row_counts_ = row_counts;
        out.row_stride_ = row_stride;
        out.timing_     = timing;
        return out;
    }

    [[nodiscard]] static const void* owner(const PendingBatch& pending) noexcept {
        return pending.owner_;
    }

    [[nodiscard]] static std::uint64_t transaction(const PendingBatch& pending) noexcept {
        return pending.transaction_;
    }

    [[nodiscard]] static std::span<const SequenceHandle>
    rows(const PendingBatch& pending) noexcept {
        return {pending.rows_.data(), pending.row_count_};
    }

    static void consume(PendingBatch& pending) noexcept {
        pending.owner_       = nullptr;
        pending.transaction_ = 0;
        pending.row_count_   = 0;
        pending.tokens_      = {};
        pending.row_counts_  = {};
        pending.row_stride_  = 0;
        pending.timing_      = {};
    }

    [[nodiscard]] static CaptureOffer make_capture_offer(const void* owner,
                                                         runtime::LaneId lane,
                                                         std::uint64_t epoch,
                                                         std::uint64_t id) noexcept {
        CaptureOffer out;
        out.owner_ = owner;
        out.lane_  = lane;
        out.epoch_ = epoch;
        out.id_    = id;
        return out;
    }

    [[nodiscard]] static const void* owner(const CaptureOffer& offer) noexcept {
        return offer.owner_;
    }

    [[nodiscard]] static runtime::LaneId lane(const CaptureOffer& offer) noexcept {
        return offer.lane_;
    }

    [[nodiscard]] static std::uint64_t epoch(const CaptureOffer& offer) noexcept {
        return offer.epoch_;
    }

    [[nodiscard]] static std::uint64_t id(const CaptureOffer& offer) noexcept {
        return offer.id_;
    }

    static void consume(CaptureOffer& offer) noexcept {
        offer.owner_ = nullptr;
        offer.lane_  = {};
        offer.epoch_ = 0;
        offer.id_    = 0;
    }

    [[nodiscard]] static AdmissionCandidate
    take_admission(ResourcePlan& plan) noexcept {
        return std::move(plan.admission_);
    }

    [[nodiscard]] static const AdmissionCandidate&
    admission(const ResourcePlan& plan) noexcept {
        return plan.admission_;
    }
};

} // namespace ninfer::models::qwen3_8_flash_next
