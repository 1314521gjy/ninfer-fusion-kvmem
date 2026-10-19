#include "models/qwen3_8_flash_next/impl/program_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime_plan.h"
#include <ninfer/models/qwen3_8_flash_next/package.h>
#include "runtime/engine/context_cache/context_cost.h"

#include "models/qwen3_8_flash_next/impl/load/materialized.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

// Upstream writes this as an explicit specialization of the variant template
// (`template <> struct PressurePlanningSessionImpl<FlashNextPressureHandleAccess>`), because there
// the session *is* `PressurePlanningSessionImpl<Variant>` and the variant supplies the handle
// packing. This tree's session is a concrete class that packs handles itself, so the variant's
// contribution is kept as the policy struct alone -- same four operations, same names, reached as
// `FlashNextPressureHandleHelper` below. `PressureTargetHandle` friends this exact type, which is
// what lets it touch the private session_/generation_/index_ fields.
namespace ninfer::models::qwen3_5::detail {

struct FlashNextPressureHandleAccess {
    static PressureTargetHandle make_handle(const void* session, std::uint32_t generation,
                                            std::uint32_t index) noexcept {
        PressureTargetHandle h;
        h.session_    = session;
        h.generation_ = generation;
        h.index_      = index;
        return h;
    }
    static const void* get_session(const PressureTargetHandle& h) noexcept { return h.session_; }
    static std::uint32_t get_generation(const PressureTargetHandle& h) noexcept {
        return h.generation_;
    }
    static std::uint32_t get_index(const PressureTargetHandle& h) noexcept { return h.index_; }
};
} // namespace ninfer::models::qwen3_5::detail

namespace ninfer::models::qwen3_8_flash_next::detail {

using FlashNextPressureHandleHelper = ninfer::models::qwen3_5::detail::FlashNextPressureHandleAccess;

ProgramImpl::ProgramImpl(const LoadedModelData* model_data, FlashNextRuntimePlan plan_in,
                         DeviceContext& dev, TextModelView text_override,
                         std::optional<VisionModelView> vision_override,
                         PleIndexMetadata ple_override)
    : model_data_(model_data), text_override_(std::move(text_override)),
      vision_override_(std::move(vision_override)),
      plan_(std::move(plan_in)), device_(dev), allocation_(plan_),
      executor_(model_data_ != nullptr ? model_data_->text : text_override_,
                model_data_ != nullptr ? kPleIndexMetadata : ple_override,
                device_, allocation_),
      lane_states_(plan_.config.max_concurrency),
      sampling_workspace_(std::max<std::size_t>(
          256,
          ops::sampling_workspace_capacity_bytes(
              248'077, 1, static_cast<std::int32_t>(plan_.config.max_concurrency)))),
      device_sampling_configs_(plan_.config.max_concurrency * sizeof(ops::SamplingConfig)),
      device_sampling_positions_(plan_.config.max_concurrency * sizeof(std::int32_t)),
      device_sampled_tokens_(plan_.config.max_concurrency * sizeof(std::int32_t)),
      host_sampled_tokens_(plan_.config.max_concurrency, 0),
      host_sampling_configs_(plan_.config.max_concurrency),
      host_sampling_positions_(plan_.config.max_concurrency, 0),
      device_constraint_masks_(plan_.config.max_concurrency * kConstraintColumns * kConstraintMaskWords * sizeof(std::int32_t)),
      host_constraint_masks_(plan_.config.max_concurrency * kConstraintColumns * kConstraintMaskWords) {
    allocation_.initialize(device_.stream);
    const std::uint32_t cont_cap = plan_.config.continuation_capacity;
    continuation_slots_.resize(cont_cap);
    for (std::uint32_t c = 0; c < cont_cap; ++c) {
        continuation_slots_[c].cache_slot =
            flash_next_floor_slots(plan_.config.max_concurrency, plan_.config.speculative_draft_tokens) + c;
    }
    if (model_data_ != nullptr && model_data_->vision.has_value() && plan_.config.vision_enabled) {
        if (!plan_.vision_workspace.has_value()) {
            throw std::logic_error("Vision enabled but plan has no vision workspace");
        }
        DeviceSpan ws{allocation_.workspace().base(), allocation_.workspace().capacity()};
        vision_session_.emplace(*model_data_->vision, device_, ws, *plan_.vision_workspace,
                                plan_.config.max_vision_tokens);
    } else if (vision_override_.has_value() && plan_.config.vision_enabled) {
        if (!plan_.vision_workspace.has_value()) {
            throw std::logic_error("Vision enabled but plan has no vision workspace");
        }
        DeviceSpan ws{allocation_.workspace().base(), allocation_.workspace().capacity()};
        vision_session_.emplace(*vision_override_, device_, ws, *plan_.vision_workspace,
                                plan_.config.max_vision_tokens);
    }
}

const std::int32_t* ProgramImpl::upload_constraint_mask(std::uint32_t lane, std::uint32_t column,
                                                       runtime::OutputConstraintState& constraint) {
    const auto mask = constraint.next_mask();
    if (mask.size() != kConstraintMaskWords || column >= kConstraintColumns) {
        throw std::logic_error("structured output mask has an invalid shape");
    }
    const auto offset = (lane * kConstraintColumns + column) * kConstraintMaskWords;
    auto* host = host_constraint_masks_.data() + offset;
    auto* device = static_cast<std::int32_t*>(device_constraint_masks_.p) + offset;
    std::copy(mask.begin(), mask.end(), host);
    CUDA_CHECK(cudaMemcpyAsync(device, host, mask.size_bytes(), cudaMemcpyHostToDevice, device_.stream));
    return device;
}

void ProgramImpl::sample_tokens(const Tensor& logits,
                                std::span<const std::uint32_t> lane_indices,
                                std::span<std::int32_t> out_tokens) {
    const std::size_t B = lane_indices.size();
    if (B == 0 || B > plan_.config.max_concurrency || out_tokens.size() < B) {
        throw std::invalid_argument("sample_tokens: batch size is invalid");
    }

    for (std::size_t b = 0; b < B; ++b) {
        const std::uint32_t lane_idx = lane_indices[b];
        auto& st                     = lane_states_[lane_idx];
        st.sampling_config.allowed_tokens = st.output_constraint
            ? upload_constraint_mask(lane_idx, 0, *st.output_constraint) : nullptr;
        host_sampling_configs_[b]   = st.sampling_config;
        host_sampling_positions_[b] = st.last_token_pos;
    }

    CUDA_CHECK(cudaMemcpyAsync(device_sampling_configs_.p, host_sampling_configs_.data(),
                               B * sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
                               device_.stream));
    CUDA_CHECK(cudaMemcpyAsync(device_sampling_positions_.p, host_sampling_positions_.data(),
                               B * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               device_.stream));

    Tensor out_tensor(device_sampled_tokens_.p, DType::I32, {static_cast<std::int32_t>(B)});
    Tensor pos_tensor(device_sampling_positions_.p, DType::I32, {static_cast<std::int32_t>(B)});

    constexpr std::int32_t kSemanticTokenDomain = 248'077;
    ops::sample(logits, out_tensor, kSemanticTokenDomain,
                static_cast<const ops::SamplingConfig*>(device_sampling_configs_.p), pos_tensor,
                ops::kSamplePurposeDecode, sampling_workspace_, device_.stream);

    // Decode/serve sampled-token D2H; not on the prefill chunk path.
    CUDA_CHECK(cudaMemcpyAsync(out_tokens.data(), device_sampled_tokens_.p,
                               B * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               device_.stream));
    CUDA_CHECK(cudaStreamSynchronize(device_.stream));
}

bool ProgramImpl::is_slot_protected(std::size_t slot_idx) const {
    if (slot_idx >= continuation_slots_.size()) { return false; }
    const auto& slot = continuation_slots_[slot_idx];
    if (slot.role != ContinuationSlotRole::Catalogued) { return false; }

    for (const auto& st : lane_states_) {
        if (!st.active) { continue; }
        if (st.turn_closure_continuation_index == slot_idx) { return true; }
        if (!st.reused_from_continuation_index.has_value()) { continue; }
        const auto owner_idx = *st.reused_from_continuation_index;
        const auto& owner = continuation_slots_[owner_idx];
        if (owner.role != ContinuationSlotRole::Catalogued ||
            st.reused_from_continuation_generation != owner.generation) {
            continue;
        }
        if (owner_idx == slot_idx ||
            (owner.paired_rewrite_slot == slot_idx &&
             owner.paired_rewrite_generation == slot.generation)) {
            return true;
        }
    }

    return false;
}

std::uint32_t ProgramImpl::count_freeable_physical_groups(
    std::optional<std::size_t> current_matching_slot) const {
    std::vector<bool> unfreeable(plan_.main_page_groups, false);

    for (const auto& st : lane_states_) {
        if (st.active) {
            for (const auto g : executor_.lane_physical_groups(st.lane_handle)) {
                if (g < unfreeable.size()) {
                    unfreeable[g] = true;
                }
            }
        }
    }

    for (std::size_t c = 0; c < continuation_slots_.size(); ++c) {
        const auto& slot = continuation_slots_[c];
        if (slot.role != ContinuationSlotRole::Catalogued) {
            continue;
        }
        if (current_matching_slot.has_value() && *current_matching_slot == c) {
            for (const auto g : slot.physical_groups) {
                if (g < unfreeable.size()) {
                    unfreeable[g] = true;
                }
            }
            continue;
        }
        if (is_slot_protected(c)) {
            for (const auto g : slot.physical_groups) {
                if (g < unfreeable.size()) {
                    unfreeable[g] = true;
                }
            }
        }
    }

    std::uint32_t unfreeable_count = 0;
    for (bool b : unfreeable) {
        if (b) { unfreeable_count += 1; }
    }
    return plan_.main_page_groups >= unfreeable_count
               ? (plan_.main_page_groups - unfreeable_count)
               : 0U;
}

void ProgramImpl::vacate_slot(std::size_t slot_idx) {
    if (slot_idx >= continuation_slots_.size()) { return; }
    auto& c_slot = continuation_slots_[slot_idx];
    if (c_slot.role == ContinuationSlotRole::Vacant) { return; }
    executor_.release_physical_groups(c_slot.physical_groups);
    c_slot.physical_groups.clear();
    c_slot.committed_tokens.clear();
    c_slot.committed_frontier   = 0;
    c_slot.role                 = ContinuationSlotRole::Vacant;
    c_slot.generation++;
    c_slot.published_checkpoints = 1;
    c_slot.paired_rewrite_slot.reset();
    c_slot.paired_rewrite_generation = 0;
}

// An Engine owner is the endpoint slot plus the TurnClosure slot its summary published as the
// rewrite checkpoint; the Engine drops both when it evicts or releases the owner.
void ProgramImpl::vacate_owner(std::size_t slot_idx) {
    if (slot_idx >= continuation_slots_.size()) { return; }
    const auto paired     = continuation_slots_[slot_idx].paired_rewrite_slot;
    const auto paired_gen = continuation_slots_[slot_idx].paired_rewrite_generation;
    vacate_slot(slot_idx);
    if (paired.has_value() && *paired < continuation_slots_.size()) {
        const auto& turn = continuation_slots_[*paired];
        if (turn.role == ContinuationSlotRole::Catalogued && turn.generation == paired_gen &&
            !is_slot_protected(*paired)) {
            vacate_slot(*paired);
        }
    }
}

std::uint32_t ProgramImpl::reserved_unowned_groups() const noexcept {
    std::uint32_t reserved = 0;
    for (const auto& st : lane_states_) {
        if (!st.active) { continue; }
        const auto owned =
            static_cast<std::uint32_t>(executor_.lane_physical_groups(st.lane_handle).size());
        if (st.planned_groups > owned) { reserved += st.planned_groups - owned; }
    }
    return reserved;
}

std::uint32_t ProgramImpl::vacant_checkpoint_slots() const noexcept {
    return static_cast<std::uint32_t>(std::count_if(
        continuation_slots_.begin(), continuation_slots_.end(),
        [](const auto& slot) { return slot.role == ContinuationSlotRole::Vacant; }));
}

std::uint32_t ProgramImpl::owner_checkpoint_slots(std::size_t owner) const {
    const auto& slot = continuation_slots_.at(owner);
    if (slot.role != ContinuationSlotRole::Catalogued) { return 0; }
    if (slot.paired_rewrite_slot && *slot.paired_rewrite_slot < continuation_slots_.size()) {
        const auto& rewrite = continuation_slots_[*slot.paired_rewrite_slot];
        if (rewrite.role == ContinuationSlotRole::Catalogued &&
            rewrite.generation == slot.paired_rewrite_generation &&
            !is_slot_protected(*slot.paired_rewrite_slot)) {
            return 2;
        }
    }
    return 1;
}

void ProgramImpl::release_checkpoint_reservations(LaneState& st) {
    if (st.reserved_endpoint_index) {
        vacate_slot(*st.reserved_endpoint_index);
        st.reserved_endpoint_index.reset();
    }
    if (st.reserved_turn_closure_index) {
        vacate_slot(*st.reserved_turn_closure_index);
        st.reserved_turn_closure_index.reset();
    }
}

std::uint32_t ProgramImpl::published_checkpoints_of(std::size_t slot_idx) const noexcept {
    return slot_idx < continuation_slots_.size() ? continuation_slots_[slot_idx].published_checkpoints
                                                 : 1U;
}

// A lane that ends without cataloguing an endpoint (Released finish, abort, cancelled commit)
// loses its Engine entry, and with it the TurnClosure it captured: drop our copy unless a
// sibling lane is still resuming from it.
void ProgramImpl::drop_unpublished_checkpoints(LaneState& st) {
    release_checkpoint_reservations(st);
    if (!st.turn_closure_continuation_index.has_value()) { return; }
    const std::size_t idx = *st.turn_closure_continuation_index;
    if (idx >= continuation_slots_.size()) { return; }
    const auto& turn = continuation_slots_[idx];
    if (turn.role != ContinuationSlotRole::Catalogued ||
        turn.kind != runtime::CheckpointKind::TurnClosure) {
        return;
    }
    for (const auto& other : lane_states_) {
        if (&other == &st || !other.active || !other.reused_from_continuation_index.has_value()) {
            continue;
        }
        const auto& owner = continuation_slots_[*other.reused_from_continuation_index];
        if (owner.generation == other.reused_from_continuation_generation &&
            owner.paired_rewrite_slot == idx && owner.paired_rewrite_generation == turn.generation) {
            return;
        }
    }
    vacate_slot(idx);
}

void ProgramImpl::evict_continuation_slot(std::size_t slot_idx) {
    if (slot_idx >= continuation_slots_.size()) { return; }
    if (continuation_slots_[slot_idx].role != ContinuationSlotRole::Catalogued) { return; }
    vacate_owner(slot_idx);
    ++resource_revision_;
}

bool ProgramImpl::ensure_physical_groups_available(std::size_t needed) {
    if (executor_.available_physical_groups() >= needed) {
        return true;
    }
    std::fprintf(stderr, "[warn] flash_next: unexpected defensive page group eviction needed=%zu available=%zu\n",
                 needed, executor_.available_physical_groups());
    while (executor_.available_physical_groups() < needed) {
        std::int32_t best_c = -1;
        std::uint64_t oldest_epoch = std::numeric_limits<std::uint64_t>::max();

        for (std::size_t c = 0; c < continuation_slots_.size(); ++c) {
            const auto& slot = continuation_slots_[c];
            if (slot.role != ContinuationSlotRole::Catalogued) { continue; }
            if (slot.physical_groups.empty()) { continue; }
            if (is_slot_protected(c)) { continue; }

            if (slot.last_used_epoch < oldest_epoch) {
                oldest_epoch = slot.last_used_epoch;
                best_c       = static_cast<std::int32_t>(c);
            }
        }

        if (best_c < 0) {
            break;
        }

        evict_continuation_slot(static_cast<std::size_t>(best_c));
    }

    return executor_.available_physical_groups() >= needed;
}

std::int32_t ProgramImpl::allocate_vacant_continuation_slot() {
    const std::size_t cap = continuation_slots_.size();
    for (std::size_t c = 0; c < cap; ++c) {
        if (continuation_slots_[c].role == ContinuationSlotRole::Vacant) {
            return static_cast<std::int32_t>(c);
        }
    }
    return -1;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail

namespace ninfer::models::qwen3_8_flash_next {

static std::optional<std::uint32_t>
derive_turn_closure_frontier(const qwen3_5::PreparedPromptData& prompt_data);

// ---------------------------------------------------------------------------
// RequestBasePlan
// ---------------------------------------------------------------------------

RequestBasePlan::RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
RequestBasePlan::RequestBasePlan(RequestBasePlan&&) noexcept            = default;
RequestBasePlan& RequestBasePlan::operator=(RequestBasePlan&&) noexcept = default;
RequestBasePlan::~RequestBasePlan()                                     = default;

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept {
    static const runtime::RequestPlanSummary empty{};
    return impl_ != nullptr ? impl_->summary : empty;
}

const qwen3_5::PreparedContextCache& RequestBasePlan::context_cache() const noexcept {
    static const qwen3_5::PreparedContextCache empty{};
    return impl_ != nullptr ? impl_->context_cache : empty;
}

std::optional<PrefixShortlistKey>
RequestBasePlan::prefix_shortlist_key(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prefix_digests.size()) {
        return std::nullopt;
    }
    const auto lookup_digests = impl_->prefix_digests.at(frontier);
    if (std::getenv("NINFER_FLASH_NEXT_TRACE_KEYS") != nullptr) {
        std::fprintf(stderr, "[fnkey] lookup f=%u tag=%u d0=%016llx d1=%016llx\n", frontier,
                     impl_->prefix_identity_tag,
                     static_cast<unsigned long long>(lookup_digests[0]),
                     static_cast<unsigned long long>(lookup_digests[1]));
    }
    return PrefixShortlistKey{
        .digests      = lookup_digests,
        .frontier     = frontier,
        .identity_tag = impl_->prefix_identity_tag,
    };
}

std::size_t RequestBasePlan::prefix_shortlist_size() const noexcept {
    // The fusion answer is the same expression over its own impl
    // (`models/qwen3_5/program/program.cpp:101-103`), and this facade's impl carries the same
    // `prefix_digests` member the key accessor above reads.
    return impl_ != nullptr ? impl_->prefix_digests.size() : 0;
}

std::optional<runtime::PrefillWork>
RequestBasePlan::shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
    if (impl_ == nullptr || frontier == 0 || frontier > impl_->prefix_digests.size()) {
        return std::nullopt;
    }
    // Flash-Next publishes no shared prefixes, but the Engine still prepares shared
    // candidates from declared or repeated boundaries and requires their canonical
    // rebuild cost. Same formula as the checkpoint summaries.
    return runtime::make_prefill_work(0, frontier, 0, 0, impl_->prefill_chunk);
}

// ---------------------------------------------------------------------------
// AdmissionCandidate
// ---------------------------------------------------------------------------

AdmissionCandidate::AdmissionCandidate(
    std::unique_ptr<detail::AdmissionCandidateImpl> impl) noexcept
    : impl_(std::move(impl)) {}
AdmissionCandidate::AdmissionCandidate(AdmissionCandidate&&) noexcept            = default;
AdmissionCandidate& AdmissionCandidate::operator=(AdmissionCandidate&&) noexcept = default;
AdmissionCandidate::~AdmissionCandidate()                                         = default;

const runtime::RequestPlanSummary& AdmissionCandidate::summary() const noexcept {
    static const runtime::RequestPlanSummary empty{};
    return impl_ != nullptr ? impl_->summary : empty;
}

const runtime::IdentityMaterializationAssessment&
AdmissionCandidate::identity_assessment() const noexcept {
    static const runtime::IdentityMaterializationAssessment empty{};
    return impl_ != nullptr ? impl_->assessment : empty;
}

// ---------------------------------------------------------------------------
// ResourcePlan
// ---------------------------------------------------------------------------

ResourcePlan::ResourcePlan(AdmissionCandidate&& admission, runtime::ProgramResourceRevision revision,
                           bool needs_transfer) noexcept
    : admission_(std::move(admission)), revision_(revision), needs_transfer_(needs_transfer) {}
ResourcePlan::ResourcePlan(ResourcePlan&&) noexcept            = default;
ResourcePlan& ResourcePlan::operator=(ResourcePlan&&) noexcept = default;
ResourcePlan::~ResourcePlan()                                  = default;

const runtime::RequestPlanSummary& ResourcePlan::summary() const noexcept {
    return admission_.summary();
}

// ---------------------------------------------------------------------------
// PressurePlanningSession
// ---------------------------------------------------------------------------

namespace detail {

PressurePlanningSessionImpl::PressurePlanningSessionImpl(
    ProgramImpl& program,
    std::span<const AdmissionCandidate* const> candidates,
    std::span<const runtime::PlanningCandidateId> candidate_ids,
    std::span<const ContinuationHandle* const> private_owners,
    std::span<const runtime::PlanningOwnerId> private_owner_ids,
    std::span<const SharedPrefixHandle* const> shared_owners,
    std::span<const runtime::PlanningOwnerId> shared_owner_ids)
    : program_(&program),
      resource_revision_(program.resource_revision_) {
    if (candidates.empty() || candidates.size() != candidate_ids.size()) {
        throw std::logic_error("pressure planning session: candidates/candidate_ids empty or size mismatch");
    }
    if (private_owners.size() != private_owner_ids.size()) {
        throw std::logic_error("pressure planning session: private owners/owner_ids size mismatch");
    }
    if (shared_owners.size() != shared_owner_ids.size()) {
        throw std::logic_error("pressure planning session: shared owners/owner_ids size mismatch");
    }
    if (program.has_context_transaction_) {
        throw std::logic_error("pressure planning session: active context transaction in progress");
    }
    if (program.pressure_planning_active_) {
        throw std::logic_error("pressure planning session: another pressure planning session is already active");
    }

    candidates_.assign(candidates.begin(), candidates.end());
    candidate_ids_.assign(candidate_ids.begin(), candidate_ids.end());
    owners_.reserve(private_owners.size());
    for (std::size_t i = 0; i < private_owners.size(); ++i) {
        const ContinuationHandle* handle = private_owners[i];
        if (handle == nullptr) {
            throw std::logic_error("pressure planning private owner is null");
        }
        const std::uint32_t slot_idx = handle->index();
        const std::uint64_t gen = handle->generation();
        if (slot_idx >= program.continuation_slots_.size() ||
            program.continuation_slots_[slot_idx].role != ContinuationSlotRole::Catalogued ||
            program.continuation_slots_[slot_idx].generation != gen) {
            throw std::logic_error("pressure planning private owner is stale");
        }
        owners_.push_back(PressurePlanningOwner{
            .handle = handle,
            .owner_id = private_owner_ids[i],
            .continuation_index = slot_idx,
            .generation = gen,
        });
    }

    std::sort(owners_.begin(), owners_.end(), [](const auto& a, const auto& b) {
        return a.owner_id.value < b.owner_id.value;
    });
    for (std::size_t i = 1; i < owners_.size(); ++i) {
        if (owners_[i - 1].owner_id.value == owners_[i].owner_id.value) {
            throw std::logic_error("pressure planning owner ID is duplicated");
        }
    }

    targets_.reserve(candidates_.size() + 16);
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
        const AdmissionCandidate* cand = candidates_[i];
        if (cand == nullptr || cand->impl_ == nullptr ||
            cand->impl_->planning_revision != resource_revision_) {
            throw std::logic_error("pressure planning candidate is stale");
        }
        targets_.push_back(PressurePlanningTargetNode{
            .candidate_index = static_cast<std::uint32_t>(i),
            .owner_evicted = std::vector<std::uint8_t>(owners_.size(), 0),
            .stable_ordinal = static_cast<std::uint32_t>(i),
            .root_maximal = false,
        });
    }

    if (++program.pressure_planning_generation_ == 0) {
        ++program.pressure_planning_generation_;
    }
    session_generation_ = program.pressure_planning_generation_;
    program.pressure_planning_active_ = true;
}

PressurePlanningSessionImpl::~PressurePlanningSessionImpl() noexcept {
    if (program_ != nullptr) {
        program_->pressure_planning_active_ = false;
    }
}

bool PressurePlanningSessionImpl::valid(PressureTargetHandle target) const noexcept {
    return FlashNextPressureHandleHelper::get_session(target) == this &&
           FlashNextPressureHandleHelper::get_generation(target) == session_generation_ &&
           FlashNextPressureHandleHelper::get_index(target) < targets_.size() &&
           program_ != nullptr &&
           program_->resource_revision_ == resource_revision_;
}

std::uint32_t PressurePlanningSessionImpl::candidate_index(runtime::PlanningCandidateId candidate) const {
    const auto it = std::find(candidate_ids_.begin(), candidate_ids_.end(), candidate);
    if (it == candidate_ids_.end()) {
        throw std::invalid_argument("pressure target candidate does not belong to this session");
    }
    return static_cast<std::uint32_t>(it - candidate_ids_.begin());
}

PressureTargetHandle PressurePlanningSessionImpl::identity_target(runtime::PlanningCandidateId candidate) const {
    const std::uint32_t idx = candidate_index(candidate);
    return FlashNextPressureHandleHelper::make_handle(this, session_generation_, idx);
}

PressureTargetHandle PressurePlanningSessionImpl::root_maximal_target(runtime::PlanningCandidateId root_candidate) {
    if (scratch_live_) {
        throw std::logic_error("pressure expansion scratch is still live");
    }
    const std::uint32_t cand_idx = candidate_index(root_candidate);
    const auto& cand_impl = *candidates_[cand_idx]->impl_;

    PressurePlanningTargetNode maximal{
        .candidate_index = cand_idx,
        .owner_evicted = std::vector<std::uint8_t>(owners_.size(), 0),
        .stable_ordinal = 0,
        .root_maximal = true,
    };

    for (std::size_t i = 0; i < owners_.size(); ++i) {
        const auto& owner = owners_[i];
        bool is_protected = false;
        if (cand_impl.has_source &&
            cand_impl.source_continuation_index == owner.continuation_index &&
            cand_impl.source_continuation_generation == owner.generation) {
            is_protected = true;
        }
        if (program_->is_slot_protected(owner.continuation_index)) {
            is_protected = true;
        }
        if (!is_protected) {
            maximal.owner_evicted[i] = 1;
        }
    }

    auto it = std::find_if(targets_.begin(), targets_.end(), [&](const auto& node) {
        return node.candidate_index == maximal.candidate_index &&
               node.owner_evicted == maximal.owner_evicted;
    });

    std::uint32_t target_idx = 0;
    if (it != targets_.end()) {
        it->root_maximal = true;
        target_idx = static_cast<std::uint32_t>(it - targets_.begin());
    } else {
        maximal.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
        targets_.push_back(std::move(maximal));
        target_idx = static_cast<std::uint32_t>(targets_.size() - 1);
    }

    return FlashNextPressureHandleHelper::make_handle(this, session_generation_, target_idx);
}

AssessedPressureTarget PressurePlanningSessionImpl::assess(PressureTargetHandle target) {
    if (!valid(target) || scratch_live_) {
        throw std::logic_error("pressure target assessment is stale or conflicts with expansion");
    }
    const std::uint32_t target_idx = FlashNextPressureHandleHelper::get_index(target);
    const auto& node = targets_[target_idx];
    const auto& cand = *candidates_[node.candidate_index];
    const auto& details = *cand.impl_;

    const bool is_identity = std::all_of(node.owner_evicted.begin(), node.owner_evicted.end(),
                                         [](std::uint8_t v) { return v == 0; });

    assessment_outcomes_.clear();
    assessment_outcomes_.reserve(owners_.size());
    assessment_impacts_.clear();
    assessment_recovery_work_.clear();
    assessment_impacts_.reserve(owners_.size() * 2);
    assessment_recovery_work_.reserve(owners_.size() * 2);
    std::uint32_t total_degradation = 0;
    std::uint32_t total_dropped = 0;
    std::uint64_t projection_work = 1;

    for (std::size_t i = 0; i < owners_.size(); ++i) {
        if (node.owner_evicted[i] == 1) {
            const std::size_t slot_idx = owners_[i].continuation_index;
            // The Engine validates this against continuation_checkpoint_count(entry.summary).
            const std::uint32_t dropped = program_->published_checkpoints_of(slot_idx);
            assessment_outcomes_.push_back(runtime::PressureOwnerOutcome{
                .owner = owners_[i].owner_id,
                .disposition = runtime::VictimDisposition::Evicted,
                .degradation_units = 1,
                .dropped_checkpoints = dropped,
            });
            total_degradation += 1;
            total_dropped += dropped;
            projection_work += 1;

            if (slot_idx < program_->continuation_slots_.size()) {
                const auto& slot = program_->continuation_slots_[slot_idx];

                // 1. Endpoint checkpoint impact (SessionEndpoint)
                const std::uint32_t endpoint_frontier =
                    slot.committed_frontier > 0 ? static_cast<std::uint32_t>(slot.committed_frontier) : 0U;
                runtime::CheckpointRef endpoint_ref{
                    .kind = slot.kind,
                    .frontier = endpoint_frontier,
                    .ordinal = 0,
                };
                runtime::CheckpointRecoveryAlternativeWork endpoint_work{};
                endpoint_work.prefill = runtime::make_prefill_work(
                    0, endpoint_frontier, 0, 0, program_->plan_.config.prefill_chunk);
                endpoint_work.transfers = {};
                assessment_recovery_work_.push_back(endpoint_work);
                assessment_impacts_.push_back(runtime::PressureCheckpointRecoveryImpact{
                    .owner = owners_[i].owner_id,
                    .checkpoint = endpoint_ref,
                    .target_recovery_work = {},
                    .survives = false,
                });

                // 2. Paired TurnClosure checkpoint impact (if published as rewrite)
                if (dropped >= 2 && slot.paired_rewrite_slot.has_value()) {
                    const std::uint32_t turn_idx = *slot.paired_rewrite_slot;
                    if (turn_idx < program_->continuation_slots_.size()) {
                        const auto& turn_slot = program_->continuation_slots_[turn_idx];
                        const std::uint32_t turn_frontier =
                            turn_slot.committed_frontier > 0 ? static_cast<std::uint32_t>(turn_slot.committed_frontier) : 0U;
                        runtime::CheckpointRef rewrite_ref{
                            .kind = runtime::CheckpointKind::TurnClosure,
                            .frontier = turn_frontier,
                            .ordinal = 0,
                        };
                        runtime::CheckpointRecoveryAlternativeWork rewrite_work{};
                        rewrite_work.prefill = runtime::make_prefill_work(
                            0, turn_frontier, 0, 0, program_->plan_.config.prefill_chunk);
                        rewrite_work.transfers = {};
                        assessment_recovery_work_.push_back(rewrite_work);
                        assessment_impacts_.push_back(runtime::PressureCheckpointRecoveryImpact{
                            .owner = owners_[i].owner_id,
                            .checkpoint = rewrite_ref,
                            .target_recovery_work = {},
                            .survives = false,
                        });
                    }
                }
            }
        }
    }

    // Bind spans into assessment_recovery_work_ after all elements are inserted
    for (std::size_t k = 0; k < assessment_impacts_.size(); ++k) {
        assessment_impacts_[k].target_recovery_work =
            std::span<const runtime::CheckpointRecoveryAlternativeWork>(&assessment_recovery_work_[k], 1);
    }

    runtime::MaterializationPhysicalStatus status = runtime::MaterializationPhysicalStatus::Infeasible;
    if (is_identity) {
        status = details.assessment.physical_status;
    } else {
        std::uint32_t available = static_cast<std::uint32_t>(program_->executor_.available_physical_groups());
        {
            const std::uint32_t reserved = program_->reserved_unowned_groups();
            available = available > reserved ? available - reserved : 0U;
        }
        {
            // A group comes back only when every reference to it belongs to a slot this target
            // evicts; an Engine owner is its endpoint slot plus the paired TurnClosure slot.
            std::unordered_map<std::uint32_t, std::uint32_t> refs_in_target;
            const auto add_slot = [&](std::size_t s) {
                if (s >= program_->continuation_slots_.size()) { return; }
                const auto& slot = program_->continuation_slots_[s];
                if (slot.role != ContinuationSlotRole::Catalogued) { return; }
                for (const std::uint32_t g : slot.physical_groups) { refs_in_target[g] += 1; }
            };
            for (std::size_t i = 0; i < owners_.size(); ++i) {
                if (node.owner_evicted[i] != 1) { continue; }
                const std::size_t s = owners_[i].continuation_index;
                add_slot(s);
                if (s < program_->continuation_slots_.size()) {
                    const auto& slot = program_->continuation_slots_[s];
                    if (slot.paired_rewrite_slot.has_value() &&
                        *slot.paired_rewrite_slot < program_->continuation_slots_.size() &&
                        program_->continuation_slots_[*slot.paired_rewrite_slot].generation ==
                            slot.paired_rewrite_generation &&
                        !program_->is_slot_protected(*slot.paired_rewrite_slot)) {
                        add_slot(*slot.paired_rewrite_slot);
                    }
                }
            }
            for (const auto& [g, refs] : refs_in_target) {
                if (program_->executor_.group_refcount(g) == refs) { available += 1; }
            }
        }
        std::uint32_t needed = details.required_page_groups;
        if (details.has_source && details.reusable_tokens > 0) {
            const std::uint32_t source_groups = (details.reusable_tokens + 256U - 1U) / 256U;
            needed = (needed > source_groups) ? (needed - source_groups) : 0;
        }
        std::uint32_t checkpoint_slots = program_->vacant_checkpoint_slots();
        if (details.has_source && details.reusable_tokens > 0 &&
            details.assessment.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            checkpoint_slots += program_->owner_checkpoint_slots(details.source_continuation_index);
        }
        for (std::size_t i = 0; i < owners_.size(); ++i) {
            if (node.owner_evicted[i] == 1) {
                checkpoint_slots += program_->owner_checkpoint_slots(owners_[i].continuation_index);
            }
        }
        const bool checkpoint_available =
            checkpoint_slots >= details.base_plan->checkpoint_slots_required;
        if (available >= needed && checkpoint_available) {
            status = runtime::MaterializationPhysicalStatus::Feasible;
        } else {
            status = runtime::MaterializationPhysicalStatus::Infeasible;
        }
    }

    bool expandable = false;
    for (std::size_t i = 0; i < owners_.size(); ++i) {
        if (node.owner_evicted[i] == 0) {
            bool is_protected = false;
            if (details.has_source &&
                details.source_continuation_index == owners_[i].continuation_index &&
                details.source_continuation_generation == owners_[i].generation) {
                is_protected = true;
            }
            if (program_->is_slot_protected(owners_[i].continuation_index)) {
                is_protected = true;
            }
            if (!is_protected) {
                expandable = true;
                break;
            }
        }
    }

    std::uint64_t digest = details.assessment.assessment_digest;
    if (!is_identity) {
        digest = 1469598103934665603ULL;
        digest ^= static_cast<std::uint64_t>(node.candidate_index); digest *= 1099511628211ULL;
        digest ^= static_cast<std::uint64_t>(node.stable_ordinal); digest *= 1099511628211ULL;
        digest ^= static_cast<std::uint64_t>(status); digest *= 1099511628211ULL;
        digest ^= static_cast<std::uint64_t>(total_degradation); digest *= 1099511628211ULL;
        for (std::uint8_t ev : node.owner_evicted) {
            digest ^= static_cast<std::uint64_t>(ev); digest *= 1099511628211ULL;
        }
    }

    runtime::PressureTargetAssessment result{
        .physical_status = status,
        .source_mode = details.assessment.source_mode,
        .machine_work = details.assessment.machine_work,
        .owner_outcomes = assessment_outcomes_,
        .checkpoint_impacts = assessment_impacts_,
        .candidate = candidate_ids_[node.candidate_index],
        .stable_target_ordinal = node.stable_ordinal,
        .degradation_units = total_degradation,
        .dropped_checkpoints = total_dropped,
        .projection_work = projection_work,
        .assessment_digest = digest,
        .expandable = expandable,
        .root_maximal = node.root_maximal,
    };

    std::optional<AdmissionCandidate> executable;
    if (status == runtime::MaterializationPhysicalStatus::Feasible) {
        std::vector<std::uint32_t> evicted_slots;
        std::vector<std::uint32_t> evicted_ordinals;
        for (std::size_t i = 0; i < owners_.size(); ++i) {
            if (node.owner_evicted[i] == 1) {
                evicted_slots.push_back(owners_[i].continuation_index);
                evicted_ordinals.push_back(owners_[i].owner_id.value);
            }
        }
        auto copy = std::make_unique<AdmissionCandidateImpl>();
        copy->summary = cand.impl_->summary;
        copy->assessment = cand.impl_->assessment;
        copy->assessment.physical_status = status;
        if (cand.impl_->base_plan) {
            copy->base_plan = std::make_unique<RequestBasePlanImpl>(*cand.impl_->base_plan);
        }
        copy->reusable_tokens = cand.impl_->reusable_tokens;
        copy->source_continuation_index = cand.impl_->source_continuation_index;
        copy->source_continuation_generation = cand.impl_->source_continuation_generation;
        copy->source_checkpoint_index = cand.impl_->source_checkpoint_index;
        copy->source_checkpoint_generation = cand.impl_->source_checkpoint_generation;
        copy->has_source = cand.impl_->has_source;
        copy->required_page_groups = cand.impl_->required_page_groups;
        copy->planning_revision = cand.impl_->planning_revision;
        copy->pressure_evicted_slots = std::move(evicted_slots);
        copy->pressure_evicted_ordinals = std::move(evicted_ordinals);
        executable.emplace(std::move(copy));
    }

    return AssessedPressureTarget(this, session_generation_, target_idx, result, 0, 0, nullptr,
                                  std::move(executable), std::nullopt);
}

runtime::PressureTargetGuidance PressurePlanningSessionImpl::guidance(PressureTargetHandle target) {
    if (!valid(target) || scratch_live_) {
        throw std::logic_error("pressure target guidance is stale or conflicts with expansion");
    }
    const std::uint32_t target_idx = FlashNextPressureHandleHelper::get_index(target);
    const auto& node = targets_[target_idx];
    const auto& cand = *candidates_[node.candidate_index];
    const auto& details = *cand.impl_;

    guidance_outcomes_.clear();
    guidance_outcomes_.reserve(owners_.size());
    std::uint32_t total_degradation = 0;
    std::uint32_t total_dropped = 0;

    for (std::size_t i = 0; i < owners_.size(); ++i) {
        if (node.owner_evicted[i] == 1) {
            const std::uint32_t dropped = program_->published_checkpoints_of(owners_[i].continuation_index);
            guidance_outcomes_.push_back(runtime::PressureOwnerOutcome{
                .owner = owners_[i].owner_id,
                .disposition = runtime::VictimDisposition::Evicted,
                .degradation_units = 1,
                .dropped_checkpoints = dropped,
            });
            total_degradation += 1;
            total_dropped += dropped;
        } else {
            guidance_outcomes_.push_back(runtime::PressureOwnerOutcome{
                .owner = owners_[i].owner_id,
                .disposition = runtime::VictimDisposition::Retained,
                .degradation_units = 0,
                .dropped_checkpoints = 0,
            });
        }
    }

    std::uint32_t available = static_cast<std::uint32_t>(program_->executor_.available_physical_groups());
    {
        const std::uint32_t reserved = program_->reserved_unowned_groups();
        available = available > reserved ? available - reserved : 0U;
    }
    {
        std::unordered_map<std::uint32_t, std::uint32_t> refs_in_target;
        const auto add_slot = [&](std::size_t s) {
            if (s >= program_->continuation_slots_.size()) { return; }
            const auto& slot = program_->continuation_slots_[s];
            if (slot.role != ContinuationSlotRole::Catalogued) { return; }
            for (const std::uint32_t g : slot.physical_groups) { refs_in_target[g] += 1; }
        };
        for (std::size_t i = 0; i < owners_.size(); ++i) {
            if (node.owner_evicted[i] != 1) { continue; }
            const std::size_t s = owners_[i].continuation_index;
            add_slot(s);
            if (s < program_->continuation_slots_.size()) {
                const auto& slot = program_->continuation_slots_[s];
                if (slot.paired_rewrite_slot.has_value() &&
                    *slot.paired_rewrite_slot < program_->continuation_slots_.size() &&
                    program_->continuation_slots_[*slot.paired_rewrite_slot].generation ==
                        slot.paired_rewrite_generation &&
                    !program_->is_slot_protected(*slot.paired_rewrite_slot)) {
                    add_slot(*slot.paired_rewrite_slot);
                }
            }
        }
        for (const auto& [g, refs] : refs_in_target) {
            if (program_->executor_.group_refcount(g) == refs) { available += 1; }
        }
    }

    std::uint32_t needed = details.required_page_groups;
    if (details.has_source && details.reusable_tokens > 0) {
        const std::uint32_t source_groups = (details.reusable_tokens + 256U - 1U) / 256U;
        needed = (needed > source_groups) ? (needed - source_groups) : 0;
    }

    const std::uint32_t deficit = (needed > available) ? (needed - available) : 0U;
    const std::uint32_t total_capacity = static_cast<std::uint32_t>(program_->plan_.main_page_groups);
    constexpr std::uint64_t kResidualOne = 1ULL << 20U;
    const std::uint64_t normalized_q20 = (deficit > 0 && total_capacity > 0)
        ? std::min<std::uint64_t>(kResidualOne, (static_cast<std::uint64_t>(deficit) * kResidualOne) / total_capacity)
        : 0ULL;

    return runtime::PressureTargetGuidance{
        .physical = runtime::PressurePhysicalGuidance{
            .unsatisfied_constraints = deficit > 0 ? 1U : 0U,
            .estimated_remaining_steps = (deficit + 3U) / 4U,
            .normalized_residual_q20 = normalized_q20,
        },
        .estimated_machine_work = details.assessment.machine_work,
        .owner_outcomes = guidance_outcomes_,
        .candidate = candidate_ids_[node.candidate_index],
        .stable_target_ordinal = node.stable_ordinal,
        .degradation_units = total_degradation,
        .dropped_checkpoints = total_dropped,
    };
}

void PressurePlanningSessionImpl::retain_assessment(PressureTargetHandle target) {
    if (!valid(target) || scratch_live_) {
        throw std::logic_error("retained pressure assessment is stale or conflicts with expansion");
    }
    retained_target_idx_ = FlashNextPressureHandleHelper::get_index(target);
}

std::optional<PressureTargetHandle>
PressurePlanningSessionImpl::guided_closure_target(
    runtime::PlanningCandidateId candidate, std::span<const runtime::PlanningOwnerId> preferred_owner_ids) {
    if (scratch_live_) {
        throw std::logic_error("guided pressure closure conflicts with expansion scratch");
    }
    const std::uint32_t cand_idx = candidate_index(candidate);
    const auto& cand_impl = *candidates_[cand_idx]->impl_;

    PressurePlanningTargetNode target{
        .candidate_index = cand_idx,
        .owner_evicted = std::vector<std::uint8_t>(owners_.size(), 0),
        .stable_ordinal = 0,
        .root_maximal = false,
    };

    std::uint32_t needed = cand_impl.required_page_groups;
    if (cand_impl.has_source && cand_impl.reusable_tokens > 0) {
        const std::uint32_t source_groups = (cand_impl.reusable_tokens + 256U - 1U) / 256U;
        needed = (needed > source_groups) ? (needed - source_groups) : 0;
    }

    const auto check_feasible = [&](const PressurePlanningTargetNode& node) -> bool {
        std::uint32_t available = static_cast<std::uint32_t>(program_->executor_.available_physical_groups());
        const std::uint32_t reserved = program_->reserved_unowned_groups();
        available = available > reserved ? available - reserved : 0U;

        std::unordered_map<std::uint32_t, std::uint32_t> refs_in_target;
        const auto add_slot = [&](std::size_t s) {
            if (s >= program_->continuation_slots_.size()) { return; }
            const auto& slot = program_->continuation_slots_[s];
            if (slot.role != ContinuationSlotRole::Catalogued) { return; }
            for (const std::uint32_t g : slot.physical_groups) { refs_in_target[g] += 1; }
        };
        for (std::size_t i = 0; i < owners_.size(); ++i) {
            if (node.owner_evicted[i] != 1) { continue; }
            const std::size_t s = owners_[i].continuation_index;
            add_slot(s);
            if (s < program_->continuation_slots_.size()) {
                const auto& slot = program_->continuation_slots_[s];
                if (slot.paired_rewrite_slot.has_value() &&
                    *slot.paired_rewrite_slot < program_->continuation_slots_.size() &&
                    program_->continuation_slots_[*slot.paired_rewrite_slot].generation ==
                        slot.paired_rewrite_generation &&
                    !program_->is_slot_protected(*slot.paired_rewrite_slot)) {
                    add_slot(*slot.paired_rewrite_slot);
                }
            }
        }
        for (const auto& [g, refs] : refs_in_target) {
            if (program_->executor_.group_refcount(g) == refs) { available += 1; }
        }
        return available >= needed;
    };

    if (check_feasible(target)) {
        return identity_target(candidate);
    }

    std::vector<std::size_t> owner_order;
    owner_order.reserve(owners_.size());
    for (const runtime::PlanningOwnerId id : preferred_owner_ids) {
        for (std::size_t i = 0; i < owners_.size(); ++i) {
            if (owners_[i].owner_id == id) {
                if (std::find(owner_order.begin(), owner_order.end(), i) == owner_order.end()) {
                    owner_order.push_back(i);
                }
                break;
            }
        }
    }
    for (std::size_t i = 0; i < owners_.size(); ++i) {
        if (std::find(owner_order.begin(), owner_order.end(), i) == owner_order.end()) {
            owner_order.push_back(i);
        }
    }

    for (const std::size_t owner_idx : owner_order) {
        const auto& owner = owners_[owner_idx];
        bool is_protected = false;
        if (cand_impl.has_source &&
            cand_impl.source_continuation_index == owner.continuation_index &&
            cand_impl.source_continuation_generation == owner.generation) {
            is_protected = true;
        }
        if (program_->is_slot_protected(owner.continuation_index)) {
            is_protected = true;
        }
        if (is_protected) { continue; }

        target.owner_evicted[owner_idx] = 1;
        if (check_feasible(target)) {
            auto it = std::find_if(targets_.begin(), targets_.end(), [&](const auto& item) {
                return item.candidate_index == target.candidate_index &&
                       item.owner_evicted == target.owner_evicted;
            });
            std::uint32_t target_idx = 0;
            if (it != targets_.end()) {
                target_idx = static_cast<std::uint32_t>(it - targets_.begin());
            } else {
                target.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
                targets_.push_back(std::move(target));
                target_idx = static_cast<std::uint32_t>(targets_.size() - 1);
            }
            return FlashNextPressureHandleHelper::make_handle(this, session_generation_, target_idx);
        }
    }

    return std::nullopt;
}

std::uint32_t PressurePlanningSessionImpl::optional_targets_remaining() const noexcept {
    // Fusion answers "canonical target slots the arena can still hold"
    // (`planning/pressure_planner.cpp:809-816`) because its session owns a fixed 4096-slot arena
    // (`kOptionalTargetCapacity`). This session has no such arena: `prepare_expansion` enumerates
    // children by setting exactly one more bit in the `owner_evicted` mask (`program.cpp:1096-1124`)
    // and rejecting duplicates, so every child is a subset of `owners_` and no sequence can sacrifice
    // more owners than exist. `owners_.size()` is therefore this session's own ceiling on the
    // sacrifice ladder, and it is what `SharedCapturePlanner` needs when it bounds
    // `min(remaining, session.optional_targets_remaining())`.
    // Deliberately NOT copying fusion's 4096: that constant sizes a structure this model does not
    // have, and a copied constant would be a second home for a fact that belongs to this session.
    return static_cast<std::uint32_t>(owners_.size());
}

bool PressurePlanningSessionImpl::try_claim_seal_window() noexcept {
    // Forwarded to the Program so the claim is Program-wide, exactly as fusion does
    // (`models/qwen3_5/program/program.cpp:239-241`: `impl_->program->try_claim_seal_window()`).
    return program_ != nullptr && program_->try_claim_seal_window();
}

void PressurePlanningSessionImpl::release_seal_window() noexcept {
    if (program_ != nullptr) { program_->release_seal_window(); }
}

bool ProgramImpl::try_claim_seal_window() noexcept {
    // Copied from fusion (`models/qwen3_5/program/planning/pressure.cpp:2410-2413`): one exclusive
    // claim, acquired by compare-exchange.
    bool expected = false;
    return seal_window_claimed_.compare_exchange_strong(expected, true, std::memory_order_acquire);
}

void ProgramImpl::release_seal_window() noexcept {
    // Copied from fusion (`models/qwen3_5/program/planning/pressure.cpp:2415-2417`); idempotent.
    seal_window_claimed_.store(false, std::memory_order_release);
}

PreparedPressureExpansion PressurePlanningSessionImpl::prepare_expansion(PressureTargetHandle parent) {
    if (!valid(parent) || scratch_live_) {
        throw std::logic_error("pressure expansion parent is stale or scratch is busy");
    }
    const std::uint32_t parent_idx = FlashNextPressureHandleHelper::get_index(parent);
    const auto& node = targets_[parent_idx];
    const auto& details = *candidates_[node.candidate_index]->impl_;

    expansion_scratch_.clear();
    scratch_new_count_ = 0;
    scratch_parent_index_ = parent_idx;

    for (std::size_t i = 0; i < owners_.size(); ++i) {
        if (node.owner_evicted[i] == 0) {
            bool is_protected = false;
            if (details.has_source &&
                details.source_continuation_index == owners_[i].continuation_index &&
                details.source_continuation_generation == owners_[i].generation) {
                is_protected = true;
            }
            if (program_->is_slot_protected(owners_[i].continuation_index)) {
                is_protected = true;
            }
            if (!is_protected) {
                PressurePlanningTargetNode child = node;
                child.owner_evicted[i] = 1;
                child.root_maximal = false;

                const bool in_scratch = std::any_of(expansion_scratch_.begin(), expansion_scratch_.end(),
                    [&](const auto& item) {
                        return item.candidate_index == child.candidate_index &&
                               item.owner_evicted == child.owner_evicted;
                    });
                if (!in_scratch) {
                    const bool in_targets = std::any_of(targets_.begin(), targets_.end(),
                        [&](const auto& item) {
                            return item.candidate_index == child.candidate_index &&
                                   item.owner_evicted == child.owner_evicted;
                        });
                    if (!in_targets) {
                        scratch_new_count_++;
                    }
                    expansion_scratch_.push_back(std::move(child));
                }
            }
        }
    }

    if (++scratch_generation_ == 0) {
        ++scratch_generation_;
    }
    scratch_live_ = true;
    return PreparedPressureExpansion(this, session_generation_, scratch_generation_, parent_idx, scratch_new_count_);
}

PressureExpansionView PressurePlanningSessionImpl::commit_expansion(PreparedPressureExpansion&& prepared) {
    if (!scratch_live_ || prepared.new_canonical_count() != scratch_new_count_) {
        throw std::logic_error("prepared pressure expansion is stale");
    }

    committed_children_.clear();
    committed_children_.reserve(expansion_scratch_.size());

    for (auto& child : expansion_scratch_) {
        auto it = std::find_if(targets_.begin(), targets_.end(), [&](const auto& item) {
            return item.candidate_index == child.candidate_index &&
                   item.owner_evicted == child.owner_evicted;
        });
        std::uint32_t idx = 0;
        if (it != targets_.end()) {
            idx = static_cast<std::uint32_t>(it - targets_.begin());
        } else {
            child.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
            targets_.push_back(std::move(child));
            idx = static_cast<std::uint32_t>(targets_.size() - 1);
        }
        committed_children_.push_back(FlashNextPressureHandleHelper::make_handle(this, session_generation_, idx));
    }

    const std::uint32_t count = scratch_new_count_;
    expansion_scratch_.clear();
    scratch_new_count_ = 0;
    scratch_live_ = false;

    return PressureExpansionView{
        .children = committed_children_,
        .new_canonical_count = count,
    };
}

void PressurePlanningSessionImpl::discard_expansion(PreparedPressureExpansion&& /*prepared*/) noexcept {
    if (scratch_live_) {
        expansion_scratch_.clear();
        scratch_new_count_ = 0;
        scratch_live_ = false;
    }
}

std::optional<ResourcePlan> PressurePlanningSessionImpl::seal(
    AssessedPressureTarget&& assessed, const qwen3_5::PreparedPrompt& /*prompt*/,
    runtime::FinalScheduleIntent /*intent*/) {
    if (assessed.session_ != this || assessed.session_generation_ != session_generation_ || scratch_live_ ||
        assessed.target_index_ >= targets_.size() || !assessed.executable_ ||
        assessed.assessment_.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        throw std::logic_error("pressure assessment is not sealable as materialization");
    }
    std::optional<AdmissionCandidate> sealed = std::move(assessed.executable_);
    assessed.reset();
    return ResourcePlan(std::move(*sealed), runtime::ProgramResourceRevision{program_->resource_revision_}, false);
}

std::optional<CapturePressurePlan>
PressurePlanningSessionImpl::seal_capture(AssessedPressureTarget&& assessed) {
    (void)assessed;
    return std::nullopt;
}

} // namespace detail

PressurePlanningSession::PressurePlanningSession(
    std::unique_ptr<detail::PressurePlanningSessionImpl> impl) noexcept
    : impl_(std::move(impl)) {}
PressurePlanningSession::PressurePlanningSession(PressurePlanningSession&&) noexcept            = default;
PressurePlanningSession& PressurePlanningSession::operator=(PressurePlanningSession&&) noexcept = default;
PressurePlanningSession::~PressurePlanningSession()                                             = default;

PressureTargetHandle
PressurePlanningSession::identity_target(runtime::PlanningCandidateId candidate) const {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->identity_target(candidate);
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(runtime::PlanningCandidateId root_candidate) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->root_maximal_target(root_candidate);
}

AssessedPressureTarget
PressurePlanningSession::assess(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->assess(target);
}

runtime::PressureTargetGuidance
PressurePlanningSession::guidance(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->guidance(target);
}

void PressurePlanningSession::retain_assessment(PressureTargetHandle target) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    impl_->retain_assessment(target);
}

std::optional<PressureTargetHandle>
PressurePlanningSession::guided_closure_target(
    runtime::PlanningCandidateId candidate, std::span<const runtime::PlanningOwnerId> preferred_owner_ids) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->guided_closure_target(candidate, preferred_owner_ids);
}

PreparedPressureExpansion
PressurePlanningSession::prepare_expansion(PressureTargetHandle parent) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->prepare_expansion(parent);
}

std::uint32_t PressurePlanningSession::optional_targets_remaining() const noexcept {
    if (impl_ == nullptr) { return 0; }
    return impl_->optional_targets_remaining();
}

bool PressurePlanningSession::try_claim_seal_window() noexcept {
    return impl_ != nullptr && impl_->try_claim_seal_window();
}

void PressurePlanningSession::release_seal_window() noexcept {
    if (impl_ != nullptr) { impl_->release_seal_window(); }
}

PressureExpansionView
PressurePlanningSession::commit_expansion(PreparedPressureExpansion&& prepared) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->commit_expansion(std::move(prepared));
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&& prepared) noexcept {
    if (impl_ != nullptr) {
        impl_->discard_expansion(std::move(prepared));
    }
}

runtime::PrefillWork
PressurePlanningSession::shared_capture_split_prefill_work(
    const AssessedPressureTarget& /*assessed*/,
    const qwen3_5::PreparedPrompt& /*prompt*/,
    std::span<const std::uint32_t> /*frontiers*/) const {
    return runtime::PrefillWork{};
}

std::optional<ResourcePlan>
PressurePlanningSession::seal(AssessedPressureTarget&& assessed,
                              const qwen3_5::PreparedPrompt& prompt,
                              runtime::FinalScheduleIntent intent) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->seal(std::move(assessed), prompt, intent);
}

std::optional<CapturePressurePlan>
PressurePlanningSession::seal_capture(AssessedPressureTarget&& assessed) {
    if (impl_ == nullptr) { throw std::logic_error("PressurePlanningSession: instance is empty"); }
    return impl_->seal_capture(std::move(assessed));
}

// ---------------------------------------------------------------------------
// SequencePlan & SequencePlanner
// ---------------------------------------------------------------------------

SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}
SequencePlan::SequencePlan(SequencePlan&&) noexcept            = default;
SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;
SequencePlan::~SequencePlan()                                  = default;

std::uint32_t SequencePlan::capacity() const noexcept {
    if (impl_ == nullptr) { return 0; }
    // task-28: this accessor feeds EngineTarget::capacity, which is what engine.cpp:406/424/431
    // compare a prompt against. With KV streaming on, what a REQUEST may occupy is the context
    // budget (max_context) -- the device pool is only a residency window; with the switch off this
    // is resolved_tokens, byte for byte the previous value.
    return ninfer::models::qwen3_8_flash_next::detail::flash_next_kv_stream_requested()
               ? impl_->plan.config.max_context
               : impl_->plan.resolved_tokens;
}

std::uint32_t SequencePlan::kv_capacity() const noexcept {
    return impl_ != nullptr ? impl_->plan.resolved_tokens : 0;
}

std::uint32_t SequencePlan::max_concurrency() const noexcept {
    return impl_ != nullptr ? impl_->plan.config.max_concurrency : 0;
}

std::uint32_t SequencePlan::continuation_capacity() const noexcept {
    return impl_ != nullptr ? impl_->plan.continuation_slots : 0;
}

std::size_t SequencePlan::device_reservation_bytes() const noexcept {
    return impl_ != nullptr ? impl_->plan.total_device_bytes : 0;
}

std::size_t SequencePlan::workspace_capacity_bytes() const noexcept {
    return impl_ != nullptr ? impl_->plan.workspace_bytes : 0;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}
SequencePlanner::SequencePlanner(SequencePlanner&&) noexcept            = default;
SequencePlanner& SequencePlanner::operator=(SequencePlanner&&) noexcept = default;
SequencePlanner::~SequencePlanner()                                     = default;

runtime::SequenceCapacityCurve SequencePlanner::capacity_curve() const {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    return detail::flash_next_capacity_curve(impl_->config);
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (impl_ == nullptr) { throw std::logic_error("sequence planner is empty"); }
    auto plan = detail::finalize_flash_next_runtime_plan(impl_->config, main_page_groups);
    impl_.reset();
    return SequencePlan(std::make_unique<detail::SequencePlanImpl>(std::move(plan)));
}

// ---------------------------------------------------------------------------
// Program Implementation
// ---------------------------------------------------------------------------

Program::Program(std::unique_ptr<detail::ProgramImpl> impl) noexcept : impl_(std::move(impl)) {}
Program::~Program() noexcept = default;

RequestBasePlan
Program::plan_request(const qwen3_5::PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    const auto& prompt_data = qwen3_5::PreparedPromptAccess::view(prompt);
    if (prompt_data.token_ids.empty()) {
        throw std::invalid_argument("prompt must contain tokens");
    }
    if (prompt_data.has_media()) {
        if (!impl_->vision_session_.has_value() &&
            (impl_->model_data_ == nullptr || !impl_->model_data_->vision.has_value())) {
            throw std::invalid_argument("Vision requests require vision features enabled in model");
        }
    }
    if (prompt_data.token_ids.size() > impl_->plan_.admissible_tokens) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    for (const TokenId id : prompt_data.token_ids) {
        if (id < 0 || id >= 248'077) {
            throw std::invalid_argument("prompt contains token outside the 248077-token domain");
        }
    }
    if (!std::isfinite(options.sampling.temperature) || !std::isfinite(options.sampling.top_p) ||
        !std::isfinite(options.sampling.min_p) || !std::isfinite(options.sampling.presence_penalty) ||
        !std::isfinite(options.sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (options.sampling.top_p < 0.0F || options.sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (options.sampling.min_p < 0.0F || options.sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }

    auto base                   = std::make_unique<detail::RequestBasePlanImpl>();
    base->context_cache         = prompt_data.context_cache;
    base->prefill_chunk         = impl_->plan_.config.prefill_chunk;
    base->summary.prompt_tokens = static_cast<std::uint32_t>(prompt_data.token_ids.size());
    base->summary.reusable_prompt_tokens = 0;
    base->summary.requested_output_tokens = options.requested_output_tokens;

    const std::uint32_t capacity_output =
        impl_->plan_.admissible_tokens >= base->summary.prompt_tokens
            ? (impl_->plan_.admissible_tokens - base->summary.prompt_tokens + 1U)
            : 0U;
    base->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    base->summary.effective_limit_reason =
        options.requested_output_tokens <= capacity_output
            ? FinishReason::OutputLimit
            : FinishReason::ContextCapacity;
    base->summary.prefix_reuse_path    = PrefixReusePath::Root;
    base->summary.publish_continuation =
        options.allow_prefix_reuse && prompt_data.identity.reusable &&
        (impl_->plan_.config.continuation_capacity > 0);
    if (base->summary.publish_continuation) {
        base->checkpoint_slots_required = 1;
        if (impl_->continuation_slots_.size() > 1 && derive_turn_closure_frontier(prompt_data)) {
            ++base->checkpoint_slots_required;
        }
    }

    const runtime::PrefillWork prefill_work =
        runtime::make_prefill_work(0, base->summary.prompt_tokens, 0, 0,
                                   impl_->plan_.config.prefill_chunk);
    base->summary.service_work_quanta =
        prefill_work.tokens + base->summary.effective_output_tokens;

    base->requested_output_tokens = options.requested_output_tokens;
    base->effective_output_tokens = base->summary.effective_output_tokens;
    base->allow_prefix_reuse      = options.allow_prefix_reuse;
    base->thinking                = options.thinking;
    base->prefix_digests.assign(prompt_data);
    base->prefix_identity_tag     = 0;

    base->sampling_config.temperature       = options.sampling.temperature;
    base->sampling_config.top_k             = options.sampling.top_k;
    base->sampling_config.top_p             = options.sampling.top_p;
    base->sampling_config.min_p             = options.sampling.min_p;
    base->sampling_config.presence_penalty  = options.sampling.presence_penalty;
    base->sampling_config.frequency_penalty = options.sampling.frequency_penalty;
    base->sampling_config.seed              = options.sampling.seed;
    base->sampling_config.token_counts      = nullptr;
    base->output_constraint = options.output_constraint;

    if (prompt_data.has_media()) {
        auto control_plan         = qwen3_5::plan_vision_control(prompt_data);
        auto control              = qwen3_5::build_vision_control(prompt_data, control_plan, 0);
        base->vision_control_plan = std::move(control_plan);
        base->vision_control      = std::move(control);
    }

    return RequestBasePlan(std::move(base));
}

std::optional<AdmissionCandidate>
Program::inspect_admission(const qwen3_5::PreparedPrompt& prompt, const RequestBasePlan& base,
                          runtime::LaneId destination, const ContinuationHandle* source,
                          const SharedPrefixHandle* shared_source,
                          std::optional<runtime::CheckpointRef> checkpoint,
                          bool must_retain_private_source) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    const auto& prompt_data = qwen3_5::PreparedPromptAccess::view(prompt);
    const std::uint32_t prompt_tokens = base.summary().prompt_tokens;
    const std::uint32_t effective_out = base.summary().effective_output_tokens;

    // Admission trace, part 1 of 2 (docs/39 section 9.4): an UNCONDITIONAL entry line, so that "no
    // line at all" stops meaning two different things. The site label comes from the arguments rather
    // than from a call-site parameter: the root call passes no source, the private loop passes an
    // owner, the shared loop passes a shared handle. `feasible` cannot be printed here -- it is only
    // computed much further down -- which is why part 2 is a separate outcome line.
    const char* const admission_site =
        shared_source != nullptr ? "shared" : (source != nullptr ? "priv" : "root");
    const bool trace_admission = std::getenv("NINFER_FLASH_NEXT_TRACE_ADMISSION") != nullptr;
    if (trace_admission) {
        std::fprintf(stderr,
                     "[flash_next admission] enter site=%s dest=%u prompt_tokens=%u effective_out=%u "
                     "checkpoint=%d retain_source=%d\n",
                     admission_site, destination.value, prompt_tokens, effective_out,
                     checkpoint.has_value() ? 1 : 0, must_retain_private_source ? 1 : 0);
    }
    // The silent exits: before 2026-10-11 each of them returned `nullopt` without a word, so a
    // declined candidate and a candidate that was never inspected were indistinguishable.
    const auto decline = [&](const char* reason) {
        if (trace_admission) {
            std::fprintf(stderr, "[flash_next admission] decline site=%s reason=%s\n", admission_site,
                         reason);
        }
    };

    std::uint32_t reusable_tokens = 0;
    const detail::ContinuationSlot* cont_slot = nullptr;

    const bool has_source = source != nullptr || shared_source != nullptr;
    if ((source != nullptr && shared_source != nullptr) || has_source != checkpoint.has_value()) {
        throw std::logic_error("Flash-Next admission requires one owner and its selected checkpoint");
    }
    if (shared_source != nullptr) {
        decline("shared_source_unsupported");
        return std::nullopt;
    }

    std::uint32_t matched_slot_index = 0;
    if (source != nullptr) {
        if (source->owner() != this || source->index() >= impl_->continuation_slots_.size()) {
            throw std::logic_error("Flash-Next admission owner does not belong to this Program");
        }
        const auto& owner = impl_->continuation_slots_[source->index()];
        if (owner.role != detail::ContinuationSlotRole::Catalogued ||
            owner.generation != source->generation() ||
            owner.kind != runtime::CheckpointKind::SessionEndpoint) {
            throw std::logic_error("Flash-Next admission owner is stale");
        }
        matched_slot_index = source->index();
        if (checkpoint->kind == runtime::CheckpointKind::TurnClosure) {
            if (!owner.paired_rewrite_slot.has_value() ||
                *owner.paired_rewrite_slot >= impl_->continuation_slots_.size()) {
                throw std::logic_error("Flash-Next catalog rewrite has no checkpoint");
            }
            matched_slot_index = *owner.paired_rewrite_slot;
            if (impl_->continuation_slots_[matched_slot_index].generation != owner.paired_rewrite_generation) {
                throw std::logic_error("Flash-Next catalog rewrite generation is stale");
            }
        }
        const auto& selected = impl_->continuation_slots_[matched_slot_index];
        if (selected.role != detail::ContinuationSlotRole::Catalogued ||
            checkpoint->ordinal != 0 || checkpoint->kind != selected.kind ||
            checkpoint->frontier != static_cast<std::uint32_t>(selected.committed_frontier) ||
            checkpoint->frontier == 0 || checkpoint->frontier > selected.committed_tokens.size()) {
            throw std::logic_error("Flash-Next catalog checkpoint disagrees with Program state");
        }
        // The Engine has already enumerated candidates and recorded this exact owner/ref.
        // A token mismatch declines this candidate; it cannot select another owner's state.
        if (checkpoint->frontier > prompt_data.token_ids.size() ||
            !std::equal(selected.committed_tokens.begin(),
                        selected.committed_tokens.begin() + checkpoint->frontier,
                        prompt_data.token_ids.begin())) {
            decline("catalog_token_mismatch");
            return std::nullopt;
        }
        reusable_tokens = checkpoint->frontier;
        cont_slot = &selected;
    }

    const std::uint32_t total_tokens = prompt_tokens + (effective_out > 0 ? effective_out - 1U : 0U);
    if (total_tokens > impl_->plan_.admissible_tokens) {
        decline("tokens_over_resolved");
        return std::nullopt;
    }

    const std::uint32_t total_required_groups = (total_tokens + 255U) / 256U;
    // M4-2: with streaming on the device pool is a residency WINDOW; availability is then governed
    // by resolve (misses / overflow), so admission moves to the LOGICAL page ceiling. Switch off:
    // byte-for-byte the previous physical group check (total_required_groups is kept for callers).
    const bool over_capacity = ninfer::models::qwen3_8_flash_next::detail::flash_next_kv_stream_requested()
                                   ? ((total_tokens + ninfer::models::qwen3_8_flash_next::detail::kPageTokens - 1U) / ninfer::models::qwen3_8_flash_next::detail::kPageTokens >
                                      impl_->plan_.attention_logical_pages)
                                   : (total_required_groups > impl_->plan_.main_page_groups);
    if (over_capacity) {
        decline("groups_over_capacity");
        return std::nullopt;
    }

    const std::uint32_t reusable_groups =
        (reusable_tokens > 0) ? ((reusable_tokens + 255U) / 256U) : 0U;
    // task-28(b): under KV streaming the request's RESERVATION is capped at the physical pool. The
    // pool is a residency WINDOW, so "a 10,483-token prompt needs 41 groups" must not turn into
    // "reserve 41 of the 32 groups that exist". Switch off: identical to the previous expression.
    const std::uint32_t required_groups_for_admission =
        ninfer::models::qwen3_8_flash_next::detail::flash_next_kv_stream_requested()
            ? std::min(total_required_groups, impl_->plan_.main_page_groups)
            : total_required_groups;
    const std::uint32_t additional_groups_needed =
        (required_groups_for_admission > reusable_groups)
            ? (required_groups_for_admission - reusable_groups)
            : 0U;

    const bool lane_available =
        destination.value < impl_->lane_states_.size() && !impl_->lane_states_[destination.value].active;

    // Identity semantics: an admission candidate assumes no evictions, so only the free list
    // counts here. Catalogued groups come back through the pressure planner's root-maximal
    // target and expansions, which the Engine commits before start_resource_transaction.
    const std::uint32_t free_now = static_cast<std::uint32_t>(impl_->executor_.available_physical_groups());
    const std::uint32_t reserved = impl_->reserved_unowned_groups();
    const std::uint32_t total_freeable_groups = free_now > reserved ? free_now - reserved : 0U;
    const bool groups_available = (total_freeable_groups >= additional_groups_needed);

    std::uint32_t checkpoint_slots = impl_->vacant_checkpoint_slots();
    if (cont_slot != nullptr && reusable_tokens > 0 && !must_retain_private_source) {
        checkpoint_slots += impl_->owner_checkpoint_slots(source->index());
    }
    const bool checkpoint_available = checkpoint_slots >= base.impl_->checkpoint_slots_required;
    const bool feasible = lane_available && groups_available && checkpoint_available &&
                          !impl_->has_context_transaction_;

    // Admission trace, part 2 of 2 (docs/39 section 9.4): the outcome line replaces the old
    // infeasible-only print, so a run now always says which of the four conjuncts failed -- including
    // the checkpoint one, which the old line could not carry. `lane_active` keeps its original,
    // inverted meaning (`lane_available ? 0 : 1`) so that reads from the older logs stay comparable.
    if (trace_admission) {
        std::fprintf(stderr,
                     "[flash_next admission] outcome site=%s feasible=%d dest=%u lane_active=%d "
                     "active_lanes=%u freeable_groups=%u needed=%u total_tokens=%u resolved_tokens=%u "
                     "checkpoint_slots=%u checkpoint_required=%u has_txn=%d\n",
                     admission_site, feasible ? 1 : 0, destination.value, lane_available ? 0 : 1,
                     static_cast<unsigned>(impl_->executor_.active_lanes_count()), total_freeable_groups,
                     additional_groups_needed, total_tokens, impl_->plan_.resolved_tokens,
                     checkpoint_slots, base.impl_->checkpoint_slots_required,
                     impl_->has_context_transaction_ ? 1 : 0);
    }

    const std::uint32_t remaining_tokens =
        prompt_tokens > reusable_tokens ? prompt_tokens - reusable_tokens : 0U;
    const runtime::PrefillWork prefill_work =
        runtime::make_prefill_work(reusable_tokens, remaining_tokens, 0, 0, impl_->plan_.config.prefill_chunk);

    auto cand_impl                            = std::make_unique<detail::AdmissionCandidateImpl>();
    cand_impl->summary                        = base.summary();
    cand_impl->summary.reusable_prompt_tokens = reusable_tokens;
    cand_impl->summary.prefix_reuse_path =
        reusable_tokens > 0
            ? (cont_slot && cont_slot->kind == runtime::CheckpointKind::TurnClosure
                   ? PrefixReusePath::PrivateTurnClosure
                   : PrefixReusePath::PrivateEndpoint)
            : PrefixReusePath::Root;
    cand_impl->assessment.physical_status = feasible ? runtime::MaterializationPhysicalStatus::Feasible
                                                   : runtime::MaterializationPhysicalStatus::Infeasible;
    cand_impl->assessment.source_mode =
        (source != nullptr && must_retain_private_source)
            ? runtime::PrivateSourceMode::Retain
            : runtime::PrivateSourceMode::ConsumeToActive;
    cand_impl->assessment.expandable                          = true;
    cand_impl->assessment.projection_work                     = 1;
    cand_impl->assessment.machine_work.remaining_prefill_work = prefill_work;
    cand_impl->assessment.machine_work.reused_prompt_tokens   = reusable_tokens;
    cand_impl->base_plan                        = std::make_unique<detail::RequestBasePlanImpl>(*base.impl_);
    cand_impl->reusable_tokens                  = reusable_tokens;
    cand_impl->has_source                       = (cont_slot != nullptr);
    cand_impl->required_page_groups             = required_groups_for_admission;
    cand_impl->planning_revision                = impl_->resource_revision_;
    if (cont_slot != nullptr) {
        cand_impl->source_continuation_index      = source->index();
        cand_impl->source_continuation_generation = source->generation();
        cand_impl->source_checkpoint_index        = matched_slot_index;
        cand_impl->source_checkpoint_generation   = cont_slot->generation;
    }

    std::uint64_t digest = 1469598103934665603ULL;
    digest ^= static_cast<std::uint64_t>(cand_impl->assessment.physical_status); digest *= 1099511628211ULL;
    digest ^= static_cast<std::uint64_t>(reusable_tokens); digest *= 1099511628211ULL;
    digest ^= static_cast<std::uint64_t>(prompt_tokens); digest *= 1099511628211ULL;
    cand_impl->assessment.assessment_digest = digest;

    return AdmissionCandidate(std::move(cand_impl));
}

std::optional<ResourcePlan>
Program::seal_identity(const AdmissionCandidate& candidate,
                       const qwen3_5::PreparedPrompt& /*prompt*/,
                       runtime::FinalScheduleIntent /*intent*/) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }
    if (candidate.impl_ == nullptr || candidate.impl_->planning_revision != impl_->resource_revision_ ||
        candidate.impl_->assessment.physical_status != runtime::MaterializationPhysicalStatus::Feasible) {
        return std::nullopt;
    }
    auto copy = std::make_unique<detail::AdmissionCandidateImpl>();
    copy->summary = candidate.impl_->summary;
    copy->assessment = candidate.impl_->assessment;
    if (candidate.impl_->base_plan) {
        copy->base_plan = std::make_unique<detail::RequestBasePlanImpl>(*candidate.impl_->base_plan);
    }
    copy->reusable_tokens = candidate.impl_->reusable_tokens;
    copy->source_continuation_index = candidate.impl_->source_continuation_index;
    copy->source_continuation_generation = candidate.impl_->source_continuation_generation;
    copy->source_checkpoint_index = candidate.impl_->source_checkpoint_index;
    copy->source_checkpoint_generation = candidate.impl_->source_checkpoint_generation;
    copy->has_source = candidate.impl_->has_source;
    copy->required_page_groups = candidate.impl_->required_page_groups;
    copy->planning_revision = candidate.impl_->planning_revision;

    AdmissionCandidate sealed_cand(std::move(copy));
    return ResourcePlan(std::move(sealed_cand), runtime::ProgramResourceRevision{impl_->resource_revision_}, false);
}

PressurePlanningSession
Program::begin_pressure_planning(std::span<const AdmissionCandidate* const> candidates,
                                 std::span<const runtime::PlanningCandidateId> candidate_ids,
                                 std::span<const ContinuationHandle* const> private_owners,
                                 std::span<const runtime::PlanningOwnerId> private_owner_ids,
                                 std::span<const SharedPrefixHandle* const> shared_owners,
                                 std::span<const runtime::PlanningOwnerId> shared_owner_ids,
                                 // Accepted so the engine's call matches; unread. Fusion threads this
                                 // into its session to rank owners by recency, which is the mechanism
                                 // this model does not implement -- it selects through
                                 // `guided_closure_target` / `guidance` / `assess`. The parameter is
                                 // not silently ignored in the harmful sense: the range of target
                                 // sequences this session can produce is defined by `owner_evicted`
                                 // enumeration (see `optional_targets_remaining` below), which is a
                                 // self-contained ladder that does not need the caller's ranking.
                                 std::span<const runtime::PlanningOwnerId> /*recency_order*/) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }
    return PressurePlanningSession(std::make_unique<detail::PressurePlanningSessionImpl>(
        *impl_, candidates, candidate_ids, private_owners, private_owner_ids,
        shared_owners, shared_owner_ids));
}

runtime::PrefillWork
Program::shared_capture_split_prefill_work(const AdmissionCandidate& /*candidate*/,
                                           const qwen3_5::PreparedPrompt& /*prompt*/,
                                           std::span<const std::uint32_t> /*frontiers*/) {
    return runtime::PrefillWork{};
}

static std::optional<std::uint32_t>
derive_turn_closure_frontier(const qwen3_5::PreparedPromptData& prompt_data) {
    if (prompt_data.identity.rewrite_checkpoint &&
        prompt_data.identity.rewrite_checkpoint->frontier > 0 &&
        prompt_data.identity.rewrite_checkpoint->frontier < prompt_data.token_ids.size()) {
        return prompt_data.identity.rewrite_checkpoint->frontier;
    }
    std::optional<std::uint32_t> last_opp;
    for (const auto& opp : prompt_data.context_cache.opportunities) {
        if (opp.frontier > 0 && opp.frontier < prompt_data.token_ids.size()) {
            if (!last_opp || opp.frontier > *last_opp) {
                last_opp = opp.frontier;
            }
        }
    }
    return last_opp;
}

runtime::ContextTransactionReserveStatus
Program::start_resource_transaction(ResourcePlan&& plan, qwen3_5::PreparedPrompt&& prompt,
                                    runtime::CancellationFlagView cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }
    if (plan.resource_revision().value == 0 || plan.resource_revision().value != impl_->resource_revision_) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (cancellation.requested()) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (impl_->has_context_transaction_) {
        throw std::logic_error("Program: a context transaction is already in progress");
    }

    const runtime::RequestPlanSummary summary = plan.summary();
    auto adm = ContractAccess::take_admission(plan);

    // 1. Execute planned pressure evictions
    impl_->transaction_victims_.clear();
    if (adm.impl_ != nullptr) {
        for (std::size_t i = 0; i < adm.impl_->pressure_evicted_slots.size(); ++i) {
            const std::uint32_t slot_idx = adm.impl_->pressure_evicted_slots[i];
            const std::uint32_t owner_ord = adm.impl_->pressure_evicted_ordinals[i];
            impl_->vacate_owner(slot_idx);
            qwen3_5::MaterializationVictimResult victim;
            victim.owner              = runtime::PlanningOwnerId{owner_ord};
            victim.disposition        = runtime::VictimDisposition::Evicted;
            victim.pressure_committed = true;
            victim.final_summary      = std::nullopt;
            impl_->transaction_victims_.push_back(victim);
        }
    }

    // 2. Allocate lane
    detail::LaneHandle handle;
    try {
        handle = impl_->executor_.allocate_lane();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[flash_next start] allocate_lane failed: %s\n", e.what());
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    const std::uint32_t lane_idx = handle.lane_index();

    auto& st                     = impl_->lane_states_[lane_idx];
    st.active                    = true;
    st.epoch                     = handle.epoch();
    st.lane_handle               = handle;
    st.planned_groups            = (summary.prompt_tokens +
                                    (summary.effective_output_tokens > 0 ? summary.effective_output_tokens - 1U : 0U) +
                                    255U) / 256U;

    auto prompt_data        = qwen3_5::PreparedPromptAccess::take(std::move(prompt));
    st.capture_frontier     = derive_turn_closure_frontier(prompt_data);
    st.capture_offered      = false;
    st.mrope_pos0.assign(prompt_data.position_axis(0).begin(), prompt_data.position_axis(0).end());
    st.mrope_pos1.assign(prompt_data.position_axis(1).begin(), prompt_data.position_axis(1).end());
    st.mrope_pos2.assign(prompt_data.position_axis(2).begin(), prompt_data.position_axis(2).end());
    st.prefix_digests.assign(prompt_data);
    st.prompt_tokens        = std::move(prompt_data.token_ids);
    st.media_payloads       = std::move(prompt_data.media_payloads);
    st.vision_items         = std::move(prompt_data.vision_items);
    // Per-request cache. A leftover 0 skips encode on the next image and reuses the prior handoff.
    st.encoded_item_index.reset();
    st.vision_control.reset();
    st.turn_closure_continuation_index = std::nullopt;
    st.reused_from_continuation_index.reset();
    st.reused_from_continuation_generation.reset();
    st.pending_capture_offer = 0;
    st.reused_from_turn_closure = (adm.impl_ != nullptr && adm.impl_->summary.prefix_reuse_path == PrefixReusePath::PrivateTurnClosure);
    st.prompt_tokens_processed = 0;
    st.reused_prompt_tokens    = 0;
    st.committed_frontier      = 0;
    st.last_token_id           = 0;
    st.last_token_pos          = 0;
    st.last_token_index        = 0;
    st.total_generated_tokens  = 0;
    st.output_constraint.reset();
    st.requested_output_tokens = summary.requested_output_tokens;
    st.effective_output_tokens = summary.effective_output_tokens;
    st.publish_continuation    = summary.publish_continuation;
    const bool is_mtp = impl_->has_mtp() && impl_->plan_.config.speculative_draft_tokens > 0;
    st.speculative_stats = SpeculativeStats{
        .backend               = is_mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
        .enabled               = is_mtp,
        .draft_window          = is_mtp ? impl_->plan_.config.speculative_draft_tokens : 0U,
        .rounds                = 0,
        .drafted_tokens        = 0,
        .accepted_tokens       = 0,
        .fallback_steps        = 0,
        .accepted_per_position = std::vector<std::uint64_t>(is_mtp ? impl_->plan_.config.speculative_draft_tokens : 0U, 0),
    };

    if (adm.impl_ != nullptr && adm.impl_->base_plan != nullptr) {
        st.sampling_config = adm.impl_->base_plan->sampling_config;
        if (adm.impl_->base_plan->output_constraint) {
            st.output_constraint = std::make_unique<runtime::OutputConstraintState>(
                adm.impl_->base_plan->output_constraint, prompt_data.starts_in_reasoning);
        }
        st.vision_control  = std::move(adm.impl_->base_plan->vision_control);
    }

    if (adm.impl_ != nullptr && adm.impl_->reusable_tokens > 0) {
        const auto owner_idx = adm.impl_->source_continuation_index;
        const auto checkpoint_idx = adm.impl_->source_checkpoint_index;
        auto& owner = impl_->continuation_slots_.at(owner_idx);
        const auto& selected = impl_->continuation_slots_.at(checkpoint_idx);
        if (owner.role != detail::ContinuationSlotRole::Catalogued ||
            owner.generation != adm.impl_->source_continuation_generation ||
            selected.role != detail::ContinuationSlotRole::Catalogued ||
            selected.generation != adm.impl_->source_checkpoint_generation) {
            throw std::logic_error("Flash-Next sealed source checkpoint is stale");
        }
        impl_->executor_.attach_physical_groups(
            handle, selected.physical_groups, selected.committed_frontier, selected.history);
        const auto active_slot = impl_->executor_.allocation().current_source_slot(lane_idx);
        impl_->executor_.copy_state_slot(selected.cache_slot, static_cast<std::uint32_t>(active_slot));

        st.prompt_tokens_processed = adm.impl_->reusable_tokens;
        st.reused_prompt_tokens = adm.impl_->reusable_tokens;
        st.last_token_pos = static_cast<std::int32_t>(adm.impl_->reusable_tokens) - 1;
        st.last_token_index = st.last_token_pos;
        st.committed_frontier = static_cast<std::int32_t>(adm.impl_->reusable_tokens);

        if (adm.impl_->assessment.source_mode == runtime::PrivateSourceMode::ConsumeToActive) {
            // The active lane now owns the selected pages and copied recurrent state.
            // Consume the complete Engine owner, including an unselected endpoint/rewrite.
            impl_->vacate_owner(owner_idx);
        } else {
            owner.last_used_epoch = ++impl_->continuation_epoch_;
            st.reused_from_continuation_index = owner_idx;
            st.reused_from_continuation_generation = owner.generation;
        }
    }

    if (st.publish_continuation && !impl_->continuation_slots_.empty()) {
        const auto endpoint = impl_->allocate_vacant_continuation_slot();
        if (endpoint < 0) {
            throw std::logic_error("Flash-Next sealed admission has no endpoint checkpoint slot");
        }
        auto& slot = impl_->continuation_slots_[endpoint];
        slot.role = detail::ContinuationSlotRole::ReservedEndpoint;
        st.reserved_endpoint_index = static_cast<std::uint32_t>(endpoint);
        if (adm.impl_->base_plan->checkpoint_slots_required > 1) {
            const auto rewrite = impl_->allocate_vacant_continuation_slot();
            if (rewrite < 0) {
                throw std::logic_error("Flash-Next sealed admission has no turn-closure checkpoint slot");
            }
            impl_->continuation_slots_[rewrite].role = detail::ContinuationSlotRole::ReservedTurnClosure;
            st.reserved_turn_closure_index = static_cast<std::uint32_t>(rewrite);
        }
    }

    st.prefill_completed = false;
    st.finished          = false;

    impl_->has_context_transaction_ = true;
    impl_->transaction_lane_        = runtime::LaneId(lane_idx);
    impl_->transaction_epoch_       = handle.epoch();
    impl_->transaction_has_source_  = (adm.impl_ != nullptr && adm.impl_->reusable_tokens > 0);
    impl_->transaction_source_mode_ =
        adm.impl_ != nullptr ? adm.impl_->assessment.source_mode : runtime::PrivateSourceMode::ConsumeToActive;
    ++impl_->resource_revision_;

    return runtime::ContextTransactionReserveStatus::Reserved;
}

std::optional<PersistentBackfillProof>
Program::prove_persistent_backfill(const RequestBasePlan& /*blocked_head*/,
                                  const ResourcePlan& /*candidate*/,
                                  std::span<const SequenceHandle> /*persistent_borrowers*/) const {
    return std::nullopt;
}

ContextTransactionProgress
Program::progress_context_transaction(runtime::CancellationFlagView cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }
    if (!impl_->has_context_transaction_) {
        throw std::logic_error("Program: no active context transaction to progress");
    }

    if (impl_->is_capture_transaction_) {
        ActiveCaptureResult res = std::move(impl_->pending_capture_result_);
        return res;
    }

    if (cancellation.requested()) {
        if (impl_->transaction_lane_) {
            const std::uint32_t lane_idx = impl_->transaction_lane_->value;
            if (impl_->lane_states_[lane_idx].active) {
                impl_->release_checkpoint_reservations(impl_->lane_states_[lane_idx]);
                impl_->executor_.release_lane(impl_->lane_states_[lane_idx].lane_handle);
                impl_->lane_states_[lane_idx].active = false;
                ++impl_->resource_revision_;
            }
        }
        impl_->has_context_transaction_ = false;
        impl_->transaction_lane_.reset();
        impl_->transaction_epoch_.reset();
        impl_->transaction_has_source_ = false;
        MaterializationResult res;
        res.status = runtime::ContextTransactionStatus::Aborted;
        res.victims = std::move(impl_->transaction_victims_);
        return res;
    }

    MaterializationResult res;
    res.status = runtime::ContextTransactionStatus::Published;
    res.published = StartResult{
        .sequence = SequenceHandle(this, impl_->transaction_lane_.value(),
                                   impl_->transaction_epoch_.value()),
    };
    if (impl_->transaction_has_source_) {
        res.source.emplace(qwen3_5::MaterializationSourceResult{
            .mode = impl_->transaction_source_mode_,
            .final_summary = std::nullopt,
        });
    }
    res.victims = std::move(impl_->transaction_victims_);
    return res;
}

void Program::finalize_context_transaction() noexcept {
    if (impl_ != nullptr) {
        impl_->has_context_transaction_ = false;
        impl_->is_capture_transaction_  = false;
        impl_->transaction_lane_.reset();
        impl_->transaction_epoch_.reset();
        impl_->transaction_has_source_ = false;
        impl_->transaction_victims_.clear();
    }
}

bool Program::has_context_transaction() const noexcept {
    return impl_ != nullptr && impl_->has_context_transaction_;
}

PrefillProgress
Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    if (sequence.owner() != this) {
        throw std::logic_error("SequenceHandle does not belong to this Program");
    }
    const std::uint32_t lane_idx = sequence.lane().value;
    if (lane_idx >= impl_->plan_.config.max_concurrency) {
        throw std::out_of_range("sequence lane out of range");
    }
    auto& st = impl_->lane_states_[lane_idx];
    if (!st.active || st.epoch != sequence.epoch()) {
        throw std::logic_error("sequence lane is not active or epoch mismatch");
    }
    if (st.prefill_completed) {
        throw std::logic_error("prefill is already completed for this sequence");
    }

    const std::size_t N         = st.prompt_tokens.size();
    const std::uint32_t start_i = st.prompt_tokens_processed;

    if (start_i == N) {
        if (!impl_->pending_round_.valid()) {
            throw std::logic_error("advance_prefill called on completed prompt without pending round");
        }
        // Sample the first output token from logits (step 2: completion)
        impl_->sample_tokens(impl_->pending_round_.logits(), std::span(&lane_idx, 1),
                             std::span(impl_->host_sampled_tokens_.data(), 1));

        impl_->pending_batch_tokens_.resize(1);
        impl_->pending_batch_row_counts_.resize(1);
        impl_->pending_batch_tokens_[0]     = static_cast<TokenId>(impl_->host_sampled_tokens_[0]);
        impl_->pending_batch_row_counts_[0] = 1;

        st.prefill_completed  = true;
        st.committed_frontier = static_cast<std::int32_t>(N);
        st.media_payloads.clear();
        if (impl_->vision_session_.has_value()) {
            impl_->vision_session_->retire_handoff();
        }

        PrefillProgress progress;
        progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
        progress.summary.reused_prompt_tokens = st.reused_prompt_tokens;
        progress.summary.prefix_reuse_path =
            st.reused_prompt_tokens > 0
                ? (st.reused_from_turn_closure
                       ? PrefixReusePath::PrivateTurnClosure
                       : PrefixReusePath::PrivateEndpoint)
                : PrefixReusePath::Root;
        progress.processed_prompt_tokens      = 0;
        progress.complete                     = true;
        // Attribute the round's blocking device stall to the wait bucket rather than
        // leaving it in the submit residual. See ExecutionTimingRecorder.
        timing.reclassify_submit_as_wait(impl_->executor_.take_round_device_wait_ns());
        progress.timing                       = timing.finish();
        progress.pending                      = ContractAccess::make_pending(
            this, 1, std::span(&sequence, 1),
            std::span(impl_->pending_batch_tokens_.data(), 1),
            std::span(impl_->pending_batch_row_counts_.data(), 1), 1, progress.timing);
        return progress;
    }

    std::uint32_t end_i =
        std::min<std::uint32_t>(start_i + impl_->plan_.config.prefill_chunk,
                                static_cast<std::uint32_t>(N));

    // Clip at capture frontier F if not yet offered
    const bool can_offer_capture =
        st.publish_continuation && (impl_->plan_.config.continuation_capacity > 0) &&
        st.capture_frontier.has_value() && st.reserved_turn_closure_index.has_value() && !st.capture_offered;

    if (can_offer_capture) {
        const std::uint32_t F = *st.capture_frontier;
        if (start_i < F && end_i > F) {
            end_i = F;
        }
    }

    // Chunk clipping for vision items: a chunk never crosses into a new vision item
    if (st.vision_control.has_value() && !st.vision_control->items.empty()) {
        for (std::size_t k = 0; k < st.vision_control->items.size(); ++k) {
            const auto& item = st.vision_control->items[k];
            if (item.scatter_indices.empty()) { continue; }
            const std::uint32_t item_begin = static_cast<std::uint32_t>(item.scatter_indices.front());
            const std::uint32_t item_end   = static_cast<std::uint32_t>(item.scatter_indices.back() + 1);

            if (item_end <= start_i) {
                continue;
            }
            if (start_i < item_begin && item_begin < end_i) {
                end_i = item_begin;
                break;
            }
            if (item_begin <= start_i && start_i < item_end) {
                end_i = std::min(end_i, item_end);
                break;
            }
        }
    }

    const std::uint32_t chunk_size = end_i - start_i;
    std::span<const std::int32_t> chunk_token_ids(st.prompt_tokens.data() + start_i, chunk_size);
    std::vector<std::array<std::int32_t, 3>> chunk_positions(chunk_size);
    for (std::uint32_t i = 0; i < chunk_size; ++i) {
        chunk_positions[i] = {st.mrope_pos0[start_i + i], st.mrope_pos1[start_i + i],
                              st.mrope_pos2[start_i + i]};
    }

    const Tensor* visual_embeddings = nullptr;
    std::vector<std::int32_t> local_scatter_indices;
    Tensor chunk_visual_tensor;

    if (st.vision_control.has_value() && !st.vision_control->items.empty()) {
        for (std::size_t k = 0; k < st.vision_control->items.size(); ++k) {
            const auto& item = st.vision_control->items[k];
            if (item.scatter_indices.empty()) { continue; }
            const auto& scatter = item.scatter_indices;
            auto begin_it = std::lower_bound(scatter.begin(), scatter.end(), static_cast<std::int32_t>(start_i));
            auto end_it   = std::lower_bound(begin_it, scatter.end(), static_cast<std::int32_t>(end_i));

            if (begin_it < end_it) {
                if (!st.encoded_item_index.has_value() || *st.encoded_item_index != k) {
                    if (!impl_->vision_session_.has_value()) {
                        throw std::logic_error("Vision item in prefill but vision session not initialized");
                    }
                    if (k >= st.media_payloads.size() || !st.media_payloads[k]) {
                        throw std::logic_error("Missing media payload for vision item");
                    }
                    impl_->vision_session_->encode(item, st.media_payloads[k]->span(), impl_->device_.stream);
                    st.encoded_item_index = k;
                }
                const std::size_t visual_begin = begin_it - scatter.begin();
                const std::size_t visual_count = end_it - begin_it;
                local_scatter_indices.resize(visual_count);
                for (std::size_t i = 0; i < visual_count; ++i) {
                    local_scatter_indices[i] = begin_it[i] - static_cast<std::int32_t>(start_i);
                }
                chunk_visual_tensor = impl_->vision_session_->output_tensor().slice(
                    1, static_cast<std::int32_t>(visual_begin), static_cast<std::int32_t>(visual_count));
                visual_embeddings = &chunk_visual_tensor;
                break;
            }
        }
    }

    const bool is_capture_split = can_offer_capture && (end_i == *st.capture_frontier);

    const std::int32_t last_token_in_chunk = static_cast<std::int32_t>(end_i - 1);
    const std::size_t req_groups =
        static_cast<std::size_t>(last_token_in_chunk / static_cast<std::int32_t>(detail::kMainPageGroupTokens)) + 1U;
    const std::size_t owned = impl_->executor_.lane_physical_groups(st.lane_handle).size();
    if (req_groups > owned) {
        const std::size_t needed = req_groups - owned;
        impl_->ensure_physical_groups_available(needed);
    }

    if (end_i < N) {
        auto round = impl_->executor_.execute_prefill_chunk(
            st.lane_handle, chunk_token_ids, chunk_positions, static_cast<std::int32_t>(start_i),
            nullptr, visual_embeddings, local_scatter_indices);
        std::array<detail::LaneCommitDecision, 1> decision = {{{.accept = true}}};
        round.commit(decision);
        st.last_token_id    = chunk_token_ids.back();
        st.last_token_pos   = chunk_positions.back()[0];
        st.last_token_index = static_cast<std::int32_t>(end_i - 1);
        st.prompt_tokens_processed = end_i;

        PrefillProgress progress;
        progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
        progress.summary.reused_prompt_tokens = st.reused_prompt_tokens;
        progress.summary.prefix_reuse_path =
            st.reused_prompt_tokens > 0
                ? (st.reused_from_turn_closure
                       ? PrefixReusePath::PrivateTurnClosure
                       : PrefixReusePath::PrivateEndpoint)
                : PrefixReusePath::Root;
        progress.processed_prompt_tokens      = chunk_size;
        progress.complete                     = false;
        // Attribute the round's blocking device stall to the wait bucket rather than
        // leaving it in the submit residual. See ExecutionTimingRecorder.
        timing.reclassify_submit_as_wait(impl_->executor_.take_round_device_wait_ns());
        progress.timing                       = timing.finish();

        if (is_capture_split) {
            st.capture_offered             = true;
            const std::uint64_t capture_id = ++impl_->capture_counter_;
            st.pending_capture_offer       = capture_id;
            progress.capture               = ContractAccess::make_capture_offer(
                this, sequence.lane(), sequence.epoch(), capture_id);
        }
        return progress;
    }

    // Final prefill chunk ending at N
    if (is_capture_split) {
        // F == N: offer capture at N before sampling
        impl_->pending_round_ = impl_->executor_.execute_prefill_chunk(
            st.lane_handle, chunk_token_ids, chunk_positions, static_cast<std::int32_t>(start_i),
            nullptr, visual_embeddings, local_scatter_indices);
        st.last_token_id    = chunk_token_ids.back();
        st.last_token_pos   = chunk_positions.back()[0];
        st.last_token_index = static_cast<std::int32_t>(end_i - 1);
        st.prompt_tokens_processed = end_i;
        st.capture_offered         = true;

        const std::uint64_t capture_id = ++impl_->capture_counter_;
        st.pending_capture_offer       = capture_id;

        PrefillProgress progress;
        progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
        progress.summary.reused_prompt_tokens = st.reused_prompt_tokens;
        progress.summary.prefix_reuse_path =
            st.reused_prompt_tokens > 0
                ? (st.reused_from_turn_closure
                       ? PrefixReusePath::PrivateTurnClosure
                       : PrefixReusePath::PrivateEndpoint)
                : PrefixReusePath::Root;
        progress.processed_prompt_tokens      = chunk_size;
        progress.complete                     = false;
        // Attribute the round's blocking device stall to the wait bucket rather than
        // leaving it in the submit residual. See ExecutionTimingRecorder.
        timing.reclassify_submit_as_wait(impl_->executor_.take_round_device_wait_ns());
        progress.timing                       = timing.finish();
        progress.capture                      = ContractAccess::make_capture_offer(
            this, sequence.lane(), sequence.epoch(), capture_id);
        return progress;
    }

    // Normal completion at N without capture offer at N
    impl_->pending_round_ = impl_->executor_.execute_prefill_chunk(
        st.lane_handle, chunk_token_ids, chunk_positions, static_cast<std::int32_t>(start_i),
        nullptr, visual_embeddings, local_scatter_indices);
    st.last_token_id    = chunk_token_ids.back();
    st.last_token_pos   = chunk_positions.back()[0];
    st.last_token_index = static_cast<std::int32_t>(end_i - 1);
    st.prompt_tokens_processed = end_i;

    impl_->sample_tokens(impl_->pending_round_.logits(), std::span(&lane_idx, 1),
                         std::span(impl_->host_sampled_tokens_.data(), 1));

    impl_->pending_batch_tokens_.resize(1);
    impl_->pending_batch_row_counts_.resize(1);
    impl_->pending_batch_tokens_[0]     = static_cast<TokenId>(impl_->host_sampled_tokens_[0]);
    impl_->pending_batch_row_counts_[0] = 1;

    st.prefill_completed  = true;
    st.committed_frontier = static_cast<std::int32_t>(N);
    st.media_payloads.clear();
    if (impl_->vision_session_.has_value()) {
        impl_->vision_session_->retire_handoff();
    }

    PrefillProgress progress;
    progress.summary.prompt_tokens        = static_cast<std::uint32_t>(N);
    progress.summary.reused_prompt_tokens = st.reused_prompt_tokens;
    progress.summary.prefix_reuse_path =
        st.reused_prompt_tokens > 0
            ? (st.reused_from_turn_closure
                   ? PrefixReusePath::PrivateTurnClosure
                   : PrefixReusePath::PrivateEndpoint)
            : PrefixReusePath::Root;
    progress.processed_prompt_tokens      = chunk_size;
    progress.complete                     = true;
    // Attribute the round's blocking device stall to the wait bucket rather than
    // leaving it in the submit residual. See ExecutionTimingRecorder.
    timing.reclassify_submit_as_wait(impl_->executor_.take_round_device_wait_ns());
    progress.timing                       = timing.finish();
    progress.pending                      = ContractAccess::make_pending(
        this, 1, std::span(&sequence, 1),
        std::span(impl_->pending_batch_tokens_.data(), 1),
        std::span(impl_->pending_batch_row_counts_.data(), 1), 1, progress.timing);
    return progress;
}

CaptureAssessment
Program::inspect_capture(const CaptureOffer& offer,
                         const SharedPrefixHandle* /*exact_shared*/,
                         const SharedPrefixHandle* /*replacement*/,
                         std::optional<runtime::CheckpointRef> /*private_replacement*/,
                         bool /*permit_shared_publication*/) const {
    if (impl_ == nullptr || offer.owner() != this) {
        return CaptureAssessment{};
    }
    const std::uint32_t lane_idx = offer.lane().value;
    if (lane_idx >= impl_->plan_.config.max_concurrency) {
        return CaptureAssessment{};
    }
    const auto& st = impl_->lane_states_[lane_idx];
    if (!st.active || st.epoch != offer.epoch()) {
        return CaptureAssessment{};
    }
    if (impl_->plan_.config.continuation_capacity == 0) {
        return CaptureAssessment{};
    }

    CaptureAssessment assessment;
    const std::uint32_t N        = st.prompt_tokens_processed;
    assessment.frontier          = N;
    assessment.publishes_private = st.publish_continuation && (impl_->plan_.config.continuation_capacity > 0);
    assessment.publishes_shared  = false;
    assessment.shortlist_key     = PrefixShortlistKey{
        .digests      = st.prefix_digests.at(N),
        .frontier     = N,
        .identity_tag = 0,
    };
    // Admission reserves this lane's rewrite together with its endpoint. Capture must not
    // depend on an unrelated vacant slot surviving until the prefill frontier is reached.
    assessment.physically_feasible = st.reserved_turn_closure_index.has_value();
    return assessment;
}

CapturePressurePlanningSession Program::begin_capture_pressure_planning(
    const CaptureAssessment& /*assessment*/,
    std::span<const ContinuationHandle* const> /*private_owners*/,
    std::span<const runtime::PlanningOwnerId> /*private_owner_ids*/,
    std::span<const SharedPrefixHandle* const> /*shared_owners*/,
    std::span<const runtime::PlanningOwnerId> /*shared_owner_ids*/,
    // Accepted so the engine's call matches; unread for the same reason as in
    // `begin_pressure_planning` below -- this model ranks through its own guidance API.
    std::span<const runtime::PlanningOwnerId> /*recency_order*/) {
    throw std::logic_error("FlashNext does not support capture pressure planning");
}

bool Program::shared_capture_matches(const CaptureOffer& /*offer*/,
                                     const SharedPrefixHandle& /*shared*/) const {
    return false;
}

void Program::skip_capture(CaptureOffer&& offer) {
    if (impl_ == nullptr || offer.owner() != this) { return; }
    const std::uint32_t lane_idx = offer.lane().value;
    if (lane_idx < impl_->plan_.config.max_concurrency) {
        impl_->lane_states_[lane_idx].pending_capture_offer = 0;
        auto& st = impl_->lane_states_[lane_idx];
        if (st.reserved_turn_closure_index) {
            impl_->vacate_slot(*st.reserved_turn_closure_index);
            st.reserved_turn_closure_index.reset();
            ++impl_->resource_revision_;
        }
    }
    ContractAccess::consume(offer);
}

runtime::ContextTransactionReserveStatus
Program::reserve_active_capture(CaptureOffer&& offer,
                                const SharedPrefixHandle* /*exact_shared*/,
                                const SharedPrefixHandle* /*replacement*/,
                                std::optional<runtime::CheckpointRef> /*private_replacement*/,
                                bool /*permit_shared_publication*/,
                                runtime::CancellationFlagView cancellation) {
    if (impl_ == nullptr || offer.owner() != this) {
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (cancellation.requested()) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    const std::uint32_t lane_idx = offer.lane().value;
    if (lane_idx >= impl_->plan_.config.max_concurrency) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    auto& st = impl_->lane_states_[lane_idx];
    if (!st.active || st.epoch != offer.epoch()) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }
    if (impl_->has_context_transaction_) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }

    const std::int32_t slot_idx = st.reserved_turn_closure_index
        ? static_cast<std::int32_t>(*st.reserved_turn_closure_index) : -1;
    if (slot_idx < 0) {
        skip_capture(std::move(offer));
        return runtime::ContextTransactionReserveStatus::Aborted;
    }

    const std::int32_t capture_frontier = static_cast<std::int32_t>(st.prompt_tokens_processed);
    auto& c_slot              = impl_->continuation_slots_[slot_idx];
    c_slot.role               = detail::ContinuationSlotRole::Catalogued;
    st.reserved_turn_closure_index.reset();
    c_slot.generation         = ++impl_->continuation_epoch_;
    c_slot.committed_tokens.assign(st.prompt_tokens.begin(),
                                   st.prompt_tokens.begin() + capture_frontier);
    c_slot.committed_frontier = capture_frontier;
    c_slot.prefix_digests     = st.prefix_digests.at(capture_frontier);
    if (std::getenv("NINFER_FLASH_NEXT_TRACE_KEYS") != nullptr) {
        std::fprintf(stderr, "[fnkey] capture publish f=%d d0=%016llx d1=%016llx\n", capture_frontier,
                     static_cast<unsigned long long>(c_slot.prefix_digests[0]),
                     static_cast<unsigned long long>(c_slot.prefix_digests[1]));
    }
    c_slot.last_used_epoch    = ++impl_->continuation_epoch_;
    c_slot.kind               = runtime::CheckpointKind::TurnClosure;
    const auto groups         = impl_->executor_.lane_physical_groups(st.lane_handle);
    c_slot.physical_groups.assign(groups.begin(), groups.end());
    c_slot.history            = impl_->executor_.lane_history(st.lane_handle);
    impl_->executor_.acquire_physical_groups(c_slot.physical_groups);

    const auto active_slot = impl_->allocation_.current_source_slot(lane_idx);
    impl_->executor_.copy_state_slot(static_cast<std::uint32_t>(active_slot), c_slot.cache_slot);

    st.turn_closure_continuation_index = static_cast<std::uint32_t>(slot_idx);
    st.pending_capture_offer           = 0;
    ContractAccess::consume(offer);

    impl_->has_context_transaction_ = true;
    impl_->is_capture_transaction_  = true;
    impl_->transaction_lane_        = runtime::LaneId(lane_idx);
    impl_->transaction_epoch_       = st.epoch;

    ActiveCaptureResult capture_res;
    capture_res.status = runtime::ContextTransactionStatus::Published;
    capture_res.active_summary.rewrite = CheckpointSummary{
        .ref = runtime::CheckpointRef{
            .kind     = runtime::CheckpointKind::TurnClosure,
            .frontier = static_cast<std::uint32_t>(capture_frontier),
            .ordinal  = 0,
        },
        .shortlist_key = PrefixShortlistKey{
            .digests      = c_slot.prefix_digests,
            .frontier     = static_cast<std::uint32_t>(capture_frontier),
            .identity_tag = 0,
        },
        .state_residency = runtime::ReplicaResidency::DeviceOnly,
        .required_kv = TargetKVRequirement{
            .main_frontier    = static_cast<std::uint32_t>(capture_frontier),
            .backend_frontier = 0,
            .main_pages       = static_cast<std::uint32_t>(c_slot.physical_groups.size() * 4),
            .backend_pages    = 0,
        },
        .rebuild_work = runtime::make_prefill_work(
            0, static_cast<std::uint32_t>(capture_frontier), 0, 0,
            impl_->plan_.config.prefill_chunk),
    };
    impl_->pending_capture_result_ = std::move(capture_res);

    return runtime::ContextTransactionReserveStatus::Reserved;
}

runtime::ContextTransactionReserveStatus Program::reserve_active_capture_with_pressure(
    CaptureOffer&& offer, const SharedPrefixHandle* exact_shared,
    const SharedPrefixHandle* replacement,
    std::optional<runtime::CheckpointRef> private_replacement, bool permit_shared_publication,
    CapturePressurePlan&& /*pressure*/,
    runtime::CancellationFlagView cancellation) {
    return reserve_active_capture(std::move(offer), exact_shared, replacement,
                                  private_replacement, permit_shared_publication,
                                  cancellation);
}

PendingBatch Program::decode(std::span<const SequenceHandle> sequences,
                             std::span<const runtime::RoundBudget> budgets,
                             runtime::ExecutionTiming* failed_timing) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    const std::size_t B = sequences.size();
    if (B == 0 || B > impl_->plan_.config.max_concurrency) {
        throw std::invalid_argument("decode batch size invalid");
    }
    if (budgets.size() != B || std::any_of(budgets.begin(), budgets.end(), [](const auto& budget) {
            return budget.generated_tokens_remaining == 0;
        })) {
        throw std::invalid_argument("decode requires a positive remaining budget for each row");
    }
    if (impl_->pending_round_.valid()) {
        throw std::logic_error("cannot decode while a pending round is uncommitted");
    }

    // Speculative path: single sequence (B == 1) with speculative decoding enabled
    if (B == 1 && impl_->plan_.config.speculative_draft_tokens > 0 &&
        impl_->has_mtp()) {
        const auto& seq = sequences[0];
        if (seq.owner() != this) {
            throw std::logic_error("SequenceHandle does not belong to this Program");
        }
        const std::uint32_t lane_idx = seq.lane().value;
        if (lane_idx >= impl_->plan_.config.max_concurrency) {
            throw std::out_of_range("sequence lane out of range");
        }
        auto& st = impl_->lane_states_[lane_idx];
        if (!st.active || st.epoch != seq.epoch()) {
            throw std::logic_error("sequence lane is not active or epoch mismatch");
        }

        const std::uint32_t K =
            std::min({4U, impl_->plan_.config.speculative_draft_tokens,
                      budgets[0].generated_tokens_remaining - 1U,
                      impl_->plan_.config.max_context -
                          static_cast<std::uint32_t>(st.last_token_index) - 1U});

        const std::int32_t last_token_index =
            st.last_token_index + static_cast<std::int32_t>(K);
        const std::size_t req_groups =
            static_cast<std::size_t>(last_token_index /
                                     static_cast<std::int32_t>(detail::kMainPageGroupTokens)) +
            1U;
        const std::size_t owned = impl_->executor_.lane_physical_groups(st.lane_handle).size();
        if (req_groups > owned) {
            impl_->ensure_physical_groups_available(req_groups - owned);
        }

        // If draft tokens are not already prepared, draft them from current state
        if (st.draft_tokens.empty()) {
            st.draft_tokens.resize(K);
            Tensor last_hidden = impl_->allocation_.state_view().mtp_backbone_hidden.slice(1, impl_->allocation_.current_source_slot(lane_idx), 1);
            std::array<std::int32_t, 3> mrope_pos = {st.last_token_pos, st.last_token_pos,
                                                     st.last_token_pos};
            impl_->executor_.draft_mtp_tokens(st.lane_handle, st.last_token_id, st.last_token_index,
                                              mrope_pos, last_hidden, K, st.draft_tokens);
        }

        std::array<std::int32_t, 3> first_mrope_pos = {st.last_token_pos, st.last_token_pos,
                                                       st.last_token_pos};
        std::array<ops::SamplingConfig, 5> verification_sampling;
        verification_sampling.fill(st.sampling_config);
        if (st.output_constraint) {
            auto preview = st.output_constraint->fork();
            for (std::size_t column = 0; column <= st.draft_tokens.size(); ++column) {
                verification_sampling[column].allowed_tokens = impl_->upload_constraint_mask(lane_idx, static_cast<std::uint32_t>(column), preview);
                if (column < st.draft_tokens.size() &&
                    (!preview.try_accept(st.draft_tokens[column]) || preview.terminated())) {
                    st.draft_tokens.resize(column);
                    break;
                }
            }
        }
        impl_->pending_round_ = impl_->executor_.execute_speculative_verify_round(
            st.lane_handle, st.last_token_id, st.draft_tokens, st.last_token_index, first_mrope_pos,
            st.sampling_config, std::span(verification_sampling.data(), st.draft_tokens.size() + 1));

        const auto sampled = impl_->pending_round_.sampled_tokens();

        // Verification loop: compare sampled[k] with draft_tokens[k]
        std::vector<std::int32_t> accepted;
        accepted.reserve(st.draft_tokens.size() + 1);

        bool mismatch = false;
        for (std::size_t k = 0; k < st.draft_tokens.size(); ++k) {
            const std::int32_t target_tok = sampled[k];
            accepted.push_back(target_tok);
            if (target_tok != st.draft_tokens[k]) {
                mismatch = true;
                break;
            }
        }
        if (!mismatch && sampled.size() > st.draft_tokens.size()) {
            accepted.push_back(sampled[st.draft_tokens.size()]);
        }

        st.pending_accepted_tokens = accepted;
        if (st.speculative_stats.enabled) {
            st.speculative_stats.rounds += 1;
            st.speculative_stats.drafted_tokens += st.draft_tokens.size();
            const std::size_t num_accepted_draft = accepted.size() > 1 ? (accepted.size() - 1) : 0;
            st.speculative_stats.accepted_tokens += num_accepted_draft;
            for (std::size_t p = 0; p < num_accepted_draft && p < st.speculative_stats.accepted_per_position.size(); ++p) {
                st.speculative_stats.accepted_per_position[p] += 1;
            }
        }

        impl_->pending_batch_tokens_.resize(accepted.size());
        for (std::size_t i = 0; i < accepted.size(); ++i) {
            impl_->pending_batch_tokens_[i] = static_cast<TokenId>(accepted[i]);
        }
        impl_->pending_batch_row_counts_.resize(1);
        impl_->pending_batch_row_counts_[0] = static_cast<std::int32_t>(accepted.size());

        // Attribute the round's blocking device stall to the wait bucket rather than
        // leaving it in the submit residual. See ExecutionTimingRecorder.
        timing.reclassify_submit_as_wait(impl_->executor_.take_round_device_wait_ns());
        const auto exec_timing = timing.finish();
        return ContractAccess::make_pending(
            this, impl_->pending_round_.valid() ? 1 : 0, sequences,
            std::span(impl_->pending_batch_tokens_.data(), accepted.size()),
            std::span(impl_->pending_batch_row_counts_.data(), 1),
            static_cast<std::uint32_t>(accepted.size()), exec_timing);
    }

    // Standard non-speculative path
    std::vector<detail::LaneStepRequest> requests(B);
    std::vector<std::uint32_t> lane_indices(B);

    for (std::size_t b = 0; b < B; ++b) {
        const auto& seq = sequences[b];
        if (seq.owner() != this) {
            throw std::logic_error("SequenceHandle does not belong to this Program");
        }
        const std::uint32_t lane_idx = seq.lane().value;
        if (lane_idx >= impl_->plan_.config.max_concurrency) {
            throw std::out_of_range("sequence lane out of range");
        }
        auto& st = impl_->lane_states_[lane_idx];
        if (!st.active || st.epoch != seq.epoch()) {
            throw std::logic_error("sequence lane is not active or epoch mismatch");
        }
        lane_indices[b] = lane_idx;
        if (st.speculative_stats.enabled) {
            st.speculative_stats.fallback_steps += 1;
            st.speculative_stats.rounds += 1;
        }

        requests[b] = detail::LaneStepRequest{
            .handle          = st.lane_handle,
            .token_id        = st.last_token_id,
            .token_index     = st.last_token_index,
            .mrope_positions = {st.last_token_pos, st.last_token_pos, st.last_token_pos},
            .sampling        = st.sampling_config,
        };
        if (st.output_constraint) {
            requests[b].sampling.allowed_tokens = impl_->upload_constraint_mask(lane_idx, 0, *st.output_constraint);
        }
    }

    std::size_t total_needed = 0;
    for (std::size_t b = 0; b < B; ++b) {
        const auto& st = impl_->lane_states_[lane_indices[b]];
        const std::int32_t next_token_index = st.last_token_index;
        const std::size_t req_groups =
            static_cast<std::size_t>(next_token_index / static_cast<std::int32_t>(detail::kMainPageGroupTokens)) + 1U;
        const std::size_t owned = impl_->executor_.lane_physical_groups(st.lane_handle).size();
        if (req_groups > owned) {
            total_needed += (req_groups - owned);
        }
    }
    if (total_needed > 0) {
        impl_->ensure_physical_groups_available(total_needed);
    }

    impl_->pending_round_ = impl_->executor_.execute_round(requests);

    const auto sampled = impl_->pending_round_.sampled_tokens();

    impl_->pending_batch_tokens_.resize(B);
    impl_->pending_batch_row_counts_.resize(B);
    for (std::size_t b = 0; b < B; ++b) {
        impl_->pending_batch_tokens_[b]     = static_cast<TokenId>(sampled[b]);
        impl_->pending_batch_row_counts_[b] = 1;
    }

    // Attribute the round's blocking device stall to the wait bucket rather than
    // leaving it in the submit residual. See ExecutionTimingRecorder.
    timing.reclassify_submit_as_wait(impl_->executor_.take_round_device_wait_ns());
    const auto exec_timing = timing.finish();
    return ContractAccess::make_pending(
        this, impl_->pending_round_.valid() ? 1 : 0, sequences,
        std::span(impl_->pending_batch_tokens_.data(), B),
        std::span(impl_->pending_batch_row_counts_.data(), B), 1, exec_timing);
}

runtime::ExecutionTiming
Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
                              std::span<const TokenId> row_major_tokens, std::uint32_t row_stride,
                              std::span<const std::optional<std::uint32_t>> prefix_execution_splits,
                              runtime::ExecutionTiming* failed_timing) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    const std::size_t B = sequences.size();
    if (B == 0 || B > impl_->plan_.config.max_concurrency || row_stride == 0 ||
        row_major_tokens.size() != B * row_stride) {
        throw std::invalid_argument("append_forced_tokens arguments invalid");
    }
    // The split extent is a fusion-generation parameter that neither reference target has. Accept the
    // span (the engine always passes one, `engine_core.h:2159`) but refuse a row that actually asks
    // for a split: running such a row whole would produce right-looking output from the wrong
    // execution, which is exactly the class of silent wrong answer this project forbids. A caller
    // that wants splitting must implement it here first.
    if (!prefix_execution_splits.empty()) {
        if (prefix_execution_splits.size() != B) {
            throw std::invalid_argument("append_forced_tokens split span does not match the batch");
        }
        for (const std::optional<std::uint32_t>& split : prefix_execution_splits) {
            if (split.has_value()) {
                throw std::invalid_argument(
                    "append_forced_tokens: prefix execution splits are not supported for this target");
            }
        }
    }
    if (impl_->pending_round_.valid()) {
        throw std::logic_error("cannot append forced tokens while pending round is uncommitted");
    }

    for (std::size_t b = 0; b < B; ++b) {
        const auto& seq = sequences[b];
        if (seq.owner() != this) {
            throw std::logic_error("SequenceHandle does not belong to this Program");
        }
        const std::uint32_t lane_idx = seq.lane().value;
        auto& st                     = impl_->lane_states_[lane_idx];
        if (!st.active || st.epoch != seq.epoch()) {
            throw std::logic_error("sequence lane not active or epoch mismatch");
        }

        for (std::uint32_t s = 0; s < row_stride; ++s) {
            const TokenId tok = row_major_tokens[b * row_stride + s];
            if (st.output_constraint) { st.output_constraint->accept(std::span(&tok, 1)); }
            detail::LaneStepRequest req{
                .handle          = st.lane_handle,
                .token_id        = static_cast<std::int32_t>(tok),
                .token_index     = st.last_token_index + 1,
                .mrope_positions = {st.last_token_pos + 1, st.last_token_pos + 1,
                                    st.last_token_pos + 1},
            };
            const std::size_t req_groups =
                static_cast<std::size_t>(req.token_index / static_cast<std::int32_t>(detail::kMainPageGroupTokens)) + 1U;
            const std::size_t owned = impl_->executor_.lane_physical_groups(st.lane_handle).size();
            if (req_groups > owned) {
                impl_->ensure_physical_groups_available(req_groups - owned);
            }

            auto round = impl_->executor_.execute_round(std::span(&req, 1));
            std::array<detail::LaneCommitDecision, 1> decision = {{{.accept = true}}};
            round.commit(decision);

            st.last_token_id    = req.token_id;
            st.last_token_pos   = req.mrope_positions[0];
            st.last_token_index = req.token_index;
            ++st.total_generated_tokens;
        }
    }

    // Attribute the round's blocking device stall to the wait bucket rather than
    // leaving it in the submit residual. See ExecutionTimingRecorder.
    timing.reclassify_submit_as_wait(impl_->executor_.take_round_device_wait_ns());
    return timing.finish();
}

CommitResult
Program::commit(PendingBatch&& pending, std::span<const runtime::CommitDecision> decisions,
                runtime::CommitObservation /*observation*/,
                runtime::ExecutionTiming* failed_timing) {
    if (impl_ == nullptr) { throw std::logic_error("Program: instance is empty"); }

    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    const auto rows     = ContractAccess::rows(pending);
    const std::size_t B = rows.size();
    if (B != decisions.size()) {
        throw std::invalid_argument("decisions size mismatch with pending rows");
    }

    CommitResult result;
    result.row_count = B;

    // Check if we are in speculative decode mode for B=1
    bool is_speculative = false;
    if (B == 1) {
        const std::uint32_t lane_idx = rows[0].lane().value;
        const auto& st               = impl_->lane_states_[lane_idx];
        if (st.active && !st.pending_accepted_tokens.empty()) {
            is_speculative = true;
        }
    }

    if (is_speculative) {
        const auto& seq              = rows[0];
        const auto& dec              = decisions[0];
        const std::uint32_t lane_idx = seq.lane().value;
        auto& st                     = impl_->lane_states_[lane_idx];

        if (dec.accepted_tokens > 0) {
            if (dec.accepted_tokens > st.pending_accepted_tokens.size()) {
                throw std::invalid_argument("speculative commit exceeds the verified token count");
            }
            st.pending_accepted_tokens.resize(dec.accepted_tokens);
            // Commit accepted tokens via speculative commit
            if (impl_->pending_round_.valid()) {
                impl_->pending_round_.commit_speculative(lane_idx, st.pending_accepted_tokens);
            }

            for (const auto tok_i32 : st.pending_accepted_tokens) {
                const TokenId sampled = static_cast<TokenId>(tok_i32);
                if (st.output_constraint) { st.output_constraint->accept(std::span(&sampled, 1)); }
                st.prompt_tokens.push_back(sampled);
                st.prefix_digests.append_generated(std::span(&sampled, 1), 0);
                st.last_token_id = tok_i32;
                st.last_token_pos += 1;
                st.last_token_index += 1;
                st.committed_frontier += 1;
                st.total_generated_tokens += 1;
            }

            st.draft_tokens.clear();
            st.pending_accepted_tokens.clear();

            if (dec.terminal) {
                result.rows[0].disposition = runtime::CommitDisposition::Finishable;
            } else {
                result.rows[0].disposition = runtime::CommitDisposition::Active;
            }
        } else {
            st.draft_tokens.clear();
            st.pending_accepted_tokens.clear();
            if (impl_->pending_round_.valid()) {
                impl_->pending_round_.abort();
            }
            impl_->drop_unpublished_checkpoints(st);
            impl_->executor_.release_lane(st.lane_handle);
            st.active   = false;
            st.finished = true;
            st.reused_from_continuation_index.reset();
            st.reused_from_continuation_generation.reset();
            st.turn_closure_continuation_index.reset();
            st.pending_capture_offer = 0;
            ++impl_->resource_revision_;
            result.rows[0].disposition = runtime::CommitDisposition::CancelledReleased;
        }
    } else {
        // Standard non-speculative path (or prefill commit, or B > 1)
        std::vector<detail::LaneCommitDecision> lane_decisions(B);
        for (std::size_t b = 0; b < B; ++b) {
            lane_decisions[b].accept = (decisions[b].accepted_tokens > 0);
        }

        if (impl_->pending_round_.valid()) {
            impl_->pending_round_.commit(lane_decisions);
        }

        for (std::size_t b = 0; b < B; ++b) {
            const auto& seq              = rows[b];
            const auto& dec              = decisions[b];
            const std::uint32_t lane_idx = seq.lane().value;
            auto& st                     = impl_->lane_states_[lane_idx];

            if (dec.accepted_tokens > 0) {
                const TokenId sampled = pending.tokens()[b];
                if (st.output_constraint) { st.output_constraint->accept(std::span(&sampled, 1)); }
                st.prompt_tokens.push_back(sampled);
                st.prefix_digests.append_generated(std::span(&sampled, 1), 0);
                st.committed_frontier += 1;
                st.last_token_id      = static_cast<std::int32_t>(sampled);
                st.last_token_pos += 1;
                st.last_token_index += 1;
                ++st.total_generated_tokens;

                st.draft_tokens.clear();

                if (dec.terminal) {
                    result.rows[b].disposition = runtime::CommitDisposition::Finishable;
                } else {
                    result.rows[b].disposition = runtime::CommitDisposition::Active;
                }
            } else {
                st.draft_tokens.clear();
                st.pending_accepted_tokens.clear();
                impl_->drop_unpublished_checkpoints(st);
                impl_->executor_.release_lane(st.lane_handle);
                st.active   = false;
                st.finished = true;
                st.reused_from_continuation_index.reset();
                st.reused_from_continuation_generation.reset();
                st.turn_closure_continuation_index.reset();
                st.pending_capture_offer = 0;
                ++impl_->resource_revision_;
                result.rows[b].disposition = runtime::CommitDisposition::CancelledReleased;
            }
        }
    }

    ContractAccess::consume(pending);
    // Attribute the round's blocking device stall to the wait bucket rather than
    // leaving it in the submit residual. See ExecutionTimingRecorder.
    timing.reclassify_submit_as_wait(impl_->executor_.take_round_device_wait_ns());
    result.timing = timing.finish();
    return result;
}

DiscardResult Program::abort_pending(PendingBatch&& pending) noexcept {
    const auto rows = ContractAccess::rows(pending);
    const auto count = rows.size();
    if (impl_ != nullptr && impl_->pending_round_.valid()) {
        impl_->pending_round_.abort();
    }
    if (impl_ != nullptr) {
        for (const auto& sequence : rows) {
            const auto lane = sequence.lane().value;
            if (lane >= impl_->lane_states_.size()) { continue; }
            auto& state = impl_->lane_states_[lane];
            if (!state.active || state.epoch != sequence.epoch()) { continue; }
            impl_->drop_unpublished_checkpoints(state);
            impl_->executor_.release_lane(state.lane_handle);
            state.active = false;
            state.finished = true;
            state.draft_tokens.clear();
            state.pending_accepted_tokens.clear();
            ++impl_->resource_revision_;
        }
    }
    ContractAccess::consume(pending);
    return DiscardResult{
        .status    = runtime::ConsumeStatus::Consumed,
        .row_count = count,
    };
}

FinishResult Program::finish(SequenceHandle sequence) noexcept {
    if (impl_ != nullptr) {
        try {
            if (impl_->pending_round_.valid()) {
                impl_->pending_round_.abort();
            }
            const std::uint32_t lane_idx = sequence.lane().value;
            if (lane_idx < impl_->plan_.config.max_concurrency) {
                auto& st = impl_->lane_states_[lane_idx];
                if (st.active && st.epoch == sequence.epoch()) {
                    SpeculativeStats lane_spec = std::move(st.speculative_stats);
                    const bool should_catalogue =
                        st.publish_continuation && (impl_->plan_.config.continuation_capacity > 0) &&
                        (st.committed_frontier > 0);

                    if (should_catalogue) {
                        const std::int32_t target_c_idx = st.reserved_endpoint_index
                            ? static_cast<std::int32_t>(*st.reserved_endpoint_index) : -1;
                        if (target_c_idx >= 0) {
                            auto& c_slot = impl_->continuation_slots_[target_c_idx];
                            c_slot.role = detail::ContinuationSlotRole::Catalogued;
                            c_slot.kind = runtime::CheckpointKind::SessionEndpoint;
                            c_slot.published_checkpoints = 1;
                            c_slot.paired_rewrite_slot.reset();
                            c_slot.paired_rewrite_generation = 0;
                            c_slot.last_used_epoch = ++impl_->continuation_epoch_;
                            c_slot.committed_frontier = st.committed_frontier;
                            c_slot.committed_tokens = st.prompt_tokens;
                            c_slot.history = impl_->executor_.lane_history(st.lane_handle);
                            c_slot.physical_groups = impl_->executor_.take_lane_physical_groups(st.lane_handle);

                            const std::int32_t active_slot = impl_->executor_.allocation().current_source_slot(lane_idx);
                            impl_->executor_.copy_state_slot(static_cast<std::uint32_t>(active_slot), c_slot.cache_slot);

                            const auto digests = st.prefix_digests.at(st.committed_frontier);
                            c_slot.prefix_digests = digests;
                            if (std::getenv("NINFER_FLASH_NEXT_TRACE_KEYS") != nullptr) {
                                std::fprintf(stderr, "[fnkey] finish endpoint f=%d d0=%016llx d1=%016llx\n",
                                             st.committed_frontier,
                                             static_cast<unsigned long long>(digests[0]),
                                             static_cast<unsigned long long>(digests[1]));
                            }

                            FinishResult out;
                            CheckpointSummary cp;
                            cp.ref = runtime::CheckpointRef{
                                .frontier = static_cast<std::uint32_t>(st.committed_frontier),
                                .ordinal  = 0,
                            };
                            cp.shortlist_key = PrefixShortlistKey{
                                .digests      = digests,
                                .frontier     = static_cast<std::uint32_t>(st.committed_frontier),
                                .identity_tag = 0,
                            };
                            cp.state_residency = runtime::ReplicaResidency::DeviceOnly;
                            cp.required_kv = TargetKVRequirement{
                                .main_frontier    = static_cast<std::uint32_t>(st.committed_frontier),
                                .backend_frontier = 0,
                                .main_pages       = static_cast<std::uint32_t>(c_slot.physical_groups.size() * 4),
                                .backend_pages    = 0,
                            };
                            cp.rebuild_work = runtime::make_prefill_work(
                                0, static_cast<std::uint32_t>(st.committed_frontier), 0, 0,
                                impl_->plan_.config.prefill_chunk);

                            out.summary.endpoint          = cp;
                            out.summary.active_references = 0;

                            if (st.turn_closure_continuation_index.has_value()) {
                                const std::uint32_t turn_idx = *st.turn_closure_continuation_index;
                                if (turn_idx < impl_->continuation_slots_.size()) {
                                    const auto& turn_slot = impl_->continuation_slots_[turn_idx];
                                    if (turn_slot.role == detail::ContinuationSlotRole::Catalogued) {
                                        CheckpointSummary rw;
                                        rw.ref = runtime::CheckpointRef{
                                            .kind     = runtime::CheckpointKind::TurnClosure,
                                            .frontier = static_cast<std::uint32_t>(turn_slot.committed_frontier),
                                            .ordinal  = 0,
                                        };
                                        if (std::getenv("NINFER_FLASH_NEXT_TRACE_KEYS") != nullptr) {
                                            std::fprintf(stderr, "[fnkey] finish rewrite f=%d d0=%016llx d1=%016llx\n",
                                                         turn_slot.committed_frontier,
                                                         static_cast<unsigned long long>(turn_slot.prefix_digests[0]),
                                                         static_cast<unsigned long long>(turn_slot.prefix_digests[1]));
                                        }
                                        rw.shortlist_key = PrefixShortlistKey{
                                            .digests      = turn_slot.prefix_digests,
                                            .frontier     = static_cast<std::uint32_t>(turn_slot.committed_frontier),
                                            .identity_tag = 0,
                                        };
                                        rw.state_residency = runtime::ReplicaResidency::DeviceOnly;
                                        rw.required_kv = TargetKVRequirement{
                                            .main_frontier    = static_cast<std::uint32_t>(turn_slot.committed_frontier),
                                            .backend_frontier = 0,
                                            .main_pages       = static_cast<std::uint32_t>(turn_slot.physical_groups.size() * 4),
                                            .backend_pages    = 0,
                                        };
                                        rw.rebuild_work = runtime::make_prefill_work(
                                            0, static_cast<std::uint32_t>(turn_slot.committed_frontier), 0, 0,
                                            impl_->plan_.config.prefill_chunk);
                                        out.summary.rewrite = rw;
                                        c_slot.published_checkpoints     = 2;
                                        c_slot.paired_rewrite_slot       = turn_idx;
                                        c_slot.paired_rewrite_generation = turn_slot.generation;
                                    }
                                }
                            }

                            st.reserved_endpoint_index.reset();
                            impl_->release_checkpoint_reservations(st);
                            impl_->executor_.release_lane(st.lane_handle);
                            st.active   = false;
                            st.finished = true;
                            st.reused_from_continuation_index.reset();
                            st.reused_from_continuation_generation.reset();
                            st.turn_closure_continuation_index.reset();
                            st.pending_capture_offer = 0;
                            out.status               = runtime::ConsumeStatus::Consumed;
                            out.disposition          = runtime::FinishDisposition::Catalogued;
                            out.speculative          = std::move(lane_spec);
                            out.continuation         = ContinuationHandle(
                                this, static_cast<std::uint32_t>(target_c_idx), c_slot.generation);
                            ++impl_->resource_revision_;
                            return out;
                        }
                    }

                    impl_->drop_unpublished_checkpoints(st);
                    impl_->executor_.release_lane(st.lane_handle);
                    st.active   = false;
                    st.finished = true;
                    st.reused_from_continuation_index.reset();
                    st.reused_from_continuation_generation.reset();
                    st.turn_closure_continuation_index.reset();
                    st.pending_capture_offer = 0;
                    ++impl_->resource_revision_;
                    return FinishResult{
                        .status      = runtime::ConsumeStatus::Consumed,
                        .disposition = runtime::FinishDisposition::Released,
                        .speculative = std::move(lane_spec),
                    };
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[flash_next finish] lane %u: swallowed exception: %s\n",
                         sequence.lane().value, e.what());
        } catch (...) {
            std::fprintf(stderr, "[flash_next finish] lane %u: swallowed unknown exception\n",
                         sequence.lane().value);
        }
    }
    return FinishResult{
        .status      = runtime::ConsumeStatus::Consumed,
        .disposition = runtime::FinishDisposition::Released,
    };
}

AbortResult Program::abort(SequenceHandle sequence) noexcept {
    if (impl_ != nullptr) {
        const std::uint32_t lane_idx = sequence.lane().value;
        if (lane_idx < impl_->plan_.config.max_concurrency) {
            auto& st = impl_->lane_states_[lane_idx];
            if (st.active && st.epoch == sequence.epoch()) {
                SpeculativeStats lane_spec = std::move(st.speculative_stats);
                if (impl_->pending_round_.valid()) {
                    impl_->pending_round_.abort();
                }
                impl_->drop_unpublished_checkpoints(st);
                impl_->executor_.release_lane(st.lane_handle);
                st.active   = false;
                st.finished = true;
                st.reused_from_continuation_index.reset();
                st.reused_from_continuation_generation.reset();
                st.turn_closure_continuation_index.reset();
                st.pending_capture_offer = 0;
                ++impl_->resource_revision_;
                return AbortResult{
                    .status      = runtime::ConsumeStatus::Consumed,
                    .speculative = std::move(lane_spec),
                };
            }
        }
    }
    return AbortResult{
        .status = runtime::ConsumeStatus::Consumed,
    };
}

ReleaseResult Program::release_continuation(ContinuationHandle&& continuation) noexcept {
    ReleaseResult out{ .status = runtime::ConsumeStatus::Consumed };
    if (impl_ == nullptr) { return out; }
    if (continuation.owner() != this) { return out; }

    const std::uint32_t c_idx = continuation.index();
    const std::uint64_t gen   = continuation.generation();

    if (c_idx < impl_->continuation_slots_.size()) {
        auto& c_slot = impl_->continuation_slots_[c_idx];
        if (c_slot.role == detail::ContinuationSlotRole::Catalogued && c_slot.generation == gen) {
            impl_->vacate_owner(c_idx);
            ++impl_->resource_revision_;
        }
    }
    return out;
}

ReleaseResult Program::release_shared_prefix(SharedPrefixHandle&& /*shared*/) noexcept {
    return ReleaseResult{
        .status = runtime::ConsumeStatus::Consumed,
    };
}

// Widened from upstream's `void` to `std::optional<PhysicalUsageSnapshot>`: the engine picks
// between this and `shutdown_cleanup` through one `decltype` (engine_core.h:2260-2263), so the two
// must return the same type. `nullopt` is "cleanup published no leak report" - the engine then
// skips its recovery diagnostic instead of printing a fabricated one. When there was no Program at
// all there is nothing to report either, which is the same answer.
std::optional<PhysicalUsageSnapshot> Program::fail_all_cleanup() noexcept {
    if (impl_ == nullptr) { return std::nullopt; }
    if (impl_->pending_round_.valid()) {
        impl_->pending_round_.abort();
    }
    for (std::uint32_t l = 0; l < impl_->plan_.config.max_concurrency; ++l) {
        auto& st = impl_->lane_states_[l];
        if (st.active) {
            impl_->drop_unpublished_checkpoints(st);
            impl_->executor_.release_lane(st.lane_handle);
            st.active   = false;
            st.finished = true;
            st.reused_from_continuation_index.reset();
            st.reused_from_continuation_generation.reset();
            st.turn_closure_continuation_index.reset();
            st.pending_capture_offer = 0;
        }
    }
    for (std::size_t c = 0; c < impl_->continuation_slots_.size(); ++c) {
        impl_->vacate_slot(c);
    }
    impl_->has_context_transaction_ = false;
    impl_->transaction_lane_.reset();
    impl_->transaction_epoch_.reset();
    impl_->transaction_has_source_ = false;
    impl_->transaction_victims_.clear();
    ++impl_->resource_revision_;
    return std::nullopt;
}

bool Program::isolated_request_feasible(const RequestBasePlan& base) const noexcept {
    if (impl_ == nullptr || base.impl_ == nullptr) { return false; }
    const std::uint32_t prompt_tokens = base.summary().prompt_tokens;
    const std::uint32_t effective_out = base.summary().effective_output_tokens;
    const std::uint32_t total_tokens  = prompt_tokens + (effective_out > 0 ? effective_out - 1U : 0U);
    if (total_tokens > impl_->plan_.admissible_tokens) { return false; }
    // M4-2: streaming => logical page ceiling; otherwise the physical group check, unchanged.
    if (ninfer::models::qwen3_8_flash_next::detail::flash_next_kv_stream_requested()) {
        return (total_tokens + ninfer::models::qwen3_8_flash_next::detail::kPageTokens - 1U) / ninfer::models::qwen3_8_flash_next::detail::kPageTokens <=
               impl_->plan_.attention_logical_pages;
    }
    return impl_->plan_.main_page_groups >= (total_tokens + 255U) / 256U;
}

runtime::ProgramResourceRevision Program::resource_revision() const noexcept {
    return runtime::ProgramResourceRevision{impl_ != nullptr ? impl_->resource_revision_ : 0ULL};
}

PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    if (impl_ == nullptr) { return {}; }
    std::uint32_t catalogued_slots = 0;
    std::uint32_t reserved_slots = 0;
    for (const auto& c : impl_->continuation_slots_) {
        if (c.role == detail::ContinuationSlotRole::Catalogued) {
            ++catalogued_slots;
        } else if (c.role == detail::ContinuationSlotRole::ReservedEndpoint ||
                   c.role == detail::ContinuationSlotRole::ReservedTurnClosure) {
            ++reserved_slots;
        }
    }
    return PhysicalUsageSnapshot{
        .resource_revision       = impl_->resource_revision_,
        .device_state_slots      = static_cast<std::uint32_t>(impl_->executor_.active_lanes_count()) + catalogued_slots,
        .host_state_slots        = 0,
        .device_main_kv_pages    = static_cast<std::uint32_t>(
            (impl_->plan_.main_page_groups - impl_->executor_.available_physical_groups()) * 4),
        .device_backend_kv_pages = 0,
        .host_kv_bytes           = 0,
        .checkpoint_slots_occupied = catalogued_slots + reserved_slots,
        .checkpoint_slots_capacity = static_cast<std::uint32_t>(impl_->continuation_slots_.size()),
        .checkpoint_slots_reserved = reserved_slots,
    };
}

MemorySummary Program::memory_summary() const noexcept {
    if (impl_ == nullptr) { return {}; }
    MemorySummary out{};
    out.device                            = impl_->device_.device;
    out.max_context                       = impl_->plan_.config.max_context;
    out.kv_capacity                       = impl_->plan_.resolved_tokens;
    out.kv_cache                          = impl_->plan_.config.kv_cache;
    out.runtime_reservation_bytes         = impl_->plan_.total_device_bytes;
    out.minimum_runtime_reservation_bytes =
        impl_->plan_.capacity_curve.minimum_device_reservation_bytes;
    out.kv_capacity_increment_bytes =
        impl_->plan_.capacity_curve.bytes_per_additional_main_page_group;
    const std::size_t weights_bytes =
        impl_->model_data_ != nullptr ? impl_->model_data_->backing.stats().device_capacity_bytes : 0;
    out.weights = ArenaMemorySummary{weights_bytes, weights_bytes, weights_bytes};
    out.sequence =
        ArenaMemorySummary{impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes,
                           impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes,
                           impl_->plan_.total_device_bytes - impl_->plan_.workspace_bytes};
    out.workspace = ArenaMemorySummary{impl_->plan_.workspace_bytes, 0, 0};
    out.cuda_graph_allowance_bytes = impl_->plan_.cuda_graph_allowance_bytes;
    if (impl_->vision_session_.has_value()) {
        out.vision_workspace = impl_->vision_session_->memory_summary(impl_->plan_.config.max_context);
    } else if (impl_->plan_.vision_workspace.has_value()) {
        out.vision_workspace = VisionWorkspaceMemorySummary{
            .aggregate_prompt_tokens = impl_->plan_.config.max_context,
            .max_item_tokens         = impl_->plan_.vision_workspace->max_merged_tokens,
            .general_capacity_bytes  = impl_->plan_.vision_workspace->general_capacity_bytes,
            .encode_peak_bytes       = impl_->plan_.vision_workspace->encode_peak_bytes,
            .handoff_offset_bytes    = impl_->plan_.vision_workspace->handoff_offset_bytes,
            .handoff_capacity_bytes  = impl_->plan_.vision_workspace->handoff_capacity_bytes,
            .handoff_active_bytes    = 0,
            .handoff_peak_bytes      = 0,
        };
    }
    return out;
}

void Program::reset_memory_peaks() noexcept {}

// ---------------------------------------------------------------------------
// Engine-side controls the ported core does not implement.
//
// The engine's ModelContract must be satisfiable at compile time whether or not a feature is
// switched on, so these few entry points exist even where Flash-Next has no counterpart. Each is
// answered in the direction that makes the engine's own guard skip the step: a call is inert
// rather than wrong. The guard that makes the answer unreachable is named with each definition,
// and the two that are NOT stubs (update_sampling, fail_all_cleanup) say so.
// ---------------------------------------------------------------------------

bool Program::vision_pending(SequenceHandle sequence) const noexcept {
    if (impl_ == nullptr) { return false; }
    // Ask the lane ledger rather than returning a constant: a lane that was admitted with vision
    // items and has not finished encoding them still owes a prefill unit. Without this the lane is
    // given no unit until it completes - the wrong diagnosis for a lane that is mid-encode.
    const std::uint32_t lane = sequence.lane().value;
    if (lane >= impl_->lane_states_.size()) { return false; }
    const auto& state = impl_->lane_states_[lane];
    return state.active && state.encoded_item_index.has_value();
}

void Program::update_sampling(SequenceHandle sequence, const ResolvedSamplingParameters& sampling) {
    if (impl_ == nullptr) { throw std::logic_error("sampling update names no Program"); }
    const std::uint32_t lane = sequence.lane().value;
    if (lane >= impl_->lane_states_.size() || !impl_->lane_states_[lane].active) {
        throw std::logic_error("sampling update names no sequence");
    }
    // The sampling op reads the lane's config on the next round and re-uploads it every time
    // (`sample_tokens` rebuilds `host_sampling_configs_` from `lane_states_`), so updating the host
    // copy here is complete. `token_counts` is deliberately untouched: the engine's version also
    // only zeroes it when penalties newly become nonzero, and this core allocates the counts buffer
    // with the lane's own admission.
    ops::SamplingConfig& host = impl_->lane_states_[lane].sampling_config;
    host.temperature       = sampling.temperature;
    host.top_k             = sampling.top_k;
    host.top_p             = sampling.top_p;
    host.min_p             = sampling.min_p;
    host.presence_penalty  = sampling.presence_penalty;
    host.frequency_penalty = sampling.frequency_penalty;
    host.seed              = sampling.seed;
}

bool Program::wait_context_transfer() noexcept {
    // The engine waits on an open context transaction's submitted Host transfer before its next
    // boundary and only `continue`s when this reports one in flight. This core stages experts and
    // context through the executor's own streams, so it has no engine-visible transfer to wait on;
    // false leaves the engine's timed wait in place (engine_core.h:2459).
    return false;
}

std::optional<DeviceKVLeaseShortfall>
Program::device_kv_lease_shortfall(SequenceHandle /*sequence*/) const noexcept {
    // Unreachable in practice: `apply_device_kv_lease_settlements` returns immediately unless
    // `--context-cache` turned `kv_lease_growth` on (engine_core.h:2049), and the lease itself is a
    // ContextCache mechanism. Reporting no shortfall keeps `resume_device_kv_lease` uncalled and
    // leaves the engine's own pool accounting as the only authority on growth.
    return std::nullopt;
}

bool Program::resume_device_kv_lease(SequenceHandle /*sequence*/) noexcept {
    // Reached only from a non-empty `device_kv_lease_shortfall`, which never happens here.
    return false;
}

std::optional<std::uint32_t>
Program::device_kv_lease_settlement_tokens(SequenceHandle /*sequence*/,
                                           std::uint32_t /*forced_span_tokens*/) const noexcept {
    // The engine caps a request's remaining budget only when this returns a limit (engine_core.h
    // :2083). No lease means no cap, so the context/KV ceiling stays the only bound.
    return std::nullopt;
}

DeviceKVPages Program::retained_device_kv_pages(const ContinuationHandle& /*continuation*/) const noexcept {
    // Retention is a ContextCache concept: this core keeps continuations as lane slots with a
    // checkpoint pool, not as retained Device KV pages, and reports that through
    // `physical_usage().checkpoint_slots_*`.
    return {};
}

DeviceKVPages Program::retained_device_kv_pages(const SharedPrefixHandle& /*shared*/) const noexcept {
    // The shared-prefix counterpart of the overload above; `ResourceManager` needs both
    // (`context_cache/resource_manager.h:1200`). The same reasoning applies: this core retains no
    // Device KV pages of its own, so the honest answer is zero.
    return {};
}

std::vector<float> Program::causal_score(qwen3_5::PreparedPrompt&& /*prompt*/,
                                        std::uint32_t /*first_target*/) {
    // Causal scoring drives the perplexity tool through the engine's scoring core. This target
    // implements generation, not scoring; an empty distribution is the honest answer, and the tool
    // reports it as absent rather than as a wrong number.
    return {};
}

std::optional<PhysicalUsageSnapshot> Program::shutdown_cleanup() noexcept {
    return fail_all_cleanup();
}

// The hybrid prefix cache is the engine's ContextCache Host tier. Flash-Next keeps its own
// continuation slots and checkpoint pool, so a hybrid tier would be a second, competing owner of
// the same Host bytes. Every entry point therefore declines the mode; `context_cache.enabled`
// selects `HybridResourceManager` (engine_core.h:186-189), and until it is off the manager is the
// only caller of these.
HybridAdmissionQuote Program::hybrid_quote(const qwen3_5::PreparedPrompt& /*prompt*/,
                                           const RequestBasePlan& /*base*/,
                                           runtime::LaneId destination) {
    // PermanentlyInfeasible, not TemporarilyBlocked: the hybrid manager turns the latter into a
    // readiness the scheduler retries forever (hybrid_resource_manager.h:144), which would park
    // every request instead of disabling the mode.
    return HybridAdmissionQuote{
        .readiness   = runtime::Readiness::PermanentlyInfeasible,
        .destination = destination,
    };
}

runtime::ContextTransactionReserveStatus
Program::hybrid_reserve_materialization(HybridAdmissionQuote&& /*quote*/,
                                        qwen3_5::PreparedPrompt&& /*prompt*/,
                                        runtime::CancellationFlagView cancellation) {
    // `ContextTransactionReserveStatus` has exactly two members, {Reserved, Aborted}; the fusion
    // version names a third (Stale) that this engine's enum does not have. `Aborted` is the one that
    // tells the hybrid manager the reservation neither stands nor should be retried.
    (void)cancellation;
    return runtime::ContextTransactionReserveStatus::Aborted;
}

void Program::set_hybrid_cost(const runtime::prefix_cache::CacheCostModel& /*cost*/) {}

void Program::set_hybrid_coalesce_wait_limit(double /*seconds*/) {}

qwen3_5::HybridCachePersistence
Program::attach_hybrid_cache_file(const std::filesystem::path& /*path*/, std::string /*fingerprint*/) {
    return qwen3_5::HybridCachePersistence{
        .ok      = false,
        .message = "Flash-Next keeps its own continuation slots; the hybrid prefix cache is not "
                   "available for this target",
    };
}

std::optional<qwen3_5::HybridCachePersistence> Program::hybrid_shutdown_save() const {
    // The engine treats `nullopt` as "did not stop cleanly" and warns without failing
    // (engine.cpp:289-295), which is the right report for a Program that never held a hybrid tier.
    return std::nullopt;
}

std::uint32_t Program::hybrid_reclaim_device_kv(std::uint32_t /*main_pages*/,
                                                std::uint32_t /*backend_pages*/) {
    return 0;
}

qwen3_5::HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return {}; }

} // namespace ninfer::models::qwen3_8_flash_next
