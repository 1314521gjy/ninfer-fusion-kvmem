#include "models/qwen3_8_flash_next/impl/lane_ledger.h"

#include "models/qwen3_8_flash_next/impl/expert_staging.h"
#include "models/qwen3_8_flash_next/impl/kv_stream_binding.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next::detail {

FlashNextLaneLedger::FlashNextLaneLedger(const FlashNextRuntimePlan& plan)
    : plan_(plan) {
    const std::uint32_t concurrency = plan_.config.max_concurrency;

    lanes_.resize(concurrency);
    lane_physical_groups_.resize(concurrency);
    previous_group_counts_.resize(concurrency, 0);

    physical_group_refcounts_.resize(plan_.main_page_groups, 0);
    free_physical_groups_.reserve(plan_.main_page_groups);
    for (std::uint32_t g = plan_.main_page_groups; g > 0; --g) {
        free_physical_groups_.push_back(g - 1U);
    }

    host_attention_table_.resize(
        static_cast<std::size_t>(plan_.attention_logical_pages) * concurrency, -1);
    host_indexer_table_.resize(static_cast<std::size_t>(plan_.indexer_logical_pages) * concurrency,
                               -1);
}

LaneHandle FlashNextLaneLedger::allocate_lane(const void* owner) {
    const std::uint32_t concurrency = plan_.config.max_concurrency;
    const void* actual_owner        = owner != nullptr ? owner : this;

    for (std::uint32_t i = 0; i < concurrency; ++i) {
        if (lanes_[i].state == LaneState::Free) {
            lanes_[i].owner = actual_owner;
            lanes_[i].state = LaneState::Active;
            lanes_[i].epoch += 1;
            lanes_[i].committed_frontier = 0;
            lanes_[i].history            = PleTokenHistory{};
            lane_physical_groups_[i].clear();
            previous_group_counts_[i] = 0;

            return LaneHandle{actual_owner, i, lanes_[i].epoch};
        }
    }
    throw std::runtime_error("FlashNextLaneLedger: no free lanes available");
}

void FlashNextLaneLedger::release_lane(LaneHandle handle) {
    validate_handle(handle, LaneState::Active);
    const std::uint32_t lane = handle.lane_index();

    for (const auto g : lane_physical_groups_[lane]) {
        if (g < physical_group_refcounts_.size()) {
            if (physical_group_refcounts_[g] > 0) {
                physical_group_refcounts_[g] -= 1;
            }
            if (physical_group_refcounts_[g] == 0) {
                free_physical_groups_.push_back(g);
            }
        }
    }
    lane_physical_groups_[lane].clear();
    previous_group_counts_[lane] = 0;

    lanes_[lane].owner = nullptr;
    lanes_[lane].state = LaneState::Free;
    lanes_[lane].epoch += 1;
    lanes_[lane].committed_frontier = 0;
    lanes_[lane].history            = PleTokenHistory{};
    lanes_[lane].history            = PleTokenHistory{};

    // Block table indexing: lane * logical_pages + page
    for (std::uint32_t p = 0; p < plan_.attention_logical_pages; ++p) {
        host_attention_table_[static_cast<std::size_t>(lane) * plan_.attention_logical_pages + p] =
            -1;
    }
    for (std::uint32_t p = 0; p < plan_.indexer_logical_pages; ++p) {
        host_indexer_table_[static_cast<std::size_t>(lane) * plan_.indexer_logical_pages + p] = -1;
    }
    block_tables_dirty_ = true;
}

std::int32_t FlashNextLaneLedger::committed_frontier(LaneHandle handle) const {
    const std::uint32_t concurrency = plan_.config.max_concurrency;
    if (handle.lane_index() >= concurrency) {
        throw std::invalid_argument("FlashNextLaneLedger: invalid lane index in handle");
    }
    const auto& lane = lanes_[handle.lane_index()];
    if (lane.owner != handle.owner()) {
        throw std::invalid_argument("FlashNextLaneLedger: cross-executor or invalid owner handle");
    }
    if (lane.epoch != handle.epoch()) {
        throw std::invalid_argument("FlashNextLaneLedger: stale lane handle epoch");
    }
    if (lane.state == LaneState::Free) {
        throw std::logic_error("FlashNextLaneLedger: lane is free");
    }
    return lane.committed_frontier;
}

const PleTokenHistory& FlashNextLaneLedger::lane_history(LaneHandle handle) const {
    const std::uint32_t concurrency = plan_.config.max_concurrency;
    if (handle.lane_index() >= concurrency) {
        throw std::invalid_argument("FlashNextLaneLedger: invalid lane index in handle");
    }
    const auto& lane = lanes_[handle.lane_index()];
    if (lane.owner != handle.owner()) {
        throw std::invalid_argument("FlashNextLaneLedger: cross-executor or invalid owner handle");
    }
    if (lane.epoch != handle.epoch()) {
        throw std::invalid_argument("FlashNextLaneLedger: stale lane handle epoch");
    }
    if (lane.state == LaneState::Free) {
        throw std::logic_error("FlashNextLaneLedger: lane is free");
    }
    return lane.history;
}

std::span<const std::uint32_t> FlashNextLaneLedger::lane_physical_groups(LaneHandle handle) const {
    const std::uint32_t concurrency = plan_.config.max_concurrency;
    if (handle.lane_index() >= concurrency) {
        throw std::invalid_argument("FlashNextLaneLedger: invalid lane index in handle");
    }
    const auto& lane = lanes_[handle.lane_index()];
    if (lane.owner != handle.owner()) {
        throw std::invalid_argument("FlashNextLaneLedger: cross-executor or invalid owner handle");
    }
    if (lane.epoch != handle.epoch()) {
        throw std::invalid_argument("FlashNextLaneLedger: stale lane handle epoch");
    }
    if (lane.state == LaneState::Free) {
        throw std::logic_error("FlashNextLaneLedger: lane is free");
    }
    return lane_physical_groups_[handle.lane_index()];
}

std::vector<std::uint32_t> FlashNextLaneLedger::take_lane_physical_groups(LaneHandle handle) {
    validate_handle(handle, LaneState::Active);
    const std::uint32_t lane = handle.lane_index();
    std::vector<std::uint32_t> groups = std::move(lane_physical_groups_[lane]);
    lane_physical_groups_[lane].clear();
    previous_group_counts_[lane] = 0;
    return groups;
}

void FlashNextLaneLedger::acquire_physical_groups(std::span<const std::uint32_t> groups) {
    for (const auto g : groups) {
        if (g < physical_group_refcounts_.size()) {
            physical_group_refcounts_[g] += 1;
        }
    }
}

void FlashNextLaneLedger::release_physical_groups(std::span<const std::uint32_t> groups) {
    for (const auto g : groups) {
        if (g < physical_group_refcounts_.size()) {
            if (physical_group_refcounts_[g] > 0) {
                physical_group_refcounts_[g] -= 1;
            }
            if (physical_group_refcounts_[g] == 0) {
                free_physical_groups_.push_back(g);
            }
        }
    }
}

void FlashNextLaneLedger::attach_physical_groups(
    LaneHandle handle, std::span<const std::uint32_t> groups,
    std::int32_t committed_frontier, const PleTokenHistory& history) {
    validate_handle(handle, LaneState::Active);
    const std::uint32_t lane = handle.lane_index();

    lane_physical_groups_[lane].assign(groups.begin(), groups.end());
    previous_group_counts_[lane] = groups.size();
    lanes_[lane].committed_frontier = committed_frontier;
    lanes_[lane].history = history;

    // Active lane acquires reference to attached groups
    for (const auto g : groups) {
        if (g < physical_group_refcounts_.size()) {
            physical_group_refcounts_[g] += 1;
        }
    }

    // Populate block tables for the attached groups
    for (std::size_t g_idx = 0; g_idx < groups.size(); ++g_idx) {
        const std::uint32_t g = groups[g_idx];
        // 4 attention pages per group
        for (std::uint32_t p = 0; p < 4; ++p) {
            const std::size_t log_page = g_idx * 4 + p;
            if (log_page < plan_.attention_logical_pages) {
                host_attention_table_[static_cast<std::size_t>(lane) * plan_.attention_logical_pages + log_page] =
                    static_cast<std::int32_t>(g * 4 + p);
            }
        }
        // 1 indexer page per group
        if (g_idx < plan_.indexer_logical_pages) {
            host_indexer_table_[static_cast<std::size_t>(lane) * plan_.indexer_logical_pages + g_idx] =
                static_cast<std::int32_t>(g);
        }
    }
    block_tables_dirty_ = true;
}

std::size_t FlashNextLaneLedger::active_lanes_count() const noexcept {
    std::size_t count = 0;
    for (const auto& lane : lanes_) {
        if (lane.state != LaneState::Free) { ++count; }
    }
    return count;
}

void FlashNextLaneLedger::validate_handle(LaneHandle handle, LaneState expected_state) const {
    const std::uint32_t concurrency = plan_.config.max_concurrency;
    if (handle.lane_index() >= concurrency) {
        throw std::invalid_argument("FlashNextLaneLedger: invalid lane index in handle");
    }
    const auto& lane = lanes_[handle.lane_index()];
    if (lane.owner != handle.owner()) {
        throw std::invalid_argument("FlashNextLaneLedger: cross-executor or invalid owner handle");
    }
    if (lane.epoch != handle.epoch()) {
        throw std::invalid_argument("FlashNextLaneLedger: stale lane handle epoch");
    }
    if (lane.state != expected_state) {
        throw std::invalid_argument("FlashNextLaneLedger: unexpected lane state");
    }
}

FlashNextLaneLedger::PreparedRound
FlashNextLaneLedger::begin_round(std::span<const LaneStepRequest> requests,
                                 const PleIndexMetadata& ple_meta) {
    if (has_pending_batch_) {
        throw std::logic_error("FlashNextLaneLedger: cannot begin round with pending batch");
    }
    const std::uint32_t concurrency = plan_.config.max_concurrency;
    const auto batch_size           = static_cast<std::uint32_t>(requests.size());

    if (batch_size < 1 || batch_size > concurrency) {
        throw std::invalid_argument(
            "FlashNextLaneLedger: requests batch size must be in [1, max_concurrency]");
    }
    if (current_transaction_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FlashNextLaneLedger: transaction id overflow");
    }

    // Phase 1: Dry-run validation (check all handles, frontiers, duplicates, PLE rows, and
    // capacity)
    std::uint32_t lane_mask         = 0;
    std::int32_t max_active_blocks  = 0;
    std::size_t total_groups_needed = 0;
    std::vector<std::size_t> required_groups_per_lane(batch_size);
    std::vector<std::array<std::int64_t, 16>> ple_indices_vec(batch_size);

    for (std::uint32_t i = 0; i < batch_size; ++i) {
        const auto& req = requests[i];
        validate_handle(req.handle, LaneState::Active);

        const std::uint32_t lane = req.handle.lane_index();
        if ((lane_mask & (1U << lane)) != 0) {
            throw std::invalid_argument("FlashNextLaneLedger: duplicate lane index in batch");
        }
        lane_mask |= (1U << lane);

        if (req.token_index != lanes_[lane].committed_frontier) {
            throw std::invalid_argument(
                "FlashNextLaneLedger: token_index must match committed frontier");
        }
        if (req.token_index < 0 ||
            static_cast<std::uint32_t>(req.token_index) >= plan_.config.max_context) {
            throw std::out_of_range("FlashNextLaneLedger: token_index exceeds max_context");
        }

        // Dry-run compute PLE indices before any mutation
        ple_indices_vec[i] = ple_indices(ple_meta, lanes_[lane].history, req.token_id);

        const std::size_t req_groups =
            static_cast<std::size_t>(req.token_index /
                                     static_cast<std::int32_t>(kMainPageGroupTokens)) +
            1U;
        required_groups_per_lane[i] = req_groups;

        const auto current_owned = lane_physical_groups_[lane].size();
        if (req_groups > current_owned) { total_groups_needed += (req_groups - current_owned); }
        max_active_blocks = std::max(max_active_blocks, (req.token_index + 1) / 4);
    }

    if (total_groups_needed > free_physical_groups_.size()) {
        throw std::runtime_error("FlashNextLaneLedger: physical page group capacity exhausted");
    }

    // Phase 2: Capacity assignment and shadow table updates
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        const std::uint32_t lane     = requests[i].handle.lane_index();
        const std::size_t req_groups = required_groups_per_lane[i];
        auto& owned_groups           = lane_physical_groups_[lane];
        previous_group_counts_[lane] = owned_groups.size();

        while (owned_groups.size() < req_groups) {
            const auto phys_group = free_physical_groups_.back();
            free_physical_groups_.pop_back();
            if (phys_group < physical_group_refcounts_.size()) {
                physical_group_refcounts_[phys_group] = 1;
            }

            const auto log_group = static_cast<std::uint32_t>(owned_groups.size());
            owned_groups.push_back(phys_group);

            for (std::uint32_t s = 0; s < 4U; ++s) {
                const auto log_att_page = log_group * 4U + s;
                if (log_att_page < plan_.attention_logical_pages) {
                    host_attention_table_[static_cast<std::size_t>(lane) *
                                              plan_.attention_logical_pages +
                                          log_att_page] =
                        static_cast<std::int32_t>(phys_group * 4U + s);
                }
            }
            if (log_group < plan_.indexer_logical_pages) {
                host_indexer_table_[static_cast<std::size_t>(lane) * plan_.indexer_logical_pages +
                                    log_group] = static_cast<std::int32_t>(phys_group);
            }
            block_tables_dirty_ = true;
        }
    }

    current_transaction_id_ += 1;
    pending_requests_.assign(requests.begin(), requests.end());
    pending_lane_indices_.resize(batch_size);
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        const auto lane          = requests[i].handle.lane_index();
        lanes_[lane].state       = LaneState::Pending;
        pending_lane_indices_[i] = lane;
    }
    has_pending_batch_ = true;

    return PreparedRound{current_transaction_id_, max_active_blocks, std::move(ple_indices_vec)};
}

FlashNextLaneLedger::PreparedRound
FlashNextLaneLedger::begin_prefill_chunk(LaneHandle handle,
                                         std::span<const std::int32_t> token_ids,
                                         std::int32_t first_token_index,
                                         const PleIndexMetadata& ple_meta) {
    (void)ple_meta;
    if (has_pending_batch_) {
        throw std::logic_error(
            "FlashNextLaneLedger: cannot begin prefill chunk with pending transaction");
    }
    validate_handle(handle, LaneState::Active);

    const auto num_tokens = static_cast<std::uint32_t>(token_ids.size());
    if (num_tokens == 0) {
        throw std::invalid_argument(
            "FlashNextLaneLedger: prefill chunk token_ids must not be empty");
    }
    const std::uint32_t lane = handle.lane_index();
    if (first_token_index != lanes_[lane].committed_frontier) {
        throw std::invalid_argument(
            "FlashNextLaneLedger: first_token_index must match committed frontier");
    }
    if (first_token_index < 0 ||
        static_cast<std::uint64_t>(first_token_index) + num_tokens > plan_.config.max_context) {
        throw std::out_of_range("FlashNextLaneLedger: prefill chunk exceeds max_context");
    }
    if (current_transaction_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FlashNextLaneLedger: transaction id overflow");
    }

    const std::int32_t last_token_index =
        first_token_index + static_cast<std::int32_t>(num_tokens) - 1;
    const std::size_t req_groups =
        static_cast<std::size_t>(last_token_index /
                                 static_cast<std::int32_t>(kMainPageGroupTokens)) +
        1U;
    auto& owned_groups           = lane_physical_groups_[lane];
    previous_group_counts_[lane] = owned_groups.size();

    if (req_groups > owned_groups.size()) {
        const std::size_t total_needed = req_groups - owned_groups.size();
        if (total_needed > free_physical_groups_.size()) {
            throw std::runtime_error("FlashNextLaneLedger: physical page group capacity exhausted");
        }
        while (owned_groups.size() < req_groups) {
            const auto phys_group = free_physical_groups_.back();
            free_physical_groups_.pop_back();
            if (phys_group < physical_group_refcounts_.size()) {
                physical_group_refcounts_[phys_group] = 1;
            }

            const auto log_group = static_cast<std::uint32_t>(owned_groups.size());
            owned_groups.push_back(phys_group);

            for (std::uint32_t s = 0; s < 4U; ++s) {
                const auto log_att_page = log_group * 4U + s;
                if (log_att_page < plan_.attention_logical_pages) {
                    host_attention_table_[static_cast<std::size_t>(lane) *
                                              plan_.attention_logical_pages +
                                          log_att_page] =
                        static_cast<std::int32_t>(phys_group * 4U + s);
                }
            }
            if (log_group < plan_.indexer_logical_pages) {
                host_indexer_table_[static_cast<std::size_t>(lane) * plan_.indexer_logical_pages +
                                    log_group] = static_cast<std::int32_t>(phys_group);
            }
            block_tables_dirty_ = true;
        }
    }

    const std::int32_t max_active_blocks = (last_token_index + 1) / 4;
    current_transaction_id_ += 1;
    has_pending_batch_                 = true;
    is_pending_prefill_chunk_          = true;
    pending_prefill_lane_              = lane;
    pending_prefill_tokens_.assign(token_ids.begin(), token_ids.end());
    pending_prefill_first_token_index_ = first_token_index;
    lanes_[lane].state                 = LaneState::Pending;

    return PreparedRound{current_transaction_id_, max_active_blocks, {}};
}

void FlashNextLaneLedger::rollback_prepared_prefill_chunk(std::uint64_t tx_id) {
    if (!has_pending_batch_ || !is_pending_prefill_chunk_ || tx_id != current_transaction_id_) {
        return;
    }
    const auto lane    = pending_prefill_lane_;
    lanes_[lane].state = LaneState::Active;
    const auto prev_count = previous_group_counts_[lane];
    auto& owned           = lane_physical_groups_[lane];
    while (owned.size() > prev_count) {
        const auto g = owned.back();
        const auto log_group = static_cast<std::uint32_t>(owned.size() - 1U);
        for (std::uint32_t s = 0; s < 4U; ++s) {
            const auto log_att = log_group * 4U + s;
            if (log_att < plan_.attention_logical_pages) {
                host_attention_table_[static_cast<std::size_t>(lane) *
                                          plan_.attention_logical_pages +
                                      log_att] = -1;
            }
        }
        if (log_group < plan_.indexer_logical_pages) {
            host_indexer_table_[static_cast<std::size_t>(lane) * plan_.indexer_logical_pages +
                                log_group] = -1;
        }
        if (g < physical_group_refcounts_.size()) {
            if (physical_group_refcounts_[g] > 0) {
                physical_group_refcounts_[g] -= 1;
            }
            if (physical_group_refcounts_[g] == 0) {
                free_physical_groups_.push_back(g);
            }
        }
        owned.pop_back();
        block_tables_dirty_ = true;
    }

    has_pending_batch_        = false;
    is_pending_prefill_chunk_ = false;
    pending_prefill_tokens_.clear();
}

void FlashNextLaneLedger::commit_prefill_chunk(std::uint64_t tx_id,
                                               FlashNextRuntimeAllocation& alloc,
                                               cudaStream_t stream) {
    (void)alloc;
    (void)stream;
    if (!has_pending_batch_ || !is_pending_prefill_chunk_ || tx_id != current_transaction_id_) {
        throw std::logic_error(
            "FlashNextLaneLedger: invalid or stale prefill chunk transaction commit");
    }
    const auto lane = pending_prefill_lane_;
    for (const auto tok : pending_prefill_tokens_) { lanes_[lane].history.commit(tok); }
    lanes_[lane].committed_frontier += static_cast<std::int32_t>(pending_prefill_tokens_.size());
    lanes_[lane].state = LaneState::Active;

    has_pending_batch_        = false;
    is_pending_prefill_chunk_ = false;
    pending_prefill_tokens_.clear();
}

void FlashNextLaneLedger::abort_prefill_chunk(std::uint64_t tx_id) noexcept {
    if (!has_pending_batch_ || !is_pending_prefill_chunk_ || tx_id != current_transaction_id_) {
        return;
    }
    lanes_[pending_prefill_lane_].state = LaneState::Active;
    has_pending_batch_                  = false;
    is_pending_prefill_chunk_           = false;
    pending_prefill_tokens_.clear();
}

void FlashNextLaneLedger::rollback_prepared_round(std::uint64_t tx_id) {
    if (!has_pending_batch_ || tx_id != current_transaction_id_) { return; }
    if (is_pending_prefill_chunk_) {
        rollback_prepared_prefill_chunk(tx_id);
        return;
    }

    for (const auto lane : pending_lane_indices_) {
        lanes_[lane].state    = LaneState::Active;
        const auto prev_count = previous_group_counts_[lane];
        auto& owned           = lane_physical_groups_[lane];
        while (owned.size() > prev_count) {
            const auto g = owned.back();
            const auto log_group = static_cast<std::uint32_t>(owned.size() - 1U);
            for (std::uint32_t s = 0; s < 4U; ++s) {
                const auto log_att = log_group * 4U + s;
                if (log_att < plan_.attention_logical_pages) {
                    host_attention_table_[static_cast<std::size_t>(lane) *
                                              plan_.attention_logical_pages +
                                          log_att] = -1;
                }
            }
            if (log_group < plan_.indexer_logical_pages) {
                host_indexer_table_[static_cast<std::size_t>(lane) * plan_.indexer_logical_pages +
                                    log_group] = -1;
            }
            if (g < physical_group_refcounts_.size()) {
                if (physical_group_refcounts_[g] > 0) {
                    physical_group_refcounts_[g] -= 1;
                }
                if (physical_group_refcounts_[g] == 0) {
                    free_physical_groups_.push_back(g);
                }
            }
            owned.pop_back();
            block_tables_dirty_ = true;
        }
    }

    has_pending_batch_ = false;
    pending_requests_.clear();
    pending_lane_indices_.clear();
}

void FlashNextLaneLedger::commit_round(std::uint64_t tx_id,
                                       std::span<const LaneCommitDecision> decisions,
                                       FlashNextRuntimeAllocation& alloc, cudaStream_t stream) {
    if (!has_pending_batch_ || tx_id != current_transaction_id_) {
        throw std::logic_error("FlashNextLaneLedger: invalid or stale transaction commit");
    }
    if (is_pending_prefill_chunk_) {
        if (decisions.size() != 1) {
            throw std::invalid_argument(
                "FlashNextLaneLedger: decisions count must be 1 for prefill chunk");
        }
        if (decisions[0].accept) {
            commit_prefill_chunk(tx_id, alloc, stream);
        } else {
            abort_prefill_chunk(tx_id);
        }
        return;
    }
    if (decisions.size() != pending_requests_.size()) {
        throw std::invalid_argument(
            "FlashNextLaneLedger: decisions count does not match pending batch size");
    }

    const auto batch_size = static_cast<std::uint32_t>(decisions.size());
    std::vector<std::uint32_t> accepted_lanes;
    accepted_lanes.reserve(batch_size);

    for (std::uint32_t i = 0; i < batch_size; ++i) {
        if (decisions[i].accept) { accepted_lanes.push_back(pending_lane_indices_[i]); }
    }

    // Commit slots before mutating ledger history and frontier
    if (!accepted_lanes.empty()) { alloc.commit_slots(accepted_lanes, stream); }

    for (std::uint32_t i = 0; i < batch_size; ++i) {
        const auto lane = pending_lane_indices_[i];
        const auto& req = pending_requests_[i];

        if (decisions[i].accept) {
            lanes_[lane].history.commit(req.token_id);
            lanes_[lane].committed_frontier += 1;
        }
        lanes_[lane].state = LaneState::Active;
    }

    has_pending_batch_ = false;
    pending_requests_.clear();
    pending_lane_indices_.clear();
}

void FlashNextLaneLedger::abort_round(std::uint64_t tx_id) noexcept {
    if (!has_pending_batch_ || tx_id != current_transaction_id_) { return; }
    if (is_pending_prefill_chunk_) {
        abort_prefill_chunk(tx_id);
        return;
    }
    for (const auto lane : pending_lane_indices_) { lanes_[lane].state = LaneState::Active; }
    has_pending_batch_ = false;
    pending_requests_.clear();
    pending_lane_indices_.clear();
}

void FlashNextLaneLedger::sync_tables_if_dirty(FlashNextRuntimeAllocation& alloc,
                                               cudaStream_t stream) {
    if (!block_tables_dirty_) { return; }
    // KV streaming (P2): once the residency map owns the device table, this host table (a pageable
    // std::vector, lane_ledger.h:183) must not be uploaded over it -- it would clobber every
    // resolve result, and a pageable H2D is illegal inside a capture anyway. One-way switch.
    if (kv_stream_binding().residency) { return; }
    auto& att_tensor = alloc.state_view().qsa_attention_caches[0].block_tables;
    auto& idx_tensor = alloc.state_view().qsa_indexer_caches[0].block_tables;

    CUDA_CHECK(cudaMemcpyAsync(att_tensor.data, host_attention_table_.data(),
                               host_attention_table_.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(idx_tensor.data, host_indexer_table_.data(),
                               host_indexer_table_.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    block_tables_dirty_ = false;
}

FlashNextLaneLedger::PreparedRound
FlashNextLaneLedger::begin_speculative_round(LaneHandle handle,
                                             std::int32_t anchor_token_id,
                                             std::span<const std::int32_t> draft_tokens,
                                             std::int32_t first_token_index,
                                             const PleIndexMetadata& ple_meta) {
    if (has_pending_batch_) {
        throw std::logic_error(
            "FlashNextLaneLedger: cannot begin speculative round with pending transaction");
    }
    validate_handle(handle, LaneState::Active);

    const auto num_tokens    = static_cast<std::uint32_t>(1U + draft_tokens.size());
    const std::uint32_t lane = handle.lane_index();
    if (first_token_index != lanes_[lane].committed_frontier) {
        throw std::invalid_argument(
            "FlashNextLaneLedger: first_token_index must match committed frontier");
    }
    if (first_token_index < 0 ||
        static_cast<std::uint64_t>(first_token_index) + num_tokens > plan_.config.max_context) {
        throw std::out_of_range("FlashNextLaneLedger: speculative round exceeds max_context");
    }
    if (current_transaction_id_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::overflow_error("FlashNextLaneLedger: transaction id overflow");
    }

    const std::int32_t last_token_index =
        first_token_index + static_cast<std::int32_t>(num_tokens) - 1;
    previous_group_counts_[lane] = lane_physical_groups_[lane].size();
    reserve_mtp_pages(handle, last_token_index);

    std::vector<std::array<std::int64_t, 16>> ple_indices_vec;
    ple_indices_vec.reserve(num_tokens);

    PleTokenHistory simulated_history = lanes_[lane].history;
    for (std::uint32_t t = 0; t < num_tokens; ++t) {
        const std::int32_t cur_token = (t == 0) ? anchor_token_id : draft_tokens[t - 1];
        std::array<std::int64_t, 16> token_indices = ple_indices(ple_meta, simulated_history, cur_token);
        ple_indices_vec.push_back(token_indices);
        simulated_history.commit(cur_token);
    }

    const std::int32_t max_active_blocks =
        std::min(static_cast<std::int32_t>(plan_.maximum_blocks), (last_token_index + 1) / 4);

    current_transaction_id_ += 1;
    pending_requests_.clear();
    pending_requests_.push_back(LaneStepRequest{.handle = handle, .token_id = anchor_token_id, .token_index = first_token_index});
    for (std::size_t d = 0; d < draft_tokens.size(); ++d) {
        pending_requests_.push_back(LaneStepRequest{
            .handle = handle,
            .token_id = draft_tokens[d],
            .token_index = first_token_index + static_cast<std::int32_t>(d + 1)
        });
    }
    pending_lane_indices_ = {lane};
    has_pending_batch_    = true;

    return PreparedRound{current_transaction_id_, max_active_blocks, std::move(ple_indices_vec)};
}

void FlashNextLaneLedger::reserve_mtp_pages(LaneHandle handle, std::int32_t last_token_index) {
    validate_handle(handle, LaneState::Active);
    if (last_token_index < 0 ||
        static_cast<std::uint32_t>(last_token_index) >= plan_.config.max_context) {
        throw std::out_of_range("MTP page reservation exceeds max_context");
    }
    const auto lane = handle.lane_index();
    const std::size_t req_groups =
        static_cast<std::size_t>(last_token_index /
                                 static_cast<std::int32_t>(kMainPageGroupTokens)) +
        1U;
    auto& owned_groups           = lane_physical_groups_[lane];

    if (req_groups > owned_groups.size()) {
        const std::size_t total_needed = req_groups - owned_groups.size();
        if (total_needed > free_physical_groups_.size()) {
            throw std::runtime_error("FlashNextLaneLedger: physical page group capacity exhausted");
        }
        while (owned_groups.size() < req_groups) {
            const auto phys_group = free_physical_groups_.back();
            free_physical_groups_.pop_back();
            if (phys_group < physical_group_refcounts_.size()) {
                physical_group_refcounts_[phys_group] = 1;
            }

            const auto log_group = static_cast<std::uint32_t>(owned_groups.size());
            owned_groups.push_back(phys_group);

            for (std::uint32_t s = 0; s < 4U; ++s) {
                const auto log_att_page = log_group * 4U + s;
                if (log_att_page < plan_.attention_logical_pages) {
                    host_attention_table_[static_cast<std::size_t>(lane) *
                                              plan_.attention_logical_pages +
                                          log_att_page] =
                        static_cast<std::int32_t>(phys_group * 4U + s);
                }
            }
            if (log_group < plan_.indexer_logical_pages) {
                host_indexer_table_[static_cast<std::size_t>(lane) * plan_.indexer_logical_pages +
                                    log_group] = static_cast<std::int32_t>(phys_group);
            }
            block_tables_dirty_ = true;
        }
    }

}

void FlashNextLaneLedger::commit_speculative_round(std::uint64_t tx_id,
                                                   std::span<const std::int32_t> accepted_tokens,
                                                   FlashNextRuntimeAllocation& alloc,
                                                   cudaStream_t stream) {
    if (!has_pending_batch_ || tx_id != current_transaction_id_) {
        throw std::logic_error("FlashNextLaneLedger: invalid or stale speculative transaction commit");
    }
    const auto lane = pending_lane_indices_[0];

    // P5-1. The verification forward has finished, so the accept/reject result exists: hand the
    // staging cache the accepted PREFIX LENGTH so experts used only by rejected draft columns are not
    // credited as uses. This is called for the EMPTY span too, and that is the point -- an empty span
    // is "every column was rejected", i.e. accepted_count == 0, which must credit NOTHING. Skipping
    // the call would instead leave the refs pending and a later round boundary would credit them in
    // full, silently turning the whole feature into a no-op. No-op with the switch off.
    flash_next_note_live_columns(static_cast<std::int32_t>(accepted_tokens.size()));

    if (!accepted_tokens.empty()) {
        alloc.advance_lane_slot(lane, static_cast<std::uint32_t>(accepted_tokens.size()), stream);
    }

    // PLE records the inputs evaluated by the verifier. The final correction/bonus
    // output is the next round's anchor and has not been consumed yet.
    for (std::size_t i = 0; i < accepted_tokens.size(); ++i) {
        lanes_[lane].history.commit(pending_requests_[i].token_id);
    }
    lanes_[lane].committed_frontier += static_cast<std::int32_t>(accepted_tokens.size());
    lanes_[lane].state = LaneState::Active;

    has_pending_batch_ = false;
    pending_requests_.clear();
    pending_lane_indices_.clear();
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
