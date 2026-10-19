#include "models/qwen3_8_flash_next/impl/runtime_state.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next::detail {

namespace {

inline std::size_t align_up_256(std::size_t bytes) noexcept { return (bytes + 255ULL) & ~255ULL; }

void validate_plan_match(const FlashNextRuntimePlan& actual) {
    const auto expected = finalize_flash_next_runtime_plan(actual.config, actual.main_page_groups);

    if (actual.resolved_tokens != expected.resolved_tokens ||
        actual.attention_physical_pages != expected.attention_physical_pages ||
        actual.indexer_physical_pages != expected.indexer_physical_pages ||
        actual.attention_logical_pages != expected.attention_logical_pages ||
        actual.indexer_logical_pages != expected.indexer_logical_pages ||
        actual.state_slots != expected.state_slots ||
        actual.continuation_slots != expected.continuation_slots ||
        actual.maximum_blocks != expected.maximum_blocks ||
        actual.attention_kv_bytes != expected.attention_kv_bytes ||
        actual.indexer_block_keys_bytes != expected.indexer_block_keys_bytes ||
        actual.block_tables_bytes != expected.block_tables_bytes ||
        actual.recurrent_state_bytes != expected.recurrent_state_bytes ||
        actual.round_tensors_bytes != expected.round_tensors_bytes ||
        actual.workspace_bytes != expected.workspace_bytes ||
        actual.cuda_graph_allowance_bytes != expected.cuda_graph_allowance_bytes ||
        actual.total_device_bytes != expected.total_device_bytes ||
        actual.capacity_curve.main_page_tokens != expected.capacity_curve.main_page_tokens ||
        actual.capacity_curve.minimum_main_page_groups !=
            expected.capacity_curve.minimum_main_page_groups ||
        actual.capacity_curve.maximum_main_page_groups !=
            expected.capacity_curve.maximum_main_page_groups ||
        actual.capacity_curve.bytes_per_additional_main_page_group !=
            expected.capacity_curve.bytes_per_additional_main_page_group ||
        actual.capacity_curve.minimum_device_reservation_bytes !=
            expected.capacity_curve.minimum_device_reservation_bytes) {
        throw std::invalid_argument(
            "FlashNextRuntimeAllocation: supplied plan does not match canonical layout");
    }
}

} // namespace

FlashNextRuntimeAllocation::FlashNextRuntimeAllocation(FlashNextRuntimePlan plan)
    : plan_(std::move(plan)), host_ingress_(sizeof(FlashNextDecodeIngress)),
      host_egress_(sizeof(FlashNextDecodeEgress)) {
    validate_plan_match(plan_);
    if (plan_.config.speculative_draft_tokens > 0) {
        host_mtp_draft_ingress_ = std::make_unique<PinnedHostBuffer>(sizeof(FlashNextMtpDraftIngress));
    }

    const std::uint32_t concurrency = plan_.config.max_concurrency;
    slots_per_lane_ =
        plan_.config.speculative_draft_tokens > 0 ? (plan_.config.speculative_draft_tokens + 1U) : 2U;
    host_ring_offsets_.assign(concurrency, 0U);
    host_active_slots_.resize(concurrency);
    host_standby_slots_.resize(concurrency);
    for (std::uint32_t b = 0; b < concurrency; ++b) {
        const std::uint32_t base_slot = b * slots_per_lane_;
        host_active_slots_[b]         = static_cast<std::int32_t>(base_slot);
        host_standby_slots_[b]        = static_cast<std::int32_t>(base_slot + (1U % slots_per_lane_));
    }

    const std::size_t persistent_bytes =
        plan_.total_device_bytes - plan_.workspace_bytes - plan_.cuda_graph_allowance_bytes;
    storage_   = std::make_unique<DeviceBuffer>(persistent_bytes);
    workspace_ = std::make_unique<WorkspaceArena>(plan_.workspace_bytes);

    materialize_views();
    validate_flash_next_decode_state(state_view_, plan_.state_slots);
}

void FlashNextRuntimeAllocation::materialize_views() {
    auto* cur                       = static_cast<std::byte*>(storage_->p);
    const std::uint32_t concurrency = plan_.config.max_concurrency;

    // 1. Attention KV pages (12 layers, Key & Value)
    // Key: [256, 64, 2, attention_physical_pages] BF16 or FP8
    // Value: [256, 64, 2, attention_physical_pages] BF16 or FP8
    const std::size_t kv_element_bytes =
        (plan_.config.kv_cache == KvCacheStorage::Fp8E4M3Row256) ? 1ULL : sizeof(std::uint16_t);
    const DType kv_dtype =
        (plan_.config.kv_cache == KvCacheStorage::Fp8E4M3Row256) ? DType::FP8_E4M3FN : DType::BF16;
    const std::size_t att_kv_bytes = align_up_256(
        256ULL * 64ULL * 2ULL * plan_.attention_physical_pages * kv_element_bytes);
    for (std::size_t i = 0; i < plan_.qsa_cache_layers(); ++i) {
        state_view_.qsa_attention_caches[i].key_pages =
            Tensor(cur, kv_dtype,
                   {256, 64, 2, static_cast<std::int32_t>(plan_.attention_physical_pages)});
        cur += att_kv_bytes;
        state_view_.qsa_attention_caches[i].value_pages =
            Tensor(cur, kv_dtype,
                   {256, 64, 2, static_cast<std::int32_t>(plan_.attention_physical_pages)});
        cur += att_kv_bytes;
    }

    // 2. Indexer compressed block keys (12 layers)
    // [128, 64, indexer_physical_pages] BF16
    const std::size_t idx_keys_bytes =
        align_up_256(128ULL * 64ULL * plan_.indexer_physical_pages * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < plan_.qsa_cache_layers(); ++i) {
        state_view_.qsa_indexer_caches[i].block_keys = Tensor(
            cur, DType::BF16, {128, 64, static_cast<std::int32_t>(plan_.indexer_physical_pages)});
        cur += idx_keys_bytes;
    }

    // 3. Shared Single Block tables (1 attention block table + 1 indexer block table)
    const std::size_t att_table_bytes =
        align_up_256(static_cast<std::size_t>(plan_.attention_logical_pages) * concurrency *
                     sizeof(std::int32_t));
    Tensor shared_att_table(cur, DType::I32,
                            {static_cast<std::int32_t>(plan_.attention_logical_pages),
                             static_cast<std::int32_t>(concurrency)});
    cur += att_table_bytes;

    const std::size_t idx_table_bytes = align_up_256(
        static_cast<std::size_t>(plan_.indexer_logical_pages) * concurrency * sizeof(std::int32_t));
    Tensor shared_idx_table(cur, DType::I32,
                            {static_cast<std::int32_t>(plan_.indexer_logical_pages),
                             static_cast<std::int32_t>(concurrency)});
    cur += idx_table_bytes;

    for (std::size_t i = 0; i < plan_.qsa_cache_layers(); ++i) {
        state_view_.qsa_attention_caches[i].block_tables = shared_att_table;
        state_view_.qsa_indexer_caches[i].block_tables   = shared_idx_table;
    }

    // 4. Recurrent states
    // 36 GDN conv: [10240, 3, state_slots] BF16
    const std::size_t gdn_conv_bytes =
        align_up_256(10'240ULL * 3ULL * plan_.state_slots * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        state_view_.gdn_convolution_states[i] =
            Tensor(cur, DType::BF16, {10'240, 3, static_cast<std::int32_t>(plan_.state_slots)});
        cur += gdn_conv_bytes;
    }

    // 36 GDN SSM: [128, 128, 48, state_slots] FP32 or BF16
    const DType ssm_dtype = (plan_.config.gdn_state_storage == GdnStateStorage::BF16)
                                ? DType::BF16
                                : DType::FP32;
    const std::size_t ssm_elem_size = dtype_size(ssm_dtype);
    const std::size_t gdn_ssm_bytes =
        align_up_256(128ULL * 128ULL * 48ULL * plan_.state_slots * ssm_elem_size);
    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        state_view_.gdn_ssm_states[i] =
            Tensor(cur, ssm_dtype, {128, 128, 48, static_cast<std::int32_t>(plan_.state_slots)});
        cur += gdn_ssm_bytes;
    }

    // 1 PLE conv: [10240, 9, state_slots] BF16
    const std::size_t ple_conv_bytes =
        align_up_256(10'240ULL * 9ULL * plan_.state_slots * sizeof(std::uint16_t));
    state_view_.ple_convolution_states =
        Tensor(cur, DType::BF16, {10'240, 9, static_cast<std::int32_t>(plan_.state_slots)});
    cur += ple_conv_bytes;

    // 12 QSA raw keys: [128, 4, state_slots] BF16
    const std::size_t raw_keys_bytes =
        align_up_256(128ULL * 4ULL * plan_.state_slots * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < plan_.qsa_cache_layers(); ++i) {
        state_view_.qsa_indexer_caches[i].raw_keys =
            Tensor(cur, DType::BF16, {128, 4, static_cast<std::int32_t>(plan_.state_slots)});
        cur += raw_keys_bytes;
    }

    // 12 QSA raw positions: [3, 4, state_slots] I32
    const std::size_t raw_pos_bytes =
        align_up_256(3ULL * 4ULL * plan_.state_slots * sizeof(std::int32_t));
    for (std::size_t i = 0; i < plan_.qsa_cache_layers(); ++i) {
        state_view_.qsa_indexer_caches[i].raw_positions =
            Tensor(cur, DType::I32, {3, 4, static_cast<std::int32_t>(plan_.state_slots)});
        cur += raw_pos_bytes;
    }

    if (plan_.config.speculative_draft_tokens > 0) {
        state_view_.mtp_backbone_hidden =
            Tensor(cur, DType::BF16, {10'240, static_cast<std::int32_t>(plan_.state_slots)});
        cur += align_up_256(10'240ULL * plan_.state_slots * sizeof(std::uint16_t));
        state_view_.mtp_backbone_positions =
            Tensor(cur, DType::I32, {3, static_cast<std::int32_t>(plan_.state_slots)});
        cur += align_up_256(3ULL * plan_.state_slots * sizeof(std::int32_t));
    }

    // 5. Round buffers (device ingress/egress structs, gathered PLE, final hidden, logits)
    device_ingress_ = cur;
    cur += align_up_256(sizeof(FlashNextDecodeIngress));

    device_egress_ = cur;
    cur += align_up_256(sizeof(FlashNextDecodeEgress));

    const std::uint32_t round_batch_tokens =
        std::max(concurrency,
                 plan_.config.speculative_draft_tokens > 0 ? (plan_.config.speculative_draft_tokens + 1U) : 1U);

    round_tensors_.token_ids =
        Tensor(static_cast<std::byte*>(device_ingress_) + offsetof(FlashNextDecodeIngress, token_ids),
               DType::I32, {static_cast<std::int32_t>(round_batch_tokens)});

    round_tensors_.token_indices =
        Tensor(static_cast<std::byte*>(device_ingress_) +
                   offsetof(FlashNextDecodeIngress, token_indices),
               DType::I32, {static_cast<std::int32_t>(round_batch_tokens)});

    round_tensors_.mrope_positions =
        Tensor(static_cast<std::byte*>(device_ingress_) +
                   offsetof(FlashNextDecodeIngress, mrope_positions),
               DType::I32, {static_cast<std::int32_t>(round_batch_tokens), 3});

    round_tensors_.table_rows =
        Tensor(static_cast<std::byte*>(device_ingress_) + offsetof(FlashNextDecodeIngress, table_rows),
               DType::I32, {static_cast<std::int32_t>(round_batch_tokens)});

    round_tensors_.source_slots =
        Tensor(static_cast<std::byte*>(device_ingress_) +
                   offsetof(FlashNextDecodeIngress, source_slots),
               DType::I32, {static_cast<std::int32_t>(round_batch_tokens)});

    round_tensors_.destination_slots =
        Tensor(static_cast<std::byte*>(device_ingress_) +
                   offsetof(FlashNextDecodeIngress, destination_slots),
               DType::I32, {static_cast<std::int32_t>(round_batch_tokens)});

    round_tensors_.sampled_tokens =
        Tensor(static_cast<std::byte*>(device_egress_) +
                   offsetof(FlashNextDecodeEgress, sampled_tokens),
               DType::I32, {static_cast<std::int32_t>(round_batch_tokens)});

    round_tensors_.gathered_ple_embedding =
        Tensor(cur, DType::BF16, {2'560, static_cast<std::int32_t>(round_batch_tokens)});
    cur += align_up_256(2'560ULL * round_batch_tokens * sizeof(std::uint16_t));

    round_tensors_.final_hidden =
        Tensor(cur, DType::BF16, {2'560, static_cast<std::int32_t>(round_batch_tokens)});
    cur += align_up_256(2'560ULL * round_batch_tokens * sizeof(std::uint16_t));

    round_tensors_.hyper_hidden =
        Tensor(cur, DType::BF16, {10'240, static_cast<std::int32_t>(round_batch_tokens)});
    cur += align_up_256(10'240ULL * round_batch_tokens * sizeof(std::uint16_t));

    round_tensors_.logits =
        Tensor(cur, DType::BF16, {248'320, static_cast<std::int32_t>(round_batch_tokens)});
    cur += align_up_256(248'320ULL * round_batch_tokens * sizeof(std::uint16_t));

    if (plan_.config.speculative_draft_tokens > 0) {
        const auto rows = plan_.config.proposal_head == ProposalHead::Optimized
                              ? plan_.config.draft_head_rows : 248'320U;
        round_tensors_.mtp_embedding = Tensor(cur, DType::BF16, {2'560, 1});
        cur += align_up_256(2'560ULL * sizeof(std::uint16_t));
        round_tensors_.mtp_carried_hidden = Tensor(cur, DType::BF16, {10'240, 1});
        cur += align_up_256(10'240ULL * sizeof(std::uint16_t));
        round_tensors_.mtp_logits = Tensor(cur, DType::BF16, {static_cast<std::int32_t>(rows), 1});
        cur += align_up_256(rows * sizeof(std::uint16_t));
        round_tensors_.mtp_token = Tensor(cur, DType::I32, {1});
        cur += 256;
        device_mtp_draft_ingress_ = reinterpret_cast<FlashNextMtpDraftIngress*>(cur);
        cur += align_up_256(sizeof(FlashNextMtpDraftIngress));
    }

    const auto used_bytes = static_cast<std::size_t>(cur - static_cast<std::byte*>(storage_->p));
    if (used_bytes > storage_->bytes) {
        throw std::overflow_error("Flash-Next runtime layout exceeded allocated device buffer");
    }
}

void FlashNextRuntimeAllocation::initialize(cudaStream_t stream) {
    // Deterministically zero all persistent / recurrent state device storage
    CUDA_CHECK(cudaMemsetAsync(storage_->p, 0, storage_->bytes, stream));
    sync_slots_to_device(stream);
}

void FlashNextRuntimeAllocation::commit_row_slot(std::uint32_t row_index, cudaStream_t stream) {
    advance_lane_slot(row_index, 1U, stream);
}

void FlashNextRuntimeAllocation::commit_slots(std::span<const std::uint32_t> accepted_lanes,
                                              cudaStream_t stream) {
    // Prevalidate: range and uniqueness before any mutation
    std::uint32_t seen_mask = 0;
    for (const auto lane : accepted_lanes) {
        if (lane >= plan_.config.max_concurrency) {
            throw std::out_of_range("commit_slots: lane index exceeds max_concurrency");
        }
        if ((seen_mask & (1U << lane)) != 0) {
            throw std::invalid_argument("commit_slots: duplicate lane index");
        }
        seen_mask |= (1U << lane);
    }
    // All validated 鈥?advance each lane.
    //
    // This is the decode-round commit (FlashNextLaneLedger::commit_round). The device copies of
    // the slot selectors are deliberately NOT refreshed here: the only readers of
    // round_tensors().source_slots/destination_slots are the two decode bodies in
    // text_executor.cpp (execute_round_body and execute_speculative_verify_round), and each is
    // preceded on the same stream by the full pinned-ingress H2D memcpy that re-derives both
    // arrays from current_source_slot()/current_destination_slot(). The two cudaMemcpyAsync this
    // used to issue per accepted lane were therefore dead 鈥?and, sourced from pageable
    // std::vector storage, host-synchronous 鈥?landing in the middle of the inter-round idle
    // window. That window measures 0.99 ms/token of GPU idle at ~1.2 gaps >100 us per token
    // where the round structure only accounts for 1.0; these copies are the extra 0.2.
    //
    // Prefill (commit_row_slot) and MTP (commit_speculative_round) keep the upload so their
    // observable behaviour is unchanged, including test_runtime_foundation, which reads the
    // device tensors back after commit_row_slot.
    for (const auto lane : accepted_lanes) {
        advance_lane_slot(lane, 1U, stream, /*upload_slots=*/false);
    }
}

void FlashNextRuntimeAllocation::advance_lane_slot(std::uint32_t lane_index,
                                                  std::uint32_t step_count,
                                                  cudaStream_t stream,
                                                  bool upload_slots) {
    if (lane_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("advance_lane_slot: lane_index exceeds max_concurrency");
    }
    host_ring_offsets_[lane_index] =
        (host_ring_offsets_[lane_index] + step_count) % slots_per_lane_;
    const std::uint32_t base_slot = lane_index * slots_per_lane_;
    host_active_slots_[lane_index] =
        static_cast<std::int32_t>(base_slot + host_ring_offsets_[lane_index]);
    host_standby_slots_[lane_index] =
        static_cast<std::int32_t>(base_slot + ((host_ring_offsets_[lane_index] + 1U) % slots_per_lane_));
    if (upload_slots) { sync_slots_to_device(stream); }
}

void FlashNextRuntimeAllocation::restore_lane_slots(std::uint32_t lane_index,
                                                    std::int32_t active_slot,
                                                    std::int32_t standby_slot,
                                                    cudaStream_t stream) {
    if (lane_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("restore_lane_slots: lane_index exceeds max_concurrency");
    }
    host_active_slots_[lane_index]  = active_slot;
    host_standby_slots_[lane_index] = standby_slot;
    const std::uint32_t base_slot   = lane_index * slots_per_lane_;
    if (active_slot >= static_cast<std::int32_t>(base_slot) &&
        active_slot < static_cast<std::int32_t>(base_slot + slots_per_lane_)) {
        host_ring_offsets_[lane_index] = static_cast<std::uint32_t>(active_slot - base_slot);
    }
    sync_slots_to_device(stream);
}

void FlashNextRuntimeAllocation::zero_slot(std::uint32_t slot_index, cudaStream_t stream) {
    if (slot_index >= plan_.state_slots) {
        throw std::out_of_range("slot_index exceeds state_slots");
    }
    // Zero out recurrent state slices for this slot
    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        auto* conv_p = static_cast<std::byte*>(state_view_.gdn_convolution_states[i].data) +
                       slot_index * 10'240ULL * 3ULL * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemsetAsync(conv_p, 0, 10'240ULL * 3ULL * sizeof(std::uint16_t), stream));

        const std::size_t ssm_elem_size = dtype_size(state_view_.gdn_ssm_states[i].dtype);
        const std::size_t ssm_slot_bytes = 128ULL * 128ULL * 48ULL * ssm_elem_size;
        auto* ssm_p = static_cast<std::byte*>(state_view_.gdn_ssm_states[i].data) +
                      slot_index * ssm_slot_bytes;
        CUDA_CHECK(cudaMemsetAsync(ssm_p, 0, ssm_slot_bytes, stream));
    }

    auto* ple_p = static_cast<std::byte*>(state_view_.ple_convolution_states.data) +
                  slot_index * 10'240ULL * 9ULL * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemsetAsync(ple_p, 0, 10'240ULL * 9ULL * sizeof(std::uint16_t), stream));

    if (state_view_.mtp_backbone_hidden.data != nullptr) {
        Tensor hidden = state_view_.mtp_backbone_hidden.slice(1, slot_index, 1);
        CUDA_CHECK(cudaMemsetAsync(hidden.data, 0, 10'240ULL * sizeof(std::uint16_t), stream));
        Tensor positions = state_view_.mtp_backbone_positions.slice(1, slot_index, 1);
        CUDA_CHECK(cudaMemsetAsync(positions.data, 0, 3 * sizeof(std::int32_t), stream));
    }

    for (std::size_t i = 0; i < plan_.qsa_cache_layers(); ++i) {
        auto* key_p = static_cast<std::byte*>(state_view_.qsa_indexer_caches[i].raw_keys.data) +
                      slot_index * 128ULL * 4ULL * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemsetAsync(key_p, 0, 128ULL * 4ULL * sizeof(std::uint16_t), stream));

        auto* pos_p =
            static_cast<std::byte*>(state_view_.qsa_indexer_caches[i].raw_positions.data) +
            slot_index * 3ULL * 4ULL * sizeof(std::int32_t);
        CUDA_CHECK(cudaMemsetAsync(pos_p, 0, 3ULL * 4ULL * sizeof(std::int32_t), stream));
    }
}

void FlashNextRuntimeAllocation::zero_lane_slots(std::uint32_t lane_index, cudaStream_t stream) {
    if (lane_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("lane_index exceeds max_concurrency");
    }
    const std::uint32_t base_slot = lane_index * slots_per_lane_;
    for (std::uint32_t s = 0; s < slots_per_lane_; ++s) {
        zero_slot(base_slot + s, stream);
    }
    host_ring_offsets_[lane_index]  = 0U;
    host_active_slots_[lane_index]  = static_cast<std::int32_t>(base_slot);
    host_standby_slots_[lane_index] = static_cast<std::int32_t>(base_slot + (1U % slots_per_lane_));
}

void FlashNextRuntimeAllocation::copy_state_slot(std::uint32_t src_slot, std::uint32_t dst_slot,
                                                 cudaStream_t stream) {
    if (src_slot >= plan_.state_slots || dst_slot >= plan_.state_slots) {
        throw std::out_of_range("copy_state_slot: slot index exceeds state_slots");
    }
    if (src_slot == dst_slot) { return; }

    if (state_view_.mtp_backbone_hidden.data != nullptr) {
        Tensor source = state_view_.mtp_backbone_hidden.slice(1, src_slot, 1);
        Tensor destination = state_view_.mtp_backbone_hidden.slice(1, dst_slot, 1);
        CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data,
                                   10'240ULL * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
                                   stream));
        Tensor source_positions = state_view_.mtp_backbone_positions.slice(1, src_slot, 1);
        Tensor destination_positions = state_view_.mtp_backbone_positions.slice(1, dst_slot, 1);
        CUDA_CHECK(cudaMemcpyAsync(destination_positions.data, source_positions.data,
                                   3 * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream));
    }

    // 1. GDN Conv & SSM states
    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        const auto* src_conv_p = static_cast<const std::byte*>(state_view_.gdn_convolution_states[i].data) +
                                 src_slot * 10'240ULL * 3ULL * sizeof(std::uint16_t);
        auto* dst_conv_p = static_cast<std::byte*>(state_view_.gdn_convolution_states[i].data) +
                           dst_slot * 10'240ULL * 3ULL * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpyAsync(dst_conv_p, src_conv_p, 10'240ULL * 3ULL * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice, stream));

        const std::size_t ssm_elem_size = dtype_size(state_view_.gdn_ssm_states[i].dtype);
        const std::size_t ssm_slot_bytes = 128ULL * 128ULL * 48ULL * ssm_elem_size;
        const auto* src_ssm_p = static_cast<const std::byte*>(state_view_.gdn_ssm_states[i].data) +
                                src_slot * ssm_slot_bytes;
        auto* dst_ssm_p = static_cast<std::byte*>(state_view_.gdn_ssm_states[i].data) +
                          dst_slot * ssm_slot_bytes;
        CUDA_CHECK(cudaMemcpyAsync(dst_ssm_p, src_ssm_p, ssm_slot_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // 2. PLE Conv states
    const auto* src_ple_p = static_cast<const std::byte*>(state_view_.ple_convolution_states.data) +
                            src_slot * 10'240ULL * 9ULL * sizeof(std::uint16_t);
    auto* dst_ple_p = static_cast<std::byte*>(state_view_.ple_convolution_states.data) +
                      dst_slot * 10'240ULL * 9ULL * sizeof(std::uint16_t);
    CUDA_CHECK(cudaMemcpyAsync(dst_ple_p, src_ple_p, 10'240ULL * 9ULL * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToDevice, stream));

    // 3. QSA Raw Keys & Positions
    for (std::size_t i = 0; i < plan_.qsa_cache_layers(); ++i) {
        const auto* src_key_p = static_cast<const std::byte*>(state_view_.qsa_indexer_caches[i].raw_keys.data) +
                                src_slot * 128ULL * 4ULL * sizeof(std::uint16_t);
        auto* dst_key_p = static_cast<std::byte*>(state_view_.qsa_indexer_caches[i].raw_keys.data) +
                          dst_slot * 128ULL * 4ULL * sizeof(std::uint16_t);
        CUDA_CHECK(cudaMemcpyAsync(dst_key_p, src_key_p, 128ULL * 4ULL * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice, stream));

        const auto* src_pos_p =
            static_cast<const std::byte*>(state_view_.qsa_indexer_caches[i].raw_positions.data) +
            src_slot * 3ULL * 4ULL * sizeof(std::int32_t);
        auto* dst_pos_p =
            static_cast<std::byte*>(state_view_.qsa_indexer_caches[i].raw_positions.data) +
            dst_slot * 3ULL * 4ULL * sizeof(std::int32_t);
        CUDA_CHECK(cudaMemcpyAsync(dst_pos_p, src_pos_p, 3ULL * 4ULL * sizeof(std::int32_t),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

void FlashNextRuntimeAllocation::sync_slots_to_device(cudaStream_t stream) {
    // Seeding/repair path only (initialize, restore_lane_slots, prefill and MTP commits). It is
    // not the authoritative writer of the device slot selectors: host_active_slots_ is indexed by
    // lane while round_tensors().source_slots is indexed by round row, and every decode body is
    // preceded by the pinned-ingress H2D that fills the row-indexed arrays from
    // current_source_slot()/current_destination_slot(). Both copies are pageable, hence
    // host-synchronous 鈥?keep them off the per-token path (see commit_slots).
    const std::size_t bytes = plan_.config.max_concurrency * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(round_tensors_.source_slots.data, host_active_slots_.data(), bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(round_tensors_.destination_slots.data, host_standby_slots_.data(),
                               bytes, cudaMemcpyHostToDevice, stream));
}

std::int32_t FlashNextRuntimeAllocation::current_source_slot(std::uint32_t row_index) const {
    if (row_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("row_index exceeds max_concurrency");
    }
    return host_active_slots_[row_index];
}

std::int32_t FlashNextRuntimeAllocation::current_destination_slot(std::uint32_t row_index) const {
    if (row_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("row_index exceeds max_concurrency");
    }
    return host_standby_slots_[row_index];
}

std::int32_t FlashNextRuntimeAllocation::lane_ring_slot(std::uint32_t lane_index,
                                                        std::uint32_t step_offset) const {
    if (lane_index >= plan_.config.max_concurrency) {
        throw std::out_of_range("lane_ring_slot: lane_index exceeds max_concurrency");
    }
    const std::uint32_t base_slot = lane_index * slots_per_lane_;
    const std::uint32_t offset    = (host_ring_offsets_[lane_index] + step_offset) % slots_per_lane_;
    return static_cast<std::int32_t>(base_slot + offset);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
