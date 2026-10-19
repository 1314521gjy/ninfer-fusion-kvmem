#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "models/qwen3_8_flash_next/impl/runtime_plan.h"
#include "models/qwen3_8_flash_next/impl/text_decode_state.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

struct FlashNextRoundTensors {
    Tensor token_ids;              // I32 [round_batch_tokens]
    Tensor token_indices;          // I32 [round_batch_tokens]
    Tensor mrope_positions;        // I32 [round_batch_tokens, 3]
    Tensor table_rows;             // I32 [round_batch_tokens]
    Tensor source_slots;           // I32 [round_batch_tokens]
    Tensor destination_slots;      // I32 [round_batch_tokens]
    Tensor sampled_tokens;         // I32 [round_batch_tokens]
    Tensor gathered_ple_embedding; // BF16 [2560, round_batch_tokens]
    Tensor final_hidden;           // BF16 [2560, round_batch_tokens]
    Tensor hyper_hidden;           // BF16 [10240, round_batch_tokens]
    Tensor logits;                 // BF16 [248320, round_batch_tokens]
    Tensor mtp_embedding;          // BF16 [2560, 1]
    Tensor mtp_carried_hidden;     // BF16 [10240, 1]
    Tensor mtp_logits;             // BF16 [draft_head_rows, 1]
    Tensor mtp_token;              // I32 [1]
};

class FlashNextRuntimeAllocation {
public:
    explicit FlashNextRuntimeAllocation(FlashNextRuntimePlan plan);
    ~FlashNextRuntimeAllocation() = default;

    FlashNextRuntimeAllocation(const FlashNextRuntimeAllocation&)                = delete;
    FlashNextRuntimeAllocation& operator=(const FlashNextRuntimeAllocation&)     = delete;
    FlashNextRuntimeAllocation(FlashNextRuntimeAllocation&&) noexcept            = default;
    FlashNextRuntimeAllocation& operator=(FlashNextRuntimeAllocation&&) noexcept = default;

    [[nodiscard]] const FlashNextRuntimePlan& plan() const noexcept { return plan_; }

    [[nodiscard]] const FlashNextDecodeStateView& state_view() const noexcept {
        return state_view_;
    }

    [[nodiscard]] FlashNextDecodeStateView& state_view() noexcept { return state_view_; }

    [[nodiscard]] const FlashNextRoundTensors& round_tensors() const noexcept {
        return round_tensors_;
    }

    [[nodiscard]] FlashNextRoundTensors& round_tensors() noexcept { return round_tensors_; }

    [[nodiscard]] WorkspaceArena& workspace() noexcept { return *workspace_; }

    [[nodiscard]] FlashNextDecodeIngress* host_ingress() noexcept {
        return static_cast<FlashNextDecodeIngress*>(host_ingress_.data());
    }
    [[nodiscard]] const FlashNextDecodeIngress* host_ingress() const noexcept {
        return static_cast<const FlashNextDecodeIngress*>(host_ingress_.data());
    }

    [[nodiscard]] FlashNextDecodeEgress* host_egress() noexcept {
        return static_cast<FlashNextDecodeEgress*>(host_egress_.data());
    }
    [[nodiscard]] const FlashNextDecodeEgress* host_egress() const noexcept {
        return static_cast<const FlashNextDecodeEgress*>(host_egress_.data());
    }

    [[nodiscard]] void* device_ingress_ptr() noexcept { return device_ingress_; }
    [[nodiscard]] const void* device_ingress_ptr() const noexcept { return device_ingress_; }

    [[nodiscard]] void* device_egress_ptr() noexcept { return device_egress_; }
    [[nodiscard]] const void* device_egress_ptr() const noexcept { return device_egress_; }

    [[nodiscard]] FlashNextMtpDraftIngress* host_mtp_draft_ingress() noexcept {
        return host_mtp_draft_ingress_
            ? static_cast<FlashNextMtpDraftIngress*>(host_mtp_draft_ingress_->data()) : nullptr;
    }
    [[nodiscard]] FlashNextMtpDraftIngress* device_mtp_draft_ingress() noexcept {
        return device_mtp_draft_ingress_;
    }

    [[nodiscard]] const ops::SamplingConfig* device_sampling_configs() const noexcept {
        return reinterpret_cast<const ops::SamplingConfig*>(
            static_cast<const std::byte*>(device_ingress_) + offsetof(FlashNextDecodeIngress, sampling));
    }

    // Initialize device slot tensors and state before the first decode round
    void initialize(cudaStream_t stream);

    // Transactional state slot mechanics:
    // Swaps active (source) and standby (destination) slot for row b in [0, max_concurrency).
    void commit_row_slot(std::uint32_t row_index, cudaStream_t stream);
    void commit_slots(std::span<const std::uint32_t> accepted_lanes, cudaStream_t stream);
    // `upload_slots == false` advances the host-side selectors only. Use it wherever the next
    // consumer of round_tensors().source_slots/destination_slots is a decode body that is
    // preceded, on the same stream, by the full pinned-ingress H2D copy (which re-derives both
    // arrays from current_source_slot()/current_destination_slot()) 鈥?the upload there is dead
    // work in the inter-round GPU-idle window. See commit_slots() for the measurement.
    void advance_lane_slot(std::uint32_t lane_index, std::uint32_t step_count, cudaStream_t stream,
                           bool upload_slots = true);
    void restore_lane_slots(std::uint32_t lane_index, std::int32_t active_slot,
                            std::int32_t standby_slot, cudaStream_t stream);
    void sync_slots_to_device(cudaStream_t stream);

    // Recurrent state zeroing and copying for assigned lane and cache slots
    void zero_slot(std::uint32_t slot_index, cudaStream_t stream);
    void zero_lane_slots(std::uint32_t lane_index, cudaStream_t stream);
    void copy_state_slot(std::uint32_t src_slot, std::uint32_t dst_slot, cudaStream_t stream);

    [[nodiscard]] std::int32_t current_source_slot(std::uint32_t row_index) const;
    [[nodiscard]] std::int32_t current_destination_slot(std::uint32_t row_index) const;
    [[nodiscard]] std::int32_t lane_ring_slot(std::uint32_t lane_index,
                                              std::uint32_t step_offset) const;

    [[nodiscard]] void* persistent_base() noexcept { return storage_->p; }
    [[nodiscard]] const void* persistent_base() const noexcept { return storage_->p; }
    [[nodiscard]] std::size_t persistent_bytes() const noexcept {
        return plan_.total_device_bytes - plan_.workspace_bytes - plan_.cuda_graph_allowance_bytes;
    }

private:
    FlashNextRuntimePlan plan_;
    std::unique_ptr<DeviceBuffer> storage_;
    std::unique_ptr<WorkspaceArena> workspace_;

    PinnedHostBuffer host_ingress_;
    PinnedHostBuffer host_egress_;
    std::unique_ptr<PinnedHostBuffer> host_mtp_draft_ingress_;
    void* device_ingress_ = nullptr;
    void* device_egress_  = nullptr;
    FlashNextMtpDraftIngress* device_mtp_draft_ingress_ = nullptr;

    FlashNextDecodeStateView state_view_{};
    FlashNextRoundTensors round_tensors_{};

    // Slot pair for each concurrency row: active (source) and standby (destination)
    std::uint32_t slots_per_lane_ = 2;
    std::vector<std::uint32_t> host_ring_offsets_;
    std::vector<std::int32_t> host_active_slots_;
    std::vector<std::int32_t> host_standby_slots_;

    void materialize_views();
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
