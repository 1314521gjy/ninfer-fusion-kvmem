#include "models/qwen3_8_flash_next/impl/vision_execute.h"
#include "models/qwen3_8_flash_next/impl/vision_adapter.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {

FlashNextVisionSession::FlashNextVisionSession(const VisionModelView& vision_view,
                                               DeviceContext& device,
                                               DeviceSpan workspace,
                                               const qwen3_vision::WorkspacePlan& workspace_plan,
                                               std::uint32_t max_merged_tokens)
    : device_(device),
      weights_(adapt_vision_weights(vision_view)),
      encoder_(device, weights_),
      max_merged_tokens_(max_merged_tokens),
      workspace_plan_(workspace_plan),
      workspace_(workspace) {
    if (workspace_.data == nullptr) {
        throw std::invalid_argument("FlashNextVisionSession: workspace data is null");
    }
    if (workspace_.bytes < workspace_plan_.capacity_bytes) {
        throw std::invalid_argument("FlashNextVisionSession: workspace bytes (" +
                                    std::to_string(workspace_.bytes) +
                                    ") is smaller than plan capacity (" +
                                    std::to_string(workspace_plan_.capacity_bytes) + ")");
    }
}

Tensor FlashNextVisionSession::encode(const qwen3_5::VisionItemControl& control,
                                      std::span<const std::uint16_t> patches,
                                      cudaStream_t stream) {
    if (control.merged_count == 0 || control.merged_count > max_merged_tokens_) {
        throw std::invalid_argument("FlashNextVisionSession: merged_count is 0 or exceeds max_merged_tokens");
    }

    Tensor output = qwen3_vision::Encoder::bind_output(
        workspace_, workspace_plan_, control.merged_count, weights_.output_hidden);

    qwen3_vision::EncodeItemView item_view{
        .patches                = patches,
        .patch_count            = control.patch_count,
        .merged_count           = control.merged_count,
        .segment_length         = control.segment_length,
        .position_ids           = control.position_ids,
        .position_table_indices = control.position_table_indices,
        .position_table_weights = control.position_table_weights,
    };

    CudaEventTimer timer(device_, stream);
    timer.start();
    encoder_.encode(item_view, output, workspace_, workspace_plan_);
    timer.record_stop();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    elapsed_seconds_ += timer.elapsed_ms() / 1000.0;

    vision_tokens_ += control.merged_count;
    ++encode_count_;
    handoff_active_bytes_ = output.bytes();
    handoff_peak_bytes_   = std::max(handoff_peak_bytes_, handoff_active_bytes_);
    output_handoff_       = output;
    return output;
}

void FlashNextVisionSession::retire_handoff() noexcept {
    handoff_active_bytes_ = 0;
}

VisionWorkspaceMemorySummary
FlashNextVisionSession::memory_summary(std::uint32_t prompt_tokens) const noexcept {
    return VisionWorkspaceMemorySummary{
        .aggregate_prompt_tokens = prompt_tokens,
        .max_item_tokens         = workspace_plan_.max_merged_tokens,
        .general_capacity_bytes  = workspace_plan_.general_capacity_bytes,
        .encode_peak_bytes       = workspace_plan_.encode_peak_bytes,
        .handoff_offset_bytes    = workspace_plan_.handoff_offset_bytes,
        .handoff_capacity_bytes  = workspace_plan_.handoff_capacity_bytes,
        .handoff_active_bytes    = handoff_active_bytes_,
        .handoff_peak_bytes      = handoff_peak_bytes_,
    };
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
