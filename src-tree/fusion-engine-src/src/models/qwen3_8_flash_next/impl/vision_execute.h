#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/types.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"
#include <models/qwen3_5/program/vision_control.h>
#include <models/qwen3_5/execution/vision.h>
#include <ninfer/models/qwen3_vision/vision.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

class FlashNextVisionSession {
public:
    FlashNextVisionSession(const VisionModelView& vision_view, DeviceContext& device,
                           DeviceSpan workspace,
                           const qwen3_vision::WorkspacePlan& workspace_plan,
                           std::uint32_t max_merged_tokens = 4096);

    ~FlashNextVisionSession() = default;

    FlashNextVisionSession(const FlashNextVisionSession&)            = delete;
    FlashNextVisionSession& operator=(const FlashNextVisionSession&) = delete;
    FlashNextVisionSession(FlashNextVisionSession&&)                 = default;
    FlashNextVisionSession& operator=(FlashNextVisionSession&&)      = delete;

    // Encodes a single vision item using its control metadata and patch payload.
    // Returns a Tensor view of shape [2560, control.merged_count] in the handoff buffer.
    Tensor encode(const qwen3_5::VisionItemControl& control,
                  std::span<const std::uint16_t> patches,
                  cudaStream_t stream);

    void retire_handoff() noexcept;

    [[nodiscard]] double elapsed_seconds() const noexcept { return elapsed_seconds_; }
    [[nodiscard]] std::uint64_t vision_tokens() const noexcept { return vision_tokens_; }
    [[nodiscard]] std::uint64_t encode_count() const noexcept { return encode_count_; }

    [[nodiscard]] VisionWorkspaceMemorySummary memory_summary(std::uint32_t prompt_tokens) const noexcept;
    [[nodiscard]] const qwen3_vision::WorkspacePlan& workspace_plan() const noexcept { return workspace_plan_; }
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept { return workspace_plan_.capacity_bytes; }
    [[nodiscard]] const Tensor& output_tensor() const noexcept { return output_handoff_; }
    [[nodiscard]] const DeviceSpan& workspace() const noexcept { return workspace_; }

private:
    DeviceContext& device_;
    qwen3_vision::Weights weights_;
    qwen3_vision::Encoder encoder_;
    std::uint32_t max_merged_tokens_ = 4096;
    qwen3_vision::WorkspacePlan workspace_plan_;
    DeviceSpan workspace_{};
    Tensor output_handoff_;

    double elapsed_seconds_           = 0.0;
    std::uint64_t vision_tokens_      = 0;
    std::uint64_t encode_count_       = 0;
    std::size_t handoff_active_bytes_ = 0;
    std::size_t handoff_peak_bytes_   = 0;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
