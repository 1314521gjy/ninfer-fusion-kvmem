#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_vision {

// Fixed tower geometry for Qwen3 Vision
struct VisionTowerConfig {
    static constexpr int layers              = 27;
    static constexpr int hidden              = 1152;
    static constexpr int intermediate        = 4304;
    static constexpr int heads               = 16;
    static constexpr int head_dim            = hidden / heads; // 72
    static constexpr int patch_dim           = 3 * 2 * 16 * 16; // 1536
    static constexpr int merge               = 2;
    static constexpr int merge_unit          = merge * merge; // 4
    static constexpr int merger_hidden       = hidden * merge_unit; // 4608
    static constexpr int position_embeddings = 48 * 48; // 2304
    static constexpr int rotary_dim          = head_dim; // 72
    static constexpr float rope_theta        = 10000.0F;
    static constexpr float norm_epsilon      = 1.0e-6F;
    static constexpr float attention_scale   = 0.11785113019775792F; // 1.0 / sqrt(72)
};

// Passive generic views for Layer weights
struct LayerWeights {
    const Tensor* norm1_weight    = nullptr;
    const Tensor* norm1_bias      = nullptr;
    const Weight* qkv             = nullptr;
    const Tensor* qkv_bias        = nullptr;
    const Weight* projection      = nullptr;
    const Tensor* projection_bias = nullptr;
    const Tensor* norm2_weight    = nullptr;
    const Tensor* norm2_bias      = nullptr;
    const Weight* fc1             = nullptr;
    const Tensor* fc1_bias        = nullptr;
    const Weight* fc2             = nullptr;
    const Tensor* fc2_bias        = nullptr;
};

// Passive generic views for Merger weights
struct MergerWeights {
    const Tensor* norm_weight = nullptr;
    const Tensor* norm_bias   = nullptr;
    const Weight* fc1         = nullptr;
    const Tensor* fc1_bias    = nullptr;
    const Weight* fc2         = nullptr;
    const Tensor* fc2_bias    = nullptr;
};

// Complete generic Vision weights view
struct Weights {
    const Weight* patch_embed      = nullptr;
    const Tensor* patch_embed_bias = nullptr;
    const Tensor* position_embed   = nullptr;
    std::array<LayerWeights, VisionTowerConfig::layers> layers{};
    MergerWeights merger{};
    std::int32_t output_hidden = 0;
};

// Input item view to encode
struct EncodeItemView {
    std::span<const std::uint16_t> patches;
    std::size_t patch_count                   = 0;
    std::size_t merged_count                  = 0;
    std::int32_t segment_length               = 0;
    std::span<const std::int32_t> position_ids;
    std::span<const std::int32_t> position_table_indices;
    std::span<const float> position_table_weights;
};

// Workspace plan for memory sizing
struct WorkspacePlan {
    std::uint32_t max_merged_tokens    = 0;
    std::size_t general_capacity_bytes = 0;
    std::size_t encode_peak_bytes      = 0;
    std::size_t handoff_offset_bytes   = 0;
    std::size_t handoff_capacity_bytes = 0;
    std::size_t capacity_bytes         = 0;
};

// The target-neutral Qwen3 Vision Encoder
class Encoder {
public:
    Encoder(DeviceContext& device, const Weights& weights);

    [[nodiscard]] static std::size_t workspace_bytes(std::size_t patches,
                                                     std::size_t merged_tokens);
    [[nodiscard]] static WorkspacePlan plan_workspace(std::uint32_t max_merged_tokens,
                                                      std::size_t general_capacity_bytes,
                                                      std::int32_t output_hidden);
    [[nodiscard]] static Tensor bind_output(DeviceSpan backing, const WorkspacePlan& plan,
                                            std::size_t merged_tokens,
                                            std::int32_t output_hidden);
    void encode(const EncodeItemView& item, Tensor& output, DeviceSpan backing,
                const WorkspacePlan& plan) const;

    [[nodiscard]] std::int32_t output_hidden() const noexcept { return weights_.output_hidden; }
    [[nodiscard]] DeviceContext& device() const noexcept { return device_; }
    [[nodiscard]] const Weights& weights() const noexcept { return weights_; }

private:
    DeviceContext& device_;
    Weights weights_;
};

} // namespace ninfer::models::qwen3_vision
