#pragma once

#include "models/qwen3_8_flash_next/impl/model_view.h"
#include <models/qwen3_5/execution/vision.h>
#include <ninfer/models/qwen3_vision/vision.h>

#include <cstddef>

namespace ninfer::models::qwen3_8_flash_next::detail {

inline ::ninfer::models::qwen3_vision::Weights adapt_vision_weights(const VisionModelView& v) {
    ::ninfer::models::qwen3_vision::Weights out;
    out.patch_embed      = &v.patch_embedding;
    out.patch_embed_bias = &v.patch_embedding_bias;
    out.position_embed   = &v.position_embedding;
    for (std::size_t i = 0; i < out.layers.size(); ++i) {
        const auto& src     = v.layers[i];
        auto& dst           = out.layers[i];
        dst.norm1_weight    = &src.norm1_weight;
        dst.norm1_bias      = &src.norm1_bias;
        dst.qkv             = &src.qkv;
        dst.qkv_bias        = &src.qkv_bias;
        dst.projection      = &src.output;
        dst.projection_bias = &src.output_bias;
        dst.norm2_weight    = &src.norm2_weight;
        dst.norm2_bias      = &src.norm2_bias;
        dst.fc1             = &src.fc1;
        dst.fc1_bias        = &src.fc1_bias;
        dst.fc2             = &src.fc2;
        dst.fc2_bias        = &src.fc2_bias;
    }
    out.merger.norm_weight = &v.merger_norm_weight;
    out.merger.norm_bias   = &v.merger_norm_bias;
    out.merger.fc1         = &v.merger_fc1;
    out.merger.fc1_bias    = &v.merger_fc1_bias;
    out.merger.fc2         = &v.merger_fc2;
    out.merger.fc2_bias    = &v.merger_fc2_bias;
    out.output_hidden      = 2560;
    return out;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
