#pragma once

#include "core/tensor.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention.h"
#include "models/qwen3_8_flash_next/impl/qsa_indexer.h"

#include <array>

namespace ninfer::models::qwen3_8_flash_next::detail {

struct FlashNextDecodeStateView {
    // The final cache belongs to MTP and is bound only when speculation is enabled.
    std::array<QsaIndexerCacheView, kFullAttentionLayers + 1> qsa_indexer_caches;
    std::array<QsaAttentionCacheView, kFullAttentionLayers + 1> qsa_attention_caches;
    std::array<Tensor, kGdnLayers> gdn_convolution_states; // BF16 [10240, 3, state_slots]
    std::array<Tensor, kGdnLayers> gdn_ssm_states;         // FP32 or BF16 [128, 128, 48, state_slots]
    Tensor ple_convolution_states;                         // BF16 [10240, 9, state_slots]
    Tensor mtp_backbone_hidden;                            // BF16 [10240, state_slots]
    Tensor mtp_backbone_positions;                         // I32 [3, state_slots]
};

void validate_flash_next_decode_state(const FlashNextDecodeStateView& state,
                                      std::int32_t state_slots);

} // namespace ninfer::models::qwen3_8_flash_next::detail
