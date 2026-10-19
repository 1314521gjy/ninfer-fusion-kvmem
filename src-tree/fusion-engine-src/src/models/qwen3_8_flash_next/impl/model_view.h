#pragma once


#include "core/weight.h"
#include "core/tensor.h"
#include "models/qwen3_8_flash_next/impl/expert_bank.h"
#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include <array>
#include <cstddef>
#include <optional>

namespace ninfer {
class DeviceArena;
}

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr std::size_t kFullAttentionLayers = 12;
inline constexpr std::size_t kGdnLayers           = 36;

struct HyperConnectionWeights {
    Weight block_inject;
    Tensor norm;
    Weight input_mix_down;
    Weight input_mix_up;
};

struct HyperMixerWeights {
    Tensor norm;
    Weight input_mix_down;
    Weight input_mix_up;
};

struct MoeWeights {
    Weight router;
    Weight shared_down;
    Weight shared_gate;
    Weight shared_up;
    Weight shared_gate_weight;
    Nvfp4ExpertBankView expert_gate_up;
    Nvfp4ExpertBankView expert_down;
    // Block-5 IQ3_S artifacts store the two banks as whole ggml block rows instead: `experts_gguf`
    // selects this pair, and the nvfp4 views above stay empty on purpose -- the MoE leaf's
    // exact-bank validation refuses a forward over them, which is the honest state until the
    // compute wiring lands (Stage 1 is load/read only).
    bool experts_gguf = false;
    GgufExpertBankView gguf_gate_up;
    GgufExpertBankView gguf_down;
};

struct MoeBf16Weights {
    Weight router;
    Weight shared_down;
    Weight shared_gate;
    Weight shared_up;
    Weight shared_gate_weight;
    Bf16ExpertBankView expert_gate_up;
    Bf16ExpertBankView expert_down;
};

struct TextLayerWeights {
    HyperConnectionWeights attention_hyper;
    MoeWeights moe;
    HyperConnectionWeights mlp_hyper;
};

struct GdnWeights {
    Tensor a_log;
    Tensor convolution;
    Tensor dt_bias;
    Weight a_b_projection;
    Tensor norm;
    Weight query_key_value_z;
    Weight output;
};

struct AttentionWeights {
    Weight indexer_query_key;
    Tensor indexer_key_norm;
    Tensor indexer_query_norm;
    Tensor key_norm;
    Tensor query_norm;
    Weight query_gate_key_value;
    Weight output;
};

struct PleWeights {
    Tensor convolution;
    Weight key_projection;
    Tensor conv_norm;
    Tensor key_norm;
    Tensor query_norm;
    Weight value_projection;
    PleTableView table;
};

struct MtpModelView {
    Weight embedding_projection;
    Weight hidden_projection;
    HyperMixerWeights mixer;
    HyperConnectionWeights attention_hyper;
    AttentionWeights attention;
    HyperConnectionWeights mlp_hyper;
    MoeWeights moe;
    Tensor embedding_norm;
    Tensor hidden_norm;
};

struct OptimizedProposalWeights {
    Weight head;
    Tensor token_ids;
};

struct TextModelView {
    DeviceArena* weights_arena = nullptr;
    Weight token_embedding;
    std::array<TextLayerWeights, 48> layers;
    std::array<AttentionWeights, kFullAttentionLayers> full_attention;
    std::array<GdnWeights, kGdnLayers> gdn;
    PleWeights ple;
    Weight output_head;
    HyperMixerWeights final_mixer;
    std::optional<MtpModelView> mtp;
    std::optional<OptimizedProposalWeights> proposal;
};

struct VisionLayerWeights {
    Weight qkv;
    Tensor qkv_bias;
    Weight output;
    Tensor output_bias;
    Weight fc1;
    Tensor fc1_bias;
    Weight fc2;
    Tensor fc2_bias;
    Tensor norm1_weight;
    Tensor norm1_bias;
    Tensor norm2_weight;
    Tensor norm2_bias;
};

struct VisionModelView {
    Weight patch_embedding;
    Tensor patch_embedding_bias;
    Tensor position_embedding;
    std::array<VisionLayerWeights, 27> layers;
    Weight merger_fc1;
    Tensor merger_fc1_bias;
    Weight merger_fc2;
    Tensor merger_fc2_bias;
    Tensor merger_norm_weight;
    Tensor merger_norm_bias;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
