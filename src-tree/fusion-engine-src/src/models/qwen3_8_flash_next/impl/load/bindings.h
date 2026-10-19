#pragma once

#include "artifact/binder.h"
#include "artifact/typed_binding.h"
#include "models/qwen3_8_flash_next/impl/port/facade.h"
#include "ninfer/types.h"
#include "models/qwen3_5/frontend/frontend_resources.h"

#include <array>
#include <cstddef>

namespace ninfer::models::qwen3_8_flash_next::detail {

using ninfer::port::Binder;
using namespace ninfer::portalias;

inline constexpr std::size_t kTextLayers   = 48;
inline constexpr std::size_t kPleShards    = 128;
inline constexpr std::size_t kVisionLayers = 27;

struct LoadFeatures {
    bool vision                            = false;
    bool mtp                               = false;
    ProposalHead proposal_head             = ProposalHead::Full;
    std::uint32_t draft_head_rows          = 32'768;
    // The FP8 flags are a *binding* decision, not only a load-time conversion: when set, the BF16
    // head / embedding is retained in the artifact's file mapping (zero device bytes) and the FP8
    // copy is quantized from the host span. Binding them on the device instead made each flag
    // cost 608 MiB of VRAM rather than saving it (bench/d17 server logs, 2026-09-05).
    bool quantize_output_head_fp8          = false;
    bool quantize_token_embedding_fp8      = false;
};

struct HyperConnectionPlan {
    artifact::ObjectHandle block_inject;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle input_mix_down;
    artifact::ObjectHandle input_mix_up;
};

struct HyperMixerPlan {
    artifact::ObjectHandle norm;
    artifact::ObjectHandle input_mix_down;
    artifact::ObjectHandle input_mix_up;
};

struct MoePlan {
    artifact::ObjectHandle router;
    artifact::ObjectHandle shared_down;
    artifact::ObjectHandle shared_gate;
    artifact::ObjectHandle shared_up;
    artifact::ObjectHandle shared_gate_weight;
    artifact::ObjectHandle expert_gate_up;
    artifact::ObjectHandle expert_down;
    bool experts_nvfp4 = true;
    // The two expert banks were retained in the artifact's file mapping (host-tiered placement,
    // NINFER_FLASH_NEXT_EXPERTS_HOST) instead of being materialized on the device. load_moe then
    // builds the bank views over the mapped span and marks them on_host.
    bool experts_on_host = false;
    // Block-5: set when the container stores the two expert objects in ggml block formats (the
    // `gguf_blocks_v1` spelling the port gained in this change). The two records below name the
    // resolved per-object format so the load layer builds the block views without re-reading the
    // directory -- gate_up and down need not share one format.
    bool experts_gguf = false;
    NumericFormat gguf_gate_up_format = NumericFormat::GGUF_Q8_0;
    NumericFormat gguf_down_format    = NumericFormat::GGUF_Q8_0;
};

struct GdnPlan {
    artifact::ObjectHandle a_log;
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle a_b_projection;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle query_key_value_z;
    artifact::ObjectHandle output;
};

struct AttentionPlan {
    artifact::ObjectHandle indexer_query_key;
    artifact::ObjectHandle indexer_key_norm;
    artifact::ObjectHandle indexer_query_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle query_gate_key_value;
    artifact::ObjectHandle output;
};

struct TextLayerPlan {
    HyperConnectionPlan attention_hyper;
    MoePlan moe;
    HyperConnectionPlan mlp_hyper;
    GdnPlan gdn{};
    AttentionPlan attention{};
    bool is_full_attention = false;
};

struct PlePlan {
    artifact::ObjectHandle convolution;
    artifact::ObjectHandle key_projection;
    artifact::ObjectHandle conv_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle value_projection;
    artifact::ObjectHandle layer_multipliers;
    artifact::ObjectHandle ngram_head_offsets;
    artifact::ObjectHandle ngram_head_vocab_sizes;
    std::array<artifact::ObjectHandle, kPleShards> shards;
};

struct MtpPlan {
    artifact::ObjectHandle embedding_projection;
    artifact::ObjectHandle hidden_projection;
    HyperMixerPlan mixer;
    HyperConnectionPlan attention_hyper;
    MoePlan moe;
    HyperConnectionPlan mlp_hyper;
    AttentionPlan attention;
    artifact::ObjectHandle embedding_norm;
    artifact::ObjectHandle hidden_norm;
};

struct VisionLayerPlan {
    artifact::ObjectHandle qkv;
    artifact::ObjectHandle qkv_bias;
    artifact::ObjectHandle output;
    artifact::ObjectHandle output_bias;
    artifact::ObjectHandle fc1;
    artifact::ObjectHandle fc1_bias;
    artifact::ObjectHandle fc2;
    artifact::ObjectHandle fc2_bias;
    artifact::ObjectHandle norm1_weight;
    artifact::ObjectHandle norm1_bias;
    artifact::ObjectHandle norm2_weight;
    artifact::ObjectHandle norm2_bias;
};

struct VisionPlan {
    artifact::ObjectHandle patch_embedding;
    artifact::ObjectHandle patch_embedding_bias;
    artifact::ObjectHandle position_embedding;
    std::array<VisionLayerPlan, kVisionLayers> layers;
    artifact::ObjectHandle merger_fc1;
    artifact::ObjectHandle merger_fc1_bias;
    artifact::ObjectHandle merger_fc2;
    artifact::ObjectHandle merger_fc2_bias;
    artifact::ObjectHandle merger_norm_weight;
    artifact::ObjectHandle merger_norm_bias;
};

struct BindingPlan {
    LoadFeatures features;
    qwen3_5::FrontendResourcePlan frontend;
    artifact::ObjectHandle token_embedding;
    std::array<TextLayerPlan, kTextLayers> text_layers;
    PlePlan ple;
    artifact::ObjectHandle output_head;
    HyperMixerPlan final_mixer;
    MtpPlan mtp;
    VisionPlan vision;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

// TWO overloads, because the reference has two and each call site needs its own:
//   `artifact::Binder&`  -- what the reference declares at its `bindings.h:156`, and what
//                           `loader.cpp:23` calls (it builds a plain `artifact::Binder`).
//   `ninfer::port::Binder&` -- what `bindings.cpp:293` DEFINES, because that implementation body uses
//                           `require_tensor` / `materialize_on_device` / `retain_mapped_tensor` /
//                           `validate_only` / `payload` / `find`, none of which exist on the engine
//                           binder; they are adapter methods. `package.cpp` calls this one.
// Spelling only the engine type (upstream's text) left `loader.cpp` undefined; spelling only the
// adapter made `loader.cpp:23` fail with "cannot convert artifact::Binder to port::Binder&". Both were
// tried, both broke one caller. The adapter keeps the engine binder reachable via `engine()`, so the
// second overload forwards and the semantics are the reference's.
[[nodiscard]] ArtifactLoadPlan bind_artifact(artifact::Binder& binder, LoadFeatures features);
[[nodiscard]] ArtifactLoadPlan bind_artifact(Binder& binder, LoadFeatures features);

} // namespace ninfer::models::qwen3_8_flash_next::detail
