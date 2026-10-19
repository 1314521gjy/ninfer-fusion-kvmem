#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/port/facade.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <models/qwen3_5/execution/vision.h>
#include <ninfer/models/qwen3_vision/vision.h>
#include <optional>

namespace ninfer::models::qwen3_8_flash_next::detail {

using namespace ninfer::portalias;

// KV streaming switch, PLAN-TIME view (M4-1). Same predicate as the runtime switch
// (text_executor.cpp): unset / empty / '0' = off. Deliberately no special case for "00"/"off"/
// "false" -- those read as ON here and at runtime, so the two can never disagree.
[[nodiscard]] inline bool flash_next_kv_stream_requested() noexcept {
    const char* env = std::getenv("NINFER_FLASH_NEXT_KV_STREAM");
    return !(env == nullptr || env[0] == '\0' || env[0] == '0');
}

// Page & block tokens for QSA attention and indexer
inline constexpr std::uint32_t kPageTokens        = 64; // QSA attention page size
inline constexpr std::uint32_t kBlockTokens       = 4;  // QSA indexer compressed block size
inline constexpr std::uint32_t kIndexerPageBlocks = 64; // Blocks per indexer page
inline constexpr std::uint32_t kIndexerPageTokens = kBlockTokens * kIndexerPageBlocks; // 256 tokens
inline constexpr std::uint32_t kMainPageGroupTokens = 256; // 256-token affine group

// Ceiling on recurrent-state slots. Each slot carries the GDN state for all 36 GDN layers, and
// the SSM part alone is 128 * 128 * 48 floats per layer, so a slot costs roughly 110 MiB and the
// full 64 reserve about 6.9 GiB. Both the planner and the package sizer must agree on this bound:
// when only one of them clamped, the package built a plan the planner then rejected.
inline constexpr std::uint32_t kMaxStateSlots = 64;

[[nodiscard]] inline constexpr std::uint32_t
flash_next_slots_per_lane(std::uint32_t speculative_draft_tokens) noexcept {
    return speculative_draft_tokens > 0 ? (speculative_draft_tokens + 1U) : 2U;
}

[[nodiscard]] inline constexpr std::uint32_t
flash_next_floor_slots(std::uint32_t max_concurrency,
                       std::uint32_t speculative_draft_tokens) noexcept {
    return flash_next_slots_per_lane(speculative_draft_tokens) * max_concurrency;
}

// Per 256-token group strides across all 12 QSA layers
// Attention K + V (BF16): 12 layers * 4 pages/group * (256 * 64 * 2 heads * 2 bytes * 2 K/V) = 6,291,456 bytes
inline constexpr std::size_t kAttentionKvBytesPerGroupBf16 =
    12ULL * 4ULL * (256ULL * 64ULL * 2ULL * 2ULL * 2ULL);
// Attention K + V (FP8): 12 layers * 4 pages/group * (256 * 64 * 2 heads * 1 byte * 2 K/V) = 3,145,728 bytes
inline constexpr std::size_t kAttentionKvBytesPerGroupFp8 =
    12ULL * 4ULL * (256ULL * 64ULL * 2ULL * 1ULL * 2ULL);
inline constexpr std::size_t kAttentionKvBytesPerGroup = kAttentionKvBytesPerGroupBf16;

// Indexer block keys: 12 layers * 1 page/group * (128 * 64 * 2 bytes) = 196,608 bytes
inline constexpr std::size_t kIndexerBlockKeysBytesPerGroup = 12ULL * (128ULL * 64ULL * 2ULL);

// Total physical stride per 256-token group: BF16 = 6,488,064 bytes; FP8 = 3,342,336 bytes
inline constexpr std::size_t kPhysicalStrideBytesPerGroupBf16 =
    kAttentionKvBytesPerGroupBf16 + kIndexerBlockKeysBytesPerGroup;
inline constexpr std::size_t kPhysicalStrideBytesPerGroupFp8 =
    kAttentionKvBytesPerGroupFp8 + kIndexerBlockKeysBytesPerGroup;
inline constexpr std::size_t kPhysicalStrideBytesPerGroup = kPhysicalStrideBytesPerGroupBf16;

[[nodiscard]] inline constexpr std::size_t
flash_next_attention_kv_bytes_per_group(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return kAttentionKvBytesPerGroupBf16;
    case KvCacheStorage::Fp8E4M3Row256:
        return kAttentionKvBytesPerGroupFp8;
    default:
        throw std::invalid_argument("Flash-Next supports only --kv-dtype bf16 and fp8");
    }
}

[[nodiscard]] inline constexpr std::size_t
flash_next_physical_stride_bytes_per_group(KvCacheStorage storage) {
    return flash_next_attention_kv_bytes_per_group(storage) + kIndexerBlockKeysBytesPerGroup;
}

// Recurrent state bytes per state slot across all recurrent modules
// GDN conv (36 layers): 36 * (10240 * 3 * 2) = 2,211,840 bytes
inline constexpr std::size_t kGdnConvBytesPerSlot = 36ULL * (10'240ULL * 3ULL * 2ULL);
// GDN SSM (36 layers): 36 * (128 * 128 * 48 * 4) = 113,246,208 bytes
inline constexpr std::size_t kGdnSsmBytesPerSlot = 36ULL * (128ULL * 128ULL * 48ULL * 4ULL);
// PLE conv (1 layer): 10240 * 9 * 2 = 184,320 bytes
inline constexpr std::size_t kPleConvBytesPerSlot = 10'240ULL * 9ULL * 2ULL;
// QSA raw keys (12 layers): 12 * (128 * 4 * 2) = 12,288 bytes
inline constexpr std::size_t kQsaRawKeysBytesPerSlot = 12ULL * (128ULL * 4ULL * 2ULL);
// QSA raw positions (12 layers): 12 * (3 * 4 * 4) = 576 bytes
inline constexpr std::size_t kQsaRawPositionsBytesPerSlot = 12ULL * (3ULL * 4ULL * 4ULL);
// Total recurrent state per slot = 115,655,232 bytes (at default FP32 GDN state)
inline constexpr std::size_t kRecurrentStateBytesPerSlot =
    kGdnConvBytesPerSlot + kGdnSsmBytesPerSlot + kPleConvBytesPerSlot + kQsaRawKeysBytesPerSlot +
    kQsaRawPositionsBytesPerSlot;

[[nodiscard]] inline constexpr std::size_t
flash_next_gdn_ssm_bytes_per_slot(GdnStateStorage storage) noexcept {
    const std::size_t elem_size =
        (storage == GdnStateStorage::BF16) ? sizeof(std::uint16_t) : sizeof(float);
    return 36ULL * (128ULL * 128ULL * 48ULL * elem_size);
}

[[nodiscard]] inline constexpr std::size_t
flash_next_recurrent_state_bytes_per_slot(GdnStateStorage storage) noexcept {
    return kGdnConvBytesPerSlot + flash_next_gdn_ssm_bytes_per_slot(storage) +
           kPleConvBytesPerSlot + kQsaRawKeysBytesPerSlot + kQsaRawPositionsBytesPerSlot;
}

inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

// Decode CUDA-graph context buckets. Token envelopes 2048 / 8192 / 32768 plus the
// startup-fixed max_context slot. Candidates at or above maximum_blocks are dropped
// except the final slot, so max_context=8192 dedups to {512, 2048} blocks.
inline constexpr std::uint32_t kFlashNextDecodeGraphBucketTokenEnvelopes[3] = {2048u, 8192u,
                                                                              32768u};
inline constexpr std::uint32_t kFlashNextDecodeGraphMaxBuckets              = 4;
inline constexpr std::size_t kFlashNextDecodeGraphBytesPerCapture =
    24ULL * 1024ULL * 1024ULL;
inline constexpr std::size_t kFlashNextMtpDraftGraphBytesPerCapture = 2ULL * 1024ULL * 1024ULL;

struct FlashNextDecodeGraphBuckets {
    std::array<std::uint32_t, kFlashNextDecodeGraphMaxBuckets> blocks{};
    std::uint32_t count = 0;
};

// topology_class packs an explicit integer key: bits [7:0] = batch_size (1..8),
// bits [15:8] = bucket_index (0..3). Lookups must use this field (or the stored
// bucket_index) and must not derive the bucket from frontier envelopes.
[[nodiscard]] inline constexpr std::uint32_t
flash_next_decode_graph_topology_class(std::uint32_t batch_size,
                                       std::uint32_t bucket_index) noexcept {
    return (bucket_index << 8) | batch_size;
}

[[nodiscard]] inline FlashNextDecodeGraphBuckets
flash_next_decode_graph_buckets(std::uint32_t maximum_blocks) {
    FlashNextDecodeGraphBuckets out{};
    if (maximum_blocks == 0) { return out; }
    for (std::uint32_t tokens : kFlashNextDecodeGraphBucketTokenEnvelopes) {
        const std::uint32_t blocks = (tokens + kBlockTokens - 1u) / kBlockTokens;
        if (blocks >= maximum_blocks) { continue; }
        if (out.count > 0 && out.blocks[out.count - 1] == blocks) { continue; }
        out.blocks[out.count++] = blocks;
    }
    if (out.count == 0 || out.blocks[out.count - 1] != maximum_blocks) {
        out.blocks[out.count++] = maximum_blocks;
    }
    return out;
}

[[nodiscard]] inline std::uint32_t
flash_next_decode_graph_select_bucket(const FlashNextDecodeGraphBuckets& buckets,
                                      std::int32_t active_blocks) {
    const auto need = static_cast<std::uint32_t>(std::max<std::int32_t>(active_blocks, 0));
    for (std::uint32_t i = 0; i < buckets.count; ++i) {
        if (buckets.blocks[i] >= need) { return i; }
    }
    return buckets.count == 0 ? 0 : buckets.count - 1;
}

// Fixed-address pinned ingress structure copied by a single captured cudaMemcpyAsync
struct FlashNextDecodeIngress {
    std::array<std::int32_t, 8> token_ids{};
    std::array<std::int32_t, 8> token_indices{};
    std::array<std::int32_t, 3 * 8> mrope_positions{}; // Planar [3, 8]
    std::array<std::int32_t, 8> table_rows{};
    std::array<std::int32_t, 8> source_slots{};
    std::array<std::int32_t, 8> destination_slots{};
    std::array<ops::SamplingConfig, 8> sampling{};
};

// Fixed-address pinned egress structure copied by a single captured D2H cudaMemcpyAsync
struct FlashNextDecodeEgress {
    std::array<std::int32_t, 8> sampled_tokens{};
};

// One immutable header per draft step, uploaded before the device-resident chain.
// Scalar views retain the alignment required by the MTP Op contracts.
struct FlashNextMtpDraftStep {
    alignas(16) std::int32_t token_index = 0;
    alignas(16) std::array<std::int32_t, 3> mrope_positions{};
    alignas(16) std::int32_t table_row = 0;
    alignas(16) std::int32_t source_slot = 0;
    alignas(16) std::int32_t destination_slot = 0;
};

struct FlashNextMtpDraftIngress {
    alignas(16) std::int32_t token_id = 0;
    std::array<FlashNextMtpDraftStep, 4> steps{};
};

struct FlashNextRuntimeConfig {
    std::uint32_t max_concurrency          = 1;    // 1..8
    std::uint32_t max_context              = 4096; // in tokens: 1..262144
    std::uint32_t state_slot_capacity      = 0;    // 0 -> default (slots_per_lane * max_concurrency + continuation_capacity)
    std::uint32_t continuation_capacity    = 0;  // Checkpoint cache slot capacity
    std::uint32_t prefill_chunk            = 1024; // nonzero multiple of 128, <= max_context
    std::uint32_t speculative_draft_tokens = 0;    // 0 -> speculative decoding off; K in [1, 4]
    ProposalHead proposal_head             = ProposalHead::Full;
    std::uint32_t draft_head_rows          = 32'768; // Used when proposal_head == ProposalHead::Optimized
    bool use_cuda_graph                    = true;
    bool vision_enabled                    = false;
    std::uint32_t max_vision_tokens        = 4096;
    // Prefill QSA attention: GQA tiled MMA (12 query heads share each KV tile). Default off.
    bool use_qsa_prefill_mma               = true;
    KvCacheStorage kv_cache                = KvCacheStorage::BFloat16;
    GdnStateStorage gdn_state_storage      = GdnStateStorage::FP32;
};

struct FlashNextRuntimePlan {
    FlashNextRuntimeConfig config;
    std::uint32_t main_page_groups         = 0;
    std::uint32_t resolved_tokens          = 0; // main_page_groups * 256
    // M4-2: what a REQUEST may occupy. With KV streaming on the device pool is a residency window,
    // so the context budget is max_context; with it off this is exactly resolved_tokens (byte for
    // byte the old behaviour). Every admission check must read THIS, never resolved_tokens.
    std::uint32_t admissible_tokens = 0;
    std::uint32_t attention_physical_pages = 0; // 4 * main_page_groups
    std::uint32_t indexer_physical_pages   = 0; // 1 * main_page_groups
    std::uint32_t attention_logical_pages  = 0; // ceil(max_context / 64)
    std::uint32_t indexer_logical_pages    = 0; // ceil(max_context / 256)
    std::uint32_t state_slots              = 0;
    std::uint32_t continuation_slots       = 0;
    std::uint32_t maximum_blocks           = 0; // ceil(max_context / 4)

    [[nodiscard]] std::size_t qsa_cache_layers() const noexcept {
        return kFullAttentionLayers + (config.speculative_draft_tokens > 0 ? 1U : 0U);
    }

    // Memory breakdown in bytes
    std::size_t attention_kv_bytes         = 0;
    std::size_t indexer_block_keys_bytes   = 0;
    std::size_t block_tables_bytes         = 0;
    std::size_t recurrent_state_bytes      = 0;
    std::size_t round_tensors_bytes        = 0;
    std::size_t workspace_bytes            = 0;
    std::size_t cuda_graph_allowance_bytes = 0;
    std::size_t total_device_bytes         = 0;

    std::optional<qwen3_vision::WorkspacePlan> vision_workspace;
    ninfer::runtime::SequenceCapacityCurve capacity_curve;
};

[[nodiscard]] ninfer::runtime::SequenceCapacityCurve
flash_next_capacity_curve(const FlashNextRuntimeConfig& config);

[[nodiscard]] FlashNextRuntimePlan
finalize_flash_next_runtime_plan(const FlashNextRuntimeConfig& config,
                                 std::uint32_t selected_main_page_groups);

} // namespace ninfer::models::qwen3_8_flash_next::detail
