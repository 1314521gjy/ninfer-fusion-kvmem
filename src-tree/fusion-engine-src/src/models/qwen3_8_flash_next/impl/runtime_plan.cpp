#include "models/qwen3_8_flash_next/impl/runtime_plan.h"

#include "ninfer/ops/sampling.h"
#include "models/qwen3_8_flash_next/impl/text_decode.h"
#include "models/qwen3_8_flash_next/impl/mtp_forward.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {

namespace {

template <typename T>
T checked_add(T a, T b) {
    if (b > 0 && a > std::numeric_limits<T>::max() - b) {
        throw std::overflow_error("runtime plan: integer addition overflow");
    }
    return a + b;
}

template <typename T>
T checked_mul(T a, T b) {
    if (a > 0 && b > 0 && a > std::numeric_limits<T>::max() / b) {
        throw std::overflow_error("runtime plan: integer multiplication overflow");
    }
    return a * b;
}

std::size_t checked_align_up_256(std::size_t bytes) {
    if (bytes > std::numeric_limits<std::size_t>::max() - 255ULL) {
        throw std::overflow_error("runtime plan: alignment overflow");
    }
    return (bytes + 255ULL) & ~255ULL;
}

void validate_config_invariants(const FlashNextRuntimeConfig& config,
                                std::uint32_t& resolved_state_slots) {
    if (config.max_concurrency < 1 || config.max_concurrency > 8) {
        throw std::invalid_argument(
            "Flash-Next max_concurrency must be in [1, 8] (CUDA graphs decode hard limit)");
    }
    if (config.max_context < 1 || config.max_context > 262'144) {
        throw std::invalid_argument("Flash-Next max_context must be in [1, 262144]");
    }
    if (config.prefill_chunk == 0 || config.prefill_chunk % kPrefillChunkAlignment != 0 ||
        config.prefill_chunk > config.max_context) {
        throw std::invalid_argument(
            "Flash-Next prefill_chunk must be a nonzero multiple of " +
            std::to_string(kPrefillChunkAlignment) + " and <= max_context (" +
            std::to_string(config.max_context) + ")");
    }
    if (config.speculative_draft_tokens > 4) {
        throw std::invalid_argument("Flash-Next speculative_draft_tokens must be in [0, 4]");
    }
    if (config.kv_cache != KvCacheStorage::BFloat16 &&
        config.kv_cache != KvCacheStorage::Fp8E4M3Row256) {
        throw std::invalid_argument(
            "Flash-Next supports only --kv-dtype bf16 and fp8 (requested unsupported kv-dtype)");
    }

    const std::uint32_t slots_per_lane =
        flash_next_slots_per_lane(config.speculative_draft_tokens);
    const std::uint32_t floor_slots =
        flash_next_floor_slots(config.max_concurrency, config.speculative_draft_tokens);
    const std::uint32_t min_state_slots = floor_slots + config.continuation_capacity;
    resolved_state_slots = config.state_slot_capacity;
    if (resolved_state_slots == 0) {
        resolved_state_slots = min_state_slots;
    } else if (resolved_state_slots < min_state_slots || resolved_state_slots > kMaxStateSlots) {
        // Defensive invariant for configs built directly -- tests, and any future caller that
        // does not come through Package::make_sequence_planner, which now clamps
        // --max-private-continuations to the state-slot budget before we ever get here. Say what
        // can actually be asked for at this concurrency rather than only what was rejected.
        //
        // This comment previously argued that clamping was unsafe because the Engine sized its
        // private catalog from the un-clamped option and would hold more entries than the target
        // had continuation slots -- the disagreement behind the concurrency-4 crash. That is no
        // longer true: construct_registered writes the clamped value back into the effective
        // options (registry.cpp) and the Engine constructs its core from those, so the two sides
        // now read the same number by construction.
        const std::uint32_t cont_cap_limit =
            kMaxStateSlots > floor_slots ? kMaxStateSlots - floor_slots : 0U;
        throw std::invalid_argument(
            "Flash-Next state_slot_capacity must be in [slots_per_lane * max_concurrency + continuation_capacity (" +
            std::to_string(min_state_slots) + "), " + std::to_string(kMaxStateSlots) +
            "]; at max_concurrency " + std::to_string(config.max_concurrency) +
            " the continuation capacity may be at most " + std::to_string(cont_cap_limit));
    }
}

std::size_t
compute_fixed_base_bytes(const FlashNextRuntimeConfig& config, std::uint32_t resolved_state_slots,
                         std::uint32_t attention_logical_pages, std::uint32_t indexer_logical_pages,
                         std::uint32_t maximum_blocks, std::size_t& block_tables_bytes,
                         std::size_t& recurrent_state_bytes, std::size_t& round_tensors_bytes,
                         std::size_t& workspace_bytes) {
    // 1. Shared Single Block tables
    const std::size_t single_att_table_plane = checked_align_up_256(checked_mul<std::size_t>(
        static_cast<std::size_t>(attention_logical_pages) * config.max_concurrency,
        sizeof(std::int32_t)));
    const std::size_t single_idx_table_plane = checked_align_up_256(checked_mul<std::size_t>(
        static_cast<std::size_t>(indexer_logical_pages) * config.max_concurrency,
        sizeof(std::int32_t)));
    block_tables_bytes = checked_add(single_att_table_plane, single_idx_table_plane);

    // 2. Recurrent states
    const std::size_t single_gdn_conv = checked_align_up_256(
        checked_mul<std::size_t>(10'240ULL * 3ULL * sizeof(std::uint16_t), resolved_state_slots));
    const std::size_t gdn_ssm_elem_size = (config.gdn_state_storage == GdnStateStorage::BF16)
                                              ? sizeof(std::uint16_t)
                                              : sizeof(float);
    const std::size_t single_gdn_ssm = checked_align_up_256(
        checked_mul<std::size_t>(128ULL * 128ULL * 48ULL * gdn_ssm_elem_size, resolved_state_slots));
    const std::size_t ple_conv = checked_align_up_256(
        checked_mul<std::size_t>(10'240ULL * 9ULL * sizeof(std::uint16_t), resolved_state_slots));
    const std::size_t single_raw_keys = checked_align_up_256(
        checked_mul<std::size_t>(128ULL * 4ULL * sizeof(std::uint16_t), resolved_state_slots));
    const std::size_t single_raw_pos = checked_align_up_256(
        checked_mul<std::size_t>(3ULL * 4ULL * sizeof(std::int32_t), resolved_state_slots));
    const std::size_t cache_layers = kFullAttentionLayers +
                                    (config.speculative_draft_tokens > 0 ? 1ULL : 0ULL);

    recurrent_state_bytes = checked_add(
        checked_add(checked_mul(36ULL, single_gdn_conv), checked_mul(36ULL, single_gdn_ssm)),
        checked_add(ple_conv, checked_add(checked_mul(cache_layers, single_raw_keys),
                                          checked_mul(cache_layers, single_raw_pos))));
    if (config.speculative_draft_tokens > 0) {
        recurrent_state_bytes = checked_add(recurrent_state_bytes, checked_align_up_256(
            10'240ULL * sizeof(std::uint16_t) * resolved_state_slots));
        recurrent_state_bytes = checked_add(recurrent_state_bytes, checked_align_up_256(
            3ULL * sizeof(std::int32_t) * resolved_state_slots));
    }

    // 3. Round buffers (pinned/device ingress & egress, plus gathered PLE, hidden, logits)
    const std::uint32_t round_batch_tokens =
        std::max(config.max_concurrency,
                 config.speculative_draft_tokens > 0 ? (config.speculative_draft_tokens + 1U) : 1U);

    round_tensors_bytes = checked_add(
        checked_align_up_256(sizeof(FlashNextDecodeIngress)),
        checked_add(
            checked_align_up_256(sizeof(FlashNextDecodeEgress)),
            checked_add(
                checked_align_up_256(2'560ULL * round_batch_tokens * sizeof(std::uint16_t)),
                checked_add(
                    checked_align_up_256(2'560ULL * round_batch_tokens * sizeof(std::uint16_t)),
                    checked_add(
                        checked_align_up_256(10'240ULL * round_batch_tokens * sizeof(std::uint16_t)),
                        checked_align_up_256(248'320ULL * round_batch_tokens *
                                             sizeof(std::uint16_t)))))));

    if (config.speculative_draft_tokens > 0) {
        const auto rows = config.proposal_head == ProposalHead::Optimized
                              ? config.draft_head_rows : 248'320U;
        round_tensors_bytes = checked_add(round_tensors_bytes,
            checked_align_up_256(2'560ULL * sizeof(std::uint16_t)) +
            checked_align_up_256(10'240ULL * sizeof(std::uint16_t)) +
            checked_align_up_256(rows * sizeof(std::uint16_t)) + 256ULL +
            checked_align_up_256(sizeof(FlashNextMtpDraftIngress)));
    }

    // 4. Text decode and prefill workspace peak
    const std::uint32_t decode_batch_capacity = std::max(
        config.max_concurrency,
        config.speculative_draft_tokens > 0 ? (config.speculative_draft_tokens + 1U) : 1U);
    const std::size_t decode_workspace =
        flash_next_text_decode_workspace_capacity_bytes(maximum_blocks, decode_batch_capacity, config.speculative_draft_tokens > 0);
    const std::size_t prefill_workspace =
        flash_next_text_prefill_workspace_capacity_bytes(maximum_blocks, config.prefill_chunk, config.speculative_draft_tokens > 0);
    const std::size_t general_workspace = std::max({decode_workspace, prefill_workspace,
        config.speculative_draft_tokens > 0
            ? flash_next_mtp_workspace_capacity_bytes(maximum_blocks, 1) : std::size_t{0}});
    workspace_bytes                     = general_workspace;

    if (config.vision_enabled) {
        const std::uint32_t merged =
            config.max_vision_tokens > 0 ? config.max_vision_tokens : 4096U;
        const auto vision_plan =
            qwen3_vision::Encoder::plan_workspace(merged, general_workspace, 2560);
        workspace_bytes = std::max(general_workspace, vision_plan.capacity_bytes);
    }

    const std::size_t sampling_workspace_bytes = std::max<std::size_t>(
        16ULL * 1024ULL * 1024ULL,
        ops::sampling_workspace_capacity_bytes(
            248'320, 1, static_cast<std::int32_t>(config.max_concurrency)));
    const std::size_t sampling_arrays_bytes = checked_align_up_256(
        config.max_concurrency * (sizeof(ops::SamplingConfig) + 2 * sizeof(std::int32_t))) +
        config.max_concurrency * 5 * ((248077 + 31) / 32) * sizeof(std::int32_t);

    return checked_add(
        block_tables_bytes,
        checked_add(
            recurrent_state_bytes,
            checked_add(round_tensors_bytes,
                        checked_add(workspace_bytes,
                                    checked_add(sampling_workspace_bytes, sampling_arrays_bytes)))));
}

} // namespace

ninfer::runtime::SequenceCapacityCurve
flash_next_capacity_curve(const FlashNextRuntimeConfig& config) {
    std::uint32_t resolved_state_slots = 0;
    validate_config_invariants(config, resolved_state_slots);

    const std::uint32_t groups_per_seq =
        (config.max_context + kMainPageGroupTokens - 1U) / kMainPageGroupTokens;
    const std::uint32_t min_groups =
        flash_next_kv_stream_requested()
            // KV streaming (M4-1): with the switch on the device pool is a RESIDENCY WINDOW, not
            // the context; a lane still needs one group, so the floor is max(1, concurrency).
            ? std::max<std::uint32_t>(1U, config.max_concurrency)
            // Switch off: byte-for-byte today's expression.
            : std::max<std::uint32_t>(groups_per_seq, config.max_concurrency);
    const std::uint32_t max_groups =
        checked_mul<std::uint32_t>(config.max_concurrency, groups_per_seq);

    const std::uint32_t attention_logical_pages =
        (config.max_context + kPageTokens - 1U) / kPageTokens;
    const std::uint32_t indexer_logical_pages =
        (config.max_context + kIndexerPageTokens - 1U) / kIndexerPageTokens;
    const std::uint32_t maximum_blocks =
        std::min<std::uint32_t>(65'536U, (config.max_context + kBlockTokens - 1U) / kBlockTokens);

    std::size_t block_tables_bytes    = 0;
    std::size_t recurrent_state_bytes = 0;
    std::size_t round_tensors_bytes   = 0;
    std::size_t workspace_bytes       = 0;
    const std::size_t fixed_base_bytes =
        compute_fixed_base_bytes(config, resolved_state_slots, attention_logical_pages,
                                 indexer_logical_pages, maximum_blocks, block_tables_bytes,
                                 recurrent_state_bytes, round_tensors_bytes, workspace_bytes);

    const std::size_t graph_allowance =
        config.use_cuda_graph
            ? checked_mul<std::size_t>(
                  kFlashNextDecodeGraphBytesPerCapture * (config.max_concurrency + config.speculative_draft_tokens) +
                      kFlashNextMtpDraftGraphBytesPerCapture * config.speculative_draft_tokens,
                  flash_next_decode_graph_buckets(maximum_blocks).count)
            : 0ULL;

    const std::size_t stride_bytes = flash_next_physical_stride_bytes_per_group(config.kv_cache) /
        kFullAttentionLayers * (kFullAttentionLayers +
                               (config.speculative_draft_tokens > 0 ? 1ULL : 0ULL));
    ninfer::runtime::SequenceCapacityCurve curve{};
    curve.main_page_tokens                     = kMainPageGroupTokens;
    curve.minimum_main_page_groups             = min_groups;
    curve.maximum_main_page_groups             = max_groups;
    curve.bytes_per_additional_main_page_group = stride_bytes;
    curve.minimum_device_reservation_bytes     = checked_add(
        checked_add(fixed_base_bytes, checked_mul<std::size_t>(min_groups, stride_bytes)),
        graph_allowance);

    return curve;
}

FlashNextRuntimePlan finalize_flash_next_runtime_plan(const FlashNextRuntimeConfig& config,
                                                      std::uint32_t selected_main_page_groups) {
    std::uint32_t resolved_state_slots = 0;
    validate_config_invariants(config, resolved_state_slots);

    const auto curve = flash_next_capacity_curve(config);
    if (selected_main_page_groups < curve.minimum_main_page_groups ||
        selected_main_page_groups > curve.maximum_main_page_groups) {
        // The bounds are worth printing: for max_context 32768 with one lane the curve
        // collapses to a single legal value (128 groups), and the message without numbers
        // sends the reader guessing.
        throw std::invalid_argument(
            "Flash-Next selected_main_page_groups " + std::to_string(selected_main_page_groups) +
            " outside capacity curve [" + std::to_string(curve.minimum_main_page_groups) + ", " +
            std::to_string(curve.maximum_main_page_groups) + "] for max_context " +
            std::to_string(config.max_context) + " and max_concurrency " +
            std::to_string(config.max_concurrency) + " (" +
            std::to_string(curve.main_page_tokens) + " tokens per group)");
    }

    FlashNextRuntimePlan plan{};
    plan.config                     = config;
    plan.config.state_slot_capacity = resolved_state_slots;
    if (const char* env = std::getenv("NINFER_FLASH_NEXT_QSA_PREFILL_MMA");
        env != nullptr && env[0] != '\0') {
        plan.config.use_qsa_prefill_mma =
            !(std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0 ||
              std::strcmp(env, "false") == 0);
    }
    std::fprintf(stderr, "qsa_prefill_mma=%s\n",
                 plan.config.use_qsa_prefill_mma ? "on" : "off");
    plan.main_page_groups           = selected_main_page_groups;
    plan.resolved_tokens =
        checked_mul<std::uint32_t>(selected_main_page_groups, kMainPageGroupTokens);
    // M4-2: must be computed HERE, inside finalize -- runtime_state.cpp:14-45 re-finalizes and
    // compares field by field, so a value set anywhere else would throw a plan mismatch.
    plan.admissible_tokens =
        flash_next_kv_stream_requested() ? config.max_context : plan.resolved_tokens;
    plan.attention_physical_pages = checked_mul<std::uint32_t>(4U, selected_main_page_groups);
    plan.indexer_physical_pages   = selected_main_page_groups;
    plan.attention_logical_pages  = (config.max_context + kPageTokens - 1U) / kPageTokens;
    plan.indexer_logical_pages =
        (config.max_context + kIndexerPageTokens - 1U) / kIndexerPageTokens;
    plan.state_slots = resolved_state_slots;
    plan.continuation_slots = config.continuation_capacity;
    plan.maximum_blocks =
        std::min<std::uint32_t>(65'536U, (config.max_context + kBlockTokens - 1U) / kBlockTokens);

    const std::size_t kv_element_bytes =
        (config.kv_cache == KvCacheStorage::Fp8E4M3Row256) ? 1ULL : sizeof(std::uint16_t);
    const std::size_t single_att_kv_plane = checked_align_up_256(checked_mul<std::size_t>(
        256ULL * 64ULL * 2ULL * kv_element_bytes, plan.attention_physical_pages));
    plan.attention_kv_bytes = checked_mul<std::size_t>(2ULL * plan.qsa_cache_layers(), single_att_kv_plane);

    const std::size_t single_indexer_keys_plane = checked_align_up_256(checked_mul<std::size_t>(
        128ULL * 64ULL * sizeof(std::uint16_t), plan.indexer_physical_pages));
    plan.indexer_block_keys_bytes = checked_mul<std::size_t>(plan.qsa_cache_layers(), single_indexer_keys_plane);

    const std::size_t fixed_base_bytes = compute_fixed_base_bytes(
        config, resolved_state_slots, plan.attention_logical_pages, plan.indexer_logical_pages,
        plan.maximum_blocks, plan.block_tables_bytes, plan.recurrent_state_bytes,
        plan.round_tensors_bytes, plan.workspace_bytes);

    const std::size_t graph_allowance =
        config.use_cuda_graph
            ? checked_mul<std::size_t>(
                  kFlashNextDecodeGraphBytesPerCapture * (config.max_concurrency + config.speculative_draft_tokens) +
                      kFlashNextMtpDraftGraphBytesPerCapture * config.speculative_draft_tokens,
                  flash_next_decode_graph_buckets(plan.maximum_blocks).count)
            : 0ULL;
    plan.cuda_graph_allowance_bytes = graph_allowance;

    plan.total_device_bytes = checked_add(
        checked_add(plan.attention_kv_bytes, plan.indexer_block_keys_bytes),
        checked_add(fixed_base_bytes, graph_allowance));
    plan.capacity_curve = curve;

    if (config.vision_enabled) {
        const std::size_t decode_workspace =
            flash_next_text_decode_workspace_capacity_bytes(plan.maximum_blocks, std::max(config.max_concurrency, config.speculative_draft_tokens + 1U), config.speculative_draft_tokens > 0);
        const std::size_t prefill_workspace =
            flash_next_text_prefill_workspace_capacity_bytes(plan.maximum_blocks, config.prefill_chunk, config.speculative_draft_tokens > 0);
        const std::size_t general_workspace = std::max({decode_workspace, prefill_workspace, config.speculative_draft_tokens > 0 ? flash_next_mtp_workspace_capacity_bytes(plan.maximum_blocks, 1) : std::size_t{0}});
        const std::uint32_t merged =
            config.max_vision_tokens > 0 ? config.max_vision_tokens : 4096U;
        plan.vision_workspace =
            qwen3_vision::Encoder::plan_workspace(merged, general_workspace, 2560);
    }

    return plan;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
