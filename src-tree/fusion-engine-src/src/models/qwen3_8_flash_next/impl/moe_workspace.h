#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "models/qwen3_8_flash_next/impl/expert_bank.h"
#include "models/qwen3_8_flash_next/impl/moe_route.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Host-tiered expert staging (NINFER_FLASH_NEXT_EXPERTS_HOST). TWO different bounds are in play here
// and conflating them is a real bug, not a style question:
//
//   * kMaxStagedPaths bounds the per-call BOOKKEEPING, which is indexed by routed path: the pinned
//     ids/alpha/slot buffers, the per-path drop flags and the used-slot list. 10 top-k x tokens.
//   * kStagedExpertSlots bounds the DESTINATION PLANES, which are indexed by SLOT. It only sizes the
//     no-cache layout (NINFER_FLASH_NEXT_EXPERTS_SLOTS == 0, planes allocated in the workspace below);
//     with a slot pool the planes belong to the pool, and this constant only floors the per-slot
//     divisor staging plane.
//
// The distinction appeared with the wide MoE arm (NINFER_FLASH_NEXT_MOE_WIDE_ARM): an 8-token decode
// call needs at most 80 paths and the two bounds coincide, but a 32-token wave needs up to 320 paths
// against a pool of 128-172 slots. stage_flash_next_moe_experts throws rather than silently dropping
// an expert if a call ever asks for more of either. The per-expert byte counts are this target's
// compiled expert geometry (gate_up [512,1280,2560] / down [512,2560,640] at 4.5 bits: elements/2
// codes + elements/16 scales); the staging code re-checks them against the bound bank before copying.
inline constexpr std::int32_t kMaxStagedTokens        = 32;
inline constexpr std::int32_t kMaxStagedPaths         = kMaxStagedTokens * 10;
inline constexpr std::int32_t kStagedExpertSlots      = kFlashNextRouteDecodeMaxTokens * 10;
inline constexpr std::int32_t kStagedGateUpCodeBytes  = 1'638'400; // 1280*2560/2
inline constexpr std::int32_t kStagedGateUpScaleBytes = 204'800;   // 1280*2560/16
inline constexpr std::int32_t kStagedDownCodeBytes    = 819'200;   // 2560*640/2
inline constexpr std::int32_t kStagedDownScaleBytes   = 102'400;   // 2560*640/16
inline constexpr std::int32_t kStagedDivisorBytes     = 512 * 4;   // one float per expert

// Block-5 Stage-2 (gguf-block experts), all on the <=8-token decode domain the leaf enforces:
//   * paths a call can route: tokens x topk = 8 x 10 = 80, and the DISTINCT expert count is at
//     most that;
//   * worst per-expert block bytes across the artifact's six stored classes: gate_up 1,740,800 B
//     (IQ4_XS, 1,280 rows x 1,360 B) and down 921,600 B (IQ4_NL, 2,560 rows x 360 B).
inline constexpr std::int32_t kGgufStagedPathsMax  = kFlashNextRouteDecodeMaxTokens * 10;
inline constexpr std::uint64_t kGgufGateUpBytesMax = 1'740'800;
inline constexpr std::uint64_t kGgufDownBytesMax   = 921'600;

struct FlashNextMoeWorkspace {
    Tensor scores;
    Tensor ids;
    Tensor alpha;
    Tensor shared_scale;
    Tensor activations;
    // Prefill grouping & amortized GEMM workspace (tokens > 8)
    Tensor expert_counts;
    Tensor expert_offsets;
    Tensor active_expert_ids;
    Tensor active_count;
    Tensor grouped_tokens;
    Tensor grouped_paths;
    Tensor grouped_experts;
    Tensor token_to_pos;
    Tensor staged_down;
    Tensor down_intermediate;
    Tensor task_counter;
    // Prefill NVFP4 MMA activation quantization buffers (tokens > 8)
    Tensor act_codes;
    Tensor act_scales;
    Tensor down_act_codes;
    Tensor down_act_scales;
    // Packed BF16 [1280, T] = gate[640,T] | shared-act[640,T] for the MMA shared path.
    Tensor shared_gemm;
    // Host-tiered expert staging: this call's routed experts' codes/scales plus both divisor planes
    // (only the first <slots> floats of each are written -- divisors are gathered per slot because
    // the kernels index them with the same rewritten id as the codes). Allocated from the same
    // workspace arena (so the capacity function accounts for it and the call's scope releases it)
    // and deliberately separate from every trunk buffer -- sharing an x/y staging buffer with the
    // resident path is the aliasing that made Colibri emit silently wrong tokens.
    Tensor staged_gate_up_codes;
    Tensor staged_gate_up_scales;
    Tensor staged_down_codes;
    Tensor staged_down_scales;
    Tensor staged_gate_up_divisors;
    Tensor staged_down_divisors;
    // Block-6 W2: the CPU-serve forward face's partial plane [2'560, tokens] BF16 -- the H2D
    // destination the served experts' contribution is added into `output` from. Allocated only
    // when the face is armed (serve master switch on), so every other arm's layout and envelope
    // are unchanged.
    Tensor cpu_serve_partial;
    // Block-5 Stage-2 gguf expert leaf (allocated only under NINFER_FLASH_NEXT_GGUF_EXPERTS, so
    // every other arm's layout is unchanged): per-call raw-block staging for up to 80 distinct
    // experts at the worst per-expert sizes, the routed f32 plane ([2560, paths]), and the leaf's
    // small intermediates (gather/h, the shared expert's g/u/h/out, the q8 activation for the
    // K=640 down product, and gguf_project's own transient).
    Tensor gguf_stage_gate_up;
    Tensor gguf_stage_down;
    Tensor gguf_routed;
    Tensor gguf_path_src;
    Tensor gguf_token_ids;
    Tensor gguf_gather;
    Tensor gguf_h;
    Tensor gguf_shared_g;
    Tensor gguf_shared_u;
    Tensor gguf_shared_h;
    Tensor gguf_shared_out;
    Tensor gguf_down_act;
    Tensor gguf_proj_workspace;
};

template <class Arena>
FlashNextMoeWorkspace allocate_flash_next_moe_workspace(Arena& arena, std::int32_t tokens) {
    FlashNextMoeWorkspace out;
    out.scores       = arena.alloc(DType::FP32, {513, tokens}, 256);
    out.ids          = arena.alloc(DType::I32, {10, tokens}, 16);
    out.alpha        = arena.alloc(DType::FP32, {10, tokens}, 16);
    out.shared_scale = arena.alloc(DType::FP32, {tokens}, 16);
    // Ten routed SwiGLU paths followed by the always-on shared path.
    out.activations = arena.alloc(DType::BF16, {640, 11, tokens}, 256);

    // With the LRU slot pool (NINFER_FLASH_NEXT_EXPERTS_SLOTS > 0) the persistent pool is the
    // staging destination, so these per-call planes are dead weight and are not allocated.
    if (flash_next_experts_on_host_enabled() && flash_next_experts_slot_count() == 0) {
        out.staged_gate_up_codes =
            arena.alloc(DType::U8, {kStagedExpertSlots * kStagedGateUpCodeBytes}, 256);
        out.staged_gate_up_scales =
            arena.alloc(DType::U8, {kStagedExpertSlots * kStagedGateUpScaleBytes}, 256);
        out.staged_down_codes = arena.alloc(DType::U8, {kStagedExpertSlots * kStagedDownCodeBytes}, 256);
        out.staged_down_scales =
            arena.alloc(DType::U8, {kStagedExpertSlots * kStagedDownScaleBytes}, 256);
        out.staged_gate_up_divisors = arena.alloc(DType::U8, {kStagedDivisorBytes}, 16);
        out.staged_down_divisors    = arena.alloc(DType::U8, {kStagedDivisorBytes}, 16);
    }

    // Block-6 W2: the CPU-serve partial plane; see the member's note. 2'560 is this target's
    // compiled hidden, the same constant every other plane in this function uses.
    if (flash_next_cpu_serve_forward_enabled()) {
        out.cpu_serve_partial = arena.alloc(DType::BF16, {2'560, tokens}, 256);
    }

    if (tokens > 8) {
        out.expert_counts     = arena.alloc(DType::I32, {512}, 16);
        out.expert_offsets    = arena.alloc(DType::I32, {513}, 16);
        out.active_expert_ids = arena.alloc(DType::I32, {512}, 16);
        out.active_count      = arena.alloc(DType::I32, {1}, 16);
        out.grouped_tokens    = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.grouped_paths     = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.grouped_experts   = arena.alloc(DType::I32, {10 * tokens}, 16);
        out.token_to_pos      = arena.alloc(DType::I32, {10 * tokens}, 16);
        if (tokens >= 512) {
            // MMA arm: BF16 staged routed outputs [2560, 10 * tokens], reduced in fixed order.
            // The envelope is computed by calling this with the chunk capacity, but a tail chunk
            // of 9..511 tokens takes the SIMT arm and carves an FP32 [2560, 10, T] intermediate,
            // which is larger than the BF16 staging when tokens < 1022. Pad the staging so the
            // envelope covers the worst SIMT tail (T = 511) at every capacity.
            constexpr std::int32_t kSimtTailColumns = 10 * 511 * 2; // FP32 [2560,10,511] in BF16 columns
            const std::int32_t staged_columns       = std::max(10 * tokens, kSimtTailColumns);
            out.staged_down   = arena.alloc(DType::BF16, {2'560, staged_columns}, 256);
        } else {
            out.down_intermediate = arena.alloc(DType::FP32, {2'560, 10, tokens}, 256);
        }
        out.task_counter      = arena.alloc(DType::I32, {4}, 16);
        out.act_codes         = arena.alloc(DType::U8, {1'280, tokens}, 256);
        out.act_scales        = arena.alloc(DType::U8, {160, tokens}, 256);
        out.down_act_codes    = arena.alloc(DType::U8, {320, 11 * tokens}, 256);
        out.down_act_scales   = arena.alloc(DType::U8, {40, 11 * tokens}, 256);
        out.shared_gemm       = arena.alloc(DType::BF16, {640, 2 * tokens}, 256);
    }

    // Block-5 Stage-2: the gguf expert leaf's planes. Gated on the same read-once switch the load
    // face and the leaf consult, so a non-gguf or OFF process allocates exactly what it always did.
    if (flash_next_gguf_experts_enabled()) {
        out.gguf_stage_gate_up = arena.alloc(
            DType::U8, {static_cast<std::int32_t>(kGgufStagedPathsMax * kGgufGateUpBytesMax)}, 256);
        out.gguf_stage_down = arena.alloc(
            DType::U8, {static_cast<std::int32_t>(kGgufStagedPathsMax * kGgufDownBytesMax)}, 256);
        out.gguf_routed     = arena.alloc(DType::FP32, {2'560, kGgufStagedPathsMax}, 256);
        out.gguf_path_src   = arena.alloc(DType::I32, {kGgufStagedPathsMax}, 16);
        out.gguf_token_ids  = arena.alloc(DType::I32, {kFlashNextRouteDecodeMaxTokens}, 16);
        out.gguf_gather     = arena.alloc(DType::BF16, {2'560, kFlashNextRouteDecodeMaxTokens}, 256);
        out.gguf_h          = arena.alloc(DType::BF16, {640, kFlashNextRouteDecodeMaxTokens}, 256);
        out.gguf_shared_g   = arena.alloc(DType::BF16, {640, kFlashNextRouteDecodeMaxTokens}, 256);
        out.gguf_shared_u   = arena.alloc(DType::BF16, {640, kFlashNextRouteDecodeMaxTokens}, 256);
        out.gguf_shared_h   = arena.alloc(DType::BF16, {640, kFlashNextRouteDecodeMaxTokens}, 256);
        out.gguf_shared_out = arena.alloc(DType::BF16, {2'560, kFlashNextRouteDecodeMaxTokens}, 256);
        out.gguf_down_act = arena.alloc(
            DType::U8,
            {static_cast<std::int32_t>(
                ops::gguf::vector_activation_bytes(640, kFlashNextRouteDecodeMaxTokens))},
            256);
        // gguf_swiglu's own transient, worst over the four gate/up classes the artifact stores
        // (computed through the same helper the call site uses, so capacity cannot drift).
        std::size_t proj_ws = 0;
        for (const QType qtype :
             {QType::GGUF_IQ2_S, QType::GGUF_IQ3_XXS, QType::GGUF_IQ3_S, QType::GGUF_IQ4_XS}) {
            proj_ws = std::max(proj_ws, ops::detail::gguf_swiglu_workspace_bytes(
                                            ops::detail::GgufShape{qtype, 1'280, 2'560}, nullptr, 1,
                                            kFlashNextRouteDecodeMaxTokens));
        }
        out.gguf_proj_workspace =
            arena.alloc(DType::U8, {static_cast<std::int32_t>(proj_ws)}, 256);
    }
    return out;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
