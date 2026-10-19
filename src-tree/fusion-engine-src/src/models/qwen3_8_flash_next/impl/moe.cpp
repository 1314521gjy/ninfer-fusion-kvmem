#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>


#include "artifact/formats.h"
#include "core/layout.h"
#include "models/qwen3_8_flash_next/impl/expert_staging.h"
#include "models/qwen3_8_flash_next/impl/moe_kernels.h"
#include "models/qwen3_8_flash_next/impl/moe_route.h"
#include "models/qwen3_8_flash_next/impl/moe_workspace.h"
#include "models/qwen3_8_flash_next/impl/stage_ledger.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/device.h"

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool exact_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    return weight.qtype == QType::BF16 && weight.layout == QuantLayout::Contiguous &&
           weight.n == rows && weight.k == columns && weight.ndim == 2 && weight.shape[0] == rows &&
           weight.shape[1] == columns && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.qdata == weight.payload &&
           weight.payload_bytes >= static_cast<std::uint64_t>(rows) * columns * 2 &&
           aligned_to(weight.qdata, 16);
}

// ---- wide prefill arm -------------------------------------------------------------------------
// The wide arm (tokens > 8, "Group tokens by expert, load weights once per chunk") is FULLY
// IMPLEMENTED in moe_kernels.cu. Two things kept it unreachable on sm_89 + host tiering:
//   (a) exact_expert_bank below demands `experts == 512`, but host tiering hands the MoE the SLOT
//       POOL view instead (expert_staging.cpp rewrites the router ids to SLOT indices and builds a
//       view with experts = slots, on_host = false). Semantically that is exactly what the wide arm
//       wants: it indexes weights by "expert id", and after the rewrite that id IS the slot.
//       The grouped workspace tables are sized for 512 ids (moe_workspace.h:90-92), so slot ids
//       (< 172) index them safely.
//   (b) the 8-token slice cap in flash_next_moe_impl, which is a DECODE width
//       (kFlashNextRouteDecodeMaxTokens is sized for MTP verify batches).
// NINFER_FLASH_NEXT_MOE_WIDE_ARM=1 lifts both, so a wave of moe_wave_tokens() tokens reaches the
// wide arm in one call instead of ceil(wave/8) decode-shaped calls. Default OFF: with it off every
// path is byte-for-byte as before.
bool moe_wide_arm_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_WIDE_ARM");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

// Wave width. The bound that actually decides everything is DISTINCT EXPERTS PER WAVE, because a wave
// must fit the device slot pool in ONE call.
//
// The estimate this was designed from is FALSIFIED. scripts/prefill_byte_bound.py read the routing
// trace and predicted ~117 distinct at 32 tokens ("so 32 fits even a 128-slot pool"). The first real
// run at width 32 asked for 181 on a fresh ~130-token prompt against a 172-slot pool, i.e. the trace
// understated it by 55%, and the wave did NOT fit. Treat 117/146/208 as unusable: the real prompt, not
// the trace, is the instrument. Two things make the first chunk the worst case: a fresh prompt routes
// cold, and its first wave has no resident experts to reuse.
//
// So the width is a RUNTIME knob, not a constant: NINFER_FLASH_NEXT_MOE_WAVE selects 8..32 (clamped;
// 8 == the decode width == "no widening"). Scanning widths then costs one engine run each instead of
// one recompile each. Whatever width is chosen must still survive the distinct-expert guard in
// expert_staging.cpp -- which is what makes a wrong guess a loud refusal instead of silent corruption.
std::int32_t moe_wave_tokens() {
    static const std::int32_t width = [] {
        constexpr std::int32_t fallback = kMaxStagedTokens;
        const char* env                 = std::getenv("NINFER_FLASH_NEXT_MOE_WAVE");
        if (env == nullptr || env[0] == '\0') { return fallback; }
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed <= kFlashNextRouteDecodeMaxTokens) { return kFlashNextRouteDecodeMaxTokens; }
        if (parsed > kMaxStagedTokens) { return kMaxStagedTokens; }
        return static_cast<std::int32_t>(parsed);
    }();
    return width;
}

bool exact_expert_bank(const Nvfp4ExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    const bool experts_ok = moe_wide_arm_enabled() ? (bank.experts >= 1) : (bank.experts == 512);
    return experts_ok && bank.rows == rows && bank.columns == columns &&
           bank.code_bytes_per_expert == elements / 2 &&
           bank.scale_bytes_per_expert == elements / 16 && aligned_to(bank.codes, 16) &&
           aligned_to(bank.scales, 16) && aligned_to(bank.weight_scale_divisors, 16);
}

bool exact_bf16_expert_bank(const Bf16ExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * columns;
    return bank.experts == 512 && bank.rows == rows && bank.columns == columns &&
           bank.bytes_per_expert == elements * sizeof(std::uint16_t) &&
           aligned_to(bank.data, 16);
}

} // namespace

std::size_t flash_next_moe_workspace_capacity_bytes(std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Flash-Next MoE workspace requires positive tokens");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_moe_workspace(layout, max_tokens);
    return layout.peak_bytes(256);
}

void flash_next_moe_impl(const Tensor& input, const MoeWeights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream,
                         std::int32_t slice_width);

// The public entry forwards with the configured slice width. Keeping the body in a separate function
// is what turns the wave width into a parameter instead of a compile-time cap, WITHOUT touching any
// of the arithmetic below.
void flash_next_moe(const Tensor& input, const MoeWeights& weights, Tensor& output,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    // The master switch stays here, at the single point where >8 tokens can enter the tree: with it
    // off the slice is the decode width and nothing below can widen, whatever the width knob says.
    flash_next_moe_impl(input, weights, output, workspace, stream,
                        moe_wide_arm_enabled() ? moe_wave_tokens()
                                               : kFlashNextRouteDecodeMaxTokens);
}

// ================================================================================================
// Block-5 Stage-2: gguf-block expert compute (NINFER_FLASH_NEXT_GGUF_EXPERTS artifacts).
//
// WHY: the IQ3_S artifact stores the 48 text layers' routed experts as whole ggml block rows,
// which the nvfp4 kernels cannot read. This leaf computes the same MoE with the gguf kernels:
// route -> dedup (host) -> stage raw block rows (device) -> per-expert gate/up SwiGLU -> per-expert
// down -> one combine reproducing the 11-path semantics (10 routed paths at alpha, plus the
// sigmoid-gated BF16 shared expert through the generic ops).
//
// SCOPE (first pass, deliberately simple; a later round adds slot pooling):
//   * the <=8-token slice is the wired domain; the wide arm is refused loudly (its wave re-entry
//     needs pooled planes and a device-side dedup);
//   * per-call staging: each distinct expert (<=80) is copied once into a per-call plane;
//   * the host dedup costs one ids D2H+sync per call -- fine at decode width, not a prefill path;
//   * nvfp4-only policies (CPU expert serve, drop/subst) are refused loudly both here and at the
//     load face (block-5 ruling: refuse, never silently ignore).
// Geometry note: the gate_up bank is [gate; up] per expert -- exactly the fused parent form
// `gguf_swiglu` takes with up == nullptr. The down product's K = 640 fails `require_gguf`'s
// K % 512 gate, so it goes through the bridge's vector product directly; the vector kernel itself
// only needs K % block-elements == 0, and 640 is whole 32/64-element blocks. The activation
// quantizer's gate is K % 32 in this change (see quantize_vector_activation).
// Determinism: host bookkeeping is path-order deterministic (path = token * topk + rank), the
// kernels are deterministic, and the combine accumulates rank 0..9 then the shared term.
// ================================================================================================

namespace {

constexpr std::int32_t kGgufTopK = 10;

bool gguf_banks_ok(const MoeWeights& weights) {
    const auto ok = [](const GgufExpertBankView& bank, std::int32_t rows, std::int32_t columns) {
        if (bank.data == nullptr || !is_gguf(bank.qtype) || bank.experts != 512 ||
            bank.rows != rows || bank.columns != columns) {
            return false;
        }
        const GgufBlockShape block = gguf_block_shape(bank.qtype);
        return block.elements > 0 &&
               bank.row_bytes == static_cast<std::uint64_t>(columns / block.elements) *
                                    static_cast<std::uint64_t>(block.bytes);
    };
    return ok(weights.gguf_gate_up, 1'280, 2'560) && ok(weights.gguf_down, 2'560, 640);
}

// Block-5 ruling: an nvfp4-only service policy next to a gguf bank refuses loudly (the load face
// already fails fast; this is the arithmetic point, so a future load-path change cannot bypass it).
void require_no_nvfp4_only_policies() {
    if (const char* conflict = flash_next_gguf_policy_conflict(); conflict != nullptr) {
        throw std::invalid_argument(
            std::string("Flash-Next gguf experts: ") + conflict +
            " is set, and this stage does not wire that policy for gguf blocks; unset it "
            "(refused loudly rather than silently ignored)");
    }
}

} // namespace

void flash_next_moe_gguf_experts(const Tensor& input, const MoeWeights& weights, Tensor& output,
                                 FlashNextMoeWorkspace& scratch, WorkspaceArena& workspace,
                                 cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    require_no_nvfp4_only_policies();
    if (tokens > kFlashNextRouteDecodeMaxTokens) {
        throw std::invalid_argument(
            "Flash-Next gguf experts: the wide arm is not wired in this stage; keep "
            "NINFER_FLASH_NEXT_MOE_WIDE_ARM off for a gguf artifact");
    }
    const GgufExpertBankView& gu = weights.gguf_gate_up;
    const GgufExpertBankView& dn = weights.gguf_down;
    const std::uint64_t gu_bytes = gu.row_bytes * static_cast<std::uint64_t>(gu.rows);
    const std::uint64_t dn_bytes = dn.row_bytes * static_cast<std::uint64_t>(dn.rows);
    const std::int32_t paths    = tokens * kGgufTopK;

    // One line per process: the direct evidence that the gguf MoE leaf ran, with the formats the
    // container declared. Default-off runs and non-gguf artifacts never reach here.
    static bool gguf_moe_reported = false;
    if (!gguf_moe_reported) {
        gguf_moe_reported = true;
        const std::string_view gu_name = artifact::format_name(gu.qtype);
        const std::string_view dn_name = artifact::format_name(dn.qtype);
        std::fprintf(stderr,
                     "flash_next gguf_moe_experts gate_up_format=%.*s down_format=%.*s "
                     "tokens_cap=%d\n",
                     static_cast<int>(gu_name.size()), gu_name.data(),
                     static_cast<int>(dn_name.size()), dn_name.data(),
                     kFlashNextRouteDecodeMaxTokens);
    }

    // 1. The routed (token, rank) pairs, host-side. path = token * 10 + rank is the ids/alpha
    //    linear index (moe_kernels.cu's decode kernels read ids[token * kTopK + path] the same way).
    std::int32_t host_ids[kFlashNextRouteDecodeMaxTokens * kGgufTopK]{};
    cudaMemcpyAsync(host_ids, scratch.ids.data, std::size_t(paths) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 2. Distinct experts (first-seen order over ascending path) and the per-path column map.
    std::int32_t distinct[kGgufStagedPathsMax]{};
    std::int32_t path_slot[kGgufStagedPathsMax]{};
    std::int32_t distinct_count = 0;
    for (std::int32_t p = 0; p < paths; ++p) {
        const std::int32_t expert = host_ids[p];
        std::int32_t slot         = 0;
        while (slot < distinct_count && distinct[slot] != expert) { ++slot; }
        if (slot == distinct_count) { distinct[distinct_count++] = expert; }
        path_slot[p] = slot;
    }
    // Packed path lists: expert `slot` owns routed columns [slot_begin[slot], slot_begin[slot+1]).
    std::int32_t slot_begin[kGgufStagedPathsMax + 1]{};
    std::int32_t packed[kGgufStagedPathsMax]{};
    std::int32_t host_path_src[kGgufStagedPathsMax]{};
    for (std::int32_t slot = 0; slot < distinct_count; ++slot) {
        slot_begin[slot + 1] = slot_begin[slot];
        for (std::int32_t p = 0; p < paths; ++p) {
            if (path_slot[p] != slot) { continue; }
            const std::int32_t column = slot_begin[slot + 1]++;
            packed[column]            = p;
            host_path_src[p]          = column * 2'560;
        }
    }
    auto* const path_src = static_cast<std::int32_t*>(scratch.gguf_path_src.data);
    cudaMemcpyAsync(path_src, host_path_src, std::size_t(paths) * sizeof(std::int32_t),
                    cudaMemcpyHostToDevice, stream);

    // 3. Stage each distinct expert's raw block rows (pageable read from the container mapping or
    //    the pinned bank; NINFER_FLASH_NEXT_EXPERTS_PIN makes this the fast path).
    auto* const gu_plane = static_cast<std::uint8_t*>(scratch.gguf_stage_gate_up.data);
    auto* const dn_plane = static_cast<std::uint8_t*>(scratch.gguf_stage_down.data);
    for (std::int32_t slot = 0; slot < distinct_count; ++slot) {
        const std::uint64_t expert = static_cast<std::uint64_t>(distinct[slot]);
        cudaMemcpyAsync(gu_plane + static_cast<std::uint64_t>(slot) * gu_bytes,
                        gu.data + expert * gu_bytes, gu_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(dn_plane + static_cast<std::uint64_t>(slot) * dn_bytes,
                        dn.data + expert * dn_bytes, dn_bytes, cudaMemcpyHostToDevice, stream);
    }

    // 4. Per-expert products over that expert's path columns (chunks of <= 8, the vector domain).
    auto* const routed    = static_cast<float*>(scratch.gguf_routed.data);
    auto* const token_ids = static_cast<std::int32_t*>(scratch.gguf_token_ids.data);
    auto* const gather    = static_cast<__nv_bfloat16*>(scratch.gguf_gather.data);
    auto* const h_buf     = static_cast<__nv_bfloat16*>(scratch.gguf_h.data);
    auto* const down_act  = scratch.gguf_down_act.data;

    const auto make_gguf_weight = [](const void* data, QType qtype, std::uint64_t row_bytes,
                                     std::int32_t rows, std::int32_t columns) {
        Weight w{};
        w.payload       = data;
        w.qdata         = data;
        w.qtype         = qtype;
        w.layout        = QuantLayout::GgufBlocks;
        w.n             = rows;
        w.k             = columns;
        w.ndim          = 2;
        w.shape[0]      = rows;
        w.shape[1]      = columns;
        w.padded_shape[0] = rows;
        w.padded_shape[1] = columns;
        w.payload_bytes = row_bytes * static_cast<std::uint64_t>(rows);
        return w;
    };

    for (std::int32_t slot = 0; slot < distinct_count; ++slot) {
        const std::int32_t count = slot_begin[slot + 1] - slot_begin[slot];
        for (std::int32_t base = 0; base < count; base += 8) {
            const std::int32_t cols = std::min(8, count - base);
            std::int32_t chunk_token[8]{};
            for (std::int32_t j = 0; j < cols; ++j) {
                chunk_token[j] = packed[slot_begin[slot] + base + j] / kGgufTopK;
            }
            cudaMemcpyAsync(token_ids, chunk_token, std::size_t(cols) * sizeof(std::int32_t),
                            cudaMemcpyHostToDevice, stream);
            // The expert's token columns, then gate/up through the fused parent form.
            flash_next_moe_gguf_gather(static_cast<const __nv_bfloat16*>(input.data), token_ids,
                                       2'560, cols, gather, stream);
            Tensor x_view(gather, DType::BF16, {2'560, cols});
            Tensor h_view(h_buf, DType::BF16, {640, cols});
            const Weight gu_w =
                make_gguf_weight(gu_plane + static_cast<std::uint64_t>(slot) * gu_bytes, gu.qtype,
                                 gu.row_bytes, gu.rows, gu.columns);
            ops::detail::gguf_swiglu(x_view, gu_w, nullptr, h_view, workspace, stream);
            // Down: bridge-level, because K = 640 fails `require_gguf`'s K % 512 gate while the
            // vector kernel only needs whole blocks (see the section comment).
            ops::gguf::quantize_vector_activation(h_buf, 640, cols, nullptr, down_act, stream);
            const Weight dn_w =
                make_gguf_weight(dn_plane + static_cast<std::uint64_t>(slot) * dn_bytes, dn.qtype,
                                 dn.row_bytes, dn.rows, dn.columns);
            ops::gguf::VectorOutput down_out{};
            down_out.f32 = routed + static_cast<std::int64_t>(slot_begin[slot] + base) * 2'560;
            down_out.column_stride = 2'560;
            ops::gguf::vector_product(ops::detail::gguf_type(dn.qtype), dn_w.qdata,
                                      static_cast<std::int64_t>(dn.row_bytes), dn.rows, 640,
                                      down_act, cols, down_out, stream);
        }
    }

    // 5. Shared expert, BF16 through the generic ops (its gate/up/down are plain BF16 weights; the
    //    router's `shared_scale` is the sigmoid gate, applied in the combine).
    auto* const shared_g   = static_cast<__nv_bfloat16*>(scratch.gguf_shared_g.data);
    auto* const shared_u   = static_cast<__nv_bfloat16*>(scratch.gguf_shared_u.data);
    auto* const shared_h   = static_cast<__nv_bfloat16*>(scratch.gguf_shared_h.data);
    auto* const shared_out = static_cast<__nv_bfloat16*>(scratch.gguf_shared_out.data);
    Tensor g_view(shared_g, DType::BF16, {640, tokens});
    Tensor u_view(shared_u, DType::BF16, {640, tokens});
    Tensor sh_view(shared_h, DType::BF16, {640, tokens});
    Tensor so_view(shared_out, DType::BF16, {2'560, tokens});
    ops::linear(input, weights.shared_gate, g_view, stream);
    ops::linear(input, weights.shared_up, u_view, stream);
    flash_next_moe_gguf_silu_mul(shared_g, shared_u, shared_h, 640 * tokens, stream);
    ops::linear(sh_view, weights.shared_down, so_view, stream);

    // 6. The 11-path combine.
    flash_next_moe_gguf_combine(routed, path_src, paths,
                                static_cast<const float*>(scratch.alpha.data),
                                static_cast<const float*>(scratch.shared_scale.data), shared_out,
                                static_cast<__nv_bfloat16*>(output.data), tokens, stream);
}

void flash_next_moe_impl(const Tensor& input, const MoeWeights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream,
                         const std::int32_t slice_width) {
    const std::int32_t tokens = input.ne[1];

#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4)
    // THIS TREE'S MACRO, not the upstream spelling. The block below was ported from a tree where
    // the gate is `NINFER_SM89`, and that macro is defined nowhere here (measured: 0 hits in
    // `build/compile_commands.json`, while the same file carries `NINFER_SM8X_COMPAT=1` on all 738
    // entries). The two name the same fact -- "this architecture has no Blackwell block-scale NVFP4
    // MMA" -- and this tree states it once, in `CMakeLists.txt:33-38`, for 80/86/89 and for the
    // 120a compatibility path. Defining `NINFER_SM89` as well would be a second source of truth for
    // one fact, so the port follows this tree's spelling instead -- including the `NINFER_SM120_NVFP4`
    // half, which is this tree's own way of saying "a 120a build that keeps the NVFP4 units does have
    // this route" (the same conjunction is used by `ops/wrapper/sparse_moe.cpp:267`,
    // `ops/weight_input.cpp:338` and `ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu:1732`).
    // That makes the gate equivalent to upstream's `NINFER_SM89` on both sm_89 and 120a.
    //
    // The consequence of the old dead gate: the slice below never ran, every prefill chunk reached
    // `flash_next_moe_kernels_launch` unsliced, its `tokens >= 512` arm was left reachable, and on
    // sm_89 that arm ends in `ops::mma_nvfp4_e4m3`, whose `__CUDA_ARCH__ < 1000` body is a plain
    // `__trap()` (ops/common/mma.cuh:118). See `docs/20 §16.9`.
    //
    // sm_89 port, Stage B: the >= 512 arm is the Blackwell block-scaled NVFP4 MMA (kind::mxf4nvf4,
    // no Ada equivalent), and the 8..511 arm indexes expert weights by id -- which with host tiering
    // means the SLOT POOL view that exact_expert_bank now accepts under MOE_WIDE_ARM. A request
    // wider than the wave width is still re-entered in wave-sized slices, because a wave is what the
    // slot pool can hold: on the real routing trace a 32-token wave asks for ~117 DISTINCT experts
    // and a 128-token wave for ~208, while the pool has 128 slots. With the switch OFF slice_width is
    // 8, so this is the decode-shaped slicing, byte-for-byte as before.
    if (tokens > slice_width) {
        for (std::int32_t offset = 0; offset < tokens; offset += slice_width) {
            const std::int32_t count = std::min(slice_width, tokens - offset);
            // Non-owning views: slice() returns by value, so the output view needs a name to bind
            // to the (non-const) Tensor& the leaf takes.
            Tensor output_slice = output.slice(1, offset, count);
            flash_next_moe_impl(input.slice(1, offset, count), weights, output_slice, workspace,
                                stream, slice_width);
        }
        return;
    }
#endif

    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) || !exact_bf16_weight(weights.router, 512, 2'560) ||
        !exact_bf16_weight(weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_gate_weight, 1, 2'560) ||
        // Block-5: a gguf bank is validated against its own contract (bit-exact rows per its
        // format), not the nvfp4 bank geometry -- and vice versa.
        !(weights.experts_gguf
              ? gguf_banks_ok(weights)
              : (exact_expert_bank(weights.expert_gate_up, 1'280, 2'560) &&
                 exact_expert_bank(weights.expert_down, 2'560, 640))) ||
        stream == nullptr) {
        throw std::invalid_argument("Flash-Next MoE received an invalid exact target view");
    }
    const auto scope              = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    flash_next_route(input, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                     scratch.alpha, scratch.shared_scale, stream);
    stage_ledger_record(stream, FlashNextStageId::MoE_Router);

    // Route-dump probe (diagnostic only, env-gated, default off). Records the RAW router output --
    // i.e. real expert ids, before stage_flash_next_moe_experts() rewrites them in place into slot
    // numbers. The router tensor's data pointer tags each record with its layer, so the MTP layer
    // (a different router) is distinguishable from the 48 main layers.
    //
    // Record layout (binary, little endian): [router_ptr u64][tokens i32][topk i32][ids i32 * tokens*topk]
    // This exists so that every routing question becomes an offline CPU analysis instead of a GPU run:
    // per-layer hit-rate distribution, expert-union size per verification width, real reuse distance,
    // and the top-k coverage of a one-layer-lookahead prefetch.
    if (const char* route_dump_path = std::getenv("NINFER_FLASH_NEXT_ROUTE_DUMP")) {
        // FIXED 2026-10-05 -- ORDERING BUG THAT VOIDED EVERY ROUTING STATISTIC TAKEN BEFORE TODAY.
        //
        // The four cudaMemcpy() calls in this block are SYNCHRONOUS, so they run on the DEFAULT
        // stream; this engine runs on a cudaStreamNonBlocking stream. Those two streams share no
        // implicit ordering, so the copies routinely completed BEFORE flash_next_route() above had
        // written scratch.ids / scratch.alpha / scratch.scores. The dump then captured whatever the
        // PREVIOUS call's staging had left in those buffers -- slot numbers, not expert ids.
        //
        // Symptoms recorded in docs/娴嬭瘯鍙拌处-鍋氳繃浠€涔堜笌缁撴灉-20260928.md 搂4: the ids histogram is
        // 99.9% packed into 0..191 and FLAT (real routing is a spiky long tail spread over 512);
        // every layer's apparent "working set" exactly equals the pool size; and the .scores
        // sidecar is byte-identical between two different records.
        //
        // CONSEQUENCE: every number ever derived from these dumps is void -- expert union size,
        // reuse distance, working set, and scripts/prefill_byte_bound.py's "~117 at k=4" estimate.
        // The only trustworthy routing figure we have is the runtime capacity gate's own 181.
        //
        // One cudaStreamSynchronize here is sufficient AND minimal: it drains the engine stream, so
        // all four blocking copies below read final values. No restructuring of the probe needed.
        // This probe is diagnostic-only and env-gated, so the extra sync costs nothing in normal runs.
        cudaStreamSynchronize(stream);
        static std::FILE* route_dump_file = nullptr;
        if (route_dump_file == nullptr) {
            route_dump_file = std::fopen(route_dump_path, "wb");
        }
        if (route_dump_file != nullptr) {
            constexpr std::int32_t kDumpTopK = 10;
            std::int32_t host_ids[8 * kDumpTopK];
            const std::size_t count =
                static_cast<std::size_t>(tokens) * static_cast<std::size_t>(kDumpTopK);
            cudaMemcpy(host_ids, scratch.ids.data, count * sizeof(std::int32_t),
                       cudaMemcpyDeviceToHost);
            const std::uint64_t router_tag =
                static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(weights.router.qdata));
            const std::int32_t header[2] = {tokens, kDumpTopK};
            std::fwrite(&router_tag, sizeof(router_tag), 1, route_dump_file);
            std::fwrite(header, sizeof(std::int32_t), 2, route_dump_file);
            std::fwrite(host_ids, sizeof(std::int32_t), count, route_dump_file);

            // Parallel alpha sidecar, <path>.alpha, written with the SAME record framing so a record
            // index pairs the two files 1:1. alpha[path + token*kDumpTopK] is that route's
            // renormalized top-10 probability -- i.e. exactly the routing weight whose loss we must
            // bound before dropping (miss-gated) or substituting (margin) an expert.
            //
            // Written as a SEPARATE file on purpose: the ids file's format is depended on by
            // scripts/analyze_routes.py and by the 34 MB routes.bin already on disk, and this probe
            // must not invalidate either. Cost is one extra 10*tokens-float D2H per layer call.
            static std::FILE* route_dump_alpha_file = nullptr;
            if (route_dump_alpha_file == nullptr) {
                char alpha_path[4096];
                std::snprintf(alpha_path, sizeof(alpha_path), "%s.alpha", route_dump_path);
                route_dump_alpha_file = std::fopen(alpha_path, "wb");
            }
            if (route_dump_alpha_file != nullptr) {
                float host_alpha[8 * kDumpTopK];
                cudaMemcpy(host_alpha, scratch.alpha.data, count * sizeof(float),
                           cudaMemcpyDeviceToHost);
                std::fwrite(&router_tag, sizeof(router_tag), 1, route_dump_alpha_file);
                std::fwrite(header, sizeof(std::int32_t), 2, route_dump_alpha_file);
                std::fwrite(host_alpha, sizeof(float), count, route_dump_alpha_file);
                std::fflush(route_dump_alpha_file);
            }

            // Third sidecar, <path>.scores: the FULL gate score row, [router_ptr][tokens][rows=513]
            // [scores f32 * 513*tokens]. Row 512 is the shared-expert gate, so real experts are 0..511.
            //
            // CORRECTED 2026-09-28 -- READ THIS BEFORE TRUSTING THE OLD TEXT.
            // This block used to say that `scratch.alpha` does NOT hold the renormalized top-10
            // weights ("its values run from -5.8 to +15.4 with a mean of 0.0098"). That is FALSE for
            // this build. It was read as a specification by a later session, which then built a
            // "threshold on the true weight" change on top of it, double-softmaxed an already
            // normalised probability, flattened the distribution, and had to revert the whole thing.
            // Measured on logs/routes2.bin (see scripts/admission_policy_screen.py):
            //   * per-token sum of alpha is 1.0            (mean |sum-1| = 0.0015 at width 8)
            //   * alpha == softmax(scores[ids]) exactly for 92.3% of prefill tokens
            //   * the ALPHA sidecar comment above already stated this correctly, so this file
            //     contradicted itself
            // => alpha IS the renormalised top-10 weight, and the R1 drop threshold in
            //    expert_staging.cpp already acts on it. There is nothing to "fix" there.
            //
            // The scores row is still worth dumping, for a DIFFERENT reason: it is the only way to
            // audit the router itself. It exposed one anomaly that is STILL OPEN: at decode width
            // (tokens == 3) alpha matches softmax(scores[ids]) for only 69.0% of samples and contains
            // values in [-2.05, 1.89] -- impossible for a renormalised weight -- while the per-token
            // sum is still ~1 (mean |sum-1| = 0.0025). R1 is armed exactly at that width and is
            // empirically fine (GSM8K 12/15 == baseline), so this is not a proven bug; it means the
            // decode-domain threshold may act on a quantity we have not characterised. Start here.
            // Row 512 is the shared-expert gate, so real experts are rows 0..511.
            static std::FILE* route_dump_scores_file = nullptr;
            if (route_dump_scores_file == nullptr) {
                char scores_path[4096];
                std::snprintf(scores_path, sizeof(scores_path), "%s.scores", route_dump_path);
                route_dump_scores_file = std::fopen(scores_path, "wb");
            }
            if (route_dump_scores_file != nullptr) {
                constexpr std::int32_t kDumpScoreRows = 513;
                static float host_scores[8 * kDumpScoreRows];
                const std::size_t score_count =
                    static_cast<std::size_t>(kDumpScoreRows) * static_cast<std::size_t>(tokens);
                cudaMemcpy(host_scores, scratch.scores.data, score_count * sizeof(float),
                           cudaMemcpyDeviceToHost);
                const std::int32_t score_header[2] = {tokens, kDumpScoreRows};
                std::fwrite(&router_tag, sizeof(router_tag), 1, route_dump_scores_file);
                std::fwrite(score_header, sizeof(std::int32_t), 2, route_dump_scores_file);
                std::fwrite(host_scores, sizeof(float), score_count, route_dump_scores_file);
                std::fflush(route_dump_scores_file);
            }

            // Hidden states, [hidden=2560, tokens] BF16 contiguous: this layer's MoE input. Dumped so
            // a one-layer-lookahead predictor can be evaluated OFFLINE on real data -- apply layer
            // L+1's router (extracted from the artifact) to layer L's hidden state and compare its
            // top-k against layer L+1's REAL ids, which the next record carries. That comparison is
            // the only thing that decides whether a predictive prefetch can pay: the transfer to hide
            // is 0.64 ms per layer inside a 1.79 ms compute window, so the coverage number is the
            // whole question.
            constexpr std::int32_t kDumpHidden = 2'560;
            static __nv_bfloat16 host_hidden[8 * kDumpHidden];
            const std::size_t hidden_count =
                static_cast<std::size_t>(kDumpHidden) * static_cast<std::size_t>(tokens);
            cudaMemcpy(host_hidden, input.data, hidden_count * sizeof(__nv_bfloat16),
                       cudaMemcpyDeviceToHost);
            std::fwrite(host_hidden, sizeof(__nv_bfloat16), hidden_count, route_dump_file);
            std::fflush(route_dump_file);

            // ---- self-describing sidecar (P1-1) ---------------------------------------------------
            // WHY: every routing reading this tree has taken was accompanied by its CONFIGURATION only
            // in whoever's shell history set the switches. Days later the numbers were revisited and
            // the configuration was gone -- measured here twice: slots=128 against a built-in default
            // of 64, and hot=20 against a default of 32. Both defaults differ from what was actually
            // run, and the reading itself could not say so.
            //
            // A SEPARATE file on purpose: the ids file's framing is depended on by
            // scripts/analyze_routes.py and by the routes.bin already on disk, so a reader that knows
            // nothing about this sidecar keeps working byte for byte. Values come from the
            // ENVIRONMENT rather than from the cache's internals, because those knobs are
            // process-local statics with no accessor -- and an env dump cannot drift from what the
            // process was actually launched with.
            //
            // Two things it deliberately does NOT claim, because this call site cannot know them:
            //   * `kind`: tokens <= kFlashNextRouteDecodeMaxTokens is decode-OR-verify, wider is
            //     prefill. A draft-verification column set is indistinguishable from a decode call
            //     here, so no phase is asserted for it.
            //   * `live`: which columns were rejected is decided by the caller, not visible here.
            static bool s_meta_written = false;
            if (!s_meta_written) {
                s_meta_written = true;
                char meta_path[4096];
                std::snprintf(meta_path, sizeof(meta_path), "%s.meta", route_dump_path);
                if (std::FILE* meta = std::fopen(meta_path, "wb")) {
                    std::fprintf(meta,
                                 "ninfer_route_dump meta v2\n"
                                 "record_layout=[router_ptr u64][tokens i32][topk i32]"
                                 "[ids i32*tokens*topk][hidden bf16*%d*tokens]\n"
                                 "topk=%d\nhidden=%d\nexpert_count=%d\n"
                                 "kind_source=width_only\n"
                                 "kind_note=tokens<=%d is decode-or-verify, wider is prefill\n"
                                 "live_source=none\n"
                                 "live_note=no live mask is written; a reader must not assume every "
                                 "column is live\n",
                                 kDumpHidden, kDumpTopK, kDumpHidden, weights.expert_gate_up.experts,
                                 kFlashNextRouteDecodeMaxTokens);
                    static const char* const kKnobs[] = {
                        "NINFER_FLASH_NEXT_EXPERTS_HOST",
                        "NINFER_FLASH_NEXT_EXPERTS_PIN",
                        "NINFER_FLASH_NEXT_EXPERTS_SLOTS",
                        "NINFER_FLASH_NEXT_EXPERTS_HOT",
                        "NINFER_FLASH_NEXT_EXPERTS_VERIFY",
                        "NINFER_FLASH_NEXT_EXPERTS_SABOTAGE",
                        "NINFER_FLASH_NEXT_EXPERTS_ROUND",
                        "NINFER_FLASH_NEXT_EXPERTS_GLOBAL_POOL",
                        "NINFER_FLASH_NEXT_EXPERTS_LFRU",
                        "NINFER_FLASH_NEXT_EXPERTS_DEVICE_STAGE",
                        "NINFER_FLASH_NEXT_MOE_DROP_F",
                        "NINFER_FLASH_NEXT_MOE_DROP_F_PREFILL",
                        "NINFER_FLASH_NEXT_MOE_SUBST",
                        "NINFER_FLASH_NEXT_MOE_WIDE_ARM",
                        "NINFER_FLASH_NEXT_MOE_WAVE",
                    };
                    for (const char* knob : kKnobs) {
                        const char* v = std::getenv(knob);
                        std::fprintf(meta, "env.%s=%s\n", knob,
                                     (v != nullptr && v[0] != '\0') ? v : "<unset>");
                    }
                    std::fflush(meta);
                    std::fclose(meta);
                }
            }
        }
    }

    // Host-tiered expert banks (NINFER_FLASH_NEXT_EXPERTS_HOST): bring this layer's routed experts
    // into the staging planes and point the launch at them. Placement only -- the kernels, the
    // arithmetic and the ids' meaning inside the kernel are unchanged; see expert_staging.h.
    // Block-5: a gguf bank never enters the nvfp4 staging path (its geometry check would throw);
    // the gguf leaf owns its own staging and the 11-path combine.
    MoeWeights staged                = weights;
    const MoeWeights* launch_weights = &weights;
    if (weights.experts_gguf) {
        flash_next_moe_gguf_experts(input, weights, output, scratch, workspace, stream);
    } else {
        if (weights.expert_gate_up.on_host || weights.expert_down.on_host) {
            // Block-6 W1: hand the staging this call's input; the CPU serve forward face mirrors
            // it to the host when armed (no-op otherwise).
            staged         = stage_flash_next_moe_experts(weights, scratch, tokens, stream, &input);
            launch_weights = &staged;
        }
        flash_next_moe_kernels_launch(input, *launch_weights, scratch, output, stream);
        // Block-6 W2: the served paths' GPU side is the R1-drop form (legal slot, alpha = 0); the
        // real CPU contribution lands here, on the same stream, right after the kernels.
        flash_next_cpu_serve_apply(output, scratch, tokens, stream);
    }

    // Diagnostic NINFER_FLASH_NEXT_TRACE_ROUTING only; not on the default prefill chunk path.
    static const char* trace_routing_env = std::getenv("NINFER_FLASH_NEXT_TRACE_ROUTING");
    if (trace_routing_env != nullptr && trace_routing_env[0] != '\0' && tokens >= 512) {
        static int s_route_call = 0;
        int h_active = 0;
        cudaMemcpyAsync(&h_active, scratch.active_count.data, sizeof(int), cudaMemcpyDeviceToHost, stream);
        std::vector<int> h_counts(512);
        cudaMemcpyAsync(h_counts.data(), scratch.expert_counts.data, 512 * sizeof(int), cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_grp = 0, min_grp = 999999, sum_grp = 0;
        for (int e = 0; e < 512; ++e) {
            if (h_counts[e] > 0) {
                max_grp = std::max(max_grp, h_counts[e]);
                min_grp = std::min(min_grp, h_counts[e]);
                sum_grp += h_counts[e];
            }
        }
        int layer_idx = (s_route_call++) % 48;
        if (layer_idx == 0) {
            std::fprintf(stderr, "\n--- MoE Routing Trace (T=%d) ---\n", tokens);
            std::fprintf(stderr, "Layer | Active Experts | %% Active | Min Group | Avg Group | Max Group\n");
            std::fprintf(stderr, "------+----------------+----------+-----------+-----------+----------\n");
        }
        std::fprintf(stderr, " L%02d  |   %3d / 512    |  %5.1f%%  |    %3d    |   %5.1f   |    %3d   \n",
                     layer_idx, h_active, h_active * 100.0f / 512.0f, min_grp, (float)sum_grp / h_active, max_grp);
        if (layer_idx == 47) {
            std::fprintf(stderr, "-----------------------------------------------------------------\n\n");
        }
    }
}

void flash_next_moe_bf16(const Tensor& input, const MoeBf16Weights& weights, Tensor& output,
                         WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.ne[0] != 2'560 ||
        output.ne[0] != 2'560 || tokens < 1 || output.ne[1] != tokens ||
        input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 || output.ne[3] != 1 ||
        !input.is_contiguous() || !output.is_contiguous() || !aligned_to(input.data, 16) ||
        !aligned_to(output.data, 16) || !exact_bf16_weight(weights.router, 512, 2'560) ||
        !exact_bf16_weight(weights.shared_down, 2'560, 640) ||
        !exact_bf16_weight(weights.shared_gate, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_up, 640, 2'560) ||
        !exact_bf16_weight(weights.shared_gate_weight, 1, 2'560) ||
        !exact_bf16_expert_bank(weights.expert_gate_up, 1'280, 2'560) ||
        !exact_bf16_expert_bank(weights.expert_down, 2'560, 640) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next BF16 MoE received an invalid exact target view");
    }
    const auto scope              = workspace.scope();
    FlashNextMoeWorkspace scratch = allocate_flash_next_moe_workspace(workspace, tokens);
    flash_next_route(input, weights.router, weights.shared_gate_weight, scratch.scores, scratch.ids,
                     scratch.alpha, scratch.shared_scale, stream);
    flash_next_moe_bf16_kernels_launch(input, weights, scratch, output, stream);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
