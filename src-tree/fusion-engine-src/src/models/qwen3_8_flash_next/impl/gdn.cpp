#include "models/qwen3_8_flash_next/impl/gdn.h"

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/scatter.h"

#include "core/layout.h"
#include "models/qwen3_8_flash_next/impl/gdn_kernels.h"
#include "models/qwen3_8_flash_next/impl/gdn_workspace.h"
#include "models/qwen3_8_flash_next/impl/stage_ledger.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool exact_tensor(const Tensor& tensor, DType dtype, std::int32_t n0, std::int32_t n1 = 1,
                  std::int32_t n2 = 1, std::int32_t n3 = 1) {
    return tensor.dtype == dtype && tensor.ne[0] == n0 && tensor.ne[1] == n1 &&
           tensor.ne[2] == n2 && tensor.ne[3] == n3 && tensor.is_contiguous() &&
           aligned_to(tensor.data, 16);
}

bool exact_bf16_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    return weight.qtype == QType::BF16 && weight.layout == QuantLayout::Contiguous &&
           weight.n == rows && weight.k == columns && weight.ndim == 2 && weight.shape[0] == rows &&
           weight.shape[1] == columns && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.qdata == weight.payload &&
           weight.payload_bytes >= static_cast<std::uint64_t>(rows) * columns * 2 &&
           aligned_to(weight.qdata, 16);
}

bool exact_fp8_f32_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    const std::uint64_t code_bytes   = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_offset = (code_bytes + 255U) & ~std::uint64_t{255U};
    const auto* payload              = static_cast<const std::byte*>(weight.payload);
    // 4-byte scale arm: this artifact's FP8 weights are `fp8_e4m3fn_row_f32` ->
    // QType::FP8_E4M3FN_ROW_F32, which is what the rest of this predicate already demands
    // (`scale_dtype == DType::FP32`, `scale_nb[0] == 4`). See the note in `qsa_attention.cpp`.
    return weight.qtype == QType::FP8_E4M3FN_ROW_F32 && weight.layout == QuantLayout::RowScale &&
           weight.scale_dtype == DType::FP32 && weight.group_size == columns &&
           weight.group == columns && weight.n == rows && weight.k == columns && weight.ndim == 2 &&
           weight.shape[0] == rows && weight.shape[1] == columns && weight.shape[2] == 1 &&
           weight.shape[3] == 1 && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.padded_shape[2] == 1 &&
           weight.padded_shape[3] == 1 && weight.scale_ne[0] == rows && weight.scale_ne[1] == 1 &&
           weight.scale_ne[2] == 1 && weight.scale_ne[3] == 1 && weight.scale_nb[0] == 4 &&
           weight.scale_nb[1] == static_cast<std::int64_t>(rows) * 4 &&
           weight.scale_nb[2] == static_cast<std::int64_t>(rows) * 4 &&
           weight.scale_nb[3] == static_cast<std::int64_t>(rows) * 4 && payload != nullptr &&
           weight.qdata == payload && weight.scales == payload + scale_offset &&
           weight.qhigh == nullptr && weight.high_plane_bytes == 0 &&
           weight.payload_bytes >= scale_offset + static_cast<std::uint64_t>(rows) * 4 &&
           aligned_to(weight.qdata, 16) && aligned_to(weight.scales, 16);
}

} // namespace

std::size_t flash_next_gdn_workspace_capacity_bytes(std::int32_t min_batch,
                                                    std::int32_t max_batch) {
    if (min_batch <= 0 || max_batch < min_batch || max_batch > 8'192) {
        throw std::invalid_argument("Flash-Next GDN workspace requires B in [1,8192]");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_gdn_workspace(layout, max_batch);
    {
        auto scope = layout.scope();
        const std::size_t qkvz_ws = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, 16'384, 2'560, ops::LinearPolicy::AllowA8, 1, max_batch);
        const std::size_t out_ws = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, 2'560, 6'144, ops::LinearPolicy::AllowA8, 1, max_batch);
        const std::size_t gdn_op_ws =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, max_batch, max_batch);
        (void)layout.alloc_bytes(std::max({qkvz_ws, out_ws, gdn_op_ws}), 256);
    }
    return layout.peak_bytes(256);
}

void flash_next_gdn_decode(const Tensor& input, const GdnWeights& weights,
                           const Tensor& source_slots, const Tensor& destination_slots,
                           Tensor& convolution_states, Tensor& ssm_states,
                           WorkspaceArena& workspace, Tensor& output, cudaStream_t stream,
                           bool aliased_recurrent_scan) {
    const std::int32_t batch = input.ne[1];
    if (!exact_tensor(input, DType::BF16, 2'560, batch) || batch < 1 || batch > 8 ||
        !exact_tensor(output, DType::BF16, 2'560, batch) ||
        !exact_tensor(source_slots, DType::I32, batch) ||
        !exact_tensor(destination_slots, DType::I32, batch) ||
        convolution_states.dtype != DType::BF16 || convolution_states.ne[0] != 10'240 ||
        convolution_states.ne[1] != 3 || convolution_states.ne[2] <= 0 ||
        convolution_states.ne[3] != 1 || !convolution_states.is_contiguous() ||
        !aligned_to(convolution_states.data, 16) ||
        (ssm_states.dtype != DType::FP32 && ssm_states.dtype != DType::BF16) ||
        ssm_states.ne[0] != 128 || ssm_states.ne[1] != 128 || ssm_states.ne[2] != 48 ||
        ssm_states.ne[3] != convolution_states.ne[2] || !ssm_states.is_contiguous() ||
        !aligned_to(ssm_states.data, 16) || !exact_tensor(weights.a_log, DType::BF16, 48) ||
        !exact_tensor(weights.convolution, DType::BF16, 10'240, 4) ||
        !exact_tensor(weights.dt_bias, DType::BF16, 48) ||
        !exact_bf16_weight(weights.a_b_projection, 96, 2'560) ||
        !exact_tensor(weights.norm, DType::BF16, 128) ||
        !exact_fp8_f32_weight(weights.query_key_value_z, 16'384, 2'560) ||
        !exact_fp8_f32_weight(weights.output, 2'560, 6'144) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next GDN received an invalid exact target view");
    }

    const auto scope              = workspace.scope();
    FlashNextGdnWorkspace scratch = allocate_flash_next_gdn_workspace(workspace, batch);
    ops::linear(input, weights.query_key_value_z, scratch.projected, ops::LinearPolicy::A16Only,
                workspace, stream);
    flash_next_gdn_controls_launch(input, weights, scratch, stream);

    if (aliased_recurrent_scan && batch > 1) {
        for (std::int32_t r = 0; r < batch; ++r) {
            flash_next_gdn_conv_launch(scratch, weights.convolution, source_slots, destination_slots,
                                       convolution_states, stream, 1, r);
            Tensor query_r            = scratch.query.view({128, 16, 1, batch}).slice(3, r, 1);
            Tensor key_r              = scratch.key.view({128, 16, 1, batch}).slice(3, r, 1);
            Tensor value_r            = scratch.value.view({128, 48, 1, batch}).slice(3, r, 1);
            Tensor recurrent_output_r = scratch.recurrent_output.view({128, 48, 1, batch}).slice(3, r, 1);
            Tensor g_r                = scratch.g.view({48, 1, batch}).slice(2, r, 1);
            Tensor beta_r             = scratch.beta.view({48, 1, batch}).slice(2, r, 1);
            Tensor src_r              = source_slots.slice(0, r, 1);
            Tensor dst_r              = destination_slots.slice(0, r, 1);
            ops::gated_delta_net_batch_update(query_r, key_r, value_r, g_r, beta_r,
                                              1.0F / std::sqrt(128.0F), true, ssm_states,
                                              src_r, dst_r, recurrent_output_r, stream);
        }
    } else {
        flash_next_gdn_conv_launch(scratch, weights.convolution, source_slots, destination_slots,
                                   convolution_states, stream);
        Tensor query            = scratch.query.view({128, 16, 1, batch});
        Tensor key              = scratch.key.view({128, 16, 1, batch});
        Tensor value            = scratch.value.view({128, 48, 1, batch});
        Tensor recurrent_output = scratch.recurrent_output.view({128, 48, 1, batch});
        Tensor g                = scratch.g.view({48, 1, batch});
        Tensor beta             = scratch.beta.view({48, 1, batch});
        ops::gated_delta_net_batch_update(query, key, value, g, beta, 1.0F / std::sqrt(128.0F), true,
                                          ssm_states, source_slots, destination_slots, recurrent_output,
                                          stream);
    }

    flash_next_gdn_output_gate_launch(scratch, weights.norm, stream);
    ops::linear(scratch.gated_output, weights.output, output, ops::LinearPolicy::A16Only, workspace,
                stream);
}

void flash_next_gdn_prefill_chunk(const Tensor& input, const GdnWeights& weights,
                                  std::int32_t source_slot, std::int32_t destination_slot,
                                  Tensor& convolution_states, Tensor& ssm_states,
                                  WorkspaceArena& workspace, Tensor& output, cudaStream_t stream) {
    const std::int32_t tokens      = input.ne[1];
    const std::int32_t state_slots = convolution_states.ne[2];
    if (!exact_tensor(input, DType::BF16, 2'560, tokens) || tokens < 1 ||
        !exact_tensor(output, DType::BF16, 2'560, tokens) ||
        convolution_states.dtype != DType::BF16 || convolution_states.ne[0] != 10'240 ||
        convolution_states.ne[1] != 3 || convolution_states.ne[2] <= 0 ||
        convolution_states.ne[3] != 1 || !convolution_states.is_contiguous() ||
        !aligned_to(convolution_states.data, 16) ||
        (ssm_states.dtype != DType::FP32 && ssm_states.dtype != DType::BF16) ||
        ssm_states.ne[0] != 128 || ssm_states.ne[1] != 128 || ssm_states.ne[2] != 48 ||
        ssm_states.ne[3] != convolution_states.ne[2] || !ssm_states.is_contiguous() ||
        !aligned_to(ssm_states.data, 16) || !exact_tensor(weights.a_log, DType::BF16, 48) ||
        !exact_tensor(weights.convolution, DType::BF16, 10'240, 4) ||
        !exact_tensor(weights.dt_bias, DType::BF16, 48) ||
        !exact_bf16_weight(weights.a_b_projection, 96, 2'560) ||
        !exact_tensor(weights.norm, DType::BF16, 128) ||
        !exact_fp8_f32_weight(weights.query_key_value_z, 16'384, 2'560) ||
        !exact_fp8_f32_weight(weights.output, 2'560, 6'144) ||
        source_slot < 0 || source_slot >= state_slots || destination_slot < 0 ||
        destination_slot >= state_slots || stream == nullptr) {
        throw std::invalid_argument("Flash-Next GDN prefill chunk received an invalid view");
    }

    const auto scope              = workspace.scope();
    FlashNextGdnWorkspace scratch = allocate_flash_next_gdn_workspace(workspace, tokens);
    const ops::LinearPolicy linear_policy =
        tokens >= 16 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::A16Only;

    // 1. Projection -> scratch.projected [16384, T]
    ops::linear(input, weights.query_key_value_z, scratch.projected, linear_policy, workspace,
                stream);
    stage_ledger_record(stream, FlashNextStageId::GDN_QkvzProjection);

    // 2. Causal conv on rows [0, 10240)
    ops::extract_bf16_columns(scratch.projected, 0, scratch.conv_in, stream);
    Tensor conv_state_in  = convolution_states.slice(2, source_slot, 1).view({10'240, 3});
    Tensor conv_state_out = convolution_states.slice(2, destination_slot, 1).view({10'240, 3});
    ops::causal_conv1d_silu(scratch.conv_in, weights.convolution, conv_state_in, conv_state_out,
                            scratch.conv_output, stream);

    // Extract q, k, v from conv_output
    ops::extract_bf16_columns(scratch.conv_output, 0, scratch.query, stream);
    ops::extract_bf16_columns(scratch.conv_output, 2'048, scratch.key, stream);
    ops::extract_bf16_columns(scratch.conv_output, 4'096, scratch.value, stream);

    // Extract z from rows [10240, 16384) of scratch.projected
    ops::extract_bf16_columns(scratch.projected, 10'240, scratch.z, stream);
    stage_ledger_record(stream, FlashNextStageId::GDN_ConvExtracts);

    // 3. Controls -> g [48, T], beta [48, T]
    flash_next_gdn_controls_launch(input, weights, scratch, stream);
    stage_ledger_record(stream, FlashNextStageId::GDN_Controls);

    // 4. Gated DeltaNet distinct-state recurrence
    Tensor query            = scratch.query.view({128, 16, tokens});
    Tensor key              = scratch.key.view({128, 16, tokens});
    Tensor value            = scratch.value.view({128, 48, tokens});
    Tensor recurrent_output = scratch.recurrent_output.view({128, 48, tokens});
    Tensor g                = scratch.g.view({48, tokens});
    Tensor beta             = scratch.beta.view({48, tokens});
    Tensor ssm_in           = ssm_states.slice(3, source_slot, 1).view({128, 128, 48});
    Tensor ssm_out          = ssm_states.slice(3, destination_slot, 1).view({128, 128, 48});

    ops::detail::gated_delta_net::chunked::GdnChunkedStageHook gdn_hook{};
    if (FlashNextStageLedger::is_enabled()) {
        gdn_hook.record_stage = [](void*, int stage_id, cudaStream_t s) {
            if (stage_id == 0) {
                stage_ledger_record(s, FlashNextStageId::GDN_Recurrence_PrepareWyWu);
            } else if (stage_id == 1) {
                stage_ledger_record(s, FlashNextStageId::GDN_Recurrence_StatePassing);
            } else if (stage_id == 2) {
                stage_ledger_record(s, FlashNextStageId::GDN_Recurrence_Output);
            }
        };
    }
    ops::gated_delta_net(query, key, value, g, beta, 1.0F / std::sqrt(128.0F), true, workspace,
                         ssm_in, ssm_out, recurrent_output, stream,
                         FlashNextStageLedger::is_enabled() ? &gdn_hook : nullptr);

    // 5. Output gate & projection
    flash_next_gdn_output_gate_launch(scratch, weights.norm, stream);
    stage_ledger_record(stream, FlashNextStageId::GDN_OutputGate);
    ops::linear(scratch.gated_output, weights.output, output, linear_policy, workspace, stream);
    stage_ledger_record(stream, FlashNextStageId::GDN_OutputProjection);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
