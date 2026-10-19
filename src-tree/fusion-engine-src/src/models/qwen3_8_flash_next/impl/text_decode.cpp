#include "models/qwen3_8_flash_next/impl/text_decode.h"
#include "models/qwen3_8_flash_next/impl/mtp_forward.h"

#include "core/device.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"

#include "core/layout.h"
#include "models/qwen3_8_flash_next/impl/gdn.h"
#include "models/qwen3_8_flash_next/impl/hyper_connection.h"
#include "models/qwen3_8_flash_next/impl/kv_stream_binding.h"
#include "models/qwen3_8_flash_next/impl/moe.h"
#include "models/qwen3_8_flash_next/impl/ple_decode.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention.h"
#include "models/qwen3_8_flash_next/impl/qsa_indexer.h"
#include "models/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"
#include "models/qwen3_8_flash_next/impl/stage_ledger.h"
#include "models/qwen3_8_flash_next/impl/text_decode_kernels.h"
#include "models/qwen3_8_flash_next/impl/text_decode_workspace.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

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
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    const auto* payload              = static_cast<const std::byte*>(weight.payload);
    const std::int64_t scale_stride  = static_cast<std::int64_t>(rows) * 4;
    // 4-byte scale arm: this artifact's FP8 weights are `fp8_e4m3fn_row_f32` ->
    // QType::FP8_E4M3FN_ROW_F32, which is what the rest of this predicate already demands
    // (`scale_dtype == DType::FP32`, `scale_nb[0] == 4`). See the note in `qsa_attention.cpp`.
    return weight.qtype == QType::FP8_E4M3FN_ROW_F32 && weight.layout == QuantLayout::RowScale &&
           weight.scale_dtype == DType::FP32 && weight.group_size == static_cast<std::uint32_t>(columns) &&
           weight.group == columns && weight.n == rows && weight.k == columns && weight.ndim == 2 &&
           weight.shape[0] == rows && weight.shape[1] == columns && weight.shape[2] == 1 &&
           weight.shape[3] == 1 && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == columns && weight.padded_shape[2] == 1 &&
           weight.padded_shape[3] == 1 && weight.scale_ne[0] == rows && weight.scale_ne[1] == 1 &&
           weight.scale_ne[2] == 1 && weight.scale_ne[3] == 1 && weight.scale_nb[0] == 4 &&
           weight.scale_nb[1] == scale_stride && weight.scale_nb[2] == scale_stride &&
           weight.scale_nb[3] == scale_stride && payload != nullptr && weight.qdata == payload &&
           weight.scales == payload + scale_offset && weight.qhigh == nullptr &&
           weight.high_plane_bytes == 0 &&
           weight.payload_bytes >= scale_offset + static_cast<std::uint64_t>(rows) * 4 &&
           aligned_to(weight.qdata, 16) && aligned_to(weight.scales, 16);
}

bool exact_output_head(const Weight& weight) {
    return exact_bf16_weight(weight, 248'320, 2'560) ||
           exact_fp8_f32_weight(weight, 248'320, 2'560);
}

bool exact_token_embedding(const Weight& weight) {
    return exact_bf16_weight(weight, 248'320, 2'560) ||
           exact_fp8_f32_weight(weight, 248'320, 2'560);
}

} // namespace

void validate_flash_next_decode_state(const FlashNextDecodeStateView& state,
                                      std::int32_t state_slots) {
    if (state_slots <= 0) {
        throw std::invalid_argument("Flash-Next state validation requires state_slots > 0");
    }
    if (!exact_tensor(state.ple_convolution_states, DType::BF16, 10'240, 9, state_slots)) {
        throw std::invalid_argument("Flash-Next PLE convolution state view is invalid");
    }
    for (std::size_t i = 0; i < kGdnLayers; ++i) {
        const DType ssm_dtype = state.gdn_ssm_states[i].dtype;
        if (!exact_tensor(state.gdn_convolution_states[i], DType::BF16, 10'240, 3, state_slots) ||
            (ssm_dtype != DType::FP32 && ssm_dtype != DType::BF16) ||
            !exact_tensor(state.gdn_ssm_states[i], ssm_dtype, 128, 128, 48, state_slots)) {
            throw std::invalid_argument("Flash-Next GDN state view is invalid");
        }
    }
    for (std::size_t i = 0; i < kFullAttentionLayers; ++i) {
        const auto& idx = state.qsa_indexer_caches[i];
        if (idx.block_keys.dtype != DType::BF16 || idx.block_keys.ne[0] != 128 ||
            idx.block_keys.ne[1] != 64 || idx.block_keys.ne[2] <= 0 || idx.block_keys.ne[3] != 1 ||
            !idx.block_keys.is_contiguous() || !aligned_to(idx.block_keys.data, 16) ||
            idx.block_tables.dtype != DType::I32 || idx.block_tables.ne[0] <= 0 ||
            idx.block_tables.ne[1] <= 0 || idx.block_tables.ne[2] != 1 ||
            idx.block_tables.ne[3] != 1 || !idx.block_tables.is_contiguous() ||
            !aligned_to(idx.block_tables.data, 16) ||
            !exact_tensor(idx.raw_keys, DType::BF16, 128, 4, state_slots) ||
            !exact_tensor(idx.raw_positions, DType::I32, 3, 4, state_slots)) {
            throw std::invalid_argument("Flash-Next QSA indexer cache view is invalid");
        }
        const auto& att = state.qsa_attention_caches[i];
        const auto att_kv_dt = att.key_pages.dtype;
        if ((att_kv_dt != DType::BF16 && att_kv_dt != DType::FP8_E4M3FN) || att.key_pages.ne[0] != 256 ||
            att.key_pages.ne[1] != 64 || att.key_pages.ne[2] != 2 || att.key_pages.ne[3] <= 0 ||
            !att.key_pages.is_contiguous() || !aligned_to(att.key_pages.data, 16) ||
            att.value_pages.dtype != att_kv_dt || att.value_pages.ne[0] != 256 ||
            att.value_pages.ne[1] != 64 || att.value_pages.ne[2] != 2 ||
            att.value_pages.ne[3] <= 0 || !att.value_pages.is_contiguous() ||
            !aligned_to(att.value_pages.data, 16) || att.block_tables.dtype != DType::I32 ||
            att.block_tables.ne[0] <= 0 || att.block_tables.ne[1] <= 0 ||
            att.block_tables.ne[2] != 1 || att.block_tables.ne[3] != 1 ||
            !att.block_tables.is_contiguous() || !aligned_to(att.block_tables.data, 16)) {
            throw std::invalid_argument("Flash-Next QSA attention cache view is invalid");
        }
    }
}

std::size_t flash_next_text_decode_workspace_capacity_bytes(std::int32_t maximum_blocks,
                                                            std::int32_t batch, bool mtp) {
    if (maximum_blocks <= 0 || maximum_blocks > 65'536 || batch <= 0 || batch > 8) {
        throw std::invalid_argument("Flash-Next text decode received an invalid envelope");
    }
    WorkspaceLayoutBuilder layout;
    layout.alloc(DType::BF16, {2'560, batch}, 256);
    (void)allocate_flash_next_text_decode_workspace(layout, batch);
    const std::size_t sort_temp = flash_next_qsa_indexer_sort_temp_bytes(maximum_blocks, batch);
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_ple_workspace(layout, batch);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_gdn_workspace(layout, batch);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_qsa_indexer_workspace(layout, maximum_blocks, batch, sort_temp);
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(flash_next_qsa_attention_workspace_capacity_bytes(batch), 256);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_moe_workspace(layout, batch);
    }
    if (mtp) {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {2'560, batch}, 256);
        (void)layout.alloc_bytes(flash_next_mtp_teacher_workspace_capacity_bytes(batch), 256);
    }
    return layout.peak_bytes(256);
}

std::size_t flash_next_text_prefill_workspace_capacity_bytes(std::int32_t maximum_blocks,
                                                             std::int32_t tokens, bool mtp) {
    if (maximum_blocks <= 0 || maximum_blocks > 65'536 || tokens <= 0) {
        throw std::invalid_argument("Flash-Next text prefill received an invalid envelope");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_prefill_chunk_staging(layout, tokens);
    (void)allocate_flash_next_text_decode_workspace(layout, tokens);
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_ple_workspace(layout, tokens);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_gdn_workspace(layout, tokens);
        const std::size_t gdn_op_ws =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, tokens, tokens);
        layout.alloc_bytes(gdn_op_ws, 256);
    }
    {
        auto scope                   = layout.scope();
        const std::int32_t tile_size = flash_next_qsa_indexer_tile_size(maximum_blocks, tokens);
        const std::size_t sort_temp =
            flash_next_qsa_indexer_sort_temp_bytes(maximum_blocks, tile_size);
        (void)allocate_flash_next_qsa_indexer_workspace(layout, maximum_blocks, tokens, tile_size,
                                                        sort_temp);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_qsa_attention_workspace(layout, tokens);
        const std::size_t qgkv_ws = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, 13'312, 2'560, ops::LinearPolicy::AllowA8, 1, tokens);
        const std::size_t out_ws = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, 2'560, 6'144, ops::LinearPolicy::AllowA8, 1, tokens);
        (void)layout.alloc_bytes(std::max(qgkv_ws, out_ws), 256);
    }
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_moe_workspace(layout, tokens);
    }
    if (mtp) {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {2'560, tokens}, 256);
        (void)layout.alloc_bytes(flash_next_mtp_teacher_workspace_capacity_bytes(tokens), 256);
    }
    return layout.peak_bytes(256);
}

void flash_next_text_decode_core(const TextModelView& model, const Tensor& embedding,
                                 const Tensor& token_indices, const Tensor& mrope_positions,
                                 const Tensor& table_rows, const Tensor& source_slots,
                                 const Tensor& destination_slots,
                                 const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                 std::int32_t active_blocks, FlashNextDecodeStateView state,
                                 WorkspaceArena& workspace, Tensor& final_hidden, Tensor& logits,
                                 cudaStream_t stream, const FlashNextDecodeStateSink* sink,
                                 Tensor* out_hyper_hidden, bool aliased_recurrent_scan,
                                 const Tensor* mtp_token_ids) {
    const std::int32_t batch       = embedding.ne[1];
    const std::int32_t state_slots = state.ple_convolution_states.ne[2];
    if (batch <= 0 || batch > 8 || maximum_blocks <= 0 || maximum_blocks > 65'536 ||
        active_blocks < 0 || active_blocks > maximum_blocks ||
        !exact_tensor(embedding, DType::BF16, 2'560, batch) ||
        !exact_tensor(token_indices, DType::I32, batch) ||
        !exact_tensor(mrope_positions, DType::I32, batch, 3) ||
        !exact_tensor(table_rows, DType::I32, batch) ||
        !exact_tensor(source_slots, DType::I32, batch) ||
        !exact_tensor(destination_slots, DType::I32, batch) ||
        !exact_tensor(gathered_ple_embedding, DType::BF16, 2'560, batch) ||
        !exact_tensor(final_hidden, DType::BF16, 2'560, batch) ||
        !exact_tensor(logits, DType::BF16, 248'320, batch) ||
        !exact_output_head(model.output_head) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next text decode core received an invalid input view");
    }
    validate_flash_next_decode_state(state, state_slots);

    auto emit_state = [&](std::string_view name, const Tensor& tensor) {
        if (sink && sink->on_state) {
            // Diagnostic sink only (TRACE_STAGES / test dumper); decode path, not prefill chunk.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            sink->on_state(name, tensor);
        }
    };

    emit_state("embedding", embedding);

    const auto round_scope = workspace.scope();
    FlashNextTextDecodeWorkspace round_ws =
        allocate_flash_next_text_decode_workspace(workspace, batch);

    // 1. Repeat embedding into 4 hyperconnection streams
    repeat_embedding_to_hyper_streams(embedding, round_ws.hyper_hidden, stream);
    emit_state("hyper_init", round_ws.hyper_hidden);

    // 2. 48-layer execution loop
    for (std::size_t layer = 0; layer < 48; ++layer) {
        char prefix_buf[32];
        std::snprintf(prefix_buf, sizeof(prefix_buf), "L%02zu_", layer);
        const std::string prefix(prefix_buf);

        // At layer 1: evaluate PLE neural injection and add residual
        if (layer == 1) {
            emit_state("ple_gathered", gathered_ple_embedding);
            flash_next_ple_decode(round_ws.hyper_hidden, gathered_ple_embedding, model.ple,
                                  source_slots, destination_slots, state.ple_convolution_states,
                                  workspace, round_ws.ple_injection, stream,
                                  aliased_recurrent_scan);
            emit_state("ple_injection", round_ws.ple_injection);
            ops::residual_add(round_ws.ple_injection, round_ws.hyper_hidden, stream);
            emit_state("hyper_after_ple", round_ws.hyper_hidden);
        }

        // Attention hyper prepare -> block_input [2560, B]
        flash_next_hyper_prepare(round_ws.hyper_hidden, model.layers[layer].attention_hyper,
                                 round_ws.hyper_scratch, round_ws.block_input, stream);
        emit_state(prefix + "attn_block_input", round_ws.block_input);

        // Execute QSA or GDN attention
        if (is_qsa_layer(layer)) {
            const std::size_t qsa_idx = qsa_ordinal(layer);
            flash_next_qsa_indexer_decode(
                round_ws.block_input, model.full_attention[qsa_idx], token_indices, mrope_positions,
                table_rows, source_slots, destination_slots, state.qsa_indexer_caches[qsa_idx],
                maximum_blocks, active_blocks, workspace, round_ws.selected_blocks,
                round_ws.selected_counts, stream, aliased_recurrent_scan);
            emit_state(prefix + "selected_counts", round_ws.selected_counts);
            // KV streaming (P2, plan §B.3): make every page this layer's selection names resident
            // before the reader runs. Host-side resolve, out of any graph by construction (A4).
            if (kv_stream_binding().residency) {
                const auto& kv = kv_stream_binding();
                ops::kv_stream_resolve_host(
                    kv.map, kv.runs,
                    static_cast<const std::int32_t*>(round_ws.selected_blocks.data),
                    round_ws.selected_blocks.ne[0],
                    static_cast<const std::int32_t*>(round_ws.selected_counts.data),
                    static_cast<const std::int32_t*>(token_indices.data), batch, 64, 4, false,
                    stream);
            }
            flash_next_qsa_attention_decode(
                round_ws.block_input, model.full_attention[qsa_idx], token_indices, mrope_positions,
                table_rows, round_ws.selected_blocks, round_ws.selected_counts,
                state.qsa_attention_caches[qsa_idx], workspace, round_ws.block_output, stream);
        } else {
            const std::size_t gdn_idx = gdn_ordinal(layer);
            flash_next_gdn_decode(round_ws.block_input, model.gdn[gdn_idx], source_slots,
                                  destination_slots, state.gdn_convolution_states[gdn_idx],
                                  state.gdn_ssm_states[gdn_idx], workspace, round_ws.block_output,
                                  stream, aliased_recurrent_scan);
        }
        emit_state(prefix + "attn_block_output", round_ws.block_output);

        // Attention hyper inject
        flash_next_hyper_inject(round_ws.block_output, round_ws.hyper_scratch.injection,
                                round_ws.hyper_hidden, stream);
        emit_state(prefix + "hyper_after_attn", round_ws.hyper_hidden);

        // MLP hyper prepare -> block_input [2560, B]
        flash_next_hyper_prepare(round_ws.hyper_hidden, model.layers[layer].mlp_hyper,
                                 round_ws.hyper_scratch, round_ws.block_input, stream);
        emit_state(prefix + "mlp_block_input", round_ws.block_input);

        // MoE
        flash_next_moe(round_ws.block_input, model.layers[layer].moe, round_ws.block_output,
                       workspace, stream);
        emit_state(prefix + "mlp_block_output", round_ws.block_output);

        // MLP hyper inject
        flash_next_hyper_inject(round_ws.block_output, round_ws.hyper_scratch.injection,
                                round_ws.hyper_hidden, stream);
        emit_state(prefix + "hyper_after_mlp", round_ws.hyper_hidden);
    }

    if (state.mtp_backbone_hidden.data != nullptr) {
        const auto mtp_scope = workspace.scope();
        Tensor mtp_embedding = embedding;
        if (mtp_token_ids != nullptr) {
            mtp_embedding = workspace.alloc(DType::BF16, {2'560, batch}, 256);
            ops::embedding(*mtp_token_ids, model.token_embedding, mtp_embedding, stream);
        }
        flash_next_mtp_teacher_extend(*model.mtp, mtp_embedding, round_ws.hyper_hidden,
            token_indices, mrope_positions, table_rows, source_slots, destination_slots,
            0, 0, 0, 0, false, aliased_recurrent_scan, state, workspace, stream);
    }

    // 3. Final hyper mixer -> final_hidden [2560, B]
    flash_next_hyper_mix(round_ws.hyper_hidden, model.final_mixer, round_ws.hyper_scratch,
                         final_hidden, stream);
    emit_state("final_hidden", final_hidden);

    if (out_hyper_hidden != nullptr && out_hyper_hidden->data != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(out_hyper_hidden->data, round_ws.hyper_hidden.data,
                                   10'240ULL * batch * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // 4. Output head linear projection -> logits [248320, B]
    ops::linear(final_hidden, model.output_head, logits, ops::LinearPolicy::A16Only, workspace,
                stream);
    emit_state("logits", logits);
}

void flash_next_text_decode(const TextModelView& model, const Tensor& token_ids,
                            const Tensor& token_indices, const Tensor& mrope_positions,
                            const Tensor& table_rows, const Tensor& source_slots,
                            const Tensor& destination_slots, const Tensor& gathered_ple_embedding,
                            std::int32_t maximum_blocks, std::int32_t active_blocks,
                            FlashNextDecodeStateView state, WorkspaceArena& workspace,
                            Tensor& final_hidden, Tensor& logits, cudaStream_t stream,
                            const FlashNextDecodeStateSink* sink) {
    const std::int32_t batch = token_ids.ne[0];
    if (batch <= 0 || batch > 8 || !exact_tensor(token_ids, DType::I32, batch) ||
        !exact_token_embedding(model.token_embedding)) {
        throw std::invalid_argument("Flash-Next text decode token embedding input is invalid");
    }
    const auto round_scope = workspace.scope();
    Tensor embedding       = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    ops::embedding(token_ids, model.token_embedding, embedding, stream);
    flash_next_text_decode_core(model, embedding, token_indices, mrope_positions, table_rows,
                                source_slots, destination_slots, gathered_ple_embedding,
                                maximum_blocks, active_blocks, state, workspace, final_hidden,
                                logits, stream, sink);
}

void flash_next_text_prefill_chunk(const TextModelView& model, const Tensor& embedding,
                                   const Tensor& token_indices, const Tensor& mrope_positions,
                                   std::int32_t table_row, std::int32_t source_slot,
                                   std::int32_t destination_slot,
                                   const Tensor& gathered_ple_embedding, std::int32_t maximum_blocks,
                                   std::int32_t first_token_index, FlashNextDecodeStateView state,
                                   WorkspaceArena& workspace, Tensor& final_hidden, Tensor& logits,
                                   cudaStream_t stream, const FlashNextDecodeStateSink* sink,
                                   bool use_qsa_prefill_mma, Tensor* out_hyper_hidden,
                                   const Tensor* mtp_token_ids) {
    const std::int32_t tokens      = embedding.ne[1];
    const std::int32_t state_slots = state.ple_convolution_states.ne[2];
    if (tokens <= 0 || maximum_blocks <= 0 || maximum_blocks > 65'536 || first_token_index < 0 ||
        table_row < 0 || source_slot < 0 || source_slot >= state_slots || destination_slot < 0 ||
        destination_slot >= state_slots ||
        !exact_tensor(embedding, DType::BF16, 2'560, tokens) ||
        !exact_tensor(token_indices, DType::I32, tokens) ||
        mrope_positions.dtype != DType::I32 || !mrope_positions.is_contiguous() ||
        !aligned_to(mrope_positions.data, 16) ||
        !((mrope_positions.ne[0] == 3 && mrope_positions.ne[1] == tokens) ||
          (mrope_positions.ne[0] == tokens && mrope_positions.ne[1] == 3) ||
          (mrope_positions.ne[0] == 1 && mrope_positions.ne[1] == 3 && mrope_positions.ne[2] == tokens)) ||
        !exact_tensor(gathered_ple_embedding, DType::BF16, 2'560, tokens) ||
        !exact_tensor(final_hidden, DType::BF16, 2'560, 1) ||
        !exact_tensor(logits, DType::BF16, 248'320, 1) ||
        !exact_output_head(model.output_head) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next text prefill chunk received an invalid input view");
    }
    validate_flash_next_decode_state(state, state_slots);

    auto emit_state = [&](std::string_view name, const Tensor& tensor) {
        if (sink && sink->on_state) {
            // Diagnostic sink only (TRACE_STAGES / test dumper): device tensor must be idle.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            sink->on_state(name, tensor);
        }
    };

    emit_state("embedding", embedding);

    const auto round_scope = workspace.scope();
    FlashNextTextDecodeWorkspace round_ws =
        allocate_flash_next_text_decode_workspace(workspace, tokens);

    // 1. Repeat embedding into 4 hyperconnection streams
    repeat_embedding_to_hyper_streams(embedding, round_ws.hyper_hidden, stream);
    stage_ledger_record(stream, FlashNextStageId::Preamble_EmbeddingStaging);
    emit_state("hyper_init", round_ws.hyper_hidden);

    // 2. 48-layer execution loop
    for (std::size_t layer = 0; layer < 48; ++layer) {
        char prefix_buf[32];
        std::snprintf(prefix_buf, sizeof(prefix_buf), "L%02zu_", layer);
        const std::string prefix(prefix_buf);

        // At layer 1: evaluate PLE neural injection and add residual
        if (layer == 1) {
            emit_state("ple_gathered", gathered_ple_embedding);
            flash_next_ple_prefill_chunk(round_ws.hyper_hidden, gathered_ple_embedding, model.ple,
                                         source_slot, destination_slot,
                                         state.ple_convolution_states, workspace,
                                         round_ws.ple_injection, stream);
            emit_state("ple_injection", round_ws.ple_injection);
            ops::residual_add(round_ws.ple_injection, round_ws.hyper_hidden, stream);
            stage_ledger_record(stream, FlashNextStageId::PLE_Injection);
            emit_state("hyper_after_ple", round_ws.hyper_hidden);
        }

        // Attention hyper prepare -> block_input [2560, T]
        flash_next_hyper_prepare(round_ws.hyper_hidden, model.layers[layer].attention_hyper,
                                 round_ws.hyper_scratch, round_ws.block_input, stream);
        stage_ledger_record(stream, FlashNextStageId::Hyper_PrepareAttn);
        emit_state(prefix + "attn_block_input", round_ws.block_input);

        // Execute QSA or GDN attention
        if (is_qsa_layer(layer)) {
            const std::size_t qsa_idx = qsa_ordinal(layer);
            flash_next_qsa_indexer_prefill_chunk(
                round_ws.block_input, model.full_attention[qsa_idx], token_indices,
                mrope_positions, table_row, source_slot, destination_slot,
                state.qsa_indexer_caches[qsa_idx], maximum_blocks, first_token_index, workspace,
                round_ws.selected_blocks, round_ws.selected_counts, stream);
            emit_state(prefix + "selected_counts", round_ws.selected_counts);
            emit_state(prefix + "selected_blocks", round_ws.selected_blocks);
            QsaStageEmitter qsa_emit;
            if (sink && sink->on_state) {
                qsa_emit = [&](std::string_view name, const Tensor& t) { emit_state(prefix + std::string(name), t); };
            }
            flash_next_qsa_attention_prefill_chunk(
                round_ws.block_input, model.full_attention[qsa_idx], token_indices,
                mrope_positions, table_row, round_ws.selected_blocks,
                round_ws.selected_counts, state.qsa_attention_caches[qsa_idx],
                workspace, round_ws.block_output, stream, qsa_emit, use_qsa_prefill_mma);
        } else {
            const std::size_t gdn_idx = gdn_ordinal(layer);
            flash_next_gdn_prefill_chunk(round_ws.block_input, model.gdn[gdn_idx], source_slot,
                                         destination_slot, state.gdn_convolution_states[gdn_idx],
                                         state.gdn_ssm_states[gdn_idx], workspace,
                                         round_ws.block_output, stream);
        }
        emit_state(prefix + "attn_block_output", round_ws.block_output);

        // Attention hyper inject
        flash_next_hyper_inject(round_ws.block_output, round_ws.hyper_scratch.injection,
                                round_ws.hyper_hidden, stream);
        stage_ledger_record(stream, FlashNextStageId::Hyper_InjectAttn);
        emit_state(prefix + "hyper_after_attn", round_ws.hyper_hidden);

        // MLP hyper prepare -> block_input [2560, T]
        flash_next_hyper_prepare(round_ws.hyper_hidden, model.layers[layer].mlp_hyper,
                                 round_ws.hyper_scratch, round_ws.block_input, stream);
        stage_ledger_record(stream, FlashNextStageId::Hyper_PrepareMlp);
        emit_state(prefix + "mlp_block_input", round_ws.block_input);

        // MoE
        flash_next_moe(round_ws.block_input, model.layers[layer].moe, round_ws.block_output,
                       workspace, stream);
        emit_state(prefix + "mlp_block_output", round_ws.block_output);

        // MLP hyper inject
        flash_next_hyper_inject(round_ws.block_output, round_ws.hyper_scratch.injection,
                                round_ws.hyper_hidden, stream);
        stage_ledger_record(stream, FlashNextStageId::Hyper_InjectMlp);
        emit_state(prefix + "hyper_after_mlp", round_ws.hyper_hidden);
    }

    if (state.mtp_backbone_hidden.data != nullptr) {
        const auto mtp_scope = workspace.scope();
        Tensor mtp_embedding = embedding;
        if (mtp_token_ids != nullptr) {
            mtp_embedding = workspace.alloc(DType::BF16, {2'560, tokens}, 256);
            ops::embedding(*mtp_token_ids, model.token_embedding, mtp_embedding, stream);
        }
        flash_next_mtp_teacher_extend(*model.mtp, mtp_embedding, round_ws.hyper_hidden,
            token_indices, mrope_positions, Tensor{}, Tensor{}, Tensor{}, table_row,
            source_slot, destination_slot, first_token_index, true, true,
            state, workspace, stream);
    }

    // 3. Final hyper mixer on last token only -> final_hidden [2560, 1]
    Tensor last_hidden = round_ws.hyper_hidden.slice(1, tokens - 1, 1);
    // The speculative draft head consumes the pre-final-mixer 4-stream hidden of the last position.
    // round_ws lives in the scoped arena, so without this copy the first draft of every request
    // reads whatever the previous request (or the graph warm-up) left in round_tensors().
    if (out_hyper_hidden != nullptr && out_hyper_hidden->data != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(out_hyper_hidden->data, last_hidden.data,
                                   10'240ULL * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
                                   stream));
    }
    flash_next_hyper_mix(last_hidden, model.final_mixer, round_ws.single_token_hyper_scratch,
                         final_hidden, stream);
    emit_state("final_hidden", final_hidden);

    // 4. Output head linear projection -> logits [248320, 1]
    ops::linear(final_hidden, model.output_head, logits, ops::LinearPolicy::A16Only, workspace,
                stream);
    stage_ledger_record(stream, FlashNextStageId::Final_NormHead);
    emit_state("logits", logits);

    FlashNextStageLedger::instance().finish_chunk(stream);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
