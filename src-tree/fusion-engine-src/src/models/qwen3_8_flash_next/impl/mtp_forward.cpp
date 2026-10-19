#include "models/qwen3_8_flash_next/impl/mtp_forward.h"

#include "core/device.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/speculative_round.h"

#include "core/layout.h"
#include "models/qwen3_8_flash_next/impl/hyper_connection.h"
#include "models/qwen3_8_flash_next/impl/hyper_workspace.h"
#include "models/qwen3_8_flash_next/impl/moe.h"
#include "models/qwen3_8_flash_next/impl/moe_workspace.h"
#include "models/qwen3_8_flash_next/impl/mtp_forward_kernels.h"
#include "models/qwen3_8_flash_next/impl/mtp_workspace.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention_workspace.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention_kernels.h"
#include "models/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"
#include "models/qwen3_8_flash_next/impl/qsa_indexer_workspace.h"

#include <iostream>
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

} // namespace

void flash_next_mtp_stem(const MtpModelView& mtp, const Tensor& embedding,
                         const Tensor& backbone_hidden, WorkspaceArena& workspace,
                         Tensor& hyper_hidden, cudaStream_t stream,
                         const FlashNextDecodeStateSink* sink) {
    const int tokens = embedding.ne[1];
    const auto scope = workspace.scope();
    auto ws = allocate_flash_next_mtp_stem_workspace(workspace, tokens);
    auto emit = [&](std::string_view name, const Tensor& tensor) {
        if (sink != nullptr && sink->on_state) { sink->on_state(name, tensor); }
    };
    ops::rmsnorm(embedding, mtp.embedding_norm, 1e-6F, true, ws.embedding_norm, stream);
    emit("mtp_embedding_norm", ws.embedding_norm);
    ops::linear(ws.embedding_norm, mtp.embedding_projection, ws.embedding_projection,
                ops::LinearPolicy::A16Only, workspace, stream);
    emit("mtp_embedding_proj", ws.embedding_projection);
    ops::rmsnorm(backbone_hidden, mtp.hidden_norm, 1e-6F, true, ws.hidden_norm, stream);
    emit("mtp_hidden_norm", ws.hidden_norm);
    // Streams are contiguous [H, 4, tokens]; the same H x H matrix acts on each.
    Tensor input_streams(ws.hidden_norm.data, DType::BF16, {2'560, 4 * tokens});
    Tensor output_streams(ws.hidden_projection.data, DType::BF16, {2'560, 4 * tokens});
    ops::linear(input_streams, mtp.hidden_projection, output_streams,
                ops::LinearPolicy::A16Only, workspace, stream);
    emit("mtp_hidden_proj", ws.hidden_projection);
    flash_next_mtp_stem_add_embedding_launch(ws.embedding_projection, ws.hidden_projection,
                                             hyper_hidden, stream);
    emit("mtp_hyper_init", hyper_hidden);
}
std::size_t flash_next_mtp_teacher_workspace_capacity_bytes(int tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_mtp_teacher_workspace(layout, tokens);
    {
        const auto scope = layout.scope();
        (void)allocate_flash_next_mtp_stem_workspace(layout, tokens);
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            QType::BF16, 2'560, 2'560, ops::LinearPolicy::A16Only, 1, 4 * tokens), 256);
    }
    {
        const auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {640, tokens}, 256);
        (void)layout.alloc(DType::BF16, {13'312, tokens}, 256);
        (void)layout.alloc(DType::BF16, {512, tokens}, 256);
        (void)layout.alloc(DType::BF16, {512, tokens}, 256);
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            QType::BF16, 13'312, 2'560, ops::LinearPolicy::A16Only, 1, tokens), 256);
    }
    return layout.peak_bytes(256);
}

void flash_next_mtp_teacher_extend(const MtpModelView& mtp, const Tensor& embedding,
    const Tensor& target_hidden, const Tensor& token_indices, const Tensor& positions,
    const Tensor& table_rows, const Tensor& source_slots, const Tensor& destination_slots,
    int table_row, int source_slot, int destination_slot, int first_token_index,
    bool prefill, bool aliased_scan, FlashNextDecodeStateView state,
    WorkspaceArena& workspace, cudaStream_t stream) {
    const auto scope = workspace.scope();
    const int offset = prefill && first_token_index == 0 ? 1 : 0;
    const int tokens = embedding.ne[1] - offset;
    if (tokens > 0) {
        auto ws = allocate_flash_next_mtp_teacher_workspace(workspace, tokens);
        flash_next_mtp_shift_inputs_launch(target_hidden, token_indices, positions,
            state.mtp_backbone_hidden, state.mtp_backbone_positions, source_slots, source_slot,
            prefill || aliased_scan, offset, ws.previous_hidden, ws.indices, ws.positions, stream);
        Tensor next_embeddings = embedding.slice(1, offset, tokens);
        flash_next_mtp_stem(mtp, next_embeddings, ws.previous_hidden, workspace,
                             ws.hyper_hidden, stream);
        flash_next_hyper_prepare(ws.hyper_hidden, mtp.attention_hyper, ws.hyper,
                                 ws.attention_input, stream);
        Tensor indexer_projected = workspace.alloc(DType::BF16, {640, tokens}, 256);
        Tensor attention_projected = workspace.alloc(DType::BF16, {13'312, tokens}, 256);
        Tensor key = workspace.alloc(DType::BF16, {512, tokens}, 256);
        Tensor value = workspace.alloc(DType::BF16, {512, tokens}, 256);
        ops::linear(ws.attention_input, mtp.attention.indexer_query_key, indexer_projected,
                    ops::LinearPolicy::A16Only, workspace, stream);
        auto indexer = state.qsa_indexer_caches[kFullAttentionLayers];
        if (prefill) {
            flash_next_qsa_indexer_store_prefill_launch(indexer_projected, ws.indices,
                ws.positions, table_row, source_slot, destination_slot,
                mtp.attention.indexer_key_norm, indexer, stream);
        } else {
            flash_next_qsa_indexer_store_launch(indexer_projected, ws.indices, ws.positions,
                table_rows, source_slots, destination_slots, mtp.attention.indexer_key_norm,
                indexer, stream, aliased_scan);
        }
        ops::linear(ws.attention_input, mtp.attention.query_gate_key_value, attention_projected,
                    ops::LinearPolicy::A16Only, workspace, stream);
        flash_next_qsa_attention_store_launch(attention_projected, ws.indices, ws.positions,
            table_rows, table_row, mtp.attention.key_norm,
            state.qsa_attention_caches[kFullAttentionLayers], key, value, stream);
    }
    flash_next_mtp_save_target_launch(target_hidden, positions, destination_slots,
        destination_slot, state.mtp_backbone_hidden, state.mtp_backbone_positions, stream);
}

std::size_t flash_next_mtp_workspace_capacity_bytes(std::int32_t maximum_blocks,
                                                    std::int32_t batch) {
    if (batch <= 0 || batch > 8) {
        throw std::invalid_argument("Flash-Next MTP received an invalid batch size");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::I32, {512, batch}, 256);
    (void)layout.alloc(DType::I32, {batch}, 256);
    (void)layout.alloc(DType::BF16, {10'240, batch}, 256); // mtp_hyper_hidden
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // attn_in
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // attn_out
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // mlp_in
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // mlp_out
    (void)layout.alloc(DType::BF16, {2'560, batch}, 256);  // mtp_final_hidden

    // Hyper workspace
    (void)allocate_flash_next_hyper_workspace(layout, batch);

    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(
            flash_next_qsa_indexer_workspace_capacity_bytes(maximum_blocks, batch), 256);
    }

    {
        auto scope = layout.scope();
        (void)allocate_flash_next_mtp_stem_workspace(layout, batch);
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            QType::BF16, 2'560, 2'560, ops::LinearPolicy::A16Only, 1, 4 * batch), 256);
    }
    // Attention workspace
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(flash_next_qsa_attention_workspace_capacity_bytes(batch), 256);
    }

    // MoE workspace
    {
        auto scope = layout.scope();
        (void)allocate_flash_next_moe_workspace(layout, batch);
    }

    // Head linear workspace
    {
        auto scope = layout.scope();
        const std::size_t head_ws = ops::linear_workspace_capacity_bytes(
            QType::BF16, 248'320, 2'560, ops::LinearPolicy::A16Only, 1, batch);
        (void)layout.alloc_bytes(head_ws, 256);
    }

    return layout.peak_bytes(256);
}

void flash_next_mtp_step(const TextModelView& model, const Tensor& input_embedding,
                         const Tensor& backbone_hyper_hidden, const Tensor& token_indices,
                         const Tensor& mrope_positions, const Tensor& table_rows,
                         const Tensor& source_slots, const Tensor& destination_slots,
                         QsaIndexerCacheView indexer_cache, QsaAttentionCacheView mtp_cache,
                         std::int32_t maximum_blocks, std::int32_t active_blocks,
                         WorkspaceArena& workspace,
                         Tensor& draft_logits, Tensor& draft_tokens, cudaStream_t stream,
                         const FlashNextDecodeStateSink* sink, Tensor* out_hyper_hidden) {
    if (!model.mtp.has_value()) {
        throw std::invalid_argument("Flash-Next MTP step called but MTP weights are not materialized");
    }
    const auto& mtp = *model.mtp;
    const std::int32_t batch = input_embedding.ne[1];

    if (!exact_tensor(input_embedding, DType::BF16, 2'560, batch) || batch < 1 || batch > 8 ||
        !exact_tensor(backbone_hyper_hidden, DType::BF16, 10'240, batch) ||
        !exact_tensor(token_indices, DType::I32, batch) ||
        !exact_tensor(mrope_positions, DType::I32, batch, 3) ||
        !exact_tensor(table_rows, DType::I32, batch) ||
        !exact_tensor(source_slots, DType::I32, batch) ||
        !exact_tensor(destination_slots, DType::I32, batch) ||
        !exact_tensor(draft_logits, DType::BF16,
                      model.proposal.has_value() ? model.proposal->head.n : 248'320, batch) ||
        !exact_tensor(draft_tokens, DType::I32, batch) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next MTP step received invalid tensor views");
    }

    auto emit_state = [&](std::string_view name, const Tensor& tensor) {
        if (sink != nullptr && sink->on_state) {
            sink->on_state(name, tensor);
        }
    };

    const auto scope = workspace.scope();
    Tensor selected_blocks = workspace.alloc(DType::I32, {512, batch}, 256);
    Tensor selected_counts = workspace.alloc(DType::I32, {batch}, 256);

    // Allocate stage tensors
    Tensor mtp_hyper_hidden = workspace.alloc(DType::BF16, {10'240, batch}, 256);
    Tensor attn_in          = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor attn_out         = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor mlp_in           = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor mlp_out          = workspace.alloc(DType::BF16, {2'560, batch}, 256);
    Tensor mtp_final_hidden = workspace.alloc(DType::BF16, {2'560, batch}, 256);

    FlashNextHyperWorkspace hyper_scratch = allocate_flash_next_hyper_workspace(workspace, batch);

    flash_next_mtp_stem(mtp, input_embedding, backbone_hyper_hidden, workspace,
                         mtp_hyper_hidden, stream, sink);
    // 4. Attention hyper prepare -> attn_in
    flash_next_hyper_prepare(mtp_hyper_hidden, mtp.attention_hyper, hyper_scratch, attn_in, stream);
    emit_state("mtp_attn_block_input", attn_in);

    flash_next_qsa_indexer_decode(attn_in, mtp.attention, token_indices, mrope_positions,
                                  table_rows, source_slots, destination_slots, indexer_cache,
                                  maximum_blocks, active_blocks, workspace, selected_blocks,
                                  selected_counts, stream);
    emit_state("mtp_selected_blocks", selected_blocks);
    emit_state("mtp_selected_counts", selected_counts);

    // 5. QSA Attention decode
    flash_next_qsa_attention_decode(attn_in, mtp.attention, token_indices, mrope_positions,
                                    table_rows, selected_blocks, selected_counts, mtp_cache,
                                    workspace, attn_out, stream);
    emit_state("mtp_attn_block_output", attn_out);

    // 6. Attention hyper inject
    flash_next_hyper_inject(attn_out, hyper_scratch.injection, mtp_hyper_hidden, stream);
    emit_state("mtp_hyper_after_attn", mtp_hyper_hidden);

    // 7. MLP hyper prepare -> mlp_in
    flash_next_hyper_prepare(mtp_hyper_hidden, mtp.mlp_hyper, hyper_scratch, mlp_in, stream);
    emit_state("mtp_mlp_block_input", mlp_in);

    // 8. NVFP4 MoE evaluation
    flash_next_moe(mlp_in, mtp.moe, mlp_out, workspace, stream);
    emit_state("mtp_mlp_block_output", mlp_out);

    // 9. MLP hyper inject
    flash_next_hyper_inject(mlp_out, hyper_scratch.injection, mtp_hyper_hidden, stream);
    emit_state("mtp_hyper_after_mlp", mtp_hyper_hidden);
    emit_state("mtp_multi_hidden", mtp_hyper_hidden);

    if (out_hyper_hidden != nullptr && out_hyper_hidden->data != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(out_hyper_hidden->data, mtp_hyper_hidden.data,
                                   10'240ULL * batch * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    // 10. Final Hyper Mixer -> mtp_final_hidden
    flash_next_hyper_mix(mtp_hyper_hidden, mtp.mixer, hyper_scratch, mtp_final_hidden, stream);
    emit_state("mtp_final_hidden", mtp_final_hidden);

    // 11. Draft Head Linear Projection -> draft_logits
    if (model.proposal.has_value()) {
        const auto& proposal = *model.proposal;
        ops::linear(mtp_final_hidden, proposal.head, draft_logits, ops::LinearPolicy::A16Only,
                    workspace, stream);
        emit_state("mtp_draft_logits", draft_logits);

        // 12. Greedy Argmax -> draft_tokens (within subset)
        ops::argmax(draft_logits, draft_tokens, draft_logits.ne[0], stream);

        // 13. Remap from subset index to true vocabulary ID
        ops::proposal_remap_token_ids(
            draft_tokens, static_cast<const std::int32_t*>(proposal.token_ids.data),
            draft_logits.ne[0], stream);
        emit_state("mtp_draft_tokens", draft_tokens);
    } else {
        ops::linear(mtp_final_hidden, model.output_head, draft_logits, ops::LinearPolicy::A16Only,
                    workspace, stream);
        emit_state("mtp_draft_logits", draft_logits);

        // 12. Greedy Argmax -> draft_tokens
        ops::argmax(draft_logits, draft_tokens, draft_logits.ne[0], stream);
        emit_state("mtp_draft_tokens", draft_tokens);
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
