#include "models/qwen3_8_flash_next/impl/qsa_attention.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/selected_block_attention.h"

#include "core/layout.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention_kernels.h"
#include "models/qwen3_8_flash_next/impl/qsa_attention_workspace.h"
#include "models/qwen3_8_flash_next/impl/stage_ledger.h"

#include <cstddef>
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
    const std::uint64_t codes        = static_cast<std::uint64_t>(rows) * columns;
    const std::uint64_t scale_offset = (codes + 255U) & ~std::uint64_t{255U};
    const auto* payload              = static_cast<const std::byte*>(weight.payload);
    const std::int64_t scale_stride  = static_cast<std::int64_t>(rows) * 4;
    // QType::FP8_E4M3FN_ROW_F32, not the 2-byte `..._ROW_BF16`: this artifact's FP8 weights carry
    // 4-byte `float` row scales (`fp8_e4m3fn_row_f32`), which is what the rest of this predicate
    // already demands (`scale_dtype == DType::FP32`, `scale_nb[0] == 4`). Naming the 2-byte qtype here
    // made the gate unsatisfiable - `artifact/typed_binding.cpp` can only produce the 4-byte one.
    return weight.qtype == QType::FP8_E4M3FN_ROW_F32 && weight.layout == QuantLayout::RowScale &&
           weight.scale_dtype == DType::FP32 && weight.group_size == columns &&
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

bool exact_projection_weight(const Weight& weight, std::int32_t rows, std::int32_t columns) {
    if (weight.qtype == QType::BF16) {
        return exact_bf16_weight(weight, rows, columns);
    }
    return exact_fp8_f32_weight(weight, rows, columns);
}

} // namespace

std::size_t flash_next_qsa_attention_workspace_capacity_bytes(std::int32_t batch) {
    if (batch <= 0 || batch > 262'144) {
        throw std::invalid_argument("Flash-Next QSA attention received an invalid batch/tokens size");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_qsa_attention_workspace(layout, batch);
    {
        auto scope = layout.scope();
        const std::size_t qgkv_ws = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, 13'312, 2'560, ops::LinearPolicy::AllowA8, 1, batch);
        const std::size_t out_ws = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, 2'560, 6'144, ops::LinearPolicy::AllowA8, 1, batch);
        const std::size_t attention_ws = batch <= 8
            ? ops::selected_block_attention_workspace_capacity_bytes(batch) : 0;
        (void)layout.alloc_bytes(std::max({qgkv_ws, out_ws, attention_ws}), 256);
    }
    return layout.peak_bytes(256);
}

void flash_next_qsa_attention_decode(const Tensor& input, const AttentionWeights& weights,
                                     const Tensor& token_indices, const Tensor& mrope_positions,
                                     const Tensor& table_rows, const Tensor& selected_blocks,
                                     const Tensor& selected_counts, QsaAttentionCacheView cache,
                                     WorkspaceArena& workspace, Tensor& output,
                                     cudaStream_t stream) {
    const std::int32_t batch = input.ne[1];
    const auto key_dt = cache.key_pages.dtype;
    if (!exact_tensor(input, DType::BF16, 2'560, batch) || batch < 1 || batch > 8 ||
        !exact_tensor(output, DType::BF16, 2'560, batch) ||
        !exact_tensor(weights.query_norm, DType::BF16, 256) ||
        !exact_tensor(weights.key_norm, DType::BF16, 256) ||
        !exact_projection_weight(weights.query_gate_key_value, 13'312, 2'560) ||
        !exact_projection_weight(weights.output, 2'560, 6'144) ||
        !exact_tensor(token_indices, DType::I32, batch) ||
        !exact_tensor(mrope_positions, DType::I32, batch, 3) ||
        !exact_tensor(table_rows, DType::I32, batch) ||
        !exact_tensor(selected_blocks, DType::I32, 512, batch) ||
        !exact_tensor(selected_counts, DType::I32, batch) ||
        (key_dt != DType::BF16 && key_dt != DType::FP8_E4M3FN) ||
        cache.key_pages.ne[0] != 256 || cache.key_pages.ne[1] != 64 || cache.key_pages.ne[2] != 2 ||
        cache.key_pages.ne[3] <= 0 || !cache.key_pages.is_contiguous() ||
        !aligned_to(cache.key_pages.data, 16) || cache.value_pages.dtype != key_dt ||
        cache.value_pages.ne[0] != 256 || cache.value_pages.ne[1] != 64 ||
        cache.value_pages.ne[2] != 2 || cache.value_pages.ne[3] != cache.key_pages.ne[3] ||
        !cache.value_pages.is_contiguous() || !aligned_to(cache.value_pages.data, 16) ||
        cache.block_tables.dtype != DType::I32 || cache.block_tables.ne[0] <= 0 ||
        cache.block_tables.ne[1] <= 0 || cache.block_tables.ne[2] != 1 ||
        cache.block_tables.ne[3] != 1 || !cache.block_tables.is_contiguous() ||
        !aligned_to(cache.block_tables.data, 16) || stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next QSA attention received an invalid exact target view");
    }

    const auto scope = workspace.scope();
    FlashNextQsaAttentionWorkspace scratch =
        allocate_flash_next_qsa_attention_workspace(workspace, batch);
    ops::linear(input, weights.query_gate_key_value, scratch.projected, ops::LinearPolicy::A16Only,
                workspace, stream);
    flash_next_qsa_attention_launch(token_indices, mrope_positions, table_rows, selected_blocks,
                                    selected_counts, weights.query_norm, weights.key_norm, cache,
                                    scratch, workspace, stream);
    ops::linear(scratch.gated, weights.output, output, ops::LinearPolicy::A16Only, workspace,
                stream);
}

void flash_next_qsa_attention_prefill_chunk(
    const Tensor& input, const AttentionWeights& weights, const Tensor& token_indices,
    const Tensor& mrope_positions, std::int32_t table_row, const Tensor& selected_blocks,
    const Tensor& selected_counts, QsaAttentionCacheView cache, WorkspaceArena& workspace,
    Tensor& output, cudaStream_t stream, const QsaStageEmitter& emit, bool use_mma) {
    const std::int32_t tokens = input.ne[1];
    const auto key_dt = cache.key_pages.dtype;
    if (tokens <= 0 || !exact_tensor(input, DType::BF16, 2'560, tokens) ||
        !exact_tensor(output, DType::BF16, 2'560, tokens) ||
        !exact_tensor(weights.query_norm, DType::BF16, 256) ||
        !exact_tensor(weights.key_norm, DType::BF16, 256) ||
        !exact_projection_weight(weights.query_gate_key_value, 13'312, 2'560) ||
        !exact_projection_weight(weights.output, 2'560, 6'144) ||
        !exact_tensor(token_indices, DType::I32, tokens) ||
        table_row < 0 || table_row >= cache.block_tables.ne[1] ||
        !exact_tensor(selected_blocks, DType::I32, 512, tokens) ||
        !exact_tensor(selected_counts, DType::I32, tokens) ||
        (key_dt != DType::BF16 && key_dt != DType::FP8_E4M3FN) ||
        cache.key_pages.ne[0] != 256 || cache.key_pages.ne[1] != 64 || cache.key_pages.ne[2] != 2 ||
        cache.key_pages.ne[3] <= 0 || !cache.key_pages.is_contiguous() ||
        !aligned_to(cache.key_pages.data, 16) || cache.value_pages.dtype != key_dt ||
        cache.value_pages.ne[0] != 256 || cache.value_pages.ne[1] != 64 ||
        cache.value_pages.ne[2] != 2 || cache.value_pages.ne[3] != cache.key_pages.ne[3] ||
        !cache.value_pages.is_contiguous() || !aligned_to(cache.value_pages.data, 16) ||
        cache.block_tables.dtype != DType::I32 || cache.block_tables.ne[0] <= 0 ||
        cache.block_tables.ne[1] <= 0 || cache.block_tables.ne[2] != 1 ||
        cache.block_tables.ne[3] != 1 || !cache.block_tables.is_contiguous() ||
        !aligned_to(cache.block_tables.data, 16) || stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next QSA attention prefill chunk received an invalid exact target view");
    }

    const auto scope = workspace.scope();
    FlashNextQsaAttentionWorkspace scratch =
        allocate_flash_next_qsa_attention_workspace(workspace, tokens);
    const ops::LinearPolicy linear_policy =
        tokens >= 16 ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::A16Only;
    ops::linear(input, weights.query_gate_key_value, scratch.projected, linear_policy, workspace,
                stream);
    stage_ledger_record(stream, FlashNextStageId::QSA_Projection);
    if (emit) { emit("qsa_projected", scratch.projected); }
    flash_next_qsa_attention_prefill_launch(token_indices, mrope_positions, table_row, selected_blocks,
                                            selected_counts, weights.query_norm, weights.key_norm,
                                            cache, scratch, stream, use_mma);
    stage_ledger_record(stream, FlashNextStageId::QSA_MainAttention);
    if (emit) {
        emit("qsa_query", scratch.query);
        emit("qsa_gate", scratch.gate);
        emit("qsa_key", scratch.key);
        emit("qsa_value", scratch.value);
        emit("qsa_attended", scratch.attended);
        emit("qsa_gated", scratch.gated);
    }
    ops::linear(scratch.gated, weights.output, output, linear_policy, workspace, stream);
    stage_ledger_record(stream, FlashNextStageId::QSA_OutputProjection);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
