#include "models/qwen3_8_flash_next/impl/qsa_indexer.h"

#include "ninfer/ops/linear.h"

#include "core/layout.h"
#include "models/qwen3_8_flash_next/impl/qsa_indexer_kernels.h"
#include "models/qwen3_8_flash_next/impl/qsa_indexer_workspace.h"
#include "models/qwen3_8_flash_next/impl/stage_ledger.h"

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

} // namespace

std::size_t flash_next_qsa_indexer_workspace_capacity_bytes(std::int32_t maximum_blocks,
                                                            std::int32_t batch) {
    if (maximum_blocks <= 0 || maximum_blocks > 65'536 || batch <= 0 || batch > 8) {
        throw std::invalid_argument("Flash-Next QSA indexer received an invalid envelope");
    }
    const std::size_t sort_temp = flash_next_qsa_indexer_sort_temp_bytes(maximum_blocks, batch);
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_qsa_indexer_workspace(layout, maximum_blocks, batch, sort_temp);
    return layout.peak_bytes(256);
}

void flash_next_qsa_indexer_decode(const Tensor& input, const AttentionWeights& weights,
                                   const Tensor& token_indices, const Tensor& mrope_positions,
                                   const Tensor& table_rows, const Tensor& source_state_slots,
                                   const Tensor& destination_state_slots, QsaIndexerCacheView cache,
                                   std::int32_t maximum_blocks, std::int32_t active_blocks,
                                    WorkspaceArena& workspace, Tensor& selected_blocks,
                                    Tensor& selected_counts, cudaStream_t stream,
                                    bool aliased_recurrent_scan) {
    const std::int32_t batch         = input.ne[1];
    const std::int32_t logical_pages = cache.block_tables.ne[0];
    if (maximum_blocks <= 0 || maximum_blocks > 65'536 || active_blocks < 0 ||
        active_blocks > maximum_blocks || logical_pages < (maximum_blocks + 63) / 64 ||
        !exact_tensor(input, DType::BF16, 2'560, batch) || batch < 1 || batch > 8 ||
        !exact_bf16_weight(weights.indexer_query_key, 640, 2'560) ||
        !exact_tensor(weights.indexer_query_norm, DType::BF16, 128) ||
        !exact_tensor(weights.indexer_key_norm, DType::BF16, 128) ||
        !exact_tensor(token_indices, DType::I32, batch) ||
        !exact_tensor(mrope_positions, DType::I32, batch, 3) ||
        !exact_tensor(table_rows, DType::I32, batch) ||
        !exact_tensor(source_state_slots, DType::I32, batch) ||
        !exact_tensor(destination_state_slots, DType::I32, batch) ||
        cache.block_keys.dtype != DType::BF16 || cache.block_keys.ne[0] != 128 ||
        cache.block_keys.ne[1] != 64 || cache.block_keys.ne[2] <= 0 ||
        cache.block_keys.ne[3] != 1 || !cache.block_keys.is_contiguous() ||
        !aligned_to(cache.block_keys.data, 16) || cache.block_tables.dtype != DType::I32 ||
        logical_pages <= 0 || cache.block_tables.ne[1] <= 0 || cache.block_tables.ne[2] != 1 ||
        cache.block_tables.ne[3] != 1 || !cache.block_tables.is_contiguous() ||
        !aligned_to(cache.block_tables.data, 16) || cache.raw_keys.dtype != DType::BF16 ||
        cache.raw_keys.ne[0] != 128 || cache.raw_keys.ne[1] != 4 || cache.raw_keys.ne[2] <= 0 ||
        cache.raw_keys.ne[3] != 1 || !cache.raw_keys.is_contiguous() ||
        !aligned_to(cache.raw_keys.data, 16) || cache.raw_positions.dtype != DType::I32 ||
        cache.raw_positions.ne[0] != 3 || cache.raw_positions.ne[1] != 4 ||
        cache.raw_positions.ne[2] != cache.raw_keys.ne[2] || cache.raw_positions.ne[3] != 1 ||
        !cache.raw_positions.is_contiguous() || !aligned_to(cache.raw_positions.data, 16) ||
        !exact_tensor(selected_blocks, DType::I32, 512, batch) ||
        !exact_tensor(selected_counts, DType::I32, batch) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next QSA indexer received an invalid exact target view");
    }

    const auto scope            = workspace.scope();
    const std::size_t sort_temp = flash_next_qsa_indexer_sort_temp_bytes(maximum_blocks, batch);
    FlashNextQsaIndexerWorkspace scratch =
        allocate_flash_next_qsa_indexer_workspace(workspace, maximum_blocks, batch, sort_temp);
    ops::linear(input, weights.indexer_query_key, scratch.projected, stream);
    flash_next_qsa_indexer_launch(token_indices, mrope_positions, table_rows, source_state_slots,
                                  destination_state_slots, weights.indexer_query_norm,
                                  weights.indexer_key_norm, cache, scratch, active_blocks,
                                  selected_blocks, selected_counts, stream, aliased_recurrent_scan);
}

void flash_next_qsa_indexer_prefill_chunk(
    const Tensor& input, const AttentionWeights& weights, const Tensor& token_indices,
    const Tensor& mrope_positions, std::int32_t table_row, std::int32_t source_state_slot,
    std::int32_t destination_state_slot, QsaIndexerCacheView cache, std::int32_t maximum_blocks,
    std::int32_t first_token_index, WorkspaceArena& workspace, Tensor& selected_blocks,
    Tensor& selected_counts, cudaStream_t stream) {
    const std::int32_t tokens        = input.ne[1];
    const std::int32_t logical_pages = cache.block_tables.ne[0];
    if (maximum_blocks <= 0 || maximum_blocks > 65'536 ||
        logical_pages < (maximum_blocks + 63) / 64 ||
        !exact_tensor(input, DType::BF16, 2'560, tokens) || tokens <= 0 ||
        !exact_bf16_weight(weights.indexer_query_key, 640, 2'560) ||
        !exact_tensor(weights.indexer_query_norm, DType::BF16, 128) ||
        !exact_tensor(weights.indexer_key_norm, DType::BF16, 128) ||
        !exact_tensor(token_indices, DType::I32, tokens) || table_row < 0 ||
        table_row >= cache.block_tables.ne[1] || source_state_slot < 0 ||
        source_state_slot >= cache.raw_keys.ne[2] || destination_state_slot < 0 ||
        destination_state_slot >= cache.raw_keys.ne[2] || cache.block_keys.dtype != DType::BF16 ||
        cache.block_keys.ne[0] != 128 || cache.block_keys.ne[1] != 64 ||
        cache.block_keys.ne[2] <= 0 || cache.block_keys.ne[3] != 1 ||
        !cache.block_keys.is_contiguous() || !aligned_to(cache.block_keys.data, 16) ||
        cache.block_tables.dtype != DType::I32 || logical_pages <= 0 ||
        cache.block_tables.ne[1] <= 0 || cache.block_tables.ne[2] != 1 ||
        cache.block_tables.ne[3] != 1 || !cache.block_tables.is_contiguous() ||
        !aligned_to(cache.block_tables.data, 16) || cache.raw_keys.dtype != DType::BF16 ||
        cache.raw_keys.ne[0] != 128 || cache.raw_keys.ne[1] != 4 || cache.raw_keys.ne[2] <= 0 ||
        cache.raw_keys.ne[3] != 1 || !cache.raw_keys.is_contiguous() ||
        !aligned_to(cache.raw_keys.data, 16) || cache.raw_positions.dtype != DType::I32 ||
        cache.raw_positions.ne[0] != 3 || cache.raw_positions.ne[1] != 4 ||
        cache.raw_positions.ne[2] != cache.raw_keys.ne[2] || cache.raw_positions.ne[3] != 1 ||
        !cache.raw_positions.is_contiguous() || !aligned_to(cache.raw_positions.data, 16) ||
        !exact_tensor(selected_blocks, DType::I32, 512, tokens) ||
        !exact_tensor(selected_counts, DType::I32, tokens) || first_token_index < 0 ||
        stream == nullptr) {
        throw std::invalid_argument(
            "Flash-Next QSA indexer prefill chunk received an invalid exact target view");
    }

    const auto scope             = workspace.scope();
    const std::int32_t tile_size = flash_next_qsa_indexer_tile_size(maximum_blocks, tokens);
    const std::size_t sort_temp =
        flash_next_qsa_indexer_sort_temp_bytes(maximum_blocks, tile_size);
    FlashNextQsaIndexerWorkspace scratch = allocate_flash_next_qsa_indexer_workspace(
        workspace, maximum_blocks, tokens, tile_size, sort_temp);
    ops::linear(input, weights.indexer_query_key, scratch.projected, stream);
    stage_ledger_record(stream, FlashNextStageId::QSA_Projection);
    flash_next_qsa_indexer_prefill_launch(
        token_indices, mrope_positions, table_row, source_state_slot, destination_state_slot,
        weights.indexer_query_norm, weights.indexer_key_norm, cache, scratch, maximum_blocks,
        first_token_index, selected_blocks, selected_counts, stream);
    stage_ledger_record(stream, FlashNextStageId::QSA_IndexerScoreSelect);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
