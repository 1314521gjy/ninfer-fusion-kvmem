#include "models/qwen3_8_flash_next/impl/ple_decode.h"

#include "ninfer/ops/linear.h"

#include "core/layout.h"
#include "models/qwen3_8_flash_next/impl/ple_decode_kernels.h"
#include "models/qwen3_8_flash_next/impl/ple_workspace.h"

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

} // namespace

std::size_t flash_next_ple_workspace_capacity_bytes(std::int32_t tokens) {
    if (tokens <= 0 || tokens > 512) {
        throw std::invalid_argument("Flash-Next PLE requires tokens in [1,512]");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_ple_workspace(layout, tokens);
    return layout.peak_bytes(256);
}

void flash_next_ple_decode(const Tensor& hidden, const Tensor& gathered_embedding,
                           const PleWeights& weights, const Tensor& source_slots,
                           const Tensor& destination_slots, Tensor& convolution_states,
                           WorkspaceArena& workspace, Tensor& output, cudaStream_t stream,
                           bool aliased_recurrent_scan) {
    const std::int32_t batch       = hidden.ne[1];
    const std::int32_t state_slots = convolution_states.ne[2];
    if (!exact_tensor(hidden, DType::BF16, 10'240, batch) || batch < 1 || batch > 8 ||
        !exact_tensor(gathered_embedding, DType::BF16, 2'560, batch) ||
        !exact_tensor(output, DType::BF16, 10'240, batch) ||
        !exact_bf16_weight(weights.key_projection, 10'240, 2'560) ||
        !exact_bf16_weight(weights.value_projection, 2'560, 2'560) ||
        !exact_tensor(weights.query_norm, DType::BF16, 10'240) ||
        !exact_tensor(weights.key_norm, DType::BF16, 10'240) ||
        !exact_tensor(weights.conv_norm, DType::BF16, 10'240) ||
        !exact_tensor(weights.convolution, DType::BF16, 10'240, 4) ||
        !exact_tensor(source_slots, DType::I32, batch) ||
        !exact_tensor(destination_slots, DType::I32, batch) ||
        !exact_tensor(convolution_states, DType::BF16, 10'240, 9, state_slots) ||
        state_slots <= 0 || stream == nullptr) {
        throw std::invalid_argument("Flash-Next PLE received an invalid exact target view");
    }

    const auto scope              = workspace.scope();
    FlashNextPleWorkspace scratch = allocate_flash_next_ple_workspace(workspace, batch);

    ops::linear(gathered_embedding, weights.key_projection, scratch.projected_key,
                ops::LinearPolicy::A16Only, workspace, stream);
    ops::linear(gathered_embedding, weights.value_projection, scratch.projected_value,
                ops::LinearPolicy::A16Only, workspace, stream);

    flash_next_ple_launch(hidden, scratch.projected_key, scratch.projected_value,
                          weights.query_norm, weights.key_norm, weights.conv_norm,
                          weights.convolution, source_slots, destination_slots, convolution_states,
                          scratch.gated, scratch.normalized_gated, output, state_slots, batch,
                          stream, aliased_recurrent_scan);
}

void flash_next_ple_prefill_chunk(const Tensor& hidden, const Tensor& gathered_embedding,
                                  const PleWeights& weights, std::int32_t source_slot,
                                  std::int32_t destination_slot, Tensor& convolution_states,
                                  WorkspaceArena& workspace, Tensor& output, cudaStream_t stream) {
    const std::int32_t tokens      = hidden.ne[1];
    const std::int32_t state_slots = convolution_states.ne[2];
    if (!exact_tensor(hidden, DType::BF16, 10'240, tokens) || tokens < 1 ||
        !exact_tensor(gathered_embedding, DType::BF16, 2'560, tokens) ||
        !exact_tensor(output, DType::BF16, 10'240, tokens) ||
        !exact_bf16_weight(weights.key_projection, 10'240, 2'560) ||
        !exact_bf16_weight(weights.value_projection, 2'560, 2'560) ||
        !exact_tensor(weights.query_norm, DType::BF16, 10'240) ||
        !exact_tensor(weights.key_norm, DType::BF16, 10'240) ||
        !exact_tensor(weights.conv_norm, DType::BF16, 10'240) ||
        !exact_tensor(weights.convolution, DType::BF16, 10'240, 4) ||
        !exact_tensor(convolution_states, DType::BF16, 10'240, 9, state_slots) ||
        source_slot < 0 || source_slot >= state_slots || destination_slot < 0 ||
        destination_slot >= state_slots || stream == nullptr) {
        throw std::invalid_argument("Flash-Next PLE prefill chunk received an invalid view");
    }

    const auto scope              = workspace.scope();
    FlashNextPleWorkspace scratch = allocate_flash_next_ple_workspace(workspace, tokens);

    ops::linear(gathered_embedding, weights.key_projection, scratch.projected_key,
                ops::LinearPolicy::A16Only, workspace, stream);
    ops::linear(gathered_embedding, weights.value_projection, scratch.projected_value,
                ops::LinearPolicy::A16Only, workspace, stream);

    flash_next_ple_chunk_launch(hidden, scratch.projected_key, scratch.projected_value,
                                weights.query_norm, weights.key_norm, weights.conv_norm,
                                weights.convolution, source_slot, destination_slot,
                                convolution_states, scratch.gated, scratch.normalized_gated,
                                output, state_slots, tokens, stream);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
