#include "models/qwen3_8_flash_next/impl/hyper_connection.h"

#include "core/layout.h"
#include "models/qwen3_8_flash_next/impl/hyper_connection_kernels.h"

#include <cstdint>
#include <stdexcept>

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

bool exact_norm(const Tensor& norm) {
    return norm.dtype == DType::BF16 && norm.ne[0] == 10'240 && norm.ne[1] == 1 &&
           norm.ne[2] == 1 && norm.ne[3] == 1 && norm.is_contiguous() && aligned_to(norm.data, 16);
}

void validate_common(const Tensor& hidden, const Tensor& block_input,
                     const FlashNextHyperWorkspace& scratch) {
    const std::int32_t tokens = hidden.ne[1];
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != 10'240 || hidden.ne[2] != 1 ||
        hidden.ne[3] != 1 || tokens < 1 || !hidden.is_contiguous() ||
        !aligned_to(hidden.data, 16) || block_input.dtype != DType::BF16 ||
        block_input.ne[0] != 2'560 || block_input.ne[1] != tokens || block_input.ne[2] != 1 ||
        block_input.ne[3] != 1 || !block_input.is_contiguous() ||
        !aligned_to(block_input.data, 16) || scratch.normalized.dtype != DType::BF16 ||
        scratch.normalized.ne[0] != 10'240 || scratch.normalized.ne[1] != tokens ||
        scratch.low_rank.dtype != DType::BF16 || scratch.low_rank.ne[0] != 320 ||
        scratch.low_rank.ne[1] != tokens || scratch.injection.dtype != DType::FP32 ||
        scratch.injection.ne[0] != 4 || scratch.injection.ne[1] != tokens ||
        scratch.up_gemm.dtype != DType::BF16 || scratch.up_gemm.ne[0] != 10'240 ||
        scratch.up_gemm.ne[1] != tokens || scratch.down_split.dtype != DType::FP32 ||
        scratch.down_split.ne[0] != 320 ||
        scratch.down_split.ne[1] != tokens * kHyperDownSplitK ||
        !scratch.normalized.is_contiguous() || !scratch.low_rank.is_contiguous() ||
        !scratch.injection.is_contiguous() || !scratch.up_gemm.is_contiguous() ||
        !scratch.down_split.is_contiguous() ||
        !aligned_to(scratch.normalized.data, 16) ||
        !aligned_to(scratch.low_rank.data, 16) || !aligned_to(scratch.injection.data, 16) ||
        !aligned_to(scratch.up_gemm.data, 16) || !aligned_to(scratch.down_split.data, 16)) {
        throw std::invalid_argument("Flash-Next hyper connection received invalid exact tensors");
    }
}

} // namespace

std::size_t flash_next_hyper_workspace_capacity_bytes(std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Flash-Next hyper workspace requires positive tokens");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_flash_next_hyper_workspace(layout, max_tokens);
    return layout.peak_bytes(256);
}

void flash_next_hyper_prepare(const Tensor& hidden, const HyperConnectionWeights& weights,
                              FlashNextHyperWorkspace& scratch, Tensor& block_input,
                              cudaStream_t stream) {
    validate_common(hidden, block_input, scratch);
    if (!exact_norm(weights.norm) || !exact_bf16_weight(weights.input_mix_down, 320, 10'240) ||
        !exact_bf16_weight(weights.input_mix_up, 10'240, 320) ||
        !exact_bf16_weight(weights.block_inject, 4, 10'240) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next hyper connection received invalid weights");
    }
    flash_next_hyper_prepare_launch(hidden, weights, scratch, block_input, stream);
}

void flash_next_hyper_inject(const Tensor& block_output, const Tensor& injection, Tensor& hidden,
                             cudaStream_t stream) {
    const std::int32_t tokens = hidden.ne[1];
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != 10'240 || hidden.ne[2] != 1 ||
        hidden.ne[3] != 1 || tokens < 1 || !hidden.is_contiguous() ||
        !aligned_to(hidden.data, 16) || block_output.dtype != DType::BF16 ||
        block_output.ne[0] != 2'560 || block_output.ne[1] != tokens ||
        !block_output.is_contiguous() || !aligned_to(block_output.data, 16) ||
        injection.dtype != DType::FP32 || injection.ne[0] != 4 || injection.ne[1] != tokens ||
        !injection.is_contiguous() || !aligned_to(injection.data, 16) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next hyper injection received invalid exact tensors");
    }
    flash_next_hyper_inject_launch(block_output, injection, hidden, stream);
}

void flash_next_hyper_mix(const Tensor& hidden, const HyperMixerWeights& weights,
                          FlashNextHyperWorkspace& scratch, Tensor& block_input,
                          cudaStream_t stream) {
    validate_common(hidden, block_input, scratch);
    if (!exact_norm(weights.norm) || !exact_bf16_weight(weights.input_mix_down, 320, 10'240) ||
        !exact_bf16_weight(weights.input_mix_up, 10'240, 320) || stream == nullptr) {
        throw std::invalid_argument("Flash-Next hyper mixer received invalid weights");
    }
    flash_next_hyper_mix_launch(hidden, weights, scratch, block_input, stream);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
