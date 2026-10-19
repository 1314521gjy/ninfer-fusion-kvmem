#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::models::qwen3_8_flash_next::detail {

void flash_next_mtp_stem_add_embedding_launch(const Tensor& emb_proj, const Tensor& hid_proj,
                                             Tensor& mtp_hyper_hidden, cudaStream_t stream);

// Pair embedding[t] with target hidden/position[t-1]. Scalar slots are for prefill;
// tensor slots select independent lanes or successive speculative ring states.
void flash_next_mtp_shift_inputs_launch(const Tensor& target_hidden,
    const Tensor& token_indices, const Tensor& positions, const Tensor& saved_hidden,
    const Tensor& saved_positions, const Tensor& source_slots, int source_slot,
    bool chain, int offset, Tensor& shifted_hidden, Tensor& shifted_indices,
    Tensor& shifted_positions, cudaStream_t stream);

void flash_next_mtp_save_target_launch(const Tensor& target_hidden, const Tensor& positions,
    const Tensor& destination_slots, int destination_slot, Tensor& saved_hidden,
    Tensor& saved_positions, cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
