#pragma once

#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/moe_workspace.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::models::qwen3_8_flash_next::detail {

void flash_next_moe_kernels_launch(const Tensor& input, const MoeWeights& weights,
                                   const FlashNextMoeWorkspace& workspace, Tensor& output,
                                   cudaStream_t stream);

// Decode-arm (tokens <= 8) down-projection kernels. Both produce bitwise-identical BF16 output;
// they differ only in how the eleven per-row partials are spread over warps and CTAs.
//   Legacy   - (320, T) CTAs x 8 warps, one warp per row walking the paths serially.
//   PathWarp - (2560, T) CTAs x 11 warps, one warp per path, thread 0 combines.
enum class FlashNextMoeDownKernel : int { Legacy = 0, PathWarp = 1 };

// The kernel flash_next_moe_kernels_launch uses on the decode arm. Read once per process:
// NINFER_FLASH_NEXT_MOE_DOWN_LEGACY set to a non-empty value other than "0" pins Legacy,
// otherwise PathWarp. Later changes to the environment are not observed.
FlashNextMoeDownKernel flash_next_moe_down_kernel_selection();

// Runs the decode-arm down projection with an explicit kernel choice on an already routed
// workspace (ids, alpha, shared_scale, activations[640, 11, T]) into output [2560, T]. This is
// the hook the bitwise gate in test_moe.cpp uses to run both kernels on identical inputs; the
// production launcher calls it with flash_next_moe_down_kernel_selection(). Throws
// std::invalid_argument unless 1 <= tokens <= 8.
void flash_next_moe_down_launch(FlashNextMoeDownKernel kernel, const MoeWeights& weights,
                                const FlashNextMoeWorkspace& workspace, int tokens,
                                Tensor& output, cudaStream_t stream);

// Compiled/launch attributes of a decode-arm down kernel, for the occupancy report in the test
// (PathWarp is designed for 4 resident CTAs per SM at 352 threads x 40 registers, no spills).
struct FlashNextMoeDownKernelAttributes {
    int threads_per_block         = 0;
    int registers_per_thread      = 0;
    std::size_t local_bytes       = 0; // per thread; nonzero means ptxas spilled
    std::size_t static_smem_bytes = 0;
    int max_blocks_per_sm         = 0; // cudaOccupancyMaxActiveBlocksPerMultiprocessor
};
FlashNextMoeDownKernelAttributes
flash_next_moe_down_kernel_attributes(FlashNextMoeDownKernel kernel);

void flash_next_moe_bf16_kernels_launch(const Tensor& input, const MoeBf16Weights& weights,
                                        const FlashNextMoeWorkspace& workspace, Tensor& output,
                                        cudaStream_t stream);

// ---- Block-5 Stage-2: gguf-block expert leaf support ------------------------------------------
// Helpers of flash_next_moe_gguf_experts (moe.cpp), all on the <=8-token decode domain:
//   gather    -- out column j = x column columns[j] (the expert's token columns), [k, n] layout;
//   silu_mul  -- h = silu(g) * u elementwise (the shared expert's activation; ops::linear_swiglu
//                has no BF16 route, so the pair is folded here);
//   combine   -- out[row, token] = sum_{rank} alpha[token*10+rank] * routed[path_src[path]+row]
//                + shared_scale[token] * shared[row, token] (the 11-path semantics in fixed order).
void flash_next_moe_gguf_gather(const __nv_bfloat16* x, const std::int32_t* columns, int k,
                                int columns_count, __nv_bfloat16* out, cudaStream_t stream);
void flash_next_moe_gguf_silu_mul(const __nv_bfloat16* g, const __nv_bfloat16* u,
                                  __nv_bfloat16* h, int n, cudaStream_t stream);
void flash_next_moe_gguf_combine(const float* routed, const std::int32_t* path_src, int paths,
                                 const float* alpha, const float* shared_scale,
                                 const __nv_bfloat16* shared_src, __nv_bfloat16* out, int tokens,
                                 cudaStream_t stream);

} // namespace ninfer::models::qwen3_8_flash_next::detail
