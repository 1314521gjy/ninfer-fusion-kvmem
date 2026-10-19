#include "models/qwen3_8_flash_next/impl/text_decode_kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

namespace {

__global__ void repeat_embedding_kernel(const __nv_bfloat16* __restrict__ embedding,
                                        __nv_bfloat16* __restrict__ hyper_hidden, int batch) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (d >= 2'560 || b >= batch) return;
    const auto val                 = embedding[static_cast<std::int64_t>(b) * 2'560 + d];
    const std::int64_t base        = static_cast<std::int64_t>(b) * 10'240 + d;
    hyper_hidden[base + 0 * 2'560] = val;
    hyper_hidden[base + 1 * 2'560] = val;
    hyper_hidden[base + 2 * 2'560] = val;
    hyper_hidden[base + 3 * 2'560] = val;
}

} // namespace

void repeat_embedding_to_hyper_streams(const Tensor& embedding, Tensor& hyper_hidden,
                                        cudaStream_t stream) {
    const int batch = embedding.ne[1];
    dim3 grid((2'560 + 255) / 256, batch);
    repeat_embedding_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(embedding.data),
        static_cast<__nv_bfloat16*>(hyper_hidden.data), batch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
