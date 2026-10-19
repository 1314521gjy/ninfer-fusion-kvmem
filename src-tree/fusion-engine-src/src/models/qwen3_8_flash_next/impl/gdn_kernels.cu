#include "models/qwen3_8_flash_next/impl/gdn_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

constexpr int kQkRows       = 2'048;
constexpr int kValueRows    = 6'144;
constexpr int kConvChannels = 10'240;
constexpr int kProjected    = 16'384;

__global__ void conv_split_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ convolution,
    const std::int32_t* __restrict__ source_slots,
    const std::int32_t* __restrict__ destination_slots, __nv_bfloat16* __restrict__ states,
    __nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value, __nv_bfloat16* __restrict__ z,
    int batch_offset = 0) {
    const int channel =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int batch = static_cast<int>(blockIdx.y) + batch_offset;
    if (channel >= kConvChannels) { return; }
    const int source               = source_slots[batch];
    const int destination          = destination_slots[batch];
    const std::int64_t source_base = static_cast<std::int64_t>(source) * kConvChannels * 3;
    const std::int64_t destination_base =
        static_cast<std::int64_t>(destination) * kConvChannels * 3;
    const float h0          = __bfloat162float(states[source_base + channel]);
    const float h1          = __bfloat162float(states[source_base + kConvChannels + channel]);
    const float h2          = __bfloat162float(states[source_base + 2 * kConvChannels + channel]);
    const auto current_bf16 = projected[static_cast<std::int64_t>(batch) * kProjected + channel];
    const float current     = __bfloat162float(current_bf16);
    float sum               = h0 * __bfloat162float(convolution[channel]);
    sum                     = fmaf(h1, __bfloat162float(convolution[kConvChannels + channel]), sum);
    sum = fmaf(h2, __bfloat162float(convolution[2 * kConvChannels + channel]), sum);
    sum = fmaf(current, __bfloat162float(convolution[3 * kConvChannels + channel]), sum);
    states[destination_base + channel] = states[source_base + kConvChannels + channel];
    states[destination_base + kConvChannels + channel] =
        states[source_base + 2 * kConvChannels + channel];
    states[destination_base + 2 * kConvChannels + channel] = current_bf16;
    const __nv_bfloat16 activated                          = __float2bfloat16_rn(ops::silu(sum));
    const std::int64_t batch_qk = static_cast<std::int64_t>(batch) * kQkRows;
    const std::int64_t batch_v  = static_cast<std::int64_t>(batch) * kValueRows;
    if (channel < kQkRows) {
        query[batch_qk + channel] = activated;
    } else if (channel < 2 * kQkRows) {
        key[batch_qk + channel - kQkRows] = activated;
    } else {
        value[batch_v + channel - 2 * kQkRows] = activated;
    }
    if (channel < kValueRows) {
        z[batch_v + channel] =
            projected[static_cast<std::int64_t>(batch) * kProjected + kConvChannels + channel];
    }
}

__global__ void controls_kernel(const __nv_bfloat16* __restrict__ input,
                                const __nv_bfloat16* __restrict__ weight,
                                const __nv_bfloat16* __restrict__ a_log,
                                const __nv_bfloat16* __restrict__ dt_bias, float* __restrict__ g,
                                float* __restrict__ beta) {
    const int batch = static_cast<int>(blockIdx.y);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int head  = static_cast<int>(blockIdx.x) * 8 + warp;
    const auto* x   = input + static_cast<std::int64_t>(batch) * 2'560;
    float a         = 0.0F;
    float b         = 0.0F;
    for (int column = lane; column < 2'560; column += 32) {
        const float v = __bfloat162float(x[column]);
        a = fmaf(__bfloat162float(weight[static_cast<std::int64_t>(head) * 2'560 + column]), v, a);
        b = fmaf(__bfloat162float(weight[static_cast<std::int64_t>(head + 48) * 2'560 + column]), v,
                 b);
    }
    a = ops::warp_reduce_sum(a);
    b = ops::warp_reduce_sum(b);
    if (lane == 0) {
        const std::int64_t index = static_cast<std::int64_t>(batch) * 48 + head;
        g[index]                 = -expf(__bfloat162float(a_log[head])) *
                   ops::softplus(a + __bfloat162float(dt_bias[head]));
        beta[index] = ops::sigmoid(b);
    }
}

__global__ void output_gate_kernel(const __nv_bfloat16* __restrict__ recurrent,
                                   const __nv_bfloat16* __restrict__ z,
                                   const __nv_bfloat16* __restrict__ norm,
                                   __nv_bfloat16* __restrict__ gated) {
    __shared__ float squares[4];
    const int head          = static_cast<int>(blockIdx.x);
    const int batch         = static_cast<int>(blockIdx.y);
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int dim           = warp * 32 + lane;
    const std::int64_t base = static_cast<std::int64_t>(batch) * kValueRows + head * 128;
    const float x           = __bfloat162float(recurrent[base + dim]);
    float square            = ops::warp_reduce_sum(x * x);
    if (lane == 0) { squares[warp] = square; }
    __syncthreads();
    const float sum        = squares[0] + squares[1] + squares[2] + squares[3];
    const float normalized = x * rsqrtf(sum / 128.0F + 1.0e-6F) * __bfloat162float(norm[dim]);
    gated[base + dim] =
        __float2bfloat16_rn(normalized * ops::sigmoid(__bfloat162float(z[base + dim])));
}

} // namespace

void flash_next_gdn_conv_launch(const FlashNextGdnWorkspace& scratch, const Tensor& convolution,
                                const Tensor& source_slots, const Tensor& destination_slots,
                                Tensor& convolution_states, cudaStream_t stream,
                                int batch_count, int batch_offset) {
    constexpr int threads = 256;
    const unsigned b_count = (batch_count > 0) ? static_cast<unsigned>(batch_count)
                                               : static_cast<unsigned>(scratch.projected.ne[1]);
    conv_split_kernel<<<dim3((kConvChannels + threads - 1) / threads, b_count),
                        threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(scratch.projected.data),
        static_cast<const __nv_bfloat16*>(convolution.data),
        static_cast<const std::int32_t*>(source_slots.data),
        static_cast<const std::int32_t*>(destination_slots.data),
        static_cast<__nv_bfloat16*>(convolution_states.data),
        static_cast<__nv_bfloat16*>(scratch.query.data),
        static_cast<__nv_bfloat16*>(scratch.key.data),
        static_cast<__nv_bfloat16*>(scratch.value.data),
        static_cast<__nv_bfloat16*>(scratch.z.data),
        batch_offset);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_gdn_controls_launch(const Tensor& input, const GdnWeights& weights,
                                    FlashNextGdnWorkspace& scratch, cudaStream_t stream) {
    controls_kernel<<<dim3(6, static_cast<unsigned>(input.ne[1])), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(weights.a_b_projection.qdata),
        static_cast<const __nv_bfloat16*>(weights.a_log.data),
        static_cast<const __nv_bfloat16*>(weights.dt_bias.data),
        static_cast<float*>(scratch.g.data), static_cast<float*>(scratch.beta.data));
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_gdn_output_gate_launch(const FlashNextGdnWorkspace& scratch, const Tensor& norm,
                                       cudaStream_t stream) {
    output_gate_kernel<<<dim3(48, static_cast<unsigned>(scratch.recurrent_output.ne[1])), 128, 0,
                         stream>>>(static_cast<const __nv_bfloat16*>(scratch.recurrent_output.data),
                                   static_cast<const __nv_bfloat16*>(scratch.z.data),
                                   static_cast<const __nv_bfloat16*>(norm.data),
                                   static_cast<__nv_bfloat16*>(scratch.gated_output.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
