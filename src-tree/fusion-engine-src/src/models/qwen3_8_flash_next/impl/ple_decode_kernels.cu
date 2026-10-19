#include "models/qwen3_8_flash_next/impl/ple_decode_kernels.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

namespace {

constexpr int kStreamDims    = 2'560;
constexpr int kTotalDims     = 10'240;
constexpr int kStreams       = 4;
constexpr float kInvSqrt2560 = 0.01976423537605237F; // 1.0f / sqrtf(2560.0f)

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFFU, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum_broadcast(float val, float* shared_mem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    val            = warp_reduce_sum(val);
    if (lane == 0) { shared_mem[warp] = val; }
    __syncthreads();
    if (warp == 0) {
        float warp_sum = (lane < 8) ? shared_mem[lane] : 0.0F;
        warp_sum       = warp_reduce_sum(warp_sum);
        if (lane == 0) { shared_mem[0] = warp_sum; }
    }
    __syncthreads();
    const float total = shared_mem[0];
    __syncthreads();
    return total;
}

__global__ void ple_gate_norm_kernel(
    const __nv_bfloat16* __restrict__ hidden, const __nv_bfloat16* __restrict__ projected_key,
    const __nv_bfloat16* __restrict__ projected_value, const __nv_bfloat16* __restrict__ query_norm,
    const __nv_bfloat16* __restrict__ key_norm, const __nv_bfloat16* __restrict__ conv_norm,
    __nv_bfloat16* __restrict__ gated, __nv_bfloat16* __restrict__ normalized_gated, int batch) {
    const int stream_idx = blockIdx.x;
    const int batch_idx  = blockIdx.y;
    const int tid        = threadIdx.x;

    if (stream_idx >= kStreams || batch_idx >= batch) { return; }

    const int stream_offset = stream_idx * kStreamDims;
    const int64_t batch_hidden_offset =
        static_cast<int64_t>(batch_idx) * kTotalDims + stream_offset;
    const int64_t batch_key_offset   = static_cast<int64_t>(batch_idx) * kTotalDims + stream_offset;
    const int64_t batch_val_offset   = static_cast<int64_t>(batch_idx) * kStreamDims;
    const int64_t batch_gated_offset = static_cast<int64_t>(batch_idx) * kTotalDims + stream_offset;
    const int64_t batch_norm_offset  = static_cast<int64_t>(batch_idx) * kTotalDims + stream_offset;

    __shared__ float s_reduce[8];

    // 1. Compute query and key sum of squares
    float sq_q = 0.0F;
    float sq_k = 0.0F;
    for (int d = tid; d < kStreamDims; d += blockDim.x) {
        const float q_val = __bfloat162float(hidden[batch_hidden_offset + d]);
        const float k_val = __bfloat162float(projected_key[batch_key_offset + d]);
        sq_q += q_val * q_val;
        sq_k += k_val * k_val;
    }
    const float total_sq_q = block_reduce_sum_broadcast(sq_q, s_reduce);
    const float total_sq_k = block_reduce_sum_broadcast(sq_k, s_reduce);
    const float r_rms_q    = rsqrtf(total_sq_q / 2560.0F + 1.0e-6F);
    const float r_rms_k    = rsqrtf(total_sq_k / 2560.0F + 1.0e-6F);

    // 2. Compute normalized query and key dot product with explicit BF16 representation boundary
    float dot = 0.0F;
    for (int d = tid; d < kStreamDims; d += blockDim.x) {
        const float q_val        = __bfloat162float(hidden[batch_hidden_offset + d]);
        const float k_val        = __bfloat162float(projected_key[batch_key_offset + d]);
        const float q_norm_scale = 1.0F + __bfloat162float(query_norm[stream_offset + d]);
        const float k_norm_scale = 1.0F + __bfloat162float(key_norm[stream_offset + d]);
        const float nq_fp32      = q_val * r_rms_q * q_norm_scale;
        const float nk_fp32      = k_val * r_rms_k * k_norm_scale;
        const float nq           = __bfloat162float(__float2bfloat16_rn(nq_fp32));
        const float nk           = __bfloat162float(__float2bfloat16_rn(nk_fp32));
        dot += nq * nk;
    }
    const float total_dot    = block_reduce_sum_broadcast(dot, s_reduce);
    const float raw_gate     = total_dot * kInvSqrt2560;
    const float sign         = (raw_gate > 0.0F) ? 1.0F : ((raw_gate < 0.0F) ? -1.0F : 0.0F);
    const float gate         = sign * sqrtf(fmaxf(fabsf(raw_gate), 1.0e-6F));
    const float sigmoid_gate = 1.0F / (1.0F + expf(-gate));

    // 3. Compute gated value and its sum of squares (using represented BF16 gated values)
    float sq_g = 0.0F;
    for (int d = tid; d < kStreamDims; d += blockDim.x) {
        const float val               = __bfloat162float(projected_value[batch_val_offset + d]);
        const float g                 = sigmoid_gate * val;
        const auto g_bf16             = __float2bfloat16_rn(g);
        gated[batch_gated_offset + d] = g_bf16;
        const float g_rep             = __bfloat162float(g_bf16);
        sq_g += g_rep * g_rep;
    }
    const float total_sq_g = block_reduce_sum_broadcast(sq_g, s_reduce);
    const float r_rms_g    = rsqrtf(total_sq_g / 2560.0F + 1.0e-6F);

    // 4. Compute normalized gated value
    for (int d = tid; d < kStreamDims; d += blockDim.x) {
        const float g_rep           = __bfloat162float(gated[batch_gated_offset + d]);
        const float conv_norm_scale = 1.0F + __bfloat162float(conv_norm[stream_offset + d]);
        const float norm_g          = g_rep * r_rms_g * conv_norm_scale;
        normalized_gated[batch_norm_offset + d] = __float2bfloat16_rn(norm_g);
    }
}

__global__ void ple_conv_inject_kernel(
    const __nv_bfloat16* __restrict__ gated, const __nv_bfloat16* __restrict__ normalized_gated,
    const __nv_bfloat16* __restrict__ convolution, const int32_t* __restrict__ source_slots,
    const int32_t* __restrict__ destination_slots, __nv_bfloat16* __restrict__ convolution_states,
    __nv_bfloat16* __restrict__ output, int state_slots, int batch, int batch_offset = 0) {
    const int channel   = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = blockIdx.y + batch_offset;

    if (channel >= kTotalDims || batch_idx >= batch) { return; }

    const int32_t src_slot = source_slots[batch_idx];
    const int32_t dst_slot = destination_slots[batch_idx];

    if (src_slot < 0 || src_slot >= state_slots || dst_slot < 0 || dst_slot >= state_slots) {
        return;
    }

    const int64_t src_base = (static_cast<int64_t>(src_slot) * 9) * kTotalDims + channel;
    const int64_t dst_base = (static_cast<int64_t>(dst_slot) * 9) * kTotalDims + channel;

    // Read taps: 0 (t-9), 3 (t-6), 6 (t-3), plus current normalized gated
    const float h0 = __bfloat162float(convolution_states[src_base + 0 * kTotalDims]);
    const float h1 = __bfloat162float(convolution_states[src_base + 3 * kTotalDims]);
    const float h2 = __bfloat162float(convolution_states[src_base + 6 * kTotalDims]);

    const int64_t batch_channel = static_cast<int64_t>(batch_idx) * kTotalDims + channel;
    const auto cur_norm_bf16    = normalized_gated[batch_channel];
    const float h3              = __bfloat162float(cur_norm_bf16);

    // Convolution weights: [10240, 4]
    const float w0 = __bfloat162float(convolution[0 * kTotalDims + channel]);
    const float w1 = __bfloat162float(convolution[1 * kTotalDims + channel]);
    const float w2 = __bfloat162float(convolution[2 * kTotalDims + channel]);
    const float w3 = __bfloat162float(convolution[3 * kTotalDims + channel]);

    float conv = fmaf(h0, w0, 0.0F);
    conv       = fmaf(h1, w1, conv);
    conv       = fmaf(h2, w2, conv);
    conv       = fmaf(h3, w3, conv);

    // silu(conv) = conv * sigmoid(conv)
    const float silu_conv = conv / (1.0F + expf(-conv));

    // Output = gated + silu(conv)
    const float g_val     = __bfloat162float(gated[batch_channel]);
    output[batch_channel] = __float2bfloat16_rn(g_val + silu_conv);

    // Write destination history: source[1..8] + current
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        convolution_states[dst_base + i * kTotalDims] =
            convolution_states[src_base + (i + 1) * kTotalDims];
    }
    convolution_states[dst_base + 8 * kTotalDims] = cur_norm_bf16;
}

__global__ void ple_conv_inject_chunk_kernel(
    const __nv_bfloat16* __restrict__ gated, const __nv_bfloat16* __restrict__ normalized_gated,
    const __nv_bfloat16* __restrict__ convolution, std::int32_t source_slot,
    std::int32_t destination_slot, __nv_bfloat16* __restrict__ convolution_states,
    __nv_bfloat16* __restrict__ output, int state_slots, int tokens) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kTotalDims) { return; }

    if (source_slot < 0 || source_slot >= state_slots || destination_slot < 0 ||
        destination_slot >= state_slots) {
        return;
    }

    const int64_t src_base = (static_cast<int64_t>(source_slot) * 9) * kTotalDims + channel;
    const int64_t dst_base = (static_cast<int64_t>(destination_slot) * 9) * kTotalDims + channel;

    const float w0 = __bfloat162float(convolution[0 * kTotalDims + channel]);
    const float w1 = __bfloat162float(convolution[1 * kTotalDims + channel]);
    const float w2 = __bfloat162float(convolution[2 * kTotalDims + channel]);
    const float w3 = __bfloat162float(convolution[3 * kTotalDims + channel]);

    // Compute convolution output for each token t in [0, tokens)
    for (int t = 0; t < tokens; ++t) {
        const int tau0 = t - 9;
        const int tau1 = t - 6;
        const int tau2 = t - 3;

        const float h0 = (tau0 < 0)
                             ? __bfloat162float(convolution_states[src_base + (9 + tau0) * kTotalDims])
                             : __bfloat162float(normalized_gated[static_cast<int64_t>(tau0) * kTotalDims + channel]);
        const float h1 = (tau1 < 0)
                             ? __bfloat162float(convolution_states[src_base + (9 + tau1) * kTotalDims])
                             : __bfloat162float(normalized_gated[static_cast<int64_t>(tau1) * kTotalDims + channel]);
        const float h2 = (tau2 < 0)
                             ? __bfloat162float(convolution_states[src_base + (9 + tau2) * kTotalDims])
                             : __bfloat162float(normalized_gated[static_cast<int64_t>(tau2) * kTotalDims + channel]);
        const float h3 = __bfloat162float(normalized_gated[static_cast<int64_t>(t) * kTotalDims + channel]);

        float conv = fmaf(h0, w0, 0.0F);
        conv       = fmaf(h1, w1, conv);
        conv       = fmaf(h2, w2, conv);
        conv       = fmaf(h3, w3, conv);

        const float silu_conv = conv / (1.0F + expf(-conv));
        const float g_val     = __bfloat162float(gated[static_cast<int64_t>(t) * kTotalDims + channel]);
        output[static_cast<int64_t>(t) * kTotalDims + channel] = __float2bfloat16_rn(g_val + silu_conv);
    }

    // Write destination history: last 9 columns of concat(history, normalized_gated)
#pragma unroll
    for (int i = 0; i < 9; ++i) {
        const int tau = tokens - 9 + i;
        const auto val = (tau < 0)
                             ? convolution_states[src_base + (9 + tau) * kTotalDims]
                             : normalized_gated[static_cast<int64_t>(tau) * kTotalDims + channel];
        convolution_states[dst_base + i * kTotalDims] = val;
    }
}

} // namespace

void flash_next_ple_launch(const Tensor& hidden, const Tensor& projected_key,
                           const Tensor& projected_value, const Tensor& query_norm,
                           const Tensor& key_norm, const Tensor& conv_norm,
                           const Tensor& convolution, const Tensor& source_slots,
                           const Tensor& destination_slots, Tensor& convolution_states,
                           Tensor& gated, Tensor& normalized_gated, Tensor& output, int state_slots,
                           int batch, cudaStream_t stream, bool aliased_recurrent_scan) {
    dim3 grid_gate(kStreams, batch);
    ple_gate_norm_kernel<<<grid_gate, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(projected_key.data),
        static_cast<const __nv_bfloat16*>(projected_value.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const __nv_bfloat16*>(key_norm.data),
        static_cast<const __nv_bfloat16*>(conv_norm.data), static_cast<__nv_bfloat16*>(gated.data),
        static_cast<__nv_bfloat16*>(normalized_gated.data), batch);
    CUDA_CHECK(cudaGetLastError());

    if (aliased_recurrent_scan && batch > 1) {
        dim3 grid_conv_single((kTotalDims + 255) / 256, 1);
        for (int r = 0; r < batch; ++r) {
            ple_conv_inject_kernel<<<grid_conv_single, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(gated.data),
                static_cast<const __nv_bfloat16*>(normalized_gated.data),
                static_cast<const __nv_bfloat16*>(convolution.data),
                static_cast<const int32_t*>(source_slots.data),
                static_cast<const int32_t*>(destination_slots.data),
                static_cast<__nv_bfloat16*>(convolution_states.data),
                static_cast<__nv_bfloat16*>(output.data), state_slots, batch, r);
        }
    } else {
        dim3 grid_conv((kTotalDims + 255) / 256, batch);
        ple_conv_inject_kernel<<<grid_conv, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gated.data),
            static_cast<const __nv_bfloat16*>(normalized_gated.data),
            static_cast<const __nv_bfloat16*>(convolution.data),
            static_cast<const int32_t*>(source_slots.data),
            static_cast<const int32_t*>(destination_slots.data),
            static_cast<__nv_bfloat16*>(convolution_states.data),
            static_cast<__nv_bfloat16*>(output.data), state_slots, batch, 0);
    }
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_ple_chunk_launch(const Tensor& hidden, const Tensor& projected_key,
                                 const Tensor& projected_value, const Tensor& query_norm,
                                 const Tensor& key_norm, const Tensor& conv_norm,
                                 const Tensor& convolution, std::int32_t source_slot,
                                 std::int32_t destination_slot, Tensor& convolution_states,
                                 Tensor& gated, Tensor& normalized_gated, Tensor& output,
                                 int state_slots, int tokens, cudaStream_t stream) {
    dim3 grid_gate(kStreams, tokens);
    ple_gate_norm_kernel<<<grid_gate, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data),
        static_cast<const __nv_bfloat16*>(projected_key.data),
        static_cast<const __nv_bfloat16*>(projected_value.data),
        static_cast<const __nv_bfloat16*>(query_norm.data),
        static_cast<const __nv_bfloat16*>(key_norm.data),
        static_cast<const __nv_bfloat16*>(conv_norm.data), static_cast<__nv_bfloat16*>(gated.data),
        static_cast<__nv_bfloat16*>(normalized_gated.data), tokens);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid_conv((kTotalDims + 255) / 256);
    ple_conv_inject_chunk_kernel<<<grid_conv, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gated.data),
        static_cast<const __nv_bfloat16*>(normalized_gated.data),
        static_cast<const __nv_bfloat16*>(convolution.data), source_slot, destination_slot,
        static_cast<__nv_bfloat16*>(convolution_states.data),
        static_cast<__nv_bfloat16*>(output.data), state_slots, tokens);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void ple_dequant_kernel(const std::uint8_t* __restrict__ codes,
                                  const std::uint16_t* __restrict__ scales,
                                  __nv_bfloat162* __restrict__ out,
                                  int total_pairs) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) { return; }

    const int t             = idx / 1280;
    const int pair_in_token = idx % 1280;
    const int head          = pair_in_token / 80;
    const int pair_in_row   = pair_in_token % 80;

    const int row_idx       = t * 16 + head;
    const std::uint8_t packed = codes[row_idx * 80 + pair_in_row];
    const std::uint8_t code0  = packed & 0x0FU;
    const std::uint8_t code1  = static_cast<std::uint8_t>(packed >> 4U);

    const int scale_group          = pair_in_row / 8;
    const std::uint16_t scale_bits = scales[row_idx * 10 + scale_group];

    const __half_raw hr{scale_bits};
    const float scale = __half2float(__half(hr));

    const float v0 = (static_cast<float>(code0) - 8.0F) * scale;
    const float v1 = (static_cast<float>(code1) - 8.0F) * scale;

    const __nv_bfloat16 bf0 = __float2bfloat16_rn(v0);
    const __nv_bfloat16 bf1 = __float2bfloat16_rn(v1);

    out[idx] = __nv_bfloat162(bf0, bf1);
}

void flash_next_ple_dequant_launch(const void* compressed, Tensor& output, int tokens,
                                   cudaStream_t stream) {
    if (tokens <= 0) { return; }
    const int total_pairs = tokens * 1280;
    const auto* codes     = static_cast<const std::uint8_t*>(compressed);
    const auto* scales    = reinterpret_cast<const std::uint16_t*>(codes + tokens * 1280);
    auto* out             = reinterpret_cast<__nv_bfloat162*>(output.data);

    constexpr int kBlock = 256;
    const int grid       = (total_pairs + kBlock - 1) / kBlock;
    ple_dequant_kernel<<<grid, kBlock, 0, stream>>>(codes, scales, out, total_pairs);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
