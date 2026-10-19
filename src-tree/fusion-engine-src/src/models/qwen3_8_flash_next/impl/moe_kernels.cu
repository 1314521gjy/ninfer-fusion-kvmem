#include "models/qwen3_8_flash_next/impl/moe_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "models/qwen3_8_flash_next/impl/moe_shared_kernels.h"
#include "models/qwen3_8_flash_next/impl/stage_ledger.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
// The staging switch below reads the environment; these were relied on
// transitively before and are now named explicitly.
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

using Activation2560Geometry = ops::detail::Nvfp4ActivationGeometry<2'560>;
using Activation640Geometry  = ops::detail::Nvfp4ActivationGeometry<640>;

using GateGeometry = ops::detail::Nvfp4GemvGeometry<1'280, 2'560>;
using GateSchedule =
    ops::detail::Nvfp4GemvSchedule<8, 2, 16, 4, ops::detail::Nvfp4ScaleAccess::Direct,
                                   ops::detail::Nvfp4CodeCache::Default, 1>;
using DownGeometry = ops::detail::Nvfp4GemvGeometry<2'560, 640>;

constexpr int kHidden       = 2'560;
constexpr int kIntermediate = 640;
constexpr int kTopK         = 10;
constexpr int kPaths        = kTopK + 1;
constexpr int kDownWarps    = 8;
// Path-per-warp decode down kernel: one CTA per (row, token); warps 0..9 own the ten routed
// paths and warp 10 the shared expert, so the CTA is exactly the eleven partials one row needs.
constexpr int kDownPathWarps   = kPaths;
constexpr int kDownPathThreads = kDownPathWarps * 32;
// Resident CTAs per SM the path-per-warp kernel is compiled for. 4 x 352 threads = 1,408 of the
// 1,536 per SM (CC 12.0) and 4 x 352 x 40 registers = 56,320 of 65,536; the register allocation
// granularity (8 per thread) means this bound effectively pins ptxas at the legacy kernel's 40
// registers per thread -- the design's 46 would already force 3 CTAs per SM.
constexpr int kDownPathMinBlocksPerSm = 4;

template <int K>
__device__ __forceinline__ void dot_bf16_pair(const __nv_bfloat16* x, const __nv_bfloat16* first,
                                              const __nv_bfloat16* second, float& first_result,
                                              float& second_result) {
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    float first_sum  = 0.0F;
    float second_sum = 0.0F;
    for (int column = lane; column < K; column += 32) {
        const float value = __bfloat162float(x[column]);
        first_sum         = fmaf(__bfloat162float(first[column]), value, first_sum);
        second_sum        = fmaf(__bfloat162float(second[column]), value, second_sum);
    }
    first_result  = ops::warp_reduce_sum(first_sum);
    second_result = ops::warp_reduce_sum(second_sum);
}

template <int K>
__device__ __forceinline__ float dot_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    float sum      = 0.0F;
    for (int column = lane; column < K; column += 32) {
        sum = fmaf(__bfloat162float(weight[column]), __bfloat162float(x[column]), sum);
    }
    return ops::warp_reduce_sum(sum);
}


// ---------------------------------------------------------------------------------------------
// Instruction-count experiment on the decode gate/up inner loop (default OFF).
//
// The shipped loop folds the K16 group scale into every FMA's multiplicand
// (`fmaf(code.x * coefficient, activation.x, accumulators)`), which is an FMUL plus an FFMA -- two
// instructions per MAC. The sibling ternary engine line (E:\infer-docs, Bonsai 涓夊厓棣栬窇-鏍瑰洜涓庝慨澶?
// :279) measured that fixing exactly this -- "only remove instructions, never add state" -- took its
// T=1 decode from 31.0 to 62.1 tok/s, and that the winning items are hoisting the per-group scale
// multiply OUT of the FMA chain and reading the scale once per group.
//
// This variant keeps the raw code x activation products in two independent chains per row and
// applies the group scale once per K16 group: 16 FFMA + 2 FADD + 1 FFMA = 19 instructions per 16
// values, against 16 FMUL + 16 FFMA = 32. It is deliberately LOCAL to this translation unit so that
// every other caller of ops/linear/nvfp4/nvfp4_gemv.cuh stays bit-identical (no header edit, no
// ODR/rebuild-exposure risk). The reassociation changes the numerics, so it is gated like any other
// numeric change: NINFER_FLASH_NEXT_MOE_HOIST=1 turns it on, default off.
[[nodiscard]] bool flash_next_moe_hoist_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_FLASH_NEXT_MOE_HOIST");
        return value != nullptr && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void compute_nvfp4_rows_hoisted(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    const ops::detail::Nvfp4GemvSharedStorage<Geometry, Schedule>& shared,
    float inverse_weight_divisor, const int (&parent_rows)[Schedule::kRowsPerWarp], int flat_row0,
    int lane, float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains]) {
    constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    constexpr int kPhases         = Geometry::kInputRows / kValuesPerPhase;
    constexpr int kGroupsPerLane =
        Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16;
    static_assert(kGroupsPerLane == 1, "hoisted variant assumes one K16 group per lane phase");
    constexpr int kChainMask     = Schedule::kAccumulatorChains - 1;
    const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(x);

#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        float coefficients[Schedule::kRowsPerWarp][1];
        ops::detail::Nvfp4CodePack<Schedule::kValuesPerLane> row_codes[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            ops::detail::load_nvfp4_coefficients<Geometry, Schedule>(
                scales, shared, parent_rows[local_row], flat_row0 + local_row, phase, lane,
                inverse_weight_divisor, coefficients[local_row]);
            const std::int64_t code_offset =
                static_cast<std::int64_t>(parent_rows[local_row]) * Geometry::kCodeBytesPerRow +
                phase * (kValuesPerPhase / 2) + lane * (Schedule::kValuesPerLane / 2);
            row_codes[local_row] =
                ops::detail::load_nvfp4_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(
                    codes + code_offset);
        }

        float group_sum[Schedule::kRowsPerWarp][2] = {};
#pragma unroll
        for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
            const int activation_index =
                phase * (kValuesPerPhase / 2) + lane * Schedule::kPairsPerLane + pair;
            const float2 activation = ops::bf16x2_bits_to_float2(activation_pairs[activation_index]);
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                const float2 code         = ops::detail::decode_nvfp4_e2m1x2(packed);
                group_sum[local_row][0] = fmaf(code.x, activation.x, group_sum[local_row][0]);
                group_sum[local_row][1] = fmaf(code.y, activation.y, group_sum[local_row][1]);
            }
        }
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const float grouped = group_sum[local_row][0] + group_sum[local_row][1];
            accumulators[local_row][phase & kChainMask] =
                fmaf(grouped, coefficients[local_row][0],
                     accumulators[local_row][phase & kChainMask]);
        }
    }
}

template <bool kHoisted>
__global__ void flash_next_moe_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const std::int32_t* __restrict__ ids,
    const std::uint8_t* __restrict__ expert_codes, const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors, std::uint64_t code_stride,
    std::uint64_t scale_stride, const __nv_bfloat16* __restrict__ shared_gate,
    const __nv_bfloat16* __restrict__ shared_up, __nv_bfloat16* __restrict__ activations) {
    __shared__ ops::detail::Nvfp4GemvSharedStorage<GateGeometry, GateSchedule> shared;
    const int token = static_cast<int>(blockIdx.y);
    const int path  = static_cast<int>(blockIdx.z);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * GateSchedule::kWarpsPerCta + warp;
    if (row >= kIntermediate) { return; }
    const auto* x = input + static_cast<std::int64_t>(token) * kHidden;
    float gate    = 0.0F;
    float up      = 0.0F;
    if (path < kTopK) {
        const int expert    = ids[token * kTopK + path];
        const auto* codes   = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* scales  = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inverse = 1.0F / expert_divisors[expert];
        const int parent_rows[GateSchedule::kRowsPerWarp] = {row, row + kIntermediate};
        float accumulators[GateSchedule::kRowsPerWarp][GateSchedule::kAccumulatorChains] = {};
        if constexpr (kHoisted) {
            compute_nvfp4_rows_hoisted<GateGeometry, GateSchedule>(
                x, codes, scales, shared, inverse, parent_rows, warp * GateSchedule::kRowsPerWarp,
                lane, accumulators);
        } else {
            ops::detail::compute_nvfp4_rows<GateGeometry, GateSchedule>(
                x, codes, scales, shared, inverse, parent_rows, warp * GateSchedule::kRowsPerWarp,
                lane, accumulators);
        }
#pragma unroll
        for (int chain = 0; chain < GateSchedule::kAccumulatorChains; ++chain) {
            gate += accumulators[0][chain];
            up += accumulators[1][chain];
        }
        gate = ops::warp_reduce_sum(gate);
        up   = ops::warp_reduce_sum(up);
    } else {
        dot_bf16_pair<kHidden>(x, shared_gate + static_cast<std::int64_t>(row) * kHidden,
                               shared_up + static_cast<std::int64_t>(row) * kHidden, gate, up);
    }
    if (lane == 0) {
        activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + row] =
            __float2bfloat16_rn(ops::silu(gate) * up);
    }
}

// ---------------------------------------------------------------------------------------------
// Decode down projection, per-lane bodies shared by the legacy and path-per-warp kernels.
//
// Everything below is the arithmetic contract that makes the two kernels bitwise equal: each
// helper is a pure function of (row, expert data, activations, lane) and never of which warp or
// CTA executes it. Any change here changes both kernels together; the memcmp gate in
// tests/models/qwen3_8_flash_next/test_moe.cpp compares the two kernels, so it can only catch
// a divergence between them, not a change to this contract itself.
// ---------------------------------------------------------------------------------------------

// Byte offset of this row's first scale word inside one expert's swizzled scale plane
// ([m_tile][k_tile][row_mod32][quartile][scale_lane], 512 B per (m_tile, k_tile) tile,
// storage_layouts.cpp / layouts.py _swizzle_nvfp4_bank_scales); the k_tile stride of 512 B is
// applied by the lane that loads that tile.
__device__ __forceinline__ std::int64_t down_scale_row_base(int row) {
    const int m_tile       = row / 128;
    const int row_inner    = row % 128;
    const int row_mod32    = row_inner & 31;
    const int row_quartile = row_inner >> 5;
    return static_cast<std::int64_t>(m_tile * 10) * 512 + row_mod32 * 16 + row_quartile * 4;
}

// One K16 group of one NVFP4 down row against its 16 BF16 activations. Even columns feed sum0
// and odd columns sum1 as two independent fmaf chains; dw * coef is a separate FMUL because nvcc
// does not contract an operand of an explicit fmaf, so the rounding sequence is fixed no matter
// what surrounds the call.
__device__ __forceinline__ void accumulate_down_group(const std::uint8_t* __restrict__ exp_codes,
                                                      const __nv_bfloat16* __restrict__ act_path,
                                                      int row, int group, float coef,
                                                      float& sum0, float& sum1) {
    const auto* packed =
        exp_codes + static_cast<std::int64_t>(row) * (kIntermediate / 2) + group * 8;
    const uint2 code_words = *reinterpret_cast<const uint2*>(packed);

    const auto* act_ptr = reinterpret_cast<const uint4*>(act_path + group * 16);
    const uint4 act_v0 = act_ptr[0];
    const uint4 act_v1 = act_ptr[1];

    const float2 dw[8] = {
        ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x)),
        ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 8)),
        ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 16)),
        ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.x >> 24)),
        ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y)),
        ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 8)),
        ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 16)),
        ops::detail::decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code_words.y >> 24))
    };

    const float2 hv[8] = {
        ops::bf16x2_bits_to_float2(act_v0.x),
        ops::bf16x2_bits_to_float2(act_v0.y),
        ops::bf16x2_bits_to_float2(act_v0.z),
        ops::bf16x2_bits_to_float2(act_v0.w),
        ops::bf16x2_bits_to_float2(act_v1.x),
        ops::bf16x2_bits_to_float2(act_v1.y),
        ops::bf16x2_bits_to_float2(act_v1.z),
        ops::bf16x2_bits_to_float2(act_v1.w)
    };

#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        sum0 = fmaf(dw[pair].x * coef, hv[pair].x, sum0);
        sum1 = fmaf(dw[pair].y * coef, hv[pair].y, sum1);
    }
}

// Warp-cooperative dot product of NVFP4 down row `row` of `expert` with the BF16 activation
// vector `act_path`. Lane l owns group l and, for l < 8, group l + 32, accumulated into the same
// two chains, pre-added, then reduced with the shfl_down tree; the result is meaningful on lane 0
// only. All 32 lanes must be converged on entry (the scale broadcast is a full-mask shuffle).
__device__ __forceinline__ float down_routed_path_value(
    int expert, const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales, const float* __restrict__ expert_divisors,
    std::uint64_t code_stride, std::uint64_t scale_stride,
    const __nv_bfloat16* __restrict__ act_path, int row, std::int64_t scale_row_base, int lane) {
    const auto* exp_codes =
        expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
    const auto* exp_scales =
        expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
    // IEEE division on purpose: a reciprocal intrinsic would change the coefficient bits.
    const float inverse = 1.0F / expert_divisors[expert];

    // Lanes 0..9 load the 10 scale words (tiles 0..9) for this row
    std::uint32_t my_tile = 0;
    if (lane < 10) {
        const std::int64_t off = scale_row_base + static_cast<std::int64_t>(lane) * 512;
        my_tile = *reinterpret_cast<const std::uint32_t*>(exp_scales + off);
    }

    // Broadcast tile words across the warp with all 32 threads active (zero divergence)
    const std::uint32_t tile_g0 = __shfl_sync(0xFFFFFFFFU, my_tile, lane >> 2);
    const std::uint32_t tile_g1 = __shfl_sync(0xFFFFFFFFU, my_tile, 8 + (lane >> 2));

    const std::uint8_t scale_g0 = static_cast<std::uint8_t>(tile_g0 >> ((lane & 3) * 8));
    const float coef_g0 = ops::detail::decode_nvfp4_e4m3(scale_g0) * inverse;

    const std::uint8_t scale_g1 = static_cast<std::uint8_t>(tile_g1 >> ((lane & 3) * 8));
    const float coef_g1 = ops::detail::decode_nvfp4_e4m3(scale_g1) * inverse;

    float sum0 = 0.0F;
    float sum1 = 0.0F;

    // Group 0..31: all 32 lanes active
    accumulate_down_group(exp_codes, act_path, row, lane, coef_g0, sum0, sum1);

    // Group 32..39: lanes 0..7 active
    if (lane < 8) {
        accumulate_down_group(exp_codes, act_path, row, lane + 32, coef_g1, sum0, sum1);
    }

    return ops::warp_reduce_sum(sum0 + sum1);
}

// Eight BF16 weight/activation pairs into the single shared-expert chain, in the fixed
// .x/.y order of the packed words.
__device__ __forceinline__ void accumulate_shared_uint4(uint4 w, uint4 a, float& shared_sum) {
    const float2 wv0 = ops::bf16x2_bits_to_float2(w.x);
    const float2 av0 = ops::bf16x2_bits_to_float2(a.x);
    const float2 wv1 = ops::bf16x2_bits_to_float2(w.y);
    const float2 av1 = ops::bf16x2_bits_to_float2(a.y);
    const float2 wv2 = ops::bf16x2_bits_to_float2(w.z);
    const float2 av2 = ops::bf16x2_bits_to_float2(a.z);
    const float2 wv3 = ops::bf16x2_bits_to_float2(w.w);
    const float2 av3 = ops::bf16x2_bits_to_float2(a.w);
    shared_sum = fmaf(wv0.x, av0.x, shared_sum);
    shared_sum = fmaf(wv0.y, av0.y, shared_sum);
    shared_sum = fmaf(wv1.x, av1.x, shared_sum);
    shared_sum = fmaf(wv1.y, av1.y, shared_sum);
    shared_sum = fmaf(wv2.x, av2.x, shared_sum);
    shared_sum = fmaf(wv2.y, av2.y, shared_sum);
    shared_sum = fmaf(wv3.x, av3.x, shared_sum);
    shared_sum = fmaf(wv3.y, av3.y, shared_sum);
}

// Warp-cooperative BF16 dot product of shared_down row `row` with the shared-path activations
// of this token: 80 uint4 (640 values) as two full-warp phases plus a 16-lane tail, one fmaf
// chain per lane, then the shfl_down tree. Meaningful on lane 0 only.
__device__ __forceinline__ float down_shared_path_value(
    const __nv_bfloat16* __restrict__ shared_down,
    const __nv_bfloat16* __restrict__ token_activations, int row, int lane) {
    // Vectorized 128-bit loads for shared expert dot product directly from L1/L2
    float shared_sum = 0.0F;
    const auto* shared_row = reinterpret_cast<const uint4*>(
        shared_down + static_cast<std::int64_t>(row) * kIntermediate);
    const auto* shared_act = reinterpret_cast<const uint4*>(
        token_activations + static_cast<std::int64_t>(kTopK) * kIntermediate);

#pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        const uint4 w = shared_row[phase * 32 + lane];
        const uint4 a = shared_act[phase * 32 + lane];
        accumulate_shared_uint4(w, a, shared_sum);
    }
    if (lane < 16) {
        const uint4 w = shared_row[64 + lane];
        const uint4 a = shared_act[64 + lane];
        accumulate_shared_uint4(w, a, shared_sum);
    }

    return ops::warp_reduce_sum(shared_sum);
}

// Legacy decode down kernel (reference): one warp per output row walks the ten routed paths
// serially, then the shared expert. Kept selectable (NINFER_FLASH_NEXT_MOE_DOWN_LEGACY=1) as the
// A/B and bitwise reference for the path-per-warp kernel below.
__global__ void flash_next_moe_down_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const __nv_bfloat16* __restrict__ activations,
    const std::uint8_t* __restrict__ expert_codes, const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors, std::uint64_t code_stride,
    std::uint64_t scale_stride, const __nv_bfloat16* __restrict__ shared_down,
    __nv_bfloat16* __restrict__ output) {
    const int token = static_cast<int>(blockIdx.y);
    const int tid   = static_cast<int>(threadIdx.x);
    const int warp  = tid >> 5;
    const int lane  = tid & 31;
    const int row   = static_cast<int>(blockIdx.x) * kDownWarps + warp;

    if (row >= kHidden) { return; }

    const std::int64_t scale_row_base = down_scale_row_base(row);

    const auto* token_activations =
        activations + static_cast<std::int64_t>(token) * kPaths * kIntermediate;

    float routed = 0.0F;

#pragma unroll
    for (int path = 0; path < kTopK; ++path) {
        const int expert = ids[token * kTopK + path];
        const auto* act_path = token_activations + static_cast<std::int64_t>(path) * kIntermediate;
        const float value =
            down_routed_path_value(expert, expert_codes, expert_scales, expert_divisors,
                                   code_stride, scale_stride, act_path, row, scale_row_base, lane);
        if (lane == 0) { routed = fmaf(alpha[token * kTopK + path], value, routed); }
    }

    const float shared_value = down_shared_path_value(shared_down, token_activations, row, lane);

    if (lane == 0) {
        output[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(fmaf(shared_scale[token], shared_value, routed));
    }
}

// Path-per-warp decode down kernel: one CTA per (row, token); warp p < 10 computes routed path
// p and warp 10 the shared expert, each with the per-lane body above, and thread 0 combines the
// eleven partials in the legacy order.
//
// Why: the legacy launch is 320 CTAs x 8 warps = 0.28 waves on 188 SMs (27% warps active, ncu
// baseline) and every warp walks its ten paths serially, so only one path's ~360 B of weights is
// in flight per warp -- roughly 1 MB GPU-wide against the ~1.2 MB Little's-law requirement at
// 1,568 GB/s x ~0.75 us, which is why it ran at 48-57% of sustained bandwidth. Here the CTA
// count is 2,560 x T (3.4 waves at T=1 with 4 resident CTAs per SM) and all 4.5 KB of a row's
// weights are requested at once, ~3.4 MB in flight GPU-wide, so the same 12.51 MB per launch
// (codes 8.19 MB + scales 1.02 MB + BF16 shared_down 3.28 MB + activations) becomes
// bandwidth-bound: floor 7.98 us, expected 9-10.5 us vs 14-16.5 us measured for the legacy one.
//
// Bitwise contract with the legacy kernel: partials cross warps unmodified (no alpha
// pre-multiplication, no +=), and the combine is the same fmaf chain in path order from 0.0F
// with the shared path last, exactly what lane 0 of the legacy warp does.
__global__ __launch_bounds__(kDownPathThreads, kDownPathMinBlocksPerSm)
void flash_next_moe_down_pathwarp_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const __nv_bfloat16* __restrict__ activations,
    const std::uint8_t* __restrict__ expert_codes, const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors, std::uint64_t code_stride,
    std::uint64_t scale_stride, const __nv_bfloat16* __restrict__ shared_down,
    __nv_bfloat16* __restrict__ output) {
    // Lane-0 value of each warp's warp_reduce_sum, indexed by path (10 = shared expert).
    __shared__ float s_path[kPaths];
    // alpha[0..9] and shared_scale of this token, staged by warp 10 so that thread 0's combine
    // does not wait on eleven global loads after the barrier and no warp carries them in
    // registers (which would lift every thread's allocation and cost a resident CTA per SM).
    __shared__ float s_coef[kPaths];

    const int token = static_cast<int>(blockIdx.y);
    const int row   = static_cast<int>(blockIdx.x);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;

    // grid.x == kHidden, so this never fires; it is uniform per CTA, so if it ever did no
    // thread of the CTA would reach the barrier below.
    if (row >= kHidden) { return; }

    const std::int64_t scale_row_base = down_scale_row_base(row);
    const auto* token_activations =
        activations + static_cast<std::int64_t>(token) * kPaths * kIntermediate;

    float value = 0.0F;
    if (warp < kTopK) {
        const int path       = warp;
        const int expert     = ids[token * kTopK + path];
        const auto* act_path = token_activations + static_cast<std::int64_t>(path) * kIntermediate;
        value = down_routed_path_value(expert, expert_codes, expert_scales, expert_divisors,
                                       code_stride, scale_stride, act_path, row, scale_row_base,
                                       lane);
    } else {
        value = down_shared_path_value(shared_down, token_activations, row, lane);
        if (lane < kTopK) { s_coef[lane] = alpha[token * kTopK + lane]; }
        if (lane == kTopK) { s_coef[kTopK] = shared_scale[token]; }
    }
    if (lane == 0) { s_path[warp] = value; }
    __syncthreads();

    if (threadIdx.x == 0) {
        float routed = 0.0F;
#pragma unroll
        for (int path = 0; path < kTopK; ++path) {
            routed = fmaf(s_coef[path], s_path[path], routed);
        }
        output[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(fmaf(s_coef[kTopK], s_path[kTopK], routed));
    }
}

// =========================================================================
// PREFILL KERNELS (T >= 128, Amortized Weight Loading & Grouped GEMMs)
// =========================================================================

// Step 1: Deterministic Expert Histogram, Prefix Scan & Token Grouping
// Step 1: Deterministic Expert Histogram, Prefix Scan, Token Grouping & Active Expert List
// CTA: 1 CTA = 512 threads (16 warps).
__global__ void flash_next_moe_prefill_build_groups_kernel(
    const std::int32_t* __restrict__ ids,
    std::int32_t* __restrict__ expert_counts,
    std::int32_t* __restrict__ expert_offsets,
    std::int32_t* __restrict__ active_experts,
    std::int32_t* __restrict__ active_count_ptr,
    std::int32_t* __restrict__ grouped_tokens,
    std::int32_t* __restrict__ grouped_paths,
    std::int32_t* __restrict__ grouped_experts,
    std::int32_t* __restrict__ token_to_pos,
    std::int32_t* __restrict__ task_counter,
    int tokens) {
    const int tid = static_cast<int>(threadIdx.x);
    if (tid >= 512) { return; }

    const int total_items = tokens * kTopK;

    // 1. Histogram: count items for expert 'tid'
    int count = 0;
    for (int i = 0; i < total_items; ++i) {
        if (ids[i] == tid) {
            count++;
        }
    }
    expert_counts[tid] = count;

    // 2. Deterministic prefix scan for expert offsets
    const int lane = tid & 31;
    const int warp = tid >> 5;

    int warp_sum = count;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int val = __shfl_up_sync(0xffffffff, warp_sum, offset);
        if (lane >= offset) warp_sum += val;
    }

    __shared__ int s_warp_totals[16];
    if (lane == 31) { s_warp_totals[warp] = warp_sum; }
    __syncthreads();

    if (warp == 0 && lane < 16) {
        int w_sum = s_warp_totals[lane];
        #pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const int val = __shfl_up_sync(0xffff, w_sum, offset);
            if (lane >= offset) w_sum += val;
        }
        s_warp_totals[lane] = w_sum;
    }
    __syncthreads();

    const int base_offset = (warp > 0) ? s_warp_totals[warp - 1] : 0;
    const int exc_offset  = base_offset + warp_sum - count;
    expert_offsets[tid]   = exc_offset;

    if (tid == 511) {
        expert_offsets[512] = exc_offset + count;
    }

    // 3. Compact active experts list using prefix scan of (count > 0)
    const int is_active = (count > 0) ? 1 : 0;
    int act_warp_sum = is_active;
    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int val = __shfl_up_sync(0xffffffff, act_warp_sum, offset);
        if (lane >= offset) act_warp_sum += val;
    }

    __shared__ int s_act_totals[16];
    if (lane == 31) { s_act_totals[warp] = act_warp_sum; }
    __syncthreads();

    if (warp == 0 && lane < 16) {
        int w_sum = s_act_totals[lane];
        #pragma unroll
        for (int offset = 1; offset < 16; offset <<= 1) {
            const int val = __shfl_up_sync(0xffff, w_sum, offset);
            if (lane >= offset) w_sum += val;
        }
        s_act_totals[lane] = w_sum;
    }
    __syncthreads();

    const int act_base_offset = (warp > 0) ? s_act_totals[warp - 1] : 0;
    const int act_exc_offset  = act_base_offset + act_warp_sum - is_active;
    if (is_active) {
        active_experts[act_exc_offset] = tid;
    }

    if (tid == 511) {
        active_count_ptr[0] = act_exc_offset + is_active;
        task_counter[0]     = 0;
        task_counter[1]     = 0;
        task_counter[2]     = 0;
        task_counter[3]     = 0;
    }

    // 4. Deterministic O(N) cooperative placement using shared memory integer atomic offsets
    __shared__ int s_heads[512];
    s_heads[tid] = exc_offset;
    __syncthreads();

    for (int i = tid; i < total_items; i += 512) {
        const int expert = ids[i];
        const int pos    = atomicAdd(&s_heads[expert], 1);
        grouped_tokens[pos]  = i / kTopK;
        grouped_paths[pos]   = i % kTopK;
        grouped_experts[pos] = expert;
        if (token_to_pos != nullptr) {
            token_to_pos[i]  = pos;
        }
    }
}



// Step 2a: Grouped Expert Gate & Up Projection (SIMT W4A16 for small token counts T < 512)
// CTA: 256 threads (8 warps). 8 pairs per CTA (8 warps x 1 pair).
// Grid: (kIntermediate / 8, kGridY) = (80, 16).
__global__ __launch_bounds__(256, 4) void flash_next_moe_prefill_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input,
    const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ expert_counts,
    const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count_ptr,
    const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths,
    const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors,
    std::uint64_t code_stride,
    std::uint64_t scale_stride,
    __nv_bfloat16* __restrict__ activations) {
    const int total_active = active_count_ptr[0];
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int local_pair   = warp;
    const int row_base     = static_cast<int>(blockIdx.x) * 8 + local_pair;
    if (row_base >= kIntermediate) { return; }

    constexpr int kBatch = 4;
    __shared__ uint4 s_input[kBatch][320];          // 20.48 KB
    __shared__ std::uint8_t s_gate_scales[16][160]; // 2.56 KB

    for (int act_idx = static_cast<int>(blockIdx.y); act_idx < total_active; act_idx += static_cast<int>(gridDim.y)) {
        const int expert      = active_experts[act_idx];
        const int token_count = expert_counts[expert];
        if (token_count <= 0) { continue; }

        const int start_idx = expert_offsets[expert];

        const auto* codes_base  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* scales_base = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inv_divisor = 1.0F / expert_divisors[expert];

        // 1. Cooperative coalesced scale staging for 16 rows x 40 tiles = 640 tasks
        for (int task = tid; task < 640; task += 256) {
            const int tile     = task >> 4;
            const int r        = task & 15;
            const int is_up    = r >= 8 ? 1 : 0;
            const int pair_idx = static_cast<int>(blockIdx.x) * 8 + (r & 7);
            const int row      = is_up ? (pair_idx + kIntermediate) : pair_idx;

            const int m_tile       = row / 128;
            const int row_inner    = row % 128;
            const int row_mod32    = row_inner & 31;
            const int row_quartile = row_inner >> 5;
            const std::int64_t off = static_cast<std::int64_t>(m_tile * 40 + tile) * 512 +
                                     row_mod32 * 16 + row_quartile * 4;

            const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(scales_base + off);
            *reinterpret_cast<std::uint32_t*>(&s_gate_scales[r][tile * 4]) = word;
        }
        __syncthreads();

        const auto* g_row_ptr = codes_base + static_cast<std::int64_t>(row_base) * (kHidden / 2);
        const auto* u_row_ptr = codes_base + static_cast<std::int64_t>(row_base + kIntermediate) * (kHidden / 2);

        for (int token_base = 0; token_base < token_count; token_base += kBatch) {
            const int batch_tokens = min(kBatch, token_count - token_base);

            // 2. Cooperative single stage of ALL 2560 input channels for up to 4 tokens
            for (int i = tid; i < batch_tokens * 320; i += 256) {
                const int b_idx = i / 320;
                const int u4    = i % 320;
                const int token = grouped_tokens[start_idx + token_base + b_idx];
                const auto* src = input + static_cast<std::int64_t>(token) * kHidden + u4 * 8;
                s_input[b_idx][u4] = *reinterpret_cast<const uint4*>(src);
            }
            __syncthreads();

            float gate_acc[kBatch] = {};
            float up_acc[kBatch]   = {};

            #pragma unroll
            for (int phase = 0; phase < 5; ++phase) {
                const int group = phase * 32 + lane;
                const auto* g_packed = g_row_ptr + group * 8;
                const auto* u_packed = u_row_ptr + group * 8;
                const uint2 g_words  = *reinterpret_cast<const uint2*>(g_packed);
                const uint2 u_words  = *reinterpret_cast<const uint2*>(u_packed);

                const float g_coeff = ops::detail::decode_nvfp4_e4m3(
                    s_gate_scales[local_pair][group]) * inv_divisor;
                const float u_coeff = ops::detail::decode_nvfp4_e4m3(
                    s_gate_scales[8 + local_pair][group]) * inv_divisor;

                float2 g_raw[8];
                float2 u_raw[8];
                #pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const std::uint32_t g_w = (pair < 4) ? g_words.x : g_words.y;
                    const std::uint32_t u_w = (pair < 4) ? u_words.x : u_words.y;
                    const std::uint8_t gp   = static_cast<std::uint8_t>(g_w >> (8 * (pair & 3)));
                    const std::uint8_t up   = static_cast<std::uint8_t>(u_w >> (8 * (pair & 3)));
                    g_raw[pair] = ops::detail::decode_nvfp4_e2m1x2(gp);
                    u_raw[pair] = ops::detail::decode_nvfp4_e2m1x2(up);
                }

                #pragma unroll
                for (int b = 0; b < kBatch; ++b) {
                    if (b < batch_tokens) {
                        const auto x_v0 = s_input[b][group * 2];
                        const auto x_v1 = s_input[b][group * 2 + 1];
                        const std::uint32_t x_raw[8] = {
                            x_v0.x, x_v0.y, x_v0.z, x_v0.w,
                            x_v1.x, x_v1.y, x_v1.z, x_v1.w
                        };

                        float g_sum_raw = 0.0F;
                        float u_sum_raw = 0.0F;
                        #pragma unroll
                        for (int pair = 0; pair < 8; ++pair) {
                            const float2 xv = ops::bf16x2_bits_to_float2(x_raw[pair]);
                            g_sum_raw = fmaf(g_raw[pair].x, xv.x, g_sum_raw);
                            g_sum_raw = fmaf(g_raw[pair].y, xv.y, g_sum_raw);
                            u_sum_raw = fmaf(u_raw[pair].x, xv.x, u_sum_raw);
                            u_sum_raw = fmaf(u_raw[pair].y, xv.y, u_sum_raw);
                        }
                        gate_acc[b] = fmaf(g_sum_raw, g_coeff, gate_acc[b]);
                        up_acc[b]   = fmaf(u_sum_raw, u_coeff, up_acc[b]);
                    }
                }
            }

            #pragma unroll
            for (int b = 0; b < kBatch; ++b) {
                if (b < batch_tokens) {
                    const int token = grouped_tokens[start_idx + token_base + b];
                    const int path  = grouped_paths[start_idx + token_base + b];

                    const float gate_sum = ops::warp_reduce_sum(gate_acc[b]);
                    const float up_sum   = ops::warp_reduce_sum(up_acc[b]);
                    if (lane == 0) {
                        const float act = ops::silu(gate_sum) * up_sum;
                        activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + row_base] =
                            __float2bfloat16_rn(act);
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

// Step 2b: Grouped Expert Gate & Up Projection (Native NVFP4 Tensor Core MMA)
// Tile: 32 rows (16 Gate + 16 Up intermediate pairs) x 16 tokens per CTA (2 warps along M x 2 warps along N)
// CTA: 128 threads (4 warps). Warps (0,1) compute pairs 0..7; Warps (2,3) compute pairs 8..15.
// Shared memory: 40960 + 5120 (weights) + 20480 + 2560 (activations) = 69120 bytes (67.5 KB)
// Grid: (kIntermediate / 16, 512)
template <bool kAsyncStaging = false>
__global__ __launch_bounds__(128, 4) void flash_next_moe_prefill_gate_up_mma_kernel(
    const std::uint8_t* __restrict__ act_codes,     // [1280, tokens]
    const std::uint8_t* __restrict__ act_scales,    // [160, tokens]
    const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ expert_counts,
    const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count_ptr,
    const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths,
    const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors,
    std::uint64_t code_stride,
    std::uint64_t scale_stride,
    __nv_bfloat16* __restrict__ activations) {

    const int total_active = active_count_ptr[0];
    const int active_idx   = static_cast<int>(blockIdx.y);
    if (active_idx >= total_active) { return; }

    const int expert = active_experts[active_idx];
    const int count  = expert_counts[expert];
    if (count <= 0) { return; }

    const int pair_base = static_cast<int>(blockIdx.x) * 16;
    if (pair_base >= kIntermediate) { return; }

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int warp_m = warp >> 1; // 0 or 1 (pairs 0..7 or pairs 8..15)
    const int warp_n = warp & 1;  // 0 or 1 (tokens 0..7 or tokens 8..15)
    const int lane   = tid & 31;

    extern __shared__ alignas(16) std::uint8_t s_dyn_mem[];
    auto* s_w_codes    = s_dyn_mem;                          // 32 * 1280 = 40960 bytes
    auto* s_w_scales   = s_dyn_mem + 40960;                  // 32 * 164 = 5248 bytes
    auto* s_a_codes_0  = s_dyn_mem + 40960 + 5248;           // 16 * 1280 = 20480 bytes
    auto* s_a_scales_0 = s_dyn_mem + 40960 + 5248 + 20480;   // 16 * 164 = 2624 bytes
    auto* s_a_codes_1  = s_dyn_mem + 40960 + 5248 + 20480 + 2624; // 16 * 1280 = 20480 bytes
    auto* s_a_scales_1 = s_dyn_mem + 40960 + 5248 + 20480 + 2624 + 20480; // 16 * 164 = 2624 bytes

    const auto* exp_codes  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
    const auto* exp_scales = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
    const int offset       = expert_offsets[expert];
    const float alpha      = 1.0F / expert_divisors[expert];

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    if constexpr (kAsyncStaging) {
        // Lever A: Cooperatively stage 32 rows of weights via direct hardware cp.async
        #pragma unroll
        for (int i = tid * 16; i < 32 * 1280; i += 128 * 16) {
            const int row = i / 1280;
            const int col = i - row * 1280;
            const int m_grp = row >> 4;   // 0 or 1
            const int r_in  = row & 15;   // 0..15
            const int p_off = m_grp * 8 + (r_in & 7);
            const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
            ops::cp_async<16, ops::Cache::cg>(s_w_codes + i,
                                              exp_codes + static_cast<std::int64_t>(global_row) * 1280 + col);
        }

        #pragma unroll
        for (int i = tid; i < 32 * 40; i += 128) {
            const int row  = i / 40;
            const int tile = i - row * 40;
            const int m_grp = row >> 4;
            const int r_in  = row & 15;
            const int p_off = m_grp * 8 + (r_in & 7);
            const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
            const int m_tile     = global_row / 128;
            const int row_inner  = global_row % 128;
            const int row_mod32  = row_inner & 31;
            const int row_quart  = row_inner >> 5;
            const std::int64_t off = static_cast<std::int64_t>(m_tile * 40 + tile) * 512 +
                                     row_mod32 * 16 + row_quart * 4;
            ops::cp_async<4, ops::Cache::ca>(s_w_scales + row * 164 + tile * 4,
                                             exp_scales + off);
        }

        // Lever B: Co-issue chunk-0 activations into Buffer 0 concurrently with weights
        #pragma unroll
        for (int i = tid * 16; i < 16 * 1280; i += 128 * 16) {
            const int tok_idx = i / 1280;
            const int col     = i - tok_idx * 1280;
            if (tok_idx < count) {
                const int pos = offset + tok_idx;
                const int tok = grouped_tokens[pos];
                ops::cp_async<16, ops::Cache::ca>(s_a_codes_0 + i,
                                                  act_codes + static_cast<std::int64_t>(tok) * 1280 + col);
            } else {
                *reinterpret_cast<uint4*>(s_a_codes_0 + i) = make_uint4(0, 0, 0, 0);
            }
        }

        #pragma unroll
        for (int i = tid; i < 16 * 40; i += 128) {
            const int tok_idx = i / 40;
            const int tile    = i - tok_idx * 40;
            if (tok_idx < count) {
                const int pos = offset + tok_idx;
                const int tok = grouped_tokens[pos];
                ops::cp_async<4, ops::Cache::ca>(s_a_scales_0 + tok_idx * 164 + tile * 4,
                                                 act_scales + static_cast<std::int64_t>(tok) * 160 + tile * 4);
            } else {
                *reinterpret_cast<std::uint32_t*>(s_a_scales_0 + tok_idx * 164 + tile * 4) = 0;
            }
        }

        ops::cp_commit();
        ops::cp_wait<0>();
        __syncthreads();

        // Lever C: Double-buffered pipelined execution across token chunks
        for (int t_chunk = 0; t_chunk < count; t_chunk += 16) {
            const int buf_idx   = (t_chunk >> 4) & 1;
            auto* curr_a_codes  = (buf_idx == 0) ? s_a_codes_0 : s_a_codes_1;
            auto* curr_a_scales = (buf_idx == 0) ? s_a_scales_0 : s_a_scales_1;
            auto* next_a_codes  = (buf_idx == 0) ? s_a_codes_1 : s_a_codes_0;
            auto* next_a_scales = (buf_idx == 0) ? s_a_scales_1 : s_a_scales_0;

            const int next_chunk = t_chunk + 16;
            if (next_chunk < count) {
                #pragma unroll
                for (int i = tid * 16; i < 16 * 1280; i += 128 * 16) {
                    const int tok_idx = i / 1280;
                    const int col     = i - tok_idx * 1280;
                    if (next_chunk + tok_idx < count) {
                        const int pos = offset + next_chunk + tok_idx;
                        const int tok = grouped_tokens[pos];
                        ops::cp_async<16, ops::Cache::ca>(next_a_codes + i,
                                                          act_codes + static_cast<std::int64_t>(tok) * 1280 + col);
                    } else {
                        *reinterpret_cast<uint4*>(next_a_codes + i) = make_uint4(0, 0, 0, 0);
                    }
                }

                #pragma unroll
                for (int i = tid; i < 16 * 40; i += 128) {
                    const int tok_idx = i / 40;
                    const int tile    = i - tok_idx * 40;
                    if (next_chunk + tok_idx < count) {
                        const int pos = offset + next_chunk + tok_idx;
                        const int tok = grouped_tokens[pos];
                        ops::cp_async<4, ops::Cache::ca>(next_a_scales + tok_idx * 164 + tile * 4,
                                                         act_scales + static_cast<std::int64_t>(tok) * 160 + tile * 4);
                    } else {
                        *reinterpret_cast<std::uint32_t*>(next_a_scales + tok_idx * 164 + tile * 4) = 0;
                    }
                }
                ops::cp_commit();
            }

            const int t_base    = t_chunk + warp_n * 8;
            const int cur_batch = max(0, min(8, count - t_base));

            if (cur_batch > 0) {
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const auto* warp_w_codes  = s_w_codes + warp_m * (16 * 1280);
                const auto* warp_w_scales = s_w_scales + warp_m * (16 * 164);
                const auto* warp_a_codes  = curr_a_codes + warp_n * (8 * 1280);
                const auto* warp_a_scales = curr_a_scales + warp_n * (8 * 164);

                #pragma unroll 4
                for (int k64 = 0; k64 < 40; ++k64) {
                    unsigned a[4];
                    unsigned b[2];

                    const int a_row = a_row_offset;
                    const auto* a_addr = warp_w_codes + a_row * 1280 + k64 * 32 + a_column_byte;
                    ops::ldmatrix_x4(a[0], a[1], a[2], a[3], ops::smem_addr(a_addr));

                    const int b_row = b_row_offset;
                    const auto* b_addr = warp_a_codes + b_row * 1280 + k64 * 32 + b_column_byte;
                    ops::ldmatrix_x2(b[0], b[1], ops::smem_addr(b_addr));

                    const uint32_t sfa = *reinterpret_cast<const uint32_t*>(warp_w_scales + sfa_row * 164 + k64 * 4);
                    const uint32_t sfb = *reinterpret_cast<const uint32_t*>(warp_a_scales + sfb_row * 164 + k64 * 4);

                    ops::mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                                       a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
                }

                const int tok0 = 2 * (lane & 3);
                const int tok1 = tok0 + 1;
                const int local_pair = lane >> 2;
                const int pair = pair_base + warp_m * 8 + local_pair;

                const float gate0 = accumulators[0] * alpha;
                const float gate1 = accumulators[1] * alpha;
                const float up0   = accumulators[2] * alpha;
                const float up1   = accumulators[3] * alpha;

                if (tok0 < cur_batch) {
                    const int pos = offset + t_base + tok0;
                    const int tok = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                        __float2bfloat16_rn(ops::silu(gate0) * up0);
                }
                if (tok1 < cur_batch) {
                    const int pos = offset + t_base + tok1;
                    const int tok = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                        __float2bfloat16_rn(ops::silu(gate1) * up1);
                }
            }

            if (next_chunk < count) {
                ops::cp_wait<0>();
                __syncthreads();
            }
        }
    } else {
        auto* s_a_codes  = s_a_codes_0;
        auto* s_a_scales = s_a_scales_0;

        // Cooperatively load 32 rows of weights (40960 bytes) across all 128 threads (20 iterations)
        #pragma unroll
        for (int i = tid * 16; i < 32 * 1280; i += 128 * 16) {
            const int row = i / 1280;
            const int col = i - row * 1280;
            const int m_grp = row >> 4;   // 0 or 1
            const int r_in  = row & 15;   // 0..15
            const int p_off = m_grp * 8 + (r_in & 7);
            const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
            *reinterpret_cast<uint4*>(s_w_codes + i) =
                *reinterpret_cast<const uint4*>(exp_codes + static_cast<std::int64_t>(global_row) * 1280 + col);
        }

        // Cooperatively unpack 1280 scale words (5120 bytes) across all 128 threads (10 iterations)
        // Stored with 164-byte row stride (41 words, coprime with 32 banks) to eliminate bank conflicts.
        #pragma unroll
        for (int i = tid; i < 32 * 40; i += 128) {
            const int row  = i / 40;
            const int tile = i - row * 40;
            const int m_grp = row >> 4;
            const int r_in  = row & 15;
            const int p_off = m_grp * 8 + (r_in & 7);
            const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
            const int m_tile     = global_row / 128;
            const int row_inner  = global_row % 128;
            const int row_mod32  = row_inner & 31;
            const int row_quart  = row_inner >> 5;
            const std::int64_t off = static_cast<std::int64_t>(m_tile * 40 + tile) * 512 +
                                     row_mod32 * 16 + row_quart * 4;
            *reinterpret_cast<std::uint32_t*>(s_w_scales + row * 164 + tile * 4) =
                *reinterpret_cast<const std::uint32_t*>(exp_scales + off);
        }
        __syncthreads();

        for (int t_chunk = 0; t_chunk < count; t_chunk += 16) {
            // Cooperatively load 16 tokens of activations across all 128 threads
            #pragma unroll
            for (int i = tid * 16; i < 16 * 1280; i += 128 * 16) {
                const int tok_idx = i / 1280;
                const int col     = i - tok_idx * 1280;
                if (t_chunk + tok_idx < count) {
                    const int pos = offset + t_chunk + tok_idx;
                    const int tok = grouped_tokens[pos];
                    *reinterpret_cast<uint4*>(s_a_codes + i) =
                        *reinterpret_cast<const uint4*>(act_codes + static_cast<std::int64_t>(tok) * 1280 + col);
                } else {
                    *reinterpret_cast<uint4*>(s_a_codes + i) = make_uint4(0, 0, 0, 0);
                }
            }

            #pragma unroll
            for (int i = tid; i < 16 * 40; i += 128) {
                const int tok_idx = i / 40;
                const int tile    = i - tok_idx * 40;
                if (t_chunk + tok_idx < count) {
                    const int pos = offset + t_chunk + tok_idx;
                    const int tok = grouped_tokens[pos];
                    *reinterpret_cast<std::uint32_t*>(s_a_scales + tok_idx * 164 + tile * 4) =
                        *reinterpret_cast<const std::uint32_t*>(act_scales + static_cast<std::int64_t>(tok) * 160 + tile * 4);
                } else {
                    *reinterpret_cast<std::uint32_t*>(s_a_scales + tok_idx * 164 + tile * 4) = 0;
                }
            }
            __syncthreads();

            const int t_base    = t_chunk + warp_n * 8;
            const int cur_batch = max(0, min(8, count - t_base));

            if (cur_batch > 0) {
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const auto* warp_w_codes  = s_w_codes + warp_m * (16 * 1280);
                const auto* warp_w_scales = s_w_scales + warp_m * (16 * 164);
                const auto* warp_a_codes  = s_a_codes + warp_n * (8 * 1280);
                const auto* warp_a_scales = s_a_scales + warp_n * (8 * 164);

                #pragma unroll 4
                for (int k64 = 0; k64 < 40; ++k64) {
                    unsigned a[4];
                    unsigned b[2];

                    const int a_row = a_row_offset;
                    const auto* a_addr = warp_w_codes + a_row * 1280 + k64 * 32 + a_column_byte;
                    ops::ldmatrix_x4(a[0], a[1], a[2], a[3], ops::smem_addr(a_addr));

                    const int b_row = b_row_offset;
                    const auto* b_addr = warp_a_codes + b_row * 1280 + k64 * 32 + b_column_byte;
                    ops::ldmatrix_x2(b[0], b[1], ops::smem_addr(b_addr));

                    const uint32_t sfa = *reinterpret_cast<const uint32_t*>(warp_w_scales + sfa_row * 164 + k64 * 4);
                    const uint32_t sfb = *reinterpret_cast<const uint32_t*>(warp_a_scales + sfb_row * 164 + k64 * 4);

                    ops::mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                                       a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
                }

                const int tok0 = 2 * (lane & 3);
                const int tok1 = tok0 + 1;
                const int local_pair = lane >> 2;
                const int pair = pair_base + warp_m * 8 + local_pair;

                const float gate0 = accumulators[0] * alpha;
                const float gate1 = accumulators[1] * alpha;
                const float up0   = accumulators[2] * alpha;
                const float up1   = accumulators[3] * alpha;

                if (tok0 < cur_batch) {
                    const int pos = offset + t_base + tok0;
                    const int tok = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                        __float2bfloat16_rn(ops::silu(gate0) * up0);
                }
                if (tok1 < cur_batch) {
                    const int pos = offset + t_base + tok1;
                    const int tok = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                        __float2bfloat16_rn(ops::silu(gate1) * up1);
                }
            }
            __syncthreads();
        }
    }
}

// Step 2c: Grouped Expert Gate & Up Projection (Optimized N=32 tokens, M=16 pairs, 256 threads / 8 warps)
// Tile: 32 rows (16 Gate + 16 Up intermediate pairs) x 32 tokens per CTA (4 warps along N x 2 warps along M)
// Warps 0..1: pairs 0..7, tokens 0..15; Warps 2..3: pairs 8..15, tokens 0..15
// Warps 4..5: pairs 0..7, tokens 16..31; Warps 6..7: pairs 8..15, tokens 16..31
// Threads: 256 threads (8 warps) per block -> doubles warp occupancy per SM!
// Loop step: 32 tokens per iteration -> halves token loop iterations!
// Shared memory: 40960 + 5248 (weights) + 40960 + 5248 (activations) = 92416 bytes (fits hardware limit 101,376).
__global__ __launch_bounds__(256, 1) void flash_next_moe_prefill_gate_up_mma_v2_kernel(
    const std::uint8_t* __restrict__ act_codes,     // [1280, tokens]
    const std::uint8_t* __restrict__ act_scales,    // [160, tokens]
    const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ expert_counts,
    const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count_ptr,
    const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths,
    const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors,
    std::uint64_t code_stride,
    std::uint64_t scale_stride,
    __nv_bfloat16* __restrict__ activations) {

    const int total_active = active_count_ptr[0];
    const int active_idx   = static_cast<int>(blockIdx.y);
    if (active_idx >= total_active) { return; }

    const int expert = active_experts[active_idx];
    const int count  = expert_counts[expert];
    if (count <= 0) { return; }

    const int pair_base = static_cast<int>(blockIdx.x) * 16;
    if (pair_base >= kIntermediate) { return; }

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int warp_n_grp = warp >> 2;                     // 0 for warps 0..3 (tok 0..15); 1 for warps 4..7 (tok 16..31)
    const int warp_sub   = warp & 3;
    const int warp_m     = warp_sub >> 1;                 // 0 for pairs 0..7; 1 for pairs 8..15
    const int warp_n_sub = warp_sub & 1;                  // 0 for tok 0..7 / 16..23; 1 for tok 8..15 / 24..31
    const int warp_n     = (warp_n_grp << 1) | warp_n_sub; // 0, 1, 2, 3 (tok 0..7, 8..15, 16..23, 24..31)
    const int lane       = tid & 31;

    extern __shared__ alignas(16) std::uint8_t s_dyn_mem[];
    auto* s_w_codes  = s_dyn_mem;                                           // 32 * 1280 = 40960 bytes
    auto* s_w_scales = s_dyn_mem + 40960;                                   // 32 * 164  =  5248 bytes
    auto* s_a_codes  = s_dyn_mem + 40960 + 5248;                           // 32 * 1280 = 40960 bytes
    auto* s_a_scales = s_dyn_mem + 40960 + 5248 + 40960;                   // 32 * 164  =  5248 bytes

    const auto* exp_codes  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
    const auto* exp_scales = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
    const int offset       = expert_offsets[expert];
    const float alpha      = 1.0F / expert_divisors[expert];

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    // 1. Cooperative load 32 rows of weights (40,960 bytes) across 256 threads (10 steps)
    #pragma unroll
    for (int i = tid * 16; i < 32 * 1280; i += 256 * 16) {
        const int row = i / 1280;
        const int col = i - row * 1280;
        const int m_grp = row >> 4;   // 0 or 1
        const int r_in  = row & 15;   // 0..15
        const int p_off = m_grp * 8 + (r_in & 7);
        const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
        ops::cp_async<16, ops::Cache::cg>(s_w_codes + i,
                                          exp_codes + static_cast<std::int64_t>(global_row) * 1280 + col);
    }

    // 2. Cooperative load 32 rows of weight scales (1,280 words) across 256 threads (5 steps)
    #pragma unroll
    for (int i = tid; i < 32 * 40; i += 256) {
        const int row  = i / 40;
        const int tile = i - row * 40;
        const int m_grp = row >> 4;
        const int r_in  = row & 15;
        const int p_off = m_grp * 8 + (r_in & 7);
        const int global_row = (r_in < 8) ? (pair_base + p_off) : (pair_base + p_off + kIntermediate);
        const int m_tile     = global_row >> 7;   // global_row / 128
        const int row_inner  = global_row & 127;  // global_row % 128
        const int row_mod32  = row_inner & 31;
        const int row_quart  = row_inner >> 5;
        const std::int64_t off = (static_cast<std::int64_t>(m_tile * 40 + tile) << 9) |
                                 (row_mod32 << 4) | (row_quart << 2);
        ops::cp_async<4, ops::Cache::ca>(s_w_scales + row * 164 + tile * 4,
                                         exp_scales + off);
    }

    // 3. Co-issue chunk-0 activations (32 tokens) concurrently with weights
    #pragma unroll
    for (int i = tid * 16; i < 32 * 1280; i += 256 * 16) {
        const int tok_idx = i / 1280;
        const int col     = i - tok_idx * 1280;
        if (tok_idx < count) {
            const int pos = offset + tok_idx;
            const int tok = grouped_tokens[pos];
            ops::cp_async<16, ops::Cache::ca>(s_a_codes + i,
                                              act_codes + static_cast<std::int64_t>(tok) * 1280 + col);
        } else {
            *reinterpret_cast<uint4*>(s_a_codes + i) = make_uint4(0, 0, 0, 0);
        }
    }

    #pragma unroll
    for (int i = tid; i < 32 * 40; i += 256) {
        const int tok_idx = i / 40;
        const int tile    = i - tok_idx * 40;
        if (tok_idx < count) {
            const int pos = offset + tok_idx;
            const int tok = grouped_tokens[pos];
            ops::cp_async<4, ops::Cache::ca>(s_a_scales + tok_idx * 164 + tile * 4,
                                             act_scales + static_cast<std::int64_t>(tok) * 160 + tile * 4);
        } else {
            *reinterpret_cast<std::uint32_t*>(s_a_scales + tok_idx * 164 + tile * 4) = 0;
        }
    }

    ops::cp_commit();
    ops::cp_wait<0>();
    __syncthreads();

    // 4. Token chunk loop (32 tokens per iteration)
    for (int t_chunk = 0; t_chunk < count; t_chunk += 32) {
        if (t_chunk > 0) {
            #pragma unroll
            for (int i = tid * 16; i < 32 * 1280; i += 256 * 16) {
                const int tok_idx = i / 1280;
                const int col     = i - tok_idx * 1280;
                if (t_chunk + tok_idx < count) {
                    const int pos = offset + t_chunk + tok_idx;
                    const int tok = grouped_tokens[pos];
                    ops::cp_async<16, ops::Cache::ca>(s_a_codes + i,
                                                      act_codes + static_cast<std::int64_t>(tok) * 1280 + col);
                } else {
                    *reinterpret_cast<uint4*>(s_a_codes + i) = make_uint4(0, 0, 0, 0);
                }
            }

            #pragma unroll
            for (int i = tid; i < 32 * 40; i += 256) {
                const int tok_idx = i / 40;
                const int tile    = i - tok_idx * 40;
                if (t_chunk + tok_idx < count) {
                    const int pos = offset + t_chunk + tok_idx;
                    const int tok = grouped_tokens[pos];
                    ops::cp_async<4, ops::Cache::ca>(s_a_scales + tok_idx * 164 + tile * 4,
                                                     act_scales + static_cast<std::int64_t>(tok) * 160 + tile * 4);
                } else {
                    *reinterpret_cast<std::uint32_t*>(s_a_scales + tok_idx * 164 + tile * 4) = 0;
                }
            }

            ops::cp_commit();
            ops::cp_wait<0>();
            __syncthreads();
        }

        const int t_base    = t_chunk + warp_n * 8;
        const int cur_batch = max(0, min(8, count - t_base));

        if (cur_batch > 0) {
            float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const auto* warp_w_codes  = s_w_codes + warp_m * (16 * 1280);
            const auto* warp_w_scales = s_w_scales + warp_m * (16 * 164);
            const auto* warp_a_codes  = s_a_codes + warp_n * (8 * 1280);
            const auto* warp_a_scales = s_a_scales + warp_n * (8 * 164);

            #pragma unroll 4
            for (int k64 = 0; k64 < 40; ++k64) {
                unsigned a[4];
                unsigned b[2];

                const int a_row = a_row_offset;
                const auto* a_addr = warp_w_codes + a_row * 1280 + k64 * 32 + a_column_byte;
                ops::ldmatrix_x4(a[0], a[1], a[2], a[3], ops::smem_addr(a_addr));

                const int b_row = b_row_offset;
                const auto* b_addr = warp_a_codes + b_row * 1280 + k64 * 32 + b_column_byte;
                ops::ldmatrix_x2(b[0], b[1], ops::smem_addr(b_addr));

                const uint32_t sfa = *reinterpret_cast<const uint32_t*>(warp_w_scales + sfa_row * 164 + k64 * 4);
                const uint32_t sfb = *reinterpret_cast<const uint32_t*>(warp_a_scales + sfb_row * 164 + k64 * 4);

                ops::mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                                   a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
            }

            const int tok0 = 2 * (lane & 3);
            const int tok1 = tok0 + 1;
            const int local_pair = lane >> 2;
            const int pair = pair_base + warp_m * 8 + local_pair;

            const float gate0 = accumulators[0] * alpha;
            const float gate1 = accumulators[1] * alpha;
            const float up0   = accumulators[2] * alpha;
            const float up1   = accumulators[3] * alpha;

            if (tok0 < cur_batch) {
                const int pos = offset + t_base + tok0;
                const int tok = grouped_tokens[pos];
                const int path = grouped_paths[pos];
                activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                    __float2bfloat16_rn(ops::silu(gate0) * up0);
            }
            if (tok1 < cur_batch) {
                const int pos = offset + t_base + tok1;
                const int tok = grouped_tokens[pos];
                const int path = grouped_paths[pos];
                activations[(static_cast<std::int64_t>(tok) * kPaths + path) * kIntermediate + pair] =
                    __float2bfloat16_rn(ops::silu(gate1) * up1);
            }
        }

        if (t_chunk + 32 < count) {
            __syncthreads();
        }
    }
}


// Step 3: Shared Expert Gate & Up Projection (Vectorized 128-bit loads)
// CTA: 256 threads (8 warps). Each warp processes 1 pair -> 8 pairs per CTA.
// Grid.x = 640 / 8 = 80, Grid.y = (tokens + 7) / 8.
__global__ void flash_next_moe_prefill_shared_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ shared_gate,
    const __nv_bfloat16* __restrict__ shared_up,
    __nv_bfloat16* __restrict__ activations,
    int tokens) {
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int pair         = static_cast<int>(blockIdx.x) * 8 + warp;
    const int token_base   = static_cast<int>(blockIdx.y) * 8;
    const int batch_tokens = min(8, tokens - token_base);

    if (pair >= kIntermediate || batch_tokens <= 0) { return; }

    const auto* g_row = shared_gate + static_cast<std::int64_t>(pair) * kHidden;
    const auto* u_row = shared_up + static_cast<std::int64_t>(pair) * kHidden;

    float gate_sum[8] = {};
    float up_sum[8]   = {};

    for (int col_base = 0; col_base < kHidden; col_base += 256) {
        const int col = col_base + lane * 8;
        if (col < kHidden) {
            const auto g_v = *reinterpret_cast<const uint4*>(&g_row[col]);
            const auto u_v = *reinterpret_cast<const uint4*>(&u_row[col]);
            const std::uint32_t g_raw[4] = {g_v.x, g_v.y, g_v.z, g_v.w};
            const std::uint32_t u_raw[4] = {u_v.x, u_v.y, u_v.z, u_v.w};

            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (b < batch_tokens) {
                    const int token  = token_base + b;
                    const auto* in_x = input + static_cast<std::int64_t>(token) * kHidden;
                    const auto x_v   = *reinterpret_cast<const uint4*>(&in_x[col]);
                    const std::uint32_t x_raw[4] = {x_v.x, x_v.y, x_v.z, x_v.w};

                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const float2 g_pair = ops::bf16x2_bits_to_float2(g_raw[p]);
                        const float2 u_pair = ops::bf16x2_bits_to_float2(u_raw[p]);
                        const float2 x_pair = ops::bf16x2_bits_to_float2(x_raw[p]);
                        gate_sum[b] = fmaf(g_pair.x, x_pair.x, gate_sum[b]);
                        gate_sum[b] = fmaf(g_pair.y, x_pair.y, gate_sum[b]);
                        up_sum[b]   = fmaf(u_pair.x, x_pair.x, up_sum[b]);
                        up_sum[b]   = fmaf(u_pair.y, x_pair.y, up_sum[b]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        if (b < batch_tokens) {
            const int token    = token_base + b;
            const float g_tot  = ops::warp_reduce_sum(gate_sum[b]);
            const float u_tot  = ops::warp_reduce_sum(up_sum[b]);
            if (lane == 0) {
                activations[(static_cast<std::int64_t>(token) * kPaths + kTopK) * kIntermediate + pair] =
                    __float2bfloat16_rn(ops::silu(g_tot) * u_tot);
            }
        }
    }
}

// Step 4a: Grouped Expert Down Projection (SIMT W4A16 for small token counts T < 512)
// CTA: 256 threads (8 warps). 16 rows per CTA (8 warps x 2 rows).
// Grid: (kHidden / 16, kGridY) = (160, 8).
__global__ __launch_bounds__(256, 4) void flash_next_moe_prefill_down_kernel(
    const __nv_bfloat16* __restrict__ activations,
    const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ expert_counts,
    const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count_ptr,
    const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths,
    const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors,
    std::uint64_t code_stride,
    std::uint64_t scale_stride,
    float* __restrict__ down_intermediate) {
    const int total_active = active_count_ptr[0];
    const int tid          = static_cast<int>(threadIdx.x);
    const int warp         = tid >> 5;
    const int lane         = tid & 31;
    const int local_row    = warp * 2;
    const int row_base     = static_cast<int>(blockIdx.x) * 16 + local_row;
    if (row_base >= kHidden) { return; }

    constexpr int kBatch = 4;
    __shared__ uint4 s_act[kBatch][80];            // 5.12 KB
    __shared__ std::uint8_t s_down_scales[16][40]; // 640 bytes

    for (int act_idx = static_cast<int>(blockIdx.y); act_idx < total_active; act_idx += static_cast<int>(gridDim.y)) {
        const int expert      = active_experts[act_idx];
        const int token_count = expert_counts[expert];
        if (token_count <= 0) { continue; }

        const int start_idx = expert_offsets[expert];

        const auto* codes_base  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
        const auto* scales_base = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
        const float inv_divisor = 1.0F / expert_divisors[expert];

        // 1. Cooperative coalesced scale staging for 16 rows x 10 tiles = 160 tasks
        for (int task = tid; task < 160; task += 256) {
            const int tile = task >> 4;
            const int r    = task & 15;
            const int row  = static_cast<int>(blockIdx.x) * 16 + r;

            const int m_tile       = row / 128;
            const int row_inner    = row % 128;
            const int row_mod32    = row_inner & 31;
            const int row_quartile = row_inner >> 5;
            const std::int64_t off = static_cast<std::int64_t>(m_tile * 10 + tile) * 512 +
                                     row_mod32 * 16 + row_quartile * 4;

            const std::uint32_t word = *reinterpret_cast<const std::uint32_t*>(scales_base + off);
            *reinterpret_cast<std::uint32_t*>(&s_down_scales[r][tile * 4]) = word;
        }
        __syncthreads();

        for (int token_base = 0; token_base < token_count; token_base += kBatch) {
            const int batch_tokens = min(kBatch, token_count - token_base);

            // 2. Cooperative stage of ALL 640 intermediate activations for up to 4 tokens
            for (int i = tid; i < batch_tokens * 80; i += 256) {
                const int b_idx = i / 80;
                const int u4    = i % 80;
                const int token = grouped_tokens[start_idx + token_base + b_idx];
                const int path  = grouped_paths[start_idx + token_base + b_idx];
                const auto* src = activations + (static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + u4 * 8;
                s_act[b_idx][u4] = *reinterpret_cast<const uint4*>(src);
            }
            __syncthreads();

            float down_acc[kBatch][2] = {};

            for (int group = lane; group < 40; group += 32) {
                float2 dw[2][8];
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    const int row = row_base + r;
                    if (row < kHidden) {
                        const auto* down_packed = codes_base + static_cast<std::int64_t>(row) * (kIntermediate / 2) + group * 8;
                        const uint2 d_words     = *reinterpret_cast<const uint2*>(down_packed);

                        #pragma unroll
                        for (int pair = 0; pair < 8; ++pair) {
                            const std::uint32_t d_w = (pair < 4) ? d_words.x : d_words.y;
                            const std::uint8_t dp   = static_cast<std::uint8_t>(d_w >> (8 * (pair & 3)));
                            dw[r][pair]             = ops::detail::decode_nvfp4_e2m1x2(dp);
                        }
                    }
                }

                #pragma unroll
                for (int b = 0; b < kBatch; ++b) {
                    if (b < batch_tokens) {
                        const auto h_v0 = s_act[b][group * 2];
                        const auto h_v1 = s_act[b][group * 2 + 1];
                        const std::uint32_t h_raw[8] = {
                            h_v0.x, h_v0.y, h_v0.z, h_v0.w,
                            h_v1.x, h_v1.y, h_v1.z, h_v1.w
                        };

                        #pragma unroll
                        for (int r = 0; r < 2; ++r) {
                            if (row_base + r < kHidden) {
                                const float down_coeff = ops::detail::decode_nvfp4_e4m3(
                                    s_down_scales[local_row + r][group]) * inv_divisor;
                                float d_sum_raw = 0.0F;
                                #pragma unroll
                                for (int pair = 0; pair < 8; ++pair) {
                                    const float2 hv = ops::bf16x2_bits_to_float2(h_raw[pair]);
                                    d_sum_raw = fmaf(dw[r][pair].x, hv.x, d_sum_raw);
                                    d_sum_raw = fmaf(dw[r][pair].y, hv.y, d_sum_raw);
                                }
                                down_acc[b][r] = fmaf(d_sum_raw, down_coeff, down_acc[b][r]);
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int b = 0; b < kBatch; ++b) {
                if (b < batch_tokens) {
                    const int token = grouped_tokens[start_idx + token_base + b];
                    const int path  = grouped_paths[start_idx + token_base + b];

                    #pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const int row = row_base + r;
                        if (row < kHidden) {
                            const float down_sum = ops::warp_reduce_sum(down_acc[b][r]);
                            if (lane == 0) {
                                down_intermediate[(static_cast<std::int64_t>(token) * kHidden + row) * kTopK + path] =
                                    down_sum;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

// Step 4b: Grouped Expert Down GEMM (Native NVFP4 Tensor Core MMA with Fused Routing Alpha Epilogue)
// Tile: 64 output rows x 16 tokens per CTA (2 warps along M x 2 warps along N, 2 sub-tiles along M)
// CTA: 128 threads (4 warps).
// Shared memory: 20480 + 2560 (weights) + 5120 + 640 (activations) = 28800 bytes (28.125 KB <= 48 KB hardware limit)
// Grid: (kHidden / 64, 512)
__global__ __launch_bounds__(128, 4) void flash_next_moe_prefill_down_mma_kernel(
    const std::uint8_t* __restrict__ act_codes,     // [320, 11 * tokens]
    const std::uint8_t* __restrict__ act_scales,    // [40, 11 * tokens]
    const std::int32_t* __restrict__ expert_offsets,
    const std::int32_t* __restrict__ expert_counts,
    const std::int32_t* __restrict__ active_experts,
    const std::int32_t* __restrict__ active_count_ptr,
    const std::int32_t* __restrict__ grouped_tokens,
    const std::int32_t* __restrict__ grouped_paths,
    const float* __restrict__ alpha_weights,        // [10, tokens]
    const std::uint8_t* __restrict__ expert_codes,
    const std::uint8_t* __restrict__ expert_scales,
    const float* __restrict__ expert_divisors,
    std::uint64_t code_stride,
    std::uint64_t scale_stride,
    __nv_bfloat16* __restrict__ staged_down) {      // [10 * tokens, 2560]

    const int total_active = active_count_ptr[0];
    const int active_idx   = static_cast<int>(blockIdx.y);
    if (active_idx >= total_active) { return; }

    const int expert = active_experts[active_idx];
    const int count  = expert_counts[expert];
    if (count <= 0) { return; }

    const int row_base = static_cast<int>(blockIdx.x) * 64;
    if (row_base >= kHidden) { return; }

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int warp_m = warp >> 1; // 0 or 1 (rows 0..31 or rows 32..63)
    const int warp_n = warp & 1;  // 0 or 1 (tokens 0..7 or tokens 8..15)
    const int lane   = tid & 31;

    __shared__ alignas(16) std::uint8_t s_w_codes[4][16 * 320];
    __shared__ alignas(16) std::uint8_t s_w_scales[4][16 * 40];
    __shared__ alignas(16) std::uint8_t s_a_codes[2][8 * 320];
    __shared__ alignas(16) std::uint8_t s_a_scales[2][8 * 40];

    const auto* exp_codes  = expert_codes + static_cast<std::uint64_t>(expert) * code_stride;
    const auto* exp_scales = expert_scales + static_cast<std::uint64_t>(expert) * scale_stride;
    const int offset       = expert_offsets[expert];
    const float inv_div    = 1.0F / expert_divisors[expert];

    // Cooperatively load 64 rows of Down weights (20480 bytes) across all 128 threads (10 iterations)
    #pragma unroll
    for (int i = tid * 16; i < 64 * 320; i += 128 * 16) {
        const int row = i / 320;
        const int col = i - row * 320;
        const int global_row = row_base + row;
        const int m_grp = row >> 4;
        const int r_in  = row & 15;
        *reinterpret_cast<uint4*>(s_w_codes[m_grp] + r_in * 320 + col) =
            *reinterpret_cast<const uint4*>(exp_codes + static_cast<std::int64_t>(global_row) * 320 + col);
    }

    // Cooperatively unpack 640 scale words (2560 bytes) across all 128 threads (5 iterations)
    #pragma unroll
    for (int i = tid; i < 64 * 10; i += 128) {
        const int row  = i / 10;
        const int tile = i - row * 10;
        const int global_row = row_base + row;
        const int m_tile     = global_row / 128;
        const int row_inner  = global_row % 128;
        const int row_mod32  = row_inner & 31;
        const int row_quart  = row_inner >> 5;
        const std::int64_t off = static_cast<std::int64_t>(m_tile * 10 + tile) * 512 +
                                 row_mod32 * 16 + row_quart * 4;
        const int m_grp = row >> 4;
        const int r_in  = row & 15;
        *reinterpret_cast<std::uint32_t*>(s_w_scales[m_grp] + r_in * 40 + tile * 4) =
            *reinterpret_cast<const std::uint32_t*>(exp_scales + off);
    }
    __syncthreads();

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    for (int t_chunk = 0; t_chunk < count; t_chunk += 16) {
        // Cooperatively load 16 tokens of intermediate activations across all 128 threads
        #pragma unroll
        for (int i = tid * 16; i < 16 * 320; i += 128 * 16) {
            const int tok_idx = i / 320;
            const int col     = i - tok_idx * 320;
            if (t_chunk + tok_idx < count) {
                const int pos     = offset + t_chunk + tok_idx;
                const int tok     = grouped_tokens[pos];
                const int path    = grouped_paths[pos];
                const int flat_id = tok * kPaths + path;
                *reinterpret_cast<uint4*>(s_a_codes[tok_idx / 8] + (tok_idx % 8) * 320 + col) =
                    *reinterpret_cast<const uint4*>(act_codes + static_cast<std::int64_t>(flat_id) * 320 + col);
            } else {
                *reinterpret_cast<uint4*>(s_a_codes[tok_idx / 8] + (tok_idx % 8) * 320 + col) = make_uint4(0, 0, 0, 0);
            }
        }

        #pragma unroll
        for (int i = tid * 4; i < 16 * 40; i += 128 * 4) {
            const int tok_idx = i / 40;
            const int col     = i - tok_idx * 40;
            if (t_chunk + tok_idx < count) {
                const int pos     = offset + t_chunk + tok_idx;
                const int tok     = grouped_tokens[pos];
                const int path    = grouped_paths[pos];
                const int flat_id = tok * kPaths + path;
                *reinterpret_cast<std::uint32_t*>(s_a_scales[tok_idx / 8] + (tok_idx % 8) * 40 + col) =
                    *reinterpret_cast<const std::uint32_t*>(act_scales + static_cast<std::int64_t>(flat_id) * 40 + col);
            } else {
                *reinterpret_cast<std::uint32_t*>(s_a_scales[tok_idx / 8] + (tok_idx % 8) * 40 + col) = 0;
            }
        }
        __syncthreads();

        const int t_base    = t_chunk + warp_n * 8;
        const int cur_batch = max(0, min(8, count - t_base));

        if (cur_batch > 0) {
            #pragma unroll
            for (int sub_m = 0; sub_m < 2; ++sub_m) {
                const int grp_idx = warp_m * 2 + sub_m;
                float accumulators[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                #pragma unroll 2
                for (int k64 = 0; k64 < 10; ++k64) {
                    unsigned a[4];
                    unsigned b[2];

                    const int a_row = a_row_offset;
                    const auto* a_addr = s_w_codes[grp_idx] + a_row * 320 + k64 * 32 + a_column_byte;
                    ops::ldmatrix_x4(a[0], a[1], a[2], a[3], ops::smem_addr(a_addr));

                    const int b_row = b_row_offset;
                    const auto* b_addr = s_a_codes[warp_n] + b_row * 320 + k64 * 32 + b_column_byte;
                    ops::ldmatrix_x2(b[0], b[1], ops::smem_addr(b_addr));

                    const uint32_t sfa = *reinterpret_cast<const uint32_t*>(s_w_scales[grp_idx] + sfa_row * 40 + k64 * 4);
                    const uint32_t sfb = *reinterpret_cast<const uint32_t*>(s_a_scales[warp_n] + sfb_row * 40 + k64 * 4);

                    ops::mma_nvfp4_e4m3(accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                                       a[0], a[1], a[2], a[3], b[0], b[1], sfa, sfb);
                }

                const int tok0 = 2 * (lane & 3);
                const int tok1 = tok0 + 1;
                const int row0 = row_base + grp_idx * 16 + (lane >> 2);
                const int row1 = row0 + 8;

                if (tok0 < cur_batch) {
                    const int pos  = offset + t_base + tok0;
                    const int tok  = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    const float w  = alpha_weights[tok * kTopK + path] * inv_div;
                    staged_down[static_cast<std::int64_t>(pos) * kHidden + row0] = __float2bfloat16_rn(accumulators[0] * w);
                    staged_down[static_cast<std::int64_t>(pos) * kHidden + row1] = __float2bfloat16_rn(accumulators[2] * w);
                }
                if (tok1 < cur_batch) {
                    const int pos  = offset + t_base + tok1;
                    const int tok  = grouped_tokens[pos];
                    const int path = grouped_paths[pos];
                    const float w  = alpha_weights[tok * kTopK + path] * inv_div;
                    staged_down[static_cast<std::int64_t>(pos) * kHidden + row0] = __float2bfloat16_rn(accumulators[1] * w);
                    staged_down[static_cast<std::int64_t>(pos) * kHidden + row1] = __float2bfloat16_rn(accumulators[3] * w);
                }
            }
        }
        __syncthreads();
    }
}

// Step 5: Prefill Shared Expert Down Kernel (BF16 SIMT FMA, not a tensor-core MMA)
// Computes SharedDown(2560 x 640) * activations_shared(640 x tokens) and scales by shared_scale[token],
// initializing the layer output (BF16 [2560, tokens]).
// Grid: (kHidden / 64, (tokens + 15) / 16)
// Threads: 128 (4 warps).
__global__ __launch_bounds__(128, 4) void flash_next_moe_prefill_shared_down_kernel(
    const __nv_bfloat16* __restrict__ shared_down,
    const __nv_bfloat16* __restrict__ activations,
    const float* __restrict__ shared_scale,
    __nv_bfloat16* __restrict__ output,
    int tokens) {

    const int row_base   = static_cast<int>(blockIdx.x) * 64;
    const int token_base = static_cast<int>(blockIdx.y) * 16;
    if (row_base >= kHidden || token_base >= tokens) { return; }

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int warp_m = warp >> 1; // 0 or 1 (rows 0..31 or rows 32..63)
    const int warp_n = warp & 1;  // 0 or 1 (tokens 0..7 or tokens 8..15)
    const int lane   = tid & 31;

    __shared__ alignas(16) __nv_bfloat16 s_w[64 * 128]; // 16 KB
    __shared__ alignas(16) __nv_bfloat16 s_x[16 * 128]; // 4 KB

    float accum[2][4] = {}; // [sub_m][4 accumulators]
    const int t_offset = token_base + warp_n * 8;

    for (int k_stage = 0; k_stage < 5; ++k_stage) {
        const int k_base = k_stage * 128;

        // Load 64 rows x 128 cols weights (8192 elements = 16384 bytes)
        #pragma unroll
        for (int i = tid * 8; i < 64 * 128; i += 128 * 8) {
            const int r = i / 128;
            const int c = i % 128;
            *reinterpret_cast<uint4*>(&s_w[r * 128 + c]) =
                *reinterpret_cast<const uint4*>(&shared_down[static_cast<std::int64_t>(row_base + r) * kIntermediate + k_base + c]);
        }

        // Load 16 tokens x 128 cols activations (2048 elements = 4096 bytes)
        #pragma unroll
        for (int i = tid * 8; i < 16 * 128; i += 128 * 8) {
            const int tok_idx = i / 128;
            const int c       = i % 128;
            const int tok     = token_base + tok_idx;
            if (tok < tokens) {
                const auto* src = activations + (static_cast<std::int64_t>(tok) * kPaths + kTopK) * kIntermediate + k_base + c;
                *reinterpret_cast<uint4*>(&s_x[tok_idx * 128 + c]) =
                    *reinterpret_cast<const uint4*>(src);
            } else {
                *reinterpret_cast<uint4*>(&s_x[tok_idx * 128 + c]) = make_uint4(0, 0, 0, 0);
            }
        }
        __syncthreads();

        // Compute 32 rows x 8 tokens for this warp
        #pragma unroll
        for (int sub_m = 0; sub_m < 2; ++sub_m) {
            const int r_local0 = warp_m * 32 + sub_m * 16 + (lane >> 2);
            const int r_local1 = r_local0 + 8;
            const int tok0_local = warp_n * 8 + 2 * (lane & 3);
            const int tok1_local = tok0_local + 1;

            #pragma unroll
            for (int k_idx = 0; k_idx < 128; k_idx += 2) {
                const float2 w0 = ops::bf16x2_bits_to_float2(*reinterpret_cast<const uint32_t*>(&s_w[r_local0 * 128 + k_idx]));
                const float2 w1 = ops::bf16x2_bits_to_float2(*reinterpret_cast<const uint32_t*>(&s_w[r_local1 * 128 + k_idx]));
                const float2 x0 = ops::bf16x2_bits_to_float2(*reinterpret_cast<const uint32_t*>(&s_x[tok0_local * 128 + k_idx]));
                const float2 x1 = ops::bf16x2_bits_to_float2(*reinterpret_cast<const uint32_t*>(&s_x[tok1_local * 128 + k_idx]));

                accum[sub_m][0] = fmaf(w0.x, x0.x, fmaf(w0.y, x0.y, accum[sub_m][0]));
                accum[sub_m][1] = fmaf(w0.x, x1.x, fmaf(w0.y, x1.y, accum[sub_m][1]));
                accum[sub_m][2] = fmaf(w1.x, x0.x, fmaf(w1.y, x0.y, accum[sub_m][2]));
                accum[sub_m][3] = fmaf(w1.x, x1.x, fmaf(w1.y, x1.y, accum[sub_m][3]));
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int sub_m = 0; sub_m < 2; ++sub_m) {
        const int row0 = row_base + warp_m * 32 + sub_m * 16 + (lane >> 2);
        const int row1 = row0 + 8;
        const int tok0 = t_offset + 2 * (lane & 3);
        const int tok1 = tok0 + 1;

        if (tok0 < tokens) {
            const float s_scale0 = shared_scale[tok0];
            output[static_cast<std::int64_t>(tok0) * kHidden + row0] = __float2bfloat16_rn(accum[sub_m][0] * s_scale0);
            output[static_cast<std::int64_t>(tok0) * kHidden + row1] = __float2bfloat16_rn(accum[sub_m][2] * s_scale0);
        }
        if (tok1 < tokens) {
            const float s_scale1 = shared_scale[tok1];
            output[static_cast<std::int64_t>(tok1) * kHidden + row0] = __float2bfloat16_rn(accum[sub_m][1] * s_scale1);
            output[static_cast<std::int64_t>(tok1) * kHidden + row1] = __float2bfloat16_rn(accum[sub_m][3] * s_scale1);
        }
    }
}

// Step 6: Fused Fixed-Order FP32 Warp-Tile Reduction Kernel
// Sums SharedDown base + 10 routed paths in strictly fixed sequential order p = 0..9
// Grid: (kHidden / 64, (tokens + 15) / 16)
// Threads: 128 (4 warps).
__global__ __launch_bounds__(128, 4) void flash_next_moe_prefill_down_reduce_fused_kernel(
    const std::int32_t* __restrict__ token_to_pos,
    const __nv_bfloat16* __restrict__ staged_down,
    __nv_bfloat16* __restrict__ output,
    int tokens) {

    const int row_base   = static_cast<int>(blockIdx.x) * 64;
    const int token_base = static_cast<int>(blockIdx.y) * 16;
    if (row_base >= kHidden || token_base >= tokens) { return; }

    const int tid = static_cast<int>(threadIdx.x);

    #pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int local_idx = tid * 8 + item;
        const int r_off     = local_idx >> 4; // 0..63
        const int t_off     = local_idx & 15; // 0..15
        const int tok       = token_base + t_off;
        const int row       = row_base + r_off;

        if (tok < tokens) {
            const auto* base_ptr = output + static_cast<std::int64_t>(tok) * kHidden + row;
            float sum = __bfloat162float(*base_ptr);

            #pragma unroll
            for (int p = 0; p < 10; ++p) {
                const int pos = token_to_pos[tok * 10 + p];
                const float v = __bfloat162float(staged_down[static_cast<std::int64_t>(pos) * kHidden + row]);
                sum += v;
            }

            output[static_cast<std::int64_t>(tok) * kHidden + row] = __float2bfloat16_rn(sum);
        }
    }
}

// Step 5 (SIMT fallback): Prefill Shared Expert Down & Routed Path Reduction (Vectorized 128-bit loads for T < 512)
// CTA: 256 threads (8 warps). Each warp processes 1 row -> 8 rows per CTA.
// Grid.x = 2560 / 8 = 320, Grid.y = (tokens + 7) / 8.
__global__ void flash_next_moe_prefill_down_reduce_kernel(
    const std::int32_t* __restrict__ ids,
    const float* __restrict__ alpha,
    const float* __restrict__ shared_scale,
    const __nv_bfloat16* __restrict__ activations,
    const float* __restrict__ down_intermediate,
    const __nv_bfloat16* __restrict__ shared_down,
    __nv_bfloat16* __restrict__ output,
    int tokens) {
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int row          = static_cast<int>(blockIdx.x) * 8 + warp;
    const int token_base   = static_cast<int>(blockIdx.y) * 8;
    const int batch_tokens = min(8, tokens - token_base);

    if (row >= kHidden || batch_tokens <= 0) { return; }

    const auto* sd_row = shared_down + static_cast<std::int64_t>(row) * kIntermediate;
    float shared_val[8] = {};

    for (int col_base = 0; col_base < kIntermediate; col_base += 256) {
        const int col = col_base + lane * 8;
        if (col < kIntermediate) {
            const auto sd_v = *reinterpret_cast<const uint4*>(&sd_row[col]);
            const std::uint32_t sd_raw[4] = {sd_v.x, sd_v.y, sd_v.z, sd_v.w};

            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                if (b < batch_tokens) {
                    const int token      = token_base + b;
                    const auto* shared_x = activations + (static_cast<std::int64_t>(token) * kPaths + kTopK) * kIntermediate;
                    const auto sx_v      = *reinterpret_cast<const uint4*>(&shared_x[col]);
                    const std::uint32_t sx_raw[4] = {sx_v.x, sx_v.y, sx_v.z, sx_v.w};

                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const float2 s_pair = ops::bf16x2_bits_to_float2(sd_raw[pair]);
                        const float2 x_pair = ops::bf16x2_bits_to_float2(sx_raw[pair]);
                        shared_val[b] = fmaf(s_pair.x, x_pair.x, shared_val[b]);
                        shared_val[b] = fmaf(s_pair.y, x_pair.y, shared_val[b]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        if (b < batch_tokens) {
            const int token   = token_base + b;
            const float s_val = ops::warp_reduce_sum(shared_val[b]);

            // Parallel coalesced load of the 10 paths across lanes 0..9
            float p_val = 0.0F;
            if (lane < kTopK) {
                p_val = alpha[token * kTopK + lane] * down_intermediate[(static_cast<std::int64_t>(token) * kHidden + row) * kTopK + lane];
            }

            // All threads in the warp participate in the collective shuffle
            float routed = p_val;
            #pragma unroll
            for (int src = 1; src < kTopK; ++src) {
                const float v = __shfl_sync(0xffffffff, p_val, src);
                if (lane == 0) {
                    routed += v;
                }
            }

            if (lane == 0) {
                const float s_scale = shared_scale[token];
                const float total   = routed + s_val * s_scale;
                output[static_cast<std::int64_t>(token) * kHidden + row] = __float2bfloat16_rn(total);
            }
        }
    }
}

__global__ void flash_next_moe_bf16_gate_up_kernel(
    const __nv_bfloat16* __restrict__ input, const std::int32_t* __restrict__ ids,
    const __nv_bfloat16* __restrict__ expert_gate_up, std::uint64_t expert_stride,
    const __nv_bfloat16* __restrict__ shared_gate, const __nv_bfloat16* __restrict__ shared_up,
    __nv_bfloat16* __restrict__ activations) {
    const int token = static_cast<int>(blockIdx.y);
    const int path  = static_cast<int>(blockIdx.z);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * GateSchedule::kWarpsPerCta + warp;
    if (row >= kIntermediate) { return; }
    const auto* x = input + static_cast<std::int64_t>(token) * kHidden;
    float gate    = 0.0F;
    float up      = 0.0F;
    if (path < kTopK) {
        const int expert    = ids[token * kTopK + path];
        const auto* exp_ptr = expert_gate_up + static_cast<std::uint64_t>(expert) * expert_stride;
        const auto* gate_w  = exp_ptr + static_cast<std::int64_t>(row) * kHidden;
        const auto* up_w    = exp_ptr + static_cast<std::int64_t>(row + kIntermediate) * kHidden;
        dot_bf16_pair<kHidden>(x, gate_w, up_w, gate, up);
    } else {
        dot_bf16_pair<kHidden>(x, shared_gate + static_cast<std::int64_t>(row) * kHidden,
                               shared_up + static_cast<std::int64_t>(row) * kHidden, gate, up);
    }
    if (lane == 0) {
        activations[(static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate + row] =
            __float2bfloat16_rn(ops::silu(gate) * up);
    }
}

__global__ void flash_next_moe_bf16_down_kernel(
    const std::int32_t* __restrict__ ids, const float* __restrict__ alpha,
    const float* __restrict__ shared_scale, const __nv_bfloat16* __restrict__ activations,
    const __nv_bfloat16* __restrict__ expert_down, std::uint64_t expert_stride,
    const __nv_bfloat16* __restrict__ shared_down, __nv_bfloat16* __restrict__ output) {
    const int token = static_cast<int>(blockIdx.y);
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int row   = static_cast<int>(blockIdx.x) * kDownWarps + warp;
    if (row >= kHidden) { return; }

    float routed = 0.0F;
#pragma unroll
    for (int path = 0; path < kTopK; ++path) {
        const int expert = ids[token * kTopK + path];
        const auto* x =
            activations + (static_cast<std::int64_t>(token) * kPaths + path) * kIntermediate;
        const auto* down_w = expert_down + static_cast<std::uint64_t>(expert) * expert_stride +
                             static_cast<std::int64_t>(row) * kIntermediate;
        const float value = dot_bf16<kIntermediate>(x, down_w);
        if (lane == 0) { routed = fmaf(alpha[token * kTopK + path], value, routed); }
    }
    const auto* shared_x =
        activations + (static_cast<std::int64_t>(token) * kPaths + kTopK) * kIntermediate;
    const float shared_value = dot_bf16<kIntermediate>(
        shared_x, shared_down + static_cast<std::int64_t>(row) * kIntermediate);
    if (lane == 0) {
        output[static_cast<std::int64_t>(token) * kHidden + row] =
            __float2bfloat16_rn(fmaf(shared_scale[token], shared_value, routed));
    }
}

// NINFER_FLASH_NEXT_LAUNCH_DEBUG=1: report the REAL decode-launch geometry -- grid, block,
// registers, the occupancy query's max CTAs/SM, and the resulting WAVE COUNT. Static grid sizes can
// be read off the source, but max_blocks_per_sm comes out of ptxas's register allocation, so only a
// runtime query gives the true wave count and the true resident warps per SM. Prints at most a
// dozen times, so it cannot flood a 1,024-token run. MUST be defined before launch_down_decode.
bool launch_debug_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_LAUNCH_DEBUG");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

template <class Kernel>
void report_launch_shape(const char* what, Kernel* kernel, dim3 grid, int threads) {
    if (!launch_debug_enabled()) { return; }
    static int printed = 0;
    if (printed >= 12) { return; }
    ++printed;
    cudaFuncAttributes attrs{};
    CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
    int max_blocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, threads, 0));
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    const double ctas  = static_cast<double>(grid.x) * grid.y * grid.z;
    const double slots = static_cast<double>(sms * (max_blocks > 0 ? max_blocks : 1));
    const double warps = ctas * (threads / 32.0);
    std::fprintf(stderr,
                 "flash_next_moe_launch %-10s grid=(%u,%u,%u) block=%d regs=%d maxcta/sm=%d "
                 "ctas=%.0f slots=%.0f waves=%.2f warps=%.0f warps/sm=%.1f resident_warps/sm=%d\n",
                 what, grid.x, grid.y, grid.z, threads, attrs.numRegs, max_blocks, ctas, slots,
                 ctas / slots, warps, warps / sms, max_blocks * (threads / 32));
}

// Decode-arm down projection with an explicit kernel choice. The argument list is the same for
// both kernels; only the grid shape differs (one row per warp vs one row per CTA).
void launch_down_decode(FlashNextMoeDownKernel kernel, const MoeWeights& weights,
                        const FlashNextMoeWorkspace& workspace, int tokens, Tensor& output,
                        cudaStream_t stream) {
    const auto* ids          = static_cast<const std::int32_t*>(workspace.ids.data);
    const auto* alpha        = static_cast<const float*>(workspace.alpha.data);
    const auto* shared_scale = static_cast<const float*>(workspace.shared_scale.data);
    const auto* activations  = static_cast<const __nv_bfloat16*>(workspace.activations.data);
    const auto* codes        = reinterpret_cast<const std::uint8_t*>(weights.expert_down.codes);
    const auto* scales       = reinterpret_cast<const std::uint8_t*>(weights.expert_down.scales);
    const float* divisors    = weights.expert_down.weight_scale_divisors;
    const std::uint64_t code_stride  = weights.expert_down.code_bytes_per_expert;
    const std::uint64_t scale_stride = weights.expert_down.scale_bytes_per_expert;
    const auto* shared_down  = static_cast<const __nv_bfloat16*>(weights.shared_down.qdata);
    auto* out                = static_cast<__nv_bfloat16*>(output.data);

    if (kernel == FlashNextMoeDownKernel::Legacy) {
        const dim3 down_grid(kHidden / kDownWarps, static_cast<unsigned>(tokens));
        report_launch_shape("down_legacy", flash_next_moe_down_kernel, down_grid, kDownWarps * 32);
        flash_next_moe_down_kernel<<<down_grid, kDownWarps * 32, 0, stream>>>(
            ids, alpha, shared_scale, activations, codes, scales, divisors, code_stride,
            scale_stride, shared_down, out);
    } else {
        const dim3 down_grid(kHidden, static_cast<unsigned>(tokens));
        report_launch_shape("down_pathwarp", flash_next_moe_down_pathwarp_kernel, down_grid,
                            kDownPathThreads);
        flash_next_moe_down_pathwarp_kernel<<<down_grid, kDownPathThreads, 0, stream>>>(
            ids, alpha, shared_scale, activations, codes, scales, divisors, code_stride,
            scale_stride, shared_down, out);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Typed so the runtime's template overloads resolve the kernel entry; no function-pointer to
// void* cast, which is only conditionally supported.
template <class Kernel>
FlashNextMoeDownKernelAttributes down_kernel_attributes(Kernel* kernel, int threads_per_block) {
    FlashNextMoeDownKernelAttributes out{};
    out.threads_per_block = threads_per_block;
    cudaFuncAttributes attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    out.registers_per_thread = attributes.numRegs;
    out.local_bytes          = attributes.localSizeBytes;
    out.static_smem_bytes    = attributes.sharedSizeBytes;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&out.max_blocks_per_sm, kernel,
                                                             threads_per_block, 0));
    return out;
}

} // namespace

// Hybrid dispatch threshold:
// At T < 512 (e.g. T=128), average active tokens per expert is small (~2.7), where SIMT W4A16
// avoids activation quantization overhead and achieves lower latency.
// At T >= 512 (e.g. T=512, 2048), tokens per expert is high (>= 10), where Native NVFP4 MMA
// achieves 1.26x+ higher throughput.
constexpr int kMmaPrefillThreshold = 512;

FlashNextMoeDownKernel flash_next_moe_down_kernel_selection() {
    // Read once: this sits on the per-layer decode path (48 launches per token), and a getenv
    // per launch is host overhead the decode graph would otherwise replay. The bitwise gate
    // therefore selects kernels through flash_next_moe_down_launch, not by flipping the
    // environment mid-process.
    static const FlashNextMoeDownKernel selection = []() {
        const char* env = std::getenv("NINFER_FLASH_NEXT_MOE_DOWN_LEGACY");
        const bool legacy = env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
        return legacy ? FlashNextMoeDownKernel::Legacy : FlashNextMoeDownKernel::PathWarp;
    }();
    return selection;
}

void flash_next_moe_down_launch(FlashNextMoeDownKernel kernel, const MoeWeights& weights,
                                const FlashNextMoeWorkspace& workspace, int tokens,
                                Tensor& output, cudaStream_t stream) {
    if (tokens < 1 || tokens > 8) {
        throw std::invalid_argument("Flash-Next MoE decode down launch requires 1 <= tokens <= 8");
    }
    launch_down_decode(kernel, weights, workspace, tokens, output, stream);
}

FlashNextMoeDownKernelAttributes
flash_next_moe_down_kernel_attributes(FlashNextMoeDownKernel kernel) {
    if (kernel == FlashNextMoeDownKernel::Legacy) {
        return down_kernel_attributes(flash_next_moe_down_kernel, kDownWarps * 32);
    }
    return down_kernel_attributes(flash_next_moe_down_pathwarp_kernel, kDownPathThreads);
}


void flash_next_moe_kernels_launch(const Tensor& input, const MoeWeights& weights,
                                   const FlashNextMoeWorkspace& workspace, Tensor& output,
                                   cudaStream_t stream) {
    const int tokens = static_cast<int>(input.ne[1]);

#if defined(NINFER_SM8X_COMPAT) && !defined(NINFER_SM120_NVFP4)
    // This tree's spelling of the gate (upstream calls it `NINFER_SM89`, which is defined nowhere
    // here). See the block comment in `impl/moe.cpp` at the same gate, and `docs/20 §16.9`: while
    // this was dead, a prefill chunk of 512 tokens or more reached the arm below it and ended in
    // `__trap()` on sm_89 instead of failing here by name.
    // Local sm_89 port. This guard used to refuse EVERY tokens > 8 call, which was one rung too
    // broad: only the tokens >= kMmaPrefillThreshold arm needs the Blackwell NVFP4 block-scaled MMA
    // (kind::mxf4nvf4 via ops::mma_nvfp4_e4m3, which has no Ada equivalent -- refuse host-side so the
    // device-side trap is never reached). The 8 < tokens < 512 arm below is SIMT W4A16 plus the BF16
    // MMA shared/route kernels (launch_bf16_mma is sm_80-floor), all of which Ada has. Widening this
    // gate is what lets NINFER_FLASH_NEXT_MOE_WIDE_ARM reach the grouped prefill arm at all: there is
    // no wgmma or other sm_90-only intrinsic anywhere on that path (only core/pdl.cuh is arch-gated,
    // and it compiles to a no-op below 900).
    if (tokens >= kMmaPrefillThreshold) {
        throw std::runtime_error(
            "Flash-Next MoE prefill at tokens >= 512 needs the NVFP4 block-scaled MMA path, which is "
            "not implemented for sm_89; the SIMT W4A16 prefill arm covers 9..511 and the decode path "
            "covers 1..8");
    }
#endif

    if (tokens <= 8) {
        // Decode path (tokens <= 8): UNTOUCHED fused kernels
        const dim3 gate_grid(kIntermediate / GateSchedule::kWarpsPerCta,
                             static_cast<unsigned>(tokens), kPaths);
        const auto launch_gate_up = [&](auto hoisted_tag) {
            report_launch_shape("gate_up", flash_next_moe_gate_up_kernel<decltype(hoisted_tag)::value>,
                                gate_grid, GateSchedule::kThreads);
            flash_next_moe_gate_up_kernel<decltype(hoisted_tag)::value>
                <<<gate_grid, GateSchedule::kThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data),
                    static_cast<const std::int32_t*>(workspace.ids.data),
                    reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
                    reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
                    weights.expert_gate_up.weight_scale_divisors,
                    weights.expert_gate_up.code_bytes_per_expert,
                    weights.expert_gate_up.scale_bytes_per_expert,
                    static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
                    static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
                    static_cast<__nv_bfloat16*>(workspace.activations.data));
        };
        // NINFER_FLASH_NEXT_MOE_HOIST=1 selects the hoisted-coefficient inner loop (see the note
        // above the kernel); default keeps the shipped order bit-for-bit.
        if (flash_next_moe_hoist_enabled()) {
            launch_gate_up(std::integral_constant<bool, true>{});
        } else {
            launch_gate_up(std::integral_constant<bool, false>{});
        }
        CUDA_CHECK(cudaGetLastError());

        // Path-per-warp by default; NINFER_FLASH_NEXT_MOE_DOWN_LEGACY=1 pins the legacy kernel.
        launch_down_decode(flash_next_moe_down_kernel_selection(), weights, workspace, tokens,
                           output, stream);
    } else {
        // Prefill path (tokens > 8): Group tokens by expert, load weights once per chunk
        // 1. Group tokens by expert and build active expert list
        if (flash_next_moe_shared_mma_enabled()) {
            flash_next_moe_prefill_build_groups_mma(
                workspace.ids, workspace.expert_counts, workspace.expert_offsets,
                workspace.active_expert_ids, workspace.active_count, workspace.grouped_tokens,
                workspace.grouped_paths, workspace.grouped_experts, workspace.token_to_pos,
                workspace.task_counter, tokens, stream);
        } else {
            flash_next_moe_prefill_build_groups_kernel<<<1, 512, 0, stream>>>(
                static_cast<const std::int32_t*>(workspace.ids.data),
                static_cast<std::int32_t*>(workspace.expert_counts.data),
                static_cast<std::int32_t*>(workspace.expert_offsets.data),
                static_cast<std::int32_t*>(workspace.active_expert_ids.data),
                static_cast<std::int32_t*>(workspace.active_count.data),
                static_cast<std::int32_t*>(workspace.grouped_tokens.data),
                static_cast<std::int32_t*>(workspace.grouped_paths.data),
                static_cast<std::int32_t*>(workspace.grouped_experts.data),
                static_cast<std::int32_t*>(workspace.token_to_pos.data),
                static_cast<std::int32_t*>(workspace.task_counter.data),
                tokens);
            CUDA_CHECK(cudaGetLastError());
        }
        stage_ledger_record(stream, FlashNextStageId::MoE_Grouping);

        if (tokens >= kMmaPrefillThreshold) {
            // Large tokens: Native NVFP4 Tensor Core MMA route
            // 2. Shared expert gate & up. Completely disjoint from routed MMA.
            if (flash_next_moe_shared_mma_enabled()) {
                flash_next_moe_prefill_shared_gate_up_mma(
                    input, weights.shared_gate, weights.shared_up, workspace.activations,
                    workspace.shared_gemm, tokens, stream);
            } else {
                const dim3 shared_grid(kIntermediate / 8, (static_cast<unsigned>(tokens) + 7) / 8);
                flash_next_moe_prefill_shared_gate_up_kernel<<<shared_grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data),
                    static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
                    static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
                    static_cast<__nv_bfloat16*>(workspace.activations.data),
                    tokens);
                CUDA_CHECK(cudaGetLastError());
            }
            stage_ledger_record(stream, FlashNextStageId::MoE_SharedGateUp);

            // 3. Quantize input activations X -> NVFP4
            constexpr int kActThreads = 256;
            const int act_tasks = tokens * Activation2560Geometry::kGroupsPerRow;
            ops::detail::nvfp4_w4a4_quantize_kernel<Activation2560Geometry>
                <<<(act_tasks + kActThreads - 1) / kActThreads, kActThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data),
                    static_cast<std::uint8_t*>(workspace.act_codes.data),
                    static_cast<std::uint8_t*>(workspace.act_scales.data),
                    tokens, 1.0F);
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_QuantInput);

            // 4. Grouped Expert Gate & Up (Native NVFP4 Tensor Core MMA)
            static const bool s_mma_smem_init = []() {
                CUDA_CHECK(cudaFuncSetAttribute(flash_next_moe_prefill_gate_up_mma_kernel<false>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, 69312));
                CUDA_CHECK(cudaFuncSetAttribute(flash_next_moe_prefill_gate_up_mma_kernel<true>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, 92416));
                CUDA_CHECK(cudaFuncSetAttribute(flash_next_moe_prefill_gate_up_mma_v2_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize, 92416));
                return true;
            }();
            (void)s_mma_smem_init;

            const char* staging_env = std::getenv("NINFER_FLASH_NEXT_MOE_STAGING");
            const bool staging_legacy = staging_env && (std::strcmp(staging_env, "0") == 0 ||
                                                         std::strcmp(staging_env, "legacy") == 0 ||
                                                         std::strcmp(staging_env, "old") == 0);
            const bool staging_v1 = staging_env && (std::strcmp(staging_env, "1") == 0 ||
                                                     std::strcmp(staging_env, "v1") == 0);

            if (staging_legacy) {
                const dim3 gate_grid(kIntermediate / 16, 512);
                flash_next_moe_prefill_gate_up_mma_kernel<false><<<gate_grid, 128, 69312, stream>>>(
                    static_cast<const std::uint8_t*>(workspace.act_codes.data),
                    static_cast<const std::uint8_t*>(workspace.act_scales.data),
                    static_cast<const std::int32_t*>(workspace.expert_offsets.data),
                    static_cast<const std::int32_t*>(workspace.expert_counts.data),
                    static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
                    static_cast<const std::int32_t*>(workspace.active_count.data),
                    static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
                    static_cast<const std::int32_t*>(workspace.grouped_paths.data),
                    reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
                    reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
                    weights.expert_gate_up.weight_scale_divisors,
                    weights.expert_gate_up.code_bytes_per_expert,
                    weights.expert_gate_up.scale_bytes_per_expert,
                    static_cast<__nv_bfloat16*>(workspace.activations.data));
            } else if (staging_v1) {
                const dim3 gate_grid(kIntermediate / 16, 512);
                flash_next_moe_prefill_gate_up_mma_kernel<true><<<gate_grid, 128, 92416, stream>>>(
                    static_cast<const std::uint8_t*>(workspace.act_codes.data),
                    static_cast<const std::uint8_t*>(workspace.act_scales.data),
                    static_cast<const std::int32_t*>(workspace.expert_offsets.data),
                    static_cast<const std::int32_t*>(workspace.expert_counts.data),
                    static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
                    static_cast<const std::int32_t*>(workspace.active_count.data),
                    static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
                    static_cast<const std::int32_t*>(workspace.grouped_paths.data),
                    reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
                    reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
                    weights.expert_gate_up.weight_scale_divisors,
                    weights.expert_gate_up.code_bytes_per_expert,
                    weights.expert_gate_up.scale_bytes_per_expert,
                    static_cast<__nv_bfloat16*>(workspace.activations.data));
            } else {
                // Default: High-throughput V2 MMA kernel (M=16 pairs, N=32 tokens, 256 threads / 8 warps)
                const dim3 gate_grid(kIntermediate / 16, 512);
                flash_next_moe_prefill_gate_up_mma_v2_kernel<<<gate_grid, 256, 92416, stream>>>(
                    static_cast<const std::uint8_t*>(workspace.act_codes.data),
                    static_cast<const std::uint8_t*>(workspace.act_scales.data),
                    static_cast<const std::int32_t*>(workspace.expert_offsets.data),
                    static_cast<const std::int32_t*>(workspace.expert_counts.data),
                    static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
                    static_cast<const std::int32_t*>(workspace.active_count.data),
                    static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
                    static_cast<const std::int32_t*>(workspace.grouped_paths.data),
                    reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
                    reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
                    weights.expert_gate_up.weight_scale_divisors,
                    weights.expert_gate_up.code_bytes_per_expert,
                    weights.expert_gate_up.scale_bytes_per_expert,
                    static_cast<__nv_bfloat16*>(workspace.activations.data));
            }
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_RoutedGateUp);

            // 5. Quantize intermediate activations -> NVFP4
            const int down_act_rows = tokens * kPaths;
            const int down_act_tasks = down_act_rows * Activation640Geometry::kGroupsPerRow;
            ops::detail::nvfp4_w4a4_quantize_kernel<Activation640Geometry>
                <<<(down_act_tasks + kActThreads - 1) / kActThreads, kActThreads, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(workspace.activations.data),
                    static_cast<std::uint8_t*>(workspace.down_act_codes.data),
                    static_cast<std::uint8_t*>(workspace.down_act_scales.data),
                    down_act_rows, 1.0F);
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_QuantDown);

            // 6. Shared Down: SharedDown(2560 x 640) * shared_act * shared_scale -> output base
            if (flash_next_moe_shared_mma_enabled()) {
                flash_next_moe_prefill_shared_down_mma(weights.shared_down, workspace.shared_gemm,
                                                       workspace.shared_scale, output, tokens,
                                                       stream);
            } else {
                const dim3 shared_down_grid(kHidden / 64, (static_cast<unsigned>(tokens) + 15) / 16);
                flash_next_moe_prefill_shared_down_kernel<<<shared_down_grid, 128, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
                    static_cast<const __nv_bfloat16*>(workspace.activations.data),
                    static_cast<const float*>(workspace.shared_scale.data),
                    static_cast<__nv_bfloat16*>(output.data),
                    tokens);
                CUDA_CHECK(cudaGetLastError());
            }
            stage_ledger_record(stream, FlashNextStageId::MoE_SharedDown);

            // 7. Grouped Down GEMM (Native NVFP4 Tensor Core MMA with Fused Routing Alpha Epilogue)
            const dim3 down_grid(kHidden / 64, 512);
            flash_next_moe_prefill_down_mma_kernel<<<down_grid, 128, 0, stream>>>(
                static_cast<const std::uint8_t*>(workspace.down_act_codes.data),
                static_cast<const std::uint8_t*>(workspace.down_act_scales.data),
                static_cast<const std::int32_t*>(workspace.expert_offsets.data),
                static_cast<const std::int32_t*>(workspace.expert_counts.data),
                static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
                static_cast<const std::int32_t*>(workspace.active_count.data),
                static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
                static_cast<const std::int32_t*>(workspace.grouped_paths.data),
                static_cast<const float*>(workspace.alpha.data),
                reinterpret_cast<const std::uint8_t*>(weights.expert_down.codes),
                reinterpret_cast<const std::uint8_t*>(weights.expert_down.scales),
                weights.expert_down.weight_scale_divisors,
                weights.expert_down.code_bytes_per_expert,
                weights.expert_down.scale_bytes_per_expert,
                static_cast<__nv_bfloat16*>(workspace.staged_down.data));
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_RoutedDown);

            // 8. Fused Fixed-Order FP32 Warp-Tile Reduction Kernel
            const dim3 reduce_grid(kHidden / 64, (static_cast<unsigned>(tokens) + 15) / 16);
            flash_next_moe_prefill_down_reduce_fused_kernel<<<reduce_grid, 128, 0, stream>>>(
                static_cast<const std::int32_t*>(workspace.token_to_pos.data),
                static_cast<const __nv_bfloat16*>(workspace.staged_down.data),
                static_cast<__nv_bfloat16*>(output.data),
                tokens);
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_Reduce);
        } else {
            // Small tokens (8 < tokens < 512): SIMT W4A16 route (avoids quant overhead)
            // 2. Grouped Expert Gate & Up (SIMT W4A16)
            constexpr int kGridY = 16;
            const dim3 gate_grid(kIntermediate / 8, kGridY);
            flash_next_moe_prefill_gate_up_kernel<<<gate_grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(input.data),
                static_cast<const std::int32_t*>(workspace.expert_offsets.data),
                static_cast<const std::int32_t*>(workspace.expert_counts.data),
                static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
                static_cast<const std::int32_t*>(workspace.active_count.data),
                static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
                static_cast<const std::int32_t*>(workspace.grouped_paths.data),
                reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.codes),
                reinterpret_cast<const std::uint8_t*>(weights.expert_gate_up.scales),
                weights.expert_gate_up.weight_scale_divisors,
                weights.expert_gate_up.code_bytes_per_expert,
                weights.expert_gate_up.scale_bytes_per_expert,
                static_cast<__nv_bfloat16*>(workspace.activations.data));
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_RoutedGateUp);

            // 3. Shared expert gate & up
            if (flash_next_moe_shared_mma_enabled()) {
                flash_next_moe_prefill_shared_gate_up_mma(
                    input, weights.shared_gate, weights.shared_up, workspace.activations,
                    workspace.shared_gemm, tokens, stream);
            } else {
                const dim3 shared_grid(kIntermediate / 8, (static_cast<unsigned>(tokens) + 7) / 8);
                flash_next_moe_prefill_shared_gate_up_kernel<<<shared_grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data),
                    static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
                    static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
                    static_cast<__nv_bfloat16*>(workspace.activations.data),
                    tokens);
                CUDA_CHECK(cudaGetLastError());
            }
            stage_ledger_record(stream, FlashNextStageId::MoE_SharedGateUp);

            // 4. Grouped Down GEMM (SIMT W4A16)
            constexpr int kDownGridY = 8;
            const dim3 down_grid(kHidden / 16, kDownGridY);
            flash_next_moe_prefill_down_kernel<<<down_grid, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(workspace.activations.data),
                static_cast<const std::int32_t*>(workspace.expert_offsets.data),
                static_cast<const std::int32_t*>(workspace.expert_counts.data),
                static_cast<const std::int32_t*>(workspace.active_expert_ids.data),
                static_cast<const std::int32_t*>(workspace.active_count.data),
                static_cast<const std::int32_t*>(workspace.grouped_tokens.data),
                static_cast<const std::int32_t*>(workspace.grouped_paths.data),
                reinterpret_cast<const std::uint8_t*>(weights.expert_down.codes),
                reinterpret_cast<const std::uint8_t*>(weights.expert_down.scales),
                weights.expert_down.weight_scale_divisors,
                weights.expert_down.code_bytes_per_expert,
                weights.expert_down.scale_bytes_per_expert,
                static_cast<float*>(workspace.down_intermediate.data));
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_RoutedDown);

            // 5. Shared Down & Top-K weighted reduction (SIMT)
            const dim3 reduce_grid(kHidden / 8, (static_cast<unsigned>(tokens) + 7) / 8);
            flash_next_moe_prefill_down_reduce_kernel<<<reduce_grid, 256, 0, stream>>>(
                static_cast<const std::int32_t*>(workspace.ids.data),
                static_cast<const float*>(workspace.alpha.data),
                static_cast<const float*>(workspace.shared_scale.data),
                static_cast<const __nv_bfloat16*>(workspace.activations.data),
                static_cast<float*>(workspace.down_intermediate.data),
                static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
                static_cast<__nv_bfloat16*>(output.data),
                tokens);
            CUDA_CHECK(cudaGetLastError());
            stage_ledger_record(stream, FlashNextStageId::MoE_Reduce);
        }
    }
}

void flash_next_moe_bf16_kernels_launch(const Tensor& input, const MoeBf16Weights& weights,
                                        const FlashNextMoeWorkspace& workspace, Tensor& output,
                                        cudaStream_t stream) {
    const int tokens = static_cast<int>(input.ne[1]);
    const std::uint64_t gate_up_stride =
        weights.expert_gate_up.bytes_per_expert / sizeof(std::uint16_t);
    const std::uint64_t down_stride =
        weights.expert_down.bytes_per_expert / sizeof(std::uint16_t);

    const dim3 gate_grid(kIntermediate / GateSchedule::kWarpsPerCta,
                         static_cast<unsigned>(tokens), kPaths);
    flash_next_moe_bf16_gate_up_kernel<<<gate_grid, GateSchedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const std::int32_t*>(workspace.ids.data),
        reinterpret_cast<const __nv_bfloat16*>(weights.expert_gate_up.data),
        gate_up_stride,
        static_cast<const __nv_bfloat16*>(weights.shared_gate.qdata),
        static_cast<const __nv_bfloat16*>(weights.shared_up.qdata),
        static_cast<__nv_bfloat16*>(workspace.activations.data));
    CUDA_CHECK(cudaGetLastError());

    const dim3 down_grid(kHidden / kDownWarps, static_cast<unsigned>(tokens));
    flash_next_moe_bf16_down_kernel<<<down_grid, kDownWarps * 32, 0, stream>>>(
        static_cast<const std::int32_t*>(workspace.ids.data),
        static_cast<const float*>(workspace.alpha.data),
        static_cast<const float*>(workspace.shared_scale.data),
        static_cast<const __nv_bfloat16*>(workspace.activations.data),
        reinterpret_cast<const __nv_bfloat16*>(weights.expert_down.data),
        down_stride,
        static_cast<const __nv_bfloat16*>(weights.shared_down.qdata),
        static_cast<__nv_bfloat16*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}

// ================================================================================================
// Block-5 Stage-2: gguf-block expert leaf support (see flash_next_moe_gguf_experts in moe.cpp).
// ================================================================================================

// Column gather: out[column, i] = x[columns[column], i], [k, columns] both sides. One launch covers
// one expert's chunk (<= 8 columns), issued from the leaf's per-expert loop.
__global__ void flash_next_moe_gguf_gather_kernel(const __nv_bfloat16* __restrict__ x,
                                                  const std::int32_t* __restrict__ columns,
                                                  int k, __nv_bfloat16* __restrict__ out) {
    const int i      = blockIdx.x * blockDim.x + threadIdx.x;
    const int column = blockIdx.y;
    if (i >= k) { return; }
    out[static_cast<std::int64_t>(column) * k + i] =
        x[static_cast<std::int64_t>(columns[column]) * k + i];
}

// The shared expert's activation: h = silu(g) * u, elementwise over the [640, tokens] pair.
// (ops::linear_swiglu has no BF16 route and its GGUF pair form is block-format only, so the fold
// lives here.)
__global__ void flash_next_moe_gguf_silu_mul_kernel(const __nv_bfloat16* __restrict__ g,
                                                    const __nv_bfloat16* __restrict__ u,
                                                    __nv_bfloat16* __restrict__ h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    h[i] = __float2bfloat16_rn(ops::silu(__bfloat162float(g[i])) * __bfloat162float(u[i]));
}

// The 11-path combine: for each token, the ten routed paths (rank order) weighted by alpha, plus
// the sigmoid-gated shared expert. Paths are token-major (path = token * kTopK + rank), and
// `path_src[path]` is the element offset of that path's column inside `routed` -- experts write
// their own contiguous column regions, so path -> column is not the identity. The rank-0..9 order
// matches the nvfp4 reduce's accumulation order.
__global__ void flash_next_moe_gguf_combine_kernel(
    const float* __restrict__ routed, const std::int32_t* __restrict__ path_src,
    const float* __restrict__ alpha, const float* __restrict__ shared_scale,
    const __nv_bfloat16* __restrict__ shared_src, __nv_bfloat16* __restrict__ out) {
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (row >= kHidden) { return; }
    float acc = 0.0f;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
        const int path = token * kTopK + rank;
        acc = fmaf(alpha[path], routed[static_cast<std::int64_t>(path_src[path]) + row], acc);
    }
    acc = fmaf(shared_scale[token],
               __bfloat162float(shared_src[static_cast<std::int64_t>(token) * kHidden + row]), acc);
    out[static_cast<std::int64_t>(token) * kHidden + row] = __float2bfloat16_rn(acc);
}

void flash_next_moe_gguf_gather(const __nv_bfloat16* x, const std::int32_t* columns, int k,
                                int columns_count, __nv_bfloat16* out, cudaStream_t stream) {
    if (k != kHidden || columns_count < 1 || columns_count > 8 || x == nullptr ||
        columns == nullptr || out == nullptr) {
        throw std::invalid_argument("Flash-Next gguf gather: unexpected geometry");
    }
    const dim3 grid((k + 255) / 256, static_cast<unsigned>(columns_count));
    flash_next_moe_gguf_gather_kernel<<<grid, 256, 0, stream>>>(x, columns, k, out);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_moe_gguf_silu_mul(const __nv_bfloat16* g, const __nv_bfloat16* u,
                                  __nv_bfloat16* h, int n, cudaStream_t stream) {
    if (n <= 0 || g == nullptr || u == nullptr || h == nullptr) {
        throw std::invalid_argument("Flash-Next gguf silu_mul: unexpected geometry");
    }
    flash_next_moe_gguf_silu_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(g, u, h, n);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_moe_gguf_combine(const float* routed, const std::int32_t* path_src, int paths,
                                 const float* alpha, const float* shared_scale,
                                 const __nv_bfloat16* shared_src, __nv_bfloat16* out, int tokens,
                                 cudaStream_t stream) {
    if (tokens < 1 || tokens > 8 || paths != tokens * kTopK || routed == nullptr ||
        path_src == nullptr || alpha == nullptr || shared_scale == nullptr ||
        shared_src == nullptr || out == nullptr) {
        throw std::invalid_argument("Flash-Next gguf combine: unexpected geometry");
    }
    const dim3 grid(kHidden / 256, static_cast<unsigned>(tokens));
    flash_next_moe_gguf_combine_kernel<<<grid, 256, 0, stream>>>(routed, path_src, alpha,
                                                                 shared_scale, shared_src, out);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
