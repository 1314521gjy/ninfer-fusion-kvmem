#pragma once

// FP8 row scales come in two word widths in this tree (`src/core/weight.h:61-78`):
//   FP8_E4M3FN_ROW_BF16 (8)  -> 2-byte `__nv_bfloat16` scale word
//   FP8_E4M3FN_ROW_F32  (28) -> 4-byte `float` scale word
// Every Qwen3.5 shape registered before 2026-10-11 uses the 2-byte arm, and the FP8 kernels were
// written against it directly (`__bfloat162float(row_scales[...])`, `static_cast<const
// __nv_bfloat16*>(weight.scales)`). The Qwen3.8-Flash-Next artifact carries its FP8 weights as
// `fp8_e4m3fn_row_f32` (4-byte), so the kernels need both arms.
//
// Upstream solves this the same way: one conversion helper plus a scale-type template parameter
// defaulted to the 2-byte arm (upstream `src/ops/linear/fp8/fp8_gemv.cuh:74-78` and `:102`). This is
// that helper. `Fp8DefaultScale` keeps every existing instantiation compiling unchanged - only a
// shape that declares the 4-byte arm names `float` explicitly.

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

using Fp8DefaultScale = __nv_bfloat16;

__device__ __forceinline__ float fp8_row_scale(float value) { return value; }

__device__ __forceinline__ float fp8_row_scale(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

} // namespace ninfer::ops::detail
