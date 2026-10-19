#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

// Qwen3.8-Flash-Next QSA attention input projection: 13312 x 2560, 4-byte FP8 row scales
// (artifact format `fp8_e4m3fn_row_f32` -> QType::FP8_E4M3FN_ROW_F32). 12 of the 48 text layers
// carry this shape. Schedule shapes are copied from the closest registered sibling
// `n14336_k5120.cu`; the kernel constraints hold for k=2560 (2560 % (32*8) == 0, 2560 % (32*16) == 0)
// and for kOutputRows 13312 with kRowsPerCta 16 (13312 % 16 == 0). No k=2560 measurement exists on
// this machine - the schedules are the sibling's, not a tuned set.
namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<13312, 2560>;
using Scale     = float;
using Gemv      = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using C2        = Fp8SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4        = Fp8SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C8        = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C11       = Fp8SimtSchedule<8, 2, 16, 11, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 2>;
using FullChunk = Fp8SimtSchedule<8, 2, 16, 11, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, Gemv, Scale>;
    if (tokens == 11) return launch_fp8_simt<Geometry, 11, FullChunk, true, Scale>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, 2, C2, false, Scale>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, 4, C4, false, Scale>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, 8, C8, false, Scale>;
    if (tokens <= 11) return launch_fp8_simt<Geometry, 11, C11, false, Scale>;
    throw std::logic_error("fp8 A16 chunk exceeds shape capacity");
}

// A16 only: FP8 A8 is unavailable on sm_89 (see the comment in `n16384_k2560.cu`), and the ported
// family passes AllowA8 directly for tokens >= 16.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N13312K2560{13312, 2560, launch_fp8_a16_chunks<11, select_a16>, nullptr,
                                     uses_a8};
} // namespace ninfer::ops::detail
