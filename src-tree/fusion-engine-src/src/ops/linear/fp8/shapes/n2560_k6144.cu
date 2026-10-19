#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

// Qwen3.8-Flash-Next residual / output projection: 2560 x 6144, 4-byte FP8 row scales
// (artifact format `fp8_e4m3fn_row_f32` -> QType::FP8_E4M3FN_ROW_F32). All 48 text layers carry this
// shape. Schedule shapes are copied from `n5120_k6144.cu` (same K, smaller N); the kernel
// constraints hold - kInputRows 6144 % (32*8) == 0 and % (32*16) == 0, kOutputRows 2560 % kRowsPerCta
// 16 == 0, and every `TokenTile <= Capacity` pair is the sibling's. No k=6144/N=2560 measurement
// exists on this machine - the schedules are the sibling's, not a tuned set.
namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<2560, 6144>;
using Scale     = float;
using Gemv      = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using C2        = Fp8SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4        = Fp8SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C8        = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C12       = Fp8SimtSchedule<8, 2, 16, 12, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using C16       = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using C24       = Fp8SimtSchedule<8, 2, 16, 12, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 2>;
using FullChunk = Fp8SimtSchedule<8, 2, 16, 12, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::TokenTilesContiguous, 1>;
using Full17    = Fp8SimtSchedule<8, 2, 16, 17, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full18    = Fp8SimtSchedule<8, 2, 16, 18, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full19    = Fp8SimtSchedule<8, 2, 16, 19, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using Full20    = Fp8SimtSchedule<8, 2, 8, 20, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, Gemv, Scale>;
    if (tokens == 24) return launch_fp8_simt<Geometry, 24, FullChunk, true, Scale>;
    // These short SIMT extents retain a measured advantage over padded capacities (sibling's note).
    if (tokens == 17) return launch_fp8_simt<Geometry, 17, Full17, true, Scale>;
    if (tokens == 18) return launch_fp8_simt<Geometry, 18, Full18, true, Scale>;
    if (tokens == 19) return launch_fp8_simt<Geometry, 19, Full19, true, Scale>;
    if (tokens == 20) return launch_fp8_simt<Geometry, 20, Full20, true, Scale>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, 2, C2, false, Scale>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, 4, C4, false, Scale>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, 8, C8, false, Scale>;
    if (tokens <= 12) return launch_fp8_simt<Geometry, 12, C12, false, Scale>;
    if (tokens <= 16) return launch_fp8_simt<Geometry, 16, C16, false, Scale>;
    if (tokens <= 24) return launch_fp8_simt<Geometry, 24, C24, false, Scale>;
    throw std::logic_error("fp8 A16 chunk exceeds shape capacity");
}

// A16 only: FP8 A8 is unavailable on sm_89 (see the comment in `n16384_k2560.cu`), and the ported
// family passes AllowA8 directly for tokens >= 16.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N2560K6144{2560, 6144, launch_fp8_a16_chunks<24, select_a16>, nullptr,
                                    uses_a8};
} // namespace ninfer::ops::detail
