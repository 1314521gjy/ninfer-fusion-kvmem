#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_launch.cuh"

// Qwen3.8-Flash-Next GDN input projection: 16384 x 2560, 4-byte FP8 row scales
// (artifact format `fp8_e4m3fn_row_f32` -> QType::FP8_E4M3FN_ROW_F32). 36 of the 48 text layers
// carry this exact shape; before this file the engine had no geometry below k=5120 and the ported
// family could not even be planned (docs/20 §16.35).
//
// Schedule shapes are copied from the k=5120 sibling `n16384_k5120.cu`: only the K extent changed,
// and both kernel constraints hold for k=2560 as they do for k=5120 - the GEMV/SIMT templates
// require `kInputRows % (32 * kValuesPerLane) == 0` (2560 % 256 == 0, 2560 % 512 == 0) and
// `kOutputRows % kRowsPerCta == 0` (16384 % 16 == 0). No k=2560 measurement exists on this machine,
// so the schedules are the sibling's, not a tuned set: record any retuning as its own reading.
namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<16384, 2560>;
using Scale     = float;
using Gemv      = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using C2        = Fp8SimtSchedule<8, 2, 16, 2, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C4        = Fp8SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C8        = Fp8SimtSchedule<8, 2, 16, 8, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
using C10       = Fp8SimtSchedule<8, 2, 16, 10, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 2>;
using FullChunk = Fp8SimtSchedule<8, 2, 16, 10, 1, Fp8SimtActivationAccess::TokenPacked,
                                  Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;

Fp8Launch select_a16(std::int32_t tokens) {
    if (tokens == 1) return launch_fp8_gemv<Geometry, Gemv, Scale>;
    if (tokens == 10) return launch_fp8_simt<Geometry, 10, FullChunk, true, Scale>;
    if (tokens <= 2) return launch_fp8_simt<Geometry, 2, C2, false, Scale>;
    if (tokens <= 4) return launch_fp8_simt<Geometry, 4, C4, false, Scale>;
    if (tokens <= 8) return launch_fp8_simt<Geometry, 8, C8, false, Scale>;
    if (tokens <= 10) return launch_fp8_simt<Geometry, 10, C10, false, Scale>;
    throw std::logic_error("fp8 A16 chunk exceeds shape capacity");
}

// A16 only, and that is a correctness decision rather than an optimisation: FP8 A8 execution is
// architecturally unavailable in this build (sm_89 - `src/ops/fp8_sm86_stubs.cpp` reject_fp8_a8()
// throws, `fp8_launch.cuh` will not instantiate fp8_mma_kernel there), and the ported family hands
// `LinearPolicy::AllowA8` straight to ops::linear for tokens >= 16 (`impl/qsa_attention.cpp:169`,
// `impl/gdn.cpp:181`), bypassing the frontend's SM8X policy downgrade. Any A8 admission here would
// therefore throw at the first prefill. `affine` to a verbatim copy of the k=5120 sibling, which
// admits A8 from max_tokens >= 11.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8N16384K2560{16384, 2560, launch_fp8_a16_chunks<10, select_a16>, nullptr,
                                     uses_a8};
} // namespace ninfer::ops::detail
