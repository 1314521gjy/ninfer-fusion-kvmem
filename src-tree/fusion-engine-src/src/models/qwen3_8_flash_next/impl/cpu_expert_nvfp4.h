#pragma once
//
// CPU-side NVFP4 expert decode -- block-2, phase I of the CPU-expert-kernel project
// (E:\视频\_flashnext\experiments\20261014-speed-campaign\立项-CPU专家核-20261014.md).
//
// WHY THIS FILE EXISTS AT ALL: the project needs a CPU path that reads the SAME expert record the
// device kernels read. The risk that decides whether the whole project is viable is not the matmul --
// it is whether the CPU interprets the record's three faces the way the device does:
//   codes  2 codes per byte, e2m1 (1 sign, 2 exponent, 1 mantissa), row-major byte = row*(K/2)+col/2
//   scales E4M3 raw bytes, one per group of 16 elements along the row, laid out K16M128x4
//          (two independent spellings of the same permutation: nvfp4_codec.cuh:31-38 nvfp4_scale_byte
//           and core/weight_view.cpp:233-236 weight_scale_offset)
//   divisor one positive fp32 per expert per matrix, and everyone DIVIDES by it:
//          value = code x scale / divisor -- moe_kernels.cu:182/292/650/813/1170/1450/1605 all
//          compute inverse = 1.0F / expert_divisors[expert], and decode_tensor
//          (models/qwen3_5/load/vision_cpu.cpp:249) does the same. Nothing stores a reciprocal.
//          (An earlier revision of this file MULTIPLIED by the stored word; that was wrong and is
//          why its probe read abs_max=2.25e6, non-physical. Corrected 2026-10-14, block-2 R3.)
// (contract coordinates: impl/moe_workspace.h:32-35, impl/expert_bank.h:13-41, and the kernel's use of
//  the divisor in impl/moe_kernels.cu.)
//
// WHAT PHASE I VERIFIES, AND WHAT IT DOES NOT: I-a verifies the DECODERS against the FORMAT SPEC's
// literal values (ground truth written out below as constants, taken from the e2m1 and E4M3 layouts),
// not against a second copy of my own code -- a decoder checked against itself is a tautology and can
// never go red. I-b (compare_matrix below) verifies the RECORD walk -- code nibble order, the
// K16M128x4 scale permutation, and the divide-by-divisor contract -- element by element against the
// tree's own independent reader decode_tensor() (models/qwen3_5/load/vision_cpu.cpp:179). It does
// NOT yet verify the device kernels' ARITHMETIC against this file; that is phase II (the int8
// straight-eat matmul). decode_tensor is a REFERENCE ONLY: it must never appear on the hot path.
//
// ASCII ONLY, like every other script/launcher in this line. Host TU only: this header must never be
// included by a .cu (the tree has been bitten by host/device header mixing, and one TU compile is the
// only thing that proves it -- see 立项书 sec-3.2).

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>  // __cpuidex: the one-shot isa self-attestation on the phase-II instr line
#endif

namespace ninfer::models::qwen3_8_flash_next::detail::cpu_expert {

// ------------------------------------------------------------------------------------------------
// e2m1: bits [s e1 e0 m]. Values are 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and their negatives).
// ------------------------------------------------------------------------------------------------
[[nodiscard]] inline float e2m1_decode(std::uint8_t v) noexcept {
    const unsigned sign = (v >> 3) & 0x1u;
    const unsigned exp  = (v >> 1) & 0x3u;
    const unsigned man  = v & 0x1u;
    float magnitude;
    if (exp == 0u) {
        magnitude = (man == 0u) ? 0.0F : 0.5F;         // subnormal: 0 or 0.5
    } else {
        // 2^(exp-1) * (1 + man/2)  ->  1, 1.5 | 2, 3 | 4, 6
        const float base = std::ldexp(1.0F, static_cast<int>(exp) - 1);
        magnitude        = base * (1.0F + 0.5F * static_cast<float>(man));
    }
    return (sign != 0u) ? -magnitude : magnitude;
}

// ------------------------------------------------------------------------------------------------
// E4M3: bits [s e3 e2 e1 e0 m2 m1 m0], bias 7, no infinities, S.1111.111 is NaN.
// ------------------------------------------------------------------------------------------------
[[nodiscard]] inline float e4m3_decode(std::uint8_t v) noexcept {
    const unsigned sign = (v >> 7) & 0x1u;
    const unsigned exp  = (v >> 3) & 0xFu;
    const unsigned man  = v & 0x7u;
    if (exp == 0xFu && man == 0x7u) { return std::numeric_limits<float>::quiet_NaN(); }
    float magnitude;
    if (exp == 0u) {
        magnitude = std::ldexp(static_cast<float>(man), -9);            // subnormal: man/8 * 2^-6
    } else {
        magnitude = std::ldexp(1.0F + static_cast<float>(man) / 8.0F,
                               static_cast<int>(exp) - 7);
    }
    return (sign != 0u) ? -magnitude : magnitude;
}

// One dequantized element: code_value * scale_value / divisor. The divisor is a plain positive fp32
// per expert per matrix and the kernels DIVIDE by it (moe_kernels.cu computes
// inverse = 1.0F / expert_divisors[expert] then multiplies by that inverse; decode_tensor does the
// same at vision_cpu.cpp:249). No reciprocal is stored anywhere.
[[nodiscard]] inline float decode_element(std::uint8_t code, std::uint8_t scale,
                                         float divisor) noexcept {
    return e2m1_decode(code & 0x0Fu) * e4m3_decode(scale) / divisor;
}

// ------------------------------------------------------------------------------------------------
// Phase-I self check. GROUND TRUTH = the two formats' literal values, written here by hand from the
// layout (not computed by the functions above). If a decoder drifts, one of these goes red.
// ------------------------------------------------------------------------------------------------
struct SpecCase {
    std::uint8_t bits;
    float expect;
};

inline const SpecCase* e2m1_spec_cases() {
    static const SpecCase cases[] = {
        {0x0, 0.0F},  {0x1, 0.5F},  {0x2, 1.0F},  {0x3, 1.5F},
        {0x4, 2.0F},  {0x5, 3.0F},  {0x6, 4.0F},  {0x7, 6.0F},
        {0x8, 0.0F},  {0x9, -0.5F}, {0xA, -1.0F}, {0xB, -1.5F},
        {0xC, -2.0F}, {0xD, -3.0F}, {0xE, -4.0F}, {0xF, -6.0F},
    };
    return cases;
}

inline const SpecCase* e4m3_spec_cases() {
    static const SpecCase cases[] = {
        {0x00, 0.0F},      {0x01, 0.001953125F},  // 2^-9, smallest subnormal
        {0x08, 0.015625F},                        // 2^-6
        {0x38, 1.0F},      {0x39, 1.125F},        {0x3C, 1.5F},
        {0x40, 2.0F},      {0x7E, 448.0F},        // largest finite E4M3
        {0x80, 0.0F},      {0xB8, -1.0F},         {0xFE, -448.0F},
    };
    return cases;
}

struct CheckResult {
    std::uint32_t elements   = 0;
    std::uint32_t mismatches = 0;
    std::uint8_t first_bits  = 0;
    float first_expect       = 0.0F;
    float first_got          = 0.0F;
    bool ok() const noexcept { return mismatches == 0 && elements > 0; }
};

// `sabotage` deliberately corrupts one expected value, so the judgement can be SEEN to fail
// (expert_staging.h:35-37: a judgement nobody has seen fail is not a judgement).
[[nodiscard]] inline CheckResult self_check(bool sabotage) noexcept {
    CheckResult r{};
    const SpecCase* e2 = e2m1_spec_cases();
    for (std::size_t i = 0; i < 16; ++i) {
        ++r.elements;
        float expect = e2[i].expect;
        if (sabotage && i == 3) { expect += 1.0F; }   // one wrong expectation, on purpose
        const float got = e2m1_decode(e2[i].bits);
        if (got != expect) {
            if (r.mismatches == 0) { r.first_bits = e2[i].bits; r.first_expect = expect; r.first_got = got; }
            ++r.mismatches;
        }
    }
    const SpecCase* e4 = e4m3_spec_cases();
    for (std::size_t i = 0; i < 11; ++i) {
        ++r.elements;
        float expect = e4[i].expect;
        if (sabotage && i == 4) { expect += 0.25F; }
        const float got = e4m3_decode(e4[i].bits);
        if (got != expect) {
            if (r.mismatches == 0) { r.first_bits = e4[i].bits; r.first_expect = expect; r.first_got = got; }
            ++r.mismatches;
        }
    }
    return r;
}

// ---- Phase I-b: walk a real record and judge it against the tree's independent reference ----------
//
// The reference is decode_tensor() (models/qwen3_5/load/vision_cpu.cpp:179), the tree's "any stored
// representation -> FP32" reader. Comparing against it settles three questions at once: nibble
// order, the K16M128x4 scale permutation, and the divide-by-divisor contract. The scale index below
// is a HOST transcription of the DEVICE spelling (nvfp4_codec.cuh:31-38 nvfp4_scale_byte, with
// ScaleTilesPerRow = columns/64) rather than a call into the shared helper -- a transcription error
// in THIS file is exactly what the comparison exists to catch.
//
// `sabotage` deliberately falls back to the flat linear scale index scales[(row*K+col)/16] -- the
// exact bug phase I-b was born from (the first probe read abs_max=2.25e6 because of it) -- so the
// judgement must be SEEN to fail (expert_staging.h:35-37: a judgement nobody has seen fail is not a
// judgement).

[[nodiscard]] inline std::uint64_t k16m128x4_scale_byte(std::uint64_t row, std::uint64_t group,
                                                        std::uint64_t scale_tiles_per_row) noexcept {
    const std::uint64_t m_tile    = row / 128;
    const std::uint64_t row_inner = row - m_tile * 128;
    const std::uint64_t scale_tile = group / 4;
    const std::uint64_t scale_lane = group & 3;
    return (m_tile * scale_tiles_per_row + scale_tile) * 512 + (row_inner & 31) * 16 +
           (row_inner >> 5) * 4 + scale_lane;
}

[[nodiscard]] inline float decode_element_at(const std::uint8_t* codes, const std::uint8_t* scales,
                                            std::uint64_t columns, float divisor, std::uint64_t row,
                                            std::uint64_t col, bool sabotage) noexcept {
    const std::uint64_t group = col / 16;
    const std::uint64_t scale_index =
        sabotage ? ((row * columns + col) / 16) : k16m128x4_scale_byte(row, group, columns / 64);
    const std::uint8_t byte = codes[row * (columns / 2) + col / 2];
    const std::uint8_t code = ((col & 1u) == 0u) ? (byte & 0x0Fu) : (byte >> 4);
    return decode_element(code, scales[scale_index], divisor);
}

struct MatrixCompare {
    std::uint64_t elements     = 0;
    double        cos          = 0.0;
    double        rel_mae      = 0.0;
    std::uint64_t mismatches   = 0;
    double        max_abs_diff = 0.0;
    float         first_expect = 0.0F;
    float         first_got    = 0.0F;
    [[nodiscard]] bool ok(double cos_gate, double rel_mae_gate) const noexcept {
        return elements > 0 && cos > cos_gate && rel_mae < rel_mae_gate;
    }
};

// `reference` is decode_tensor's row-major FP32 output for the same matrix (rows*columns floats).
// Gates live at the call site so this file never hardcodes a verdict policy: cos>0.99 and
// rel_MAE<0.04 (K1 of the shared caliber) are what the campaign judges with.
[[nodiscard]] inline MatrixCompare
compare_matrix(const std::uint8_t* codes, const std::uint8_t* scales, std::uint64_t rows,
               std::uint64_t columns, float divisor, bool sabotage, const float* reference) noexcept {
    MatrixCompare r{};
    if (codes == nullptr || scales == nullptr || reference == nullptr || rows == 0 || columns == 0 ||
        !(divisor > 0.0F)) {
        return r;
    }
    double dot = 0.0, norm_mine = 0.0, norm_ref = 0.0, abs_diff = 0.0, abs_ref = 0.0;
    for (std::uint64_t row = 0; row < rows; ++row) {
        for (std::uint64_t col = 0; col < columns; ++col) {
            const float got = decode_element_at(codes, scales, columns, divisor, row, col, sabotage);
            const float ref = reference[row * columns + col];
            const double g  = static_cast<double>(got);
            const double f  = static_cast<double>(ref);
            dot += g * f;
            norm_mine += g * g;
            norm_ref += f * f;
            const double d = std::fabs(g - f);
            abs_diff += d;
            abs_ref += std::fabs(f);
            if (d > r.max_abs_diff) { r.max_abs_diff = d; }
            if (d > 1.0e-3 * (std::fabs(f) + 1.0e-6)) {
                if (r.mismatches == 0) { r.first_expect = ref; r.first_got = got; }
                ++r.mismatches;
            }
            ++r.elements;
        }
    }
    if (norm_mine > 0.0 && norm_ref > 0.0) { r.cos = dot / std::sqrt(norm_mine * norm_ref); }
    if (abs_ref > 0.0) { r.rel_mae = abs_diff / abs_ref; }
    return r;
}

// ================================================================================================
// Phase II -- one-expert CPU W4A8 forward, integer straight-eat (block-3).
//
// WHAT THIS IS: the phase-II kernel plus its diagnostic harness, for the CPU expert core. The
// kernel consumes the same expert record the device kernels read (codes / K16M128x4 scales /
// per-expert divisor), quantizes activations to u8 with zero point 128, and accumulates through
// VEX vpdpbusd (first version: 128-bit, one instruction per 16-k group; the probe in
// port-probe\_isa-probe.* proves no /arch is needed and zmm stays at 0). No float dequantizer
// appears in the kernel section: weights stay in the integer domain end to end (CPU-expert brief
// v2 sec-3.4).
// The whole section runs once per process inside the CPU_EXPERT_CHECK guard; nothing here is
// reachable from the hot path, and the default (check off) behaviour is byte-identical.
//
// CONTRACT (frozen with QA v3; fields and verdict rules are re-judged independently by the
// block-3 harness from the raw log lines):
//   q9   = e4m3 x 512        (exact integer for every finite e4m3, subnormals included)
//   w_i8 = 2 x e2m1          (exact int8 lanes, {-12..12})
//   P_g  = sum_{16} a_u8 * w_i8        S_g = sum_{16} w_i8
//   C_r  = 128 * sum_g q9_g * S_g      (per-row constant, precomputed; input independent)
//   out_fixed[r] = sum_g q9_g * P_g - C_r     (int64; |out_fixed| <= ~1.8e12)
//   value        = out_fixed * 2^-10 * act_scale / divisor     (fixed_shift = 10)
//
// REFERENCES (per-stage caliber frozen with QA; all diagnostic-only):
//   gate_up : reference A = W_gu (decode_tensor fp32) x raw unquantized x, double accumulate.
//   down    : reference A = W_dn x act of the exact chain (fp32 silu + contract bf16 round, NO u8
//             quantization of the act) -- the kernel's own u8 act must never be the reference here
//             (hybrid reference would erase the act quantization loss from the down floor).
//   whole   : reference A = the entire unquantized fp32 chain (raw x, fp32 act face, no u8 at all).
//   ideal   = the same numbers recomputed exactly from the kernel's u8 codes (oracle integers,
//             same-domain fp32 conversion) -> floor = rel_MAE(ideal, reference A) per stage,
//             rel_mae_vs_ideal = rel_MAE(kernel, ideal), gate <= 1e-4.
//
// SABOTAGE (red arms; env NINFER_FLASH_NEXT_CPU_EXPERT_FWD_SABOTAGE=<int>, read once in the .cpp):
//   mode 1 = G(1): skip the zero-point compensation C_r
//   mode 2 = G(2): flat linear scale index scales[(row*cols+col)/16] (the phase-I-b bug form)
//   Modes 3..5 are QA-list reserves, not implemented in this round (they run as no sabotage).
// ================================================================================================

inline constexpr int kFixedShift = 10;

// e2m1 code -> 2 x value as int8: {0,0.5,1,1.5,2,3,4,6} x 2 = {0,1,2,3,4,6,8,12}. The x2 keeps
// every e2m1 value an exact integer (CPU-expert brief v2 sec-3.4 item 4); the sign bit negates the
// magnitude.
alignas(16) inline constexpr std::int8_t kE2m1X2Lut[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ---- kernel section (integer only; no float dequantizers on this path) --------------------------

// 8 packed code bytes -> 16 int8 lanes in k order (low nibble of byte j is k = 2j).
[[nodiscard]] inline __m128i expand_codes_i8(const std::uint8_t* codes8) noexcept {
    const __m128i lut = _mm_load_si128(reinterpret_cast<const __m128i*>(kE2m1X2Lut));
    const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes8));
    const __m128i lo  = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
    const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
    return _mm_unpacklo_epi8(_mm_shuffle_epi8(lut, lo), _mm_shuffle_epi8(lut, hi));
}

// One 16-k group: exact sum of a_u8 x w_i8 (int32; |P_g| <= 48960).
[[nodiscard]] inline std::int32_t dot_group_i32(const std::uint8_t* act16,
                                                const std::uint8_t* codes8) noexcept {
    const __m128i w = expand_codes_i8(codes8);
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(act16));
    const __m128i p = _mm_dpbusd_avx_epi32(_mm_setzero_si128(), a, w);
    const __m128i s = _mm_hadd_epi32(_mm_hadd_epi32(p, p), _mm_hadd_epi32(p, p));
    return _mm_cvtsi128_si32(s);
}

// E4M3 byte -> q9 = value x 512, exact for every finite e4m3 (subnormals included). The only
// rejected pattern is S.1111.111 (NaN); callers turn that into a failed instrument, never a value.
[[nodiscard]] inline bool e4m3_to_q9(std::uint8_t v, std::int32_t& q9_out) noexcept {
    const unsigned sign = (v >> 7) & 0x1U;
    const unsigned exp  = (v >> 3) & 0xFU;
    const unsigned man  = v & 0x7U;
    if (exp == 0xFU && man == 0x7U) { return false; }
    const std::uint32_t magnitude =
        (exp == 0U) ? man : (static_cast<std::uint32_t>(8U + man) << (exp - 1U));
    q9_out = (sign != 0U) ? -static_cast<std::int32_t>(magnitude)
                          : static_cast<std::int32_t>(magnitude);
    return true;
}

// Scale byte of (row, group); sabotage mode 2 swaps in the flat linear index on purpose.
[[nodiscard]] inline std::uint64_t phase2_scale_index(std::uint64_t row, std::uint64_t group,
                                                      std::uint64_t columns,
                                                      std::int32_t sab_mode) noexcept {
    if (sab_mode == 2) { return row * (columns / 16) + group; }
    return k16m128x4_scale_byte(row, group, columns / 64);
}

// ---- W0-b(B): SIMD row-sum + q9 table (16-k group math, integer-identical to the scalar form) ---
//
// S_g = sum of the 16 w_i8 lanes of one group: expand the 8 code bytes with the same pshufb LUT
// expand_codes_i8 uses, then _mm_maddubs_epi16(u8 ones, w) -> _mm_madd_epi16(., 1) -> two hadds.
// |w_i8| <= 12, so every partial sum is far inside int16/int32 and |S_g| <= 192 exactly. Integer
// sums do not depend on the order of the addends, so this is bit-identical to the per-byte loop
// it replaces -- the frozen contract (:261-268) and the int64 accumulation below are untouched.
// 128-bit intrinsics only (SSSE3/SSE2 class; no AVX-512 emission path in this TU).
[[nodiscard]] inline std::int32_t row_sum_i32_simd(const std::uint8_t* codes8) noexcept {
    const __m128i w     = expand_codes_i8(codes8);
    const __m128i pairs = _mm_maddubs_epi16(_mm_set1_epi8(1), w);   // 8 x i16: w[2i]+w[2i+1]
    const __m128i quads = _mm_madd_epi16(pairs, _mm_set1_epi16(1)); // 4 x i32: |each| <= 48
    const __m128i half  = _mm_hadd_epi32(quads, quads);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(half, half));           // |S_g| <= 192
}

// 256-entry q9 table, filled FROM e4m3_to_q9 (single source of truth -- a second hand-written
// decoder would be the same bug written twice). The one rejected pattern, S.1111.111, is stored
// as kQ9NonFinite, so callers keep the "failed instrument, never a value" semantic.
inline constexpr std::int32_t kQ9NonFinite = std::numeric_limits<std::int32_t>::min();

[[nodiscard]] inline const std::int32_t* q9_lut256() noexcept {
    static const std::array<std::int32_t, 256> table = [] {
        std::array<std::int32_t, 256> out{};
        for (std::size_t b = 0; b < out.size(); ++b) {
            std::int32_t q9 = 0;
            out[b] = e4m3_to_q9(static_cast<std::uint8_t>(b), q9) ? q9 : kQ9NonFinite;
        }
        return out;
    }();
    return table.data();
}

// Per-row zero-point compensation C_r = 128 * sum_g q9_g * S_g (input independent).
// Phase IV W0-b(B): the 16-lane sum is SIMD (row_sum_i32_simd) and q9 comes from the table; both
// are integer-exact re-spellings of the frozen scalar math, so every bias value below is
// bit-identical to the pre-W0-b implementation. Contract block (:261-268) stays word for word.
[[nodiscard]] inline bool precompute_row_bias(const std::uint8_t* codes, const std::uint8_t* scales,
                                              std::uint64_t rows, std::uint64_t columns,
                                              std::int32_t sab_mode, std::vector<std::int64_t>& bias,
                                              bool& q9_finite) {
    const std::uint64_t groups = columns / 16;
    bias.assign(static_cast<std::size_t>(rows), 0);
    const std::int32_t* const q9_table = q9_lut256();
    for (std::uint64_t row = 0; row < rows; ++row) {
        const std::uint8_t* row_codes = codes + row * (columns / 2);
        std::int64_t c               = 0;
        for (std::uint64_t g = 0; g < groups; ++g) {
            const std::int32_t q9 = q9_table[scales[phase2_scale_index(row, g, columns, sab_mode)]];
            if (q9 == kQ9NonFinite) {
                q9_finite = false;
                return false;
            }
            c += static_cast<std::int64_t>(q9) * row_sum_i32_simd(row_codes + g * 8);
        }
        bias[static_cast<std::size_t>(row)] = 128 * c;
    }
    return true;
}

// (rows x columns) . (u8 activation vector) -> out_fixed[rows] (int64), integer only.
// Sabotage mode 1 drops the compensation; mode 2 also corrupts the scale index upstream in
// precompute_row_bias, so the whole flat-index bug is coherent across bias and accumulation.
[[nodiscard]] inline bool gemv_w4a8(const std::uint8_t* codes, const std::uint8_t* scales,
                                    std::uint64_t rows, std::uint64_t columns, std::int32_t sab_mode,
                                    const std::uint8_t* act_u8,
                                    const std::vector<std::int64_t>& bias,
                                    std::vector<std::int64_t>& out_fixed) {
    const std::uint64_t groups = columns / 16;
    out_fixed.assign(static_cast<std::size_t>(rows), 0);
    for (std::uint64_t row = 0; row < rows; ++row) {
        const std::uint8_t* row_codes = codes + row * (columns / 2);
        std::int64_t acc              = 0;
        for (std::uint64_t g = 0; g < groups; ++g) {
            std::int32_t q9 = 0;
            if (!e4m3_to_q9(scales[phase2_scale_index(row, g, columns, sab_mode)], q9)) {
                return false;
            }
            const std::int32_t p = dot_group_i32(act_u8 + g * 16, row_codes + g * 8);
            acc += static_cast<std::int64_t>(q9) * p;
        }
        out_fixed[static_cast<std::size_t>(row)] =
            (sab_mode == 1) ? acc : (acc - bias[static_cast<std::size_t>(row)]);
    }
    return true;
}

// The single float scale of the frozen equation: value = out_fixed * 2^-10 * act_scale / divisor.
[[nodiscard]] inline float dequant_scale_f32(float act_scale, float divisor) noexcept {
    return act_scale / (1024.0F * divisor);
}

// One rounding (int64 -> float32) plus one multiply; kernel and ideal share this exact expression.
[[nodiscard]] inline float fixed_to_float(std::int64_t out_fixed, float scale) noexcept {
    return static_cast<float>(out_fixed) * scale;
}

// u8 zero-point-128 quantization, per-vector symmetric absmax. Frozen rule (QA v3):
//   scale = absmax / 127;  q = clamp(lround(x / scale) + 128, 1, 255);  absmax == 0 -> q == 128.
struct QuantizedAct {
    std::vector<std::uint8_t> codes;
    float                     scale  = 1.0F;
    float                     absmax = 0.0F;
};

[[nodiscard]] inline QuantizedAct quantize_u8_zp128(const float* x, std::size_t n) {
    QuantizedAct out;
    out.codes.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        if (a > out.absmax) { out.absmax = a; }
    }
    if (out.absmax > 0.0F) { out.scale = out.absmax / 127.0F; }
    for (std::size_t i = 0; i < n; ++i) {
        long q = std::lround(x[i] / out.scale) + 128L;
        if (q < 1L) { q = 1L; }
        if (q > 255L) { q = 255L; }
        out.codes[i] = static_cast<std::uint8_t>(q);
    }
    return out;
}

// RNE to bfloat16 via the classic bit trick; mirrors __float2bfloat16_rn on normal values (the
// |v| < 2^-126 corner is a known boundary reported on, never silently accepted).
[[nodiscard]] inline float round_to_bf16(float v) noexcept {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    u += 0x7FFFU + ((u >> 16U) & 1U);
    u &= 0xFFFF0000U;
    float r = 0.0F;
    std::memcpy(&r, &u, sizeof(r));
    return r;
}

[[nodiscard]] inline std::uint16_t bf16_bits(float v) noexcept {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return static_cast<std::uint16_t>(u >> 16U);
}

// Same spelling as ops/common/math.cuh:38 (the compiled branch: NINFER_SFU_SIGMOID_SILU is OFF),
// transcribed to the host with std::expf. Kernel and oracle share this exact expression.
[[nodiscard]] inline float silu_f32(float x) noexcept { return x / (1.0F + std::expf(-x)); }

// ---- diagnostic section: oracles, references, metrics, probe runner (never on the hot path) -----

struct IsaFlags {
    int avx2     = -1;  // -1 = host build without __cpuidex (non-MSVC); 0/1 otherwise
    int avx_vnni = -1;
};

// Runtime self-attestation for the instr line: true CPUID reads, not build-time constants.
[[nodiscard]] inline IsaFlags isa_flags() noexcept {
#if defined(_MSC_VER)
    int leaf7[4]  = {0, 0, 0, 0};
    int leaf7s1[4] = {0, 0, 0, 0};
    __cpuidex(leaf7, 7, 0);
    __cpuidex(leaf7s1, 7, 1);
    IsaFlags f;
    f.avx2     = (leaf7[1] >> 5) & 1;     // CPUID.(EAX=7,ECX=0):EBX.AVX2
    f.avx_vnni = (leaf7s1[0] >> 4) & 1;   // CPUID.(EAX=7,ECX=1):EAX.AVX-VNNI
    return f;
#else
    return IsaFlags{};
#endif
}

// SIMD expansion vs the scalar spec table: catches lane order / mask / nibble-order bugs.
[[nodiscard]] inline bool expand_scalar_match_check() noexcept {
    const std::uint8_t pat[8] = {0x01, 0xA5, 0xF0, 0x3C, 0x77, 0x88, 0xDE, 0x09};
    alignas(16) std::int8_t got[16] = {};
    _mm_store_si128(reinterpret_cast<__m128i*>(got), expand_codes_i8(pat));
    for (int i = 0; i < 8; ++i) {
        const int lo_exp = kE2m1X2Lut[pat[i] & 0x0FU];
        const int hi_exp = kE2m1X2Lut[pat[i] >> 4U];
        if (got[2 * i] != lo_exp || got[2 * i + 1] != hi_exp) { return false; }
    }
    return true;
}

// Scalar transcription of the frozen fixed-point equation (correct form, no sabotage): the
// independent oracle for the integer face. Different code path, same formula.
[[nodiscard]] inline std::int64_t oracle_dot_row(const std::uint8_t* row_codes,
                                                 const std::uint8_t* scales, std::uint64_t columns,
                                                 std::uint64_t row, const std::uint8_t* act_u8) {
    const std::uint64_t groups = columns / 16;
    std::int64_t acc           = 0;
    std::int64_t c_r           = 0;
    for (std::uint64_t g = 0; g < groups; ++g) {
        std::int32_t q9 = 0;
        if (!e4m3_to_q9(scales[k16m128x4_scale_byte(row, g, columns / 64)], q9)) { return 0; }
        std::int64_t p = 0;
        std::int32_t s = 0;
        for (int i = 0; i < 8; ++i) {
            const std::uint8_t b = row_codes[g * 8 + i];
            const int wl         = kE2m1X2Lut[b & 0x0FU];
            const int wh         = kE2m1X2Lut[b >> 4U];
            p += static_cast<std::int64_t>(wl) * act_u8[g * 16 + 2 * i];
            p += static_cast<std::int64_t>(wh) * act_u8[g * 16 + 2 * i + 1];
            s += wl + wh;
        }
        acc += static_cast<std::int64_t>(q9) * p;
        c_r += static_cast<std::int64_t>(q9) * s;
    }
    return acc - 128 * c_r;
}

// Scalar u8 quantizer, independent transcription of the same frozen rule.
[[nodiscard]] inline std::vector<std::uint8_t> oracle_quantize_u8(const float* x, std::size_t n,
                                                                  float& scale_out,
                                                                  float& absmax_out) {
    float amax = 0.0F;
    for (std::size_t i = 0; i < n; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax) { amax = a; }
    }
    const float scale = (amax > 0.0F) ? (amax / 127.0F) : 1.0F;
    std::vector<std::uint8_t> q(n);
    for (std::size_t i = 0; i < n; ++i) {
        long v = std::lround(x[i] / scale) + 128L;
        if (v < 1L) { v = 1L; }
        if (v > 255L) { v = 255L; }
        q[i] = static_cast<std::uint8_t>(v);
    }
    scale_out  = scale;
    absmax_out = amax;
    return q;
}

// Reference A: fp32 weights (decode_tensor output) times a DOUBLE activation vector, double
// accumulation. The activation vector decides the stage caliber (window comment at the top).
inline void ref_gemm_f32w_d64x(const float* w, std::uint64_t rows, std::uint64_t cols,
                               const double* x, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(rows), 0.0);
    for (std::uint64_t r = 0; r < rows; ++r) {
        const float* wr = w + r * cols;
        double acc      = 0.0;
        for (std::uint64_t k = 0; k < cols; ++k) {
            acc += static_cast<double>(wr[k]) * x[k];
        }
        out[static_cast<std::size_t>(r)] = acc;
    }
}

// The same spelling as silu_f32, in double: reference A's own chain may use it (QA v3 sec-3.3).
[[nodiscard]] inline double silu_d64(double x) noexcept { return x / (1.0 + std::exp(-x)); }

struct VecCompare {
    std::uint64_t elements = 0;
    double        cos      = 0.0;
    double        rel_mae  = 0.0;
};

[[nodiscard]] inline VecCompare compare_vec(const double* got, const double* ref,
                                            std::size_t n) noexcept {
    VecCompare r;
    double dot = 0.0, ng = 0.0, nr = 0.0, abs_diff = 0.0, abs_ref = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double g = got[i];
        const double f = ref[i];
        dot += g * f;
        ng += g * g;
        nr += f * f;
        abs_diff += std::fabs(g - f);
        abs_ref += std::fabs(f);
        ++r.elements;
    }
    if (ng > 0.0 && nr > 0.0) { r.cos = dot / std::sqrt(ng * nr); }
    if (abs_ref > 0.0) { r.rel_mae = abs_diff / abs_ref; }
    return r;
}

[[nodiscard]] inline std::vector<double> widen_f32(const std::vector<float>& v) {
    return std::vector<double>(v.begin(), v.end());
}

[[nodiscard]] inline std::uint64_t compare_bf16_bits(const float* got, const float* ref,
                                                     std::size_t n) noexcept {
    std::uint64_t m = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (bf16_bits(got[i]) != bf16_bits(ref[i])) { ++m; }
    }
    return m;
}

[[nodiscard]] inline std::uint64_t compare_fixed_bits(const std::vector<std::int64_t>& got,
                                                      const std::vector<std::int64_t>& ref,
                                                      std::int64_t& max_abs_diff) noexcept {
    max_abs_diff       = 0;
    std::uint64_t mism = 0;
    if (got.size() != ref.size()) { return got.size() + ref.size() + 1; }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const std::int64_t d = got[i] - ref[i];
        const std::int64_t a = (d < 0) ? -d : d;
        if (a > max_abs_diff) { max_abs_diff = a; }
        if (d != 0) { ++mism; }
    }
    return mism;
}

[[nodiscard]] inline std::uint64_t count_diff_u8(const std::vector<std::uint8_t>& a,
                                                 const std::vector<std::uint8_t>& b) noexcept {
    if (a.size() != b.size()) { return a.size() + b.size() + 1; }
    std::uint64_t m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) { ++m; }
    }
    return m;
}

// P1: deterministic xorshift32, uniform [-1, 1); fixed seed, prompt independent by construction.
inline void fill_probe_p1(std::vector<float>& x, bool& ok) {
    std::uint32_t s = 0x9E3779B9U;
    float lo = 2.0F, hi = -2.0F;
    for (auto& v : x) {
        s ^= s << 13U;
        s ^= s >> 17U;
        s ^= s << 5U;
        v = static_cast<float>(s) / 2147483648.0F - 1.0F;
        if (v < lo) { lo = v; }
        if (v > hi) { hi = v; }
    }
    ok = (lo < -0.9F) && (hi > 0.9F);
}

// P2: integer code grid c in [-127, 127] with the extremes pinned -> lossless under the frozen
// quantizer (absmax 127 -> scale 1 -> q = c + 128 exactly). Only the x face is lossless; the
// act/down/whole stages keep their measured quantization floor (QA v3 says so explicitly).
inline void fill_probe_p2(std::vector<float>& x, bool& ok) {
    std::uint32_t s = 0x85EBCA6BU;
    for (auto& v : x) {
        s ^= s << 13U;
        s ^= s >> 17U;
        s ^= s << 5U;
        const int c = static_cast<int>(s % 255U) - 127;
        v = static_cast<float>(c);
    }
    if (x.size() >= 2) {
        x[0] = -127.0F;
        x[1] = 127.0F;
    }
    ok = (x.size() >= 2) && (x[0] == -127.0F) && (x[1] == 127.0F);
}

struct Phase2Matrix {
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    std::uint64_t       rows;
    std::uint64_t       columns;
    float               divisor;
    std::uint64_t       code_bytes;
    std::uint64_t       scale_bytes;
};

[[nodiscard]] inline bool phase2_geometry_ok(const Phase2Matrix& m) noexcept {
    return m.codes != nullptr && m.scales != nullptr && m.divisor > 0.0F && m.rows > 0 &&
           m.columns > 0 && (m.code_bytes == m.rows * m.columns / 2) &&
           (m.scale_bytes == m.rows * m.columns / 16) && (m.rows % 128 == 0) &&
           (m.columns % 64 == 0);
}

struct Phase2FaceMetrics {
    std::uint64_t elements           = 0;
    std::uint64_t codes_mismatches   = 0;
    std::uint64_t int_mismatches     = 0;
    std::int64_t  max_abs_diff_fixed = 0;
    double        cos                = 0.0;
    double        rel_mae            = 0.0;
    double        floor_v            = 0.0;
    double        rel_mae_vs_ideal   = 0.0;
    bool          computed           = false;
};

struct Phase2ProbeResult {
    Phase2FaceMetrics gu;
    std::uint64_t     act_elements        = 0;
    std::uint64_t     act_bf16_mismatches = 0;
    bool              act_computed        = false;
    Phase2FaceMetrics dn;
    Phase2FaceMetrics whole;
};

struct Phase2InstrState {
    bool          expand_ok          = false;
    bool          q9_finite          = true;
    bool          quant_lossless_p2  = false;
    bool          zero_x_ok          = false;
    bool          probe_p1_ok        = false;
    bool          probe_p2_ok        = false;
    std::uint64_t oracle_elements_gu = 0;
    std::uint64_t oracle_elements_dn = 0;
};

struct Phase2RunResult {
    Phase2ProbeResult probes[2];  // [0] = P1, [1] = P2
    Phase2InstrState  instr;
    bool              ran = false;  // false = geometry / scale preconditions failed
};

// The whole phase-II run, once per process: two probes (P1, P2) x four stages, integer faces
// judged against the scalar oracle, floats against the per-stage reference A, plus the instr
// flags. The .cpp prints one line per stage per probe from the metrics below.
[[nodiscard]] inline Phase2RunResult run_phase2(const Phase2Matrix& gu, const Phase2Matrix& dn,
                                                const float* w_gu, const float* w_dn,
                                                std::int32_t sab_mode) {
    Phase2RunResult out;
    if (!phase2_geometry_ok(gu) || !phase2_geometry_ok(dn) || w_gu == nullptr || w_dn == nullptr) {
        return out;
    }
    out.ran                      = true;
    out.instr.oracle_elements_gu = gu.rows;
    out.instr.oracle_elements_dn = dn.rows;
    out.instr.expand_ok          = expand_scalar_match_check();
    out.instr.q9_finite          = true;

    std::vector<std::int64_t> bias_gu;
    std::vector<std::int64_t> bias_dn;
    if (!precompute_row_bias(gu.codes, gu.scales, gu.rows, gu.columns, sab_mode, bias_gu,
                             out.instr.q9_finite) ||
        !precompute_row_bias(dn.codes, dn.scales, dn.rows, dn.columns, sab_mode, bias_dn,
                             out.instr.q9_finite)) {
        out.ran = false;
        return out;
    }
    const std::uint64_t half = gu.rows / 2;

    for (int pi = 0; pi < 2; ++pi) {
        bool probe_ok = false;
        std::vector<float> x(static_cast<std::size_t>(gu.columns), 0.0F);
        if (pi == 0) {
            fill_probe_p1(x, probe_ok);
            out.instr.probe_p1_ok = probe_ok;
        } else {
            fill_probe_p2(x, probe_ok);
            out.instr.probe_p2_ok = probe_ok;
        }
        Phase2ProbeResult& pr = out.probes[pi];

        // ---- the kernel pipeline (the thing under test) ----
        const QuantizedAct qx = quantize_u8_zp128(x.data(), x.size());
        if (pi == 1) {
            bool lossless = (qx.absmax == 127.0F);
            for (std::size_t i = 0; i < x.size() && lossless; ++i) {
                if (static_cast<long>(qx.codes[i]) != std::lround(static_cast<double>(x[i])) + 128L) {
                    lossless = false;
                }
            }
            out.instr.quant_lossless_p2 = lossless;
        }
        std::vector<std::int64_t> fx_gu;
        std::vector<std::int64_t> fx_dn;
        if (!gemv_w4a8(gu.codes, gu.scales, gu.rows, gu.columns, sab_mode, qx.codes.data(), bias_gu,
                       fx_gu)) {
            continue;
        }
        const float dq_gu = dequant_scale_f32(qx.scale, gu.divisor);
        std::vector<float> y_gu(fx_gu.size());
        for (std::size_t i = 0; i < y_gu.size(); ++i) {
            y_gu[i] = fixed_to_float(fx_gu[i], dq_gu);
        }
        std::vector<float> act(static_cast<std::size_t>(half));
        for (std::uint64_t r = 0; r < half; ++r) {
            act[static_cast<std::size_t>(r)] =
                round_to_bf16(silu_f32(y_gu[static_cast<std::size_t>(r)]) *
                              y_gu[static_cast<std::size_t>(r + half)]);
        }
        const QuantizedAct qa = quantize_u8_zp128(act.data(), act.size());
        if (!gemv_w4a8(dn.codes, dn.scales, dn.rows, dn.columns, sab_mode, qa.codes.data(), bias_dn,
                       fx_dn)) {
            continue;
        }
        const float dq_dn = dequant_scale_f32(qa.scale, dn.divisor);
        std::vector<float> y_dn(fx_dn.size());
        for (std::size_t i = 0; i < y_dn.size(); ++i) {
            y_dn[i] = fixed_to_float(fx_dn[i], dq_dn);
        }
        std::vector<float> whole(y_dn.size());
        for (std::size_t i = 0; i < whole.size(); ++i) {
            whole[i] = round_to_bf16(y_dn[i]);
        }

        // ---- oracle integer faces: exact recompute from the kernel's u8 codes ----
        std::vector<std::int64_t> ox_gu(static_cast<std::size_t>(gu.rows));
        for (std::uint64_t r = 0; r < gu.rows; ++r) {
            ox_gu[static_cast<std::size_t>(r)] =
                oracle_dot_row(gu.codes + r * (gu.columns / 2), gu.scales, gu.columns, r,
                               qx.codes.data());
        }
        std::vector<std::int64_t> ox_dn(static_cast<std::size_t>(dn.rows));
        for (std::uint64_t r = 0; r < dn.rows; ++r) {
            ox_dn[static_cast<std::size_t>(r)] =
                oracle_dot_row(dn.codes + r * (dn.columns / 2), dn.scales, dn.columns, r,
                               qa.codes.data());
        }
        // ---- ideal chain: same-domain fp32 floats fed by the oracle integers ----
        std::vector<float> y_gu_id(y_gu.size());
        for (std::size_t i = 0; i < y_gu_id.size(); ++i) {
            y_gu_id[i] = fixed_to_float(ox_gu[i], dq_gu);
        }
        std::vector<float> act_id(act.size());
        for (std::uint64_t r = 0; r < half; ++r) {
            act_id[static_cast<std::size_t>(r)] =
                round_to_bf16(silu_f32(y_gu_id[static_cast<std::size_t>(r)]) *
                              y_gu_id[static_cast<std::size_t>(r + half)]);
        }
        std::vector<float> y_dn_id(y_dn.size());
        for (std::size_t i = 0; i < y_dn_id.size(); ++i) {
            y_dn_id[i] = fixed_to_float(ox_dn[i], dq_dn);
        }
        std::vector<float> whole_id(y_dn_id.size());
        for (std::size_t i = 0; i < whole_id.size(); ++i) {
            whole_id[i] = round_to_bf16(y_dn_id[i]);
        }

        // ---- reference A, per-stage caliber (frozen with QA) ----
        // gate_up: W_gu x raw unquantized x, double accumulate.
        const std::vector<double> x_d(widen_f32(x));
        std::vector<double> gu_a;
        ref_gemm_f32w_d64x(w_gu, gu.rows, gu.columns, x_d.data(), gu_a);
        // down: W_dn x act of the exact chain (fp32 silu + bf16 round, NO u8 of the act).
        const std::vector<double> act_id_d(widen_f32(act_id));
        std::vector<double> dn_exact_a;
        ref_gemm_f32w_d64x(w_dn, dn.rows, dn.columns, act_id_d.data(), dn_exact_a);
        // whole: the entire unquantized fp32 chain (act face in fp32 from the raw chain too).
        std::vector<float> act_raw(static_cast<std::size_t>(half));
        for (std::uint64_t r = 0; r < half; ++r) {
            act_raw[static_cast<std::size_t>(r)] =
                round_to_bf16(silu_f32(static_cast<float>(gu_a[static_cast<std::size_t>(r)])) *
                              static_cast<float>(gu_a[static_cast<std::size_t>(r + half)]));
        }
        const std::vector<double> act_raw_d(widen_f32(act_raw));
        std::vector<double> dn_raw_a;
        ref_gemm_f32w_d64x(w_dn, dn.rows, dn.columns, act_raw_d.data(), dn_raw_a);

        // ---- metrics: gate_up stage ----
        {
            float oz_scale = 1.0F, oz_absmax = 0.0F;
            const std::vector<std::uint8_t> ox_codes =
                oracle_quantize_u8(x.data(), x.size(), oz_scale, oz_absmax);
            pr.gu.elements         = gu.rows;
            pr.gu.codes_mismatches = count_diff_u8(qx.codes, ox_codes);
            pr.gu.int_mismatches   = compare_fixed_bits(fx_gu, ox_gu, pr.gu.max_abs_diff_fixed);
            const std::vector<double> got = widen_f32(y_gu);
            const VecCompare cmp = compare_vec(got.data(), gu_a.data(), gu_a.size());
            pr.gu.cos     = cmp.cos;
            pr.gu.rel_mae = cmp.rel_mae;
            const std::vector<double> idv = widen_f32(y_gu_id);
            pr.gu.floor_v          = compare_vec(idv.data(), gu_a.data(), gu_a.size()).rel_mae;
            pr.gu.rel_mae_vs_ideal = compare_vec(got.data(), idv.data(), got.size()).rel_mae;
            pr.gu.computed         = true;
        }
        // ---- act stage: bf16 bit patterns, kernel vs ideal (same fp32 face on both sides) ----
        pr.act_elements        = act.size();
        pr.act_bf16_mismatches = compare_bf16_bits(act.data(), act_id.data(), act.size());
        pr.act_computed        = true;
        // ---- metrics: down stage ----
        {
            float oa_scale = 1.0F, oa_absmax = 0.0F;
            const std::vector<std::uint8_t> oa_codes =
                oracle_quantize_u8(act.data(), act.size(), oa_scale, oa_absmax);
            pr.dn.elements         = dn.rows;
            pr.dn.codes_mismatches = count_diff_u8(qa.codes, oa_codes);
            pr.dn.int_mismatches   = compare_fixed_bits(fx_dn, ox_dn, pr.dn.max_abs_diff_fixed);
            const std::vector<double> got = widen_f32(y_dn);
            const VecCompare cmp = compare_vec(got.data(), dn_exact_a.data(), dn_exact_a.size());
            pr.dn.cos     = cmp.cos;
            pr.dn.rel_mae = cmp.rel_mae;
            const std::vector<double> idv = widen_f32(y_dn_id);
            pr.dn.floor_v = compare_vec(idv.data(), dn_exact_a.data(), dn_exact_a.size()).rel_mae;
            pr.dn.rel_mae_vs_ideal = compare_vec(got.data(), idv.data(), got.size()).rel_mae;
            pr.dn.computed         = true;
        }
        // ---- metrics: whole stage ----
        {
            pr.whole.elements = dn.rows;
            const std::vector<double> got = widen_f32(whole);
            const VecCompare cmp = compare_vec(got.data(), dn_raw_a.data(), dn_raw_a.size());
            pr.whole.cos     = cmp.cos;
            pr.whole.rel_mae = cmp.rel_mae;
            const std::vector<double> idv = widen_f32(whole_id);
            pr.whole.floor_v = compare_vec(idv.data(), dn_raw_a.data(), dn_raw_a.size()).rel_mae;
            pr.whole.rel_mae_vs_ideal = compare_vec(got.data(), idv.data(), got.size()).rel_mae;
            pr.whole.computed         = true;
        }
    }

    // ---- zero-x micro control: with x = 0 the compensation must cancel the accumulation exactly
    {
        std::vector<float> zx(static_cast<std::size_t>(gu.columns), 0.0F);
        const QuantizedAct qz  = quantize_u8_zp128(zx.data(), zx.size());
        std::vector<float> zact(static_cast<std::size_t>(half), 0.0F);
        const QuantizedAct qza = quantize_u8_zp128(zact.data(), zact.size());
        std::vector<std::int64_t> z_gu;
        std::vector<std::int64_t> z_dn;
        bool ok = gemv_w4a8(gu.codes, gu.scales, gu.rows, gu.columns, sab_mode, qz.codes.data(),
                            bias_gu, z_gu) &&
                  gemv_w4a8(dn.codes, dn.scales, dn.rows, dn.columns, sab_mode, qza.codes.data(),
                            bias_dn, z_dn);
        for (std::size_t i = 0; ok && i < z_gu.size(); ++i) {
            if (z_gu[i] != 0) { ok = false; }
        }
        for (std::size_t i = 0; ok && i < z_dn.size(); ++i) {
            if (z_dn[i] != 0) { ok = false; }
        }
        out.instr.zero_x_ok = ok;
    }
    return out;
}

// ================================================================================================
// Phase IV -- "merge into the main path" (block 3, stage IV; DEFAULT OFF). W0-b(A) C_r cache,
// W0-b(B) SIMD precompute, W1 serve channel + 6-worker team, W2 share derivation. The W3 wiring
// lives in expert_staging.cpp (the miss fork, :1503/:1504), which is the only caller of this TU.
//
// WHAT THIS SECTION IS, IN ONE PARAGRAPH. The miss fork in expert_staging.cpp may hand a routed
// path's expert to a CPU-side serve engine instead of copying its record over PCIe. The engine is
// the Infernix shape, ported: a single-slot request channel (fields -> release fence -> sequence),
// a serve loop that watches the sequence and answers by writing `done`, and a worker team whose
// worker 0 IS the serve thread (it starts computing at once; the others spin, then park). The
// served work today is the record landing + unpack step: the record is read in full from the host
// bank and C_r for both matrices is computed and cached under a SEMANTIC key, so the per-serve
// cost falls from the scalar precompute (~1.4 ms/expert) to a cache hit. This is why W0-b(A)
// (the cache) and W0-b(B) (SIMD + q9 table) live exactly here: this is the only place that pays
// for precompute_row_bias on the hot service face, and the block-3 caliber measured its cost at
// 1.6-1.8x the whole per-expert pipeline when uncached.
//
// KEY DISCIPLINE (hard condition #1 of the W0-b ruling): the cache key is (key, sab_mode, rows,
// columns) with `key` = the resident identity expert_staging.cpp:1473 computes --
// table_index*kExpertsPerBank + expert. Pointers into (codes, scales) are NEVER the key: under
// the one-pool layout a slot changes owners, so "same address" does not mean "same record".
// A failed q9 (S.1111.111, q9_finite == false) is never stored as a success.
//
// WHAT IT DID NOT DO (BLOCK-3/4), AND WHAT BLOCK-6 W1 WIRED: the forward with the activation face
// needs x, and x was not reachable from the staging call. Blocks 3/4 therefore served the record
// face -- landing + unpack + C_r -- which is exactly the cost the block-3 calibers measured
// (_r-probe: ~135 us/expert = 2.76 MB at ~20 GB/s, cold; the scalar C_r on top of it is the
// 1.4 ms). Block-6 W1 extends the staging signature with the MoE input tensor and the request with
// the x/partial pair, so the decode-width face now computes the served expert with the ported
// W4A16 forward (`forward_job` below) instead of only unpacking it; the record face stays as the
// wider-call and OFF-arm behaviour, untouched. W2 carries the computed partial into the residual
// stream after the MoE kernels (flash_next_cpu_serve_apply, expert_staging.cpp).
//
// COUNTER DISCIPLINE (so the recorded false green cannot repeat here): a served path is counted
// as a MISS and as cpu_served -- never as a hit. The reference's ledger records exactly that
// failure ("CPU-served although the policy counts them as hits", design :1508-1512); the counter
// pair added to the staging line exists so it can be seen, not papered over.
// ================================================================================================

inline constexpr int kCpuServeMaxJobs         = 512;  // = Infernix kMaxCpuJobs (miss_request.h:11)
inline constexpr int kCpuServeWorkers         = 6;    // stage-IV fixed (no worker sweep this round)
inline constexpr std::size_t kCpuServeBiasCap = 512;  // records; 512 x 30,720 B ~= 15.7 MB
inline constexpr std::size_t kCpuServeBytesPerRecord = 30'720; // gu 10,240 + dn 20,480 bytes
inline constexpr int kCpuServePublishWaitMs   = 250;  // bounded wait for the previous request

// Block-8 item 7 (wait aggregation): the serve channel is a RING of this many request slots -- the
// publisher may run this many rounds ahead of the serve thread's answers (the old channel was one
// slot: a new round could not even be filled until the previous one was answered). 3 = one round
// being served, one just published, one free for the next publish; 2 would leave no slack, 4+
// starts paying pinned slab memory for depth the layer-paced apply cannot use yet. This is a
// constant, not an env: both ends of the channel must move together.
inline constexpr std::uint32_t kCpuServeRingSlots = 3;

// ---- block-6 W1/W2: the forward face's constants -------------------------------------------------
// The forward face serves at most one COLUMN per token and the ported forward is capped at eight
// columns (ported::kMaxColumns, asserted at the bottom of this file). kCpuServeApplyWaitMs bounds
// the apply side's wait for the round's partial: the serve team computes while the GPU kernels
// run, so reaching the bound means the engine is broken, not slow -- and a partial that is not
// fully answered must NOT be added (a torn read is the one thing that can never be papered over).
inline constexpr int kCpuServeForwardCols = 8;
inline constexpr int kCpuServeApplyWaitMs = 250;

// Block-8 item 8 (adaptive d): the live CPU throughput, published by the serve loop (single
// writer, only after >=100ms of accumulated service time) and read lock-free by the staging
// decision. 0.0 means "no reading yet" -- callers fall back to the frozen seed. A namespace-scope
// accessor on purpose: the decision path reads it WITHOUT forcing the service singleton into
// existence (instance() starts threads; a report or a landing-only call must not).
inline std::atomic<double>& cpu_serve_c_live() noexcept {
    static std::atomic<double> value{0.0};
    return value;
}

// Block-8 item 8: the adaptive divisor -- the startup formula d = max(1, floor(1.65 + C*t))
// re-run on the LIVE throughput. t stays the configured link time (env override / startup
// default); until the live C exists the frozen seed is returned unchanged. Clamped to [1, 4096]:
// the live input is untrusted in a way the startup doubles never were. Default OFF (the caller
// only reaches this when the share says adaptive), so the OFF path is bit-for-bit serve_share.d.
[[nodiscard]] inline std::int32_t adaptive_serve_d(std::int32_t frozen_d, double t_us) noexcept {
    const double c = cpu_serve_c_live().load(std::memory_order_relaxed);
    if (!(c > 0.0) || !(t_us > 0.0)) { return frozen_d; }
    const long long d = static_cast<long long>(1.65 + c * t_us * 1.0e-6);
    if (d < 1) { return 1; }
    if (d > 4096) { return 4096; }
    return static_cast<std::int32_t>(d);
}

#if defined(_WIN32)
// Declared by hand, like GetProcessIoCounters in expert_staging.cpp: this TU also pulls
// CUDA-facing headers and host/device header mixing has bitten this tree more than once.
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread(void);
extern "C" __declspec(dllimport) std::uintptr_t __stdcall SetThreadAffinityMask(void* thread,
                                                                              std::uintptr_t mask);
// Block-8 item 10: the pin map is chosen against the process affinity mask, so a worker is never
// pinned to a CPU this process cannot use (same hand-declared discipline as the two above).
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) int __stdcall GetProcessAffinityMask(void* process,
                                                                      std::uintptr_t* process_mask,
                                                                      std::uintptr_t* system_mask);
#endif

// Shape of the reference's pin(): optional, OFF by default, failure said out loud, never silently
// ignored (the reference measured pinning to P-cores SLOWER than not pinning: 46.65 vs 50.35).
[[nodiscard]] inline bool cpu_serve_pin_current_thread(int cpu) noexcept {
#if defined(_WIN32)
    if (cpu < 0 || cpu >= 64) { return false; }
    return SetThreadAffinityMask(GetCurrentThread(), std::uintptr_t{1} << cpu) != 0;
#else
    (void)cpu;
    return false;
#endif
}

// Block-8 item 10 (worker orchestration, S6 A4 shape): choose one CPU per worker -- one physical
// core each, the host side left free. On this machine's numbering the first logical CPUs come as
// sibling pairs (0&1 one core, 2&3 the next, ...), so an EVEN STRIDE from the first usable CPU
// lands distinct physical cores; the stride falls back to 1 when the mask is too small for
// pairing to be plausible (fewer than two logical CPUs per worker), which is the old flat
// behaviour. The process affinity mask is consulted so nothing is pinned outside the mask; CPUs
// 0 and 1 (host core and its sibling) are always left clear. PIN_FIRST values below 2 are
// clamped to 2. The chosen map is printed once by the team constructor -- a knob with no line is
// a knob nobody can tell from an unused one.
[[nodiscard]] inline std::vector<int> cpu_serve_pin_targets(int workers, int first_cpu) {
    std::vector<int> out;
    if (workers < 1) { return out; }
    std::uintptr_t proc_mask = ~std::uintptr_t{0};
#if defined(_WIN32)
    std::uintptr_t sys_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) == 0) {
        proc_mask = ~std::uintptr_t{0};
    }
#endif
    int present = 0;
    for (int cpu = 0; cpu < 64; ++cpu) {
        if ((proc_mask >> cpu) & std::uintptr_t{1}) { ++present; }
    }
    const bool even_stride = present >= 2 * workers;
    int cpu                = first_cpu < 2 ? 2 : first_cpu;
    if (even_stride && (cpu & 1) != 0) { ++cpu; }
    for (const int step = even_stride ? 2 : 1;
         cpu < 64 && static_cast<int>(out.size()) < workers; cpu += step) {
        if ((proc_mask >> cpu) & std::uintptr_t{1}) { out.push_back(cpu); }
    }
    return out;
}

// ---- W0-b(A): the semantic-keyed C_r cache ------------------------------------------------------
struct CpuServeBiasPair {
    std::vector<std::int64_t> gu; // C_r per gate_up row
    std::vector<std::int64_t> dn; // C_r per down row
};

class CpuServeBiasCache {
public:
    CpuServeBiasCache() : records_(kCpuServeBiasCap) {}

    CpuServeBiasCache(const CpuServeBiasCache&)            = delete;
    CpuServeBiasCache& operator=(const CpuServeBiasCache&) = delete;

    // Returns the cached pair, or nullptr on miss. The shared_ptr copy keeps the pair alive while
    // the caller uses it, even if the ring recycles the record underneath -- the cache is written
    // by several workers of one round, so a raw reference would be a data race waiting to happen.
    [[nodiscard]] std::shared_ptr<const CpuServeBiasPair>
    lookup(std::uint32_t key, std::int32_t sab_mode, std::uint64_t rows_gu, std::uint64_t columns_gu,
           std::uint64_t rows_dn, std::uint64_t columns_dn) {
        const std::lock_guard<std::mutex> lock(mutex_);
        for (const Record& r : records_) {
            if (r.pair != nullptr && r.key == key && r.sab_mode == sab_mode &&
                r.rows_gu == rows_gu && r.columns_gu == columns_gu && r.rows_dn == rows_dn &&
                r.columns_dn == columns_dn) {
                ++hits_;
                return r.pair;
            }
        }
        ++misses_;
        return nullptr;
    }

    void insert(std::uint32_t key, std::int32_t sab_mode, std::uint64_t rows_gu,
                std::uint64_t columns_gu, std::uint64_t rows_dn, std::uint64_t columns_dn,
                std::shared_ptr<const CpuServeBiasPair> pair) {
        const std::lock_guard<std::mutex> lock(mutex_);
        Record& slot = records_[next_++ % records_.size()];
        if (slot.pair != nullptr) { ++evictions_; }
        slot = Record{key, sab_mode, rows_gu, columns_gu, rows_dn, columns_dn, std::move(pair)};
    }

    void note_failed() {
        const std::lock_guard<std::mutex> lock(mutex_);
        ++failed_;
    }

    struct Stats {
        std::uint64_t hits      = 0;
        std::uint64_t misses    = 0;
        std::uint64_t evictions = 0;
        std::uint64_t failed    = 0;
    };

    [[nodiscard]] Stats stats() {
        const std::lock_guard<std::mutex> lock(mutex_);
        return Stats{hits_, misses_, evictions_, failed_};
    }

private:
    struct Record {
        std::uint32_t key        = 0;
        std::int32_t  sab_mode   = 0;
        std::uint64_t rows_gu    = 0;
        std::uint64_t columns_gu = 0;
        std::uint64_t rows_dn    = 0;
        std::uint64_t columns_dn = 0;
        std::shared_ptr<const CpuServeBiasPair> pair;
    };
    std::vector<Record> records_;
    std::size_t   next_     = 0;
    std::uint64_t hits_     = 0;
    std::uint64_t misses_   = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t failed_   = 0;
    std::mutex    mutex_;
};

// ---- W1: the request (shape of Infernix miss_request.h:16-29, host-internal) --------------------
//
// Fields first, sequence last. The publisher fills every field, then a release store to the
// service's `published` counter -- the modern spelling of the reference's "every field, a
// system-scope fence, then `sequence`". The service acquires, reads the fields, and answers by
// storing the same sequence to `done` after every job is done. A service that never answers is
// visible as answered < published plus stalls > 0 -- never silent.
struct CpuServeRequest {
    std::uint32_t sequence   = 0;
    std::int32_t  slot       = 0; // Block-8 item 7: which ring slot this request occupies
    std::int32_t  layer      = 0;
    std::int32_t  jobs       = 0;
    std::int32_t  sab_mode   = 0;
    std::uint64_t rows_gu    = 0;
    std::uint64_t columns_gu = 0;
    std::uint64_t rows_dn    = 0;
    std::uint64_t columns_dn = 0;
    // ---- block-6 W1: the forward face (default off; armed by the staging side) ------------------
    // WHAT CHANGED AND WHY: the record face above lands + unpacks; the forward face actually
    // computes the served expert. It replaces the per-job x/y placeholder arrays of the Infernix
    // job shape with the buffer pair the compute chain really needs, all HOST-PINNED and owned by
    // the staging side: x mirrors this call's MoE input [kHidden, cols] BF16 (one D2H per call);
    // partial [kHidden, cols] FP32 is the accumulator the team adds alpha*y into and the apply
    // side reads after the MoE kernels. Column c of both is token c, laid out exactly like the
    // input/output tensors (element (row, c) at c*kHidden + row). Per job, the columns that route
    // to this expert and the summed routing weight per column -- a repeat identity in one call
    // shares one job, and two ranks on one token add their weights (same y, so the weights add).
    bool forward = false;
    std::int32_t tokens = 0;                       // columns of x/partial (<= kCpuServeForwardCols)
    const std::uint16_t* x = nullptr;              // [kHidden * tokens] BF16, host-pinned
    float* partial = nullptr;                      // [kHidden * tokens] FP32, host-pinned
    // Block-8 item 3a: the stream-wait channel's two extra pointers (staging-filled; null = the
    // plain blocking shape). `partial_bf16` is this round's converted slab -- the serve thread
    // converts after the ordered reduce, BEFORE the answer, so BOTH consumption paths (the blocking
    // H2D and the stream wait) read a finished slab. `done_flag` is the slot's completion word in
    // pinned host memory: the serve thread release-stores the round's sequence once the slab is
    // complete; the apply side's stream waits for exactly that value.
    std::uint16_t* partial_bf16 = nullptr;
    std::uint32_t* done_flag    = nullptr;
    // Block-7 fix: per-job y slots, `kCpuServeForwardCols * kHidden` BF16 cells per job, written
    // ONLY by the worker that picked that job. The serve thread then folds the slots into `partial`
    // in job order (reduce_forward) -- one writer per slot, one fixed summation order, so the
    // served sum is bit-stable run to run. (The parallel `+=` into `partial` this replaces was a
    // data race: two jobs sharing a token column accumulated the same cells concurrently, which
    // both loses updates and lets the float sum order follow completion timing.)
    std::uint16_t* job_y = nullptr;
    std::int32_t jn[kCpuServeMaxJobs]{};           // this job's column count (0 = empty)
    std::int32_t jt[kCpuServeMaxJobs][kCpuServeForwardCols]{};
    float        ja[kCpuServeMaxJobs][kCpuServeForwardCols]{};
    float        divisor_gu[kCpuServeMaxJobs]{};   // this expert's gu divisor (host bank, by expert)
    float        divisor_dn[kCpuServeMaxJobs]{};
    // The record face. `key` is the semantic identity (layer*kExpertsPerBank + expert); the four
    // pointers address the record inside the (stable) host bank and are read immediately by the
    // serving round, never used as identity.
    std::uint32_t key[kCpuServeMaxJobs]{};
    const std::uint8_t* codes_gu[kCpuServeMaxJobs]{};
    const std::uint8_t* scales_gu[kCpuServeMaxJobs]{};
    const std::uint8_t* codes_dn[kCpuServeMaxJobs]{};
    const std::uint8_t* scales_dn[kCpuServeMaxJobs]{};
};

// Per-request counters, written by all workers of a round (atomics: the workers share them).
struct CpuServeCounters {
    std::atomic<std::uint64_t> jobs{0};
    std::atomic<std::uint64_t> landed_records{0}; // lookups that actually read the record
    std::atomic<std::uint64_t> landed_bytes{0};   // and the bytes that read touched
    std::atomic<std::uint64_t> jobs_failed{0};
    std::atomic<std::uint32_t> status{0};         // 0 = ok; first non-zero wins
    // Block-6 W1: the forward face (decode width). fwd_jobs counts served-expert forward jobs the
    // team computed; fwd_elements is the hidden elements they added into the partial. Cumulative
    // across rounds, like `jobs` -- the self-report line names both faces.
    std::atomic<std::uint64_t> fwd_jobs{0};
    std::atomic<std::uint64_t> fwd_elements{0};
};

// ---- W1: the worker team (shape of Infernix expert_team.h:45-109) -------------------------------
//
// A round hands out work through one atomic counter (faster cores simply take more of it); worker
// 0 is the calling (serve) thread and starts at once; idle workers spin for a bounded number of
// pause iterations, then park on the epoch word (std::atomic::wait is the portable spelling of
// the reference's futex park). `epoch_` counts runs from 1 and a freshly started worker begins at
// 0, so the first run can never be missed; `finished_` is the round's completion count that
// worker 0 waits on. The protocol is strictly one round at a time (the serve loop serializes).
class CpuServeTeam {
public:
    CpuServeTeam(int workers, int spin_iterations, bool pin, int pin_first_cpu)
        : workers_(workers < 1 ? 1 : workers),
          spin_iterations_(spin_iterations < 1 ? 1 : spin_iterations),
          pin_(pin),
          pin_first_cpu_(pin_first_cpu) {
        // Block-8 item 10: build the pin map before any worker starts; the evidence line below is
        // the single place the chosen CPUs are visible (printed only when pinning is on).
        if (pin_) {
            pin_cpus_ = cpu_serve_pin_targets(workers_, pin_first_cpu_);
            std::fprintf(stderr, "flash_next_cpu_expert_serve pin=1 workers=%d cpus=", workers_);
            for (std::size_t i = 0; i < pin_cpus_.size(); ++i) {
                std::fprintf(stderr, "%s%d", (i == 0) ? "" : ",", pin_cpus_[i]);
            }
            std::fprintf(stderr, " host_reserved=0,1 stride=%s\n",
                         (pin_cpus_.size() >= 2 && pin_cpus_[1] == pin_cpus_[0] + 2) ? "even"
                                                                                     : "flat");
        }
        for (int w = 1; w < workers_; ++w) {
            worker_threads_.emplace_back([this, w] { worker_main(w); });
        }
    }

    ~CpuServeTeam() {
        stop_.store(true, std::memory_order_release);
        epoch_.fetch_add(1, std::memory_order_release);
        epoch_.notify_all();
        for (std::thread& t : worker_threads_) {
            if (t.joinable()) { t.join(); }
        }
    }

    CpuServeTeam(const CpuServeTeam&)            = delete;
    CpuServeTeam& operator=(const CpuServeTeam&) = delete;

    // Serves every job of `request`; returns when every worker of this round has finished.
    void run(const CpuServeRequest& request, CpuServeBiasCache& cache, CpuServeCounters& counters) {
        if (request.jobs <= 0) { return; }
        request_  = &request;
        cache_    = &cache;
        counters_ = &counters;
        next_job_.store(0, std::memory_order_relaxed);
        finished_.store(0, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release); // publishes the pointers above
        epoch_.notify_all();
        try {
            work(0); // worker 0 = the calling (serve) thread; it starts computing at once
        } catch (...) {
            note_status(2); // a failure never hangs the round and never answers "ok"
        }
        while (finished_.load(std::memory_order_acquire) < workers_ - 1) { _mm_pause(); }
    }

    // Block-7 fix (deterministic reduce): fold every job's y slot into the request's partial in
    // (job, column, hidden) order on the calling -- serve -- thread, after the round's workers have
    // finished (call it between run() and answering the request). One writer per slot plus one
    // fixed summation order make the served sum bit-stable run to run; the parallel `+=` this
    // replaces was a data race on shared token columns and an order lottery.
    void reduce_forward();

    // Block-8 item 10: the serve thread IS worker 0; pin it like the worker threads so the whole
    // team sits one per physical core. Called from the serve loop's own entry (that thread).
    void pin_worker_zero() {
        if (!pin_ || pin_cpus_.empty()) { return; }
        const int cpu = pin_cpus_.front();
        if (!cpu_serve_pin_current_thread(cpu)) {
            static std::atomic<bool> warned{false};
            bool expected = false;
            if (warned.compare_exchange_strong(expected, true)) {
                std::fprintf(stderr,
                             "flash_next_cpu_expert_serve pin=1 worker=0 cpu=%d "
                             "SetThreadAffinityMask failed; serve thread runs unpinned\n",
                             cpu);
            }
        }
    }

private:
    void note_status(std::uint32_t code) noexcept {
        std::uint32_t expected = 0;
        counters_->status.compare_exchange_strong(expected, code, std::memory_order_relaxed);
    }

    void worker_main(int w) {
        if (pin_) {
            const int cpu = (w >= 0 && w < static_cast<int>(pin_cpus_.size()))
                                ? pin_cpus_[static_cast<std::size_t>(w)]
                                : -1;
            if (cpu < 0 || !cpu_serve_pin_current_thread(cpu)) {
                static std::atomic<bool> warned{false};
                bool expected = false;
                if (warned.compare_exchange_strong(expected, true)) {
                    std::fprintf(stderr,
                                 "flash_next_cpu_expert_serve pin=1 worker=%d cpu=%d has no "
                                 "usable core; worker runs unpinned\n",
                                 w, cpu);
                }
            }
        }
        std::uint64_t my_epoch = 0; // epoch counts runs from 1; begin at 0 so run 1 is never missed
        for (;;) {
            int spins = spin_iterations_;
            while (epoch_.load(std::memory_order_acquire) == my_epoch && spins-- > 0) {
                _mm_pause();
            }
            if (epoch_.load(std::memory_order_acquire) == my_epoch) {
                // bounded spin exhausted: park on the epoch word (the futex shape)
                sleepers_.fetch_add(1, std::memory_order_acq_rel);
                epoch_.wait(my_epoch, std::memory_order_acquire);
                sleepers_.fetch_sub(1, std::memory_order_acq_rel);
                continue; // either woken or epoch moved -- re-evaluate from the top
            }
            if (stop_.load(std::memory_order_acquire)) { return; }
            my_epoch = epoch_.load(std::memory_order_acquire);
            try {
                work(w);
            } catch (...) {
                note_status(2); // counted, not silent; the round still completes
            }
            finished_.fetch_add(1, std::memory_order_release);
        }
    }

    void work(int w) {
        (void)w; // every worker takes from the same atomic counter (the reference's block shape)
        for (int j = next_job_.fetch_add(1, std::memory_order_relaxed); j < request_->jobs;
             j = next_job_.fetch_add(1, std::memory_order_relaxed)) {
            if (request_->forward) {
                forward_job(j); // block-6 W1: compute the served expert and add into the partial
            } else {
                unpack_job(j);
            }
        }
    }

    // One served record: land it (read the record in full) and unpack it (C_r for both matrices),
    // through the semantic-keyed cache. A cache hit means this identity was already landed and
    // unpacked -- no read, no compute; that reuse IS the W0-b(A) saving, and its measured rate r
    // is reported on the staging line. A miss reads codes+scales of both matrices and computes
    // C_r exactly once per identity.
    void unpack_job(int j) {
        const CpuServeRequest& r = *request_;
        counters_->jobs.fetch_add(1, std::memory_order_relaxed);
        const std::uint32_t key = r.key[j];
        if (cache_->lookup(key, r.sab_mode, r.rows_gu, r.columns_gu, r.rows_dn, r.columns_dn) !=
            nullptr) {
            return; // reused: the record was landed and unpacked by an earlier service of this key
        }
        std::vector<std::int64_t> bias_gu;
        std::vector<std::int64_t> bias_dn;
        bool q9_finite = true;
        if (!precompute_row_bias(r.codes_gu[j], r.scales_gu[j], r.rows_gu, r.columns_gu, r.sab_mode,
                                 bias_gu, q9_finite) ||
            !precompute_row_bias(r.codes_dn[j], r.scales_dn[j], r.rows_dn, r.columns_dn, r.sab_mode,
                                 bias_dn, q9_finite)) {
            // A failed q9 (S.1111.111) is NEVER stored as a success. The job is counted failed and
            // the request status turns non-zero -- a failed service, not a silent skip.
            cache_->note_failed();
            counters_->jobs_failed.fetch_add(1, std::memory_order_relaxed);
            note_status(1);
            return;
        }
        auto pair = std::make_shared<CpuServeBiasPair>();
        pair->gu  = std::move(bias_gu);
        pair->dn  = std::move(bias_dn);
        cache_->insert(key, r.sab_mode, r.rows_gu, r.columns_gu, r.rows_dn, r.columns_dn,
                       std::move(pair));
        counters_->landed_records.fetch_add(1, std::memory_order_relaxed);
        counters_->landed_bytes.fetch_add(kCpuServeBytesPerRecord, std::memory_order_relaxed);
    }

    // Block-6 W1: one job of the forward face -- the served expert's whole W4A16 forward for its
    // routed columns, written into the job's own slot of `job_y` (block-7 fix: no shared-memory
    // accumulate here; the sum is folded by reduce_forward above). Defined OUT OF LINE at the
    // bottom of this header: it calls the ported forward, which is declared after this class.
    void forward_job(int j);

    int workers_;
    int spin_iterations_;
    bool pin_;
    int pin_first_cpu_;
    std::vector<int> pin_cpus_; // Block-8 item 10: chosen per-worker CPUs (empty when unpinned)

    const CpuServeRequest* request_  = nullptr;
    CpuServeBiasCache*     cache_    = nullptr;
    CpuServeCounters*      counters_ = nullptr;

    alignas(64) std::atomic<std::uint64_t> epoch_{0};
    alignas(64) std::atomic<int> sleepers_{0};
    alignas(64) std::atomic<int> finished_{0};
    alignas(64) std::atomic<int> next_job_{0};
    alignas(64) std::atomic<bool> stop_{false};
    std::vector<std::thread> worker_threads_;
};

// ---- W1: the service (shape of Infernix miss_service.cpp:54-91, :151-252) -----------------------
class CpuServeService {
public:
    // Process-lifetime singleton, built on first use. The first call's (pin, pin_first_cpu)
    // arguments win; the only caller is the staging wiring, which reads the env once.
    static CpuServeService& instance(bool pin = false, int pin_first_cpu = 2) {
        static CpuServeService* service = new CpuServeService(pin, pin_first_cpu);
        return *service;
    }

    CpuServeService(const CpuServeService&)            = delete;
    CpuServeService& operator=(const CpuServeService&) = delete;

    // Publisher side (the staging call thread). A single-slot channel: the previous request must
    // be answered before its fields may be rewritten, so this waits -- bounded. Expiry returns
    // nullptr and counts a stall; the staging side then keeps the PCIe path for everything, which
    // is the safe direction (and a stall is loudly visible on the self-report line).
    //
    // Block-7 fix (wait primitive): this used to poll `sleep_for(50us)`, which on Windows quantises
    // to one ~15.6ms timer tick -- a measured 11.1-16.5ms per call, ~300x the nominal step, on the
    // per-round critical path (730+ rounds per token). Polling is now a condition-variable wait:
    // woken by the serve thread's notify as soon as the round answers, still bounded by the same
    // deadline so a stalled or dead service keeps its exact old behaviour (stall counted, PCIe
    // path kept).
    CpuServeRequest* begin_request() {
        // Block-8 item 7 (ring): a new round takes the next ring slot. The slot may only be reused
        // once the round that last occupied it (N rounds earlier) has been ANSWERED -- its fields
        // must not change under the serve thread or the team, and its workspace slabs stay read-
        // only-in-flight. This is the old "previous request answered" condition generalised 1 -> N:
        // the publisher no longer stalls behind the round it just published.
        const std::uint32_t next = published_.load(std::memory_order_acquire) + 1;
        const std::uint32_t s    = (next - 1) % kCpuServeRingSlots;
        if (next > kCpuServeRingSlots) {
            const std::uint32_t needed = next - kCpuServeRingSlots;
            std::unique_lock<std::mutex> lock(wait_mutex_);
            const bool slot_free = answered_cv_.wait_for(
                lock, std::chrono::milliseconds(kCpuServePublishWaitMs),
                [this, needed] { return answered_.load(std::memory_order_acquire) >= needed; });
            if (!slot_free) {
                stalls_.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
        }
        request_[s].jobs = 0;
        request_[s].slot = static_cast<std::int32_t>(s);
        return &request_[s];
    }

    // Fields already written; publish. Sequence+release: every field write stays before the
    // visible sequence (the reference's published contract, spelled with the C++ memory model).
    // Single publisher (the staging host thread): begin and commit derive the same (seq, slot).
    void commit_request() {
        const std::uint32_t seq = published_.load(std::memory_order_relaxed) + 1;
        const std::uint32_t s   = (seq - 1) % kCpuServeRingSlots;
        std::atomic_thread_fence(std::memory_order_release);
        request_[s].sequence = seq;
        published_.store(seq, std::memory_order_release);
        // Block-7 fix (wait primitive): wake the serve thread's channel wait immediately instead of
        // letting it discover the commit after its next (timer-quantised) idle rest.
        published_.notify_all();
        // Block-6 W2: name what the round just published serves. `last_forward_` is consumed by
        // the apply side on the same host thread; `scope_forward_` only feeds the self-report line.
        last_forward_  = request_[s].forward;
        scope_forward_ = request_[s].forward;
    }

    // ---- block-6 W2: the apply side (same host thread that published) ---------------------------
    // Did the round that just committed run the forward face? Consumed once per round: the apply
    // must add exactly one partial per forward-armed round, and a round that published nothing
    // leaves the flag clear.
    [[nodiscard]] bool take_forward_round() noexcept {
        const bool armed = last_forward_;
        last_forward_     = false;
        return armed;
    }

    // Bounded wait for the round to be ANSWERED (partial complete). A late round returns false and
    // is counted by the caller; the partial must never be read while the team is still writing it.
    //
    // Block-7 fix (wait primitive): the poll `sleep_for(50us)` here was the main offender -- one
    // measured ~15.6ms timer tick of dead host time per call while the team needed microseconds to
    // milliseconds (the round it overlapped usually finished during that very tick). The wait is
    // now woken by the serve thread's notify; the deadline keeps the old bounded semantics.
    [[nodiscard]] bool await_last_round(int wait_ms) {
        const std::uint32_t want = published_.load(std::memory_order_acquire);
        if (want == 0) { return false; }
        const auto wait_t0 = std::chrono::steady_clock::now(); // Block-8 item 7: apply-side segment
        std::unique_lock<std::mutex> lock(wait_mutex_);
        const bool ok = answered_cv_.wait_for(lock, std::chrono::milliseconds(wait_ms), [this, want] {
            return answered_.load(std::memory_order_acquire) == want;
        });
        prof_wait_ns_.fetch_add(
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - wait_t0)
                                           .count()),
            std::memory_order_relaxed);
        if (ok) {
            // Block-8 item 7: the slot this round lives in, so forward_partial() reads the right
            // slab (same host thread as the publisher; no atomics needed).
            last_applied_slot_ = (want - 1) % kCpuServeRingSlots;
        }
        return ok;
    }

    [[nodiscard]] const float* forward_partial() const noexcept {
        return request_[last_applied_slot_].partial;
    }

    // Block-8 item 7: the ring slot the last successful await resolved to (the apply side's bf16
    // staging copy is per-slot too; see flash_next_cpu_serve_apply).
    [[nodiscard]] std::uint32_t last_applied_slot() const noexcept { return last_applied_slot_; }

    // Block-8 item 3a: the sequence of the most recently committed round. The apply side reads it
    // on the same host thread that published, so the value it sees is this round's.
    [[nodiscard]] std::uint32_t last_committed_seq() const noexcept {
        return published_.load(std::memory_order_acquire);
    }

    // Block-8 item 3a: start the stream-watch monitor (once). Called by the apply side the first
    // time the stream-wait path is used -- the monitor exists only where that channel does.
    void arm_stream_watchdog() noexcept {
        bool expected = false;
        if (stream_watch_armed_.compare_exchange_strong(expected, true)) {
            std::thread(&CpuServeService::stream_watch_loop, this).detach();
        }
    }

    void note_forward_applied() noexcept { fwd_applied_.fetch_add(1, std::memory_order_relaxed); }
    void note_forward_stall() noexcept { fwd_stalls_.fetch_add(1, std::memory_order_relaxed); }

    struct Snapshot {
        std::uint64_t published       = 0;
        std::uint64_t answered        = 0;
        std::uint64_t requests        = 0;
        std::uint64_t jobs            = 0;
        std::uint64_t landed_records  = 0;
        std::uint64_t landed_bytes    = 0;
        std::uint64_t jobs_failed     = 0;
        std::uint64_t stalls          = 0;
        std::uint64_t errors          = 0;
        std::uint32_t last_status     = 0;
        // Block-6: the forward face's counters (see CpuServeCounters) plus the apply side's.
        std::uint64_t fwd_jobs        = 0;
        std::uint64_t fwd_elements    = 0;
        std::uint64_t fwd_applied     = 0;
        std::uint64_t fwd_stalls      = 0;
        // Block-8 item 7: segmented serve timings (nanosecond accumulators; reported only on the
        // env-gated serve_prof line -- the frozen serve line's fields are untouched).
        std::uint64_t prof_run_ns     = 0;
        std::uint64_t prof_reduce_ns  = 0;
        std::uint64_t prof_wait_ns    = 0;
        // Block-8 item 3a: rounds the stream-watch monitor had to fuse (dropped contributions).
        std::uint64_t fwd_fused       = 0;
        bool          scope_forward   = false; // what the LAST committed round served
        CpuServeBiasCache::Stats cache;
    };

    [[nodiscard]] Snapshot snapshot() {
        Snapshot s;
        s.published      = published_.load(std::memory_order_acquire);
        s.answered       = answered_.load(std::memory_order_acquire);
        s.requests       = requests_served_.load(std::memory_order_relaxed);
        s.jobs           = counters_.jobs.load(std::memory_order_relaxed);
        s.landed_records = counters_.landed_records.load(std::memory_order_relaxed);
        s.landed_bytes   = counters_.landed_bytes.load(std::memory_order_relaxed);
        s.jobs_failed    = counters_.jobs_failed.load(std::memory_order_relaxed);
        s.stalls         = stalls_.load(std::memory_order_relaxed);
        s.errors         = errors_.load(std::memory_order_relaxed);
        s.last_status    = status_.load(std::memory_order_relaxed);
        s.fwd_jobs       = counters_.fwd_jobs.load(std::memory_order_relaxed);
        s.fwd_elements   = counters_.fwd_elements.load(std::memory_order_relaxed);
        s.fwd_applied    = fwd_applied_.load(std::memory_order_relaxed);
        s.fwd_stalls     = fwd_stalls_.load(std::memory_order_relaxed);
        s.prof_run_ns    = prof_run_ns_.load(std::memory_order_relaxed);
        s.prof_reduce_ns = prof_reduce_ns_.load(std::memory_order_relaxed);
        s.prof_wait_ns   = prof_wait_ns_.load(std::memory_order_relaxed);
        s.fwd_fused      = fwd_fused_.load(std::memory_order_relaxed);
        s.scope_forward  = scope_forward_;
        s.cache          = cache_.stats();
        return s;
    }

private:
    CpuServeService(bool pin, int pin_first_cpu)
        : team_(kCpuServeWorkers, 1 << 16, pin, pin_first_cpu),
          server_(&CpuServeService::serve_loop, this) {}

    void serve_loop() {
        team_.pin_worker_zero(); // Block-8 item 10: worker 0's own affinity, set by this thread
        try {
            std::uint32_t seen  = 0;
            std::uint64_t spins = 0;
            for (;;) {
                const std::uint32_t seq = published_.load(std::memory_order_acquire);
                heartbeat_.fetch_add(1, std::memory_order_relaxed);
                if (seq == seen) {
                    // Idle: spin briefly, then park on the channel word -- a loop that burns a core
                    // between calls would tax the very bandwidth the CPU side is competing for.
                    // Block-7 fix (wait primitive): the rest step is a futex wait on `published_`
                    // (woken by commit_request's notify), never a timer-quantised sleep. The old
                    // `sleep_for(50us)` parked this thread for ~15.6ms per rest, so the next commit
                    // waited out the rest before the round could even start -- one of the two ticks
                    // per round that made B 10x slower than A.
                    if (++spins % 4096 == 0) {
                        published_.wait(seen, std::memory_order_acquire);
                    }
                    _mm_pause();
                    continue;
                }
                // Block-8 item 7 (ring): process exactly the NEXT round. `seen = seq` would skip
                // any rounds published in between (their answers would never arrive); one round per
                // pass, and the loop returns for whatever else has been published.
                std::atomic_thread_fence(std::memory_order_acquire);
                const std::uint32_t round = seen + 1;
                seen                      = round;
                const CpuServeRequest& r =
                    request_[static_cast<std::size_t>((round - 1) % kCpuServeRingSlots)];
                requests_served_.fetch_add(1, std::memory_order_relaxed);
                counters_.status.store(0, std::memory_order_relaxed);
                // Block-8 item 8: service timing for the live C -- the window covers run + reduce,
                // i.e. the round latency the apply side actually waits out.
                const auto round_t0 = std::chrono::steady_clock::now();
                try {
                    team_.run(r, cache_, counters_);
                } catch (...) {
                    note_status(2);
                }
                // Block-8 item 7: the run segment of the env-gated serve_prof line.
                prof_run_ns_.fetch_add(
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - round_t0)
                                                   .count()),
                    std::memory_order_relaxed);
                // Block-7 fix (deterministic reduce): fold the jobs' y slots into the partial here,
                // in job order, single-threaded, BEFORE the answer -- so the apply side can never
                // read a partial still being summed.
                if (r.forward) {
                    const auto reduce_t0 = std::chrono::steady_clock::now();
                    team_.reduce_forward();
                    prof_reduce_ns_.fetch_add(
                        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                       std::chrono::steady_clock::now() - reduce_t0)
                                                       .count()),
                        std::memory_order_relaxed);
                    // Block-8 item 3a: convert + publish the completion flag BEFORE the answer --
                    // every consumer of this round reads a finished bf16 slab once `answered_`
                    // (or the flag) says the round is done. Defined out of line at the bottom: it
                    // calls the ported converter, which is declared later in this header.
                    publish_forward_round(r);
                }
                const std::uint32_t status = counters_.status.load(std::memory_order_relaxed);
                if (status != 0) { errors_.fetch_add(1, std::memory_order_relaxed); }
                status_.store(status, std::memory_order_relaxed);
                done_.store(round, std::memory_order_release);
                {
                    // Store under the mutex: a waiter that just checked the predicate must not be
                    // able to slip into its wait between this store and the notify.
                    const std::lock_guard<std::mutex> lock(wait_mutex_);
                    answered_.store(round, std::memory_order_release);
                }
                answered_cv_.notify_all();
                // Block-8 item 8: feed the live CPU throughput (the serve thread is the only
                // writer). Published only after >=100ms of accumulated service time, so the
                // decision side never sees a rate built from one or two rounds.
                if (r.jobs > 0) {
                    const auto round_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              std::chrono::steady_clock::now() - round_t0)
                                              .count();
                    c_jobs_total_ += static_cast<std::uint64_t>(r.jobs);
                    c_ns_total_ += static_cast<std::uint64_t>(round_ns > 0 ? round_ns : 1);
                    if (c_ns_total_ >= 100'000'000ULL) {
                        cpu_serve_c_live().store(
                            static_cast<double>(c_jobs_total_) * 1.0e9 /
                                static_cast<double>(c_ns_total_),
                            std::memory_order_relaxed);
                    }
                }
                spins = 0;
            }
        } catch (...) {
            // The service died: it will never answer again, and the publisher's bounded wait
            // turns that into stalls on the self-report line -- a red instrument, not silence.
            errors_.fetch_add(1, std::memory_order_relaxed);
            // Block-7 fix: the waits are condition-variable waits now; wake them so they can still
            // observe their deadline instead of sleeping through it.
            answered_cv_.notify_all();
        }
    }

    void note_status(std::uint32_t code) noexcept {
        std::uint32_t expected = 0;
        counters_.status.compare_exchange_strong(expected, code, std::memory_order_relaxed);
    }

    // Block-8 item 3a: convert the round's fp32 partial into its bf16 slab and store the completion
    // flag. Defined out of line at the bottom (it calls the ported converter, declared later in
    // this header). Runs on the serve thread between the ordered reduce and the answer.
    void publish_forward_round(const CpuServeRequest& r);

    // Block-8 item 3a (watchdog M1): the stream-wait channel has NO timeout -- a serve thread that
    // dies with a round in flight would leave the stream waiting forever. This monitor is the
    // escape: while a round is unanswered AND the serve heartbeat has not moved for ten 100ms
    // ticks, the round is declared dead, its slab zeroed (zero contribution = the old drop
    // semantics), its flag stored and its answer advanced -- the ring drains instead of hanging.
    // A merely slow-but-alive round can never be fused: the heartbeat moves with every serve-loop
    // pass. After the first fuse the monitor serves every later round within ~100ms (degraded
    // mode), so a dead service costs bounded lateness instead of a deadlock.
    void stream_watch_loop() {
        std::uint64_t last_hb = heartbeat_.load(std::memory_order_relaxed);
        int frozen_ticks      = 0;
        bool declared_dead    = false;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const std::uint64_t hb  = heartbeat_.load(std::memory_order_relaxed);
            const std::uint32_t ans = answered_.load(std::memory_order_acquire);
            const std::uint32_t pub = published_.load(std::memory_order_acquire);
            if (pub != ans) {
                frozen_ticks = (hb == last_hb) ? (frozen_ticks + 1) : 0;
                if (frozen_ticks >= (declared_dead ? 1 : 10)) {
                    // The slot for the OLDEST unanswered round (ordinal ans+1). It cannot have been
                    // reused: reuse requires answered >= its ordinal (begin_request's rule).
                    CpuServeRequest& rr = request_[ans % kCpuServeRingSlots];
                    // Defence in depth: fuse only the request this ordinal names.
                    if (rr.done_flag != nullptr && rr.sequence == ans + 1) {
                        if (rr.partial_bf16 != nullptr && rr.tokens >= 1 &&
                            rr.tokens <= kCpuServeForwardCols) {
                            const std::size_t n = static_cast<std::size_t>(2'560) *
                                                  static_cast<std::size_t>(rr.tokens);
                            for (std::size_t k = 0; k < n; ++k) { rr.partial_bf16[k] = 0; }
                        }
                        fwd_fused_.fetch_add(1, std::memory_order_relaxed);
                        std::atomic_ref<std::uint32_t>(*rr.done_flag)
                            .store(rr.sequence, std::memory_order_release);
                        {
                            const std::lock_guard<std::mutex> lock(wait_mutex_);
                            answered_.store(rr.sequence, std::memory_order_release);
                        }
                        answered_cv_.notify_all();
                        static std::atomic<bool> fused_warned{false};
                        bool expected = false;
                        if (fused_warned.compare_exchange_strong(expected, true)) {
                            std::fprintf(stderr,
                                         "flash_next_cpu_expert_serve stream-watch: fused a dead "
                                         "round (serve heartbeat frozen); contributions are dropped "
                                         "and counted by fwd_fused\n");
                        }
                        declared_dead = true;
                        frozen_ticks  = 0;
                    }
                }
            } else {
                frozen_ticks = 0;
            }
            last_hb = hb;
        }
    }

    alignas(64) CpuServeRequest request_[kCpuServeRingSlots];
    CpuServeBiasCache cache_;
    CpuServeTeam      team_;

    std::atomic<std::uint32_t> published_{0};
    std::atomic<std::uint32_t> answered_{0};
    std::atomic<std::uint32_t> done_{0};
    std::atomic<std::uint32_t> status_{0};
    std::atomic<std::uint64_t> heartbeat_{0};
    std::atomic<std::uint64_t> requests_served_{0};
    std::atomic<std::uint64_t> stalls_{0};
    std::atomic<std::uint64_t> errors_{0};
    std::atomic<std::uint64_t> fwd_applied_{0};
    std::atomic<std::uint64_t> fwd_stalls_{0};
    // Block-7 fix (wait primitive): the answer channel's condition variable. The serve thread
    // stores `answered_` under this mutex and notifies; the apply side and the next publisher wait
    // with wait_for -- bounded exactly like the polls they replace, but woken by the notify
    // instead of a ~15.6ms timer tick. (The channel word itself `published_` is waited on by the
    // serve thread directly, notify-driven, no mutex needed: it has a single waiter and the store
    // precedes the notify.)
    std::mutex wait_mutex_;
    std::condition_variable answered_cv_;
    // Block-6 W2: written at commit, consumed/read by the apply side and the report point -- all
    // on the same host thread. The serve thread never touches either.
    bool last_forward_  = false;
    bool scope_forward_ = false;
    // Block-8 item 7: the ring slot the last successful await resolved to (host thread only -- the
    // publisher, the apply side and this member all live on the staging thread).
    std::uint32_t last_applied_slot_ = 0;
    // Block-8 item 3a: the stream-watch monitor latch and the fused-round counter.
    std::atomic<bool> stream_watch_armed_{false};
    std::atomic<std::uint64_t> fwd_fused_{0};
    // Block-8 item 7: the serve_prof accumulators. The serve thread writes run/reduce, the host
    // thread writes wait; every reader goes through snapshot().
    std::atomic<std::uint64_t> prof_run_ns_{0};
    std::atomic<std::uint64_t> prof_reduce_ns_{0};
    std::atomic<std::uint64_t> prof_wait_ns_{0};
    CpuServeCounters counters_;
    std::uint64_t c_jobs_total_{0}; // Block-8 item 8: serve-thread-only accumulators for the live C
    std::uint64_t c_ns_total_{0};

    // Last member: its thread starts inside this constructor and must find every field above
    // already initialised. Never joined -- the singleton is process-lifetime (never destroyed).
    std::thread server_;
};

// ================================================================================================
// Block-4 P1 -- ported CPU expert forward (W4A16) for the real serve path.
//
// LICENSE / ATTRIBUTION (compliance three, PM-signed):
//   Portions derived from Infernix (github.com/Wallawalla47/Infernix, Apache-2.0).
//   Refs: refs/infernix-20261013 (fork of Neroued/ninfer); upstream NInfer (Apache-2.0).
//   Modified for NInfer fusion line: K16M128x4 face adaptation (phi1/phi2/phi3), kAvx512Vnni
//   tier omitted (this host has no AVX-512; executing it would SIGILL). Part extraction only --
//   no whole-file copy (the reference's w4a4/w4a16 sources carry 512-bit vector intrinsics in
//   22/28 places; only the
//   kScalar/kAvx2/kAvxVnni tiers and the A16 math closure are taken, see the extraction marks).
//
// WHAT THIS IS: the W4A16 canonical arithmetic (per expert: encode x -> gate/up units + SwiGLU ->
// encode h -> down rows -> y in BF16), plus this tree's own layout-adapted kernels that read THIS
// tree's expert bank (row-major codes + K16M128x4 scales) through the first-check-A maps:
//   phi1 codes : (r,c) -> byte (r*C/2 + c/2), nibble (c&1)
//   phi2 scales: k16m128x4_scale_byte(row, col/16, C/64)
//   phi3 rows  : gu record interleaves gate/up rows (even=gate_i, odd=up_i); this tree stores
//                gate rows [0:640) and up rows [640:1280) -> their_row -> ported_row_gu().
// The ORIGINAL unit-order kernels are kept as the throughput reference arm of the port probe
// (values on this tree's bytes are meaningless there -- the arm exists to measure the access
// pattern, not the data).
//
// SABOTAGE (diagnostic plumbing for the V2 red arms; DEFAULT OFF, no production code touched):
// every forward entry takes `sab_mode` -- 0 = off; 1 = output row shift +1 ("logits row offset"
// family); 2 = code nibble swap (column-pair swap); 3 = drop the biased-code correction.
// The staging side reads NINFER_FLASH_NEXT_CPU_EXPERT_PORT_SABOTAGE and passes it through.
// ================================================================================================

namespace ported {

// ---- canon A16 math (part-extracted; semantics identical to the reference's canonical_math.h) ---

inline constexpr std::uint32_t kCanonicalNan = 0x7FC00000U;
inline constexpr int kA16NonFinite = -1; // the column exponent of a column holding Inf or NaN

[[nodiscard]] inline std::uint32_t f32_bits(float x) noexcept {
    std::uint32_t u = 0;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
[[nodiscard]] inline float f32_from_bits(std::uint32_t u) noexcept {
    float x = 0.0F;
    std::memcpy(&x, &u, sizeof x);
    return x;
}
[[nodiscard]] inline float mul_rn(float a, float b) noexcept { return a * b; }
[[nodiscard]] inline float add_rn(float a, float b) noexcept { return a + b; }
[[nodiscard]] inline float div_rn(float a, float b) noexcept { return a / b; }
[[nodiscard]] inline float fma_rn(float a, float b, float c) noexcept { return std::fmaf(a, b, c); }

// Round-to-nearest-even conversions (host defaults; the reference relies on the same).
[[nodiscard]] inline float i64_to_f32_rn(std::int64_t v) noexcept { return static_cast<float>(v); }
[[nodiscard]] inline float rint_rn(float x) noexcept {
    const float magic = 12582912.0F; // 1.5 * 2^23
    return add_rn(add_rn(x, magic), -magic);
}
[[nodiscard]] inline float bf16_to_f32(std::uint16_t h) noexcept {
    return f32_from_bits(std::uint32_t{h} << 16);
}
// Every NaN becomes the canonical quiet NaN 0x7FC0 (payloads/signs differ between x86 and CUDA).
[[nodiscard]] inline std::uint16_t f32_to_bf16_rn(float x) noexcept {
    const std::uint32_t u = f32_bits(x);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) { return 0x7FC0U; }
    const std::uint32_t lsb = (u >> 16) & 1U;
    return static_cast<std::uint16_t>((u + 0x7FFFU + lsb) >> 16);
}

// Twice the E2M1 value of a 4-bit code, negated by bit 3 (host spelling of the reference's).
[[nodiscard]] inline int e2m1_x2(unsigned code) noexcept {
    const int mag = static_cast<int>((0x0C08060403020100ULL >> (8U * (code & 7U))) & 0xFFU);
    return (code & 8U) != 0 ? -mag : mag;
}
// E4M3FN value times 2^9 for a non-negative scale word (sign bit ignored): [0, 229376].
[[nodiscard]] inline std::int32_t e4m3_scaled(unsigned word) noexcept {
    const unsigned e = (word >> 3) & 15U;
    const unsigned m = word & 7U;
    return e == 0 ? static_cast<std::int32_t>(m)
                  : static_cast<std::int32_t>((8U + m) << (e - 1U));
}

// The same decode for all 256 scale bytes in one table: the 8-row-batch kernel's scale side turns
// its 16 per-block row scales into multipliers by lookup (same 1 KB and same idiom as q9_lut256
// above; the table is the function above, so the two can never drift apart).
[[nodiscard]] inline const std::int32_t* e4m3_scaled_lut256() noexcept {
    static const std::array<std::int32_t, 256> table = [] {
        std::array<std::int32_t, 256> out{};
        for (std::size_t b = 0; b < out.size(); ++b) {
            out[b] = e4m3_scaled(static_cast<unsigned>(b));
        }
        return out;
    }();
    return table.data();
}

// exp(x) with Cody-Waite reduction and a fixed degree-7 Taylor polynomial by explicit fma.
// Overflow -> +inf, deep underflow -> +0, NaN -> the canonical NaN. Deterministic on purpose:
// the CPU and the device must produce the same bits (the reference's own note).
[[nodiscard]] inline float exp_c(float x) noexcept {
    if (x != x) { return f32_from_bits(kCanonicalNan); }
    if (x > 88.72283935546875F) { return f32_from_bits(0x7F800000U); }
    if (x < -103.97208404541015625F) { return 0.0F; }
    const float n  = rint_rn(mul_rn(x, 1.44269502162933349609375F));
    const float r0 = fma_rn(-n, 0.693145751953125F, x);         // ln2 high part
    const float r  = fma_rn(-n, 1.428606765330187045e-06F, r0); // ln2 low part
    float p        = 1.0F / 5040.0F;
    p              = fma_rn(p, r, 1.0F / 720.0F);
    p              = fma_rn(p, r, 1.0F / 120.0F);
    p              = fma_rn(p, r, 1.0F / 24.0F);
    p              = fma_rn(p, r, 1.0F / 6.0F);
    p              = fma_rn(p, r, 0.5F);
    p              = fma_rn(p, r, 1.0F);
    p              = fma_rn(p, r, 1.0F);
    const int k    = static_cast<int>(n);
    const int k1   = k / 2;
    const int k2   = k - k1;
    const float s1 = f32_from_bits(static_cast<std::uint32_t>(k1 + 127) << 23);
    const float s2 = f32_from_bits(static_cast<std::uint32_t>(k2 + 127) << 23);
    return mul_rn(mul_rn(p, s1), s2);
}

// SiLU on binary32 in the form that never overflows exp; NaN -> canonical NaN.
[[nodiscard]] inline float silu_c(float g) noexcept {
    float r;
    if (g >= 0.0F) {
        r = div_rn(g, add_rn(1.0F, exp_c(-g)));
    } else {
        const float e = exp_c(g);
        r             = div_rn(mul_rn(g, e), add_rn(1.0F, e));
    }
    return r != r ? f32_from_bits(kCanonicalNan) : r;
}

// SiLU(g) * u on BF16 inputs, rounded to BF16.
[[nodiscard]] inline std::uint16_t swiglu_bf16(std::uint16_t gate_bf16, std::uint16_t up_bf16) noexcept {
    return f32_to_bf16_rn(mul_rn(silu_c(bf16_to_f32(gate_bf16)), bf16_to_f32(up_bf16)));
}

// ---- A16 encoding (design §16.2.1): a BF16 column is encoded once at its largest exponent ------

// emax of a BF16 vector: in [1, 254] (1 for an all-zero column), or kA16NonFinite.
[[nodiscard]] inline int a16_column_exponent(const std::uint16_t* v, int n) noexcept {
    unsigned key = 0; // the largest magnitude's bits: its exponent field is emax
    for (int j = 0; j < n; ++j) {
        const unsigned m = v[j] & 0x7FFFU;
        key              = m > key ? m : key;
    }
    if (key >= 0x7F80U) { return kA16NonFinite; }
    const int e = static_cast<int>(key >> 7);
    return e == 0 ? 1 : e;
}

// X of one finite element of a column with exponent emax.
[[nodiscard]] inline std::int32_t a16_value(std::uint16_t v, int emax) noexcept {
    const unsigned field  = (v >> 7) & 0xFFU;
    const std::uint32_t m = field == 0 ? (v & 0x7FU) : (0x80U | (v & 0x7FU));
    const int d           = emax - (field == 0 ? 1 : static_cast<int>(field));
    const std::uint32_t q = m << 13; // < 2^21
    std::uint32_t x;
    if (d == 0) {
        x = q;
    } else if (d >= 22) {
        x = 0; // q / 2^d < 1/2
    } else {
        x = (q + (1U << (d - 1)) - 1U + ((q >> d) & 1U)) >> d; // round half to even
    }
    return (v & 0x8000U) != 0 ? -static_cast<std::int32_t>(x) : static_cast<std::int32_t>(x);
}

// One encoded 16-element block: X = lo + 256 mid + 65536 hi (lo/mid unsigned low bytes of X's
// two's complement; hi = X >> 16 in [-32, 31]); the biased-code dot products subtract
// 12 * hi_sum.
struct A16Block {
    std::uint8_t lo[16];
    std::uint8_t mid[16];
    std::int8_t hi[16];
    std::int32_t hi_sum;
};

[[nodiscard]] inline A16Block a16_block(const std::uint16_t* v_bf16, int emax) noexcept {
    A16Block out{};
    if (emax == kA16NonFinite) { return out; }
    std::int32_t sum = 0;
    for (int j = 0; j < 16; ++j) {
        const std::int32_t x = a16_value(v_bf16[j], emax);
        const auto u         = static_cast<std::uint32_t>(x);
        out.lo[j]            = static_cast<std::uint8_t>(u & 0xFFU);
        out.mid[j]           = static_cast<std::uint8_t>((u >> 8) & 0xFFU);
        out.hi[j]            = static_cast<std::int8_t>(x >> 16);
        sum += out.hi[j];
    }
    out.hi_sum = sum;
    return out;
}

// X of element j of an encoded block.
[[nodiscard]] inline std::int32_t a16_x(const A16Block& b, int j) noexcept {
    return static_cast<std::int32_t>(b.lo[j]) + 256 * static_cast<std::int32_t>(b.mid[j]) +
           65536 * static_cast<std::int32_t>(b.hi[j]);
}

// y = bf16_rn((fl32_rn(S) * 2^(emax - 157)) * m), m = fl32(1 / weight global scale) -- this
// tree's divisor adapts as m. A non-finite column gives the canonical NaN.
[[nodiscard]] inline std::uint16_t a16_row_output(std::int64_t s, int emax, float multiplier) noexcept {
    if (emax == kA16NonFinite) { return 0x7FC0U; }
    const int k    = emax - 157; // in [-156, 97]
    const int k1   = k / 2;
    const int k2   = k - k1;
    const float s1 = f32_from_bits(static_cast<std::uint32_t>(k1 + 127) << 23);
    const float s2 = f32_from_bits(static_cast<std::uint32_t>(k2 + 127) << 23);
    return f32_to_bf16_rn(mul_rn(mul_rn(mul_rn(i64_to_f32_rn(s), s1), s2), multiplier));
}

// ---- weight code tables and ISA tiers -----------------------------------------------------------

// Code -> doubled E2M1 value as a signed byte, and biased by +12 as an unsigned byte.
alignas(16) inline constexpr std::int8_t kSignedCodes[16] = {0, 1, 2, 3, 4, 6, 8, 12,
                                                             0, -1, -2, -3, -4, -6, -8, -12};
alignas(16) inline constexpr std::int8_t kBiasedCodes[16] = {12, 13, 14, 15, 16, 18, 20, 24,
                                                             12, 11, 10, 9,  8,  6,  4,  0};
inline constexpr int kBias = 12;

// kAvx512Vnni is DELETED (this host has no AVX-512; a runtime-dispatched path must not exist).
enum class CpuIsa { kScalar, kAvx2, kAvxVnni };

[[nodiscard]] inline bool isa_supported(CpuIsa isa) noexcept {
    if (isa == CpuIsa::kScalar) { return true; }
    const IsaFlags f = isa_flags(); // MSVC's __cpuidex path; non-MSVC returns -1 (scalar only)
    if (isa == CpuIsa::kAvx2) { return f.avx2 == 1; }
    return f.avx_vnni == 1;
}
[[nodiscard]] inline CpuIsa best_isa() noexcept {
    if (isa_supported(CpuIsa::kAvxVnni)) { return CpuIsa::kAvxVnni; }
    if (isa_supported(CpuIsa::kAvx2)) { return CpuIsa::kAvx2; }
    return CpuIsa::kScalar;
}

// ---- geometry (same numbers as the reference's w4a4_expert.h and this tree's expert frame) ------

inline constexpr int kHidden         = 2560;
inline constexpr int kIntermediate   = 640;
inline constexpr int kGateUpBlocks   = kHidden / 16;                    // 160
inline constexpr int kDownBlocks     = kIntermediate / 16;              // 40
inline constexpr int kHBlocks        = kIntermediate / 16;              // 40
inline constexpr int kGateUpRowGroups = 2 * kIntermediate / 16;         // 80
inline constexpr int kDownRowGroups  = kHidden / 16;                    // 160
inline constexpr int kMaxColumns     = 8;                               // narrow route: n <= 8
inline constexpr std::size_t kGateUpBytes = static_cast<std::size_t>(kGateUpRowGroups)
                                            * kGateUpBlocks * 144;      // 1,843,200 (reference)
inline constexpr std::size_t kDownBytes = static_cast<std::size_t>(kDownRowGroups)
                                          * kDownBlocks * 144;          // 921,600

// ---- shared helpers -----------------------------------------------------------------------------

// Code nibble of row r (0..15) at k (0..15) inside one reference 144-byte unit (kept verbatim).
[[nodiscard]] inline unsigned unit_code(const std::uint8_t* unit, int r, int k) noexcept {
    const std::uint8_t byte = unit[32 * (k / 4) + 4 * (r % 8) + k % 4];
    return r < 8 ? (byte & 15U) : (byte >> 4);
}

// This tree's codes face: row-major byte = row*(C/2) + col/2, nibble = col & 1 (phi1).
// sab 2 swaps the column pair (diagnostic red-arm plumbing).
[[nodiscard]] inline unsigned ours_code(const std::uint8_t* codes, std::size_t code_row_bytes,
                                        int row, int col, std::int32_t sab) noexcept {
    const int c             = (sab == 2) ? (col ^ 1) : col;
    const std::uint8_t byte = codes[static_cast<std::size_t>(row) * code_row_bytes + (c >> 1)];
    return (c & 1) == 0 ? (byte & 15U) : (byte >> 4);
}

// gu record interleaves gate/up rows (even = gate_i, odd = up_i); this tree splits gate [0:I) and
// up [I:2I). their_row -> this tree's row (phi3 inverse).
[[nodiscard]] inline int ported_row_gu(int their_row, int intermediate) noexcept {
    return (their_row & 1) == 0 ? (their_row >> 1) : (intermediate + ((their_row - 1) >> 1));
}

// Software prefetch along the unit stream (reference prefetch_unit, kept for the original arm).
inline void prefetch_unit(const std::uint8_t* unit, int distance) noexcept {
    if (distance > 0) {
        const char* p = reinterpret_cast<const char*>(unit) + distance;
        _mm_prefetch(p, _MM_HINT_T0);
        _mm_prefetch(p + 64, _MM_HINT_T0);
        _mm_prefetch(p + 128, _MM_HINT_T0);
    }
}

#if defined(__x86_64__) || defined(_M_X64)

[[nodiscard]] inline __m256i dot4_avx2(__m256i acc, __m256i u, __m256i s) noexcept {
    const __m256i pairs = _mm256_maddubs_epi16(u, s); // |pair| <= 2*255*12 or 2*24*32: no saturation
    return _mm256_add_epi32(acc, _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)));
}
[[nodiscard]] inline __m256i dot4_avxvnni(__m256i acc, __m256i u, __m256i s) noexcept {
    return _mm256_dpbusd_avx_epi32(acc, u, s);
}
[[nodiscard]] inline __m128i dot4_avx2_128(__m128i acc, __m128i u, __m128i s) noexcept {
    const __m128i pairs = _mm_maddubs_epi16(u, s); // same bound as the 256-bit form: no saturation
    return _mm_add_epi32(acc, _mm_madd_epi16(pairs, _mm_set1_epi16(1)));
}
[[nodiscard]] inline std::int32_t hsum4_epi32(__m128i v) noexcept {
    const __m128i h = _mm_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(h, h));
}
// 8 packed bytes -> 16 raw codes (low nibble of byte j = k 2j), one byte per code.
[[nodiscard]] inline __m128i expand8_codes(const std::uint8_t* p) noexcept {
    const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    const __m128i lo  = _mm_and_si128(raw, _mm_set1_epi8(0x0F));
    const __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F));
    return _mm_unpacklo_epi8(lo, hi);
}

#endif // x86

// ---- arm ORIGINAL (unit order): the reference's access pattern, for the port probe only ---------
// Values computed on THIS tree's bytes are meaningless here (the layout differs); the arm exists
// to measure the reference access pattern's throughput on the same records. Kept verbatim from the
// reference (minus the deleted kAvx512Vnni tier and the target attributes MSVC does not need).

inline void rs_orig_scalar(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end,
                           const A16Block* const* acts, int ncols, std::int64_t* out) {
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        for (int c = 0; c < ncols; ++c) {
            std::int64_t s[16] = {};
            for (int b = 0; b < blocks; ++b) {
                const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * 144;
                const A16Block& a        = acts[c][b];
                for (int r = 0; r < 16; ++r) {
                    std::int32_t p = 0;
                    for (int k = 0; k < 16; ++k) { p += e2m1_x2(unit_code(unit, r, k)) * a16_x(a, k); }
                    s[r] += static_cast<std::int64_t>(p) * e4m3_scaled(unit[128 + r]);
                }
            }
            for (int r = 0; r < 16; ++r) {
                out[(static_cast<std::size_t>(rg - rg_begin) * 16 + r) * ncols + c] = s[r];
            }
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)

// One body, two dot products: AVX-VNNI must not leak into the plain AVX2 variant (reference note).
#define INFERNIX_AVX2_ROW_SUMS_ORIG(NAME, DOT) \
inline void NAME(const std::uint8_t* matrix, int blocks, int rg_begin, int rg_end, \
                 const A16Block* const* acts, int ncols, std::int64_t* out, int prefetch_bytes) { \
    const __m256i lut_s = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kSignedCodes))); \
    const __m256i lut_b = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes))); \
    const __m256i low4  = _mm256_set1_epi8(0x0F); \
    for (int rg = rg_begin; rg < rg_end; ++rg) { \
        __m256i even[2][kMaxColumns], odd[2][kMaxColumns]; \
        for (int h = 0; h < 2; ++h) { \
            for (int c = 0; c < ncols; ++c) { even[h][c] = odd[h][c] = _mm256_setzero_si256(); } \
        } \
        for (int b = 0; b < blocks; ++b) { \
            const std::uint8_t* unit = matrix + (static_cast<std::size_t>(rg) * blocks + b) * 144; \
            prefetch_unit(unit, prefetch_bytes); \
            __m256i plo[2][kMaxColumns], pmid[2][kMaxColumns], phi[2][kMaxColumns]; \
            for (int h = 0; h < 2; ++h) { \
                for (int c = 0; c < ncols; ++c) { plo[h][c] = pmid[h][c] = phi[h][c] = _mm256_setzero_si256(); } \
            } \
            for (int q = 0; q < 4; ++q) { \
                const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(unit + 32 * q)); \
                const __m256i idx[2] = {_mm256_and_si256(v, low4), \
                                        _mm256_and_si256(_mm256_srli_epi16(v, 4), low4)}; \
                for (int h = 0; h < 2; ++h) { \
                    const __m256i ws = _mm256_shuffle_epi8(lut_s, idx[h]); \
                    const __m256i wb = _mm256_shuffle_epi8(lut_b, idx[h]); \
                    for (int c = 0; c < ncols; ++c) { \
                        const A16Block& a = acts[c][b]; \
                        std::int32_t qlo, qmid, qhi; \
                        std::memcpy(&qlo, &a.lo[4 * q], 4); \
                        std::memcpy(&qmid, &a.mid[4 * q], 4); \
                        std::memcpy(&qhi, &a.hi[4 * q], 4); \
                        plo[h][c]  = DOT(plo[h][c], _mm256_set1_epi32(qlo), ws); \
                        pmid[h][c] = DOT(pmid[h][c], _mm256_set1_epi32(qmid), ws); \
                        phi[h][c]  = DOT(phi[h][c], wb, _mm256_set1_epi32(qhi)); \
                    } \
                } \
            } \
            for (int h = 0; h < 2; ++h) { \
                const __m256i sw     = scales8_avx2_impl(unit + 128 + 8 * h); \
                const __m256i sw_odd = _mm256_srli_epi64(sw, 32); \
                for (int c = 0; c < ncols; ++c) { \
                    const __m256i hi = _mm256_sub_epi32(phi[h][c], _mm256_set1_epi32(kBias * acts[c][b].hi_sum)); \
                    const __m256i p  = _mm256_add_epi32(_mm256_add_epi32(plo[h][c], _mm256_slli_epi32(pmid[h][c], 8)), \
                                                        _mm256_slli_epi32(hi, 16)); \
                    even[h][c] = _mm256_add_epi64(even[h][c], _mm256_mul_epi32(p, sw)); \
                    odd[h][c]  = _mm256_add_epi64(odd[h][c], _mm256_mul_epi32(_mm256_srli_epi64(p, 32), sw_odd)); \
                } \
            } \
        } \
        for (int h = 0; h < 2; ++h) { \
            for (int c = 0; c < ncols; ++c) { \
                alignas(32) std::int64_t e[4], o[4]; \
                _mm256_store_si256(reinterpret_cast<__m256i*>(e), even[h][c]); \
                _mm256_store_si256(reinterpret_cast<__m256i*>(o), odd[h][c]); \
                for (int i = 0; i < 4; ++i) { \
                    const std::size_t base = static_cast<std::size_t>(rg - rg_begin) * 16 + 8 * h; \
                    out[(base + 2 * i) * ncols + c]     = e[i]; \
                    out[(base + 2 * i + 1) * ncols + c] = o[i]; \
                } \
            } \
        } \
    } \
}

// The reference's scales8_avx2 (8 scale bytes -> 8 x int32 e4m3_scaled values).
[[nodiscard]] inline __m256i scales8_avx2_impl(const std::uint8_t* s) noexcept {
    const __m256i w    = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s)));
    const __m256i m    = _mm256_and_si256(w, _mm256_set1_epi32(7));
    const __m256i e    = _mm256_and_si256(_mm256_srli_epi32(w, 3), _mm256_set1_epi32(15));
    const __m256i norm = _mm256_sllv_epi32(_mm256_or_si256(m, _mm256_set1_epi32(8)),
                                           _mm256_sub_epi32(e, _mm256_set1_epi32(1)));
    const __m256i sub  = _mm256_cmpeq_epi32(e, _mm256_setzero_si256());
    return _mm256_blendv_epi8(norm, m, sub);
}

INFERNIX_AVX2_ROW_SUMS_ORIG(rs_orig_avx2, dot4_avx2)
INFERNIX_AVX2_ROW_SUMS_ORIG(rs_orig_avxvnni, dot4_avxvnni)

#undef INFERNIX_AVX2_ROW_SUMS_ORIG
#endif // x86

// ---- arm OURS (layout-adapted): row-major K16M128x4 access, same arithmetic ----------------------
// Reads THIS tree's faces through phi1/phi2/phi3; same lo/mid/hi split, same dpbusd direction and
// the same -12*hi_sum correction as the reference. `sab` is the diagnostic red-arm plumbing (see
// the section header); `pf_rows` prefetches the same block `pf_rows` rows ahead (0 = off).

inline void rs_ours_scalar(bool gu, int intermediate, const std::uint8_t* codes,
                           std::size_t code_row_bytes, const std::uint8_t* scales,
                           int scale_tiles_per_row, int blocks, int rg_begin, int rg_end,
                           const A16Block* const* acts, int ncols, std::int64_t* out,
                           std::int32_t sab) {
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        for (int c = 0; c < ncols; ++c) {
            std::int64_t s[16] = {};
            for (int b = 0; b < blocks; ++b) {
                const A16Block& a = acts[c][b];
                for (int r = 0; r < 16; ++r) {
                    const int row = gu ? ported_row_gu(rg * 16 + r, intermediate) : rg * 16 + r;
                    std::int32_t p = 0;
                    for (int k = 0; k < 16; ++k) {
                        p += e2m1_x2(ours_code(codes, code_row_bytes, row, b * 16 + k, sab)) *
                             a16_x(a, k);
                    }
                    if (sab == 3) { // simulate the skipped correction (same wrong value as the SIMD arm)
                        std::int32_t hs = 0;
                        for (int k = 0; k < 16; ++k) { hs += a.hi[k]; }
                        p += (kBias * hs) << 16;
                    }
                    const std::int32_t sw =
                        e4m3_scaled(scales[k16m128x4_scale_byte(row, b, scale_tiles_per_row)]);
                    s[r] += static_cast<std::int64_t>(p) * sw;
                }
            }
            for (int r = 0; r < 16; ++r) {
                const int dst = (sab == 1) ? ((r + 1) & 15) : r; // sab 1: output row shift
                out[(static_cast<std::size_t>(rg - rg_begin) * 16 + dst) * ncols + c] = s[r];
            }
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)

// The round-1 AVX2 arm (one 128-bit dot product per metric; no leakage into other tiers). It and
// the scalar arm above stay as-is: both are the untouched correctness references of the Plan A
// rewrite (only the AVX-VNNI tier below is rebuilt).
#define INFERNIX_ROW_SUMS_OURS(NAME, DOT128) \
inline void NAME(bool gu, int intermediate, const std::uint8_t* codes, std::size_t code_row_bytes, \
                 const std::uint8_t* scales, int scale_tiles_per_row, int blocks, int rg_begin, \
                 int rg_end, const A16Block* const* acts, int ncols, std::int64_t* out, \
                 std::int32_t sab, int pf_rows) { \
    const __m128i lut_s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kSignedCodes)); \
    const __m128i lut_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes)); \
    for (int rg = rg_begin; rg < rg_end; ++rg) { \
        std::int64_t s[16 * kMaxColumns]; \
        for (int i = 0; i < 16 * kMaxColumns; ++i) { s[i] = 0; } \
        for (int b = 0; b < blocks; ++b) { \
            for (int r = 0; r < 16; ++r) { \
                const int their = rg * 16 + r; \
                const int row   = gu ? ported_row_gu(their, intermediate) : their; \
                const std::uint8_t* src = codes + static_cast<std::size_t>(row) * code_row_bytes + \
                                          static_cast<std::size_t>(b) * 8; \
                if (pf_rows > 0 && r + pf_rows < 16) { \
                    const int prow = gu ? ported_row_gu(their + pf_rows, intermediate) : (their + pf_rows); \
                    _mm_prefetch(reinterpret_cast<const char*>(codes + \
                                     static_cast<std::size_t>(prow) * code_row_bytes + \
                                     static_cast<std::size_t>(b) * 8), _MM_HINT_T0); \
                } \
                __m128i w4 = expand8_codes(src); \
                if (sab == 2) { /* column-pair swap (diagnostic) */ \
                    w4 = _mm_shuffle_epi8(w4, _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, \
                                                            13, 12, 15, 14)); \
                } \
                const __m128i ws = _mm_shuffle_epi8(lut_s, w4); \
                const __m128i wb = _mm_shuffle_epi8(lut_b, w4); \
                const std::int32_t sc = \
                    e4m3_scaled(scales[k16m128x4_scale_byte(row, b, scale_tiles_per_row)]); \
                for (int c = 0; c < ncols; ++c) { \
                    const A16Block& a  = acts[c][b]; \
                    const __m128i alo  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.lo)); \
                    const __m128i amid = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.mid)); \
                    const __m128i ahi  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.hi)); \
                    const std::int32_t lo_v  = hsum4_epi32(DOT128(_mm_setzero_si128(), alo, ws)); \
                    const std::int32_t mid_v = hsum4_epi32(DOT128(_mm_setzero_si128(), amid, ws)); \
                    std::int32_t hi_v        = hsum4_epi32(DOT128(_mm_setzero_si128(), wb, ahi)); \
                    if (sab != 3) { hi_v -= kBias * a.hi_sum; } /* sab 3: skip the correction */ \
                    const std::int32_t p = lo_v + (mid_v << 8) + (hi_v << 16); \
                    s[r * kMaxColumns + c] += static_cast<std::int64_t>(p) * sc; \
                } \
            } \
        } \
        for (int r = 0; r < 16; ++r) { \
            const int dst = (sab == 1) ? ((r + 1) & 15) : r; \
            for (int c = 0; c < ncols; ++c) { \
                out[(static_cast<std::size_t>(rg - rg_begin) * 16 + dst) * ncols + c] = \
                    s[r * kMaxColumns + c]; \
            } \
        } \
    } \
}

INFERNIX_ROW_SUMS_OURS(rs_ours_avx2, dot4_avx2_128)

#undef INFERNIX_ROW_SUMS_OURS

// Plan A (block-4 P1 round 2): the AVX-VNNI arm rebuilt as an 8-row batch. Four tightenings, per
// the pre-registered spec (期Ⅳ-P1-6.7x回审 §4):
//   ① transposed load in 8-row batches: each batch's eight 8-byte code words are expanded and
//      transposed so that a 256-bit "quarter" vector's byte 4r+j is code(row r, col 4q+j);
//   ② lanes are ROWS after that transpose -- the four quarters chain straight into one dpbusd
//      accumulator per metric, so there is no hsum serial chain at all (the round-1 arm paid three
//      hadds per metric, six per row and column);
//   ③ the scale side fetches all 16 rows' scale bytes once per block and decodes through the
//      256-entry e4m3_scaled table (no per-row index arithmetic, no per-row branchy decode);
//   ④ 256-bit width throughout (the round-1 arm was 128-bit).
//
// Scale stride, why +16 per row: phi2's index for row R at block b is
//   (R/128)*tiles*512 + (b/4)*512 + ((R%128)&31)*16 + ((R%128)>>5)*4 + (b&3).
// Every 8-row run this kernel reads starts on a multiple of 8 (gate rg*8, up 640+rg*8, down
// rg*16), so (start%128)&31 is in {0,8,16,24} and start+7 -- or +15 for the down face, where
// start%32 is in {0,16} -- never crosses a 32-row sub-block: the m-tile and sub-block terms stay
// constant and the index moves by exactly +16 per row. One base per run (mg, the index of its
// first row at b=0) plus tb = 512*(b/4) + (b&3) reaches every row in it: base + 16*k, k < 8.
// The gu face has two runs per group; its lanes interleave gate/up exactly as ported_row_gu maps
// rg*16 + li (even lane -> gate run, odd lane -> up run), so one 16-entry fill covers a block.
//
// Correctness against the untouched scalar and AVX2 arms: same products, same
// lo + 256*mid + 65536*(hi - 12*hi_sum) split, same -12*hi_sum correction (skipped under sab 3),
// same int64 accumulation -- dpbusd and int64 adds are exact, and the writeback maps
// lane -> their-row -> the same out[(rg-rg_begin)*16 + dst]*ncols + c cells the scalar arm fills.
// Sab modes keep their round-1 semantics: 1 = output row shift, 2 = code pair swap (applied to
// each expanded row before the transpose), 3 = skip the correction.
// (Structure follows the reference's row-lane dpbusd organization; the implementation is this
// tree's own, under the license block above.)
inline void rs_ours_avxvnni(bool gu, int intermediate, const std::uint8_t* codes,
                            std::size_t code_row_bytes, const std::uint8_t* scales,
                            int scale_tiles_per_row, int blocks, int rg_begin, int rg_end,
                            const A16Block* const* acts, int ncols, std::int64_t* out,
                            std::int32_t sab, int pf_rows) {
    const __m256i lut_s = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(kSignedCodes)));
    const __m256i lut_b = _mm256_broadcastsi128_si256(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(kBiasedCodes)));
    const __m128i swap2 = _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
    const std::int32_t* const scale_lut = e4m3_scaled_lut256();
    const auto join = [](__m128i lo, __m128i hi) {
        return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    };
    const auto word_at = [](const void* p) {
        std::int32_t v = 0;
        std::memcpy(&v, p, 4);
        return v;
    };
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        const int r0 = rg * 16;
        // Starts of the two 8-row weight runs this group reads (gate/up on the gu face; the single
        // row_g run on the down face). Both are multiples of 8 -- the stride-16 property above.
        int row_g = r0;
        int row_u = r0 + 1;
        if (gu) {
            row_g = ported_row_gu(r0, intermediate);
            row_u = ported_row_gu(r0 + 1, intermediate);
        }
        const std::uint64_t mg = k16m128x4_scale_byte(static_cast<std::uint64_t>(row_g), 0,
                                                      static_cast<std::uint64_t>(scale_tiles_per_row));
        const std::uint64_t mu = k16m128x4_scale_byte(static_cast<std::uint64_t>(row_u), 0,
                                                      static_cast<std::uint64_t>(scale_tiles_per_row));
        for (int c = 0; c < ncols; ++c) {
            const A16Block* const ac = acts[c];
            __m256i acc_e0 = _mm256_setzero_si256(); // their rows 0, 2, 4, 6
            __m256i acc_o0 = _mm256_setzero_si256(); // their rows 1, 3, 5, 7
            __m256i acc_e1 = _mm256_setzero_si256(); // their rows 8, 10, 12, 14
            __m256i acc_o1 = _mm256_setzero_si256(); // their rows 9, 11, 13, 15
            for (int b = 0; b < blocks; ++b) {
                // Scale side: 16 rows' values for this block, fetched once, decoded by table.
                const std::uint64_t tb = 512ULL * static_cast<std::uint64_t>(b >> 2) +
                                         static_cast<std::uint64_t>(b & 3);
                std::int32_t sv[16];
                if (gu) {
                    const std::uint8_t* const sg = scales + mg + tb;
                    const std::uint8_t* const su = scales + mu + tb;
                    for (int li = 0; li < 16; ++li) { // even lane -> gate run, odd -> up run
                        const std::uint8_t byte =
                            (li & 1) == 0 ? sg[16 * (li >> 1)] : su[16 * (li >> 1)];
                        sv[li] = scale_lut[byte];
                    }
                } else {
                    const std::uint8_t* const sd = scales + mg + tb;
                    for (int li = 0; li < 16; ++li) { sv[li] = scale_lut[sd[16 * li]]; }
                }
                const __m256i sc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sv));
                const __m256i sc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sv + 8));
                const A16Block& a = ac[b];
                for (int batch = 0; batch < 2; ++batch) {
                    const int t0 = 8 * batch;
                    const auto row_src = [&](int t) {
                        const int row = gu ? ported_row_gu(r0 + t, intermediate) : (r0 + t);
                        return codes + static_cast<std::size_t>(row) * code_row_bytes +
                               static_cast<std::size_t>(b) * 8;
                    };
                    if (pf_rows > 0 && b + pf_rows < blocks) {
                        for (int i = 0; i < 8; ++i) {
                            const int t   = t0 + i;
                            const int row = gu ? ported_row_gu(r0 + t, intermediate) : (r0 + t);
                            _mm_prefetch(reinterpret_cast<const char*>(
                                             codes + static_cast<std::size_t>(row) * code_row_bytes +
                                             static_cast<std::size_t>(b + pf_rows) * 8),
                                         _MM_HINT_T0);
                        }
                    }
                    __m128i e0 = expand8_codes(row_src(t0 + 0));
                    __m128i e1 = expand8_codes(row_src(t0 + 1));
                    __m128i e2 = expand8_codes(row_src(t0 + 2));
                    __m128i e3 = expand8_codes(row_src(t0 + 3));
                    __m128i e4 = expand8_codes(row_src(t0 + 4));
                    __m128i e5 = expand8_codes(row_src(t0 + 5));
                    __m128i e6 = expand8_codes(row_src(t0 + 6));
                    __m128i e7 = expand8_codes(row_src(t0 + 7));
                    if (sab == 2) { // column-pair swap, per row, before the transpose
                        e0 = _mm_shuffle_epi8(e0, swap2);
                        e1 = _mm_shuffle_epi8(e1, swap2);
                        e2 = _mm_shuffle_epi8(e2, swap2);
                        e3 = _mm_shuffle_epi8(e3, swap2);
                        e4 = _mm_shuffle_epi8(e4, swap2);
                        e5 = _mm_shuffle_epi8(e5, swap2);
                        e6 = _mm_shuffle_epi8(e6, swap2);
                        e7 = _mm_shuffle_epi8(e7, swap2);
                    }
                    // 8x16 byte transpose: pair dword interleaves (L = d0/d1, H = d2/d3), then the
                    // quarter halves join. Byte 4r+j of Qq = code(row r, col 4q+j).
                    const __m128i l01 = _mm_unpacklo_epi32(e0, e1);
                    const __m128i l23 = _mm_unpacklo_epi32(e2, e3);
                    const __m128i l45 = _mm_unpacklo_epi32(e4, e5);
                    const __m128i l67 = _mm_unpacklo_epi32(e6, e7);
                    const __m128i h01 = _mm_unpackhi_epi32(e0, e1);
                    const __m128i h23 = _mm_unpackhi_epi32(e2, e3);
                    const __m128i h45 = _mm_unpackhi_epi32(e4, e5);
                    const __m128i h67 = _mm_unpackhi_epi32(e6, e7);
                    const __m256i q0 = join(_mm_unpacklo_epi64(l01, l23), _mm_unpacklo_epi64(l45, l67));
                    const __m256i q1 = join(_mm_unpackhi_epi64(l01, l23), _mm_unpackhi_epi64(l45, l67));
                    const __m256i q2 = join(_mm_unpacklo_epi64(h01, h23), _mm_unpacklo_epi64(h45, h67));
                    const __m256i q3 = join(_mm_unpackhi_epi64(h01, h23), _mm_unpackhi_epi64(h45, h67));
                    __m256i plo  = _mm256_setzero_si256();
                    __m256i pmid = _mm256_setzero_si256();
                    __m256i phi  = _mm256_setzero_si256();
                    const __m256i ws0 = _mm256_shuffle_epi8(lut_s, q0);
                    const __m256i wb0 = _mm256_shuffle_epi8(lut_b, q0);
                    plo  = _mm256_dpbusd_avx_epi32(plo, _mm256_set1_epi32(word_at(a.lo)), ws0);
                    pmid = _mm256_dpbusd_avx_epi32(pmid, _mm256_set1_epi32(word_at(a.mid)), ws0);
                    phi  = _mm256_dpbusd_avx_epi32(phi, wb0, _mm256_set1_epi32(word_at(a.hi)));
                    const __m256i ws1 = _mm256_shuffle_epi8(lut_s, q1);
                    const __m256i wb1 = _mm256_shuffle_epi8(lut_b, q1);
                    plo  = _mm256_dpbusd_avx_epi32(plo, _mm256_set1_epi32(word_at(a.lo + 4)), ws1);
                    pmid = _mm256_dpbusd_avx_epi32(pmid, _mm256_set1_epi32(word_at(a.mid + 4)), ws1);
                    phi  = _mm256_dpbusd_avx_epi32(phi, wb1, _mm256_set1_epi32(word_at(a.hi + 4)));
                    const __m256i ws2 = _mm256_shuffle_epi8(lut_s, q2);
                    const __m256i wb2 = _mm256_shuffle_epi8(lut_b, q2);
                    plo  = _mm256_dpbusd_avx_epi32(plo, _mm256_set1_epi32(word_at(a.lo + 8)), ws2);
                    pmid = _mm256_dpbusd_avx_epi32(pmid, _mm256_set1_epi32(word_at(a.mid + 8)), ws2);
                    phi  = _mm256_dpbusd_avx_epi32(phi, wb2, _mm256_set1_epi32(word_at(a.hi + 8)));
                    const __m256i ws3 = _mm256_shuffle_epi8(lut_s, q3);
                    const __m256i wb3 = _mm256_shuffle_epi8(lut_b, q3);
                    plo  = _mm256_dpbusd_avx_epi32(plo, _mm256_set1_epi32(word_at(a.lo + 12)), ws3);
                    pmid = _mm256_dpbusd_avx_epi32(pmid, _mm256_set1_epi32(word_at(a.mid + 12)), ws3);
                    phi  = _mm256_dpbusd_avx_epi32(phi, wb3, _mm256_set1_epi32(word_at(a.hi + 12)));
                    if (sab != 3) { // sab 3: skip the biased-code correction (round-1 semantics)
                        phi = _mm256_sub_epi32(phi, _mm256_set1_epi32(kBias * a.hi_sum));
                    }
                    const __m256i p = _mm256_add_epi32(
                        _mm256_add_epi32(plo, _mm256_slli_epi32(pmid, 8)),
                        _mm256_slli_epi32(phi, 16));
                    const __m256i scv = (batch == 0) ? sc0 : sc1;
                    const __m256i ev = _mm256_mul_epi32(p, scv); // even lanes (i64)
                    const __m256i ov = _mm256_mul_epi32(_mm256_srli_epi64(p, 32),
                                                        _mm256_srli_epi64(scv, 32)); // odd lanes
                    if (batch == 0) {
                        acc_e0 = _mm256_add_epi64(acc_e0, ev);
                        acc_o0 = _mm256_add_epi64(acc_o0, ov);
                    } else {
                        acc_e1 = _mm256_add_epi64(acc_e1, ev);
                        acc_o1 = _mm256_add_epi64(acc_o1, ov);
                    }
                }
            }
            { // writeback: int64 lanes -> out[(rg-rg_begin)*16 + their row][c]
                const std::size_t base = static_cast<std::size_t>(rg - rg_begin) * 16;
                alignas(32) std::int64_t we[4], wo[4];
                _mm256_store_si256(reinterpret_cast<__m256i*>(we), acc_e0);
                _mm256_store_si256(reinterpret_cast<__m256i*>(wo), acc_o0);
                for (int j = 0; j < 4; ++j) {
                    const int t0 = 2 * j, t1 = 2 * j + 1;
                    out[(base + ((sab == 1) ? ((t0 + 1) & 15) : t0)) * ncols + c] = we[j];
                    out[(base + ((sab == 1) ? ((t1 + 1) & 15) : t1)) * ncols + c] = wo[j];
                }
                _mm256_store_si256(reinterpret_cast<__m256i*>(we), acc_e1);
                _mm256_store_si256(reinterpret_cast<__m256i*>(wo), acc_o1);
                for (int j = 0; j < 4; ++j) {
                    const int t0 = 8 + 2 * j, t1 = 8 + 2 * j + 1;
                    out[(base + ((sab == 1) ? ((t0 + 1) & 15) : t0)) * ncols + c] = we[j];
                    out[(base + ((sab == 1) ? ((t1 + 1) & 15) : t1)) * ncols + c] = wo[j];
                }
            }
        }
    }
}
#endif // x86

// ---- top level: dispatch, A16 encode, gate/up units, down rows, whole expert --------------------

// Dispatch for the layout-adapted arm. kAvx512Vnni is not a case: the tier does not exist here.
inline void rg_row_sums_ours(CpuIsa isa, bool gu, int intermediate, const std::uint8_t* codes,
                             std::size_t code_row_bytes, const std::uint8_t* scales,
                             int scale_tiles_per_row, int blocks, int rg_begin, int rg_end,
                             const A16Block* const* acts, int ncols, std::int64_t* out,
                             std::int32_t sab, int pf_rows) {
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == CpuIsa::kAvxVnni) {
        rs_ours_avxvnni(gu, intermediate, codes, code_row_bytes, scales, scale_tiles_per_row,
                        blocks, rg_begin, rg_end, acts, ncols, out, sab, pf_rows);
        return;
    }
    if (isa == CpuIsa::kAvx2) {
        rs_ours_avx2(gu, intermediate, codes, code_row_bytes, scales, scale_tiles_per_row, blocks,
                     rg_begin, rg_end, acts, ncols, out, sab, pf_rows);
        return;
    }
#else
    (void)isa;
#endif
    rs_ours_scalar(gu, intermediate, codes, code_row_bytes, scales, scale_tiles_per_row, blocks,
                   rg_begin, rg_end, acts, ncols, out, sab);
}

// Dispatch for the original arm (throughput reference only).
inline void rg_row_sums_orig(CpuIsa isa, const std::uint8_t* matrix, int blocks, int rg_begin,
                             int rg_end, const A16Block* const* acts, int ncols,
                             std::int64_t* out, int prefetch_bytes) {
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == CpuIsa::kAvxVnni) {
        rs_orig_avxvnni(matrix, blocks, rg_begin, rg_end, acts, ncols, out, prefetch_bytes);
        return;
    }
    if (isa == CpuIsa::kAvx2) {
        rs_orig_avx2(matrix, blocks, rg_begin, rg_end, acts, ncols, out, prefetch_bytes);
        return;
    }
#else
    (void)isa; (void)prefetch_bytes;
#endif
    rs_orig_scalar(matrix, blocks, rg_begin, rg_end, acts, ncols, out);
}

// Encodes blocks [block_begin, block_end) of a column whose exponent is already known.
inline void encode_a16_blocks(const std::uint16_t* v, int emax, int block_begin, int block_end,
                              A16Block* out) noexcept {
    for (int b = block_begin; b < block_end; ++b) { out[b] = a16_block(v + 16 * b, emax); }
}

// Phase A: gate/up units [unit_begin, unit_end) + SwiGLU + h in BF16. gu divisor -> multiplier
// (adaptation: this tree divides by every matrix's divisor; the reference multiplies by m).
inline void gate_up_units_ours(CpuIsa isa, const std::uint8_t* gu_codes, std::size_t gu_row_bytes,
                               const std::uint8_t* gu_scales, float alpha_gate, float alpha_up,
                               const A16Block* const* x, const int* x_exp, int ncols,
                               int unit_begin, int unit_end, std::uint16_t* const* h,
                               std::int32_t sab, int pf_rows) {
    std::int64_t s[32 * kMaxColumns]; // one unit's sums: no allocation on the miss path
    for (int u = unit_begin; u < unit_end; ++u) {
        // Unit u is row groups 2u, 2u+1: 32 interleaved rows, gate_i at even rows, up_i at odd.
        rg_row_sums_ours(isa, /*gu=*/true, kIntermediate, gu_codes, gu_row_bytes, gu_scales,
                         kHidden / 64, kGateUpBlocks, 2 * u, 2 * u + 2, x, ncols, s, sab, pf_rows);
        for (int c = 0; c < ncols; ++c) {
            for (int i = 0; i < 16; ++i) {
                const std::uint16_t g = a16_row_output(
                    s[static_cast<std::size_t>(2 * i) * ncols + c], x_exp[c], alpha_gate);
                const std::uint16_t p = a16_row_output(
                    s[static_cast<std::size_t>(2 * i + 1) * ncols + c], x_exp[c], alpha_up);
                h[c][16 * u + i] = swiglu_bf16(g, p);
            }
        }
    }
}

// Phase B: down row groups [rg_begin, rg_end) -> y in BF16.
inline void down_rows_ours(CpuIsa isa, const std::uint8_t* dn_codes, std::size_t dn_row_bytes,
                           const std::uint8_t* dn_scales, float alpha_down,
                           const A16Block* const* h, const int* h_exp, int ncols, int rg_begin,
                           int rg_end, std::uint16_t* const* y, std::int32_t sab, int pf_rows) {
    std::int64_t s[16 * kMaxColumns]; // one row group at a time
    for (int rg = rg_begin; rg < rg_end; ++rg) {
        rg_row_sums_ours(isa, /*gu=*/false, 0, dn_codes, dn_row_bytes, dn_scales, kIntermediate / 64,
                         kDownBlocks, rg, rg + 1, h, ncols, s, sab, pf_rows);
        for (int r = 0; r < 16; ++r) {
            for (int c = 0; c < ncols; ++c) {
                y[c][rg * 16 + r] = a16_row_output(s[r * ncols + c], h_exp[c], alpha_down);
            }
        }
    }
}

// Whole expert on one thread: y[col][0..2560) from x[col][0..2560), both BF16 (ncols in [1, 8]).
// Returns false when the preconditions fail (the caller turns that into a failed instrument).
inline bool ported_forward_a16(CpuIsa isa, const std::uint8_t* gu_codes, std::size_t gu_row_bytes,
                               const std::uint8_t* gu_scales, float gu_divisor,
                               const std::uint8_t* dn_codes, std::size_t dn_row_bytes,
                               const std::uint8_t* dn_scales, float dn_divisor, int ncols,
                               const std::uint16_t* const* x, std::uint16_t* const* y,
                               std::int32_t sab, int pf_rows) {
    if (ncols < 1 || ncols > kMaxColumns) { return false; }
    if (!(gu_divisor > 0.0F) || !(dn_divisor > 0.0F)) { return false; }
    if (gu_codes == nullptr || gu_scales == nullptr || dn_codes == nullptr ||
        dn_scales == nullptr || x == nullptr || y == nullptr) {
        return false;
    }
    sab = (sab >= 0 && sab <= 3) ? sab : 0;
    const float alpha_gate = 1.0F / gu_divisor; // divisor -> multiplier (adaptation 5)
    const float alpha_up   = alpha_gate;
    const float alpha_down = 1.0F / dn_divisor;
    std::vector<A16Block> xb(static_cast<std::size_t>(ncols) * kGateUpBlocks);
    std::vector<A16Block> hb(static_cast<std::size_t>(ncols) * kHBlocks);
    std::vector<std::uint16_t> hv(static_cast<std::size_t>(ncols) * kIntermediate);
    const A16Block* x_ptr[kMaxColumns];
    const A16Block* h_ptr[kMaxColumns];
    std::uint16_t* hv_ptr[kMaxColumns];
    int x_exp[kMaxColumns], h_exp[kMaxColumns];
    for (int c = 0; c < ncols; ++c) {
        x_exp[c] = a16_column_exponent(x[c], kHidden);
        encode_a16_blocks(x[c], x_exp[c], 0, kGateUpBlocks,
                          &xb[static_cast<std::size_t>(c) * kGateUpBlocks]);
        x_ptr[c]  = &xb[static_cast<std::size_t>(c) * kGateUpBlocks];
        hv_ptr[c] = &hv[static_cast<std::size_t>(c) * kIntermediate];
        h_ptr[c]  = &hb[static_cast<std::size_t>(c) * kHBlocks];
    }
    gate_up_units_ours(isa, gu_codes, gu_row_bytes, gu_scales, alpha_gate, alpha_up, x_ptr, x_exp,
                       ncols, 0, kHBlocks, hv_ptr, sab, pf_rows);
    for (int c = 0; c < ncols; ++c) {
        h_exp[c] = a16_column_exponent(hv_ptr[c], kIntermediate);
        encode_a16_blocks(hv_ptr[c], h_exp[c], 0, kHBlocks,
                          &hb[static_cast<std::size_t>(c) * kHBlocks]);
    }
    down_rows_ours(isa, dn_codes, dn_row_bytes, dn_scales, alpha_down, h_ptr, h_exp, ncols, 0,
                   kDownRowGroups, y, sab, pf_rows);
    return true;
}

// ---- port probe: throughput of the two arms on the same records (zero-GPU / diagnostic) ----------
// V4 caliber (QA pre-registration): T_new = ours best, T_ref = original best; converge <= 1.25x,
// grey 1.25-2x (requires a written net-account recompute), diverge > 2x or link headroom <= 0 =>
// stop and re-review. Probe grade ONLY: never quoted as an arm-level tok/s number.

struct PortProbeResult {
    CpuIsa isa            = CpuIsa::kScalar;
    int    experts        = 0;
    int    ncols          = 0;
    double ours_best_us   = 0.0; // per expert, single thread
    double ours_worst_us  = 0.0;
    double orig_best_us   = 0.0;
    double orig_worst_us  = 0.0;
    double ratio_best     = 0.0; // ours_best / orig_best
    bool   ok             = false;
};

inline PortProbeResult port_probe_throughput(CpuIsa isa, bool gu, const std::uint8_t* codes,
                                             std::size_t code_row_bytes, const std::uint8_t* scales,
                                             std::size_t code_expert_stride,
                                             std::size_t scale_expert_stride, const int* experts,
                                             int expert_count, int ncols, int rounds) {
    PortProbeResult r;
    r.isa = isa;
    if (expert_count <= 0 || ncols < 1 || ncols > kMaxColumns || rounds < 1) { return r; }
    // Synthetic activation blocks, deterministic; the values do not affect throughput.
    std::vector<A16Block> ab(static_cast<std::size_t>(ncols) * kGateUpBlocks);
    for (std::size_t i = 0; i < ab.size(); ++i) {
        std::int32_t hs = 0;
        for (int j = 0; j < 16; ++j) {
            ab[i].lo[j]  = static_cast<std::uint8_t>((i * 7U + static_cast<std::size_t>(j) * 13U) & 0xFFU);
            ab[i].mid[j] = static_cast<std::uint8_t>((i * 5U + static_cast<std::size_t>(j) * 11U) & 0xFFU);
            ab[i].hi[j]  = static_cast<std::int8_t>(
                static_cast<int>((i + static_cast<std::size_t>(j)) % 17U) - 8);
            hs += ab[i].hi[j];
        }
        ab[i].hi_sum = hs;
    }
    std::vector<const A16Block*> acts(static_cast<std::size_t>(ncols));
    std::vector<std::int64_t> out(static_cast<std::size_t>(kGateUpRowGroups) * 16 * kMaxColumns);
    double ours_best = 1.0e300, ours_worst = 0.0, orig_best = 1.0e300, orig_worst = 0.0;
    for (int round = 0; round < rounds; ++round) {
        { // layout-adapted arm (this tree's bytes, phi1/phi2/phi3)
            const auto t0 = std::chrono::steady_clock::now();
            for (int e = 0; e < expert_count; ++e) {
                for (int c = 0; c < ncols; ++c) {
                    acts[c] = &ab[static_cast<std::size_t>(c) * kGateUpBlocks];
                }
                const std::size_t ec = static_cast<std::size_t>(experts[e]) * code_expert_stride;
                const std::size_t es = static_cast<std::size_t>(experts[e]) * scale_expert_stride;
                rg_row_sums_ours(isa, gu, kIntermediate, codes + ec, code_row_bytes, scales + es,
                                 kHidden / 64, kGateUpBlocks, 0, kGateUpRowGroups, acts.data(), ncols,
                                 out.data(), 0, 2);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                              static_cast<double>(expert_count);
            if (us < ours_best) { ours_best = us; }
            if (us > ours_worst) { ours_worst = us; }
        }
        { // original arm: reference access pattern; values on this tree's bytes are not meaningful
            const auto t0 = std::chrono::steady_clock::now();
            for (int e = 0; e < expert_count; ++e) {
                for (int c = 0; c < ncols; ++c) {
                    acts[c] = &ab[static_cast<std::size_t>(c) * kGateUpBlocks];
                }
                const std::size_t ec = static_cast<std::size_t>(experts[e]) * code_expert_stride;
                rg_row_sums_orig(isa, codes + ec, kGateUpBlocks, 0, kGateUpRowGroups, acts.data(),
                                 ncols, out.data(), 2048);
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                              static_cast<double>(expert_count);
            if (us < orig_best) { orig_best = us; }
            if (us > orig_worst) { orig_worst = us; }
        }
    }
    r.experts       = expert_count;
    r.ncols         = ncols;
    r.ours_best_us  = ours_best;
    r.ours_worst_us = ours_worst;
    r.orig_best_us  = orig_best;
    r.orig_worst_us = orig_worst;
    r.ratio_best    = (orig_best > 0.0) ? (ours_best / orig_best) : 0.0;
    r.ok            = true;
    return r;
}

// ---- one-shot red-arm feasibility check: the three sabotage modes must CHANGE the output ---------
// Runs the layout-adapted arm on one expert with sab 0 then 1/2/3 and counts differing elements.
// A zero difference for any mode means that mode cannot turn red and must not be used as an arm.

struct PortSabCheck {
    std::uint64_t elements = 0;
    std::uint64_t diff1    = 0; // output row shift +1
    std::uint64_t diff2    = 0; // code nibble swap (column-pair swap)
    std::uint64_t diff3    = 0; // dropped biased-code correction
    bool          ok       = false;
};

inline PortSabCheck port_probe_sab_check(CpuIsa isa, bool gu, const std::uint8_t* codes,
                                         std::size_t code_row_bytes, const std::uint8_t* scales,
                                         std::size_t code_expert_stride,
                                         std::size_t scale_expert_stride, int expert, int ncols) {
    PortSabCheck r;
    if (ncols < 1 || ncols > kMaxColumns) { return r; }
    std::vector<A16Block> ab(static_cast<std::size_t>(ncols) * kGateUpBlocks);
    for (std::size_t i = 0; i < ab.size(); ++i) {
        std::int32_t hs = 0;
        for (int j = 0; j < 16; ++j) {
            ab[i].lo[j]  = static_cast<std::uint8_t>((i * 7U + static_cast<std::size_t>(j) * 13U) & 0xFFU);
            ab[i].mid[j] = static_cast<std::uint8_t>((i * 5U + static_cast<std::size_t>(j) * 11U) & 0xFFU);
            ab[i].hi[j]  = static_cast<std::int8_t>(
                static_cast<int>((i + static_cast<std::size_t>(j)) % 17U) - 8);
            hs += ab[i].hi[j];
        }
        ab[i].hi_sum = hs;
    }
    std::vector<const A16Block*> acts(static_cast<std::size_t>(ncols));
    for (int c = 0; c < ncols; ++c) { acts[c] = &ab[static_cast<std::size_t>(c) * kGateUpBlocks]; }
    const std::size_t n =
        static_cast<std::size_t>(kGateUpRowGroups) * 16 * static_cast<std::size_t>(ncols);
    std::vector<std::int64_t> base(n, 0), probe(n, 0);
    const std::size_t ec = static_cast<std::size_t>(expert) * code_expert_stride;
    const std::size_t es = static_cast<std::size_t>(expert) * scale_expert_stride;
    const auto run = [&](std::int32_t sab, std::vector<std::int64_t>& out) {
        rg_row_sums_ours(isa, gu, kIntermediate, codes + ec, code_row_bytes, scales + es,
                         kHidden / 64, kGateUpBlocks, 0, kGateUpRowGroups, acts.data(), ncols,
                         out.data(), sab, 0);
    };
    run(0, base);
    for (std::int32_t sab = 1; sab <= 3; ++sab) {
        for (std::size_t i = 0; i < n; ++i) { probe[i] = 0; }
        run(sab, probe);
        std::uint64_t d = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (probe[i] != base[i]) { ++d; }
        }
        if (sab == 1) { r.diff1 = d; }
        if (sab == 2) { r.diff2 = d; }
        if (sab == 3) { r.diff3 = d; }
    }
    r.elements = static_cast<std::uint64_t>(n);
    r.ok       = (r.diff1 > 0 && r.diff2 > 0 && r.diff3 > 0);
    return r;
}

} // namespace ported

// ================================================================================================
// Block-6 W1 -- the serve team's forward face, defined out of line: it needs the ported W4A16
// forward above (declared after CpuServeTeam). One job = one served expert record on this tree's
// faces; `jn/jt/ja` carry the columns (tokens) that routed to it and the summed weight per column.
// The team computes y (BF16, the ported forward's own dtype) straight into the job's own slot of
// `job_y`; the sum alpha*y over jobs is folded later, in job order, by reduce_forward (block-7
// fix -- see the note at the request's `job_y`). A job that fails leaves zeros in its slot, so a
// failed expert contributes exactly nothing, as before.
// ================================================================================================
inline void CpuServeTeam::forward_job(int j) {
    const CpuServeRequest& r = *request_;
    const int ncols          = r.jn[j];
    if (ncols <= 0) { return; }
    if (r.x == nullptr || r.partial == nullptr || r.job_y == nullptr ||
        ncols > kCpuServeForwardCols) {
        counters_->jobs_failed.fetch_add(1, std::memory_order_relaxed);
        note_status(1);
        return;
    }
    std::uint16_t* const slot =
        r.job_y + static_cast<std::size_t>(j) * kCpuServeForwardCols * ported::kHidden;
    // Zero the slot BEFORE computing: a failure anywhere below then contributes zeros, never the
    // previous round's leftovers.
    std::memset(slot, 0, static_cast<std::size_t>(ncols) * ported::kHidden * sizeof(std::uint16_t));
    std::vector<const std::uint16_t*> xp(static_cast<std::size_t>(ncols));
    std::vector<std::uint16_t*> yp(static_cast<std::size_t>(ncols));
    for (int c = 0; c < ncols; ++c) {
        xp[c] = r.x + static_cast<std::size_t>(r.jt[j][c]) * ported::kHidden;
        yp[c] = slot + static_cast<std::size_t>(c) * ported::kHidden;
    }
    const bool ok = ported::ported_forward_a16(
        ported::best_isa(), r.codes_gu[j], static_cast<std::size_t>(r.columns_gu / 2),
        r.scales_gu[j], r.divisor_gu[j], r.codes_dn[j], static_cast<std::size_t>(r.columns_dn / 2),
        r.scales_dn[j], r.divisor_dn[j], ncols, xp.data(), yp.data(), r.sab_mode, /*pf_rows=*/2);
    if (!ok) {
        // The slot stays zeroed: this expert contributes exactly nothing, loudly counted.
        counters_->jobs_failed.fetch_add(1, std::memory_order_relaxed);
        note_status(1);
        return;
    }
    counters_->fwd_jobs.fetch_add(1, std::memory_order_relaxed);
    counters_->fwd_elements.fetch_add(static_cast<std::uint64_t>(ncols) * ported::kHidden,
                                      std::memory_order_relaxed);
}

// Block-7 fix (deterministic reduce): partial[col][h] += Sum_j alpha_j,col * y_j,col,h, folded in
// (job, column, hidden) order on the serve thread after every worker of the round has stopped.
// Deterministic by construction: each slot has exactly one writer, and this single-threaded fold
// fixes the float summation order run to run (the parallel form's order followed completion
// timing, and its concurrent `+=` on shared token columns could drop whole contributions).
inline void CpuServeTeam::reduce_forward() {
    const CpuServeRequest& r = *request_;
    if (r.job_y == nullptr || r.partial == nullptr) { return; }
    for (std::int32_t j = 0; j < r.jobs; ++j) {
        const int ncols = r.jn[j];
        if (ncols <= 0) { continue; }
        const std::uint16_t* const slot =
            r.job_y + static_cast<std::size_t>(j) * kCpuServeForwardCols * ported::kHidden;
        for (int c = 0; c < ncols; ++c) {
            const float a = r.ja[j][c];
            float* dst = r.partial + static_cast<std::size_t>(r.jt[j][c]) * ported::kHidden;
            const std::uint16_t* src = slot + static_cast<std::size_t>(c) * ported::kHidden;
            for (int h = 0; h < ported::kHidden; ++h) {
                dst[h] += a * ported::bf16_to_f32(src[h]);
            }
        }
    }
}

// Block-8 item 3a: publish a forward round for BOTH consumption paths. The conversion runs after
// the ordered reduce and before the answer (single writer on the serve thread, fixed order -- the
// same discipline the reduce follows); the completion flag is stored with a release so a stream
// waiting on that value cannot observe it before the slab bytes it guards.
inline void CpuServeService::publish_forward_round(const CpuServeRequest& r) {
    if (r.partial_bf16 != nullptr && r.partial != nullptr && r.tokens >= 1 &&
        r.tokens <= kCpuServeForwardCols) {
        const std::size_t n =
            static_cast<std::size_t>(2'560) * static_cast<std::size_t>(r.tokens);
        for (std::size_t k = 0; k < n; ++k) {
            r.partial_bf16[k] = ported::f32_to_bf16_rn(r.partial[k]);
        }
    }
    if (r.done_flag != nullptr) {
        std::atomic_ref<std::uint32_t>(*r.done_flag).store(r.sequence, std::memory_order_release);
    }
}

// The forward face is sized for the ported forward's column cap; if either side ever moves, the
// request's jt/ja arrays must move with it -- this is the assert that fails first when they don't.
static_assert(kCpuServeForwardCols == ported::kMaxColumns,
              "the serve forward face must track ported::kMaxColumns");

} // namespace ninfer::models::qwen3_8_flash_next::detail::cpu_expert
