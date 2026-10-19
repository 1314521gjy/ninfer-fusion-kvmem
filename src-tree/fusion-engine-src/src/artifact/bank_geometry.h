#pragma once

#include "artifact/schema.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <cstdint>
#include <span>

// Geometry of a rank-three block-scale *bank*: `experts` whole matrices sharing one encoded parent.
//
// Why this is not `weight_geometry()`: that function describes ONE matrix and rejects any quantized
// shape whose rank is not 2 (`core/weight_view.cpp`). A routed-expert bank materializes every
// expert's matrix into a single object, so the planes are laid out once for the whole rank-three
// tensor: one code plane for all experts, then one scale plane for all experts, then one divisor per
// expert. Reading it as a per-expert matrix would land every expert but the first at the wrong
// offset -- silently, because the bytes are still in range.
//
// Layout (`expert-blockscale-k16-m128x4-v1`, alignment 256 -- the same wire name and the same
// alignment our `artifact/formats.cpp` and `tools/artifact/layouts.py` already record for
// `block_scale_k16_m128x4_v1`):
//
//   [ code plane ][ pad ][ scale plane ][ weight divisors ]
//
//   code plane        experts * rows * columns / 2      (NVFP4 packs two values per byte)
//   scale plane       experts * rows * columns / 16     (one scale word per 16-value group)
//   weight divisors   experts * 4                       (one fp32 divisor per expert matrix)
//
// The result is written into a `WeightGeometry` so the bank shares the engine's existing geometry
// vocabulary instead of introducing a parallel one; the bank-specific fields are the leading three.

namespace ninfer::artifact {

// The engine spells numeric encodings `QType`; the ported Flash-Next target code was written against
// the upstream name `NumericFormat`. They name the same enum, so this is an alias rather than a
// second enumeration -- a parallel enum would let the two drift apart silently.
using NumericFormat = QType;

// Rank-three block-scale bank geometry, in the field names the bank code addresses it by.
//
// `experts` whole matrices share one encoded parent:
//   [ code plane ][ pad to 256 ][ scale plane ][ weight divisors ]
struct BlockScaleBankGeometry {
    std::uint64_t experts               = 0;
    std::uint64_t rows                  = 0;
    std::uint64_t columns               = 0;
    std::uint64_t groups_per_row        = 0;
    std::uint64_t k_tiles               = 0;
    std::uint64_t code_plane_bytes      = 0;
    std::uint64_t scale_plane_offset    = 0;
    std::uint64_t scale_plane_bytes     = 0;
    std::uint64_t weight_divisor_offset = 0;
    std::uint64_t weight_divisor_bytes  = 0;
    std::uint64_t encoded_bytes         = 0;
};

// `shape` must be `{experts, rows, columns}` with every entry positive; `rows` divisible by 128 and
// `columns` divisible by 64 are the format's own requirements and are enforced rather than assumed.
// Throws `ArtifactError` on a non-NVFP4 format, a shape that is not positive rank-three, or an
// encoding size that would overflow.
[[nodiscard]] BlockScaleBankGeometry block_scale_bank_geometry(QType format,
                                                               std::span<const std::uint64_t> shape);

} // namespace ninfer::artifact
