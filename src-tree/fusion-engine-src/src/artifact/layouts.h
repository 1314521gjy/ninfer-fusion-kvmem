#pragma once

#include "core/weight_view.h"

#include <cstdint>
#include <span>

namespace ninfer::artifact {

struct TensorObject;

// Is this declared object the rank-three block-scale EXPERT BANK rather than a rank-two
// block-scale matrix?
//
// WHY THE SHAPE IS PART OF THE QUESTION: upstream keeps two distinct storage layouts
// (`BlockScaleK16M128x4V1` vs `ExpertBlockScaleK16M128x4V1`) and dispatches on that value alone.
// This engine carries both spellings in ONE `QuantLayout` value (`artifact/formats.cpp` maps
// `blockscale-k16-m128x4-v1`, `expert-blockscale-k16-m128x4-v1` and
// `expert_blockscale_k16_m128x4_v1` all to `QuantLayout::BlockScaleK16M128x4`), so the layout value
// alone cannot tell them apart -- asking only about (format, layout) would send every rank-two
// NVFP4 block-scale matrix, which other targets publish (`src/targets/.../bindings.cpp`), into the
// bank rules, where `block_scale_bank_geometry` would reject it. Rank three is what the bank
// demands and the matrix never is, so the rank closes the ambiguity.
//
// WHY NOT DISPATCH ON RANK ALONE: the MTP draft layer's experts are rank-three BF16 planes. They
// are ordinary contiguous parents; the format test is what keeps them out of the NVFP4 bank path.
[[nodiscard]] constexpr bool is_block_scale_bank(QType format, QuantLayout layout,
                                                 std::span<const std::uint64_t> shape) {
    return format == QType::NVFP4 && layout == QuantLayout::BlockScaleK16M128x4 && shape.size() == 3;
}

// Encoded-parent geometry for one declared tensor object -- the ONE dispatch between
// `weight_geometry` (one matrix) and `block_scale_bank_geometry` (one rank-three bank).
//
// Both the engine (`describe_tensor` below) and `tools/artifact_geometry_check.cpp` call this, so
// the tool's verdict and the engine's read cannot drift apart: a green geometry check means the
// engine would compute the same byte count, which is the only thing that check claims.
// `divisors` is forwarded to `weight_geometry`; a bank carries one divisor per expert instead and
// ignores it.
[[nodiscard]] WeightGeometry tensor_geometry(QType format, QuantLayout layout,
                                             std::span<const std::uint64_t> shape,
                                             std::uint64_t divisors = 1);

// Interpret and validate a requested tensor's complete encoded parent.
[[nodiscard]] WeightGeometry describe_tensor(const TensorObject& object);

} // namespace ninfer::artifact
