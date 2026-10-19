#include "artifact/layouts.h"

#include "artifact/bank_geometry.h"
#include "artifact/formats.h"
#include "artifact/schema.h"

namespace ninfer::artifact {

namespace {

// The bank layout's own alignment. `artifact/bank_geometry.h:19-20` records 256 as the wire
// property of `expert-blockscale-k16-m128x4-v1`, and `block_scale_bank_geometry` pads its scale
// plane to it -- but the struct it returns does not carry the number, and a geometry whose
// alignment field disagreed with the padding would let a misaligned object pass validation.
constexpr std::uint64_t kBlockScaleBankAlignment = 256;

} // namespace

WeightGeometry tensor_geometry(QType format, QuantLayout layout,
                               std::span<const std::uint64_t> shape, std::uint64_t divisors) {
    if (!is_block_scale_bank(format, layout, shape)) {
        return weight_geometry(format, layout, shape, divisors);
    }
    // A bank is one encoded parent holding every expert's matrix: code plane, then scale plane,
    // then one fp32 divisor per expert. `block_scale_bank_geometry` enforces the rank, the format
    // and the 128/64 divisibility itself, and throws by name rather than returning zeros.
    const BlockScaleBankGeometry bank = block_scale_bank_geometry(format, shape);
    WeightGeometry out;
    out.format   = format;
    out.layout   = layout;
    out.shape.assign(shape.begin(), shape.end());
    out.elements = weight_element_count(shape);
    out.bytes    = bank.encoded_bytes;
    out.alignment = kBlockScaleBankAlignment;
    out.padded_columns = bank.columns;
    out.group_size     = 16; // `k16`: one scale word per 16 values (bank_geometry: groups_per_row).
    out.code_bytes     = bank.code_plane_bytes;
    out.scale_offset   = bank.scale_plane_offset;
    out.scale_bytes    = bank.scale_plane_bytes;
    out.divisor_offset = bank.weight_divisor_offset;
    out.divisor_count  = bank.experts;
    // The row-split/row-scale fields (`code_bytes_per_row`, `high_*`, `scale_bytes_per_row`) are
    // deliberately left zero: they describe rows of ONE matrix, and filling them with per-bank
    // numbers would be a second, quietly wrong description of the same bytes. Consumers that need
    // a per-expert view ask `make_nvfp4_expert_bank_view` (impl/expert_bank.cpp), which owns that
    // arithmetic.
    return out;
}

WeightGeometry describe_tensor(const TensorObject& object) {
    try {
        auto geometry = tensor_geometry(parse_format(object.format), parse_layout(object.layout),
                                        object.shape, object.divisors);
        if (geometry.bytes != object.bytes || object.offset % geometry.alignment) {
            throw ArtifactError("encoded size or object alignment differs from layout");
        }
        return geometry;
    } catch (const std::exception& error) { throw ArtifactError(object.id + ": " + error.what()); }
}

} // namespace ninfer::artifact
