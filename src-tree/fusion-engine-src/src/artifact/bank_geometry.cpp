#include "artifact/bank_geometry.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::artifact {
namespace {

constexpr std::uint64_t kBankAlignment = 256;

std::uint64_t bank_mul(std::uint64_t a, std::uint64_t b, const char* what) {
    if (b != 0 && a > (std::numeric_limits<std::uint64_t>::max() / b)) {
        throw ArtifactError(std::string("NVFP4 bank overflow: ") + what);
    }
    return a * b;
}

std::uint64_t bank_add(std::uint64_t a, std::uint64_t b, const char* what) {
    if (a > (std::numeric_limits<std::uint64_t>::max() - b)) {
        throw ArtifactError(std::string("NVFP4 bank overflow: ") + what);
    }
    return a + b;
}

std::uint64_t bank_align_up(std::uint64_t value, std::uint64_t alignment, const char* what) {
    if (value > (std::numeric_limits<std::uint64_t>::max() - (alignment - 1))) {
        throw ArtifactError(std::string("NVFP4 bank overflow: ") + what);
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

} // namespace

BlockScaleBankGeometry block_scale_bank_geometry(QType format, std::span<const std::uint64_t> shape) {
    if (format != QType::NVFP4) {
        throw ArtifactError("expert-blockscale-k16-m128x4-v1 requires NVFP4");
    }
    if (shape.size() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0) {
        throw ArtifactError("expert-blockscale-k16-m128x4-v1 requires a positive rank-three shape");
    }
    const auto experts = shape[0];
    const auto rows    = shape[1];
    const auto columns = shape[2];
    if (rows % 128 != 0 || columns % 64 != 0) {
        throw ArtifactError(
            "expert-blockscale-k16-m128x4-v1 requires N divisible by 128 and K divisible by 64");
    }

    BlockScaleBankGeometry out;
    out.experts            = experts;
    out.rows               = rows;
    out.columns            = columns;
    out.groups_per_row     = columns / 16;
    out.k_tiles            = columns / 64;
    const auto matrices    = bank_mul(experts, rows, "NVFP4 bank row count");
    const auto elements    = bank_mul(matrices, columns, "NVFP4 bank element count");
    out.code_plane_bytes   = elements / 2;
    out.scale_plane_offset = bank_align_up(out.code_plane_bytes, kBankAlignment,
                                           "NVFP4 bank scale plane offset");
    out.scale_plane_bytes  = elements / 16;
    out.weight_divisor_offset =
        bank_add(out.scale_plane_offset, out.scale_plane_bytes, "NVFP4 bank weight divisor offset");
    out.weight_divisor_bytes = bank_mul(experts, 4, "NVFP4 bank divisor bytes");
    out.encoded_bytes        = bank_add(out.weight_divisor_offset, out.weight_divisor_bytes,
                                        "NVFP4 bank tensor encoded size");
    return out;
}

} // namespace ninfer::artifact
