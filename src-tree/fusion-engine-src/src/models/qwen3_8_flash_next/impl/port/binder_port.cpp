#include "models/qwen3_8_flash_next/impl/port/binder_port.h"

#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::port {

Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                           artifact::ObjectHandle handle, NumericFormat format, std::int32_t rows,
                           std::int32_t columns) {
    if (rows <= 0 || columns <= 0) {
        throw std::invalid_argument("port: materialized_weight needs positive rows and columns");
    }
    // Upstream rejects NVFP4 here for the same reason: an NVFP4 plane carries target-validated
    // weight and input divisors, so it cannot be described by rows/columns alone.
    if (format == QType::NVFP4) {
        throw std::invalid_argument(
            "port: materialized_weight cannot describe NVFP4, which needs its divisors");
    }

    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    // The byte extents come from THIS engine's geometry, not from a transcription of the upstream
    // one: it is the single source for how a plane of a given format and layout is packed, and it
    // already backs `WeightGeometry` everywhere else in the engine.
    const WeightGeometry geometry = weight_geometry(format, QuantLayout::Contiguous, shape);

    Weight out{};
    out.payload       = bytes;
    out.qdata         = bytes;
    out.payload_bytes = geometry.bytes;
    out.qtype         = format;
    out.layout        = QuantLayout::Contiguous;
    out.n             = rows;
    out.k             = columns;
    out.ndim          = 2;
    out.shape[0]      = rows;
    out.shape[1]      = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

} // namespace ninfer::port
