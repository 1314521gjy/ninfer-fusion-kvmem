// Copied from upstream `src/artifact/typed_binding.cpp` (igorls/ninfer @ c81c88f6).
//
// WHAT CHANGED, AND WHY ONLY THIS
//   Upstream's Binder is a different generation: it takes a `StorageLayout` and a heap `span` shape,
//   and answers `require_resource` / `validate_only` / `retain_on_host`. This engine's Binder takes a
//   `QuantLayout` and a `Shape`, and expresses the same choices as a residency on one call. The
//   adapter in `models/qwen3_8_flash_next/impl/port/binder_port.h` supplies upstream's surface, so
//   the calls in this file are upstream's, unedited.
//
//   Nothing else was rewritten. `contiguous_weight` / `row_split_weight` / `row_scale_weight` keep
//   upstream's field assignments; the byte extents come from this engine's `weight_geometry`, which
//   is the single source for how a plane of a given format and layout is packed and already backs
//   `WeightGeometry` everywhere else here. Upstream split the same numbers out into
//   `RowSplitGeometry` and `RowScaleGeometry`; mapping onto the existing struct is what avoids a
//   second set of geometry types in this tree, and the names line up one for one:
//
//     upstream encoded_bytes         ->  WeightGeometry::bytes
//     upstream high_plane_offset     ->  WeightGeometry::high_offset
//     upstream high_plane_bytes      ->  WeightGeometry::high_bytes
//     upstream scale_plane_offset    ->  WeightGeometry::scale_offset
//     upstream padded_columns        ->  WeightGeometry::padded_columns
//     upstream group_size            ->  WeightGeometry::group_size

#include "artifact/typed_binding.h"

#include "artifact/materializer.h"
#include "core/weight_view.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::artifact {

using namespace ninfer::portalias;

namespace {

// Upstream `qtype_for`. I64 and U4Z8G16_F16S are mapped tensors rather than device weights, which is
// why upstream throws for both; keeping that here means a caller that asks for a device `Weight`
// over a PLE shard is told so instead of being handed a mis-typed view.
QType qtype_for(NumericFormat format) {
    switch (format) {
    case QType::BF16:
        return QType::BF16;
    case QType::FP32:
        return QType::FP32;
    case QType::INT32:
        return QType::INT32;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q8_G32_FP16:
    case QType::NVFP4:
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::FP8_E4M3FN_ROW_F32:
        return format;
    case QType::T2_G128_FP16:
    case QType::PTQ1_G128_FP16:
    case QType::GGUF_Q8_0:
    case QType::GGUF_Q2_K:
    case QType::GGUF_Q3_K:
    case QType::GGUF_Q4_K:
    case QType::GGUF_Q5_K:
    case QType::GGUF_Q6_K:
    case QType::GGUF_IQ2_XXS:
    case QType::GGUF_IQ2_XS:
    case QType::GGUF_IQ2_S:
    case QType::GGUF_IQ3_XXS:
    case QType::GGUF_IQ3_S:
    case QType::GGUF_IQ1_S:
    case QType::GGUF_IQ1_M:
    case QType::GGUF_IQ4_NL:
    case QType::GGUF_IQ4_XS:
    case QType::GGUF_Q2_0:
        return format;
    case QType::I64:
        throw std::logic_error("mapped I64 tensors have no device Weight type");
    case QType::U4Z8G16_F16S:
        throw std::logic_error("mapped U4 PLE tensors have no device Weight type");
    }
    throw std::logic_error("unhandled numeric format");
}

// Upstream `dtype_for`.
DType dtype_for(NumericFormat format) {
    switch (format) {
    case QType::BF16:
        return DType::BF16;
    case QType::FP32:
        return DType::FP32;
    case QType::INT32:
        return DType::I32;
    case QType::I64:
        throw std::logic_error("mapped I64 tensors have no device dtype");
    default:
        throw std::logic_error("quantized format has no direct dtype");
    }
}

// Upstream `storage_layout_for`, verbatim in behaviour: it answers which of the two layout
// enumerations a format is stored under. The return type is the PORT's `StorageLayout` because that
// is the argument the adapter's `require_tensor` takes; the adapter is what maps it onto this
// engine's `QuantLayout`, so the mapping exists once rather than twice.
port::StorageLayout layout_for(NumericFormat format) {
    // Block-5: the whole ggml block family stores under the one gguf layout, so it is answered here
    // rather than by sixteen switch cases; Q2_0 is included through `is_gguf`.
    if (is_gguf(format)) { return port::StorageLayout::GgufBlocksV1; }
    switch (format) {
    case QType::BF16:
    case QType::FP32:
    case QType::INT32:
    case QType::I64:
        return port::StorageLayout::ContiguousLeV1;
    case QType::Q4_G64_FP16:
    case QType::Q5_G64_FP16:
    case QType::Q6_G64_FP16:
    case QType::Q8_G32_FP16:
        return port::StorageLayout::RowSplitK128V1;
    case QType::NVFP4:
        return port::StorageLayout::BlockScaleK16M128x4V1;
    case QType::FP8_E4M3FN_ROW_BF16:
        return port::StorageLayout::RowScaleV1;
    case QType::FP8_E4M3FN_ROW_F32:
        return port::StorageLayout::RowScaleF32V1;
    case QType::U4Z8G16_F16S:
        return port::StorageLayout::PackedU4G16V1;
    default:
        throw std::logic_error("unhandled numeric format for a storage layout");
    }
}

// The same mapping in this engine's vocabulary, for the code below that has to compute a geometry
// rather than call the binder.
QuantLayout engine_layout_for(NumericFormat format) {
    return port::engine_layout(layout_for(format));
}

} // namespace

// Upstream `bind_tensor`, unedited: the placement decides between a device demand and a validation.
ObjectHandle bind_tensor(port::Binder& binder, std::string_view name, NumericFormat format,
                         std::initializer_list<std::uint64_t> shape,
                         TensorPlacement placement) {
    // Upstream passed the shape as a heap span; the adapter takes the initializer list form so the
    // conversion to this engine's `Shape` happens inside it, in one place.
    const ObjectHandle handle = binder.require_tensor(name, format, layout_for(format), shape);
    if (placement == TensorPlacement::Device) {
        binder.materialize_on_device(handle);
    } else {
        binder.validate_only(handle);
    }
    return handle;
}

ObjectHandle bind_device_tensor(port::Binder& binder, std::string_view name, NumericFormat format,
                                std::initializer_list<std::uint64_t> shape) {
    return bind_tensor(binder, name, format, shape, TensorPlacement::Device);
}

ObjectHandle bind_raw_resource(port::Binder& binder, std::string_view name) {
    const ObjectHandle handle = binder.require_resource(name, ResourceEncoding::RawBytesV1);
    binder.retain_on_host(handle);
    return handle;
}

Tensor materialized_tensor(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::initializer_list<std::int32_t> internal_shape) {
    return Tensor(materialized.device_data(handle), dtype_for(format), internal_shape);
}

Weight materialized_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::int32_t rows, std::int32_t columns) {
    if (rows <= 0 || columns <= 0) {
        throw std::invalid_argument("materialized_weight needs positive rows and columns");
    }
    const QType qtype = qtype_for(format);
    const QuantLayout layout = engine_layout_for(format);
    const std::vector<std::uint64_t> shape = {static_cast<std::uint64_t>(rows),
                                              static_cast<std::uint64_t>(columns)};
    const WeightGeometry geometry = weight_geometry(qtype, layout, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload         = bytes;
    out.qdata           = bytes;
    out.payload_bytes   = geometry.bytes;
    out.qtype           = qtype;
    out.layout          = layout;
    out.n               = rows;
    out.k               = columns;
    out.group_size      = static_cast<std::uint32_t>(geometry.group_size);
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = static_cast<std::int32_t>(geometry.padded_columns);
    out.ndim            = 2;

    // The plane offsets are where upstream's two dedicated builders put them; reading them from the
    // shared geometry is what keeps one source for the packing.
    if (layout == QuantLayout::RowSplit || layout == QuantLayout::RowSplitPanel) {
        out.high_plane_bytes = geometry.high_bytes;
        out.qhigh = geometry.high_bytes == 0 ? nullptr : bytes + geometry.high_offset;
        out.scales = bytes + geometry.scale_offset;
        out.group  = static_cast<std::int32_t>(geometry.group_size);
    } else if (layout == QuantLayout::RowScale) {
        out.scales = bytes + geometry.scale_offset;
        // `group_size` is the element count of one quantization group and `group` is its width. Row
        // scaling has exactly one group per row, so both equal the row length: `weight_geometry`
        // already sets `group_size = k` for this layout (`core/weight_view.cpp:151`), and
        // `validate_fp8_weight` demands `group_size == k` AND `group == k`
        // (`ops/linear/fp8/fp8_format.cpp:64`). Leaving `group` at its default 0 is exactly what
        // rejected every 4-byte FP8 weight; `core/weight_view.cpp:322` is the same assignment on the
        // fusion path, and this is the ported path's copy of it.
        out.group = static_cast<std::int32_t>(columns);
        // Scale metadata, in the shape `core/weight_view.cpp:332-336` uses for the fusion path, and
        // widened to both arms. Without this an FP8 weight leaves the Weight defaults
        // (`scale_ne = {1,1,1,1}`, `scale_nb = {0,0,0,0}`, `core/weight.h:221-222`), and every FP8
        // validator rejects it: the engine's `validate_fp8_weight` wants `scale_ne[0] == n` and
        // `scale_nb[0] == scale_word_bytes`, and the ported family's `exact_fp8_f32_weight` wants
        // `scale_ne[0] == rows` and `scale_nb[0] == 4`. The ported family's weights are
        // `fp8_e4m3fn_row_f32` -> QType::FP8_E4M3FN_ROW_F32 (4-byte word); only the fusion path's
        // are the 2-byte FP8_E4M3FN_ROW_BF16.
        const bool f32_scales               = qtype == QType::FP8_E4M3FN_ROW_F32;
        const std::int64_t scale_word_bytes = f32_scales ? 4 : 2;
        out.scale_dtype                     = f32_scales ? DType::FP32 : DType::BF16;
        out.scale_ne[0]                     = rows;
        out.scale_nb[0]                     = scale_word_bytes;
        out.scale_nb[1] = out.scale_nb[2] = out.scale_nb[3] =
            static_cast<std::int64_t>(rows) * scale_word_bytes;
    } else if (layout == QuantLayout::BlockScaleK16M128x4) {
        out.scales = bytes + geometry.scale_offset;
    } else if (layout == QuantLayout::PackedU4G16) {
        out.scales = bytes + geometry.scale_offset;
    }
    return out;
}

} // namespace ninfer::artifact
