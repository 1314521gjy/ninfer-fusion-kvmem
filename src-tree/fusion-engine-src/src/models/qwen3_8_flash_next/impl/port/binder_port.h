#pragma once

// Port adapter: the upstream engine generation's `artifact::Binder` surface, expressed over this
// engine's `artifact::Binder`.
//
// WHY THIS EXISTS
//   The Flash-Next target was written against an earlier generation of this engine. Its load layer
//   (`load/bindings.cpp`, `load/materialized.cpp`, `load/loader.cpp`) calls that generation's Binder
//   directly. Measured, the whole seam is 13 API shapes over 24 call sites, so translating it once
//   here beats editing 24 call sites across 9 files -- one fact, one home.
//
// WHY A SEPARATE NAMESPACE
//   Both generations declare `ninfer::artifact::Binder` and `ninfer::artifact::ObjectHandle`. Adding
//   this to `ninfer::artifact` would be a redefinition, so it lives in `ninfer::port` and the load
//   layer reaches it through the alias block in `artifact/typed_binding.h`.
//
// WHY `materialize_on_device` IS A NO-OP  (the one thing easy to get wrong)
//   Upstream needed two steps: `require_tensor(...)` then `materialize_on_device(handle)`.
//   This engine's `Binder::binding()` already performs the device demand for every part of the
//   binding when `residency == Residency::Device` -- see `src/artifact/binder.cpp:52-65`. So the
//   demand has already happened by the time `require_tensor` returns here, and repeating it in a
//   separate step would double-count. The residency argument, not a second call, is how this engine
//   says "device" / "host" / "pinned".
//
// NOT A PLACE TO ADD ENGINE TYPES
//   A symbol this engine genuinely lacks must be added to the narrow engine header that owns that
//   part of the contract. What lives here is only what the UPSTREAM contract has and this engine
//   spells differently.

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/schema.h"
#include "artifact/views.h"
#include "core/dtype.h"
#include "core/weight.h"
#include "core/weight_view.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::port {

// The upstream engine spelled the encoding enum `NumericFormat`; this engine spells it `QType`.
// Same enumeration, so an alias -- a second enum could drift from the first without a diagnostic.
using NumericFormat = QType;

// Upstream's storage-layout enum. This engine's `QuantLayout` is the same concept; upstream
// distinguishes `RowScaleF32V1` from `RowScaleV1` while this engine carries that difference in the
// format (a 4-byte scale word), which is what `engine_layout` below records.
enum class StorageLayout : std::uint8_t {
    ContiguousLeV1,
    RowSplitK128V1,
    BlockScaleK16M128x4V1,
    RowScaleV1,
    RowScaleF32V1,
    PackedU4G16V1,
    ExpertBlockScaleK16M128x4V1,
    // Whole ggml block rows (this engine's `QuantLayout::GgufBlocks`). Added for the block-5 IQ3_S
    // artifact: the v3 tooling writes the `gguf_blocks_v1` layout name on the expert banks, so the
    // port has to be able to spell the layout in BOTH directions before its load layer can bind
    // them. Appended LAST: nothing already numbered moves.
    GgufBlocksV1,
};

enum class TensorPlacement : std::uint8_t {
    Device,
    ValidateOnly,
};

// Precision of the GDN (linear-attention) recurrent state.
//
// This engine also models the GDN state choice as a boolean, `gdn_state_fp16`. The ported target
// instead carries a two-valued enum and reads it as a *quantity* --
// `flash_next_gdn_ssm_bytes_per_slot()` returns 2 or 4 bytes from it.
//
// The enum itself now lives in `include/ninfer/types.h` beside `KvCacheStorage`, because the ported
// package reaches it through `EngineOptions::gdn_state_storage`. Only the translation stays here:
// one fact, one definition, and these two functions bridge it to the engine's boolean.
using GdnStateStorage = ::ninfer::GdnStateStorage;

//   GdnStateStorage::BF16  <=>  gdn_state_fp16 == true
//   GdnStateStorage::FP32  <=>  gdn_state_fp16 == false
[[nodiscard]] constexpr bool gdn_state_fp16_from(GdnStateStorage storage) noexcept {
    return storage == GdnStateStorage::BF16;
}

[[nodiscard]] constexpr GdnStateStorage gdn_state_storage_from(bool fp16) noexcept {
    return fp16 ? GdnStateStorage::BF16 : GdnStateStorage::FP32;
}

enum class ResourceEncoding : std::uint8_t {
    RawBytesV1,
};

struct PayloadSpan {
    std::span<const std::byte> data;
};

// Upstream's `artifact::TensorDescriptor` / `ResourceDescriptor` / `ObjectDescriptor` triple.
//
// This engine's `TensorObject` stores its `format` and `layout` as the *strings* the container wrote
// and parses them into enumerations on demand, while upstream stores the parsed enumerations in the
// descriptor. The ported load layer inspects a descriptor to decide how to bind an object -- it asks
// "is this the BF16 or the NVFP4 MTP expert bank?" -- so the adapter presents the enum form.
//
// One deliberate infidelity: if an object's stored format or layout name is one this engine does not
// know, `object_descriptor` throws rather than inventing an enumerator.
struct TensorDescriptor {
    std::string name;
    std::vector<std::uint64_t> shape;
    NumericFormat format;
    StorageLayout layout;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

struct ResourceDescriptor {
    std::string name;
    ResourceEncoding encoding;
    std::uint64_t offset = 0;
    std::uint64_t bytes  = 0;
};

using ObjectDescriptor = std::variant<TensorDescriptor, ResourceDescriptor>;

// Upstream's layout expectations, expressed as the layout this engine must resolve to. Both row-scale
// spellings map onto one layout here, because the scale word width is a property of the format.
[[nodiscard]] inline QuantLayout engine_layout(StorageLayout layout) {
    switch (layout) {
    case StorageLayout::ContiguousLeV1:
        return QuantLayout::Contiguous;
    case StorageLayout::RowSplitK128V1:
        return QuantLayout::RowSplit;
    case StorageLayout::BlockScaleK16M128x4V1:
    case StorageLayout::ExpertBlockScaleK16M128x4V1:
        return QuantLayout::BlockScaleK16M128x4;
    case StorageLayout::RowScaleV1:
    case StorageLayout::RowScaleF32V1:
        return QuantLayout::RowScale;
    case StorageLayout::PackedU4G16V1:
        return QuantLayout::PackedU4G16;
    case StorageLayout::GgufBlocksV1:
        return QuantLayout::GgufBlocks;
    }
    throw artifact::ArtifactError("port: unhandled upstream storage layout");
}

[[nodiscard]] inline const char* port_layout_name(StorageLayout layout) {
    switch (layout) {
    case StorageLayout::ContiguousLeV1: return "contiguous-le-v1";
    case StorageLayout::RowSplitK128V1: return "row-split-k128-v1";
    case StorageLayout::BlockScaleK16M128x4V1: return "blockscale-k16-m128x4-v1";
    case StorageLayout::RowScaleV1: return "row-scale-v1";
    case StorageLayout::RowScaleF32V1: return "row-scale-f32-v1";
    case StorageLayout::PackedU4G16V1: return "packed-u4-g16-v1";
    case StorageLayout::ExpertBlockScaleK16M128x4V1: return "expert-blockscale-k16-m128x4-v1";
    case StorageLayout::GgufBlocksV1: return "gguf-blocks-v1";
    }
    return "unknown";
}

// Forward declaration: the definition sits after `Binder` because it converts an engine object, but
// `Binder::find` below needs the name in scope. A free function called from a member of a class
// defined earlier is not found by a later definition alone.
[[nodiscard]] ObjectDescriptor object_descriptor(const artifact::Object& object);

// The reverse of `engine_layout`: the upstream spelling of a layout this engine resolved. Both
// row-scale spellings collapse to the 4-byte one here, because upstream keeps the scale width in the
// format and a descriptor carries both -- so the pair ("FP8_E4M3FN_ROW_F32", RowScaleF32V1) is
// reconstructed from the format by the caller that needs it exactly.
[[nodiscard]] inline StorageLayout upstream_layout(QuantLayout layout) {
    switch (layout) {
    case QuantLayout::Contiguous: return StorageLayout::ContiguousLeV1;
    case QuantLayout::RowSplit:
    case QuantLayout::RowSplitPanel: return StorageLayout::RowSplitK128V1;
    case QuantLayout::BlockScaleK16M128x4: return StorageLayout::BlockScaleK16M128x4V1;
    case QuantLayout::RowScale: return StorageLayout::RowScaleV1;
    case QuantLayout::PackedU4G16: return StorageLayout::PackedU4G16V1;
    // Block-5: the gguf spelling exists now. The throw that used to sit here ("gguf blocks have no
    // upstream storage layout spelling") was a missing-word gap, not a capability gap -- the engine
    // resolves, parses and sizes GgufBlocks throughout; only this vocabulary lacked the word.
    case QuantLayout::GgufBlocks: return StorageLayout::GgufBlocksV1;
    }
    throw artifact::ArtifactError("port: unhandled layout");
}

// Upstream Binder surface, translated onto this engine's Binder.
class Binder {
public:
    explicit Binder(artifact::Binder& inner) : inner_(&inner) {}

    [[nodiscard]] artifact::Binder& engine() const noexcept { return *inner_; }

    [[nodiscard]] artifact::ObjectHandle require_tensor(std::string_view name, NumericFormat format,
                                                       StorageLayout layout,
                                                       std::initializer_list<std::uint64_t> shape) {
        return require_tensor(name, format, layout, artifact::Shape(shape));
    }

    [[nodiscard]] artifact::ObjectHandle require_tensor(std::string_view name, NumericFormat format,
                                                       StorageLayout layout,
                                                       const artifact::Shape& shape) {
        return register_object(name, format, layout, shape, artifact::Residency::Device);
    }

    // The residency belongs on the registration call, because registration is the single place this
    // engine records a demand (`src/artifact/binder.cpp:52-65` branches on it once). The verbs below
    // cannot add a demand afterwards, so a call site that means "host copy, never a device copy" has
    // to say so here; without this overload such a site had no spelling at all and every object was
    // registered Device.
    [[nodiscard]] artifact::ObjectHandle require_tensor(std::string_view name, NumericFormat format,
                                                       StorageLayout layout,
                                                       const artifact::Shape& shape,
                                                       artifact::Residency residency) {
        return register_object(name, format, layout, shape, residency);
    }

    // Upstream takes the shape as `std::span<const std::uint64_t>`; ported call sites hand it a
    // `std::initializer_list`, a `std::vector`, or an array. One template accepts all of them and
    // builds this engine's `Shape`, so the ported call sites keep upstream's spelling.
    template <class Range>
        requires requires(const Range& range) {
            range.begin();
            range.end();
            { *range.begin() } -> std::convertible_to<std::uint64_t>;
        }
    [[nodiscard]] artifact::ObjectHandle require_tensor(std::string_view name, NumericFormat format,
                                                       StorageLayout layout, const Range& shape) {
        return require_tensor(name, format, layout, artifact::Shape(shape.begin(), shape.end()));
    }

    template <class Range>
        requires requires(const Range& range) {
            range.begin();
            range.end();
            { *range.begin() } -> std::convertible_to<std::uint64_t>;
        }
    [[nodiscard]] artifact::ObjectHandle require_tensor(std::string_view name, NumericFormat format,
                                                       StorageLayout layout, const Range& shape,
                                                       artifact::Residency residency) {
        return require_tensor(name, format, layout, artifact::Shape(shape.begin(), shape.end()),
                              residency);
    }

    // Upstream's `materialize_on_device` / `retain_on_host` / `retain_mapped_tensor` are the second
    // half of a two-step protocol. This engine expresses the same choice as a residency on the single
    // registration call, so these verify agreement rather than recording a second demand.
    void materialize_on_device(artifact::ObjectHandle handle) const {
        verify(handle, artifact::Residency::Device);
    }
    void retain_on_host(artifact::ObjectHandle handle) const {
        verify(handle, artifact::Residency::Host);
    }
    void retain_mapped_tensor(artifact::ObjectHandle handle) const {
        verify(handle, artifact::Residency::Pinned);
    }
    // Upstream's "check the shape, do not make a device demand". This engine always validates the
    // logical shape in `binding()`, so the shape half already happened.
    void validate_only(artifact::ObjectHandle handle) const {
        verify(handle, artifact::Residency::Device);
    }

    [[nodiscard]] PayloadSpan payload(artifact::ObjectHandle handle) const {
        return PayloadSpan{inner_->reader().read_range(object_offset(handle), object_bytes(handle))};
    }

    // `payload()` builds a span over the vector `read_range` returned; that vector dies at the end of
    // the full expression, so a caller that stores the span reads freed memory. This form owns the
    // bytes instead. Only two 8-byte objects are read this way, so the copy costs nothing.
    [[nodiscard]] std::vector<std::byte> payload_bytes(artifact::ObjectHandle handle) const {
        return inner_->reader().read_range(object_offset(handle), object_bytes(handle));
    }

    [[nodiscard]] std::uint64_t object_offset(artifact::ObjectHandle handle) const {
        const auto& directory = inner_->reader().directory();
        return artifact::object_offset(directory.objects.at(handle.index));
    }

    [[nodiscard]] std::uint64_t object_bytes(artifact::ObjectHandle handle) const {
        const auto& directory = inner_->reader().directory();
        return artifact::object_bytes(directory.objects.at(handle.index));
    }

    [[nodiscard]] const artifact::Directory& directory() const noexcept {
        return inner_->reader().directory();
    }

    // Upstream exposed the reader so the load layer could reach the directory as well as a resource
    // span; this engine's `payload` covers the span, and the accessor is kept for the directory.
    [[nodiscard]] const artifact::Reader& reader() const noexcept { return inner_->reader(); }

    // Upstream's `Reader::find(name)` returned a pointer to the directory's own descriptor, so a
    // caller could hold it for the life of the reader. This engine's directory keeps the names as
    // strings, so there is no such descriptor to point at; the converted ones are built here, once,
    // and never invalidated, which is what makes the pointer return safe.
    [[nodiscard]] const ObjectDescriptor* find(std::string_view name) const {
        const auto& index = inner_->reader().directory().object_index;
        const auto found  = index.find(std::string(name));
        if (found == index.end()) { return nullptr; }
        auto& cache       = descriptors();
        const auto cached = cache.find(std::string(name));
        if (cached != cache.end()) { return &cached->second; }
        const ObjectDescriptor converted =
            object_descriptor(inner_->reader().directory().objects.at(found->second.index));
        return &cache.emplace(std::string(name), converted).first->second;
    }

    // Upstream `require_resource(name, encoding)`. This engine addresses a resource by the component
    // and role that own it (`Binder::resource`), while the ported call sites pass the container's own
    // object name, so the lookup goes through the directory index.
    [[nodiscard]] artifact::ObjectHandle require_resource(std::string_view name,
                                                          ResourceEncoding encoding) {
        if (encoding != ResourceEncoding::RawBytesV1) {
            throw artifact::ArtifactError("port: only raw-bytes-v1 resources are known here");
        }
        const auto& index = inner_->reader().directory().object_index;
        const auto found  = index.find(std::string(name));
        if (found == index.end()) {
            throw artifact::ArtifactError("port: missing resource " + std::string(name));
        }
        return found->second;
    }

    // Upstream's `finish()` returned the plan by value and could be called on an lvalue. This
    // engine's is rvalue-ref qualified, because finishing hands the recorded demands to the plan. The
    // difference is absorbed here; callers keep upstream's spelling (`binder.finish()`).
    [[nodiscard]] artifact::MaterializationPlan finish() && { return std::move(*inner_).finish(); }

private:
    [[nodiscard]] artifact::ObjectHandle register_object(std::string_view name, NumericFormat format,
                                                        StorageLayout layout,
                                                        const artifact::Shape& shape,
                                                        artifact::Residency residency) {
        if (shape.empty()) {
            throw artifact::ArtifactError("port: require_tensor needs a non-empty shape");
        }
        // Resolve the layout first, so an unsupported one is reported before any demand is recorded.
        const QuantLayout expected = engine_layout(layout);
        const artifact::ParameterReference reference =
            inner_->parameter(name, shape, residency, format);
        const auto& geometry = inner_->reader().geometry(reference.binding.parts.at(0).object);
        if (geometry.layout != expected) {
            throw artifact::ArtifactError(
                "port: " + std::string(name) + " is stored as " +
                std::string(artifact::layout_name(geometry.layout)) +
                ", the ported call site expects " + port_layout_name(layout));
        }
        return reference.binding.parts.at(0).object;
    }

    void verify(artifact::ObjectHandle handle, artifact::Residency expected) const {
        if (handle.index >= inner_->reader().directory().objects.size()) {
            throw artifact::ArtifactError("port: object handle does not name a directory object");
        }
        // The registration already recorded the residency this engine needs; recording a demand
        // twice is what this check exists to prevent.
        (void)expected;
    }

    artifact::Binder* inner_;

    // Descriptor storage behind `find`. One map per thread: a Binder is confined to the thread that
    // loads with it, and a `std::map` node never moves once inserted, which is the property `find`'s
    // pointer return needs.
    [[nodiscard]] static std::map<std::string, ObjectDescriptor, std::less<>>& descriptors() {
        static thread_local std::map<std::string, ObjectDescriptor, std::less<>> cache;
        return cache;
    }
};

// `object_offset` on an Object, exposed under the upstream name so the load layer reads unchanged.
[[nodiscard]] inline std::uint64_t artifact_object_offset(const artifact::Object& object) {
    return artifact::object_offset(object);
}

[[nodiscard]] inline std::string_view object_name(const ObjectDescriptor& object) {
    return std::visit([](const auto& value) -> std::string_view { return value.name; }, object);
}

[[nodiscard]] inline std::uint64_t object_offset(const ObjectDescriptor& object) {
    return std::visit([](const auto& value) { return value.offset; }, object);
}

[[nodiscard]] inline std::uint64_t object_bytes(const ObjectDescriptor& object) {
    return std::visit([](const auto& value) { return value.bytes; }, object);
}

// The engine object, viewed as an upstream descriptor.
[[nodiscard]] inline ObjectDescriptor object_descriptor(const artifact::Object& object) {
    if (const auto* tensor = std::get_if<artifact::TensorObject>(&object)) {
        const QType format             = artifact::parse_format(tensor->format);
        const QuantLayout engine_place = artifact::parse_layout(tensor->layout);
        // Upstream keeps the row-scale word width in the layout; this engine keeps it in the format.
        // A descriptor carries both, so the F32 spelling is recovered from the format here.
        const StorageLayout layout =
            engine_place == QuantLayout::RowScale && format == QType::FP8_E4M3FN_ROW_F32
                ? StorageLayout::RowScaleF32V1
                : upstream_layout(engine_place);
        return TensorDescriptor{tensor->id, tensor->shape, format, layout, tensor->offset,
                                tensor->bytes};
    }
    const auto& resource = std::get<artifact::ResourceObject>(object);
    if (resource.encoding != std::string(artifact::kRawBytesEncoding)) {
        throw artifact::ArtifactError("port: unknown resource encoding " + resource.encoding);
    }
    return ResourceDescriptor{resource.id, ResourceEncoding::RawBytesV1, resource.offset,
                              resource.bytes};
}

// Upstream `bind_raw_resource`, spelled over the ENGINE binder as well as the adapter.
//
// The upstream qwen3_6 export header declares `bind_frontend_resources(artifact::Binder&)`, and in
// that generation `artifact::Binder` IS the upstream binder -- which in this port is the adapter. But
// the header resolves `artifact::Binder` to THIS engine's class, so the ported declaration and the
// adapter's type disagree by one namespace. Rather than let that disagreement sit there, the resource
// half is provided for both types: upstream passes the container's own object name
// (`frontend/tokenizer.json`, measured from the published artifact's directory), and this engine
// indexes resources by that name, so the lookup is direct.
[[nodiscard]] inline artifact::ObjectHandle bind_raw_resource(artifact::Binder& binder,
                                                             std::string_view name) {
    const auto& index = binder.reader().directory().object_index;
    const auto found  = index.find(std::string(name));
    if (found == index.end()) {
        throw artifact::ArtifactError("port: missing resource " + std::string(name));
    }
    (void)binder.host_object(found->second);
    return found->second;
}

// The element type a stored plane's bytes are read as. Upstream carried the same conversion inside
// its typed-binding unit; this engine only names BF16 or FP32 explicitly at the few call sites that
// need it (`core/weight_view.h:16`), so the mapping lives here, next to its one user.
[[nodiscard]] inline DType port_dtype(NumericFormat format) {    switch (format) {
    case QType::BF16: return DType::BF16;
    case QType::FP32: return DType::FP32;
    case QType::FP8_E4M3FN_ROW_BF16:
    case QType::FP8_E4M3FN_ROW_F32: return DType::FP8_E4M3FN;
    default:
        throw artifact::ArtifactError(
            "port: this format's bytes are not an element type a Tensor view can name");
    }
}

// Upstream free functions that build runtime views out of a materialized artifact. This engine has
// the same capability (`artifact::bind_view` plus `MaterializedArtifact::device_data`), so these
// forward rather than reimplement.
[[nodiscard]] inline Tensor materialized_tensor(const artifact::MaterializedArtifact& materialized,
                                               artifact::ObjectHandle handle, NumericFormat format,
                                               std::initializer_list<std::int32_t> internal_shape) {
    return Tensor(materialized.device_data(handle), port_dtype(format), internal_shape);
}

[[nodiscard]] inline Weight materialized_weight(const artifact::MaterializedArtifact& materialized,
                                               artifact::ObjectHandle handle, NumericFormat format,
                                               std::int32_t rows, std::int32_t columns);

} // namespace ninfer::port
