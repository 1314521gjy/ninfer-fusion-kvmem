#pragma once

// Copied from upstream `src/artifact/typed_binding.h` (igorls/ninfer @ c81c88f6) with this engine's
// spelling of the two enumerations. The five entry points and their argument order are upstream's,
// verbatim: the ported Flash-Next load layer calls them by these names.
//
// One difference to be aware of when reading this against upstream: `NumericFormat` here is an alias
// of this engine's `QType`, and upstream's `StorageLayout` corresponds to this engine's
// `QuantLayout`. Neither is a second enumeration -- see `artifact/bank_geometry.h`.

#include "artifact/binder.h"
#include "artifact/bank_geometry.h" // NumericFormat
#include "core/tensor.h"
// Upstream's own header would not reach into a model target; this one has to, because the upstream
// surface it declares (`require_tensor`, `TensorPlacement`, `materialize_on_device`) is presented by
// the port adapter that lives beside the ported target. This header is also where the upstream names
// are declared once, so a ported file activates them by including it.
#include "models/qwen3_8_flash_next/impl/port/binder_port.h"

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace ninfer::portalias {

using port::NumericFormat;
using port::StorageLayout;
using port::TensorPlacement;
using port::ResourceEncoding;
using port::PayloadSpan;
using port::materialized_tensor;
using port::materialized_weight;
using port::artifact_object_offset;
// `GdnStateStorage` deliberately has no alias line: it is declared in `ninfer` itself (beside
// `KvCacheStorage`), so ordinary enclosing-namespace lookup already reaches it from here, and the
// port only re-exports the same type under `port::`.
using port::gdn_state_fp16_from;
using port::gdn_state_storage_from;
// The upstream descriptor triple. `ninfer::artifact` has no member of these names -- its directory
// keeps formats and layouts as strings and parses on demand -- so these are new names here rather
// than a shadowing of existing ones, which is the case that forced `Binder` to stay unaliased.
using port::TensorDescriptor;
using port::ResourceDescriptor;
using port::ObjectDescriptor;
using port::object_name;
using port::object_descriptor;
// The descriptor overloads. `object_offset` / `object_bytes` have engine-side overloads that take an
// `Object`; these take an `ObjectDescriptor`, so both are reachable by the same name.
using port::object_offset;
using port::object_bytes;

} // namespace ninfer::portalias

namespace ninfer::artifact {

class MaterializedArtifact;

// Places `name` on the device (or validates it only, for `port::TensorPlacement::ValidateOnly`) and
// returns the object that carries it.
[[nodiscard]] ObjectHandle bind_tensor(port::Binder& binder, std::string_view name,
                                       NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       port::TensorPlacement placement);

[[nodiscard]] ObjectHandle bind_device_tensor(port::Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);

[[nodiscard]] ObjectHandle bind_raw_resource(port::Binder& binder, std::string_view name);

// A `Tensor` over the object's device bytes, read as the format's element type. Throws for a format
// whose bytes are not an element type a Tensor view can name.
[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

// A `Weight` over the object's device bytes, laid out as the format's encoding dictates. `rows` and
// `columns` are the logical shape; the byte extents come from the format's geometry.
[[nodiscard]] Weight materialized_weight(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::int32_t rows, std::int32_t columns);

} // namespace ninfer::artifact
