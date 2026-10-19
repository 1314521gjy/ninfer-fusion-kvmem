#pragma once

#include "artifact/framing.h"
#include "artifact/schema.h"
#include "core/weight_view.h"

#include <filesystem>
#include <memory>
#include <span>

namespace ninfer::artifact {

struct ReadSegment {
    std::size_t file_index           = 0;
    std::uint64_t file_offset        = 0;
    std::uint64_t destination_offset = 0;
    std::uint64_t bytes              = 0;
};

// Owns the cold directory and lazily opened files. Runtime views borrow materialized storage,
// never this Reader. Encodings are interpreted only for requested objects.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path);
    ~Reader();
    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;
    Reader(const Reader&)            = delete;
    Reader& operator=(const Reader&) = delete;

    [[nodiscard]] const Directory& directory() const noexcept;
    [[nodiscard]] const ArtifactId& artifact_id() const noexcept;
    // What the container says it is. Upstream exposes this as `Reader::identity()`; here the value
    // is parsed from the directory's top-level `identity` object and required to be present.
    [[nodiscard]] const ArtifactIdentity& identity() const noexcept;
    [[nodiscard]] std::uint64_t file_bytes() const noexcept;
    // Byte offset of the logical payload inside the entry file. A view into the entry's read-only
    // mapping starts at `payload_offset() + object_offset(object)`. Upstream carries this as
    // `Reader::payload_offset()` (`reader.h:182` of the reference); the port dropped it, which is why
    // `PayloadSpan::absolute_offset` could never be reconstructed here.
    [[nodiscard]] std::uint64_t payload_offset() const noexcept;
    // How many physical files the container spans (1 = entry only, no continuations). The mapping
    // path supports the single-file case and refuses the rest rather than reading the wrong bytes.
    [[nodiscard]] std::size_t file_count() const noexcept;
    [[nodiscard]] ObjectHandle find(std::string_view id) const;
    [[nodiscard]] const WeightGeometry& geometry(ObjectHandle handle) const;
    void validate_object(ObjectHandle handle) const;

    // Read-only whole-entry mapping (`CreateFileMappingW(PAGE_READONLY)` + `MapViewOfFile(FILE_MAP_READ)`
    // on Windows, `mmap(PROT_READ, MAP_PRIVATE)` elsewhere -- the reference's `reader.cpp:191-338`).
    // The lease keeps the mapping alive: a mapped tensor's bytes must outlive this Reader, which is a
    // local in `impl/load/loader.cpp` and `runtime/engine/model_instance.cpp`.
    [[nodiscard]] std::shared_ptr<const void> mapping_lease() const;
    // The object's bytes as a span into that mapping. Refuses multi-file containers instead of
    // computing an offset from the wrong base.
    [[nodiscard]] std::span<const std::byte> mapped_bytes(ObjectHandle handle) const;

    [[nodiscard]] std::vector<ReadSegment> segments(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    void read_into(std::uint64_t offset, std::span<std::byte> destination) const;
    [[nodiscard]] std::vector<std::byte> read_range(std::uint64_t offset,
                                                    std::uint64_t bytes) const;
    [[nodiscard]] std::vector<std::byte> read_object(ObjectHandle handle) const;
    [[nodiscard]] std::size_t read_direct(std::size_t file_index, std::uint64_t file_offset,
                                          std::span<std::byte> destination) const;

private:
    // Shared, not unique: `mapping_lease()` hands the same control block out as a `const void`
    // lease, and the mapping must outlive this Reader for a mapped tensor's span to stay valid.
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace ninfer::artifact
