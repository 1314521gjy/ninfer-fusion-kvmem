#include "artifact/reader.h"

#include "artifact/file_io.h"
#include "artifact/formats.h"
#include "artifact/layouts.h"

#include <algorithm>
#include <array>
#include <limits>
#include <system_error>

// WIN32_LEAN_AND_MEAN/NOMINMAX before windows.h: the read-only mapping is the only Win32 call this
// file needs, and the lean header avoids the min/max macros colliding with <algorithm>.
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {

// The reference maps the entry once and every mapped tensor is a span into that mapping
// (`reader.cpp:191-338` of the reference). Same shape here, except the mapping is created on first
// use: a container that never asks for mapped bytes never pays for one, and a multi-file container
// is refused before anything is mapped.
class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
#if defined(_WIN32)
        file_ = ::CreateFileW(path.wstring().c_str(), GENERIC_READ,
                              FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::system_error(::GetLastError(), std::generic_category(),
                                    "open " + path.string());
        }
        LARGE_INTEGER size{};
        if (!::GetFileSizeEx(file_, &size) || size.QuadPart <= 0 ||
            static_cast<std::uintmax_t>(size.QuadPart) >
                std::numeric_limits<std::size_t>::max()) {
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw ArtifactError(path.string() + ": artifact size does not fit the address space");
        }
        mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_ == nullptr) {
            const DWORD error = ::GetLastError();
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(error, std::generic_category(),
                                    "CreateFileMapping " + path.string());
        }
        data_ = static_cast<const std::byte*>(::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (data_ == nullptr) {
            const DWORD error = ::GetLastError();
            ::CloseHandle(mapping_);
            ::CloseHandle(file_);
            mapping_ = nullptr;
            file_    = INVALID_HANDLE_VALUE;
            throw std::system_error(error, std::generic_category(),
                                    "MapViewOfFile " + path.string());
        }
        size_ = static_cast<std::size_t>(size.QuadPart);
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }
        struct stat status {};
        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size <= 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError(path.string() + ": artifact size does not fit the address space");
        }
        map_ = ::mmap(nullptr, static_cast<std::size_t>(status.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
        if (map_ == MAP_FAILED) {
            const int error = errno;
            ::close(fd);
            map_ = nullptr;
            throw std::system_error(error, std::generic_category(), "mmap " + path.string());
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(map_);
        size_ = static_cast<std::size_t>(status.st_size);
#endif
    }

    ~MappedFile() {
#if defined(_WIN32)
        if (data_ != nullptr) { ::UnmapViewOfFile(data_); }
        if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
        if (file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(file_); }
#else
        if (map_ != nullptr) { ::munmap(map_, size_); }
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }

private:
#if defined(_WIN32)
    HANDLE file_    = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
    void* map_ = nullptr;
#endif
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

struct Reader::Impl {
    std::filesystem::path entry;
    Directory directory;
    ArtifactId id{};
    std::uint64_t entry_payload_start = 0;
    std::uint64_t file_bytes          = 0;
    mutable std::vector<std::unique_ptr<InputFile>> files;
    mutable std::vector<std::optional<WeightGeometry>> geometries;
    // Lazily created by `mapped()`; the lease handed out by `Reader::mapping_lease()` shares this
    // Impl's control block, so the view stays mapped until the last lease is dropped.
    mutable std::unique_ptr<MappedFile> mapping;

    // The entry's read-only mapping. A multi-file container is refused here rather than reading an
    // object from the wrong base: only file 0 starts at `entry_payload_start`, continuations start
    // at `kPayloadAlignment`.
    const MappedFile& mapped() const {
        if (!mapping) {
            if (directory.files.size() != 1) {
                throw ArtifactError(entry.string() + ": mapped weights need a single-file container (" +
                                    std::to_string(directory.files.size()) + " files)");
            }
            mapping = std::make_unique<MappedFile>(entry);
        }
        return *mapping;
    }

    explicit Impl(const std::filesystem::path& path) : entry(path) {
        auto file = std::make_unique<InputFile>(entry);
        std::array<std::byte, kHeaderBytes> header{};
        file->read_exact(0, header);
        if (!std::equal(kEntryMagic.begin(), kEntryMagic.end(), header.begin())) {
            if (std::equal(kEntryMagic.begin(), kEntryMagic.end() - 1, header.begin()) &&
                header[kEntryMagic.size() - 1] == std::byte{2}) {
                throw ArtifactError(
                    entry.string() +
                    ": NInfer v2 artifact is not supported. Upgrade to v3 with: "
                    "python3 tools/upgrade_ninfer_v2_to_v3.py INPUT.ninfer OUTPUT.ninfer");
            }
            throw ArtifactError(entry.string() + ": expected NInfer v3 entry magic");
        }
        const auto json_bytes = read_u64_le(header.data() + 8);
        if (!json_bytes || json_bytes > file->bytes() - kHeaderBytes ||
            json_bytes > std::numeric_limits<std::size_t>::max()) {
            throw ArtifactError(entry.string() + ": invalid directory length");
        }
        std::copy_n(header.begin() + 16, id.size(), id.begin());
        std::string text(static_cast<std::size_t>(json_bytes), '\0');
        file->read_exact(kHeaderBytes, std::as_writable_bytes(std::span(text.data(), text.size())));
        directory =
            parse_directory(parse_json(text, "artifact directory"), entry.filename().string());
        entry_payload_start = align_up(checked_add(kHeaderBytes, json_bytes, "metadata end"),
                                       kPayloadAlignment, "entry payload start");
        const auto expected =
            checked_add(entry_payload_start, directory.files[0].payload_bytes, "entry length");
        if (expected != file->bytes()) {
            throw ArtifactError(entry.string() + ": entry length differs from directory");
        }
        file_bytes = checked_add(entry_payload_start, directory.payload_bytes, "file bytes");
        file_bytes = checked_add(
            file_bytes,
            checked_mul(directory.files.size() - 1, kPayloadAlignment, "continuation headers"),
            "file bytes");
        files.resize(directory.files.size());
        files[0] = std::move(file);
        geometries.resize(directory.objects.size());
    }

    InputFile& file(std::size_t index) const {
        if (index >= files.size()) { throw ArtifactError("invalid continuation index"); }
        if (!files[index]) {
            const auto& record = directory.files[index];
            auto input         = std::make_unique<InputFile>(entry.parent_path() / *record.path);
            std::array<std::byte, kHeaderBytes> header{};
            input->read_exact(0, header);
            if (!std::equal(kPartMagic.begin(), kPartMagic.end(), header.begin()) ||
                read_u64_le(header.data() + 8) != index ||
                !std::equal(id.begin(), id.end(), header.begin() + 16)) {
                throw ArtifactError(*record.path +
                                    ": continuation header does not belong to this entry");
            }
            if (input->bytes() !=
                checked_add(kPayloadAlignment, record.payload_bytes, "part length")) {
                throw ArtifactError(*record.path + ": continuation length differs from directory");
            }
            files[index] = std::move(input);
        }
        return *files[index];
    }
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_shared<Impl>(path)) {}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const Directory& Reader::directory() const noexcept { return impl_->directory; }

const ArtifactId& Reader::artifact_id() const noexcept { return impl_->id; }

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->directory.identity; }

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file_bytes; }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->entry_payload_start; }

std::size_t Reader::file_count() const noexcept { return impl_->directory.files.size(); }

std::shared_ptr<const void> Reader::mapping_lease() const {
    (void)impl_->mapped(); // create it now, so a refused container fails at the call site
    return std::static_pointer_cast<const void>(impl_);
}

std::span<const std::byte> Reader::mapped_bytes(ObjectHandle handle) const {
    const auto& object = directory().object(handle);
    const std::uint64_t begin =
        checked_add(impl_->entry_payload_start, object_offset(object), "mapped object offset");
    const std::uint64_t bytes = object_bytes(object);
    const MappedFile& file    = impl_->mapped();
    if (begin > file.size() || bytes > file.size() - begin) {
        throw ArtifactError(object_id(object) + ": mapped object lies outside the entry mapping");
    }
    return {file.data() + begin, static_cast<std::size_t>(bytes)};
}

ObjectHandle Reader::find(std::string_view id) const {
    const auto found = directory().object_index.find(id);
    if (found == directory().object_index.end()) {
        throw ArtifactError("missing object " + std::string(id));
    }
    return found->second;
}

const WeightGeometry& Reader::geometry(ObjectHandle handle) const {
    const auto& tensor = directory().tensor(handle);
    auto& result       = impl_->geometries[handle.index];
    if (!result) { result = describe_tensor(tensor); }
    return *result;
}

void Reader::validate_object(ObjectHandle handle) const {
    const auto& object = directory().object(handle);
    if (std::holds_alternative<TensorObject>(object)) {
        (void)geometry(handle);
    } else if (std::get<ResourceObject>(object).encoding != kRawBytesEncoding) {
        throw ArtifactError(object_id(object) + ": unsupported resource encoding");
    }
}

std::vector<ReadSegment> Reader::segments(std::uint64_t offset, std::uint64_t bytes) const {
    const auto& data = directory();
    if (offset > data.payload_bytes || bytes > data.payload_bytes - offset) {
        throw ArtifactError("read range exceeds logical payload");
    }
    std::vector<ReadSegment> result;
    if (!bytes) { return result; }
    const auto found     = std::upper_bound(data.files.begin(), data.files.end(), offset,
                                            [](std::uint64_t position, const FileRecord& file) {
                                            return position < file.logical_begin;
                                        });
    std::size_t index    = static_cast<std::size_t>(found - data.files.begin() - 1);
    std::uint64_t copied = 0;
    while (copied < bytes) {
        const auto& file = data.files[index];
        const auto local = offset + copied - file.logical_begin;
        const auto count = std::min(bytes - copied, file.payload_bytes - local);
        const auto start = index == 0 ? impl_->entry_payload_start : kPayloadAlignment;
        result.push_back({index, checked_add(start, local, "file offset"), copied, count});
        copied += count;
        ++index;
    }
    return result;
}

void Reader::read_into(std::uint64_t offset, std::span<std::byte> destination) const {
    for (const auto& segment : segments(offset, destination.size())) {
        impl_->file(segment.file_index)
            .read_exact(segment.file_offset,
                        destination.subspan(static_cast<std::size_t>(segment.destination_offset),
                                            static_cast<std::size_t>(segment.bytes)));
    }
}

std::vector<std::byte> Reader::read_range(std::uint64_t offset, std::uint64_t bytes) const {
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw ArtifactError("Host read size exceeds size_t");
    }
    (void)segments(offset, bytes);
    std::vector<std::byte> result(static_cast<std::size_t>(bytes));
    read_into(offset, result);
    return result;
}

std::vector<std::byte> Reader::read_object(ObjectHandle handle) const {
    validate_object(handle);
    const auto& object = directory().object(handle);
    return read_range(object_offset(object), object_bytes(object));
}

std::size_t Reader::read_direct(std::size_t file_index, std::uint64_t file_offset,
                                std::span<std::byte> destination) const {
    return impl_->file(file_index).read_direct(file_offset, destination);
}

} // namespace ninfer::artifact
