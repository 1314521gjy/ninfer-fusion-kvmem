#pragma once

#include <cstddef>
#include <cstdint>
#include <array>
#include <span>

namespace ninfer::artifact {
class InputFile;
} // namespace ninfer::artifact

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr std::uint64_t kPleRowsPerShard = 2'500'012;
inline constexpr std::uint64_t kPleShardCount   = 128;
inline constexpr std::uint64_t kPleTableRows    = kPleRowsPerShard * kPleShardCount;
inline constexpr std::uint64_t kPleRowWidth     = 160;

struct PleRowAddress {
    std::uint32_t shard;
    std::uint32_t row;
};

[[nodiscard]] PleRowAddress locate_ple_row(std::uint64_t global_row);

struct PleShardView {
    std::span<const std::byte> codes;
    std::span<const std::byte> scales;
    std::uint64_t rows;
    std::uint64_t width;
    std::uint64_t groups_per_row;
    // Block-7 alignment ④: absolute offset of this shard's payload (the `codes` byte above) in the
    // artifact's entry file. Filled by load_ple from the container's own mapping record; 0 means
    // "unknown" and the direct-read path refuses rather than guessing. The mapped spans stay the
    // source of truth for every existing caller -- this field only feeds the optional unbuffered
    // read path (NINFER_FLASH_NEXT_PLE_IO_DIRECT), which never dereferences the mapping.
    std::uint64_t file_offset = 0;
};

struct PleTableView {
    std::array<PleShardView, kPleShardCount> shards;
};

// Block-7 alignment ④: one direct (unbuffered) read of a byte range through an aligned 4096-byte
// window. `file` must be the artifact entry file opened as `artifact::InputFile`; each handle
// serves ONE outstanding direct read at a time, so concurrent gather workers must each hold their
// own InputFile (the Win32 direct handle is OVERLAPPED with hEvent = NULL, and concurrent
// outstanding reads through one such handle silently misreport completion -- measured by
// _ple-dirprobe; per-handle or per-request-event usage is unaffected). `window` is caller-owned
// scratch of at least 8192 bytes and needs 4096-byte alignment: a row that starts near the end of
// a 4096-byte block is read through the window that spans the two blocks it crosses.
struct PleDirectReadIo {
    const artifact::InputFile* file = nullptr;
    std::byte* window               = nullptr;
    std::size_t window_bytes        = 0;
};

[[nodiscard]] PleShardView make_ple_shard_view(std::span<const std::byte> encoded,
                                               std::uint64_t rows  = kPleRowsPerShard,
                                               std::uint64_t width = kPleRowWidth);

// Independent scalar oracle for U4Z8 group-16 with FP16 multipliers.
void dequantize_ple_row(const PleShardView& shard, std::uint64_t row, std::span<float> output);

void gather_ple_rows(const PleTableView& table, std::span<const std::int64_t, 16> global_rows,
                     std::span<float> output);

// Runtime gather contract: FP32 dequantization followed by round-to-nearest-even BF16 storage.
void gather_ple_rows_bf16(const PleTableView& table, std::span<const std::int64_t, 16> global_rows,
                          std::span<std::uint16_t> output);

// Runtime compressed gather: gathers raw U4 codes and FP16 scales into pinned staging buffers.
void gather_ple_rows_compressed(const PleTableView& table,
                                std::span<const std::array<std::int64_t, 16>> global_rows,
                                std::span<std::byte> out_codes,
                                std::span<std::byte> out_scales);

// Block-7 alignment ④: the two gather faces read through the file instead of the mapping. Byte
// layout of the outputs is IDENTICAL to the mapped versions above (the same 80-byte code rows and
// 20-byte scale rows are copied/written in the same order), so a direct-read run and a mapped run
// are byte-comparable row by row. Requires every shard's `file_offset` to be non-zero (load_ple
// fills it; a zero shard is refused loudly, never read from the wrong base).
void gather_ple_rows_compressed_direct(const PleTableView& table, const PleDirectReadIo& io,
                                       std::span<const std::array<std::int64_t, 16>> global_rows,
                                       std::span<std::byte> out_codes,
                                       std::span<std::byte> out_scales);

void gather_ple_rows_bf16_direct(const PleTableView& table, const PleDirectReadIo& io,
                                 std::span<const std::int64_t, 16> global_rows,
                                 std::span<std::uint16_t> output);

} // namespace ninfer::models::qwen3_8_flash_next::detail
