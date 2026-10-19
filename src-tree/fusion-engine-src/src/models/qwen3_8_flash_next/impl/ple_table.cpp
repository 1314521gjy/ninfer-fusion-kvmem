#include "models/qwen3_8_flash_next/impl/ple_table.h"

#include "artifact/file_io.h" // block-7 alignment ④: InputFile::read_direct (unbuffered, aligned)

#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

constexpr std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

float fp16_to_float(std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    std::uint32_t exponent   = (bits >> 10U) & 0x1FU;
    std::uint32_t mantissa   = bits & 0x03FFU;
    std::uint32_t fp32       = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            fp32 = sign;
        } else {
            std::int32_t unbiased = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --unbiased;
            }
            mantissa &= 0x03FFU;
            fp32 = sign | (static_cast<std::uint32_t>(unbiased + 127) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        fp32 = sign | 0x7F80'0000U | (mantissa << 13U);
    } else {
        fp32 = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    return std::bit_cast<float>(fp32);
}

std::uint16_t float_to_bf16(float value) noexcept {
    std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F80'0000U) == 0x7F80'0000U && (bits & 0x007F'FFFFU) != 0) {
        return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
    }
    const std::uint32_t tie = (bits >> 16U) & 1U;
    return static_cast<std::uint16_t>((bits + 0x7FFFU + tie) >> 16U);
}

float dequantize_ple_value(const PleShardView& shard, std::uint64_t row,
                           std::uint64_t column) noexcept {
    const std::uint64_t code_row_offset = row * shard.width / 2;
    const std::uint8_t packed =
        std::to_integer<std::uint8_t>(shard.codes[code_row_offset + column / 2]);
    const std::uint8_t code =
        column % 2 == 0 ? packed & 0x0FU : static_cast<std::uint8_t>(packed >> 4U);
    const std::uint64_t scale_row_offset = row * shard.groups_per_row * 2;
    const std::uint64_t scale_offset     = scale_row_offset + (column / 16) * 2;
    std::uint16_t scale_bits             = 0;
    std::memcpy(&scale_bits, shard.scales.data() + scale_offset, sizeof(scale_bits));
    return (static_cast<float>(code) - 8.0F) * fp16_to_float(scale_bits);
}

} // namespace

PleRowAddress locate_ple_row(std::uint64_t global_row) {
    if (global_row >= kPleTableRows) { throw std::out_of_range("PLE row is outside the table"); }
    return {
        .shard = static_cast<std::uint32_t>(global_row / kPleRowsPerShard),
        .row   = static_cast<std::uint32_t>(global_row % kPleRowsPerShard),
    };
}

PleShardView make_ple_shard_view(std::span<const std::byte> encoded, std::uint64_t rows,
                                 std::uint64_t width) {
    if (rows == 0 || width == 0 || width % 16 != 0 ||
        rows > std::numeric_limits<std::uint64_t>::max() / width) {
        throw std::invalid_argument("invalid PLE shard geometry");
    }
    const std::uint64_t elements     = rows * width;
    const std::uint64_t code_bytes   = elements / 2;
    const std::uint64_t scale_offset = align_up(code_bytes, 256);
    const std::uint64_t scale_bytes  = rows * (width / 16) * 2;
    if (scale_offset > encoded.size() || scale_bytes > encoded.size() - scale_offset ||
        scale_offset + scale_bytes != encoded.size()) {
        throw std::invalid_argument("PLE shard payload does not match packed-u4-g16-v1");
    }
    return {
        .codes          = encoded.first(static_cast<std::size_t>(code_bytes)),
        .scales         = encoded.subspan(static_cast<std::size_t>(scale_offset),
                                          static_cast<std::size_t>(scale_bytes)),
        .rows           = rows,
        .width          = width,
        .groups_per_row = width / 16,
    };
}

void dequantize_ple_row(const PleShardView& shard, std::uint64_t row, std::span<float> output) {
    if (row >= shard.rows) { throw std::out_of_range("PLE row is outside the shard"); }
    if (output.size() != shard.width) {
        throw std::invalid_argument("PLE row output has the wrong width");
    }
    for (std::uint64_t column = 0; column < shard.width; ++column) {
        output[column] = dequantize_ple_value(shard, row, column);
    }
}

void gather_ple_rows(const PleTableView& table, std::span<const std::int64_t, 16> global_rows,
                     std::span<float> output) {
    constexpr std::size_t kOutputWidth = 16 * kPleRowWidth;
    if (output.size() != kOutputWidth) {
        throw std::invalid_argument("PLE gather output must contain 2560 values");
    }
    for (std::size_t head = 0; head < global_rows.size(); ++head) {
        if (global_rows[head] < 0) { throw std::out_of_range("PLE row must be non-negative"); }
        const PleRowAddress address = locate_ple_row(static_cast<std::uint64_t>(global_rows[head]));
        dequantize_ple_row(table.shards[address.shard], address.row,
                           output.subspan(head * kPleRowWidth, kPleRowWidth));
    }
}

void gather_ple_rows_bf16(const PleTableView& table, std::span<const std::int64_t, 16> global_rows,
                          std::span<std::uint16_t> output) {
    constexpr std::size_t kOutputWidth = 16 * kPleRowWidth;
    if (output.size() != kOutputWidth) {
        throw std::invalid_argument("PLE BF16 gather output must contain 2560 values");
    }
    for (std::size_t head = 0; head < global_rows.size(); ++head) {
        if (global_rows[head] < 0) { throw std::out_of_range("PLE row must be non-negative"); }
        const PleRowAddress address = locate_ple_row(static_cast<std::uint64_t>(global_rows[head]));
        const PleShardView& shard   = table.shards[address.shard];
        for (std::uint64_t column = 0; column < kPleRowWidth; ++column) {
            output[head * kPleRowWidth + column] =
                float_to_bf16(dequantize_ple_value(shard, address.row, column));
        }
    }
}

void gather_ple_rows_compressed(const PleTableView& table,
                                std::span<const std::array<std::int64_t, 16>> global_rows,
                                std::span<std::byte> out_codes,
                                std::span<std::byte> out_scales) {
    const std::size_t tokens            = global_rows.size();
    constexpr std::size_t kCodesPerRow  = 80;
    constexpr std::size_t kScalesPerRow = 20;
    if (out_codes.size() < tokens * 16 * kCodesPerRow ||
        out_scales.size() < tokens * 16 * kScalesPerRow) {
        throw std::invalid_argument("PLE compressed gather output buffer is too small");
    }

    auto* codes_ptr  = reinterpret_cast<std::uint8_t*>(out_codes.data());
    auto* scales_ptr = reinterpret_cast<std::uint8_t*>(out_scales.data());

    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t head = 0; head < 16; ++head) {
            const std::int64_t g_row = global_rows[t][head];
            if (g_row < 0) { throw std::out_of_range("PLE row must be non-negative"); }
            const PleRowAddress addr  = locate_ple_row(static_cast<std::uint64_t>(g_row));
            const PleShardView& shard = table.shards[addr.shard];
            const std::size_t row_idx = t * 16 + head;

            const std::size_t code_src_offset = addr.row * kCodesPerRow;
            std::memcpy(codes_ptr + row_idx * kCodesPerRow,
                        shard.codes.data() + code_src_offset,
                        kCodesPerRow);

            const std::size_t scale_src_offset = addr.row * kScalesPerRow;
            std::memcpy(scales_ptr + row_idx * kScalesPerRow,
                        shard.scales.data() + scale_src_offset,
                        kScalesPerRow);
        }
    }
}

// ---- block-7 alignment ④: the unbuffered direct read face ---------------------------------------
//
// One row's codes (80 B) or scales (20 B) is read through the aligned 4096-byte window that
// contains it: the same window arithmetic the loader uses for payload blocks, applied to a row
// that can start anywhere. `InputFile::read_direct` is positional (OVERLAPPED on Windows) but one
// handle serves ONE outstanding read at a time -- concurrent gather workers each hold their own
// InputFile (see PleDirectReadIo in ple_table.h). A window that runs past EOF returns short --
// allowed by the file layer -- and the bounds check below is what turns "short of the row we need"
// into a loud failure instead of zero bytes.

namespace {

void read_ple_row_bytes_direct(const PleShardView& shard, std::uint64_t row,
                               const PleDirectReadIo& io, std::uint8_t* codes_out,
                               std::uint8_t* scales_out) {
    constexpr std::size_t kCodesPerRow  = 80;
    constexpr std::size_t kScalesPerRow = 20;
    if (io.file == nullptr || io.window == nullptr || io.window_bytes < 8192) {
        throw std::invalid_argument("PLE direct read: missing or undersized window buffer (needs "
                                    "8192 bytes: a row window can span two 4096-byte blocks)");
    }
    if (shard.file_offset == 0) {
        throw std::runtime_error(
            "PLE direct read: shard file offset is unset (load_ple did not record one)");
    }
    // Same in-shard layout make_ple_shard_view validated: codes, 256-aligned, then scales.
    const std::uint64_t scale_base = align_up(shard.rows * (shard.width / 2), 256);
    const auto window_read         = [&io](std::uint64_t position, std::uint8_t* dst,
                                   std::size_t len) {
        const std::uint64_t lo = position & ~std::uint64_t(4095);
        const std::uint64_t hi = (position + len + 4095) & ~std::uint64_t(4095);
        const std::size_t   n  = static_cast<std::size_t>(hi - lo);
        if (n > io.window_bytes) {
            throw std::invalid_argument("PLE direct read: window buffer is too small");
        }
        const std::size_t got = io.file->read_direct(lo, std::span<std::byte>(io.window, n));
        const std::size_t need =
            static_cast<std::size_t>(position - lo) + len;
        if (got < need) {
            throw std::runtime_error("PLE direct read: short read at file position " +
                                     std::to_string(position));
        }
        std::memcpy(dst, io.window + (position - lo), len);
    };
    window_read(shard.file_offset + row * kCodesPerRow, codes_out, kCodesPerRow);
    window_read(shard.file_offset + scale_base + row * kScalesPerRow, scales_out, kScalesPerRow);
}

} // namespace

void gather_ple_rows_compressed_direct(const PleTableView& table, const PleDirectReadIo& io,
                                       std::span<const std::array<std::int64_t, 16>> global_rows,
                                       std::span<std::byte> out_codes,
                                       std::span<std::byte> out_scales) {
    const std::size_t tokens            = global_rows.size();
    constexpr std::size_t kCodesPerRow  = 80;
    constexpr std::size_t kScalesPerRow = 20;
    if (out_codes.size() < tokens * 16 * kCodesPerRow ||
        out_scales.size() < tokens * 16 * kScalesPerRow) {
        throw std::invalid_argument("PLE compressed gather output buffer is too small");
    }

    auto* codes_ptr  = reinterpret_cast<std::uint8_t*>(out_codes.data());
    auto* scales_ptr = reinterpret_cast<std::uint8_t*>(out_scales.data());

    std::uint8_t row_codes[kCodesPerRow];
    std::uint8_t row_scales[kScalesPerRow];
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t head = 0; head < 16; ++head) {
            const std::int64_t g_row = global_rows[t][head];
            if (g_row < 0) { throw std::out_of_range("PLE row must be non-negative"); }
            const PleRowAddress addr  = locate_ple_row(static_cast<std::uint64_t>(g_row));
            const PleShardView& shard = table.shards[addr.shard];
            const std::size_t row_idx = t * 16 + head;

            read_ple_row_bytes_direct(shard, addr.row, io, row_codes, row_scales);
            std::memcpy(codes_ptr + row_idx * kCodesPerRow, row_codes, kCodesPerRow);
            std::memcpy(scales_ptr + row_idx * kScalesPerRow, row_scales, kScalesPerRow);
        }
    }
}

void gather_ple_rows_bf16_direct(const PleTableView& table, const PleDirectReadIo& io,
                                 std::span<const std::int64_t, 16> global_rows,
                                 std::span<std::uint16_t> output) {
    constexpr std::size_t kOutputWidth = 16 * kPleRowWidth;
    if (output.size() != kOutputWidth) {
        throw std::invalid_argument("PLE BF16 gather output must contain 2560 values");
    }
    std::uint8_t row_codes[80];
    std::uint8_t row_scales[20];
    for (std::size_t head = 0; head < global_rows.size(); ++head) {
        if (global_rows[head] < 0) { throw std::out_of_range("PLE row must be non-negative"); }
        const PleRowAddress address = locate_ple_row(static_cast<std::uint64_t>(global_rows[head]));
        const PleShardView& shard   = table.shards[address.shard];
        read_ple_row_bytes_direct(shard, address.row, io, row_codes, row_scales);
        // The SAME dequantizer the mapped path runs, over a one-row view of the freshly read
        // bytes: the bits entering float_to_bf16 are identical by construction.
        const PleShardView row_view{
            .codes = std::span<const std::byte>(reinterpret_cast<const std::byte*>(row_codes),
                                                sizeof(row_codes)),
            .scales = std::span<const std::byte>(reinterpret_cast<const std::byte*>(row_scales),
                                                 sizeof(row_scales)),
            .rows           = 1,
            .width          = kPleRowWidth,
            .groups_per_row = kPleRowWidth / 16,
            .file_offset    = 0,
        };
        for (std::uint64_t column = 0; column < kPleRowWidth; ++column) {
            output[head * kPleRowWidth + column] =
                float_to_bf16(dequantize_ple_value(row_view, 0, column));
        }
    }
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
