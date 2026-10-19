#include "models/qwen3_8_flash_next/impl/ple_index.h"

#include <bit>
#include <cstddef>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

std::uint64_t wrapped_product(std::int64_t lhs, std::int64_t rhs) noexcept {
    return static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs);
}

std::int64_t positive_remainder(std::uint64_t bits, std::int64_t divisor) {
    if (divisor <= 0) { throw std::invalid_argument("PLE head vocabulary must be positive"); }
    const std::int64_t value = std::bit_cast<std::int64_t>(bits);
    std::int64_t result      = value % divisor;
    if (result < 0) { result += divisor; }
    return result;
}

} // namespace

PleTokenHistory::PleTokenHistory() noexcept
    : previous_(kPleBoundaryTokenId), second_previous_(kPleBoundaryTokenId) {}

void PleTokenHistory::commit(std::int64_t token) noexcept {
    if (token == kPleBoundaryTokenId) {
        previous_        = kPleBoundaryTokenId;
        second_previous_ = kPleBoundaryTokenId;
        return;
    }
    second_previous_ = previous_;
    previous_        = token;
}

std::array<std::int64_t, 16> ple_indices(const PleIndexMetadata& metadata,
                                         const PleTokenHistory& history, std::int64_t token) {
    const std::uint64_t bigram = wrapped_product(token, metadata.multipliers[0]) ^
                                 wrapped_product(history.previous_token(), metadata.multipliers[1]);
    const std::uint64_t trigram =
        bigram ^ wrapped_product(history.second_previous_token(), metadata.multipliers[2]);

    std::array<std::int64_t, 16> out{};
    for (std::size_t head = 0; head < out.size(); ++head) {
        const std::uint64_t hash  = head < 8 ? bigram : trigram;
        const std::int64_t offset = metadata.head_offsets[head];
        if (offset < 0) { throw std::invalid_argument("PLE head offset must be non-negative"); }
        out[head] = offset + positive_remainder(hash, metadata.head_vocab_sizes[head]);
    }
    return out;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
