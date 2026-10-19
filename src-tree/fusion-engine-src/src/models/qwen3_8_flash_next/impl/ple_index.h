#pragma once

#include <array>
#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr std::array<std::int64_t, 3> kPleLayerMultipliers = {
    23'703'573'157'769LL, 20'109'073'645'365LL, 8'052'911'324'071LL};

inline constexpr std::array<std::int64_t, 16> kPleHeadOffsets = {
    0,           20'000'003,  40'000'026,  60'000'059,  80'000'106,  100'000'165,
    120'000'228, 140'000'297, 160'000'374, 180'000'455, 200'000'548, 220'000'655,
    240'000'802, 260'000'955, 280'001'114, 300'001'275};

inline constexpr std::array<std::int64_t, 16> kPleHeadVocabSizes = {
    20'000'003, 20'000'023, 20'000'033, 20'000'047, 20'000'059, 20'000'063, 20'000'069, 20'000'077,
    20'000'081, 20'000'093, 20'000'107, 20'000'147, 20'000'153, 20'000'159, 20'000'161, 20'000'171};

struct PleIndexMetadata {
    std::array<std::int64_t, 3> multipliers;
    std::array<std::int64_t, 16> head_offsets;
    std::array<std::int64_t, 16> head_vocab_sizes;
};

inline constexpr PleIndexMetadata kPleIndexMetadata{
    .multipliers      = kPleLayerMultipliers,
    .head_offsets     = kPleHeadOffsets,
    .head_vocab_sizes = kPleHeadVocabSizes,
};

inline constexpr std::int64_t kPleBoundaryTokenId = 248'044;

class PleTokenHistory {
public:
    PleTokenHistory() noexcept;

    void commit(std::int64_t token) noexcept;

    std::int64_t previous_token() const noexcept { return previous_; }

    std::int64_t second_previous_token() const noexcept { return second_previous_; }

private:
    std::int64_t previous_;
    std::int64_t second_previous_;
};

// Computes the eight bigram and eight trigram embedding rows for one token.
// The history is deliberately not mutated: speculative callers commit only
// tokens accepted by the output transaction.
[[nodiscard]] std::array<std::int64_t, 16>
ple_indices(const PleIndexMetadata& metadata, const PleTokenHistory& history, std::int64_t token);

} // namespace ninfer::models::qwen3_8_flash_next::detail
