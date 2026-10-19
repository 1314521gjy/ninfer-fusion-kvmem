#pragma once

#include <string_view>

namespace ninfer::models {

// Qwen4Exp: the Qwen3.8-Flash-Next preview architecture (model_type "qwen4_exp"). Appended last so
// no existing enumerator is renumbered -- the value is written into startup measurements
// (measurement.cpp) and compared across runs.
enum class Architecture { Qwen3_5, Qwen3_5Moe, Qwen4Exp };

[[nodiscard]] Architecture resolve_architecture(std::string_view architecture,
                                                std::string_view model_type);
[[nodiscard]] std::string_view architecture_name(Architecture architecture) noexcept;

} // namespace ninfer::models
