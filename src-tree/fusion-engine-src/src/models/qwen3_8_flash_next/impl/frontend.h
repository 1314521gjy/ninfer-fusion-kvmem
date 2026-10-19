#pragma once

#include "models/qwen3_8_flash_next/impl/load/loader.h"
#include "models/qwen3_5/frontend/frontend.h"

namespace ninfer::models::qwen3_8_flash_next::detail {

[[nodiscard]] qwen3_5::Frontend make_frontend(const StandaloneLoadedModel& model,
                                              const qwen3_5::FrontendOptions& options);

} // namespace ninfer::models::qwen3_8_flash_next::detail