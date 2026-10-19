#include "models/qwen3_8_flash_next/impl/frontend.h"

namespace ninfer::models::qwen3_8_flash_next::detail {

qwen3_5::Frontend make_frontend(const StandaloneLoadedModel& model,
                                const qwen3_5::FrontendOptions& options) {
    return qwen3_5::make_frontend(model.frontend_resources(), options);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail