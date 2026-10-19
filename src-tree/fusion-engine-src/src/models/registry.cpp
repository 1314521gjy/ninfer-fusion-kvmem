#include "models/registry.h"

#include <stdexcept>
#include <string>

namespace ninfer::models {

Architecture resolve_architecture(std::string_view architecture, std::string_view model_type) {
    if (architecture == "Qwen3_5ForCausalLM" && model_type == "qwen3_5_text") {
        return Architecture::Qwen3_5;
    }
    if (architecture == "Qwen3_5MoeForCausalLM" && model_type == "qwen3_5_moe_text") {
        return Architecture::Qwen3_5Moe;
    }
    // Qwen3.8-Flash-Next carries the preview of the Qwen4 architecture. The artifact's own config
    // (Qwen4ExpForConditionalGeneration / qwen4_exp) is the value observed on the published
    // igorls/Qwen3.8-Flash-Next-mixed-NInfer container and on the official tiny fixture.
    if (architecture == "Qwen4ExpForConditionalGeneration" && model_type == "qwen4_exp") {
        return Architecture::Qwen4Exp;
    }
    throw std::invalid_argument("unsupported architecture/config pair " +
                                std::string(architecture) + "/" + std::string(model_type));
}

std::string_view architecture_name(Architecture architecture) noexcept {
    switch (architecture) {
    case Architecture::Qwen3_5:
        return "Qwen3_5ForCausalLM";
    case Architecture::Qwen3_5Moe:
        return "Qwen3_5MoeForCausalLM";
    case Architecture::Qwen4Exp:
        return "Qwen4ExpForConditionalGeneration";
    }
    return {};
}

} // namespace ninfer::models
