#include "models/qwen3_8_flash_next/impl/expert_bank.h"

#include "artifact/bank_geometry.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {

bool flash_next_experts_on_host_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_EXPERTS_HOST");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool flash_next_experts_pinned_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_EXPERTS_PIN");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

std::int32_t flash_next_experts_slot_count() {
    static const std::int32_t slots = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_EXPERTS_SLOTS");
        if (env == nullptr || env[0] == '\0') { return 64; } // community value; 0 disables the cache
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed <= 0) { return 0; }
        return static_cast<std::int32_t>(std::min<long>(parsed, 512));
    }();
    return slots;
}

bool flash_next_gguf_experts_enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("NINFER_FLASH_NEXT_GGUF_EXPERTS");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

const char* flash_next_gguf_policy_conflict() noexcept {
    static const char* const switches[] = {
        "NINFER_FLASH_NEXT_CPU_EXPERT_SERVE",
        "NINFER_FLASH_NEXT_MOE_DROP_F",
        "NINFER_FLASH_NEXT_MOE_DROP_F_PREFILL",
        "NINFER_FLASH_NEXT_MOE_SUBST",
    };
    for (const char* name : switches) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0) { return name; }
    }
    return nullptr;
}

Nvfp4ExpertMatrixView Nvfp4ExpertBankView::expert(std::int32_t index) const {
    if (index < 0 || index >= experts) {
        throw std::out_of_range("NVFP4 expert index is outside the bank");
    }
    return {
        .codes                = codes + static_cast<std::uint64_t>(index) * code_bytes_per_expert,
        .scales               = scales + static_cast<std::uint64_t>(index) * scale_bytes_per_expert,
        .weight_scale_divisor = weight_scale_divisors + index,
        .input_scale_divisor  = 1.0F,
        .rows                 = rows,
        .columns              = columns,
    };
}

Nvfp4ExpertBankView make_nvfp4_expert_bank_view(const void* payload, std::uint64_t payload_bytes,
                                                std::int32_t experts, std::int32_t rows,
                                                std::int32_t columns) {
    if (payload == nullptr || experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("NVFP4 expert bank has invalid storage or shape");
    }
    const std::array<std::uint64_t, 3> shape = {static_cast<std::uint64_t>(experts),
                                                static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleBankGeometry geometry =
        artifact::block_scale_bank_geometry(artifact::NumericFormat::NVFP4, shape);
    if (payload_bytes != geometry.encoded_bytes) {
        throw std::invalid_argument("NVFP4 expert bank payload has the wrong size");
    }
    const auto* bytes = static_cast<const std::byte*>(payload);
    return {
        .codes  = bytes,
        .scales = bytes + geometry.scale_plane_offset,
        .weight_scale_divisors =
            reinterpret_cast<const float*>(bytes + geometry.weight_divisor_offset),
        .experts                = experts,
        .rows                   = rows,
        .columns                = columns,
        .code_bytes_per_expert  = geometry.code_plane_bytes / static_cast<std::uint64_t>(experts),
        .scale_bytes_per_expert = geometry.scale_plane_bytes / static_cast<std::uint64_t>(experts),
    };
}

Nvfp4ExpertBankView
materialized_nvfp4_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                    artifact::ObjectHandle handle, std::int32_t experts,
                                    std::int32_t rows, std::int32_t columns) {
    if (experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("NVFP4 expert bank has invalid shape");
    }
    const std::array<std::uint64_t, 3> shape = {static_cast<std::uint64_t>(experts),
                                                static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const artifact::BlockScaleBankGeometry geometry =
        artifact::block_scale_bank_geometry(artifact::NumericFormat::NVFP4, shape);
    return make_nvfp4_expert_bank_view(materialized.device_data(handle), geometry.encoded_bytes,
                                       experts, rows, columns);
}

GgufExpertMatrixView GgufExpertBankView::expert(std::int32_t index) const {
    if (index < 0 || index >= experts) {
        throw std::out_of_range("gguf expert index is outside the bank");
    }
    return {
        .data =
            data + static_cast<std::uint64_t>(index) * static_cast<std::uint64_t>(rows) * row_bytes,
        .rows      = rows,
        .columns   = columns,
        .row_bytes = row_bytes,
        .qtype     = qtype,
    };
}

GgufExpertBankView make_gguf_expert_bank_view(const void* payload, std::uint64_t payload_bytes,
                                              std::int32_t experts, std::int32_t rows,
                                              std::int32_t columns, QType qtype) {
    if (payload == nullptr || experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("gguf expert bank has invalid storage or shape");
    }
    // Same contract as the NVFP4 view: the payload's byte count must be EXACTLY what the block
    // geometry says, so a differently shaped authority cannot be silently reinterpreted. The
    // check runs through the single block table (core/weight.h), the same source the engine's
    // GgufBlocks geometry uses.
    const GgufBlockShape block = gguf_block_shape(qtype);
    if (!is_gguf(qtype) || block.elements <= 0 || columns % block.elements != 0) {
        throw std::invalid_argument("gguf expert bank is not whole blocks of its format");
    }
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(columns / block.elements) * static_cast<std::uint64_t>(block.bytes);
    const std::uint64_t expected =
        static_cast<std::uint64_t>(experts) * static_cast<std::uint64_t>(rows) * row_bytes;
    if (payload_bytes != expected) {
        throw std::invalid_argument("gguf expert bank payload has the wrong size");
    }
    return {
        .data    = static_cast<const std::byte*>(payload),
        .experts = experts,
        .rows    = rows,
        .columns = columns,
        .row_bytes = row_bytes,
        .qtype   = qtype,
    };
}

GgufExpertBankView
materialized_gguf_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                   artifact::ObjectHandle handle, std::int32_t experts,
                                   std::int32_t rows, std::int32_t columns, QType qtype) {
    if (experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("gguf expert bank has invalid shape");
    }
    const GgufBlockShape block = gguf_block_shape(qtype);
    if (!is_gguf(qtype) || block.elements <= 0 || columns % block.elements != 0) {
        throw std::invalid_argument("gguf expert bank is not whole blocks of its format");
    }
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(columns / block.elements) * static_cast<std::uint64_t>(block.bytes);
    const std::uint64_t expected =
        static_cast<std::uint64_t>(experts) * static_cast<std::uint64_t>(rows) * row_bytes;
    return make_gguf_expert_bank_view(materialized.device_data(handle), expected, experts, rows,
                                      columns, qtype);
}

Bf16ExpertMatrixView Bf16ExpertBankView::expert(std::int32_t index) const {
    if (index < 0 || index >= experts) {
        throw std::out_of_range("BF16 expert index is outside the bank");
    }
    return {
        .data    = data + static_cast<std::uint64_t>(index) * bytes_per_expert,
        .rows    = rows,
        .columns = columns,
    };
}

Bf16ExpertBankView make_bf16_expert_bank_view(const void* payload, std::uint64_t payload_bytes,
                                              std::int32_t experts, std::int32_t rows,
                                              std::int32_t columns) {
    if (payload == nullptr || experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("BF16 expert bank has invalid storage or shape");
    }
    const std::uint64_t bytes_per_expert =
        static_cast<std::uint64_t>(rows) * columns * sizeof(std::uint16_t);
    const std::uint64_t expected_total_bytes =
        bytes_per_expert * static_cast<std::uint64_t>(experts);
    if (payload_bytes < expected_total_bytes) {
        throw std::invalid_argument("BF16 expert bank payload has the wrong size");
    }
    return {
        .data             = static_cast<const std::byte*>(payload),
        .experts          = experts,
        .rows             = rows,
        .columns          = columns,
        .bytes_per_expert = bytes_per_expert,
    };
}

Bf16ExpertBankView
materialized_bf16_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                   artifact::ObjectHandle handle, std::int32_t experts,
                                   std::int32_t rows, std::int32_t columns) {
    if (experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("BF16 expert bank has invalid shape");
    }
    const std::uint64_t total_bytes = static_cast<std::uint64_t>(experts) * rows * columns * 2;
    return make_bf16_expert_bank_view(materialized.device_data(handle), total_bytes, experts, rows,
                                      columns);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
