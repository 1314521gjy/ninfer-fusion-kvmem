#pragma once

#include "artifact/materializer.h"
#include "core/arena.h"
#include "core/tensor.h"
#include "models/qwen3_8_flash_next/impl/load/bindings.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

class LoadedModelData {
public:
    LoadedModelData(BindingPlan plan, artifact::MaterializedArtifact materialized,
                    bool quantize_output_head_fp8 = false,
                    bool quantize_token_embedding_fp8 = false);

    LoadedModelData(const LoadedModelData&)            = delete;
    LoadedModelData& operator=(const LoadedModelData&) = delete;
    LoadedModelData(LoadedModelData&&)                 = delete;
    LoadedModelData& operator=(LoadedModelData&&)      = delete;

    artifact::MaterializedArtifact backing;
    DeviceBuffer output_head_fp8;
    DeviceBuffer token_embedding_fp8;
    DeviceBuffer mtp_expert_gate_up_nvfp4;
    DeviceBuffer mtp_expert_down_nvfp4;
    DeviceBuffer proposal_head_payload;
    DeviceBuffer proposal_head_fp8;
    DeviceBuffer proposal_token_ids;
    qwen3_5::FrontendResources frontend;
    TextModelView text;
    std::optional<VisionModelView> vision;
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
