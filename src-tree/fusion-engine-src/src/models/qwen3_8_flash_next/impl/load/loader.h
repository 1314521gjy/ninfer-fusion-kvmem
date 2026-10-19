#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen3_5/frontend/frontend_resources.h"
#include "models/qwen3_8_flash_next/impl/load/bindings.h"
#include "models/qwen3_8_flash_next/impl/load/materialized.h"
#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/ple_index.h"
#include "models/qwen3_8_flash_next/impl/runtime_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

inline constexpr std::string_view kExpectedModelId   = "qwen3.8-flash-next";
inline constexpr std::string_view kExpectedWeightsId = "mixed-nvfp4-fp8-ple-int4";
inline constexpr std::string_view kExpectedIdentity = "qwen3.8-flash-next/mixed-nvfp4-fp8-ple-int4";

void validate_identity(const artifact::ArtifactIdentity& identity);

struct LoadQuantization {
    bool output_head_fp8 = false;
};

struct FlashNextPreflightReport {
    artifact::ArtifactIdentity identity;
    std::uint64_t file_bytes                     = 0;
    std::uint64_t planned_device_weights_bytes   = 0;
    std::size_t planned_device_tensors_count     = 0;
    std::size_t planned_retained_resources_count = 0;
    std::size_t planned_mapped_tensors_count     = 0;
    FlashNextRuntimePlan runtime_plan;
};

// Inspects artifact without allocating device memory for weights.
[[nodiscard]] FlashNextPreflightReport preflight_text_artifact(const artifact::Reader& reader,
                                                               const FlashNextRuntimeConfig& config,
                                                               std::uint32_t main_page_groups = 0);

[[nodiscard]] FlashNextPreflightReport preflight_text_file(const std::filesystem::path& path,
                                                           const FlashNextRuntimeConfig& config,
                                                           std::uint32_t main_page_groups = 0);

class StandaloneLoadedModel {
public:
    ~StandaloneLoadedModel() = default;

    StandaloneLoadedModel(const StandaloneLoadedModel&)                = delete;
    StandaloneLoadedModel& operator=(const StandaloneLoadedModel&)     = delete;
    StandaloneLoadedModel(StandaloneLoadedModel&&) noexcept            = default;
    StandaloneLoadedModel& operator=(StandaloneLoadedModel&&) noexcept = default;

    [[nodiscard]] static StandaloneLoadedModel load(const artifact::Reader& reader,
                                                    DeviceContext& device,
                                                    LoadFeatures features                  = {},
                                                    LoadQuantization quantization          = {},
                                                    const StartupObserver& startup_observer = {});

    [[nodiscard]] static StandaloneLoadedModel
    load_from_file(const std::filesystem::path& path, DeviceContext& device,
                   LoadFeatures features                  = {},
                   LoadQuantization quantization          = {},
                   const StartupObserver& startup_observer = {});

    [[nodiscard]] const TextModelView& text_view() const noexcept { return data_->text; }

    [[nodiscard]] bool has_vision() const noexcept { return data_->vision.has_value(); }

    [[nodiscard]] const VisionModelView& vision_view() const {
        if (!data_->vision) {
            throw std::logic_error(
                "StandaloneLoadedModel: vision was not materialized for this instance");
        }
        return *data_->vision;
    }

    [[nodiscard]] const PleIndexMetadata& ple_metadata() const noexcept { return ple_metadata_; }

    [[nodiscard]] const qwen3_5::FrontendResources& frontend_resources() const noexcept {
        return data_->frontend;
    }

    [[nodiscard]] const artifact::MaterializedArtifact& backing() const noexcept {
        return data_->backing;
    }

    [[nodiscard]] const artifact::MaterializationStats& stats() const noexcept {
        return data_->backing.stats();
    }

private:
    StandaloneLoadedModel(BindingPlan plan, artifact::MaterializedArtifact materialized);
    StandaloneLoadedModel(BindingPlan plan, artifact::MaterializedArtifact materialized,
                          LoadQuantization quantization);

    std::unique_ptr<LoadedModelData> data_;
    PleIndexMetadata ple_metadata_{kPleIndexMetadata};
};

} // namespace ninfer::models::qwen3_8_flash_next::detail
