#include "models/qwen3_8_flash_next/impl/load/loader.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next::detail {

void validate_identity(const artifact::ArtifactIdentity& identity) {
    if (identity.model_id != kExpectedModelId || identity.weights_id != kExpectedWeightsId) {
        throw artifact::ArtifactError("FlashNextTextLoader: unsupported model identity: expected " +
                                      std::string(kExpectedIdentity) + ", got " +
                                      identity.model_id + "/" + identity.weights_id);
    }
}

FlashNextPreflightReport preflight_text_artifact(const artifact::Reader& reader,
                                                 const FlashNextRuntimeConfig& config,
                                                 std::uint32_t main_page_groups) {
    validate_identity(reader.identity());

    artifact::Binder binder(reader);
    const auto load_plan = bind_artifact(binder, LoadFeatures{.vision = false, .mtp = false});

    const auto curve = flash_next_capacity_curve(config);
    const std::uint32_t resolved_groups =
        main_page_groups == 0 ? curve.maximum_main_page_groups : main_page_groups;

    if (resolved_groups < curve.minimum_main_page_groups ||
        resolved_groups > curve.maximum_main_page_groups) {
        throw std::invalid_argument("FlashNextTextLoader: main_page_groups " +
                                    std::to_string(resolved_groups) + " is out of valid range [" +
                                    std::to_string(curve.minimum_main_page_groups) + ", " +
                                    std::to_string(curve.maximum_main_page_groups) + "]");
    }

    auto runtime_plan = finalize_flash_next_runtime_plan(config, resolved_groups);

    return FlashNextPreflightReport{
        .identity                         = reader.identity(),
        .file_bytes                       = reader.file_bytes(),
        .planned_device_weights_bytes     = load_plan.materialization.total_device_capacity_bytes(),
        .planned_device_tensors_count     = load_plan.materialization.device_objects.size(),
        .planned_retained_resources_count = load_plan.materialization.host_objects.size(),
        .planned_mapped_tensors_count     = load_plan.materialization.mapped_tensor_objects().size(),
        .runtime_plan                     = std::move(runtime_plan),
    };
}

FlashNextPreflightReport preflight_text_file(const std::filesystem::path& path,
                                             const FlashNextRuntimeConfig& config,
                                             std::uint32_t main_page_groups) {
    const artifact::Reader reader(path);
    return preflight_text_artifact(reader, config, main_page_groups);
}

StandaloneLoadedModel::StandaloneLoadedModel(BindingPlan plan, artifact::MaterializedArtifact materialized)
    : data_(std::make_unique<LoadedModelData>(std::move(plan), std::move(materialized))) {}

StandaloneLoadedModel::StandaloneLoadedModel(BindingPlan plan, artifact::MaterializedArtifact materialized,
                                             LoadQuantization quantization)
    : data_(std::make_unique<LoadedModelData>(std::move(plan), std::move(materialized),
                                              quantization.output_head_fp8)) {}

StandaloneLoadedModel StandaloneLoadedModel::load(const artifact::Reader& reader, DeviceContext& device,
                                                  LoadFeatures features, LoadQuantization quantization,
                                                  const StartupObserver& startup_observer) {
    validate_identity(reader.identity());

    artifact::Binder binder(reader);
    auto load_plan    = bind_artifact(binder, features);
    auto materialized = artifact::materialize(reader, load_plan.materialization, device, &startup_observer);

    return StandaloneLoadedModel(std::move(load_plan.bindings), std::move(materialized),
                                 quantization);
}

StandaloneLoadedModel StandaloneLoadedModel::load_from_file(const std::filesystem::path& path,
                                                            DeviceContext& device,
                                                            LoadFeatures features,
                                                            LoadQuantization quantization,
                                                            const StartupObserver& startup_observer) {
    const artifact::Reader reader(path);
    return load(reader, device, features, quantization, startup_observer);
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
