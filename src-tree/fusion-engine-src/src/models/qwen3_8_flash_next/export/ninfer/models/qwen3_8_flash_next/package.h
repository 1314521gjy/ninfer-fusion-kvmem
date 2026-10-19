#pragma once

#include "artifact/binder.h"
// `plan_load` takes the port adapter (`ninfer::port::Binder`), so the declaration needs its
// definition. Included the way the other consumer writes the path (`artifact/typed_binding.h:18`).
#include "models/qwen3_8_flash_next/impl/port/binder_port.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include <models/qwen3_5/frontend/frontend.h>
#include <models/qwen3_5/frontend/frontend_resources.h>
#include <models/qwen3_5/frontend/prepared_prompt.h>
#include <ninfer/models/qwen3_8_flash_next/runtime.h>

#include <memory>
#include <string_view>

namespace ninfer {
namespace models::qwen3_8_flash_next {

struct Package;

namespace detail {

enum class WeightsProfile {
    MixedNvfp4Fp8PleInt4,
};

class LoadPlan {
public:
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    ~LoadPlan();

    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;

public:
    class Impl;
    explicit LoadPlan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_flash_next::Package;
};

class LoadedModel {
public:
    ~LoadedModel();

    LoadedModel(LoadedModel&&)                 = delete;
    LoadedModel& operator=(LoadedModel&&)      = delete;
    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;

public:
    class Impl;
    explicit LoadedModel(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend struct qwen3_8_flash_next::Package;
};

} // namespace detail

struct Package {
    static constexpr std::string_view model_id   = "qwen3.8-flash-next";
    static constexpr std::string_view target_key = "qwen3_8_flash_next";

    using WeightsProfile             = detail::WeightsProfile;
    using LoadPlan                   = detail::LoadPlan;
    using LoadedModel                = detail::LoadedModel;
    using Frontend                   = qwen3_5::Frontend;
    using PreparedPrompt             = qwen3_5::PreparedPrompt;
    using OutputSession              = qwen3_5::OutputSession;
    using PublishedOutput            = qwen3_5::PublishedOutput;
    using SequencePlanner            = qwen3_8_flash_next::SequencePlanner;
    using SequencePlan               = qwen3_8_flash_next::SequencePlan;
    using RequestBasePlan            = qwen3_8_flash_next::RequestBasePlan;
    using AdmissionCandidate         = qwen3_8_flash_next::AdmissionCandidate;
    using ResourcePlan               = qwen3_8_flash_next::ResourcePlan;
    using PersistentBackfillProof    = qwen3_8_flash_next::PersistentBackfillProof;
    using SequenceHandle             = qwen3_8_flash_next::SequenceHandle;
    using ContinuationHandle         = qwen3_8_flash_next::ContinuationHandle;
    using SharedPrefixHandle         = qwen3_8_flash_next::SharedPrefixHandle;
    using CaptureOffer               = qwen3_8_flash_next::CaptureOffer;
    using CacheSessionKey            = qwen3_5::PreparedSessionKey;
    using ContinuationSummary        = qwen3_5::ContinuationSummary;
    using SharedPrefixSummary        = qwen3_5::SharedPrefixSummary;
    using PressurePlanningSession        = qwen3_8_flash_next::PressurePlanningSession;
    using PressureTargetHandle           = qwen3_5::PressureTargetHandle;
    using AssessedPressureTarget         = qwen3_8_flash_next::AssessedPressureTarget;
    using CapturePressurePlan            = qwen3_8_flash_next::CapturePressurePlan;
    using CapturePressurePlanningSession = qwen3_8_flash_next::CapturePressurePlanningSession;
    using MaterializationResult      = qwen3_8_flash_next::MaterializationResult;
    using ContextTransactionProgress = qwen3_8_flash_next::ContextTransactionProgress;
    using CaptureAssessment          = qwen3_5::CaptureAssessment;
    using ActiveCaptureResult        = qwen3_8_flash_next::ActiveCaptureResult;
    using PendingBatch               = qwen3_8_flash_next::PendingBatch;
    using StartResult                = qwen3_8_flash_next::StartResult;
    using PrefillProgress            = qwen3_8_flash_next::PrefillProgress;
    using CommitResult               = qwen3_8_flash_next::CommitResult;
    using DiscardResult              = qwen3_8_flash_next::DiscardResult;
    using FinishResult               = qwen3_8_flash_next::FinishResult;
    using AbortResult                = qwen3_8_flash_next::AbortResult;
    using ReleaseResult              = qwen3_8_flash_next::ReleaseResult;
    using Program                    = qwen3_8_flash_next::Program;

    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    [[nodiscard]] static WeightsProfile resolve_weights(const artifact::ArtifactIdentity& identity);
    // `artifact::Binder&`: this engine's own binder. Upstream wrote exactly this and it is correct --
    // `bind_artifact` (`impl/load/bindings.h:161`) takes the engine binder too, and the port adapter
    // is built INSIDE that layer (`impl/port/facade.h:6-13` describes the call-site edit, but the
    // call site here is `bind_artifact`, not this facade). Measured: changing this declaration to
    // `ninfer::port::Binder` produced C2511 (definition no longer matched) and C2664 (the adapter
    // cannot convert to the engine binder at `package.cpp`'s `bind_artifact` call). Reverted.
    [[nodiscard]] static LoadPlan plan_load(artifact::Binder& binder, const EngineOptions& options,
                                            WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model,
                                                const EngineOptions& options);
    [[nodiscard]] static SequencePlanner make_sequence_planner(DeviceContext& device,
                                                               const EngineOptions& options,
                                                               WeightsProfile weights_profile);
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                   const StartupObserver& startup_observer = {});
};

} // namespace models::qwen3_8_flash_next
} // namespace ninfer
