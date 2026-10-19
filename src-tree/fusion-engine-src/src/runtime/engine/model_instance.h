#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/kv_capacity.h"

#include <ninfer/models/qwen3_8_flash_next/package.h>
#include <ninfer/models/qwen3_8_flash_next/target_contract.h>

#include <memory>
#include <variant>

namespace ninfer::runtime {

[[nodiscard]] EngineOptions normalize_engine_options(EngineOptions options);

// One loaded model per target family, in the shape the reference engine uses.
//
// Reference: `src/targets/registry.h:18-114` declares, for each target, a `Loaded*` struct that
// owns the loaded weights and the frontend, and an `*Instance` struct that owns the loaded struct,
// the resolved KV capacity, `capacity` and the Program; `registry.cpp:305-313` and `:345-352` are
// its two constructors. There is no shared instance type and no template over a contract - the
// engine is instantiated once per target (`engine.cpp:243-250`) and picked by
// `identity.model_id` (`registry.cpp:366-382`). This file mirrors that arrangement, because the
// two families' runtime types are not interchangeable: 14 of the ported target's 17 handle/result
// names are its own class definitions (measured, `docs/22 §4.20`), so one Program cannot satisfy
// both contracts.
//
// The fusion model keeps its own names: `models::qwen3_5::Model` is the loaded weights,
// `execution::Parameters` derives from it, and `create_program` takes those Parameters. The ported
// target's `Package` owns both its `LoadedModel` and its Parameters-free `create_program`, so its
// instance holds the `Package::LoadedModel` instead - which is also where its frontend-resource
// bundle lives, mirroring the reference's `Loaded*::frontend`.

// --- the fusion model (unchanged behaviour: this is what `construct_model` built before) ---------

struct LoadedQwen3_5 {
    std::unique_ptr<models::qwen3_5::Model> model;
    models::qwen3_5::execution::Parameters parameters;
    models::qwen3_5::Frontend frontend;

    LoadedQwen3_5(std::unique_ptr<models::qwen3_5::Model> stable_model,
                  const EngineOptions& options);

    LoadedQwen3_5(const LoadedQwen3_5&)            = delete;
    LoadedQwen3_5& operator=(const LoadedQwen3_5&) = delete;
};

struct ModelInstance {
    using ModelContract = models::qwen3_5::RuntimeTypes;

    std::unique_ptr<LoadedQwen3_5> loaded;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_5::Program> program;

    ModelInstance(std::unique_ptr<LoadedQwen3_5> stable_loaded, KvCapacityResolution resolution,
                  models::qwen3_5::SequencePlan sequence_plan, DeviceContext& device,
                  const StartupObserver& startup_observer);
    ~ModelInstance();

    ModelInstance(const ModelInstance&)            = delete;
    ModelInstance& operator=(const ModelInstance&) = delete;
};

// --- the ported Flash-Next target (Qwen3.8-Flash-Next = the Qwen4 preview) -----------------------

struct LoadedQwen4FlashNext {
    std::unique_ptr<models::qwen3_8_flash_next::Package::LoadedModel> model;
    models::qwen3_5::Frontend frontend;

    LoadedQwen4FlashNext(
        std::unique_ptr<models::qwen3_8_flash_next::Package::LoadedModel> stable_model,
        const EngineOptions& options);

    LoadedQwen4FlashNext(const LoadedQwen4FlashNext&)            = delete;
    LoadedQwen4FlashNext& operator=(const LoadedQwen4FlashNext&) = delete;
};

struct FlashNextInstance {
    using ModelContract = models::qwen3_8_flash_next::TargetContract;

    std::unique_ptr<LoadedQwen4FlashNext> loaded;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::qwen3_8_flash_next::Program> program;

    FlashNextInstance(std::unique_ptr<LoadedQwen4FlashNext> stable_loaded,
                      KvCapacityResolution resolution,
                      models::qwen3_8_flash_next::SequencePlan sequence_plan, DeviceContext& device,
                      const StartupObserver& startup_observer);
    ~FlashNextInstance();

    FlashNextInstance(const FlashNextInstance&)            = delete;
    FlashNextInstance& operator=(const FlashNextInstance&) = delete;
};

// Exactly the reference's `ActiveTarget` (`registry.h:112`): one alternative per target.
using ActiveTarget = std::variant<std::unique_ptr<ModelInstance>,
                                  std::unique_ptr<FlashNextInstance>>;

struct ConstructedModel {
    ActiveTarget active;
    LoadSummary load;
    ModelMetadata model_metadata;
    ContextMachineCostModel context_cost;
    // The options this instance was built from, with any value the model had to resolve (today the
    // single host RAM budget's Host split and long-anchor count) replaced by what the plan actually
    // uses. Carrying it keeps the Engine's copy, its ResourceManager and the frontend grid on one
    // resolved value instead of a plan that silently differs from the reported options.
    EngineOptions options;
};

[[nodiscard]] ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device);

} // namespace ninfer::runtime
