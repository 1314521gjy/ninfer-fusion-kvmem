#pragma once

// What `EngineCore<Instance>` asks of a Flash-Next instance.
//
// `EngineCore`, `ResourceManager`, `CausalScoreCore` and `MaterializationPlanner` are all
// parameterised on a contract type (`engine_core.h:51` onward takes 31 of its names from
// `typename Instance::ModelContract`). The fusion engine's own contract is
// `models::qwen3_5::RuntimeTypes`. This header is the Flash-Next counterpart: the same 36 names,
// resolving to the ported target's own types.
//
// WHY A SECOND CONTRACT INSTEAD OF ONE SHARED ONE
//   The two families do not share their runtime types. Measured 2026-10-11 in the ported
//   `runtime.h`: 14 of the 17 handle/result names are the ported target's OWN class or struct
//   definitions, not aliases of the fusion ones - only `CaptureAssessment` and
//   `ContextTransactionProgress` are true aliases. So `qwen3_5::SequenceHandle` and
//   `qwen3_8_flash_next::SequenceHandle` are two different classes, and a single Program cannot
//   satisfy both contracts. This is the same shape upstream uses: `src/targets/registry.cpp`
//   instantiates `construct_registered<Target, Loaded, Instance>` once per target and selects by
//   `identity.model_id`.
//
// THE NAMES THAT DO COME FROM THE FUSION TREE
//   `PreparedPrompt`, `NgramArchive`, `OutputSession`, `PublishedOutput` and `Frontend` are the
//   engine's own and are shared deliberately: the controller layer speaks them, and the ported
//   target was adapted onto them (its `runtime.h` includes
//   `<models/qwen3_5/frontend/prepared_prompt.h>`). `HybridAdmissionQuote` is declared here rather
//   than taken from the target because a Flash-Next instance refuses the hybrid cache tier, but the
//   contract still has to name the type for `HybridResourceManager` to compile.

#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/program/program.h"
#include "models/qwen3_5/program/runtime_types.h"

#include <ninfer/models/qwen3_8_flash_next/runtime.h>

namespace ninfer::models::qwen3_8_flash_next {

// The contract `EngineCore<FlashNextInstance>` reads. Every name below is spelled exactly as
// `models::qwen3_5::RuntimeTypes` spells it; only the resolution differs, and only where the ported
// target genuinely has its own type.
struct TargetContract {
    // Shared with the engine: the controller layer and the ported target both speak these.
    using Frontend        = qwen3_5::Frontend;
    using PreparedPrompt  = qwen3_5::PreparedPrompt;
    using NgramArchive    = qwen3_5::NgramArchive;
    using OutputSession   = qwen3_5::OutputSession;
    using PublishedOutput = qwen3_5::PublishedOutput;

    // The ported target's own runtime types. These are the 14 measured to be independent
    // definitions rather than aliases.
    using SequencePlanner    = qwen3_8_flash_next::SequencePlanner;
    using SequencePlan       = qwen3_8_flash_next::SequencePlan;
    using RequestBasePlan    = qwen3_8_flash_next::RequestBasePlan;
    using AdmissionCandidate = qwen3_8_flash_next::AdmissionCandidate;
    using ResourcePlan       = qwen3_8_flash_next::ResourcePlan;
    using PersistentBackfillProof = qwen3_8_flash_next::PersistentBackfillProof;
    using SequenceHandle          = qwen3_8_flash_next::SequenceHandle;
    using ContinuationHandle      = qwen3_8_flash_next::ContinuationHandle;
    using SharedPrefixHandle      = qwen3_8_flash_next::SharedPrefixHandle;
    using CaptureOffer            = qwen3_8_flash_next::CaptureOffer;
    using PendingBatch            = qwen3_8_flash_next::PendingBatch;
    using PressurePlanningSession = qwen3_8_flash_next::PressurePlanningSession;
    using CapturePressurePlanningSession =
        qwen3_8_flash_next::CapturePressurePlanningSession;
    using PressureTargetHandle    = qwen3_5::PressureTargetHandle;
    using AssessedPressureTarget  = qwen3_8_flash_next::AssessedPressureTarget;
    using CapturePressurePlan     = qwen3_8_flash_next::CapturePressurePlan;
    using MaterializationResult   = qwen3_8_flash_next::MaterializationResult;
    using ContextTransactionProgress = qwen3_8_flash_next::ContextTransactionProgress;
    using CaptureAssessment          = qwen3_8_flash_next::CaptureAssessment;
    using ActiveCaptureResult        = qwen3_8_flash_next::ActiveCaptureResult;
    using StartResult                = qwen3_8_flash_next::StartResult;
    using PrefillProgress            = qwen3_8_flash_next::PrefillProgress;
    using CommitResult               = qwen3_8_flash_next::CommitResult;
    using DiscardResult              = qwen3_8_flash_next::DiscardResult;
    using FinishResult               = qwen3_8_flash_next::FinishResult;
    using AbortResult                = qwen3_8_flash_next::AbortResult;
    using ReleaseResult              = qwen3_8_flash_next::ReleaseResult;

    // The controller's admission cost model speaks the fusion summaries; the ported target
    // re-exports both, so they resolve to the engine's definitions.
    using CacheSessionKey          = qwen3_5::PreparedSessionKey;
    using ContinuationSummary      = qwen3_5::ContinuationSummary;
    using SharedPrefixSummary      = qwen3_5::SharedPrefixSummary;

    // Hybrid prefix cache: the contract must name these for `HybridResourceManager` to compile.
    // A Flash-Next instance never selects that manager (its loader leaves the hybrid tier off) and
    // its Program refuses the tier, so nothing is expected to call them with real work.
    using HybridAdmissionQuote = qwen3_5::HybridAdmissionQuote;

    using Program = qwen3_8_flash_next::Program;
};

} // namespace ninfer::models::qwen3_8_flash_next
