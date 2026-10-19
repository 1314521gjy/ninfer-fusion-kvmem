#include <ninfer/models/qwen3_8_flash_next/package.h>
#include <ninfer/models/qwen3_8_flash_next/runtime.h>
#include "models/qwen3_5/frontend/frontend_resources.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"

#include "artifact/reader.h"
#include "models/qwen3_8_flash_next/impl/load/bindings.h"
#include "models/qwen3_8_flash_next/impl/load/loader.h"
#include "models/qwen3_8_flash_next/impl/load/materialized.h"
#include "models/qwen3_8_flash_next/impl/program_impl.h"
#include "models/qwen3_8_flash_next/impl/runtime_plan.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_8_flash_next::detail {

class LoadPlan::Impl {
public:
    Impl(WeightsProfile weights_profile_in, ArtifactLoadPlan target_plan,
         bool quantize_output_head_fp8_in, bool quantize_token_embedding_fp8_in)
        : weights_profile(weights_profile_in), plan(std::move(target_plan)),
          quantize_output_head_fp8(quantize_output_head_fp8_in),
          quantize_token_embedding_fp8(quantize_token_embedding_fp8_in) {}

    WeightsProfile weights_profile;
    ArtifactLoadPlan plan;
    bool quantize_output_head_fp8 = false;
    bool quantize_token_embedding_fp8 = false;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadPlan::LoadPlan(LoadPlan&&) noexcept                 = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept      = default;
LoadPlan::~LoadPlan()                                   = default;

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    if (impl_ == nullptr) { throw std::logic_error("target load plan is empty"); }
    return impl_->plan.materialization;
}

class LoadedModel::Impl {
public:
    Impl(BindingPlan plan, artifact::MaterializedArtifact materialized,
         bool quantize_output_head_fp8, bool quantize_token_embedding_fp8)
        : data(std::move(plan), std::move(materialized), quantize_output_head_fp8,
               quantize_token_embedding_fp8) {}

    LoadedModelData data;
};

LoadedModel::LoadedModel(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}
LoadedModel::~LoadedModel()                                   = default;

} // namespace ninfer::models::qwen3_8_flash_next::detail

namespace ninfer::models::qwen3_8_flash_next {

// ---------------------------------------------------------------------------
// Package Static Definitions
// ---------------------------------------------------------------------------

namespace {

constexpr ModelSamplingDefaults kFlashNextDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kFlashNextDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

Package::WeightsProfile Package::resolve_weights(const artifact::ArtifactIdentity& identity) {
    detail::validate_identity(identity);
    if (identity.model_id == model_id && identity.weights_id == "mixed-nvfp4-fp8-ple-int4") {
        return WeightsProfile::MixedNvfp4Fp8PleInt4;
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' is not supported by target '" + std::string(target_key) + "'");
}

// `artifact::Binder&` -- this engine's own binder, NOT the port adapter. The compiler named the answer
// the first time and I did not read it: `bind_artifact` is declared at `bindings.h:161` taking
// `artifact::Binder&` (the `using ninfer::port::Binder;` in that header lives in ITS OWN detail scope,
// not in this one), so anything else fails at the call below with "cannot convert
// ninfer::port::Binder to ninfer::artifact::Binder&". The upstream signature was already correct; what
// I had changed in the wrong direction was the port-side DECLARATION in `package.h`, since reverted.
// Cost of the detour: one narrow probe that had recompiled `program.cpp` but not this file, plus C2511
// and C2065 before the real C2664 could speak.
Package::LoadPlan Package::plan_load(artifact::Binder& binder, const EngineOptions& options,
                                     WeightsProfile weights_profile) {
    (void)weights_profile;
    const bool enable_mtp = options.speculative.backend == SpeculativeBackend::Mtp;
    std::uint32_t draft_rows = 32'768;
    if (const char* env = std::getenv("NINFER_FLASH_NEXT_DRAFT_HEAD_ROWS"); env && env[0] != '\0') {
        draft_rows = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
    }
    // The quantize flags have to reach the binder, not just the materializer: they decide whether
    // the BF16 head/embedding is uploaded into the artifact arena or left in the file mapping.
    auto target_plan = detail::bind_artifact(
        binder, detail::LoadFeatures{
                    .vision                       = options.enable_vision,
                    .mtp                          = enable_mtp,
                    .proposal_head                = options.speculative.proposal_head,
                    .draft_head_rows              = draft_rows,
                    .quantize_output_head_fp8     = options.quantize_output_head_fp8,
                    .quantize_token_embedding_fp8 = options.quantize_token_embedding_fp8,
                });
    return LoadPlan(std::make_unique<LoadPlan::Impl>(
        weights_profile, std::move(target_plan), options.quantize_output_head_fp8,
        options.quantize_token_embedding_fp8));
}

std::unique_ptr<Package::LoadedModel>
Package::construct_loaded_model(LoadPlan&& plan, artifact::MaterializedArtifact&& materialized) {
    if (plan.impl_ == nullptr) { throw std::invalid_argument("load plan is empty"); }

    auto impl = std::make_unique<detail::LoadedModel::Impl>(
        std::move(plan.impl_->plan.bindings), std::move(materialized),
        plan.impl_->quantize_output_head_fp8, plan.impl_->quantize_token_embedding_fp8);
    plan.impl_.reset();
    return std::unique_ptr<LoadedModel>(new LoadedModel(std::move(impl)));
}

Package::Frontend Package::make_frontend(const LoadedModel& model, const EngineOptions& options) {
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    // Field for field the canonical list (`runtime/engine/model_instance.cpp:448-463`, thirteen
    // fields). The port used to pass five, which silently dropped `--chat-template` and four engine
    // options (`--thinking-budget-message`, `--vision-max-merged`, the two ngram switches and the two
    // long-anchor knobs). Every field is set here in DECLARATION order, as C++20 designated
    // initializers require (models/qwen3_5/frontend/frontend.h:23-49).
    //
    // `vision_enabled` deliberately keeps this port's own gate: `data.vision` is set only when the
    // artifact was bound with vision (`impl/load/bindings.cpp:368` ->
    // `impl/load/materialized.cpp:780`), which is `options.enable_vision` on the load path. The
    // program side still reads `options.enable_vision` (`impl/package.cpp:204`), so the same gate is
    // spelled twice -- recorded with the other latent gaps in docs/34 §7.7.
    //
    // `architecture` is label-only in this tree: `options.architecture` is read in exactly one place
    // (`models/qwen3_5/frontend/frontend.cpp:720` -> `default_sampling`), whose only
    // architecture-dependent value is `presence_penalty` for `Qwen3_5Moe` (`:664`) -- so Qwen4Exp and
    // the previous default (Qwen3_5) are identical value for value. `models/registry.h:7-10` names
    // Qwen4Exp as this model's preview architecture.
    return qwen3_5::make_frontend(
        model.impl_->data.frontend,
        qwen3_5::FrontendOptions{
            .chat_template_path       = options.chat_template_path,
            .architecture             = Architecture::Qwen4Exp,
            .vision_enabled           = model.impl_->data.vision.has_value(),
            .max_context              = options.max_context,
            .media_cache_bytes        = options.media_cache_bytes,
            .media_live_bytes         = options.media_live_bytes,
            .media_preprocess_threads = options.media_preprocess_threads,
            .vision_max_merged_tokens = options.vision_max_merged_tokens,
            .thinking_budget_message  = options.thinking_budget_message,
            .ngram_sources_enabled    = options.speculative.ngram_draft_tokens != 0,
            .ngram_archive_enabled    = options.speculative.ngram_archive_bytes != 0,
            .automatic_long_anchors =
                options.context_cache.automatic_long_anchors
                    ? options.context_cache.max_long_anchors_per_continuation.value_or(0U)
                    : 0U,
            .long_anchor_min_spacing_tokens =
                options.context_cache.long_anchor_min_spacing_tokens,
        });
}

Package::SequencePlanner Package::make_sequence_planner(DeviceContext& device,
                                                        const EngineOptions& options,
                                                        WeightsProfile weights_profile) {
    (void)device;
    (void)weights_profile;
    const std::uint32_t max_concurrency = std::clamp(options.max_concurrency, 1u, 8u);
    const bool is_mtp = options.speculative.backend == SpeculativeBackend::Mtp;
    const std::uint32_t draft_tokens = is_mtp ? options.speculative.draft_tokens : 0u;
    const std::uint32_t floor_slots =
        detail::flash_next_floor_slots(max_concurrency, draft_tokens);
    const std::uint32_t cont_cap_limit =
        detail::kMaxStateSlots > floor_slots ? (detail::kMaxStateSlots - floor_slots) : 0u;
    const std::uint32_t requested_cont =
        options.context_cache.enabled && options.context_cache.max_private_continuations
            ? *options.context_cache.max_private_continuations
            : 0u;
    const std::uint32_t cont_cap = std::min(requested_cont, cont_cap_limit);
    if (requested_cont > cont_cap_limit) {
        std::fprintf(stderr,
                     "[state-sizer] Continuation capacity clamped from %u to %u (state slot ceiling %u, floor slots %u at concurrency %u).\n",
                     requested_cont, cont_cap, detail::kMaxStateSlots, floor_slots, max_concurrency);
    }
    if (options.context_cache.enabled && cont_cap == 0) {
        std::fprintf(stderr,
                     "[state-sizer] WARNING: continuation capacity is 0 (all %u state slots reserved "
                     "for active decode lanes and rollback). Prefix reuse and context cache are disabled.\n",
                     detail::kMaxStateSlots);
    }
    if (options.kv_cache != KvCacheStorage::BFloat16 &&
        options.kv_cache != KvCacheStorage::Fp8E4M3Row256) {
        throw std::invalid_argument(
            "Flash-Next supports only --kv-dtype bf16 and fp8 (requested unsupported kv-dtype)");
    }
    const std::uint32_t total_state_slots = floor_slots + cont_cap;
    std::uint32_t draft_rows = 32'768;
    if (const char* env = std::getenv("NINFER_FLASH_NEXT_DRAFT_HEAD_ROWS"); env && env[0] != '\0') {
        draft_rows = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
    }
    detail::FlashNextRuntimeConfig config{
        .max_concurrency          = max_concurrency,
        .max_context              = options.max_context,
        .state_slot_capacity      = total_state_slots,
        .continuation_capacity    = cont_cap,
        .prefill_chunk            = options.prefill_chunk,
        .speculative_draft_tokens = draft_tokens,
        .proposal_head            = options.speculative.proposal_head,
        .draft_head_rows          = draft_rows,
        .use_cuda_graph           = options.use_cuda_graph,
        .vision_enabled           = options.enable_vision,
        .max_vision_tokens        = 4096,
        .use_qsa_prefill_mma      = options.use_qsa_prefill_mma, // G18 serve flag; dropped by the upstream merge e650ee62, restored after window 6
        .kv_cache                 = options.kv_cache,
        .gdn_state_storage        = options.gdn_state_storage,
    };
    return SequencePlanner(std::make_unique<detail::SequencePlannerImpl>(config));
}

std::unique_ptr<Package::Program>
Package::create_program(const LoadedModel& model, SequencePlan&& plan, DeviceContext& device,
                       const StartupObserver& startup_observer) {
    (void)startup_observer;
    if (model.impl_ == nullptr) { throw std::invalid_argument("loaded model is empty"); }
    if (plan.impl_ == nullptr) { throw std::invalid_argument("sequence plan is empty"); }

    auto program_impl = std::make_unique<detail::ProgramImpl>(
        &model.impl_->data, std::move(plan.impl_->plan), device);
    plan.impl_.reset();
    return std::unique_ptr<Program>(new Program(std::move(program_impl)));
}

} // namespace ninfer::models::qwen3_8_flash_next
