#include "runtime/engine/model_instance.h"
#include "calibration/device_calibration.h"
#include "core/arena.h"
#include "ops/common/device_route.h"
#include "runtime/engine/context_cache/context_cost.h"
#include "runtime/engine/device_profile.h"
#include "runtime/engine/diagnostics.h"
#include "artifact/reader.h"
#include "artifact/formats.h"
#include "core/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/measurement.h"
// The ported facade takes the *upstream* `artifact::Binder`, and this generation names that type
// differently, so the assembler reaches it through the adapter. Include written the way the other
// consumer writes it (`src/artifact/typed_binding.h:18`).
#include "models/qwen3_8_flash_next/impl/port/binder_port.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <chrono>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.cuda_graph_allowance_bytes != 0 && !options.use_cuda_graph) {
        throw std::invalid_argument(
            "Engine cuda_graph_allowance_bytes requires CUDA graphs to be enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
    if (!std::isfinite(options.rope_yarn_factor) || options.rope_yarn_factor < 1.0F ||
        options.rope_yarn_factor > 4.0F) {
        throw std::invalid_argument("Engine rope_yarn_factor must be finite and in [1,4]");
    }
    if (!std::isfinite(options.rope_scaling_factor) || options.rope_scaling_factor < 1.0F ||
        options.rope_scaling_factor > 32.0F) {
        throw std::invalid_argument("Engine rope_scaling_factor must be finite and in [1,32]");
    }
    if (options.rope_scaling_factor > 1.0F &&
        (options.rope_yarn || options.rope_yarn_factor > 1.0F)) {
        throw std::invalid_argument(
            "Engine rope_scaling_factor excludes rope_yarn and rope_yarn_factor");
    }
}

// The hybrid index ranks admission sources and values snapshots with the same calibrated
// prefill and Host-to-Device coefficients the Legacy ResourceManager prices materialization with.
// Uncalibrated (zero) terms keep the index's generic defaults.
prefix_cache::CacheCostModel hybrid_cache_cost(const ContextMachineCostModel& model) {
    constexpr double kSecondsPerNs = 1.0e-9;
    constexpr double kQ32          = 4294967296.0;
    prefix_cache::CacheCostModel cost;
    if (model.prefill.chunk_ns != 0) {
        cost.chunk_seconds = static_cast<double>(model.prefill.chunk_ns) * kSecondsPerNs;
    }
    if (model.prefill.token_ns_q32 != 0) {
        cost.token_seconds = static_cast<double>(model.prefill.token_ns_q32) / kQ32 * kSecondsPerNs;
    }
    if (model.prefill.attention_pair_ns_q32 != 0) {
        cost.attention_pair_seconds =
            static_cast<double>(model.prefill.attention_pair_ns_q32) / kQ32 * kSecondsPerNs;
    }
    const ContextTransferCost& h2d =
        model.transfer[static_cast<std::size_t>(ContextTransferDirection::HostToDevice)];
    if (h2d.ns_per_byte_q32 != 0) {
        cost.h2d_bytes_per_second =
            1.0 / (static_cast<double>(h2d.ns_per_byte_q32) / kQ32 * kSecondsPerNs);
    }
    if (h2d.batch_ns != 0) {
        cost.transfer_batch_seconds = static_cast<double>(h2d.batch_ns) * kSecondsPerNs;
    }
    return cost;
}

// Everything the bytes of a persisted hybrid Host tier depend on besides its geometry (which the
// file records itself): the exact artifact, its execution signature, the KV and speculative
// formats, RoPE scaling and the product binary's build identity. Any difference makes the saved
// state meaningless, so the file is ignored.
std::string hybrid_cache_fingerprint(const EngineOptions& options, const std::string& signature) {
    std::error_code error;
    const auto size = std::filesystem::file_size(options.artifact_path, error);
    const auto time = std::filesystem::last_write_time(options.artifact_path, error);
    std::string out = "artifact=" + std::filesystem::absolute(options.artifact_path).string();
    out += ";size=" + std::to_string(error ? 0U : size);
    out += ";mtime=" +
           std::to_string(error ? 0LL : static_cast<long long>(time.time_since_epoch().count()));
    out += ";signature=" + signature;
    out += ";kv=" + std::to_string(static_cast<int>(options.kv_cache));
    out += ";speculative=" + std::to_string(static_cast<int>(options.speculative.backend));
    out += ";yarn=" + std::to_string(options.rope_yarn_factor);
    out += ";interpolation=" + std::to_string(options.rope_scaling_factor) + "@" +
           std::to_string(options.rope_scaling_original_context);
    out += ";build=" + options.context_cache.hybrid.persistent_identity;
    return out;
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// Free memory on each rank's device, in rank order. Ranks that share a physical device (a test mode
// that exercises the multi-stage path on one card) split what is free between them, since each
// one's budget is spent from the same memory. Under the WDDM evictable budget the primary device
// may also count what other processes hold beyond the desktop floor: all of it but the resident
// weights.
std::vector<std::size_t> free_bytes_by_rank(const DeviceContext& device,
                                            bool evictable_budget      = false,
                                            std::size_t resident_bytes = 0) {
    std::vector<std::size_t> out;
    for (std::size_t rank = 0; rank < device.size(); ++rank) {
        std::size_t sharing = 0;
        for (std::size_t other = 0; other < device.size(); ++other) {
            if (device.same_physical_device(rank, other)) { ++sharing; }
        }
        DeviceBinding bind(device.rank(rank).device);
        std::size_t free_bytes = current_free_device_bytes();
        if (evictable_budget && rank == 0) {
            constexpr std::size_t kDesktopFloor = 512ULL << 20;
            std::size_t free_now                = 0;
            std::size_t total                   = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_now, &total));
            if (total > resident_bytes + kDesktopFloor) {
                free_bytes = std::max(free_bytes, total - resident_bytes - kDesktopFloor);
            }
        }
        out.push_back(free_bytes / sharing);
    }
    return out;
}

// Disk-tier pages are keyed by token digests, which say nothing about the weights that produced
// them: each artifact gets its own directory so one model can never restore another's KV.
EngineOptions artifact_scoped_disk_tier(const EngineOptions& options,
                                        const artifact::ArtifactId& artifact) {
    EngineOptions scoped = options;
    if (scoped.context_cache.disk_kv_path.empty()) { return scoped; }
    std::string name = "artifact_";
    for (const std::byte byte : artifact) {
        char hex[3];
        std::snprintf(hex, sizeof(hex), "%02x", static_cast<unsigned>(byte));
        name += hex;
    }
    scoped.context_cache.disk_kv_path /= name;
    return scoped;
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.prefill_chunk        = 1024;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{.enabled = false};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache = options.context_cache;
    if (options.speculative.ngram_archive_bytes != 0 &&
        options.speculative.ngram_draft_tokens == 0) {
        throw std::invalid_argument(
            "the cross-request ngram archive requires ngram drafting; enable --ngram-draft-tokens");
    }
    if (options.speculative.ngram_archive_bytes != 0 &&
        (options.speculative.ngram_session_bytes < (1ULL << 20) ||
         options.speculative.ngram_session_bytes > options.speculative.ngram_archive_bytes)) {
        throw std::invalid_argument("ngram session capacity must be between 1 MiB and the total "
                                    "archive capacity");
    }
    // The GDN conv-record workspace admits at most 16 verification columns once a batch holds
    // more than one request.
    if (options.speculative.ngram_draft_tokens > 15 && options.max_concurrency != 1) {
        throw std::invalid_argument("ngram draft widths above 15 require engine concurrency one");
    }
    const std::uint32_t concurrency = options.max_concurrency;
    if (cache.enabled && cache.mode == ContextCacheMode::Hybrid) {
        if (cache.device_state_slots || cache.max_private_continuations ||
            cache.max_shared_prefixes || cache.max_long_anchors_per_continuation) {
            throw std::invalid_argument(
                "the hybrid prefix cache does not accept Legacy capacity options (Device state "
                "slots, private/shared catalogs, long anchors)");
        }
        if (cache.rolling_retention || cache.release_diverged_checkpoints ||
            cache.thorough_admission_search || cache.recency_eviction || cache.value_aware_demote ||
            cache.automatic_long_anchors || !cache.disk_kv_path.empty()) {
            throw std::invalid_argument(
                "the hybrid prefix cache does not accept Legacy cache policies (rolling retention, "
                "diverged-checkpoint release, thorough admission search, recency eviction, "
                "value-aware demotion, automatic long anchors, disk tier)");
        }
        // One pinned Host slab pool serves blocks and snapshots alike; its size is the only
        // capacity a deployment has to choose (docs/maintainer/hybrid-prefix-cache-spec.md §5.4).
        cache.host_cache_budget_bytes =
            cache.host_cache_budget_bytes.value_or(kDefaultHybridHostCacheBytes);
        const bool host_tier             = *cache.host_cache_budget_bytes != 0;
        HybridPrefixCacheOptions& hybrid = cache.hybrid;
        // One resident snapshot per request lane keeps every live conversation's latest
        // endpoint restorable without PCIe traffic; one more slot stages taps and endpoints while
        // their Host copies are written. Without a Host tier these slots are the only snapshot
        // storage, so one more is kept for shared prefixes.
        hybrid.device_snapshot_slots =
            hybrid.device_snapshot_slots.value_or(concurrency + (host_tier ? 1U : 2U));
        // Taps without a Host tier would evict other conversations' resident snapshots.
        hybrid.max_new_taps = hybrid.max_new_taps.value_or(host_tier ? 8U : 2U);
        // Ladder taps are realized on prefill chunk boundaries, so the ladder never refines below
        // the chunk: coarser ladders only waste snapshots on taps that share one boundary.
        const std::uint32_t chunk = std::max<std::uint32_t>(options.prefill_chunk, 64U);
        hybrid.tap_ladder_tokens =
            hybrid.tap_ladder_tokens.value_or(std::max<std::uint32_t>(4096U, 2U * chunk));
        hybrid.tap_min_gap_tokens =
            hybrid.tap_min_gap_tokens.value_or(std::max<std::uint32_t>(1024U, chunk));
        if (*hybrid.device_snapshot_slots == 0 || *hybrid.device_snapshot_slots > 64) {
            throw std::invalid_argument("hybrid device snapshot slots must be in [1,64]");
        }
        if (*hybrid.max_new_taps > 64) {
            throw std::invalid_argument("hybrid taps per request must be at most 64");
        }
        if (*hybrid.tap_ladder_tokens < 64 || *hybrid.tap_min_gap_tokens < 64) {
            throw std::invalid_argument(
                "hybrid tap ladder and minimum gap must be at least 64 tokens");
        }
        // The hybrid tree keeps no catalog of owners: every lane holds one active continuation
        // and retained context lives in the Program's prefix index. The Legacy Host pools are
        // replaced by the hybrid slab pool.
        cache.device_state_slots                = hybrid.device_snapshot_slots;
        cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        return options;
    }
    if (!cache.enabled) {
        cache.mode = ContextCacheMode::Legacy;
        if ((cache.device_state_slots && *cache.device_state_slots != 0) ||
            (cache.max_private_continuations && *cache.max_private_continuations != concurrency) ||
            (cache.max_shared_prefixes && *cache.max_shared_prefixes != 0) ||
            (cache.max_long_anchors_per_continuation &&
             *cache.max_long_anchors_per_continuation != 0) ||
            cache.host_cache_budget_bytes) {
            throw std::invalid_argument("disabled context cache accepts only root-only capacities");
        }
        cache.device_state_slots                = 0;
        cache.host_state_slots                  = 0;
        cache.host_kv_capacity_bytes            = 0;
        cache.max_private_continuations         = concurrency;
        cache.max_shared_prefixes               = 0;
        cache.max_long_anchors_per_continuation = 0;
        cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);
        // Without a catalog no continuation is ever evicted or restored.
        cache.disk_kv_path.clear();
        cache.disk_kv_restore       = false;
        cache.disk_kv_directstorage = false;
        cache.rolling_retention            = false;
        cache.release_diverged_checkpoints = false;
        cache.thorough_admission_search    = false;
        cache.recency_eviction             = false;
        cache.value_aware_demote           = false;
        cache.kv_lease_growth              = false;
        cache.automatic_long_anchors       = false;
        return options;
    }

    cache.device_state_slots            = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t default_private = 2ULL * concurrency;
    cache.max_private_continuations =
        cache.max_private_continuations.value_or(static_cast<std::uint32_t>(default_private));
    cache.max_shared_prefixes = cache.max_shared_prefixes.value_or(std::max(
        concurrency, static_cast<std::uint32_t>(kMaximumPreparedPromptCacheCandidatesPerRequest)));
    cache.max_long_anchors_per_continuation =
        cache.max_long_anchors_per_continuation.value_or(cache.automatic_long_anchors ? 4U : 2U);
    cache.max_cache_markers_per_request     = cache.max_cache_markers_per_request.value_or(4U);

    if (*cache.max_private_continuations < concurrency) {
        throw std::invalid_argument(
            "context cache max_private_continuations must cover every active request");
    }
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    const std::uint64_t address_spaces =
        static_cast<std::uint64_t>(*cache.max_private_continuations) + *cache.max_shared_prefixes;
    if (address_spaces > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache address-space capacity exceeds uint32");
    }
    if (*cache.max_long_anchors_per_continuation != 0 &&
        *cache.max_private_continuations >
            std::numeric_limits<std::size_t>::max() / *cache.max_long_anchors_per_continuation) {
        throw std::overflow_error("context cache long-anchor capacity exceeds size_t");
    }
    return options;
}

namespace {

// Installs each rank's GPU route profile before any Op runs (see EngineOptions::device_profile).
// A device with no measured profile is calibrated once, in this process, before the weights claim
// its memory; the result goes to the profile file for the next start.
void install_device_route_profile_on(const EngineOptions& options, int device) {
    if (options.device_profile == "off") {
        ops::install_device_route_profile(device, nullptr);
        return;
    }
    const bool calibrate = options.device_profile == "calibrate";
    if (!calibrate && options.device_profile != "auto") {
        throw std::invalid_argument("device_profile must be auto, off or calibrate");
    }
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    const std::string hardware_class = context_cost_hardware_class(props.name, props.major, props.minor);
    const int multiprocessors        = props.multiProcessorCount;
    const std::filesystem::path path = options.device_profile_path.empty()
                                           ? default_device_profile_path()
                                           : options.device_profile_path;
    std::optional<ops::DeviceRouteProfile> profile;
    if (!calibrate) {
        try {
            profile = find_device_route_profile(hardware_class, multiprocessors, path);
        } catch (const std::exception& error) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                               "device profile %s ignored: %s", path.string().c_str(),
                               error.what());
        }
    }
    if (!profile) {
        publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                           "calibrating routes for %s (%d SMs)", hardware_class.c_str(),
                           multiprocessors);
        int previous = 0;
        CUDA_CHECK(cudaGetDevice(&previous));
        CUDA_CHECK(cudaSetDevice(device));
        ops::install_device_route_profile(device, nullptr);
        calibration::CalibrationOptions calibration_options;
        profile = calibration::calibrate_device_routes(calibration_options);
        CUDA_CHECK(cudaSetDevice(previous));
        profile->hardware_class = hardware_class;
        try {
            upsert_device_route_profile_atomic(path, *profile);
        } catch (const std::exception& error) {
            publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Warning,
                               "device profile not saved to %s: %s", path.string().c_str(),
                               error.what());
        }
    }
    publish_diagnostic(options.diagnostic_observer, DiagnosticLevel::Info,
                       "device profile %s: %zu routed keys (%s)", hardware_class.c_str(),
                       profile->routes.size(), profile->origin.c_str());
    ops::install_device_route_profile(
        device, std::make_shared<const ops::DeviceRouteProfile>(std::move(*profile)));
}

void install_device_route_profile_for(const EngineOptions& options, const DeviceContext& device) {
    std::vector<int> installed;
    for (const int id : device.device_ids()) {
        if (std::find(installed.begin(), installed.end(), id) != installed.end()) { continue; }
        install_device_route_profile_on(options, id);
        installed.push_back(id);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// The two target families' loaders and instances.
//
// Reference shape: `src/targets/registry.cpp:299-356` - a `Loaded*` constructor that takes the
// loaded weights and the options, and an `*Instance` constructor that takes the loaded struct, the
// resolved KV capacity, the finalized sequence plan, the device and the startup observer. The
// bodies below follow those two constructors; where the fusion model's own API differs from the
// reference target's static facade, the call used is the one this engine already made before this
// change (so behaviour on the fusion path is unchanged).
// ---------------------------------------------------------------------------

LoadedQwen3_5::LoadedQwen3_5(std::unique_ptr<models::qwen3_5::Model> stable_model,
                             const EngineOptions& options)
    : model(std::move(stable_model)), parameters(*model),
      frontend(models::qwen3_5::make_frontend(
          model->resources(),
          {.chat_template_path       = options.chat_template_path,
           .architecture             = model->config().text.architecture,
           .vision_enabled           = options.enable_vision,
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
               options.context_cache.long_anchor_min_spacing_tokens})) {}

ModelInstance::ModelInstance(std::unique_ptr<LoadedQwen3_5> stable_loaded,
                             KvCapacityResolution resolution,
                             models::qwen3_5::SequencePlan sequence_plan, DeviceContext& device,
                             const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(models::qwen3_5::create_program(loaded->parameters, std::move(sequence_plan), device,
                                              startup_observer)) {}

ModelInstance::~ModelInstance() = default;

// The ported target owns a `Package::LoadedModel` (weights + frontend-resource bundle + the model
// views its executor reads), so its instance is built the way the reference builds its targets:
// `Loaded` holds the frontend, `Instance` builds the Program from `create_program`.
LoadedQwen4FlashNext::LoadedQwen4FlashNext(
    std::unique_ptr<models::qwen3_8_flash_next::Package::LoadedModel> stable_model,
    const EngineOptions& options)
    : model(std::move(stable_model)),
      frontend(models::qwen3_8_flash_next::Package::make_frontend(*model, options)) {}

FlashNextInstance::FlashNextInstance(std::unique_ptr<LoadedQwen4FlashNext> stable_loaded,
                                     KvCapacityResolution resolution,
                                     models::qwen3_8_flash_next::SequencePlan sequence_plan,
                                     DeviceContext& device,
                                     const StartupObserver& startup_observer)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      capacity(sequence_plan.capacity()),
      program(models::qwen3_8_flash_next::Package::create_program(*loaded->model,
                                                                  std::move(sequence_plan), device,
                                                                  startup_observer)) {}

FlashNextInstance::~FlashNextInstance() = default;

ConstructedModel assemble_fusion(const EngineOptions& requested, DeviceContext& device,
                                 artifact::Reader& reader, Clock::time_point start) {
    EngineOptions options = requested;
    if (options.devices.size() > 1 && options.stage_layers.empty()) {
        const std::vector<std::size_t> free_now = free_bytes_by_rank(device);
        const std::vector<std::uint64_t> free_bytes(free_now.begin(), free_now.end());
        const models::qwen3_5::StageSizing sizing{
            .kv_storage = options.kv_cache,
            .state_slots =
                options.max_concurrency + options.context_cache.device_state_slots.value_or(0U)};
        options.stage_layers = models::qwen3_5::default_stage_layers(
            reader, models::load_options(options), sizing, free_bytes);
    }
    core::set_wddm_residency_lock_enabled(options.wddm_evictable_budget);
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    auto plan = models::qwen3_5::plan_load(reader, models::load_options(options));
    binding.complete();
    auto model =
        models::qwen3_5::materialize_model(std::move(plan), device, &options.startup_observer);
    device.synchronize();
    StartupPhaseScope frontend(options.startup_observer, StartupPhase::FrontendInitialize);
    // Two stages because the Program cannot be built before the sequence plan exists: the reference
    // does the same split (`src/targets/registry.cpp:215` makes the `Loaded`, `:221` the `Instance`
    // that creates the Program from the finalized plan).
    auto loaded = std::make_unique<LoadedQwen3_5>(std::move(model), options);
    frontend.complete();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    const std::size_t overlay_window_bytes =
        models::qwen3_5::prepare_vision_overlay(loaded->parameters, device, options);
    const auto signature = models::qwen3_5::prefill_signature(*loaded->model);
    auto context_cost    = resolve_context_machine_cost(
        {.hardware_class =
                context_cost_hardware_class(device.props.name, device.props.major, device.props.minor),
            .prefill_signature = signature},
        options.context_cost.preset_path);
    auto planner = models::qwen3_5::make_sequence_planner(
        loaded->parameters, device,
        artifact_scoped_disk_tier(options, loaded->model->info().artifact_id));
    const std::vector<std::size_t> free_by_rank =
        free_bytes_by_rank(device, options.wddm_evictable_budget,
                           loaded->model->storage_stats().device_capacity_bytes);
    auto resolution = resolve_kv_capacity(
        options.kv_capacity, planner.capacity_curve(), free_by_rank.front(),
        std::span<const std::size_t>(free_by_rank).subspan(1));
    auto sequence   = std::move(planner).finalize(resolution.main_page_groups);
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens ||
        !std::ranges::equal(sequence.extra_rank_reservation_bytes(),
                            resolution.extra_rank_reservation_bytes)) {
        throw std::logic_error("resolved KV capacity does not match the finalized Program plan");
    }
    // The plan is the one authority for the resolved context-cache shape: its Host state slots,
    // Host KV bytes and long-anchor count may have been derived from the single host RAM budget.
    // Publishing that shape to the options the Engine keeps — and to the frontend grid built
    // before the plan existed — keeps the reported options, the ResourceManager and the Program
    // on the same capacity instead of a silently divergent default.
    EngineOptions resolved = options;
    resolved.context_cache = sequence.context_cache_options();
    loaded->frontend.publish_long_anchor_limit(
        resolved.context_cache.max_long_anchors_per_continuation.value_or(0));
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    // The instance builds its own Program from the finalized plan (reference:
    // `src/targets/registry.cpp:221-223`).
    auto instance = std::make_unique<ModelInstance>(std::move(loaded), resolution,
                                                    std::move(sequence), device,
                                                    options.startup_observer);
    LoadSummary::PrefixCacheRestore restore;
    if (resolved.context_cache.enabled && resolved.context_cache.mode == ContextCacheMode::Hybrid) {
        instance->program->set_hybrid_cost(hybrid_cache_cost(context_cost.model));
        // A request waiting for a sibling's snapshot stays in the FIFO, so the predicted wait
        // is kept well inside its queue timeout.
        instance->program->set_hybrid_coalesce_wait_limit(
            static_cast<double>(options.pending_timeout_ms) / 1000.0 / 2.0);
        const std::filesystem::path& file = resolved.context_cache.hybrid.persistent_file;
        if (!file.empty()) {
            const models::qwen3_5::HybridCachePersistence loaded =
                instance->program->attach_hybrid_cache_file(
                    file, hybrid_cache_fingerprint(options, signature));
            restore = LoadSummary::PrefixCacheRestore{
                .attempted = true,
                .restored  = loaded.ok,
                .message   = loaded.message,
                .blocks    = loaded.blocks,
                .snapshots = loaded.snapshots,
                .bytes     = loaded.bytes,
                .seconds   = loaded.seconds,
            };
        }
    }
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();
    const auto& stats = instance->loaded->model->storage_stats();
    LoadSummary summary;
    summary.architecture = models::architecture_name(instance->loaded->model->config().text.architecture);
    summary.model_name   = instance->loaded->model->info().name;
    summary.prefill_signature = signature;
    std::set<std::string> formats;
    for (const auto& weight : instance->loaded->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.pinned_weight_bytes  = stats.pinned_bytes;
    summary.overlay_window_bytes = overlay_window_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    summary.context_cost         = std::move(context_cost.summary);
    summary.prefix_cache         = std::move(restore);

    // Static /v1/models metadata: the model identity and dimension facts come from the loaded
    // model; the parameters, weight bytes, and weights profile are pure functions of the
    // artifact's tensor inventory.
    const auto& text_config = instance->loaded->model->config().text;
    ModelMetadata metadata;
    metadata.model_id       = summary.model_name;
    metadata.vocab_size     = text_config.vocab_size;
    metadata.embedding_size = text_config.hidden_size;
    metadata.native_context = text_config.max_position_embeddings;
    std::uint64_t parameters   = 0;
    std::uint64_t weight_bytes = 0;
    std::set<std::string> tensor_formats;
    for (const auto& object : reader.directory().objects) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr) { continue; } // Non-weight resources (tokenizer, templates).
        std::uint64_t elements = 1;
        for (const auto dimension : tensor->shape) { elements *= dimension; }
        parameters     += elements;
        weight_bytes   += tensor->bytes;
        tensor_formats.emplace(tensor->format);
    }
    metadata.parameters   = parameters;
    metadata.weight_bytes = weight_bytes;
    for (const auto& format : tensor_formats) {
        if (!metadata.weights_id.empty()) { metadata.weights_id += "+"; }
        metadata.weights_id += format;
    }
    return {std::move(instance), std::move(summary), std::move(metadata),
            std::move(context_cost.model), std::move(resolved)};
}

// ---------------------------------------------------------------------------
// The ported Flash-Next target (Qwen3.8-Flash-Next = the Qwen4 preview).
//
// Reference: `src/targets/registry.cpp:358-385` selects a target by `identity.model_id` and hands
// the whole assembly to `construct_registered<Target, Loaded, Instance>` (`:102-283`). That
// function is a template, so its `Instance` is a compile-time type and every `instance->...` keeps
// working unchanged. This engine's `construct_model` is one non-template function whose body reads
// model-specific members in twelve places spread over 150 lines, so the ported family gets its own
// assembler here instead of a branch inside that body. The fusion family's assembler is
// `construct_model` itself, untouched.
//
// `new_write_note`: the reference has no second assembler because its families are template
// parameters; splitting by function is how this engine's single entry point carries two families.
// ---------------------------------------------------------------------------
ConstructedModel assemble_flash_next(const EngineOptions& requested, DeviceContext& device,
                                    artifact::Reader& reader, Clock::time_point start) {
    EngineOptions options = requested;
    StartupPhaseScope binding(options.startup_observer, StartupPhase::TargetPlan);
    // `construct_registered` builds through the target's static facade
    // (`src/targets/registry.cpp:120-147`); the calls below are that sequence.
    artifact::Binder engine_binder(reader);
    // No adapter here. Reference: `registry.cpp:120-121` is `artifact::Binder binder(reader);` then
    // `Target::plan_load(binder, options, weights_profile)` -- the ENGINE binder goes straight in,
    // and the port layer builds its own adapter inside (`impl/load/bindings.h`). Constructing a
    // `ninfer::port::Binder` at this call site was my invention, and it cannot work: `plan_load`
    // takes `ninfer::artifact::Binder&`, which an adapter cannot convert to.
    const auto weights_profile =
        models::qwen3_8_flash_next::Package::resolve_weights(reader.identity());
    auto load_plan =
        models::qwen3_8_flash_next::Package::plan_load(engine_binder, options, weights_profile);
    // `load_plan.materialization()` is this engine's own MaterializationPlan: the adapter's
    // `finish()` forwards to the engine binder (`binder_port.h:296`).
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                             &options.startup_observer);
    // Read the stats here, while `materialized` is live. The ported `LoadedModel` is an opaque
    // pimpl with no accessor (`package.h:47-62`: one private `impl_`, friend `Package` only), so
    // this is the only reachable source -- and it is the same object the loaded model retains.
    const artifact::MaterializationStats materialized_stats = materialized.stats();
    auto loaded = std::make_unique<LoadedQwen4FlashNext>(
        models::qwen3_8_flash_next::Package::construct_loaded_model(std::move(load_plan),
                                                                   std::move(materialized)),
        options);
    const std::uint64_t planned_materialization_bytes = materialized_stats.device_capacity_bytes;
    binding.complete();
    device.synchronize();
    StartupPhaseScope planning(options.startup_observer, StartupPhase::TargetFinalize);
    auto planner = models::qwen3_8_flash_next::Package::make_sequence_planner(device, options,
                                                                             weights_profile);
    const std::vector<std::size_t> free_by_rank =
        free_bytes_by_rank(device, options.wddm_evictable_budget, planned_materialization_bytes);
    auto resolution = resolve_kv_capacity(
        options.kv_capacity, planner.capacity_curve(), free_by_rank.front(),
        std::span<const std::size_t>(free_by_rank).subspan(1));
    auto sequence = std::move(planner).finalize(resolution.main_page_groups);
    // The reference checks three quantities here (`registry.cpp:203-206`); the ported SequencePlan
    // exposes two of them (no `extra_rank_reservation_bytes`, `runtime.h:625-629`).
    if (sequence.device_reservation_bytes() != resolution.runtime_reservation_bytes ||
        sequence.kv_capacity() != resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized plan");
    }
    // `resolved.context_cache` is not taken from the plan here: the fusion SequencePlan exposes
    // `context_cache_options()` and the ported one does not, so the requested options stand - they
    // are what the ported Program planned against.
    EngineOptions resolved = options;
    loaded->frontend.publish_long_anchor_limit(
        resolved.context_cache.max_long_anchors_per_continuation.value_or(0));
    planning.complete();
    StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
    auto instance = std::make_unique<FlashNextInstance>(
        std::move(loaded), resolution, std::move(sequence), device, options.startup_observer);
    device.synchronize();
    program.complete();
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    const artifact::MaterializationStats& stats = materialized_stats;
    const artifact::ArtifactIdentity& identity = reader.identity();
    LoadSummary summary;
    // Reference: `summary.model_id = identity.model_id` (`registry.cpp:280`). The fusion engine
    // reports the architecture name and a model name as well; the ported family has no
    // `qwen3_5::Config`, so `architecture` comes from the identity pair the artifact carries and
    // `model_name` from its `model_id`.
    summary.architecture        = std::string(identity.model_id);
    summary.model_name          = std::string(identity.model_id);
    summary.prefill_signature   = std::string(identity.model_id) + "/" + identity.weights_id;
    summary.load_seconds        = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds      = stats.upload_seconds;
    summary.artifact_bytes_read = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes  = stats.peak_staging_bytes;
    summary.pinned_weight_bytes = stats.pinned_bytes;
    summary.overlay_window_bytes = 0;
    summary.device_object_count = stats.device_object_count;
    summary.host_object_count   = stats.host_object_count;
    // The engine's context-cost identity wants a non-empty `prefill_signature`
    // (`context_cache/context_cost.cpp:568-570` throws on an empty component). The ported family has
    // no `qwen3_5::Model` to derive one from, so it uses the artifact identity pair - which is what
    // the reference engine's own identity carries (`runtime/engine/context_cost.h:93-97`:
    // `model_id` + `weights_id`, filled at `src/targets/registry.cpp:114-115`).
    const runtime::ContextCostIdentity context_cost_identity{
        .hardware_class = context_cost_hardware_class(device.props.name, device.props.major,
                                                      device.props.minor),
        .prefill_signature = summary.prefill_signature,
    };
    auto context_cost = resolve_context_machine_cost(context_cost_identity,
                                                     options.context_cost.preset_path);
    summary.context_cost = std::move(context_cost.summary);

    // `/v1/models` metadata. The parameter count, weight bytes and format list are pure functions
    // of the artifact's tensor inventory (`construct_model` computes the same three from
    // `reader.directory().objects`), so they are shared verbatim. The three model dimensions that
    // the fusion path reads from `qwen3_5::Config` have no source here yet: the ported family keeps
    // its geometry in `TextModelView` (a fixed-shape view, `impl/model_view.h:108-119`) and the
    // reference engine does not report them at all (no `vocab_size` / `embedding_size` /
    // `native_context` anywhere in its tree). Left zero, to be filled from the artifact's stored
    // text config later.
    ModelMetadata metadata;
    metadata.model_id = summary.model_name;
    std::uint64_t parameters   = 0;
    std::uint64_t weight_bytes = 0;
    std::set<std::string> tensor_formats;
    for (const auto& object : reader.directory().objects) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (tensor == nullptr) { continue; } // Non-weight resources (tokenizer, templates).
        std::uint64_t elements = 1;
        for (const auto dimension : tensor->shape) { elements *= dimension; }
        parameters     += elements;
        weight_bytes   += tensor->bytes;
        tensor_formats.emplace(tensor->format);
    }
    metadata.parameters   = parameters;
    metadata.weight_bytes = weight_bytes;
    for (const auto& format : tensor_formats) {
        if (!metadata.weights_id.empty()) { metadata.weights_id += "+"; }
        metadata.weights_id += format;
    }
    summary.weight_formats.assign(tensor_formats.begin(), tensor_formats.end());

    ActiveTarget active = std::move(instance);
    return {std::move(active), std::move(summary), std::move(metadata),
            std::move(context_cost.model), std::move(resolved)};
}

// Picks the assembler by the artifact identity, exactly as the reference engine picks a target
// (`src/targets/registry.cpp:365-382`). The two assemblers exist because the families' planning
// types are not interchangeable; `assemble_fusion` is this engine's original body, unchanged.
ConstructedModel construct_model(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    // Block-7 alignment ④: snapshot the artifact path into the line's EXISTING by-path read
    // convention (NINFER_FLASH_NEXT_ARTIFACT_PATH -- read by impl/load/materialized.cpp:425 and now
    // by the optional PLE direct-read face). Set from the artifact the engine actually opened, which
    // is the ground truth; harmless to the older consumer, whose activation gate
    // (NINFER_FLASH_NEXT_PIN_READFILE mode > 0) is separate and unchanged. Deleting this block rolls
    // the direct face back to the mapped path with no other edit.
    {
        std::error_code path_ec;
        const std::filesystem::path absolute_path =
            std::filesystem::absolute(options.artifact_path, path_ec);
        const std::u8string wide = (path_ec ? options.artifact_path : absolute_path).u8string();
        const std::string utf8(reinterpret_cast<const char*>(wide.c_str()), wide.size());
#if defined(_WIN32)
        _putenv_s("NINFER_FLASH_NEXT_ARTIFACT_PATH", utf8.c_str());
#else
        setenv("NINFER_FLASH_NEXT_ARTIFACT_PATH", utf8.c_str(), 1);
#endif
    }
    install_device_route_profile_for(options, device);
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    if (reader.identity().model_id == models::qwen3_8_flash_next::Package::model_id) {
        return assemble_flash_next(options, device, reader, start);
    }
    return assemble_fusion(options, device, reader, start);
}

} // namespace ninfer::runtime
