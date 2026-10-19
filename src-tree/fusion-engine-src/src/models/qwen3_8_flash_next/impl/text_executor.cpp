#include "models/qwen3_8_flash_next/impl/text_executor.h"

#include "models/qwen3_8_flash_next/impl/mtp_forward.h"
#include "models/qwen3_8_flash_next/impl/kv_stream_binding.h"
#include "models/qwen3_8_flash_next/impl/stage_ledger.h"
#include "models/qwen3_8_flash_next/impl/text_decode.h"
#include "models/qwen3_8_flash_next/impl/text_decode_workspace.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

// ---------------------------------------------------------------------------
// PendingRound
// ---------------------------------------------------------------------------

PendingRound::PendingRound(FlashNextTextExecutor* owner, std::uint64_t transaction_id,
                           std::uint32_t batch_size, Tensor logits, Tensor final_hidden,
                           Tensor hyper_hidden,
                           std::span<const std::int32_t> sampled_tokens) noexcept
    : owner_(owner), transaction_id_(transaction_id), batch_size_(batch_size), logits_(logits),
      final_hidden_(final_hidden), hyper_hidden_(hyper_hidden) {
    const std::size_t n = std::min<std::size_t>(batch_size, sampled_tokens.size());
    for (std::size_t i = 0; i < n; ++i) {
        sampled_tokens_[i] = sampled_tokens[i];
    }
}

PendingRound::PendingRound(PendingRound&& other) noexcept
    : owner_(other.owner_), transaction_id_(other.transaction_id_), batch_size_(other.batch_size_),
      logits_(other.logits_), final_hidden_(other.final_hidden_),
      hyper_hidden_(other.hyper_hidden_), sampled_tokens_(other.sampled_tokens_) {
    other.owner_          = nullptr;
    other.transaction_id_ = 0;
    other.batch_size_     = 0;
}

PendingRound& PendingRound::operator=(PendingRound&& other) noexcept {
    if (this != &other) {
        if (valid()) { abort(); }
        owner_                = other.owner_;
        transaction_id_       = other.transaction_id_;
        batch_size_           = other.batch_size_;
        logits_               = other.logits_;
        final_hidden_         = other.final_hidden_;
        hyper_hidden_         = other.hyper_hidden_;
        sampled_tokens_       = other.sampled_tokens_;
        other.owner_          = nullptr;
        other.transaction_id_ = 0;
        other.batch_size_     = 0;
    }
    return *this;
}

bool PendingRound::valid() const noexcept { return owner_ != nullptr && transaction_id_ != 0; }

std::uint32_t PendingRound::batch_size() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return batch_size_;
}

Tensor PendingRound::logits() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return logits_;
}

Tensor PendingRound::final_hidden() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return final_hidden_;
}

Tensor PendingRound::hyper_hidden() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return hyper_hidden_;
}

std::span<const std::int32_t> PendingRound::sampled_tokens() const {
    if (!valid()) { throw std::logic_error("PendingRound: transaction is not valid"); }
    return std::span(sampled_tokens_.data(), batch_size_);
}

void PendingRound::commit(std::span<const LaneCommitDecision> decisions) {
    if (!valid()) { throw std::logic_error("PendingRound: cannot commit invalid transaction"); }
    auto* owner      = owner_;
    const auto tx_id = transaction_id_;
    owner->commit_transaction(tx_id, decisions);
    owner_          = nullptr;
    transaction_id_ = 0;
    batch_size_     = 0;
}

void PendingRound::commit_speculative(std::uint32_t lane_index,
                                      std::span<const std::int32_t> accepted_tokens) {
    if (!valid()) { throw std::logic_error("PendingRound: cannot commit invalid transaction"); }
    auto* owner      = owner_;
    const auto tx_id = transaction_id_;
    owner->commit_speculative_transaction(tx_id, lane_index, accepted_tokens);
    owner_          = nullptr;
    transaction_id_ = 0;
    batch_size_     = 0;
}

void PendingRound::abort() noexcept {
    if (!valid()) { return; }
    auto* owner      = owner_;
    const auto tx_id = transaction_id_;
    owner->abort_transaction(tx_id);
    owner_          = nullptr;
    transaction_id_ = 0;
    batch_size_     = 0;
}

// ---------------------------------------------------------------------------
// FlashNextTextExecutor
// ---------------------------------------------------------------------------

FlashNextTextExecutor::FlashNextTextExecutor(const TextModelView& model,
                                             PleIndexMetadata ple_metadata, DeviceContext& device,
                                             FlashNextRuntimeAllocation& allocation)
    : model_(model), ple_metadata_(ple_metadata), device_(device), alloc_(allocation),
      ple_pipeline_(model.ple.table, device,
                    std::max(allocation.plan().config.max_concurrency,
                             allocation.plan().config.prefill_chunk)),
      ledger_(allocation.plan()),
      use_cuda_graph_(allocation.plan().config.use_cuda_graph),
      sampling_workspace_(std::max<std::size_t>(
          256,
          ops::sampling_workspace_capacity_bytes(
              248'077, 1,
              static_cast<std::int32_t>(std::max(
                  allocation.plan().config.max_concurrency,
                  allocation.plan().config.speculative_draft_tokens > 0
                      ? (allocation.plan().config.speculative_draft_tokens + 1U)
                      : 1U))))),
      round_completion_(device) {
    instantiate_graphs();
}

void FlashNextTextExecutor::instantiate_graphs() {
    decode_graphs_.buckets = flash_next_decode_graph_buckets(alloc_.plan().maximum_blocks);
    if (!use_cuda_graph_ || model_.token_embedding.payload == nullptr) { return; }
    if (decode_graphs_.buckets.count == 0) { return; }

    const std::uint32_t max_concurrency = alloc_.plan().config.max_concurrency;
    const std::uint32_t n_buckets       = decode_graphs_.buckets.count;
    const std::size_t total_topologies  = static_cast<std::size_t>(max_concurrency) * n_buckets;

    decode_graphs_.profiles.clear();
    decode_graphs_.topologies.clear();
    decode_graphs_.profiles.reserve(total_topologies);
    decode_graphs_.topologies.reserve(total_topologies);

    std::memset(alloc_.host_ingress(), 0, sizeof(FlashNextDecodeIngress));
    std::memset(const_cast<void*>(ple_pipeline_.fixed_host_buffer()), 0,
                std::max(max_concurrency, alloc_.plan().config.speculative_draft_tokens + 1U) * 2'560 * sizeof(std::uint16_t));

    // Zero out persistent storage and synchronize slot tables before dummy capture runs
    CUDA_CHECK(cudaMemsetAsync(alloc_.persistent_base(), 0, alloc_.persistent_bytes(), device_.stream));
    alloc_.sync_slots_to_device(device_.stream);
    CUDA_CHECK(cudaStreamSynchronize(device_.stream));

    pending_custom_embeddings_.clear();
    for (std::uint32_t bucket_idx = 0; bucket_idx < n_buckets; ++bucket_idx) {
        const auto bucket_blocks = static_cast<std::int32_t>(decode_graphs_.buckets.blocks[bucket_idx]);
        for (std::uint32_t B = 1; B <= max_concurrency; ++B) {
            bool ok = false;
            try {
                execute_round_body(B, bucket_blocks, nullptr);
                device_.synchronize();

                if (install_captured_graph(B, bucket_idx, bucket_blocks)) {
                    ok = true;
                }
            } catch (const std::exception& ex) {
                std::fprintf(stderr,
                             "FlashNextTextExecutor: capture failed for B=%u bucket=%u blocks=%d: %s\n",
                             B, bucket_idx, bucket_blocks, ex.what());
            } catch (...) {
                std::fprintf(stderr,
                             "FlashNextTextExecutor: capture failed for B=%u bucket=%u blocks=%d: unknown exception\n",
                             B, bucket_idx, bucket_blocks);
            }
            device_.synchronize();

            if (!ok) {
                if (bucket_idx == 0) {
                    throw std::runtime_error(
                        "FlashNextTextExecutor: failed to capture decode graph for B=" + std::to_string(B) +
                        " bucket=0 blocks=" + std::to_string(bucket_blocks));
                } else {
                    graph_pinned_eager_[B - 1][bucket_idx] = true;
                    std::fprintf(stderr,
                                 "FlashNextTextExecutor: WARNING: pinning B=%u bucket=%u to permanent eager decode\n",
                                 B, bucket_idx);
                }
            }
        }
    }

    speculative_graphs_.buckets = decode_graphs_.buckets;
    speculative_graphs_.profiles.clear();
    speculative_graphs_.topologies.clear();
    const auto max_drafts = alloc_.plan().config.speculative_draft_tokens;
    for (std::uint32_t bucket = 0; bucket < n_buckets && max_drafts > 0; ++bucket) {
        const auto blocks = static_cast<std::int32_t>(decode_graphs_.buckets.blocks[bucket]);
        for (std::uint32_t rows = 2; rows <= max_drafts + 1U; ++rows) {
            auto* ingress = alloc_.host_ingress();
            for (std::uint32_t row = 0; row < rows; ++row) {
                ingress->token_indices[row] = static_cast<std::int32_t>(row);
                for (int axis = 0; axis < 3; ++axis) { ingress->mrope_positions[axis * rows + row] = row; }
                ingress->source_slots[row] = alloc_.lane_ring_slot(0, row);
                ingress->destination_slots[row] = alloc_.lane_ring_slot(0, row + 1U);
            }
            execute_round_body(rows, blocks, nullptr, true);
            device_.synchronize();
            if (!install_captured_graph(rows, bucket, blocks, true)) {
                throw std::runtime_error("FlashNextTextExecutor: failed to capture speculative verification");
            }
        }
    }
    mtp_draft_graphs_.clear();
    if (max_drafts > 0) {
        auto* header = alloc_.host_mtp_draft_ingress();
        *header = {};
        for (std::uint32_t k = 0; k < max_drafts; ++k) {
            header->steps[k].token_index = static_cast<std::int32_t>(k);
            header->steps[k].mrope_positions.fill(static_cast<std::int32_t>(k));
            header->steps[k].source_slot = alloc_.lane_ring_slot(0, k);
            header->steps[k].destination_slot = alloc_.lane_ring_slot(0, k + 1U);
        }
        CUDA_CHECK(cudaMemcpyAsync(alloc_.device_mtp_draft_ingress(), header, sizeof(*header),
                                   cudaMemcpyHostToDevice, device_.stream));
        for (std::uint32_t bucket = 0; bucket < n_buckets; ++bucket) {
            const auto blocks = static_cast<std::int32_t>(decode_graphs_.buckets.blocks[bucket]);
            for (std::uint32_t k = 0; k < max_drafts; ++k) {
                execute_mtp_draft_step(k, blocks);
                device_.synchronize();
                MtpDraftGraph graph;
                graph.step_index = k;
                graph.bucket_index = bucket;
                graph.definition.capture(device_.stream, [this, k, blocks] {
                    execute_mtp_draft_step(k, blocks);
                });
                graph.executable.instantiate(graph.definition);
                graph.executable.upload(device_.stream);
                mtp_draft_graphs_.push_back(std::move(graph));
            }
        }
    }
    // Zero out persistent state (KV caches, recurrent states, table rows) modified by dummy capture runs
    CUDA_CHECK(cudaMemsetAsync(alloc_.persistent_base(), 0, alloc_.persistent_bytes(), device_.stream));
    CUDA_CHECK(cudaStreamSynchronize(device_.stream));
}

LaneHandle FlashNextTextExecutor::allocate_lane() {
    auto handle = ledger_.allocate_lane(this);
    try {
        alloc_.zero_lane_slots(handle.lane_index(), device_.stream);
    } catch (...) {
        ledger_.release_lane(handle);
        throw;
    }
    return handle;
}

void FlashNextTextExecutor::release_lane(LaneHandle handle) {
    if (handle.owner() != this) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: cross-executor or invalid owner handle");
    }
    ledger_.release_lane(handle);
}

std::int32_t FlashNextTextExecutor::committed_frontier(LaneHandle handle) const {
    if (handle.owner() != this) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: cross-executor or invalid owner handle");
    }
    return ledger_.committed_frontier(handle);
}

std::size_t FlashNextTextExecutor::active_lanes_count() const noexcept {
    return ledger_.active_lanes_count();
}

bool FlashNextTextExecutor::decode_graph_pinned_eager(std::uint32_t batch_size,
                                                      std::uint32_t bucket_index) const noexcept {
    if (batch_size < 1 || batch_size > 8 || bucket_index >= kFlashNextDecodeGraphMaxBuckets) {
        return false;
    }
    return graph_pinned_eager_[batch_size - 1][bucket_index];
}

DecodeGraphTopology* FlashNextTextExecutor::find_topology(std::uint32_t batch_size,
                                                          std::uint32_t bucket_index, bool speculative) noexcept {
    for (auto& topology : (speculative ? speculative_graphs_ : decode_graphs_).topologies) {
        if (topology.batch_size == batch_size && topology.bucket_index == bucket_index) {
            return &topology;
        }
    }
    return nullptr;
}

const DecodeGraphTopology*
FlashNextTextExecutor::find_topology(std::uint32_t batch_size,
                                     std::uint32_t bucket_index, bool speculative) const noexcept {
    for (const auto& topology : (speculative ? speculative_graphs_ : decode_graphs_).topologies) {
        if (topology.batch_size == batch_size && topology.bucket_index == bucket_index) {
            return &topology;
        }
    }
    return nullptr;
}

bool FlashNextTextExecutor::install_captured_graph(std::uint32_t batch_size,
                                                   std::uint32_t bucket_index,
                                                   std::int32_t bucket_blocks, bool speculative) {
    auto& family = speculative ? speculative_graphs_ : decode_graphs_;
    DecodeGraphProfile profile;
    profile.batch_size             = batch_size;
    profile.bucket_index           = bucket_index;
    profile.bucket_blocks          = static_cast<std::uint32_t>(bucket_blocks);
    profile.min_execution_frontier = 0;
    profile.max_execution_frontier = static_cast<std::uint32_t>(bucket_blocks);
    profile.topology_class = flash_next_decode_graph_topology_class(batch_size, bucket_index);

    profile.definition.capture(device_.stream, [this, batch_size, bucket_blocks, speculative] {
        execute_round_body(batch_size, bucket_blocks, nullptr, speculative);
    });
    if (!profile.definition.ready()) { return false; }

    DecodeGraphTopology topology;
    topology.topology_class = profile.topology_class;
    topology.batch_size     = batch_size;
    topology.bucket_index   = bucket_index;
    topology.executable.instantiate(profile.definition);
    family.profiles.push_back(std::move(profile));
    topology.installed_profile = family.profiles.size() - 1;
    topology.executable.upload(device_.stream);
    family.topologies.push_back(std::move(topology));
    return true;
}

void FlashNextTextExecutor::execute_round_body(std::uint32_t batch_size,
                                              std::int32_t active_blocks,
                                              const FlashNextDecodeStateSink* sink, bool aliased_recurrent_scan) {
    if (active_blocks < 0 ||
        static_cast<std::uint32_t>(active_blocks) > alloc_.plan().maximum_blocks) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: active_blocks must be in [0, maximum_blocks]");
    }
    alloc_.workspace().reset();

    // 1. Single captured H2D memcpy of pinned ingress
    CUDA_CHECK(cudaMemcpyAsync(alloc_.device_ingress_ptr(), alloc_.host_ingress(),
                               sizeof(FlashNextDecodeIngress), cudaMemcpyHostToDevice,
                               device_.stream));

    // 2. Fixed-address H2D memcpy of PLE gathered rows
    Tensor gathered_ple(alloc_.round_tensors().gathered_ple_embedding.data, DType::BF16,
                        {2'560, static_cast<std::int32_t>(batch_size)});
    CUDA_CHECK(cudaMemcpyAsync(gathered_ple.data, ple_pipeline_.fixed_host_buffer(),
                               batch_size * 2'560 * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice, device_.stream));

    // 3. Token embedding
    Tensor token_ids =
        alloc_.round_tensors().token_ids.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor embedding =
        alloc_.workspace().alloc(DType::BF16, {2'560, static_cast<std::int32_t>(batch_size)}, 256);
    ops::embedding(token_ids, model_.token_embedding, embedding, device_.stream);
    for (std::uint32_t i = 0; i < batch_size && i < pending_custom_embeddings_.size(); ++i) {
        const Tensor* custom = pending_custom_embeddings_[i];
        if (custom == nullptr) { continue; }
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(embedding.data) +
                                       static_cast<std::size_t>(i) * 2'560 * sizeof(std::uint16_t),
                                   custom->data, 2'560 * sizeof(std::uint16_t),
                                   cudaMemcpyDeviceToDevice, device_.stream));
    }

    // 4. Target decode core
    Tensor token_indices =
        alloc_.round_tensors().token_indices.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor mrope_positions(alloc_.round_tensors().mrope_positions.data, DType::I32,
                           {static_cast<std::int32_t>(batch_size), 3});
    Tensor table_rows =
        alloc_.round_tensors().table_rows.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor source_slots =
        alloc_.round_tensors().source_slots.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor destination_slots =
        alloc_.round_tensors().destination_slots.slice(0, 0, static_cast<std::int32_t>(batch_size));
    Tensor final_hidden =
        alloc_.round_tensors().final_hidden.slice(1, 0, static_cast<std::int32_t>(batch_size));
    Tensor hyper_hidden =
        alloc_.round_tensors().hyper_hidden.slice(1, 0, static_cast<std::int32_t>(batch_size));
    Tensor logits =
        alloc_.round_tensors().logits.slice(1, 0, static_cast<std::int32_t>(batch_size));

    const auto max_blocks = static_cast<std::int32_t>(alloc_.plan().maximum_blocks);
    const bool has_visual = std::any_of(pending_custom_embeddings_.begin(),
        pending_custom_embeddings_.end(), [](const Tensor* tensor) { return tensor != nullptr; });
    flash_next_text_decode_core(model_, embedding, token_indices, mrope_positions, table_rows,
                                source_slots, destination_slots, gathered_ple, max_blocks,
                                active_blocks, alloc_.state_view(), alloc_.workspace(),
                                final_hidden, logits, device_.stream, sink, &hyper_hidden,
                                aliased_recurrent_scan, has_visual ? &token_ids : nullptr);

    // 5. Sampler
    Tensor sampled_tokens =
        alloc_.round_tensors().sampled_tokens.slice(0, 0, static_cast<std::int32_t>(batch_size));
    constexpr std::int32_t kSemanticTokenDomain = 248'077;
    ops::sample(logits, sampled_tokens, kSemanticTokenDomain,
                alloc_.device_sampling_configs(), token_indices,
                ops::kSamplePurposeDecode, sampling_workspace_, device_.stream);

    // 6. Single captured D2H memcpy of pinned egress
    CUDA_CHECK(cudaMemcpyAsync(alloc_.host_egress(), alloc_.device_egress_ptr(),
                               sizeof(FlashNextDecodeEgress), cudaMemcpyDeviceToHost,
                               device_.stream));
}

PendingRound FlashNextTextExecutor::execute_round(std::span<const LaneStepRequest> requests,
                                                 const FlashNextDecodeStateSink* sink) {
    for (const auto& req : requests) {
        if (req.handle.owner() != this) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: cross-executor or invalid owner handle");
        }
        if (req.custom_embedding != nullptr &&
            (req.custom_embedding->dtype != DType::BF16 || req.custom_embedding->ne[0] != 2'560 ||
             !req.custom_embedding->is_contiguous())) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: custom embedding must be a contiguous BF16 [2560] column");
        }
    }

    const auto batch_size = static_cast<std::uint32_t>(requests.size());
    if (batch_size == 0 || batch_size > alloc_.plan().config.max_concurrency) {
        throw std::invalid_argument("FlashNextTextExecutor: invalid round batch size");
    }
    if (round_in_flight_) {
        throw std::logic_error(
            "FlashNextTextExecutor: cannot start round while previous round is in flight");
    }

    auto prepared = ledger_.begin_round(requests, ple_metadata_);
    const std::uint32_t bucket_index =
        flash_next_decode_graph_select_bucket(decode_graphs_.buckets, prepared.max_active_blocks);
    const auto bucket_blocks = static_cast<std::int32_t>(decode_graphs_.buckets.blocks[bucket_index]);
    return finish_prepared_round(requests, std::move(prepared), sink, false, bucket_blocks);
}

PendingRound FlashNextTextExecutor::execute_round_eager(std::span<const LaneStepRequest> requests,
                                                        std::int32_t active_blocks,
                                                        const FlashNextDecodeStateSink* sink) {
    for (const auto& req : requests) {
        if (req.handle.owner() != this) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: cross-executor or invalid owner handle");
        }
        if (req.custom_embedding != nullptr &&
            (req.custom_embedding->dtype != DType::BF16 || req.custom_embedding->ne[0] != 2'560 ||
             !req.custom_embedding->is_contiguous())) {
            throw std::invalid_argument(
                "FlashNextTextExecutor: custom embedding must be a contiguous BF16 [2560] column");
        }
    }

    const auto batch_size = static_cast<std::uint32_t>(requests.size());
    if (batch_size == 0 || batch_size > alloc_.plan().config.max_concurrency) {
        throw std::invalid_argument("FlashNextTextExecutor: invalid round batch size");
    }
    if (round_in_flight_) {
        throw std::logic_error(
            "FlashNextTextExecutor: cannot start round while previous round is in flight");
    }
    if (active_blocks < 0 ||
        static_cast<std::uint32_t>(active_blocks) > alloc_.plan().maximum_blocks) {
        throw std::invalid_argument("FlashNextTextExecutor: active_blocks outside maximum_blocks");
    }

    auto prepared = ledger_.begin_round(requests, ple_metadata_);
    return finish_prepared_round(requests, std::move(prepared), sink, true, active_blocks);
}

namespace {

// KV streaming P2 switch (plan v2.1 E.1-3 / E.3). Called once, before the first decode round is
// staged. Decode only replays a graph: the engine's single cudaStreamBeginCapture lives in
// core/decode_graph.cpp:63 and runs at startup from instantiate_graphs(), so this point is always
// outside a capture and may allocate. A1/A2/A4/A5 are asserted here, before anything is allocated.
void flash_next_kv_stream_stage(FlashNextRuntimeAllocation& alloc, const void* owner,
                                cudaStream_t stream, bool arm) {
    if (kv_stream_binding().host_mirror) {
        if (arm) { kv_stream_binding().arm_residency(stream); }
        return;
    }
    // M4-1: the plan-time capacity curve reads the SAME predicate, so "off" and "on" are decided by
    // one piece of code (runtime_plan.h), not by two copies that could drift.
    if (!flash_next_kv_stream_requested()) { return; }

    const auto& plan   = alloc.plan();
    const auto& config = plan.config;
    // A1: the extra draft QSA layer shares qsa_attention_caches[0].block_tables but has no host
    // pool, so streaming with a draft layer would read -1 and write past the host pool.
    if (config.speculative_draft_tokens != 0) {
        throw std::runtime_error(
            "Flash-Next KV streaming requires speculative_draft_tokens==0 (the draft QSA layer "
            "shares the block table but has no host pool)");
    }
    // A2: P2 does not assume an identity page table, so it only runs a single lane.
    if (config.max_concurrency != 1) {
        throw std::runtime_error("Flash-Next KV streaming (P2) requires max_concurrency==1");
    }
    // A4: the host-side resolve reads its counters back, which a capture forbids.
    if (config.use_cuda_graph) {
        throw std::runtime_error(
            "Flash-Next KV streaming (P2) requires --no-cuda-graph (the resolve reads counters back)");
    }
    // A5: the page geometry below is the BF16 one.
    if (config.kv_cache != KvCacheStorage::BFloat16) {
        throw std::runtime_error("Flash-Next KV streaming (P2) supports --kv-dtype bf16 only");
    }

    const auto& layers = alloc.state_view().qsa_attention_caches;
    if (layers[0].key_pages.dtype != DType::BF16 || layers[0].key_pages.ne[0] != 256 ||
        layers[0].key_pages.ne[1] != 64 || layers[0].key_pages.ne[2] != 2 ||
        layers[0].block_tables.data == nullptr) {
        throw std::runtime_error("Flash-Next KV streaming: unexpected QSA cache geometry");
    }

    const std::int64_t pages = plan.attention_logical_pages;
    std::int64_t slots       = pages / 2 > 0 ? pages / 2 : 1;
    if (const char* value = std::getenv("NINFER_FLASH_NEXT_KV_STREAM_SLOTS");
        value != nullptr && value[0] != '\0') {
        slots = std::atoll(value);
    }
    if (const char* value = std::getenv("NINFER_FLASH_NEXT_KV_STREAM_DEBUG_SLOTS");
        value != nullptr && value[0] != '\0') {
        slots = std::atoll(value);  // J4 red arm: may go below the physical pool on purpose
    }
    if (slots <= 0 || slots > static_cast<std::int64_t>(plan.attention_physical_pages)) {
        throw std::runtime_error(
            "Flash-Next KV streaming: slots must be in [1, attention_physical_pages]");
    }

    const void* keys[kKvStreamMaxLayers]   = {};
    const void* values[kKvStreamMaxLayers] = {};
    for (std::size_t layer = 0; layer < kFullAttentionLayers; ++layer) {
        keys[layer]   = layers[layer].key_pages.data;
        values[layer] = layers[layer].value_pages.data;
    }
    // One plane of one page: 2 KV heads x 64 tokens x 256 dim x 2 B = 65,536 B.
    const std::int32_t page_plane_bytes = 2 * 64 * 256 * static_cast<std::int32_t>(sizeof(std::uint16_t));

    auto& binding = kv_stream_binding();
    binding.enable_host_mirror(owner, static_cast<void*>(layers[0].block_tables.data), 1, pages,
                              static_cast<std::int32_t>(slots), 64, 4, page_plane_bytes,
                              static_cast<int>(kFullAttentionLayers), keys, values);
    if (arm) { binding.arm_residency(stream); }
    std::fprintf(stderr,
                 "flash_next_kv_stream=on slots=%lld pages=%lld host_pool_mib=%.1f map_kib=%.1f\n",
                 static_cast<long long>(slots), static_cast<long long>(pages),
                 static_cast<double>(binding.host_layer_stride * binding.layers) /
                     (1024.0 * 1024.0),
                 static_cast<double>(binding.map_storage->bytes) / 1024.0);
}

}  // namespace

FlashNextTextExecutor::~FlashNextTextExecutor() {
    // Third knife M-4: the binding is a process-wide singleton whose host pool and map belong to
    // THIS executor; releasing them with the owner is what keeps a later executor from reusing a
    // dangling map (UAF).
    kv_stream_binding().shutdown_if_owner(this);
}

PendingRound FlashNextTextExecutor::finish_prepared_round(
    std::span<const LaneStepRequest> requests, FlashNextLaneLedger::PreparedRound prepared,
    const FlashNextDecodeStateSink* sink, bool force_eager, std::int32_t active_blocks) {
    const auto batch_size = static_cast<std::uint32_t>(requests.size());
    try {
        pending_is_prefill_chunk_ = false;

        // KV streaming P2: one-way switch, taken before this round's staging and outside any graph.
        flash_next_kv_stream_stage(alloc_, this, device_.stream, /*arm=*/true);

        // 1. Sync dirty tables to device (out of captured graph)
        ledger_.sync_tables_if_dirty(alloc_, device_.stream);

        // 2. Synchronous host gather into fixed pinned host buffer before launch
        ple_pipeline_.gather_pinned(prepared.ple_indices);

        // 3. Populate pinned host ingress
        auto* host_ing = alloc_.host_ingress();
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            const auto lane            = requests[i].handle.lane_index();
            host_ing->token_ids[i]     = requests[i].token_id;
            host_ing->token_indices[i] = requests[i].token_index;
            for (std::uint32_t d = 0; d < 3; ++d) {
                host_ing->mrope_positions[d * batch_size + i] = requests[i].mrope_positions[d];
            }
            host_ing->table_rows[i]        = static_cast<std::int32_t>(lane);
            host_ing->source_slots[i]      = alloc_.current_source_slot(lane);
            host_ing->destination_slots[i] = alloc_.current_destination_slot(lane);
            host_ing->sampling[i]          = requests[i].sampling;
        }

        // 4. Launch either CUDA graph replay or eager decode body. A state sink needs the eager
        //    body: replay cannot emit per-stage states, and silently ignoring the sink would
        //    break the teacher-forced dump harness. Rounds carrying custom embeddings (the
        //    reference tool's per-token vision path) also run eagerly: their columns are copied
        //    after the token embedding, which a captured graph cannot do.
        pending_custom_embeddings_.assign(batch_size, nullptr);
        bool has_custom = false;
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            pending_custom_embeddings_[i] = requests[i].custom_embedding;
            has_custom = has_custom || requests[i].custom_embedding != nullptr;
        }

        const std::uint32_t bucket_index = flash_next_decode_graph_select_bucket(
            decode_graphs_.buckets, active_blocks);
        bool ran = false;
        if (!force_eager && use_cuda_graph_ && sink == nullptr && !has_custom &&
            model_.token_embedding.payload != nullptr && decode_graphs_.buckets.count > 0 &&
            !graph_pinned_eager_[batch_size - 1][bucket_index]) {
            if (auto* topology = find_topology(batch_size, bucket_index);
                topology != nullptr && topology->executable.ready()) {
                topology->executable.launch(device_.stream);
                ran = true;
            }
        }
        if (!ran) { execute_round_body(batch_size, active_blocks, sink); }

        // 5. Complete round on event wait
        round_completion_.record(device_.stream);
        {
            const auto wait_started = std::chrono::steady_clock::now();
            round_completion_.synchronize();
            round_device_wait_ns_ += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_started)
                    .count());
        }

        // 6. Copy sampled tokens from pinned host egress
        std::array<std::int32_t, 8> sampled_tokens{};
        const auto* host_egr = alloc_.host_egress();
        for (std::uint32_t i = 0; i < batch_size; ++i) {
            sampled_tokens[i] = host_egr->sampled_tokens[i];
        }

        Tensor final_hidden =
            alloc_.round_tensors().final_hidden.slice(1, 0, static_cast<std::int32_t>(batch_size));
        Tensor hyper_hidden =
            alloc_.round_tensors().hyper_hidden.slice(1, 0, static_cast<std::int32_t>(batch_size));
        Tensor logits =
            alloc_.round_tensors().logits.slice(1, 0, static_cast<std::int32_t>(batch_size));

        round_in_flight_ = true;
        return PendingRound(this, prepared.transaction_id, batch_size, logits, final_hidden,
                            hyper_hidden, std::span(sampled_tokens.data(), batch_size));
    } catch (...) {
        ledger_.rollback_prepared_round(prepared.transaction_id);
        round_in_flight_ = false;
        throw;
    }
}

PendingRound FlashNextTextExecutor::execute_prefill_chunk(
    LaneHandle handle, std::span<const std::int32_t> token_ids,
    std::span<const std::array<std::int32_t, 3>> positions, std::int32_t first_token_index,
    const FlashNextDecodeStateSink* sink, const Tensor* visual_embeddings,
    std::span<const std::int32_t> chunk_local_scatter_indices) {
    if (handle.owner() != this) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: cross-executor or invalid owner handle");
    }
    if (token_ids.empty() || token_ids.size() != positions.size()) {
        throw std::invalid_argument(
            "FlashNextTextExecutor: token_ids and positions must be non-empty and matching size");
    }
    if (round_in_flight_) {
        throw std::logic_error(
            "FlashNextTextExecutor: cannot start prefill chunk while previous round is in flight");
    }

    const auto num_tokens                  = static_cast<std::uint32_t>(token_ids.size());
    const std::uint32_t lane               = handle.lane_index();
    const std::int32_t initial_active_slot = alloc_.current_source_slot(lane);
    const std::int32_t initial_standby_slot =
        alloc_.current_destination_slot(lane);

    auto prepared =
        ledger_.begin_prefill_chunk(handle, token_ids, first_token_index, ple_metadata_);

    try {
        pending_is_prefill_chunk_             = true;
        // KV streaming (third knife M-2): the authoritative host copy must exist from the FIRST
        // prefill chunk, or the first decode round finds no historical page and J8 throws. Mirror
        // only -- the residency map is not armed here, so block_tables keeps its legacy meaning.
        flash_next_kv_stream_stage(alloc_, this, device_.stream, /*arm=*/false);
        pending_prefill_lane_                 = lane;
        pending_prefill_initial_active_slot_  = initial_active_slot;
        pending_prefill_initial_standby_slot_ = initial_standby_slot;

        alloc_.workspace().reset();

        // Sync table updates if dirty
        ledger_.sync_tables_if_dirty(alloc_, device_.stream);

        FlashNextStageLedger::instance().begin_chunk(device_.stream, static_cast<std::int32_t>(num_tokens));

        // 1. Gather all 16*T PLE rows for the chunk at once
        PleTokenHistory temp_history = ledger_.lane_history(handle);
        std::vector<std::array<std::int64_t, 16>> chunk_ple_indices(num_tokens);
        for (std::uint32_t t = 0; t < num_tokens; ++t) {
            chunk_ple_indices[t] = ple_indices(ple_metadata_, temp_history, token_ids[t]);
            temp_history.commit(token_ids[t]);
        }
        auto ticket = ple_pipeline_.prepare(std::span(chunk_ple_indices));
        // Staging comes from the workspace through the same layout the capacity estimate uses.
        FlashNextPrefillChunkStaging staging = allocate_flash_next_prefill_chunk_staging(
            alloc_.workspace(), static_cast<std::int32_t>(num_tokens));
        Tensor& gathered_ple        = staging.gathered_ple;
        Tensor& dev_token_ids       = staging.token_ids;
        Tensor& dev_token_indices   = staging.token_indices;
        Tensor& dev_mrope_positions = staging.mrope_positions;
        Tensor& embedding           = staging.embedding;
        ple_pipeline_.enqueue_copy(std::move(ticket), gathered_ple);

        // 2. Upload chunk input metadata to the staging tensors
        std::vector<std::int32_t> host_indices(num_tokens);
        std::vector<std::int32_t> host_flat_positions(3 * num_tokens);
        for (std::uint32_t t = 0; t < num_tokens; ++t) {
            host_indices[t]                         = first_token_index + static_cast<std::int32_t>(t);
            host_flat_positions[0 * num_tokens + t] = positions[t][0];
            host_flat_positions[1 * num_tokens + t] = positions[t][1];
            host_flat_positions[2 * num_tokens + t] = positions[t][2];
        }

        // Payload H2D of chunk token ids/indices/positions; not a host decision round-trip.
        CUDA_CHECK(cudaMemcpyAsync(dev_token_ids.data, token_ids.data(),
                                   num_tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(dev_token_indices.data, host_indices.data(),
                                   num_tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));
        CUDA_CHECK(cudaMemcpyAsync(dev_mrope_positions.data, host_flat_positions.data(),
                                   3 * num_tokens * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                   device_.stream));

        // 3. Compute embedding [2560, T]
        ops::embedding(dev_token_ids, model_.token_embedding, embedding, device_.stream);

        // 3b. Scatter visual embeddings if provided
        if (visual_embeddings != nullptr && !chunk_local_scatter_indices.empty()) {
            const auto count = static_cast<std::int32_t>(chunk_local_scatter_indices.size());
            CUDA_CHECK(cudaMemcpyAsync(staging.visual_indices.data,
                                       chunk_local_scatter_indices.data(),
                                       count * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device_.stream));
            Tensor indices_slice = staging.visual_indices.slice(0, 0, count);
            ops::scatter(*visual_embeddings, indices_slice, embedding, device_.stream);
        }
        stage_ledger_record(device_.stream, FlashNextStageId::Preamble_EmbeddingStaging);

        // 4. Output tensors
        Tensor logits(alloc_.round_tensors().logits.data, DType::BF16, {248'320, 1});
        Tensor final_hidden(alloc_.round_tensors().final_hidden.data, DType::BF16, {2'560, 1});

        // 5. Execute T-wide prefill core
        // Diagnostic hook: NINFER_FLASH_NEXT_TRACE_STAGES=<dir> dumps every stage of every prefill
        // chunk (raw device bytes, chunk<k>_<stage>.bin) through the core's state sink. This is
        // how the sparse-attention shared-memory race was localised in a served process.
        static const char* stage_dir = std::getenv("NINFER_FLASH_NEXT_TRACE_STAGES");
        static int chunk_counter     = 0;
        FlashNextDecodeStateSink stage_sink;
        const FlashNextDecodeStateSink* effective_sink = sink;
        if (stage_dir != nullptr && stage_dir[0] != '\0' && sink == nullptr) {
            const int chunk_id  = chunk_counter++;
            stage_sink.on_state = [chunk_id](std::string_view name, const Tensor& t) {
                std::size_t count = 1;
                for (int d = 0; d < 4; ++d) { if (t.ne[d] > 0) { count *= static_cast<std::size_t>(t.ne[d]); } }
                const std::size_t elem = (t.dtype == DType::FP32 || t.dtype == DType::I32) ? 4 : 2;
                std::vector<unsigned char> host(count * elem);
                cudaMemcpy(host.data(), t.data, host.size(), cudaMemcpyDeviceToHost);
                const std::string path = std::string(stage_dir) + "/chunk" + std::to_string(chunk_id) +
                                         "_" + std::string(name) + ".bin";
                std::FILE* f = std::fopen(path.c_str(), "wb");
                if (f != nullptr) { std::fwrite(host.data(), 1, host.size(), f); std::fclose(f); }
            };
            effective_sink = &stage_sink;
        }
        // The prefill's hyper hidden lives in the scoped workspace arena, so it must be exported
        // into the round tensor the speculative draft head reads; otherwise the first draft of
        // every request runs on the previous round's leftovers.
        Tensor hyper_hidden(alloc_.round_tensors().hyper_hidden.data, DType::BF16, {10'240, 1});
        flash_next_text_prefill_chunk(
            model_, embedding, dev_token_indices, dev_mrope_positions,
            static_cast<std::int32_t>(lane), initial_active_slot, initial_standby_slot,
            gathered_ple, static_cast<std::int32_t>(alloc_.plan().maximum_blocks),
            first_token_index, alloc_.state_view(), alloc_.workspace(), final_hidden, logits,
            device_.stream, effective_sink, alloc_.plan().config.use_qsa_prefill_mma,
            &hyper_hidden, visual_embeddings != nullptr ? &dev_token_ids : nullptr);

        round_in_flight_ = true;
        return PendingRound(this, prepared.transaction_id, 1, logits, final_hidden, hyper_hidden);
    } catch (...) {
        pending_is_prefill_chunk_ = false;
        round_in_flight_          = false;
        alloc_.restore_lane_slots(lane, initial_active_slot, initial_standby_slot, device_.stream);
        ledger_.rollback_prepared_prefill_chunk(prepared.transaction_id);
        throw;
    }
}

void FlashNextTextExecutor::commit_transaction(std::uint64_t tx_id,
                                               std::span<const LaneCommitDecision> decisions) {
    round_in_flight_ = false;
    if (pending_is_prefill_chunk_) {
        if (!decisions.empty() && decisions[0].accept) {
            alloc_.commit_row_slot(pending_prefill_lane_, device_.stream);
        } else {
            alloc_.restore_lane_slots(pending_prefill_lane_, pending_prefill_initial_active_slot_,
                                      pending_prefill_initial_standby_slot_, device_.stream);
        }
        pending_is_prefill_chunk_ = false;
    }
    ledger_.commit_round(tx_id, decisions, alloc_, device_.stream);
}

void FlashNextTextExecutor::commit_speculative_transaction(
    std::uint64_t tx_id, std::uint32_t lane_index,
    std::span<const std::int32_t> accepted_tokens) {
    (void)lane_index;
    round_in_flight_          = false;
    pending_is_prefill_chunk_ = false;
    ledger_.commit_speculative_round(tx_id, accepted_tokens, alloc_, device_.stream);
}

void FlashNextTextExecutor::abort_transaction(std::uint64_t tx_id) noexcept {
    round_in_flight_ = false;
    if (pending_is_prefill_chunk_) {
        alloc_.restore_lane_slots(pending_prefill_lane_, pending_prefill_initial_active_slot_,
                                  pending_prefill_initial_standby_slot_, device_.stream);
        pending_is_prefill_chunk_ = false;
    }
    ledger_.abort_round(tx_id);
}

PendingRound FlashNextTextExecutor::execute_speculative_verify_round(
    LaneHandle handle, std::int32_t anchor_token_id, std::span<const std::int32_t> draft_tokens,
    std::int32_t first_token_index, std::array<std::int32_t, 3> first_mrope_position,
    const ops::SamplingConfig& sampling, std::span<const ops::SamplingConfig> column_sampling) {
    if (handle.owner() != this) {
        throw std::invalid_argument("FlashNextTextExecutor: cross-executor or invalid owner handle");
    }
    if (round_in_flight_) {
        throw std::logic_error(
            "FlashNextTextExecutor: cannot start speculative round while previous round is in flight");
    }

    const auto num_tokens    = static_cast<std::uint32_t>(1U + draft_tokens.size());
    if (!column_sampling.empty() && column_sampling.size() != num_tokens) {
        throw std::invalid_argument("speculative verification requires one sampling configuration per column");
    }
    const std::uint32_t lane = handle.lane_index();

    auto prepared =
        ledger_.begin_speculative_round(handle, anchor_token_id, draft_tokens, first_token_index,
                                        ple_metadata_);

    try {
        pending_is_prefill_chunk_ = false;
        ledger_.sync_tables_if_dirty(alloc_, device_.stream);
        ple_pipeline_.gather_pinned(prepared.ple_indices);

        auto* host_ing = alloc_.host_ingress();
        for (std::uint32_t i = 0; i < num_tokens; ++i) {
            host_ing->token_ids[i]     = (i == 0) ? anchor_token_id : draft_tokens[i - 1];
            host_ing->token_indices[i] = first_token_index + static_cast<std::int32_t>(i);
            for (std::uint32_t d = 0; d < 3; ++d) {
                host_ing->mrope_positions[d * num_tokens + i] =
                    first_mrope_position[d] + static_cast<std::int32_t>(i);
            }
            host_ing->table_rows[i]        = static_cast<std::int32_t>(lane);
            host_ing->source_slots[i]      = alloc_.lane_ring_slot(lane, i);
            host_ing->destination_slots[i] = alloc_.lane_ring_slot(lane, i + 1U);
            host_ing->sampling[i]          = column_sampling.empty() ? sampling : column_sampling[i];
        }

        const auto bucket_index = flash_next_decode_graph_select_bucket(
            decode_graphs_.buckets, prepared.max_active_blocks);
        const auto bucket_blocks = static_cast<std::int32_t>(decode_graphs_.buckets.blocks[bucket_index]);
        pending_custom_embeddings_.clear();
        auto* topology = find_topology(num_tokens, bucket_index, num_tokens > 1);
        if (use_cuda_graph_ && topology != nullptr && topology->executable.ready()) {
            topology->executable.launch(device_.stream);
        } else {
            execute_round_body(num_tokens, bucket_blocks, nullptr, true);
        }
        Tensor final_hidden = alloc_.round_tensors().final_hidden.slice(1, 0, num_tokens);
        Tensor hyper_hidden = alloc_.round_tensors().hyper_hidden.slice(1, 0, num_tokens);
        Tensor logits = alloc_.round_tensors().logits.slice(1, 0, num_tokens);
        round_completion_.record(device_.stream);
        {
            const auto wait_started = std::chrono::steady_clock::now();
            round_completion_.synchronize();
            round_device_wait_ns_ += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_started)
                    .count());
        }

        std::array<std::int32_t, 8> sampled_tokens_host{};
        const auto* host_egr = alloc_.host_egress();
        for (std::uint32_t i = 0; i < num_tokens; ++i) {
            sampled_tokens_host[i] = host_egr->sampled_tokens[i];
        }

        round_in_flight_ = true;
        return PendingRound(this, prepared.transaction_id, num_tokens, logits, final_hidden,
                            hyper_hidden, std::span(sampled_tokens_host.data(), num_tokens));
    } catch (...) {
        ledger_.rollback_prepared_round(prepared.transaction_id);
        round_in_flight_ = false;
        throw;
    }
}

void FlashNextTextExecutor::draft_mtp_tokens(LaneHandle handle, std::int32_t token_id,
                                             std::int32_t token_index,
                                             std::array<std::int32_t, 3> mrope_pos,
                                             const Tensor& backbone_hidden,
                                             std::uint32_t draft_count,
                                             std::span<std::int32_t> out_draft_tokens) {
    if (!model_.mtp.has_value()) {
        throw std::logic_error("FlashNextTextExecutor: cannot draft MTP tokens without MTP model view");
    }
    if (draft_count == 0) { return; }
    if (draft_count > alloc_.plan().config.speculative_draft_tokens ||
        out_draft_tokens.size() < draft_count) {
        throw std::invalid_argument("FlashNextTextExecutor: invalid MTP draft window");
    }

    const std::uint32_t lane = handle.lane_index();
    const auto max_blocks    = static_cast<std::int32_t>(alloc_.plan().maximum_blocks);
    ledger_.reserve_mtp_pages(handle, token_index + static_cast<std::int32_t>(draft_count) - 1);
    ledger_.sync_tables_if_dirty(alloc_, device_.stream);


    auto* host_ingress = alloc_.host_mtp_draft_ingress();
    auto* device_ingress = alloc_.device_mtp_draft_ingress();
    host_ingress->token_id = token_id;
    for (std::uint32_t k = 0; k < draft_count; ++k) {
        auto& step = host_ingress->steps[k];
        step.token_index = token_index - 1 + static_cast<std::int32_t>(k);
        step.mrope_positions = {
            mrope_pos[0] - 1 + static_cast<std::int32_t>(k),
            mrope_pos[1] - 1 + static_cast<std::int32_t>(k),
            mrope_pos[2] - 1 + static_cast<std::int32_t>(k)
        };

        step.table_row = static_cast<std::int32_t>(lane);
        step.source_slot = alloc_.lane_ring_slot(lane, k);
        step.destination_slot = alloc_.lane_ring_slot(lane, k + 1U);
    }
    CUDA_CHECK(cudaMemcpyAsync(device_ingress, host_ingress, sizeof(FlashNextMtpDraftIngress),
                               cudaMemcpyHostToDevice, device_.stream));
    // The first draft consumes the saved target position, which need not equal
    // the generated text position for a multimodal prompt.
    Tensor saved_positions = alloc_.state_view().mtp_backbone_positions.slice(
        1, alloc_.current_source_slot(lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(device_ingress->steps[0].mrope_positions.data(), saved_positions.data,
                               3 * sizeof(std::int32_t), cudaMemcpyDeviceToDevice, device_.stream));

    Tensor carried_hidden = alloc_.round_tensors().mtp_carried_hidden;
    CUDA_CHECK(cudaMemcpyAsync(carried_hidden.data, backbone_hidden.data,
                               10'240 * sizeof(std::uint16_t), cudaMemcpyDeviceToDevice,
                               device_.stream));
    for (std::uint32_t k = 0; k < draft_count; ++k) {
        const auto blocks = std::min(max_blocks, (token_index + static_cast<std::int32_t>(k)) / 4);
        if (use_cuda_graph_) {
            const auto bucket = flash_next_decode_graph_select_bucket(decode_graphs_.buckets, blocks);
            auto graph = std::find_if(mtp_draft_graphs_.begin(), mtp_draft_graphs_.end(),
                [k, bucket](const MtpDraftGraph& entry) {
                    return entry.step_index == k && entry.bucket_index == bucket;
                });
            if (graph == mtp_draft_graphs_.end()) {
                throw std::logic_error("FlashNextTextExecutor: missing MTP draft graph");
            }
            graph->executable.launch(device_.stream);
        } else {
            execute_mtp_draft_step(k, blocks);
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(alloc_.host_egress(), alloc_.device_egress_ptr(),
                               draft_count * sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               device_.stream));
    const auto wait_start = std::chrono::steady_clock::now();
    device_.synchronize();
    round_device_wait_ns_ += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_start).count());
    std::copy_n(alloc_.host_egress()->sampled_tokens.begin(), draft_count, out_draft_tokens.begin());
}

void FlashNextTextExecutor::execute_mtp_draft_step(std::uint32_t k, std::int32_t active_blocks) {
    alloc_.workspace().reset();
    auto* device_ingress = alloc_.device_mtp_draft_ingress();
    auto& step = device_ingress->steps[k];
    Tensor draft_logits = alloc_.round_tensors().mtp_logits;
    Tensor draft_tokens_tensor = alloc_.round_tensors().mtp_token;
    Tensor carried_hidden = alloc_.round_tensors().mtp_carried_hidden;
    Tensor dev_token_indices(&step.token_index, DType::I32, {1});
    Tensor dev_mrope_positions(step.mrope_positions.data(), DType::I32, {1, 3});
    Tensor dev_table_rows(&step.table_row, DType::I32, {1});
    Tensor source_slots(&step.source_slot, DType::I32, {1});
    Tensor destination_slots(&step.destination_slot, DType::I32, {1});
    Tensor cur_token_id_tensor = k == 0
        ? Tensor(&device_ingress->token_id, DType::I32, {1}) : draft_tokens_tensor;
    Tensor cur_embedding = alloc_.round_tensors().mtp_embedding;
    ops::embedding(cur_token_id_tensor, model_.token_embedding, cur_embedding, device_.stream);

    flash_next_mtp_step(
        model_, cur_embedding, carried_hidden, dev_token_indices, dev_mrope_positions,
        dev_table_rows, source_slots, destination_slots,
        alloc_.state_view().qsa_indexer_caches[kFullAttentionLayers],
        alloc_.state_view().qsa_attention_caches[kFullAttentionLayers],
        static_cast<std::int32_t>(alloc_.plan().maximum_blocks), active_blocks,
        alloc_.workspace(), draft_logits, draft_tokens_tensor, device_.stream, nullptr,
        &carried_hidden);

    CUDA_CHECK(cudaMemcpyAsync(alloc_.round_tensors().sampled_tokens.slice(0, k, 1).data,
                               draft_tokens_tensor.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, device_.stream));
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
