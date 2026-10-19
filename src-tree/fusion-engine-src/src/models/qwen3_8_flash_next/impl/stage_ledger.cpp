#include "models/qwen3_8_flash_next/impl/stage_ledger.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ninfer::models::qwen3_8_flash_next::detail {

bool FlashNextStageLedger::check_env() noexcept {
    const char* env = std::getenv("NINFER_FLASH_NEXT_STAGE_LEDGER");
    if (env == nullptr) { return false; }
    while (*env == ' ' || *env == '\t') { ++env; }
    return *env == '1' || *env == 't' || *env == 'T' || *env == 'y' || *env == 'Y';
}

FlashNextStageLedger& FlashNextStageLedger::instance() {
    static FlashNextStageLedger s_instance;
    return s_instance;
}

FlashNextStageLedger::FlashNextStageLedger() {
    if (!enabled_) { return; }
    events_.resize(kMaxEvents, nullptr);
    for (std::size_t i = 0; i < kMaxEvents; ++i) {
        cudaEventCreate(&events_[i]);
    }
    entries_.reserve(kMaxEvents);
}

FlashNextStageLedger::~FlashNextStageLedger() {
    for (auto ev : events_) {
        if (ev != nullptr) {
            cudaEventDestroy(ev);
        }
    }
}

std::string_view FlashNextStageLedger::stage_name(FlashNextStageId stage) noexcept {
    switch (stage) {
    case FlashNextStageId::Preamble_EmbeddingStaging:
        return "Preamble: embedding & staging";
    case FlashNextStageId::PLE_Injection:
        return "PLE injection";
    case FlashNextStageId::Hyper_PrepareAttn:
        return "Hyper prepare (attention side)";
    case FlashNextStageId::QSA_Projection:
        return "Attention: QSA projection";
    case FlashNextStageId::QSA_IndexerScoreSelect:
        return "Attention: QSA indexer score+select";
    case FlashNextStageId::QSA_MainAttention:
        return "Attention: QSA main attention";
    case FlashNextStageId::QSA_OutputProjection:
        return "Attention: QSA output projection";
    case FlashNextStageId::GDN_QkvzProjection:
        return "Attention: GDN qkvz projection";
    case FlashNextStageId::GDN_ConvExtracts:
        return "Attention: GDN conv+extracts";
    case FlashNextStageId::GDN_Controls:
        return "Attention: GDN controls";
    case FlashNextStageId::GDN_Recurrence_PrepareWyWu:
        return "Attention: GDN recurrence (prepare_wy_wu)";
    case FlashNextStageId::GDN_Recurrence_StatePassing:
        return "Attention: GDN recurrence (state_passing)";
    case FlashNextStageId::GDN_Recurrence_Output:
        return "Attention: GDN recurrence (output)";
    case FlashNextStageId::GDN_OutputGate:
        return "Attention: GDN output gate";
    case FlashNextStageId::GDN_OutputProjection:
        return "Attention: GDN output projection";
    case FlashNextStageId::Hyper_InjectAttn:
        return "Hyper inject (attention side)";
    case FlashNextStageId::Hyper_PrepareMlp:
        return "Hyper prepare (mlp side)";
    case FlashNextStageId::MoE_Router:
        return "MoE: router";
    case FlashNextStageId::MoE_Grouping:
        return "MoE: grouping";
    case FlashNextStageId::MoE_SharedGateUp:
        return "MoE: shared gate-up";
    case FlashNextStageId::MoE_QuantInput:
        return "MoE: quant (input activations)";
    case FlashNextStageId::MoE_RoutedGateUp:
        return "MoE: routed gate-up";
    case FlashNextStageId::MoE_QuantDown:
        return "MoE: quant (down activations)";
    case FlashNextStageId::MoE_SharedDown:
        return "MoE: shared down";
    case FlashNextStageId::MoE_RoutedDown:
        return "MoE: routed down";
    case FlashNextStageId::MoE_Reduce:
        return "MoE: reduce";
    case FlashNextStageId::Hyper_InjectMlp:
        return "Hyper inject (mlp side)";
    case FlashNextStageId::Final_NormHead:
        return "Final norm + head";
    default:
        return "Unknown";
    }
}

void FlashNextStageLedger::begin_chunk(cudaStream_t stream, std::int32_t tokens) {
    if (!enabled_) { return; }
    entries_.clear();
    current_event_idx_ = 0;
    current_tokens_    = tokens;
    in_chunk_          = true;

    cudaEventRecord(events_[0], stream);
    current_event_idx_ = 1;
}

void FlashNextStageLedger::record_stage(cudaStream_t stream, FlashNextStageId stage) {
    if (!enabled_ || !in_chunk_ || current_event_idx_ >= kMaxEvents) { return; }
    cudaEventRecord(events_[current_event_idx_], stream);
    entries_.push_back({stage, current_event_idx_});
    current_event_idx_++;
}

void FlashNextStageLedger::finish_chunk(cudaStream_t stream) {
    if (!enabled_ || !in_chunk_) { return; }
    in_chunk_ = false;

    if (current_event_idx_ >= kMaxEvents) {
        std::fprintf(stderr, "[STAGE_LEDGER] Error: event buffer overflow\n");
        return;
    }

    const std::size_t final_event_idx = current_event_idx_;
    cudaEventRecord(events_[final_event_idx], stream);
    cudaEventSynchronize(events_[final_event_idx]);

    last_stats_ = StageLedgerStats{};
    cudaEventElapsedTime(&last_stats_.total_chunk_ms, events_[0], events_[final_event_idx]);

    for (std::size_t i = 0; i < entries_.size(); ++i) {
        float dt = 0.0f;
        const std::size_t prev_idx = (i == 0) ? 0 : entries_[i - 1].event_idx;
        const std::size_t curr_idx = entries_[i].event_idx;
        cudaEventElapsedTime(&dt, events_[prev_idx], events_[curr_idx]);

        const auto stage_idx = static_cast<std::size_t>(entries_[i].stage);
        if (stage_idx < static_cast<std::size_t>(FlashNextStageId::Count)) {
            last_stats_.stage_ms[stage_idx] += dt;
            last_stats_.stage_calls[stage_idx]++;
            last_stats_.sum_accounted_ms += dt;
        }
    }

    last_stats_.residual_ms = last_stats_.total_chunk_ms - last_stats_.sum_accounted_ms;
    if (last_stats_.total_chunk_ms > 0.0f) {
        last_stats_.accounted_pct = (last_stats_.sum_accounted_ms / last_stats_.total_chunk_ms) * 100.0f;
        last_stats_.residual_pct  = (last_stats_.residual_ms / last_stats_.total_chunk_ms) * 100.0f;
    }

    // Print Detailed Stage Table
    std::fprintf(stderr, "\n========================================================================================================\n");
    std::fprintf(stderr, "FLASH-NEXT PREFILL CHUNK STAGE LEDGER (T=%d, Chunk %zu)\n", current_tokens_, chunk_counter_);
    std::fprintf(stderr, "========================================================================================================\n");
    std::fprintf(stderr, "%-48s | %5s | %12s | %10s | %15s\n",
                 "Stage Name", "Calls", "Total (ms)", "% of Chunk", "Avg (ms/call)");
    std::fprintf(stderr, "-------------------------------------------------+-------+--------------+------------+----------------\n");

    int total_calls = 0;
    for (std::size_t s = 0; s < static_cast<std::size_t>(FlashNextStageId::Count); ++s) {
        const auto stage_id = static_cast<FlashNextStageId>(s);
        const float ms      = last_stats_.stage_ms[s];
        const int   calls   = last_stats_.stage_calls[s];
        if (calls == 0) { continue; }
        total_calls += calls;
        const float pct = (last_stats_.total_chunk_ms > 0.0f) ? (ms / last_stats_.total_chunk_ms * 100.0f) : 0.0f;
        const float avg = ms / static_cast<float>(calls);
        std::fprintf(stderr, "%-48s | %5d | %12.3f | %9.2f%% | %15.4f\n",
                     std::string(stage_name(stage_id)).c_str(), calls, ms, pct, avg);
    }
    std::fprintf(stderr, "-------------------------------------------------+-------+--------------+------------+----------------\n");
    std::fprintf(stderr, "%-48s | %5d | %12.3f | %9.2f%% |\n",
                 "Sum of Accounted Stages", total_calls, last_stats_.sum_accounted_ms, last_stats_.accounted_pct);
    std::fprintf(stderr, "%-48s | %5d | %12.3f | %9.2f%% |\n",
                 "Total Chunk Time (cudaEvent)", 1, last_stats_.total_chunk_ms, 100.0f);
    std::fprintf(stderr, "%-48s | %5s | %12.3f | %9.2f%% |\n",
                 "Residual (unaccounted / inter-stage gap)", "", last_stats_.residual_ms, last_stats_.residual_pct);
    std::fprintf(stderr, "========================================================================================================\n");

    // Rollup Categories
    float qsa_ms = 0.0f;
    qsa_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::QSA_Projection)];
    qsa_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::QSA_IndexerScoreSelect)];
    qsa_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::QSA_MainAttention)];
    qsa_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::QSA_OutputProjection)];

    float gdn_ms = 0.0f;
    gdn_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::GDN_QkvzProjection)];
    gdn_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::GDN_ConvExtracts)];
    gdn_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::GDN_Controls)];
    gdn_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::GDN_Recurrence_PrepareWyWu)];
    gdn_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::GDN_Recurrence_StatePassing)];
    gdn_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::GDN_Recurrence_Output)];
    gdn_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::GDN_OutputGate)];
    gdn_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::GDN_OutputProjection)];

    float moe_ms = 0.0f;
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_Router)];
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_Grouping)];
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_SharedGateUp)];
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_QuantInput)];
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_RoutedGateUp)];
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_QuantDown)];
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_SharedDown)];
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_RoutedDown)];
    moe_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::MoE_Reduce)];

    float hyper_ms = 0.0f;
    hyper_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::Hyper_PrepareAttn)];
    hyper_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::Hyper_InjectAttn)];
    hyper_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::Hyper_PrepareMlp)];
    hyper_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::Hyper_InjectMlp)];

    float ple_head_misc_ms = 0.0f;
    ple_head_misc_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::Preamble_EmbeddingStaging)];
    ple_head_misc_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::PLE_Injection)];
    ple_head_misc_ms += last_stats_.stage_ms[static_cast<std::size_t>(FlashNextStageId::Final_NormHead)];

    std::fprintf(stderr, "\n========================================================================================================\n");
    std::fprintf(stderr, "STAGE LEDGER CATEGORY ROLLUP (T=%d) vs EXTERNAL PRIOR\n", current_tokens_);
    std::fprintf(stderr, "========================================================================================================\n");
    std::fprintf(stderr, "%-48s | %12s | %10s | %10s | %14s\n",
                 "Category", "Measured(ms)", "% of Chunk", "Prior (ms)", "Delta vs Prior");
    std::fprintf(stderr, "-------------------------------------------------+--------------+------------+------------+----------------\n");
    std::fprintf(stderr, "%-48s | %12.3f | %9.2f%% | %10.1f | %+13.3f ms\n",
                 "QSA projections + attention (12 layers)", qsa_ms,
                 last_stats_.total_chunk_ms > 0 ? (qsa_ms / last_stats_.total_chunk_ms * 100.0f) : 0.0f,
                 115.0f, qsa_ms - 115.0f);
    std::fprintf(stderr, "%-48s | %12.3f | %9.2f%% | %10.1f | %+13.3f ms\n",
                 "GDN attention (36 layers)", gdn_ms,
                 last_stats_.total_chunk_ms > 0 ? (gdn_ms / last_stats_.total_chunk_ms * 100.0f) : 0.0f,
                 90.0f, gdn_ms - 90.0f);
    std::fprintf(stderr, "%-48s | %12.3f | %9.2f%% | %10.1f | %+13.3f ms\n",
                 "MoE (48 layers: router, shared, routed, reduce)", moe_ms,
                 last_stats_.total_chunk_ms > 0 ? (moe_ms / last_stats_.total_chunk_ms * 100.0f) : 0.0f,
                 70.0f, moe_ms - 70.0f);
    std::fprintf(stderr, "%-48s | %12.3f | %9.2f%% | %10.1f | %+13.3f ms\n",
                 "Hyper-connection boundaries (48 layers)", hyper_ms,
                 last_stats_.total_chunk_ms > 0 ? (hyper_ms / last_stats_.total_chunk_ms * 100.0f) : 0.0f,
                 50.0f, hyper_ms - 50.0f);
    std::fprintf(stderr, "%-48s | %12.3f | %9.2f%% | %10.1f | %+13.3f ms\n",
                 "PLE / head / misc", ple_head_misc_ms,
                 last_stats_.total_chunk_ms > 0 ? (ple_head_misc_ms / last_stats_.total_chunk_ms * 100.0f) : 0.0f,
                 22.0f, ple_head_misc_ms - 22.0f);
    std::fprintf(stderr, "%-48s | %12.3f | %9.2f%% | %10.1f | %+13.3f ms\n",
                 "Residual", last_stats_.residual_ms, last_stats_.residual_pct, 0.0f, last_stats_.residual_ms);
    std::fprintf(stderr, "-------------------------------------------------+--------------+------------+------------+----------------\n");
    std::fprintf(stderr, "%-48s | %12.3f | %9.2f%% | %10.1f | %+13.3f ms\n",
                 "Total Prefill Chunk", last_stats_.total_chunk_ms, 100.0f, 347.0f,
                 last_stats_.total_chunk_ms - 347.0f);
    std::fprintf(stderr, "========================================================================================================\n\n");

    chunk_counter_++;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
