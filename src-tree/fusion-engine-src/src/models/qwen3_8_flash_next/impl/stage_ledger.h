#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_8_flash_next::detail {

enum class FlashNextStageId : std::uint16_t {
    Preamble_EmbeddingStaging = 0,
    PLE_Injection,
    Hyper_PrepareAttn,
    QSA_Projection,
    QSA_IndexerScoreSelect,
    QSA_MainAttention,
    QSA_OutputProjection,
    GDN_QkvzProjection,
    GDN_ConvExtracts,
    GDN_Controls,
    GDN_Recurrence_PrepareWyWu,
    GDN_Recurrence_StatePassing,
    GDN_Recurrence_Output,
    GDN_OutputGate,
    GDN_OutputProjection,
    Hyper_InjectAttn,
    Hyper_PrepareMlp,
    MoE_Router,
    MoE_Grouping,
    MoE_SharedGateUp,
    MoE_QuantInput,
    MoE_RoutedGateUp,
    MoE_QuantDown,
    MoE_SharedDown,
    MoE_RoutedDown,
    MoE_Reduce,
    Hyper_InjectMlp,
    Final_NormHead,
    Count
};

struct StageLedgerStats {
    float total_chunk_ms     = 0.0f;
    float sum_accounted_ms   = 0.0f;
    float residual_ms        = 0.0f;
    float accounted_pct      = 0.0f;
    float residual_pct       = 0.0f;
    float stage_ms[static_cast<std::size_t>(FlashNextStageId::Count)]{};
    int   stage_calls[static_cast<std::size_t>(FlashNextStageId::Count)]{};
};

class FlashNextStageLedger {
public:
    static FlashNextStageLedger& instance();

    static bool is_enabled() noexcept {
        return enabled_;
    }

    void begin_chunk(cudaStream_t stream, std::int32_t tokens = 2048);
    void record_stage(cudaStream_t stream, FlashNextStageId stage);
    void finish_chunk(cudaStream_t stream);

    [[nodiscard]] const StageLedgerStats& last_stats() const noexcept {
        return last_stats_;
    }

    static std::string_view stage_name(FlashNextStageId stage) noexcept;

private:
    FlashNextStageLedger();
    ~FlashNextStageLedger();

    FlashNextStageLedger(const FlashNextStageLedger&) = delete;
    FlashNextStageLedger& operator=(const FlashNextStageLedger&) = delete;

    static bool check_env() noexcept;

    static inline const bool enabled_ = check_env();

    struct Entry {
        FlashNextStageId stage;
        std::size_t event_idx;
    };

    static constexpr std::size_t kMaxEvents = 4096;
    std::vector<cudaEvent_t> events_;
    std::vector<Entry> entries_;
    std::size_t current_event_idx_ = 0;
    std::int32_t current_tokens_   = 0;
    std::size_t chunk_counter_     = 0;
    bool in_chunk_                 = false;
    StageLedgerStats last_stats_{};
};

inline void stage_ledger_record(cudaStream_t stream, FlashNextStageId stage) {
#if defined(__GNUC__) || defined(__clang__)
    if (__builtin_expect(FlashNextStageLedger::is_enabled(), 0)) {
        FlashNextStageLedger::instance().record_stage(stream, stage);
    }
#else
    if (FlashNextStageLedger::is_enabled()) {
        FlashNextStageLedger::instance().record_stage(stream, stage);
    }
#endif
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
