#pragma once

#include "core/weight.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::artifact {
class MaterializedArtifact;
struct ObjectHandle;
} // namespace ninfer::artifact

namespace ninfer::models::qwen3_8_flash_next::detail {

struct Nvfp4ExpertMatrixView {
    const std::byte* codes;
    const std::byte* scales;
    const float* weight_scale_divisor;
    float input_scale_divisor;
    std::int32_t rows;
    std::int32_t columns;
};

struct Nvfp4ExpertBankView {
    const std::byte* codes;
    const std::byte* scales;
    const float* weight_scale_divisors;
    std::int32_t experts;
    std::int32_t rows;
    std::int32_t columns;
    std::uint64_t code_bytes_per_expert;
    std::uint64_t scale_bytes_per_expert;

    // Set when the three pointers above name the artifact's read-only *file mapping* instead of the
    // device arena, i.e. the bank was bound with retain_experts_on_host (bind_moe). A device kernel
    // must never dereference them: the MoE leaf stages the experts the router selected into device
    // memory first and rewrites ids to slot indices (impl/expert_staging.h). This is the host-tiered
    // placement that makes the 63.3 GiB main expert bank fit beside a 32 GB card; it is placement
    // only -- it does not change any arithmetic.
    bool on_host = false;

    [[nodiscard]] Nvfp4ExpertMatrixView expert(std::int32_t index) const;
};

// Host-tiering switch, read once per process from NINFER_FLASH_NEXT_EXPERTS_HOST (non-empty and
// not "0" = on; unset = off, which is the upstream behaviour byte for byte). Read exactly once so
// the binding decision in bind_artifact, the workspace capacity that accounts for the staging
// planes, and the per-layer staging in the MoE leaf can never disagree.
[[nodiscard]] bool flash_next_experts_on_host_enabled();

// NINFER_FLASH_NEXT_EXPERTS_PIN=1 additionally copies the host expert banks into CUDA-pinned memory
// at load and points the bank views at that copy. This is the community's "load-mode none + full
// pin" setup: a memory-mapped source cannot be registered, so every pageable->device copy pays the
// driver's packing through an internal pinned staging buffer. Measured on this machine
// (lever_probe.exe): the engine's pageable mapping delivers 8.61 GB/s, the identical call sequence
// from pinned memory delivers 22.55 GB/s against a 23.97 GB/s one-big-copy ceiling -- and
// cudaHostAlloc grants 64 GiB here, more than the whole 63.3 GiB bank. Costs: one pinned copy of
// the bank at load and it occupies RAM, so the PLE pre-fault is skipped when this is on.
[[nodiscard]] bool flash_next_experts_pinned_enabled();

// NINFER_FLASH_NEXT_EXPERTS_SLOTS=N (default 64, 0 disables) keeps a persistent per-layer LRU slot
// pool in device memory, so a call only copies the experts that MISS. This is the community's
// --moe-expert-cache-size (they run 56-80 slots per expert tensor and report 46-60% hits). Reads
// once per process; the pool is sized layers x slots x 2,764,800 B and allocated on first use,
// so the workspace capacity function must know whether it will be needed.
[[nodiscard]] std::int32_t flash_next_experts_slot_count();

// Block-6 W1/W2: the CPU expert-serve FORWARD face rides the serve master switch
// (NINFER_FLASH_NEXT_CPU_EXPERT_SERVE; no new env name, D-1). When true, the MoE workspace also
// allocates the partial plane the served experts' contribution is carried in, and the staging
// arms the compute face for decode-width calls. Defined in expert_staging.cpp; read once per
// process, like every other switch here.
[[nodiscard]] bool flash_next_cpu_serve_forward_enabled();

[[nodiscard]] Nvfp4ExpertBankView
make_nvfp4_expert_bank_view(const void* payload, std::uint64_t payload_bytes, std::int32_t experts,
                            std::int32_t rows, std::int32_t columns);

[[nodiscard]] Nvfp4ExpertBankView
materialized_nvfp4_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                    artifact::ObjectHandle handle, std::int32_t experts,
                                    std::int32_t rows, std::int32_t columns);

// NINFER_FLASH_NEXT_GGUF_EXPERTS=1 accepts the gguf-block expert banks of a block-5 IQ3_S artifact
// (the text layers' `mlp/experts/*` objects stored in QuantLayout::GgufBlocks). Default OFF (unset
// or "0"): a container whose expert bank is stored in ggml blocks is refused at binding, and every
// non-gguf artifact behaves exactly as before. Read once per process, like the rest of this family.
[[nodiscard]] bool flash_next_gguf_experts_enabled();

// Block-5 ruling: the nvfp4-only service policies (CPU expert serve, drop, substitute) have no
// gguf counterpart in this stage. Returns the first such switch found ON, or nullptr when none is
// set ("0"/unset counts as off). Both the load face (fail-fast) and the gguf MoE leaf consult
// this, so a gguf bank next to one of them refuses loudly instead of silently ignoring the policy.
[[nodiscard]] const char* flash_next_gguf_policy_conflict() noexcept;

// Block-5 IQ3_S: whole ggml block rows (QuantLayout::GgufBlocks) as an expert bank. Stage 1 is
// placement/read only -- nothing here interprets the blocks beyond their geometry, and the MoE
// leaf's exact-bank validation refuses a forward over such weights until the compute wiring lands.
struct GgufExpertMatrixView {
    const std::byte* data;
    std::int32_t rows;
    std::int32_t columns;
    std::uint64_t row_bytes;
    QType qtype;
};

struct GgufExpertBankView {
    const std::byte* data = nullptr;
    std::int32_t experts = 0;
    std::int32_t rows = 0;
    std::int32_t columns = 0;
    std::uint64_t row_bytes = 0;
    QType qtype = QType::GGUF_Q8_0;

    // Set when the pointer above names the artifact's read-only *file mapping* instead of the
    // device arena, i.e. the bank was bound with retain_experts_on_host (bind_moe) -- the same
    // host-tiered placement Nvfp4ExpertBankView documents.
    bool on_host = false;

    [[nodiscard]] GgufExpertMatrixView expert(std::int32_t index) const;
};

[[nodiscard]] GgufExpertBankView
make_gguf_expert_bank_view(const void* payload, std::uint64_t payload_bytes, std::int32_t experts,
                           std::int32_t rows, std::int32_t columns, QType qtype);

[[nodiscard]] GgufExpertBankView
materialized_gguf_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                   artifact::ObjectHandle handle, std::int32_t experts,
                                   std::int32_t rows, std::int32_t columns, QType qtype);

struct Bf16ExpertMatrixView {
    const std::byte* data;
    std::int32_t rows;
    std::int32_t columns;
};

struct Bf16ExpertBankView {
    const std::byte* data;
    std::int32_t experts;
    std::int32_t rows;
    std::int32_t columns;
    std::uint64_t bytes_per_expert;

    [[nodiscard]] Bf16ExpertMatrixView expert(std::int32_t index) const;
};

[[nodiscard]] Bf16ExpertBankView
make_bf16_expert_bank_view(const void* payload, std::uint64_t payload_bytes, std::int32_t experts,
                           std::int32_t rows, std::int32_t columns);

[[nodiscard]] Bf16ExpertBankView
materialized_bf16_expert_bank_view(const artifact::MaterializedArtifact& materialized,
                                   artifact::ObjectHandle handle, std::int32_t experts,
                                   std::int32_t rows, std::int32_t columns);

} // namespace ninfer::models::qwen3_8_flash_next::detail
