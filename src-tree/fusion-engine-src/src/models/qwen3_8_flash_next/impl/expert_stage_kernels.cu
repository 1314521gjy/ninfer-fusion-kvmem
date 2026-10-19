#include "models/qwen3_8_flash_next/impl/expert_stage_kernels.h"

#include "core/device.h"

#include <cstdio>
#include <stdexcept>

namespace ninfer::models::qwen3_8_flash_next::detail {
namespace {

// Slots and paths are small (<=128 slots, <=80 routed paths), so one thread does the whole decision.
// The victim scan is the only loop with real length: on a warm cache a layer has about one or two
// misses, and even a cold layer scans 128 slots at most ten times.
constexpr int kCopySlicesPerItem = 4;
constexpr int kCopyThreads       = 256;
// Upper bound on routed paths in one call: 10 top-k x kFlashNextRouteDecodeMaxTokens(8). The host
// side rejects anything larger before it gets here (expert_staging.cpp: the device path throws on
// tokens > 8, and the host path bounds paths by kMaxStagedPaths on its own way), so this is a stack
// bound, not a policy. It stays at the decode width on purpose: with the wide MoE arm the device path
// is refused outright, because it cannot count a call's distinct experts without the readback it
// exists to avoid.
constexpr int kMaxDeviceStagePaths = 80;

__global__ void flash_next_expert_decide_kernel(
    std::int32_t* __restrict__ ids, std::int32_t paths, std::int32_t slots, std::int32_t pass,
    std::int32_t* __restrict__ tick, std::int32_t* __restrict__ slot_of_expert,
    std::int32_t* __restrict__ slot_expert, std::int32_t* __restrict__ slot_age,
    std::int32_t* __restrict__ slot_pass, std::int32_t* __restrict__ work,
    const float* __restrict__ source_divisors_gate_up,
    const float* __restrict__ source_divisors_down, float* __restrict__ slot_divisors_gate_up,
    float* __restrict__ slot_divisors_down, unsigned long long* __restrict__ counters,
    float* __restrict__ alpha, float drop_threshold) {
    if (blockIdx.x != 0 || threadIdx.x != 0) { return; }

    const std::int32_t t = *tick + 1;
    *tick                = t;
    std::int32_t count   = 0;

    // ---- R1 (miss-gated dropping), replicated from the host path EXACTLY -------------------------
    // drop a path iff (a) its expert is NOT resident RIGHT NOW and (b) its weight < drop_threshold.
    // Three details of the host implementation are easy to get subtly wrong and are reproduced here
    // on purpose:
    //   * top_path is the argmax over ALL paths of the call (every token), not per token;
    //   * resident-ness is read from the table BEFORE this call allocates any victim, which is the
    //     order the host loop uses (its drop pass runs before its slot-allocation pass);
    //   * the two sums are taken over ALL paths split by drop[], and the survivor scale is applied
    //     ONLY IF something was actually dropped. Without that guard keep_scale would become
    //     1/(sum of all weights) = 1/tokens and silently rescale every call that dropped nothing.
    // Substitution (MOE_SUBST) is NOT implemented on this path; the host call site refuses it.
    bool drop[kMaxDeviceStagePaths];
    std::int32_t top_path     = 0;
    std::int32_t dropped_here = 0;
    float drop_sum            = 0.0F;
    float kept_sum            = 0.0F;
    const bool r1_armed       = drop_threshold > 0.0F && alpha != nullptr;
    if (r1_armed) {
        for (std::int32_t i = 0; i < paths; ++i) { drop[i] = false; }
        for (std::int32_t i = 1; i < paths; ++i) {
            if (alpha[i] > alpha[top_path]) { top_path = i; }
        }
        for (std::int32_t i = 0; i < paths; ++i) {
            if (i == top_path) { continue; }
            const std::int32_t expert = ids[i];
            if (expert < 0 || expert >= 512) { continue; } // raised on the host side
            if (slot_of_expert[expert] >= 0) { continue; } // resident: free, never dropped
            if (alpha[i] < drop_threshold) { drop[i] = true; }
        }
        for (std::int32_t i = 0; i < paths; ++i) {
            if (drop[i]) {
                ++dropped_here;
                drop_sum += alpha[i];
            } else {
                kept_sum += alpha[i];
            }
        }
    }

    for (std::int32_t i = 0; i < paths; ++i) {
        if (r1_armed && drop[i]) { continue; } // never reaches `work`: this is where the bytes are saved
        const std::int32_t expert = ids[i];
        std::int32_t slot         = slot_of_expert[expert];
        if (slot >= 0) {
            atomicAdd(&counters[0], 1ULL); // hits
            slot_age[slot]  = t;
            slot_pass[slot] = pass;
        } else {
            atomicAdd(&counters[1], 1ULL); // misses
            // Least-Stale ahead of plain LRU: an entry evicted inside the same forward pass and
            // needed again immediately after is what plain LRU gets wrong.
            std::int32_t victim = -1;
            for (std::int32_t s = 0; s < slots; ++s) {
                if (slot_expert[s] < 0) {
                    victim = s;
                    break;
                }
            }
            if (victim < 0) {
                for (std::int32_t s = 0; s < slots; ++s) {
                    if (slot_pass[s] == pass) { continue; }
                    if (victim < 0 || slot_age[s] < slot_age[victim]) { victim = s; }
                }
            }
            if (victim < 0) {
                for (std::int32_t s = 0; s < slots; ++s) {
                    if (victim < 0 || slot_age[s] < slot_age[victim]) { victim = s; }
                }
            }
            const std::int32_t evicted = slot_expert[victim];
            if (evicted >= 0) {
                atomicAdd(&counters[2], 1ULL); // evictions
                slot_of_expert[evicted] = -1;
            }
            slot_expert[victim] = expert;
            slot_of_expert[expert] = victim;
            slot_age[victim]       = t;
            slot_pass[victim]      = pass;

            work[1 + 2 * count] = expert;
            work[2 + 2 * count] = victim;
            ++count;

            // The kernels index the divisor plane with the same (rewritten) id they index the codes
            // with, so slot s must carry the divisor of ITS expert. Four bytes over PCIe per plane.
            slot_divisors_gate_up[victim] = source_divisors_gate_up[expert];
            slot_divisors_down[victim]    = source_divisors_down[expert];
            slot                          = victim;
        }
        ids[i] = slot; // expert id -> slot index, in place
    }

    // Fixup, mirroring the host path: dropped paths take the TOP path's slot (final by now, since the
    // loop above assigned it) and a zero weight, so the MoE kernel reads a COMMITTED slot and
    // multiplies it by zero; survivors are renormalised by 1/kept_sum. Nothing is touched when
    // nothing was dropped -- that is what keeps the unarmed case bit-identical.
    if (r1_armed && dropped_here > 0) {
        const float keep_scale = (kept_sum > 1e-6F) ? (1.0F / kept_sum) : 0.0F;
        for (std::int32_t i = 0; i < paths; ++i) {
            if (drop[i]) {
                ids[i]   = ids[top_path];
                alpha[i] = 0.0F;
            } else {
                alpha[i] *= keep_scale;
            }
        }
        atomicAdd(&counters[3], static_cast<unsigned long long>(dropped_here));
    }
    work[0] = count;
}

__global__ void flash_next_expert_copy_kernel(
    const std::int32_t* __restrict__ work, const std::uint8_t* __restrict__ gate_up_codes,
    std::uint64_t gate_up_code_stride, const std::uint8_t* __restrict__ gate_up_scales,
    std::uint64_t gate_up_scale_stride, const std::uint8_t* __restrict__ down_codes,
    std::uint64_t down_code_stride, const std::uint8_t* __restrict__ down_scales,
    std::uint64_t down_scale_stride, std::uint8_t* __restrict__ slot_gate_up_codes,
    std::uint8_t* __restrict__ slot_gate_up_scales, std::uint8_t* __restrict__ slot_down_codes,
    std::uint8_t* __restrict__ slot_down_scales) {
    const std::int32_t count = work[0];
    const std::int32_t item  = blockIdx.x / kCopySlicesPerItem;
    if (item >= count) { return; }
    const std::int32_t slice = blockIdx.x % kCopySlicesPerItem;
    const std::int32_t plane = blockIdx.y;

    const std::int32_t expert = work[1 + 2 * item];
    const std::int32_t slot   = work[2 + 2 * item];

    const std::uint8_t* source      = nullptr;
    std::uint8_t* destination       = nullptr;
    std::uint64_t bytes             = 0;
    switch (plane) {
    case 0:
        source      = gate_up_codes + static_cast<std::uint64_t>(expert) * gate_up_code_stride;
        destination = slot_gate_up_codes + static_cast<std::uint64_t>(slot) * gate_up_code_stride;
        bytes       = gate_up_code_stride;
        break;
    case 1:
        source      = gate_up_scales + static_cast<std::uint64_t>(expert) * gate_up_scale_stride;
        destination = slot_gate_up_scales + static_cast<std::uint64_t>(slot) * gate_up_scale_stride;
        bytes       = gate_up_scale_stride;
        break;
    case 2:
        source      = down_codes + static_cast<std::uint64_t>(expert) * down_code_stride;
        destination = slot_down_codes + static_cast<std::uint64_t>(slot) * down_code_stride;
        bytes       = down_code_stride;
        break;
    default:
        source      = down_scales + static_cast<std::uint64_t>(expert) * down_scale_stride;
        destination = slot_down_scales + static_cast<std::uint64_t>(slot) * down_scale_stride;
        bytes       = down_scale_stride;
        break;
    }

    // 16-byte units; both the pinned bank planes and the pool planes are 256-byte aligned and every
    // stride is a multiple of 256, so the reinterpreted views stay aligned.
    const auto count16 = bytes / sizeof(uint4);
    const auto* source16 = reinterpret_cast<const uint4*>(source);
    auto* destination16  = reinterpret_cast<uint4*>(destination);
    const std::uint64_t step =
        static_cast<std::uint64_t>(kCopySlicesPerItem) * static_cast<std::uint64_t>(blockDim.x);
    for (std::uint64_t i = static_cast<std::uint64_t>(slice) * blockDim.x + threadIdx.x; i < count16;
         i += step) {
        destination16[i] = source16[i];
    }
}

} // namespace

void flash_next_expert_decide_launch(std::int32_t* ids, std::int32_t paths, std::int32_t slots,
                                     std::int32_t pass, std::int32_t* tick,
                                     std::int32_t* slot_of_expert, std::int32_t* slot_expert,
                                     std::int32_t* slot_age, std::int32_t* slot_pass,
                                     std::int32_t* work, const float* source_divisors_gate_up,
                                     const float* source_divisors_down, float* slot_divisors_gate_up,
                                     float* slot_divisors_down, unsigned long long* counters,
                                     float* alpha, float drop_threshold, cudaStream_t stream) {
    flash_next_expert_decide_kernel<<<1, 1, 0, stream>>>(
        ids, paths, slots, pass, tick, slot_of_expert, slot_expert, slot_age, slot_pass, work,
        source_divisors_gate_up, source_divisors_down, slot_divisors_gate_up, slot_divisors_down,
        counters, alpha, drop_threshold);
    CUDA_CHECK(cudaGetLastError());
}

void flash_next_expert_copy_launch(const std::int32_t* work, std::int32_t max_items,
                                   const std::uint8_t* gate_up_codes,
                                   std::uint64_t gate_up_code_stride,
                                   const std::uint8_t* gate_up_scales,
                                   std::uint64_t gate_up_scale_stride,
                                   const std::uint8_t* down_codes, std::uint64_t down_code_stride,
                                   const std::uint8_t* down_scales, std::uint64_t down_scale_stride,
                                   std::uint8_t* slot_gate_up_codes, std::uint8_t* slot_gate_up_scales,
                                   std::uint8_t* slot_down_codes, std::uint8_t* slot_down_scales,
                                   cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(max_items * kCopySlicesPerItem), 4u);
    flash_next_expert_copy_kernel<<<grid, kCopyThreads, 0, stream>>>(
        work, gate_up_codes, gate_up_code_stride, gate_up_scales, gate_up_scale_stride, down_codes,
        down_code_stride, down_scales, down_scale_stride, slot_gate_up_codes, slot_gate_up_scales,
        slot_down_codes, slot_down_scales);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------------------------------------------
// Per-layer device state. Keyed by the layer's host bank pointer, exactly like the host-side slot
// pool, so the two never have to agree on an index.
// ------------------------------------------------------------------------------------------------
namespace {

constexpr int kMaxDeviceTables = 49;
constexpr int kMaxSlots        = 512;
constexpr int kMaxPaths        = 80;

struct DeviceStageState {
    bool ready        = false;
    std::int32_t slots = 0;
    std::int32_t* slot_of_expert = nullptr; // [kMaxDeviceTables][512]
    std::int32_t* slot_expert    = nullptr; // [kMaxDeviceTables][slots]
    std::int32_t* slot_age       = nullptr;
    std::int32_t* slot_pass      = nullptr;
    std::int32_t* tick           = nullptr;
    std::int32_t* work           = nullptr; // [1 + 2*kMaxPaths]
    unsigned long long* counters = nullptr; // [4], device: hits, misses, evictions, dropped
    unsigned long long* counters_host = nullptr; // [4], pinned mirror
    const void* keys[kMaxDeviceTables] = {};
    std::uint64_t calls = 0;
};

DeviceStageState& device_stage_state() {
    static DeviceStageState state;
    return state;
}

bool ensure_device_state(DeviceStageState& s, std::int32_t slots) {
    if (s.ready) { return true; }
    void* tables = nullptr;
    if (cudaMalloc(&tables, static_cast<std::size_t>(kMaxDeviceTables) * kMaxSlots *
                                sizeof(std::int32_t)) != cudaSuccess) {
        return false;
    }
    s.slot_of_expert = static_cast<std::int32_t*>(tables);
    CUDA_CHECK(cudaMemset(s.slot_of_expert, 0xFF,
                          static_cast<std::size_t>(kMaxDeviceTables) * kMaxSlots *
                              sizeof(std::int32_t)));
    const auto slots_bytes =
        static_cast<std::size_t>(kMaxDeviceTables) * slots * sizeof(std::int32_t);
    void* per_slot = nullptr;
    CUDA_CHECK(cudaMalloc(&per_slot, slots_bytes));
    s.slot_expert = static_cast<std::int32_t*>(per_slot);
    CUDA_CHECK(cudaMemset(s.slot_expert, 0xFF, slots_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.slot_age), slots_bytes));
    CUDA_CHECK(cudaMemset(s.slot_age, 0, slots_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.slot_pass), slots_bytes));
    CUDA_CHECK(cudaMemset(s.slot_pass, 0xFF, slots_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.tick), sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemset(s.tick, 0, sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.work),
                          (1 + 2 * kMaxPaths) * sizeof(std::int32_t)));
    // FOUR counters: hits, misses, evictions, dropped. The slot count is a single constant in four
    // places (malloc, memset, the mirror copy, and the host zeroing) -- changing only some of them
    // makes the decide kernel write out of bounds AND makes the host read a stale zero, which is
    // exactly how "dropped=0" appeared while misses visibly fell.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&s.counters), 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(s.counters, 0, 4 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&s.counters_host),
                             4 * sizeof(unsigned long long), cudaHostAllocDefault));
    s.counters_host[0] = s.counters_host[1] = s.counters_host[2] = s.counters_host[3] = 0;
    s.slots = slots;
    s.ready = true;
    const double table_mib = static_cast<double>(kMaxDeviceTables * kMaxSlots *
                                                     sizeof(std::int32_t) +
                                                 slots_bytes * 3) /
                             (1024.0 * 1024.0);
    std::fprintf(stderr, "flash_next_experts_device_stage tables_mib=%.1f slots=%d\n", table_mib,
                 slots);
    return true;
}

int table_index_for(DeviceStageState& s, const void* key) {
    for (int i = 0; i < kMaxDeviceTables; ++i) {
        if (s.keys[i] == key) { return i; }
    }
    for (int i = 0; i < kMaxDeviceTables; ++i) {
        if (s.keys[i] == nullptr) {
            s.keys[i] = key;
            return i;
        }
    }
    throw std::runtime_error("Flash-Next device staging ran out of per-layer tables");
}

} // namespace

void flash_next_expert_device_stage_launch(
    const void* layer_key, std::int32_t* ids, std::int32_t paths, std::int32_t slots,
    float* alpha, float drop_threshold, const std::uint8_t* source_gate_up_codes,
    std::uint64_t source_gate_up_code_stride,
    const std::uint8_t* source_gate_up_scales, std::uint64_t source_gate_up_scale_stride,
    const std::uint8_t* source_down_codes, std::uint64_t source_down_code_stride,
    const std::uint8_t* source_down_scales, std::uint64_t source_down_scale_stride,
    const float* source_divisors_gate_up, const float* source_divisors_down,
    std::uint8_t* slot_gate_up_codes, std::uint8_t* slot_gate_up_scales, std::uint8_t* slot_down_codes,
    std::uint8_t* slot_down_scales, float* slot_divisors_gate_up, float* slot_divisors_down,
    std::int32_t pass, cudaStream_t stream) {
    DeviceStageState& s = device_stage_state();
    if (!ensure_device_state(s, slots)) {
        throw std::runtime_error("Flash-Next device staging could not allocate its slot tables");
    }
    const auto index    = static_cast<std::size_t>(table_index_for(s, layer_key));
    const auto slot_cnt = static_cast<std::size_t>(s.slots);
    auto* slot_of_expert = s.slot_of_expert + index * kMaxSlots;
    auto* slot_expert    = s.slot_expert + index * slot_cnt;
    auto* slot_age       = s.slot_age + index * slot_cnt;
    auto* slot_pass      = s.slot_pass + index * slot_cnt;

    flash_next_expert_decide_launch(ids, paths, slots, pass, s.tick, slot_of_expert, slot_expert,
                                    slot_age, slot_pass, s.work, source_divisors_gate_up,
                                    source_divisors_down, slot_divisors_gate_up, slot_divisors_down,
                                    s.counters, alpha, drop_threshold, stream);
    flash_next_expert_copy_launch(s.work, kMaxPaths, source_gate_up_codes,
                                  source_gate_up_code_stride, source_gate_up_scales,
                                  source_gate_up_scale_stride, source_down_codes,
                                  source_down_code_stride, source_down_scales,
                                  source_down_scale_stride, slot_gate_up_codes, slot_gate_up_scales,
                                  slot_down_codes, slot_down_scales, stream);
    if (++s.calls % 48 == 0) {
        // Pinned mirror, no sync: the printed values lag the device by one call.
        CUDA_CHECK(cudaMemcpyAsync(s.counters_host, s.counters, 4 * sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost, stream));
    }
}

const unsigned long long* flash_next_expert_device_stage_counters() {
    return device_stage_state().counters_host;
}

} // namespace ninfer::models::qwen3_8_flash_next::detail
