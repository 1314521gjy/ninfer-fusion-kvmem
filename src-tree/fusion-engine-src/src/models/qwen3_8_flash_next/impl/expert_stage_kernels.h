#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Device-side expert staging: the per-layer host round trip (read the router ids back, decide which
// slot each expert takes, then issue the copies) is replaced by two kernels on the compute stream.
// Measured premise (lever_probe.exe, this machine): a kernel copying straight out of the pinned host
// bank reaches 22.58 GB/s for the cached miss volume, against 22.57 GB/s for the driver DMA path --
// so moving the whole staging step to the device costs no bandwidth and removes ~24 ms/token of
// host latency from the 48-per-token critical path.
//
// decide: one thread walks the routed paths, resolves expert -> slot through the device slot table
// (Least-Stale then LRU victim), rewrites ids in place to slot indices, appends (expert, slot) to the
// work list, and gathers that slot's two weight-scale divisors. Must be launched with <<<1, 1>>>.
void flash_next_expert_decide_launch(std::int32_t* ids, std::int32_t paths, std::int32_t slots,
                                     std::int32_t pass, std::int32_t* tick,
                                     std::int32_t* slot_of_expert, std::int32_t* slot_expert,
                                     std::int32_t* slot_age, std::int32_t* slot_pass,
                                     std::int32_t* work, const float* source_divisors_gate_up,
                                     const float* source_divisors_down, float* slot_divisors_gate_up,
                                     float* slot_divisors_down, unsigned long long* counters,
                                     const float* alpha, float drop_threshold,
                                     cudaStream_t stream);

// copy: one grid over (work items x slices, four planes); every block copies its slice of one
// expert's plane out of the pinned host bank into the slot pool. Reads `work[0]` for the item count,
// so a fixed maximum-size grid covers a short list.
void flash_next_expert_copy_launch(const std::int32_t* work, std::int32_t max_items,
                                   const std::uint8_t* gate_up_codes,
                                   std::uint64_t gate_up_code_stride,
                                   const std::uint8_t* gate_up_scales,
                                   std::uint64_t gate_up_scale_stride,
                                   const std::uint8_t* down_codes, std::uint64_t down_code_stride,
                                   const std::uint8_t* down_scales, std::uint64_t down_scale_stride,
                                   std::uint8_t* slot_gate_up_codes, std::uint8_t* slot_gate_up_scales,
                                   std::uint8_t* slot_down_codes, std::uint8_t* slot_down_scales,
                                   cudaStream_t stream);

// Orchestration: owns the per-layer device slot tables and launches decide+copy for one layer.
// `layer_key` is the layer's host bank pointer (unique per layer, same key the host path uses).
// `slot_*` are this layer's sub-array of the slot pool; `source_*` are the pinned host banks.
// `alpha` is scratch.alpha (the renormalised top-k weights, device-resident and WRITABLE -- the R1
// fixup zeroes dropped paths' weights and rescales the survivors in place) and `drop_threshold`
// is drop_f/10 with 0 meaning "R1 not armed". Without them the device path silently ran with R1 OFF,
// which is what made it look slower than the host path: it was compared against a host arm that was
// dropping roughly half its misses, and R1 alone is worth +90%.
// Also mirrors the device hit/miss/eviction/dropped counters into pinned host memory without a sync.
void flash_next_expert_device_stage_launch(const void* layer_key, std::int32_t* ids,
                                           std::int32_t paths, std::int32_t slots,
                                           float* alpha, float drop_threshold,
                                           const std::uint8_t* source_gate_up_codes,
                                           std::uint64_t source_gate_up_code_stride,
                                           const std::uint8_t* source_gate_up_scales,
                                           std::uint64_t source_gate_up_scale_stride,
                                           const std::uint8_t* source_down_codes,
                                           std::uint64_t source_down_code_stride,
                                           const std::uint8_t* source_down_scales,
                                           std::uint64_t source_down_scale_stride,
                                           const float* source_divisors_gate_up,
                                           const float* source_divisors_down,
                                           std::uint8_t* slot_gate_up_codes,
                                           std::uint8_t* slot_gate_up_scales,
                                           std::uint8_t* slot_down_codes,
                                           std::uint8_t* slot_down_scales,
                                           float* slot_divisors_gate_up, float* slot_divisors_down,
                                           std::int32_t pass, cudaStream_t stream);

// Mirrored counters (host view): [0] hits, [1] misses, [2] evictions, [3] dropped.
[[nodiscard]] const unsigned long long* flash_next_expert_device_stage_counters();

} // namespace ninfer::models::qwen3_8_flash_next::detail
