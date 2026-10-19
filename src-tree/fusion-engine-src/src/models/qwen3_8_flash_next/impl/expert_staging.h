#pragma once

#include "models/qwen3_8_flash_next/impl/model_view.h"
#include "models/qwen3_8_flash_next/impl/moe_workspace.h"

#include <cuda_runtime.h>

namespace ninfer::models::qwen3_8_flash_next::detail {

// Host-tiered expert staging for one MoE call.
//
// The engine's expert banks normally live in the device arena, and the main bank is 63.3 GiB across
// the 48 text layers -- more than a 32 GB card can hold (the target plan asks for 70.01 GiB). With
// NINFER_FLASH_NEXT_EXPERTS_HOST the banks are bound into the artifact's read-only file mapping
// instead (bind_moe's retain_experts_on_host), and this function brings just the experts the router
// selected into device memory for the call.
//
// It copies each unique expert's codes and scales plus both full divisor planes from the mapping,
// rewrites scratch.ids in place from expert ids to staging slot indices, and returns a MoeWeights
// whose expert banks point at the staging planes. The MoE kernels are untouched: they already treat
// ids as an index into `codes + expert * code_bytes_per_expert`, so a slot index is a drop-in
// replacement. Placement only -- no arithmetic changes.
//
// Preconditions, all enforced (throwing, never silent):
//   * both banks report on_host, and their strides equal this target's compiled expert geometry
//   * 1 <= tokens <= kMaxStagedTokens (32) and the staging planes are allocated
//   * the router's DISTINCT expert count fits the destination: the slot pool when there is one,
//     kStagedExpertSlots for the no-cache planes otherwise. The count is computed from the ids this
//     call already brings to the host, because exceeding it would not crash -- it would make the LRU
//     re-use a slot that earlier paths of the same call still point at, silently feeding them another
//     expert's weights. On the device staging path, which has no such readback, tokens > 8 is refused
//     for exactly this reason.
// Judgements: NINFER_FLASH_NEXT_EXPERTS_VERIFY=1 byte-compares every staged plane against the
// mapping and aborts on any mismatch, printing "flash_next_experts_verify slots=.. bytes=..
// mismatched_planes=.. sabotage=.."; NINFER_FLASH_NEXT_EXPERTS_SABOTAGE=1 stages a neighbouring
// expert on purpose so that the verifier must go red (a judgement nobody has seen fail is not a
// judgement). Both are read once per process.
// Block-6 W1: `serve_input` is this call's MoE input [kHidden, tokens] BF16 (device). When the CPU
// serve forward face is armed (serve master switch on, decode width, this target's geometry) the
// staging mirrors it to host and the serve team computes the missed experts' real contribution;
// nullptr keeps the landing-only round, which is what every non-decode caller passes.
[[nodiscard]] MoeWeights stage_flash_next_moe_experts(const MoeWeights& weights,
                                                      const FlashNextMoeWorkspace& scratch,
                                                      int tokens, cudaStream_t stream,
                                                      const Tensor* serve_input);

// Block-6 W2: after the MoE kernels have written `output`, add the CPU-served experts' real
// contribution (the serve team's partial from the forward face) into it. Called on the same host
// thread that staged the call, right after the kernel launch; a no-op unless the round that was
// just committed ran the forward face. Bounded wait -- a partial that is not fully answered is not
// added (never a torn read) and is counted in the serve self-report line's fwd_stalls.
void flash_next_cpu_serve_apply(Tensor& output, const FlashNextMoeWorkspace& scratch, int tokens,
                                cudaStream_t stream);

// P5-1. Tell the staging cache how many draft columns of the round just staged were actually
// ACCEPTED, so experts used only by rejected columns do not count as uses.
//
// Why a separate call and not a parameter: the acceptance does not exist when the experts are
// staged. Staging runs inside the verification forward, layer by layer; accept/reject is only known
// once that forward has finished (lane_ledger.cpp commit_speculative_round). So the credit is
// DEFERRED, not retracted: with NINFER_FLASH_NEXT_MOE_LIVE_COLUMNS on, a hit records the pair
// (resident key, column) instead of incrementing the frequency plane, and this call commits only the
// pairs whose column is below `accepted_count`.
//
// A scalar suffices because acceptance is a PREFIX: speculative_round.cuh:164-165 "the first failed
// event determines the prefix", and :193 writes a live column as `column < accepted_count`. So 0
// means "every column was rejected". Ignored when the switch is off, and when nothing is pending --
// a call with nothing to commit is counted, not an error.
void flash_next_note_live_columns(std::int32_t accepted_count);

} // namespace ninfer::models::qwen3_8_flash_next::detail
