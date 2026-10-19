#pragma once

// Port shim: the upstream `runtime/contract/types.h` this engine's ported Flash-Next target code
// includes.
//
// Upstream kept the whole execution/request/resource contract in one 25 KB header. This engine
// splits the same contract across four narrower headers, and every symbol the upstream header
// declared is already present here:
//
//   execution.h   LaneId · ConsumeStatus · CommitDisposition · CommitObservation · CommitDecision
//                 BeginSummary · RoundBudget
//   timing.h      ExecutionTiming · ExecutionTimingPhase · ExecutionTimingRecorder
//   request.h     ResolvedExecutionOptions · CancellationFlagView · RequestPlanSummary
//   resources.h   PrefillWork · CheckpointKind · CheckpointRef · ContextTransferObservation ·
//                 ContextOperationCounts · ContextTransaction{Status,ReserveStatus,InProgress} ·
//                 PlanningOwnerId · PlanningCandidateId · ProgramResourceRevision ·
//                 FinalScheduleIntent · PrivateSourceMode · VictimDisposition · FinishDisposition ·
//                 CheckpointRecoveryAlternativeWork · MaterializationPhysicalStatus ·
//                 IdentityMaterializationAssessment · Pressure* · SequenceCapacityCurve · ...
//
// So this is a forward, NOT a copy. Copying the upstream header was measured and rejected: it
// raises the target's compile errors (40 -> 526) because its declarations bind to the upstream
// engine's shapes, not this one's.
//
// A symbol the ported code needs but this engine lacks must be added to the narrow header that owns
// that part of the contract -- never by pasting a definition here, which would give one fact two
// sources.

#include "runtime/contract/execution.h"
#include "runtime/contract/request.h"
#include "runtime/contract/resources.h"
#include "runtime/contract/timing.h"

namespace ninfer::runtime {

// Upstream declares this one forward in the same header, because its `ResolvedExecutionOptions`
// holds a `std::shared_ptr<const CompiledOutputConstraint>` and the definition lives in
// `runtime/contract/structured_output.h` (which this header must not include: that one pulls in the
// schema types, and every request type would pay for it). Same reason, same placement.
class CompiledOutputConstraint;

} // namespace ninfer::runtime
