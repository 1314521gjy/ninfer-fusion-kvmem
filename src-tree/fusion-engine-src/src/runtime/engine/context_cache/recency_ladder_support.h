#pragma once

namespace ninfer::runtime {

// An opt-in tag, and the ONLY thing a model's Program header needs from the engine's context-cache
// layer. A contract declares that its `CapturePressurePlanningSession` implements the ContextCache
// recency ladder by deriving from this tag; `runtime/engine/context_cache/resource_manager.h` then
// selects the engine's `MaterializationPlanner` for it. A contract that does not derive from it gets
// the refusing `NoPressurePlanner` instead.
//
// WHY A SEPARATE HEADER
//   It lived in `resource_manager.h` at first, and the fusion `models/qwen3_5/program/program.h`
//   then could not see it: that model header does not include the engine's resource manager and must
//   not (measured -- the build failed with `"RecencyLadderSupported": 不是 "ninfer::runtime" 的成员`,
//   `未定义基类`). Putting the tag in its own file keeps the dependency to one line with no cycle:
//   `program.h` -> this header, and `resource_manager.h` -> this header.
//
// WHY A TAG RATHER THAN A NAME PROBE
//   The engine header must not name any model, so the selection has to be detected. Detecting a
//   member name does not work here: both contracts alias `CapturePressurePlanningSession`
//   (`models/qwen3_8_flash_next/.../target_contract.h:62`), so a name probe matched everywhere and
//   the ported contract was still sent to `MaterializationPlanner` (measured: the identical 19
//   errors). A tag cannot be aliased into existence by accident.
struct RecencyLadderSupported {};

} // namespace ninfer::runtime
