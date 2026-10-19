#pragma once
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

namespace ninfer::ops::detail {
struct Fp8LinearShape {
    std::int32_t n, k;
    Fp8Launch a16;
    void (*a8)(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace, cudaStream_t);
    bool (*uses_a8)(std::int32_t min_tokens, std::int32_t max_tokens);
};

extern const Fp8LinearShape kFp8N14336K5120;
extern const Fp8LinearShape kFp8N16384K5120;
extern const Fp8LinearShape kFp8N34816K5120;
extern const Fp8LinearShape kFp8N5120K6144;
extern const Fp8LinearShape kFp8N5120K17408;
extern const Fp8LinearShape kFp8N248320K5120;
// Qwen3.8-Flash-Next geometries (k = 2560 / 6144, 4-byte row scales). Registered in the standard
// table only: the unified twins would need the unified A16 path templated on the scale type as well
// (`fp8_a16_*`/`fp8_operands.h` still cast `weight.scales` to `__nv_bfloat16*`), so the bootstrap
// runs with `NINFER_LINEAR_ROUTES=legacy` instead. Widening a miss in the unified table stays loud:
// resolve_shape names the geometry and the table after this change.
extern const Fp8LinearShape kFp8N13312K2560;
extern const Fp8LinearShape kFp8N16384K2560;
extern const Fp8LinearShape kFp8N2560K6144;
} // namespace ninfer::ops::detail
