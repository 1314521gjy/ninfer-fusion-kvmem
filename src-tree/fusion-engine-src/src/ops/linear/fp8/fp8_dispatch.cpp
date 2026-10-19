#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/common/route_table.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_shapes_unified.h"
#include "ops/linear/fp8/fp8_format.h"
#include <array>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
                         &kFp8N5120K6144,  &kFp8N5120K17408, &kFp8N248320K5120,
                         &kFp8N13312K2560, &kFp8N16384K2560, &kFp8N2560K6144};
const std::array kUnifiedShapes{&unified::kFp8N14336K5120, &unified::kFp8N16384K5120,
                                &unified::kFp8N34816K5120, &unified::kFp8N5120K6144,
                                &unified::kFp8N5120K17408, &unified::kFp8N248320K5120};

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                    LinearRouteTable table) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    // Two explicit branches rather than a ternary over the two arrays: they no longer hold the same
    // number of entries (the standard table carries the three Qwen3.8-Flash-Next geometries, the
    // unified table does not), so a conditional expression cannot unify their types.
    if (table == LinearRouteTable::Unified) {
        for (const auto* shape : kUnifiedShapes)
            if (shape->n == n && shape->k == k) return *shape;
    } else {
        for (const auto* shape : kShapes)
            if (shape->n == n && shape->k == k) return *shape;
    }
    // Name the offending geometry and the table it was looked up in: a bare "unsupported shape" cost
    // a whole round on 2026-10-11 (docs/20 §16.35), and the two tables hold different sets.
    throw std::invalid_argument(std::string("fp8 linear: unsupported shape n=") + std::to_string(n) +
                                " k=" + std::to_string(k) + " table=" +
                                (table == LinearRouteTable::Unified ? "unified" : "standard"));
}
} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    return capacity_by_table(
        min_tokens, max_tokens,
        [](std::int32_t width) { return linear_route_table(LinearRouteFamily::Fp8, width); },
        [&](LinearRouteTable table, std::int32_t first, std::int32_t last) -> std::size_t {
            const bool a8 = resolve_shape(n, k, policy, table).uses_a8(first, last);
            return allows_a8(policy) && a8 ? fp8_a8_workspace_capacity_bytes(last, k) : 0;
        });
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy,
                                      linear_route_table(LinearRouteFamily::Fp8, x.ne[1]));
    if (!allows_a8(policy) || !shape.uses_a8(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
    shape.a8(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
