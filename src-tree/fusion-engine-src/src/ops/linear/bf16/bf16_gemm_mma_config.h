#pragma once

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"

namespace ninfer::ops::detail {

// Measured large-T production schedule for the BF16 computation core. Geometry remains a template
// argument so an exact problem can replace any tile, pipeline, cache, raster, or fragment choice
// without changing either Linear or semantic-Op dispatch.
template <class Geometry>
struct Bf16MmaProductionScheduleSelector {
    using Type = Bf16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
};

template <>
struct Bf16MmaProductionScheduleSelector<Bf16GemvGeometry<4304, 1152>> {
    using Type = Bf16MmaSchedule<16, 128, 64, 16, 32, 2, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;
};

template <>
struct Bf16MmaProductionScheduleSelector<Bf16GemvGeometry<1152, 4304>> {
    using Type = Bf16MmaSchedule<64, 64, 16, 32, 32, 2, 2, Cache::cg, Cache::cg,
                                 Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
                                 Bf16MmaSwizzle::Plain>;
};

template <class Geometry>
using Bf16MmaProductionSchedule = typename Bf16MmaProductionScheduleSelector<Geometry>::Type;

} // namespace ninfer::ops::detail
