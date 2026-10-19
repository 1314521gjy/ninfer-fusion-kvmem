# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

add_library(ninfer_model_runtime STATIC)
ninfer_internal_includes(ninfer_model_runtime)
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support
  PRIVATE ninfer_media_decode ninfer_jinja)

include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)

# The ported Qwen3.8-Flash-Next (qwen4_exp) target and the vision tower it uses. Their exported API is
# reached as <ninfer/models/<name>/...>; that prefix lives inside each layer's `export` directory, so
# the flat spelling resolves through the include paths added below (the same shape the qwen3_5 prefix
# has via PROJECT_SOURCE_DIR/src).
include(qwen3_8_flash_next/loading_sources.cmake)
include(qwen3_8_flash_next/runtime_sources.cmake)
include(qwen3_vision/sources.cmake)

# PUBLIC, not PRIVATE: the load path that reaches these exported headers is `src/runtime/engine/
# model_instance.{h,cpp}` (the assembler that calls `Package::plan_load` and friends), and that lives
# in `ninfer_engine`, a different library. PRIVATE made the flat spelling resolve only inside the two
# model libraries, so the real build failed with C1083 on
# `<ninfer/models/qwen3_8_flash_next/package.h>` while a hand-built probe with an extra -I passed --
# a probe whose flags differed from the build's is not evidence. `ninfer_engine` links
# `ninfer_model_runtime` PRIVATE, which links `ninfer_model_loading` PUBLIC, so a PUBLIC include here
# reaches it.
foreach(target ninfer_model_loading ninfer_model_runtime)
  target_include_directories(${target} PUBLIC
    "${CMAKE_CURRENT_LIST_DIR}/qwen3_8_flash_next/export"
    "${CMAKE_CURRENT_LIST_DIR}/qwen3_vision/export")
endforeach()
