# The measured device route profiles ship inside the binary: device_profiles.json (ninfer-calibrate
# --print output of each part, merged) becomes a string in the generated source.
set(NINFER_DEVICE_PROFILES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/engine/device_profiles.json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${NINFER_DEVICE_PROFILES_FILE})
file(READ ${NINFER_DEVICE_PROFILES_FILE} NINFER_DEVICE_PROFILES_JSON)
configure_file(engine/device_profiles_builtin.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/engine/device_profiles_builtin.cpp @ONLY)

add_library(ninfer_runtime_support STATIC
  engine/admission_policy.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/device_profile.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/engine/device_profiles_builtin.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
  prefix_cache/block_hash.cpp
  prefix_cache/prefix_index.cpp
  prefix_cache/tap_planner.cpp
)
ninfer_internal_includes(ninfer_runtime_support)
target_link_libraries(ninfer_runtime_support PUBLIC ninfer_core ninfer_ops PRIVATE ninfer::json)

# Public Engine and common contracts; model execution remains in the model layer.
add_library(ninfer_engine STATIC
  engine/model_instance.cpp
  contract/sampling.cpp
  # Port-era addition left out of this list. Defines `ninfer::runtime::OutputConstraintState`
  # (constructor, destructor, next_mask, accept, try_accept, fork, terminated) -- the type the ported
  # Program calls. Note `src/text/structured_output.cpp` is a DIFFERENT file in namespace
  # `ninfer::text` (GrammarState / StructuredCompiler) and does not define it; that similarity of
  # names is why this was easy to miss. Without this unit the only symptom was LNK2019.
  contract/structured_output.cpp
  engine/engine.cpp
)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
# `engine/model_instance.h` (a PUBLIC header of THIS library) includes
# `<ninfer/models/qwen3_8_flash_next/package.h>` and `<ninfer/models/qwen3_8_flash_next/target_contract.h>`.
# `src/models/CMakeLists.txt` adds those two `export/` directories as PUBLIC includes on
# `ninfer_model_loading` / `ninfer_model_runtime`, but `ninfer_engine` links `ninfer_model_runtime`
# PRIVATE (line 43) -- so they never reach a consumer of `ninfer_engine`, and any target that only
# links `ninfer_engine` (e.g. `tests/test_context_cache_defaults.cpp`, which includes model_instance.h)
# used to die with `fatal error C1083`. Adding the two directories here is the minimal fix; do NOT
# switch the link to PUBLIC (that would also export llama-jinja and the other model_runtime deps).
target_include_directories(ninfer_engine PUBLIC
  ${PROJECT_SOURCE_DIR}/src/models/qwen3_8_flash_next/export
  ${PROJECT_SOURCE_DIR}/src/models/qwen3_vision/export)
# `contract/structured_output.cpp` includes `<xgrammar/xgrammar.h>` and `<nlohmann/json.hpp>`. Written
# the way the only other consumer of that header does it (`src/text/CMakeLists.txt:8,7`).
target_include_directories(ninfer_engine PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_engine PRIVATE
  ninfer_calibration
  ninfer_model_runtime
  ninfer_artifact
  ninfer_core
  ninfer_runtime_support
  ninfer_ops
  ninfer_text
  ninfer_xgrammar
  ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)
