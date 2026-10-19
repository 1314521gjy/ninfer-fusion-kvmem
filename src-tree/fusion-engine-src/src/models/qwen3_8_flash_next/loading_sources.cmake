# qwen3_8_flash_next, immutable side: what the loader binds and materializes.
#
# Moved in from igorls@c81c88f6 (`src/targets/qwen3_8_flash_next/impl/load/`). The list is exactly
# the compile units that pass the single-file probe (`E:\infer-build\port-probe\scan-final`, 39/39,
# 0 errors), so a file can never be silently missing from the build.
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/impl/load/bindings.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/load/loader.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/load/materialized.cpp"
)
