# qwen3_vision: the target-neutral Qwen3 Vision encoder tower.
#
# Moved in from igorls@c81c88f6 (`src/targets/qwen3_vision/`). One compile unit, plus its public
# header tree under export/ninfer/models/qwen3_vision/ (which the CMake include path below points at).
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/impl/encoder.cpp"
)
