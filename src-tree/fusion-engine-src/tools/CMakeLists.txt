# Standalone CUDA enumeration probe, plus a real DeviceContext construction check.
#
# It deliberately does not link ninfer_core. The enumeration half needs to answer "which CUDA
# indices are the two cards, and can they reach each other" before the model loader is involved,
# on a machine where the answer is not obvious -- a display GPU commonly takes cuda=0 even when
# nvidia-smi lists it elsewhere, so the indices for --devices must be read from a CUDA-level
# enumeration rather than assumed.
#
# device.cu is compiled in directly rather than linked from the core. It only includes core/device.h
# and standard headers, so this costs a single translation unit and keeps the probe around 1 MB
# instead of the ~500 MB a core link would produce -- which matters because this binary gets
# uploaded to rented hardware to be run there.
add_executable(ninfer-multi-gpu-probe
  multi_gpu_probe.cu
  ${CMAKE_SOURCE_DIR}/src/core/device.cu)
target_include_directories(ninfer-multi-gpu-probe PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(ninfer-multi-gpu-probe PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# Go/no-go probe for tensor-parallel all-reduce transports (peer push, mapped host, staged host,
# optionally NCCL): per-collective latency in graphs, a correctness soak, and the pinned-memory
# portability check. Standalone like the probe above, so it can be uploaded to rented hardware.
add_executable(ninfer-tp-probe tp_probe.cu)
set_target_properties(ninfer-tp-probe PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# Measures what the dequantise-to-int8 prefill route would cost on the weight side: the group-scale
# spread within a row, and the relative L2 of reconstructing a weight as int8 with one scale per
# row. Reads the real artifact, because the answer depends on the actual scale distribution.
add_executable(ninfer-w4-row-scale-error w4_row_scale_error.cpp)
target_link_libraries(ninfer-w4-row-scale-error PRIVATE ninfer_artifact)
ninfer_internal_includes(ninfer-w4-row-scale-error)
set_target_properties(ninfer-w4-row-scale-error PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# Per-object geometry check against a real artifact's own directory: recomputes every object's
# encoded size with the engine's own `parse_format` / `parse_layout` / `tensor_geometry` and compares
# it with the `bytes` the container recorded. No device, no model load, no reference engine -- the
# cheapest instrument for the question "can this engine read this artifact".
#
# Registered late on purpose being noted: this source existed and was run before, but no target
# listed it, so in THIS build tree the binary did not exist and the offline regression could not be
# re-run. That is the same failure class as `src/artifact/CMakeLists.txt`'s port-era additions:
# a file that is in the tree but in no target, whose only symptom is much later.
#
# `tensor_geometry` is shared with the engine (`artifact/layouts.h`), so a green run here means the
# engine computes the same byte count -- which is all this tool claims.
add_executable(ninfer-artifact-geometry-check artifact_geometry_check.cpp)
target_link_libraries(ninfer-artifact-geometry-check PRIVATE ninfer_artifact)
ninfer_internal_includes(ninfer-artifact-geometry-check)
set_target_properties(ninfer-artifact-geometry-check PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
