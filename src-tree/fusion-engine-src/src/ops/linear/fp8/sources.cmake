target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/fp8_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k5120_unified.cu"
  # Qwen3.8-Flash-Next geometries: 4-byte row scales, A16 only (sm_89: FP8 A8 is a hard stub), and
  # registered in the standard table only - the bootstrap runs with NINFER_LINEAR_ROUTES=legacy.
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n13312_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k6144.cu"
)
