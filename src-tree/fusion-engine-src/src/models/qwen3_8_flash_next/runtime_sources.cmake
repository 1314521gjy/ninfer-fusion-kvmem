# qwen3_8_flash_next, mutable side: the architecture execution layer.
#
# Moved in from igorls@c81c88f6 (`src/targets/qwen3_8_flash_next/impl/`). Together with
# loading_sources.cmake this is exactly the 39 compile units that pass the single-file probe
# (`E:\infer-build\port-probe\scan-final`, 39/39, 0 errors). `ninfer_cuda_archive` is used the same
# way the other model targets use it: the .cu units are compiled into this static library.
target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/impl/expert_bank.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/expert_stage_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/expert_staging.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/gdn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/gdn_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/hyper_connection.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/hyper_connection_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/lane_ledger.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/moe.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/moe_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/moe_route.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/moe_shared_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/mtp_forward.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/mtp_forward_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/package.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/ple_decode.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/ple_decode_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/ple_index.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/ple_pipeline.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/ple_table.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/qsa_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/qsa_attention_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/qsa_indexer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/qsa_indexer_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/runtime_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/runtime_state.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/stage_ledger.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/text_decode.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/text_decode_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/text_executor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/vision_execute.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/port/binder_port.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/impl/load/quantize_nvfp4_expert_bank.cu"
  "${CMAKE_CURRENT_LIST_DIR}/impl/load/quantize_output_head.cu"
)
