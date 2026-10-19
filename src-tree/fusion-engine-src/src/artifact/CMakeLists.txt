add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  binder.cpp
  materializer.cpp
  transcode.cpp
  permute_row_split.cu
  views.cpp
  # Port-era additions that were left out of this list. Without them the symbols exist in headers
  # but in no library, and the only symptom is a link error much later:
  #   typed_binding.cpp -> materialized_tensor / materialized_weight (called by the ported loader)
  #   bank_geometry.cpp -> block_scale_bank_geometry (called by the ported expert bank)
  typed_binding.cpp
  bank_geometry.cpp
)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json)
