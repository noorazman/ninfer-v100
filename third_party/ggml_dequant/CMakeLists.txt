add_library(ninfer_ggml_dequant STATIC
  ggml-quants.c
  dequant.cpp
)

target_include_directories(ninfer_ggml_dequant PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/ggml-cpu
)

target_compile_options(ninfer_ggml_dequant PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-O3 -fPIC -mavx2 -mfma>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -fPIC -mavx2 -mfma>
)
