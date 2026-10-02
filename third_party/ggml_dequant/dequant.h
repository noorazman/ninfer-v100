#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, -1 on unsupported type
int ninfer_ggml_dequantize_row(int ggml_type, const void* src, float* dst, int64_t n_elements);

// Dequantize directly to bfloat16 (stored as uint16_t words)
int ninfer_ggml_dequantize_to_bf16(int ggml_type, const void* src, uint16_t* dst_bf16, int64_t n_elements);

const char* ninfer_ggml_type_name(int ggml_type);
size_t ninfer_ggml_type_size(int ggml_type);
int64_t ninfer_ggml_blck_size(int ggml_type);
size_t ninfer_ggml_row_size(int ggml_type, int64_t n_elements);

#ifdef __cplusplus
}
#endif
