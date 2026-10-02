# GGUF Support in NInfer-v100

This document outlines the GGUF integration architecture in NInfer-v100, including parser implementation, dequantization design decisions, tensor mapping, and conversion pipeline.

---

## 1. Dequantization Strategy: Vendored GGML vs. Custom CUDA Kernels

### Explicit Architectural Decision
NInfer-v100 vendors the core dequantization functions from `ggml-quants` (MIT licensed) under `third_party/ggml_dequant/`, wrapping them in an efficient, vectorized C++ host-side conversion and dequantization pipeline.

### Justification & Tradeoff Analysis

1. **Build Complexity**:
   - `ggml-quants.c` is pure ANSI C / C99 with optional AVX2/FMA/NEON vectorization. It compiles without external dependencies (no complex GGML computation graph or runtime framework required).
   - In contrast, writing custom CUDA dequantization kernels for each GGML quant format (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q2_K`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`) would require thousands of lines of complex GPU bit-manipulation, block scale/min unpacking, and non-linear importance codebook lookups.
   - Using vendored `ggml-quants` keeps build complexity minimal and avoids code bloat in the GPU kernel tree.

2. **License Compatibility**:
   - `ggml-quants` is distributed under the permissive **MIT License** (by Georgi Gerganov and GGML authors).
   - NInfer is licensed under the **Apache License 2.0**.
   - The MIT license is fully permissive and compatible with Apache 2.0 (matching `third_party/llama_cpp_fattn/`). Attribution notices are preserved in `third_party/ggml_dequant/LICENSE`.

3. **Performance & Hardware Alignment**:
   - Dequantization in NInfer-v100 occurs at load/conversion time. Converting weights into NInfer's internal native layouts (`RowSplitK128V1` or contiguous BF16) allows NInfer's custom Tesla V100 CUTLASS and SIMT GEMM kernels to run at peak memory bandwidth without runtime dequantization overhead during token generation.
   - Vectorized CPU dequantization (with `-mavx2 -mfma`) processes gigabytes of weights in seconds, producing clean, aligned tensors ready for direct GPU execution.

4. **Format Coverage**:
   - In addition to standard `Q4_K`, `Q5_K`, and `Q8_0`, the vendored dequantizer immediately supports mixed-precision Importance Quantization (IQ) models, such as `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf`.

---

## 2. GGUF Parser Implementation

The parser is implemented in `include/ninfer/gguf/gguf_parser.h` and `src/gguf/gguf_parser.cpp`:
- **Header Parsing**: Supports GGUF v2 and v3 (magic `0x47475546`).
- **Metadata KV**: Reads typed key-value pairs (integers, floats, booleans, strings, arrays).
- **Tensor Descriptors**: Extracts logical shapes, dimensions, `ggml_type`, and byte offsets within the data blob.
- **Memory Mapping**: Uses `mmap` for fast, zero-copy access to tensor blobs.
- **Split File Support**: Automatically detects and loads split files matching the pattern `*-00001-of-000NN.gguf` using `split.count` and `split.no` metadata.

---

## 3. Tensor Mapping & Repacking

NInfer expects fused and aligned tensors (e.g. `mlp/gate_up`, `attention/query_key`, `gdn/query_key`). The GGUF converter maps individual GGUF tensors (`blk.N.ffn_gate`, `blk.N.ffn_up`, etc.) into NInfer's format:
- Gate and Up projections are concatenated into `gate_up` matrices.
- GDN SSM tensors (`ssm_a`, `ssm_conv1d`, `ssm_alpha`, `ssm_beta`, `ssm_norm`, `ssm_out`) are mapped to `gdn/*`.
- Attention projections are fused and padded to 128-column alignment for V100 GEMM kernels.
- The output artifact is saved with `NINFER\0\2` framing and direct device layout metadata.
