# Phase 1 Report — GGUF Support

## 1. Overview and Objectives
Phase 1 implements complete GGUF model ingestion capabilities in NInfer-v100 on branch `feature/gguf-loader`. It delivers a native GGUF v2/v3 parser, a comprehensive dequantization engine supporting standard and Importance Quantization (IQ) formats, a tensor name mapping and repacking pipeline, a standalone converter binary, CLI flags, and verification test suites.

---

## 2. Deliverables Summary

### 2.1 GGUF Parser
- **Source**: `include/ninfer/gguf/gguf_parser.h` and `src/gguf/gguf_parser.cpp`.
- **Functionality**:
  - Validates GGUF magic (`0x47475546`) and versions 2 & 3.
  - Parses typed metadata key-value pairs (integers, floats, strings, bools, arrays).
  - Reads tensor descriptors (names, dimensions, GGML type, data blob offset).
  - Uses `mmap` for zero-copy binary access to tensor payloads.
  - Automatically discovers and stitches multi-file split GGUFs (`*-00001-of-000NN.gguf`).

### 2.2 Dequantization Engine
- **Source**: `third_party/ggml_dequant/` (`dequant.h`, `dequant.cpp`, `ggml-quants.c`).
- **Functionality**:
  - Dequantizes GGML blocks to FP32 and BF16 with AVX2/FMA vectorization.
  - Format support includes:
    - Standard: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K`.
    - Importance Quantization (IQ): `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`.
    - Direct: `F32`, `F16`, `BF16`.
- **Architectural Rationale**: Documented in `docs/gguf.md`. Explains the decision to vendor `ggml-quants` (MIT licensed) rather than writing custom CUDA kernels, based on build simplicity, broad format coverage, license compatibility, and V100 runtime performance.

### 2.3 Tensor Name Mapper & Repacking Converter
- **Source**: `src/gguf/converter.h`, `src/gguf/converter.cpp`, `apps/gguf_convert/main.cpp`.
- **Functionality**:
  - Maps GGUF tensor names to NInfer internal inventory:
    - Fuses MLP gate and up matrices into `mlp/gate_up` $[34816, 5120]$.
    - Fuses attention projections into `attention/query_key` and `attention/gate_value`.
    - Fuses GDN projections into `gdn/query_key` and `gdn/value_z`.
    - Maps SSM tensors (`ssm_a`, `ssm_conv1d`, `ssm_dt`, `ssm_alpha`, `ssm_beta`, `ssm_norm`, `ssm_out`) to `gdn/*`.
    - Converts MTP layer 64 tensors when present.
  - Repacks dequantized matrices into planar `RowSplitK128V1` (`Q4G64_F16S`, `Q5G64_F16S`, `W8G32_F16S`) with 128-column padding and 256-byte alignment.
  - Emits valid `.ninfer` container with `NINFER\0\2` binary header and embedded tokenizer resources.
  - Standalone conversion binary: `ninfer-gguf-convert`.

### 2.4 Config Extraction & CLI Flag
- **Config Extraction**: `GGUFFile::extract_config()` extracts hidden size, intermediate size, layer count, head count, head dim, RoPE parameters, RMSNorm epsilon, vocab size, and hybrid SSM geometry.
- **CLI Flag**: Added `--gguf-model <path>` to `ninfer` (`apps/cli/options.h`, `apps/cli/options.cpp`, `apps/cli/main.cpp`). Automatically converts and loads GGUF models while preserving the existing `.ninfer` execution path unchanged.

---

## 3. Verification & Test Results

Executed on **Tesla V100 GPU 3**:

1. **`ninfer_gguf_parser_test`** (`tests/test_gguf_parser.cpp`):
   - Verified GGUF v3 single-file metadata on `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf` (866 tensors, 49 KV pairs).
   - Verified GGUF v3 metadata on `qwen2.5-1.5b-instruct-q8_0.gguf` (339 tensors).
   - Verified split GGUF multi-shard loading on `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (2 shards, 1224 tensors).
   - **Result**: **PASS** (0.27s).

2. **`ninfer_gguf_loader_test`** (`tests/test_gguf_loader.cpp`):
   - Verified shape extraction, dequantization of Q8_0 and IQ2_S weights, and executed `ops::rmsnorm` forward passes on GPU 3.
   - Outputs verified 100% finite.
   - **Result**: **PASS** (1.95s).

3. **`ninfer_cli_options_test`** (`tests/test_cli_options.cpp`):
   - Verified `--gguf-model` flag parsing, usage display, and backward compatibility.
   - **Result**: **PASS**.

4. **Full Test Suite Status**:
   - Total Tests: 106
   - Passed: 98
   - Skipped: 7 (unrelated offline full-model benchmarks)
   - Failed: 1 (pre-existing `ninfer_gdn_gating_proj_test` baseline tolerance discrepancy on Volta sm_70)
   - Zero regressions introduced.

---

## 4. Git Metadata
- **Branch**: `feature/gguf-loader`
- **Commit SHA**: `5c6a2cf1a29e4d2519c97d7ca60520a86d7fedfe`
- **Remote**: Pushed to `origin` (`https://github.com/noorazman/ninfer-v100`)
