# Phase 0 Report — Fork and Reconnaissance

## 1. Overview and Objectives
Phase 0 establishes the legal baseline, sets up the repository fork under account `noorazman` with strict safeguards preventing accidental pushes to upstream `geoffwatts/ninfer-v100`, maps the codebase architecture, and evaluates the critical feasibility question of loading GGUF format models into NInfer's internal tensor inventory.

---

## 2. Step 0.1 — License Check
- **Upstream Repository**: `geoffwatts/ninfer-v100` (master branch)
- **License**: **Apache License, Version 2.0** (January 2004)
- **Clauses & Permissions**:
  - **Section 2 (Grant of Copyright License)**: Grants perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, sublicense, and distribute the Work.
  - **Section 3 (Grant of Patent License)**: Grants perpetual, worldwide patent license.
  - **Section 4 (Redistribution & Modification Conditions)**:
    - Must provide recipients a copy of the License.
    - Modified files must carry prominent notices stating they were changed.
    - Must retain all copyright, patent, trademark, and attribution notices.
- **Finding**: The license explicitly permits forking, modification, redistribution, and derivative works.

---

## 3. Step 0.2 — Fork and Remote Configuration
- **Fork URL**: `https://github.com/noorazman/ninfer-v100`
- **Local Clone Path**: `/home/noorazman/dsh/ninfer-v100-gguf`
- **Tracked Default Branch**: `master` (synchronized with `origin/master`, identical SHA `b37d0dd3e1163b9d802d8bccfa89918bf68d793e` with `upstream/master`).
- **Configured Remotes**:
  ```text
  origin    https://github.com/noorazman/ninfer-v100 (fetch)
  origin    https://github.com/noorazman/ninfer-v100 (push)
  upstream  https://github.com/geoffwatts/ninfer-v100 (fetch)
  upstream  DISABLED (push)
  ```
- **Push Protection**:
  - `git remote set-url --push upstream DISABLED`
  - Installed executable `.git/hooks/pre-push` refusing any push to URLs containing `geoffwatts/ninfer-v100`. Verified by direct invocation and dry-run tests.

---

## 4. Step 0.3 — Reconnaissance Findings

### 4.1 Subsystem Mapping
1. **Weight Loading and Artifact Parsing**:
   - `src/artifact/reader.h` & `src/artifact/reader.cpp`: parses `NINFER\0\2` binary framing, JSON headers, and payload offsets.
   - `src/artifact/binder.h` & `src/artifact/binder.cpp`: binds logical tensor shapes and types to descriptors.
   - `src/artifact/materializer.h` & `src/artifact/materializer.cpp`: streams tensors into GPU memory.
   - `src/targets/registry.cpp`: orchestrates artifact loading and target initialization.
2. **Internal Tensor Naming & Layout**:
   - `src/targets/qwen3_6_27b/impl/load/bindings.h` & `bindings.cpp`: defines exact hierarchy (`text/token_embedding`, `text/layers/{i}/...`, `text/output_head`).
   - `src/artifact/storage_layouts.cpp`: defines planar layouts (`RowSplitK128V1`, `BlockScaleK16M128x4V1`, `RowScaleV1`).
3. **CUDA Device and Context Initialization**:
   - `src/core/device.h` & `src/core/device.cu`: `DeviceContext` manages CUDA streams and device properties.
4. **Forward Pass and Layer Execution Loop**:
   - `src/targets/qwen3_6/impl/runtime/text_context_impl.h`: `run_layers` executes sequential layer mixers (attention / GDN) and MLPs.
   - `src/targets/qwen3_6/impl/runtime/decode_impl.h`: executes decode passes under CUDA graph capture.
5. **KV Cache Allocation & Management**:
   - `src/core/paged_kv_cache.h` & `src/core/paged_kv_cache.cpp`: manages 64-token paged KV storage pools.
6. **CLI Argument Parsing**:
   - `apps/cli/options.h` & `apps/cli/options.cpp`: parses positional `.ninfer` model path and execution flags.

### 4.2 Critical Feasibility Question
- **Direct Mapping vs. Converter**: A model-specific converter tool is **strictly required**. GGUF weights cannot be mapped directly into NInfer's internal tensor inventory because:
  1. GGUF quantization formats (e.g. Q4_K, Q5_K, Q8_0, IQ quants) use interleaved block encodings with affine offsets (`d`, `dmin`, scales).
  2. NInfer's Volta GEMM kernels require planar `RowSplitK128V1` storage (separate low-nibble, high-bit, and scale planes) with 128-column row padding.
  3. NInfer requires fused projection matrices (e.g. `mlp/gate_up`, attention `query_key`, GDN `value_z`), whereas GGUF provides individual unfused tensors.

### 4.3 Hardware Verification
- **Target GPUs**: Tesla V100-SXM2-16GB (GPU 3 and GPU 4), each with 16,142 MiB free VRAM.
- **P2P Capability**: Direct peer access between GPU 3 and GPU 4 confirmed active (`cudaDeviceCanAccessPeer(3, 4) == 1`).
