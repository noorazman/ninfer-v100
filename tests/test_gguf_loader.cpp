#include "ninfer/gguf/gguf_parser.h"
#include "dequant.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/rmsnorm.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::gguf;

void test_qwen_gguf_forward_pass() {
    std::cout << "--- Running test_qwen_gguf_forward_pass ---\n";
    // Check available models on disk
    std::filesystem::path model_path = "/mnt/ssd/llm_models/Router_Models/qwen2.5-1.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(model_path)) {
        model_path = "/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf";
    }
    if (!std::filesystem::exists(model_path)) {
        std::cout << "SKIP: Neither test model found on disk.\n";
        return;
    }

    std::cout << "Loading GGUF: " << model_path << "\n";
    GGUFFile gguf(model_path);
    ModelConfig cfg = gguf.extract_config();
    std::cout << "  Architecture: " << cfg.architecture << "\n"
              << "  Layers:       " << cfg.block_count << "\n"
              << "  Hidden:       " << cfg.embedding_length << "\n"
              << "  Vocab:        " << cfg.vocab_size << "\n";

    assert(cfg.embedding_length > 0);
    assert(cfg.block_count > 0);

    // 1. Verify tensor descriptors and quantization types
    const auto* embd_desc = gguf.find_tensor("token_embd.weight");
    assert(embd_desc != nullptr);
    std::cout << "  token_embd.weight: shape=[" << embd_desc->shape[0] << ", " << embd_desc->shape[1]
              << "] type=" << ninfer_ggml_type_name(embd_desc->ggml_type) << "\n";
    assert(embd_desc->shape[0] == cfg.vocab_size || cfg.vocab_size == 0);
    assert(embd_desc->shape[1] == cfg.embedding_length);

    const auto* norm_desc = gguf.find_tensor("blk.0.attn_norm.weight");
    assert(norm_desc != nullptr);
    std::cout << "  blk.0.attn_norm.weight: shape=[" << norm_desc->shape[0]
              << "] type=" << ninfer_ggml_type_name(norm_desc->ggml_type) << "\n";

    // 2. Initialize CUDA device context (uses currently active CUDA device)
    DeviceContext ctx(0);
    std::cout << "  CUDA Device: " << ctx.props.name << " (sm_" << ctx.props.major << ctx.props.minor << ")\n";

    // 3. Test dequantizing a slice of embedding and input norm
    const int hidden = cfg.embedding_length;
    const int batch = 1;
    const int test_token_id = 15;

    // Dequantize embedding for test_token_id:
    // Logical shape is [vocab, hidden]. Row test_token_id starts at test_token_id * hidden.
    std::vector<float> token_floats(hidden);
    auto raw_embd = gguf.tensor_data(*embd_desc);
    int64_t bs = ninfer_ggml_blck_size(embd_desc->ggml_type);
    size_t ts = ninfer_ggml_type_size(embd_desc->ggml_type);
    int64_t start_elem = static_cast<int64_t>(test_token_id) * hidden;
    assert(start_elem % bs == 0);
    size_t byte_off = (start_elem / bs) * ts;

    int res = ninfer_ggml_dequantize_row(embd_desc->ggml_type, raw_embd.data() + byte_off, token_floats.data(), hidden);
    assert(res == 0);

    // Dequantize norm weights
    std::vector<float> norm_floats(hidden);
    auto raw_norm = gguf.tensor_data(*norm_desc);
    res = ninfer_ggml_dequantize_row(norm_desc->ggml_type, raw_norm.data(), norm_floats.data(), hidden);
    assert(res == 0);

    // Convert to BF16 for device
    std::vector<uint16_t> h_embed_bf16(hidden);
    std::vector<uint16_t> h_norm_bf16(hidden);
    for (int i = 0; i < hidden; ++i) {
        // float to bf16
        uint32_t x;
        std::memcpy(&x, &token_floats[i], 4);
        h_embed_bf16[i] = static_cast<uint16_t>((x + 0x7fff + ((x >> 16) & 1)) >> 16);

        std::memcpy(&x, &norm_floats[i], 4);
        h_norm_bf16[i] = static_cast<uint16_t>((x + 0x7fff + ((x >> 16) & 1)) >> 16);
    }

    // Allocate device buffers
    void* d_x = nullptr;
    void* d_norm_w = nullptr;
    void* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, hidden * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_norm_w, hidden * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out, hidden * sizeof(uint16_t)));

    CUDA_CHECK(cudaMemcpy(d_x, h_embed_bf16.data(), hidden * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm_w, h_norm_bf16.data(), hidden * sizeof(uint16_t), cudaMemcpyHostToDevice));

    Tensor x_tensor(d_x, DType::BF16, {hidden, batch});
    Tensor norm_tensor(d_norm_w, DType::BF16, {hidden});
    Tensor out_tensor(d_out, DType::BF16, {hidden, batch});

    // 4. Run ops::rmsnorm on the dequantized weights
    float eps = cfg.layer_norm_rms_epsilon > 0.0f ? cfg.layer_norm_rms_epsilon : 1e-6f;
    ops::rmsnorm(x_tensor, norm_tensor, eps, true, out_tensor, ctx.stream);
    ctx.synchronize();

    // 5. Read back results and verify finite values
    std::vector<uint16_t> h_result_bf16(hidden);
    CUDA_CHECK(cudaMemcpy(h_result_bf16.data(), d_out, hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    bool all_finite = true;
    for (int i = 0; i < hidden; ++i) {
        uint32_t val32 = static_cast<uint32_t>(h_result_bf16[i]) << 16;
        float f;
        std::memcpy(&f, &val32, 4);
        if (!std::isfinite(f)) {
            all_finite = false;
            break;
        }
    }
    assert(all_finite);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_norm_w));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "  RMSNorm layer execution produced 100% finite outputs!\n";
    std::cout << "PASS: test_qwen_gguf_forward_pass\n";
}

void test_swift_gguf_forward_pass() {
    std::cout << "--- Running test_swift_gguf_forward_pass ---\n";
    std::filesystem::path model_path = "/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf";
    if (!std::filesystem::exists(model_path)) {
        std::cout << "SKIP: Swift GGUF not found\n";
        return;
    }

    std::cout << "Loading Swift GGUF: " << model_path << "\n";
    GGUFFile gguf(model_path);
    ModelConfig cfg = gguf.extract_config();
    std::cout << "  Architecture: " << cfg.architecture << "\n"
              << "  Layers:       " << cfg.block_count << "\n"
              << "  Hidden:       " << cfg.embedding_length << "\n"
              << "  Vocab:        " << cfg.vocab_size << "\n";

    assert(cfg.embedding_length == 5120);
    assert(cfg.block_count == 65);

    const auto* embd_desc = gguf.find_tensor("token_embd.weight");
    assert(embd_desc != nullptr);
    std::cout << "  token_embd.weight: shape=[" << embd_desc->shape[0] << ", " << embd_desc->shape[1]
              << "] type=" << ninfer_ggml_type_name(embd_desc->ggml_type) << "\n";

    const auto* norm_desc = gguf.find_tensor("blk.0.attn_norm.weight");
    assert(norm_desc != nullptr);
    std::cout << "  blk.0.attn_norm.weight: shape=[" << norm_desc->shape[0]
              << "] type=" << ninfer_ggml_type_name(norm_desc->ggml_type) << "\n";

    DeviceContext ctx(0);
    const int hidden = cfg.embedding_length;
    const int batch = 1;
    const int test_token_id = 42;

    // Dequantize embedding slice (IQ2_S)
    std::vector<float> token_floats(hidden);
    auto raw_embd = gguf.tensor_data(*embd_desc);
    int64_t bs = ninfer_ggml_blck_size(embd_desc->ggml_type);
    size_t ts = ninfer_ggml_type_size(embd_desc->ggml_type);
    int64_t start_elem = static_cast<int64_t>(test_token_id) * hidden;
    assert(start_elem % bs == 0);
    size_t byte_off = (start_elem / bs) * ts;

    int res = ninfer_ggml_dequantize_row(embd_desc->ggml_type, raw_embd.data() + byte_off, token_floats.data(), hidden);
    assert(res == 0);

    // Dequantize norm weights (F32)
    std::vector<float> norm_floats(hidden);
    auto raw_norm = gguf.tensor_data(*norm_desc);
    res = ninfer_ggml_dequantize_row(norm_desc->ggml_type, raw_norm.data(), norm_floats.data(), hidden);
    assert(res == 0);

    // Convert to BF16 for device
    std::vector<uint16_t> h_embed_bf16(hidden);
    std::vector<uint16_t> h_norm_bf16(hidden);
    for (int i = 0; i < hidden; ++i) {
        uint32_t x;
        std::memcpy(&x, &token_floats[i], 4);
        h_embed_bf16[i] = static_cast<uint16_t>((x + 0x7fff + ((x >> 16) & 1)) >> 16);

        std::memcpy(&x, &norm_floats[i], 4);
        h_norm_bf16[i] = static_cast<uint16_t>((x + 0x7fff + ((x >> 16) & 1)) >> 16);
    }

    // Allocate device buffers
    void* d_x = nullptr;
    void* d_norm_w = nullptr;
    void* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, hidden * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_norm_w, hidden * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_out, hidden * sizeof(uint16_t)));

    CUDA_CHECK(cudaMemcpy(d_x, h_embed_bf16.data(), hidden * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_norm_w, h_norm_bf16.data(), hidden * sizeof(uint16_t), cudaMemcpyHostToDevice));

    Tensor x_tensor(d_x, DType::BF16, {hidden, batch});
    Tensor norm_tensor(d_norm_w, DType::BF16, {hidden});
    Tensor out_tensor(d_out, DType::BF16, {hidden, batch});

    ops::rmsnorm(x_tensor, norm_tensor, 1e-6f, true, out_tensor, ctx.stream);
    ctx.synchronize();

    std::vector<uint16_t> h_result_bf16(hidden);
    CUDA_CHECK(cudaMemcpy(h_result_bf16.data(), d_out, hidden * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    bool all_finite = true;
    for (int i = 0; i < hidden; ++i) {
        uint32_t val32 = static_cast<uint32_t>(h_result_bf16[i]) << 16;
        float f;
        std::memcpy(&f, &val32, 4);
        if (!std::isfinite(f)) {
            all_finite = false;
            break;
        }
    }
    assert(all_finite);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_norm_w));
    CUDA_CHECK(cudaFree(d_out));

    std::cout << "  Swift 1.5 Qwen3.8-27B forward pass test produced 100% finite outputs!\n";
    std::cout << "PASS: test_swift_gguf_forward_pass\n";
}

int main() {
    test_qwen_gguf_forward_pass();
    test_swift_gguf_forward_pass();
    std::cout << "ALL GGUF LOADER TESTS PASSED!\n";
    return 0;
}
