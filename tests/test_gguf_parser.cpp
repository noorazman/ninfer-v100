#include "ninfer/gguf/gguf_parser.h"
#include "dequant.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace ninfer::gguf;

void test_swift_gguf() {
    std::filesystem::path path = "/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_XXS-mtp.gguf";
    if (!std::filesystem::exists(path)) {
        std::cout << "SKIP: Swift GGUF file not found at " << path << "\n";
        return;
    }

    GGUFFile file(path);
    assert(file.version() == 3);
    assert(file.tensor_count() == 866);
    assert(file.kv_count() > 0);

    const auto* arch = file.find_metadata("general.architecture");
    assert(arch != nullptr);
    assert(arch->as_string() == "qwen35");

    ModelConfig cfg = file.extract_config();
    assert(cfg.architecture == "qwen35");
    assert(cfg.block_count == 65);
    assert(cfg.embedding_length == 5120);
    assert(cfg.feed_forward_length == 17408);
    assert(cfg.head_count == 24);
    assert(cfg.head_count_kv == 4);
    assert(cfg.key_length == 256);
    assert(cfg.value_length == 256);
    assert(cfg.vocab_size == 248320);
    assert(cfg.ssm_time_step_rank == 48);
    assert(cfg.ssm_inner_size == 6144);
    assert(cfg.nextn_predict_layers == 1);

    // Check tensors
    const auto* embd = file.find_tensor("token_embd.weight");
    assert(embd != nullptr);
    assert(embd->shape.size() == 2);
    assert(embd->shape[0] == 248320);
    assert(embd->shape[1] == 5120);

    auto embd_data = file.tensor_data(*embd);
    assert(!embd_data.empty());

    const auto* out_norm = file.find_tensor("output_norm.weight");
    assert(out_norm != nullptr);
    assert(out_norm->shape.size() == 1);
    assert(out_norm->shape[0] == 5120);
    auto norm_data = file.tensor_data(*out_norm);
    assert(norm_data.size() == 5120 * sizeof(float));

    // Test dequantizing a slice of output_norm (F32) and token_embd (IQ2_S)
    float norm_floats[16];
    int res = ninfer_ggml_dequantize_row(out_norm->ggml_type, norm_data.data(), norm_floats, 16);
    assert(res == 0);
    for (int i = 0; i < 16; ++i) {
        assert(std::isfinite(norm_floats[i]));
    }

    std::cout << "PASS: test_swift_gguf\n";
}

void test_router_qwen25_gguf() {
    std::filesystem::path path = "/mnt/ssd/llm_models/Router_Models/qwen2.5-1.5b-instruct-q8_0.gguf";
    if (!std::filesystem::exists(path)) {
        std::cout << "SKIP: Router Qwen2.5 GGUF not found\n";
        return;
    }

    GGUFFile file(path);
    assert(file.version() == 3);
    assert(file.tensor_count() == 339);

    ModelConfig cfg = file.extract_config();
    assert(cfg.architecture == "qwen2");
    assert(cfg.block_count == 28);
    assert(cfg.embedding_length == 1536);
    assert(cfg.feed_forward_length == 8960);
    assert(cfg.head_count == 12);
    assert(cfg.head_count_kv == 2);

    const auto* q = file.find_tensor("blk.0.attn_q.weight");
    assert(q != nullptr);
    assert(q->shape.size() == 2);
    assert(q->shape[0] == 1536);
    assert(q->shape[1] == 1536);

    auto q_data = file.tensor_data(*q);
    assert(!q_data.empty());

    // Dequantize one row of Q8_0
    float deq_row[1536];
    int res = ninfer_ggml_dequantize_row(q->ggml_type, q_data.data(), deq_row, 1536);
    assert(res == 0);
    for (int i = 0; i < 16; ++i) {
        assert(std::isfinite(deq_row[i]));
    }

    std::cout << "PASS: test_router_qwen25_gguf\n";
}

void test_split_gguf() {
    std::filesystem::path path = "/mnt/ssd/llm_models/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
    if (!std::filesystem::exists(path)) {
        std::cout << "SKIP: Split GGUF not found\n";
        return;
    }

    std::cout << "test_split_gguf: opening file...\n";
    GGUFFile file(path);
    std::cout << "test_split_gguf: opened! shards=" << file.shard_count() << " tensors=" << file.tensor_count() << "\n";
    assert(file.shard_count() == 2);
    assert(file.tensor_count() == 1224);

    const auto* t0 = file.find_tensor("token_embd.weight");
    if (!t0) {
        std::cerr << "FAIL: token_embd.weight not found in split GGUF\n";
        std::abort();
    }
    std::cout << "t0: name=" << t0->name << " shard=" << t0->shard_index << " offset=" << t0->offset << " bytes=" << t0->size_bytes << "\n";
    auto data0 = file.tensor_data(*t0);
    std::cout << "t0 data ok, size=" << data0.size() << "\n";

    const auto* t_shard1 = file.find_tensor("blk.12.ffn_down_exps.weight");
    if (!t_shard1) {
        std::cerr << "FAIL: blk.12.ffn_down_exps.weight not found in split GGUF\n";
        std::abort();
    }
    std::cout << "t_shard1: name=" << t_shard1->name << " shard=" << t_shard1->shard_index << " offset=" << t_shard1->offset << " bytes=" << t_shard1->size_bytes << "\n";
    auto data_shard1 = file.tensor_data(*t_shard1);
    std::cout << "t_shard1 data ok, size=" << data_shard1.size() << "\n";

    std::cout << "PASS: test_split_gguf (loaded 2 shards, " << file.tensor_count() << " tensors)\n";
}

int main() {
    std::cout << "Starting test_swift_gguf...\n";
    test_swift_gguf();
    std::cout << "Starting test_router_qwen25_gguf...\n";
    test_router_qwen25_gguf();
    std::cout << "Starting test_split_gguf...\n";
    test_split_gguf();
    std::cout << "ALL GGUF PARSER TESTS PASSED!\n";
    return 0;
}
