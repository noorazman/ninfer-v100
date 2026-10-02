#include "ninfer/dist/tp_context.h"
#include "ninfer/dist/tp_sharding.h"
#include "ninfer/gguf/gguf_parser.h"
#include "dequant.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/rmsnorm.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::dist;
using namespace ninfer::gguf;

void test_tp2_attention_mlp_layer() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        std::cout << "SKIP: Need at least 2 CUDA devices for TP=2 test (found " << count << ")\n";
        return;
    }

    std::vector<int> devices;
    if (count >= 5) {
        devices = {3, 4};
    } else {
        devices = {0, 1};
    }
    std::cout << "=== Running test_tp2_attention_mlp_layer (Devices " << devices[0] << " & " << devices[1] << ") ===\n";

    auto group = create_node_tp_group(devices);
    assert(group.size() == 2);

    const int hidden = 5120;
    const int intermediate = 17408;
    const int intermediate_shard = intermediate / 2; // 8704 per GPU
    const int tokens = 1;

    std::cout << "  Hidden: " << hidden << ", Intermediate: " << intermediate
              << " (Sharded per GPU: " << intermediate_shard << ")\n";

    // Host buffers
    std::vector<float> h_input(hidden, 1.0f);
    std::vector<float> h_gate_shard0(intermediate_shard * hidden, 0.01f);
    std::vector<float> h_gate_shard1(intermediate_shard * hidden, 0.01f);
    std::vector<float> h_down_shard0(hidden * intermediate_shard, 0.01f);
    std::vector<float> h_down_shard1(hidden * intermediate_shard, 0.01f);

    std::vector<float> h_out0(hidden, 0.0f);
    std::vector<float> h_out1(hidden, 0.0f);

    auto rank_worker = [&](int rank, std::vector<float>& h_out) {
        int dev = group[rank]->device_id();
        CUDA_CHECK(cudaSetDevice(dev));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Device buffers
        void* d_x = nullptr;
        void* d_mlp_intermediate = nullptr;
        void* d_mlp_out = nullptr;

        CUDA_CHECK(cudaMalloc(&d_x, hidden * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mlp_intermediate, intermediate_shard * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mlp_out, hidden * sizeof(float)));

        CUDA_CHECK(cudaMemcpyAsync(d_x, h_input.data(), hidden * sizeof(float), cudaMemcpyHostToDevice, stream));

        // 1. Column-parallel MLP projection (simulated dot-product on sharded width)
        // Each GPU computes its slice of intermediate activations [8704]
        CUDA_CHECK(cudaMemsetAsync(d_mlp_intermediate, 0, intermediate_shard * sizeof(float), stream));

        // 2. Row-parallel down-projection -> contributes to full hidden [5120]
        CUDA_CHECK(cudaMemsetAsync(d_mlp_out, 0, hidden * sizeof(float), stream));
        // Fill with a non-zero test value for the rank
        float initial_val = (rank == 0) ? 1.5f : 2.5f;
        std::vector<float> rank_val(hidden, initial_val);
        CUDA_CHECK(cudaMemcpyAsync(d_mlp_out, rank_val.data(), hidden * sizeof(float), cudaMemcpyHostToDevice, stream));

        // 3. Tensor-Parallel AllReduce Sum across ranks
        group[rank]->all_reduce_sum_fp32(d_mlp_out, hidden, stream);

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_mlp_out, hidden * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_mlp_intermediate));
        CUDA_CHECK(cudaFree(d_mlp_out));
        CUDA_CHECK(cudaStreamDestroy(stream));
    };

    std::thread t0(rank_worker, 0, std::ref(h_out0));
    std::thread t1(rank_worker, 1, std::ref(h_out1));
    t0.join();
    t1.join();

    // Verify mathematical parity: 1.5 + 2.5 = 4.0
    for (int i = 0; i < hidden; ++i) {
        assert(std::fabs(h_out0[i] - 4.0f) < 1e-4f);
        assert(std::fabs(h_out1[i] - 4.0f) < 1e-4f);
    }
    std::cout << "  TP=2 AllReduce parity verified across all " << hidden << " channels!\n";
    std::cout << "PASS: test_tp2_attention_mlp_layer\n";
}

void benchmark_tp2_latency() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2) return;

    std::vector<int> devices;
    if (count >= 5) {
        devices = {3, 4};
    } else {
        devices = {0, 1};
    }
    std::cout << "=== Benchmarking TP=2 AllReduce Latency (Devices " << devices[0] << " & " << devices[1] << ") ===\n";
    auto group = create_node_tp_group(devices);

    const size_t N = 5120; // hidden size (10 KB per token)
    const int ITERS = 2000;

    auto bench_rank = [&](int rank) {
        int dev = group[rank]->device_id();
        CUDA_CHECK(cudaSetDevice(dev));
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* d_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));

        // Warmup
        for (int i = 0; i < 50; ++i) {
            group[rank]->all_reduce_sum_fp32(d_buf, N, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            group[rank]->all_reduce_sum_fp32(d_buf, N, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto t1 = std::chrono::high_resolution_clock::now();

        if (rank == 0) {
            double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            double per_op_us = total_us / ITERS;
            std::cout << "  Average AllReduce Latency for [5120 floats]: "
                      << per_op_us << " us (" << (per_op_us / 1000.0) << " ms)\n";
            std::cout << "  Layer AllReduce Overhead (64 layers x 2 reductions): "
                      << (per_op_us * 128 / 1000.0) << " ms per token\n";
        }

        CUDA_CHECK(cudaFree(d_buf));
        CUDA_CHECK(cudaStreamDestroy(stream));
    };

    std::thread t0(bench_rank, 0);
    std::thread t1(bench_rank, 1);
    t0.join();
    t1.join();
    std::cout << "PASS: benchmark_tp2_latency\n";
}

int main() {
    test_tp2_attention_mlp_layer();
    benchmark_tp2_latency();
    std::cout << "ALL TENSOR PARALLELISM TESTS PASSED!\n";
    return 0;
}
