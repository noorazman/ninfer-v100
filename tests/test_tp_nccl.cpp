#include "ninfer/dist/tp_context.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <thread>
#include <vector>

using namespace ninfer::dist;

int main() {
    std::cout << "Testing NCCL TP=2 with multi-threading...\n";
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        std::cout << "SKIP: Need at least 2 CUDA devices for TP=2 test (found " << count << ")\n";
        return 0;
    }

    std::vector<int> devices;
    if (count >= 5) {
        devices = {3, 4};
    } else {
        devices = {0, 1};
    }

    auto group = create_node_tp_group(devices);
    assert(group.size() == 2);
    assert(group[0]->rank() == 0);
    assert(group[1]->rank() == 1);
    assert(group[0]->device_id() == 3);
    assert(group[1]->device_id() == 4);

    const size_t N = 5120;
    std::vector<float> h_recv0(N, 0.0f);
    std::vector<float> h_recv1(N, 0.0f);

    auto run_rank = [&](int rank, float send_val, std::vector<float>& h_recv) {
        int dev = group[rank]->device_id();
        CUDA_CHECK(cudaSetDevice(dev));
        cudaStream_t s = nullptr;
        CUDA_CHECK(cudaStreamCreate(&s));

        void* d_buf = nullptr;
        CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));

        std::vector<float> h_send(N, send_val);
        CUDA_CHECK(cudaMemcpyAsync(d_buf, h_send.data(), N * sizeof(float), cudaMemcpyHostToDevice, s));

        group[rank]->all_reduce_sum_fp32(d_buf, N, s);

        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaMemcpy(h_recv.data(), d_buf, N * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_buf));
        CUDA_CHECK(cudaStreamDestroy(s));
    };

    std::thread t0(run_rank, 0, 1.0f, std::ref(h_recv0));
    std::thread t1(run_rank, 1, 2.0f, std::ref(h_recv1));

    t0.join();
    t1.join();

    for (size_t i = 0; i < N; ++i) {
        assert(std::fabs(h_recv0[i] - 3.0f) < 1e-4f);
        assert(std::fabs(h_recv1[i] - 3.0f) < 1e-4f);
    }

    std::cout << "PASS: NCCL TP=2 AllReduce between GPU 3 and GPU 4 verified (result=3.0 across all 5120 elements)!\n";
    return 0;
}
