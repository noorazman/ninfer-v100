#pragma once

#include "core/device.h"
#include <nccl.h>

#include <memory>
#include <span>
#include <vector>

namespace ninfer::dist {

struct TPConfig {
    int tp_size = 1;
    int rank = 0;
    std::vector<int> device_ids = {0};

    [[nodiscard]] bool is_distributed() const noexcept { return tp_size > 1; }
};

class TPContext {
public:
    TPContext(const TPConfig& config);
    ~TPContext();

    TPContext(const TPContext&) = delete;
    TPContext& operator=(const TPContext&) = delete;
    TPContext(TPContext&&) noexcept;
    TPContext& operator=(TPContext&&) noexcept;

    [[nodiscard]] int rank() const noexcept { return config_.rank; }
    [[nodiscard]] int size() const noexcept { return config_.tp_size; }
    [[nodiscard]] int device_id() const noexcept { return config_.device_ids[config_.rank]; }
    [[nodiscard]] ncclComm_t comm() const noexcept { return comm_; }

    void all_reduce_sum_bf16(void* sendrecv_buf, size_t count, cudaStream_t stream) const;
    void all_reduce_sum_fp32(void* sendrecv_buf, size_t count, cudaStream_t stream) const;
    void all_gather(const void* sendbuf, void* recvbuf, size_t count_bytes, cudaStream_t stream) const;
    void broadcast(void* buf, size_t count_bytes, int root, cudaStream_t stream) const;
    void barrier(cudaStream_t stream) const;

    friend std::vector<std::unique_ptr<TPContext>> create_node_tp_group(const std::vector<int>& device_ids);

private:
    TPConfig config_;
    ncclComm_t comm_ = nullptr;
};

// Creates a group of TPContexts initialized together for a multi-GPU group on the same node
std::vector<std::unique_ptr<TPContext>> create_node_tp_group(const std::vector<int>& device_ids);

} // namespace ninfer::dist
