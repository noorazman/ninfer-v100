#include "ninfer/dist/tp_context.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ninfer::dist {
namespace {

void check_nccl(ncclResult_t res, const char* msg) {
    if (res != ncclSuccess) {
        throw std::runtime_error(std::string("NCCL Error [") + msg + "]: " + ncclGetErrorString(res));
    }
}

} // namespace

TPContext::TPContext(const TPConfig& config) : config_(config) {
    if (!config_.is_distributed()) {
        comm_ = nullptr;
    }
}

TPContext::~TPContext() {
    if (comm_ != nullptr) {
        ncclCommDestroy(comm_);
        comm_ = nullptr;
    }
}

TPContext::TPContext(TPContext&& other) noexcept
    : config_(std::move(other.config_)), comm_(other.comm_) {
    other.comm_ = nullptr;
}

TPContext& TPContext::operator=(TPContext&& other) noexcept {
    if (this != &other) {
        if (comm_ != nullptr) {
            ncclCommDestroy(comm_);
        }
        config_ = std::move(other.config_);
        comm_   = other.comm_;
        other.comm_ = nullptr;
    }
    return *this;
}

void TPContext::all_reduce_sum_bf16(void* sendrecv_buf, size_t count, cudaStream_t stream) const {
    if (!config_.is_distributed() || comm_ == nullptr) return;
    check_nccl(ncclAllReduce(sendrecv_buf, sendrecv_buf, count, ncclBfloat16, ncclSum, comm_, stream),
               "all_reduce_sum_bf16");
}

void TPContext::all_reduce_sum_fp32(void* sendrecv_buf, size_t count, cudaStream_t stream) const {
    if (!config_.is_distributed() || comm_ == nullptr) return;
    check_nccl(ncclAllReduce(sendrecv_buf, sendrecv_buf, count, ncclFloat32, ncclSum, comm_, stream),
               "all_reduce_sum_fp32");
}

void TPContext::all_gather(const void* sendbuf, void* recvbuf, size_t count_bytes, cudaStream_t stream) const {
    if (!config_.is_distributed() || comm_ == nullptr) {
        if (sendbuf != recvbuf) {
            CUDA_CHECK(cudaMemcpyAsync(recvbuf, sendbuf, count_bytes, cudaMemcpyDeviceToDevice, stream));
        }
        return;
    }
    check_nccl(ncclAllGather(sendbuf, recvbuf, count_bytes, ncclChar, comm_, stream), "all_gather");
}

void TPContext::broadcast(void* buf, size_t count_bytes, int root, cudaStream_t stream) const {
    if (!config_.is_distributed() || comm_ == nullptr) return;
    check_nccl(ncclBroadcast(buf, buf, count_bytes, ncclChar, root, comm_, stream), "broadcast");
}

void TPContext::barrier(cudaStream_t stream) const {
    if (!config_.is_distributed() || comm_ == nullptr) return;
    // Single-byte all-reduce as lightweight barrier
    char dummy = 0;
    check_nccl(ncclAllReduce(&dummy, &dummy, 1, ncclChar, ncclSum, comm_, stream), "barrier");
}

std::vector<std::unique_ptr<TPContext>> create_node_tp_group(const std::vector<int>& device_ids) {
    const int tp_size = static_cast<int>(device_ids.size());
    std::vector<std::unique_ptr<TPContext>> group;
    group.reserve(tp_size);

    if (tp_size <= 1) {
        TPConfig cfg;
        cfg.tp_size = 1;
        cfg.rank = 0;
        cfg.device_ids = device_ids.empty() ? std::vector<int>{0} : device_ids;
        group.push_back(std::make_unique<TPContext>(cfg));
        return group;
    }

    std::vector<ncclComm_t> comms(tp_size);
    check_nccl(ncclCommInitAll(comms.data(), tp_size, device_ids.data()), "ncclCommInitAll");

    for (int r = 0; r < tp_size; ++r) {
        TPConfig cfg;
        cfg.tp_size = tp_size;
        cfg.rank = r;
        cfg.device_ids = device_ids;
        auto ctx = std::make_unique<TPContext>(cfg);
        ctx->comm_ = comms[r];
        group.push_back(std::move(ctx));
    }
    return group;
}

} // namespace ninfer::dist
