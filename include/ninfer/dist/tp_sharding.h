#pragma once

#include "core/tensor.h"
#include <vector>

namespace ninfer::dist {

// Sharding strategies for 2D weight matrices [N, K]
enum class ShardStrategy {
    Row,    // Shard along N (output dimension)
    Column, // Shard along K (input dimension)
    Replicate // Replicate across all ranks (e.g. LayerNorms, biases)
};

struct WeightShard {
    int rank = 0;
    int world_size = 1;
    ShardStrategy strategy = ShardStrategy::Replicate;
    std::vector<int64_t> sharded_shape; // [N_shard, K] or [N, K_shard]
};

// Computes sharded dimensions for rank in [0..world_size-1]
inline WeightShard compute_weight_shard(int64_t n, int64_t k, ShardStrategy strat, int rank, int world_size) {
    WeightShard ws;
    ws.rank = rank;
    ws.world_size = world_size;
    ws.strategy = strat;
    if (strat == ShardStrategy::Row) {
        int64_t n_shard = (n + world_size - 1) / world_size;
        ws.sharded_shape = {n_shard, k};
    } else if (strat == ShardStrategy::Column) {
        int64_t k_shard = (k + world_size - 1) / world_size;
        ws.sharded_shape = {n, k_shard};
    } else {
        ws.sharded_shape = {n, k};
    }
    return ws;
}

} // namespace ninfer::dist
