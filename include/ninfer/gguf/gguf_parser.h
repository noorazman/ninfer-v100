#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ninfer::gguf {

constexpr uint32_t kGgufMagic = 0x46554747; // 'GGUF' ('G'=0x47, 'G'=0x47, 'U'=0x55, 'F'=0x46 in LE)

enum class MetadataValueType : uint32_t {
    UInt8   = 0,
    Int8    = 1,
    UInt16  = 2,
    Int16   = 3,
    UInt32  = 4,
    Int32   = 5,
    Float32 = 6,
    Bool    = 7,
    String  = 8,
    Array   = 9,
    UInt64  = 10,
    Int64   = 11,
    Float64 = 12,
};

struct MetadataValue;

using MetadataArray = std::vector<MetadataValue>;

struct MetadataValue {
    MetadataValueType type;
    std::variant<
        uint8_t, int8_t,
        uint16_t, int16_t,
        uint32_t, int32_t,
        float, bool,
        std::string,
        MetadataArray,
        uint64_t, int64_t,
        double
    > value;

    // Helper accessors
    [[nodiscard]] std::string as_string(std::string_view default_val = "") const;
    [[nodiscard]] int64_t as_int(int64_t default_val = 0) const;
    [[nodiscard]] uint64_t as_uint(uint64_t default_val = 0) const;
    [[nodiscard]] float as_float(float default_val = 0.0f) const;
    [[nodiscard]] bool as_bool(bool default_val = false) const;
};

struct TensorDescriptor {
    std::string name;
    std::vector<uint64_t> shape; // In logical row-major [rows, cols, ...]
    std::vector<uint64_t> raw_dims; // As stored in GGUF [ne0, ne1, ...]
    int ggml_type = 0;
    uint64_t offset = 0; // Offset in shard tensor data
    size_t shard_index = 0;
    size_t size_bytes = 0;
};

struct ModelConfig {
    std::string architecture;
    std::string name;
    uint32_t block_count = 0;
    uint32_t embedding_length = 0; // hidden size
    uint32_t feed_forward_length = 0; // intermediate size
    uint32_t head_count = 0;
    uint32_t head_count_kv = 0;
    uint32_t key_length = 0; // head dim
    uint32_t value_length = 0;
    uint32_t vocab_size = 0;
    float rope_freq_base = 10000.0f;
    float layer_norm_rms_epsilon = 1e-6f;
    uint32_t ssm_time_step_rank = 0;
    uint32_t ssm_inner_size = 0;
    uint32_t full_attention_interval = 0;
    uint32_t nextn_predict_layers = 0; // MTP layers
    bool is_hybrid_ssm = false;
};

class GGUFFile {
public:
    explicit GGUFFile(const std::filesystem::path& path);
    ~GGUFFile();

    GGUFFile(GGUFFile&&) noexcept;
    GGUFFile& operator=(GGUFFile&&) noexcept;
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    [[nodiscard]] uint32_t version() const noexcept;
    [[nodiscard]] uint64_t tensor_count() const noexcept;
    [[nodiscard]] uint64_t kv_count() const noexcept;
    [[nodiscard]] size_t shard_count() const noexcept;

    // Metadata access
    [[nodiscard]] const std::unordered_map<std::string, MetadataValue>& metadata() const noexcept;
    [[nodiscard]] const MetadataValue* find_metadata(std::string_view key) const noexcept;

    // Model configuration extracted from metadata
    [[nodiscard]] ModelConfig extract_config() const;

    // Tensor access
    [[nodiscard]] const std::vector<TensorDescriptor>& tensors() const noexcept;
    [[nodiscard]] const TensorDescriptor* find_tensor(std::string_view name) const noexcept;
    [[nodiscard]] std::span<const std::byte> tensor_data(const TensorDescriptor& desc) const;
    [[nodiscard]] std::span<const std::byte> tensor_data(std::string_view name) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::gguf
