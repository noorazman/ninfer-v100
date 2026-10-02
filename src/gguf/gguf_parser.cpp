#include "ninfer/gguf/gguf_parser.h"
#include "dequant.h"

#include <cassert>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <regex>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ninfer::gguf {
namespace {

constexpr uint64_t kDefaultAlignment = 32;

uint64_t align_up(uint64_t offset, uint64_t alignment) {
    return (offset + alignment - 1) / alignment * alignment;
}

class BinaryReader {
public:
    BinaryReader(const std::byte* data, size_t size) : data_(data), size_(size), pos_(0) {}

    [[nodiscard]] size_t pos() const noexcept { return pos_; }
    [[nodiscard]] size_t remaining() const noexcept { return pos_ < size_ ? size_ - pos_ : 0; }

    void seek(size_t pos) {
        if (pos > size_) throw std::runtime_error("BinaryReader: seek out of bounds");
        pos_ = pos;
    }

    void skip(size_t bytes) {
        if (pos_ + bytes > size_) throw std::runtime_error("BinaryReader: skip out of bounds");
        pos_ += bytes;
    }

    template <typename T>
    T read() {
        if (pos_ + sizeof(T) > size_) throw std::runtime_error("BinaryReader: read out of bounds");
        T val;
        std::memcpy(&val, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        if (pos_ + len > size_) throw std::runtime_error("BinaryReader: string length out of bounds");
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }

    const std::byte* current_ptr() const noexcept { return data_ + pos_; }

private:
    const std::byte* data_;
    size_t size_;
    size_t pos_;
};

MetadataValue read_metadata_value(BinaryReader& reader, MetadataValueType type) {
    MetadataValue mv;
    mv.type = type;
    switch (type) {
    case MetadataValueType::UInt8:
        mv.value = reader.read<uint8_t>();
        break;
    case MetadataValueType::Int8:
        mv.value = reader.read<int8_t>();
        break;
    case MetadataValueType::UInt16:
        mv.value = reader.read<uint16_t>();
        break;
    case MetadataValueType::Int16:
        mv.value = reader.read<int16_t>();
        break;
    case MetadataValueType::UInt32:
        mv.value = reader.read<uint32_t>();
        break;
    case MetadataValueType::Int32:
        mv.value = reader.read<int32_t>();
        break;
    case MetadataValueType::Float32:
        mv.value = reader.read<float>();
        break;
    case MetadataValueType::Bool:
        mv.value = (reader.read<uint8_t>() != 0);
        break;
    case MetadataValueType::String:
        mv.value = reader.read_string();
        break;
    case MetadataValueType::Array: {
        auto elem_type = static_cast<MetadataValueType>(reader.read<uint32_t>());
        uint64_t count = reader.read<uint64_t>();
        MetadataArray arr;
        arr.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            arr.push_back(read_metadata_value(reader, elem_type));
        }
        mv.value = std::move(arr);
        break;
    }
    case MetadataValueType::UInt64:
        mv.value = reader.read<uint64_t>();
        break;
    case MetadataValueType::Int64:
        mv.value = reader.read<int64_t>();
        break;
    case MetadataValueType::Float64:
        mv.value = reader.read<double>();
        break;
    default:
        throw std::runtime_error("Unknown GGUF metadata type: " + std::to_string(static_cast<uint32_t>(type)));
    }
    return mv;
}

} // namespace

std::string MetadataValue::as_string(std::string_view default_val) const {
    if (const auto* s = std::get_if<std::string>(&value)) {
        return *s;
    }
    return std::string(default_val);
}

int64_t MetadataValue::as_int(int64_t default_val) const {
    if (const auto* v = std::get_if<int8_t>(&value)) return *v;
    if (const auto* v = std::get_if<int16_t>(&value)) return *v;
    if (const auto* v = std::get_if<int32_t>(&value)) return *v;
    if (const auto* v = std::get_if<int64_t>(&value)) return *v;
    if (const auto* v = std::get_if<uint8_t>(&value)) return static_cast<int64_t>(*v);
    if (const auto* v = std::get_if<uint16_t>(&value)) return static_cast<int64_t>(*v);
    if (const auto* v = std::get_if<uint32_t>(&value)) return static_cast<int64_t>(*v);
    if (const auto* v = std::get_if<uint64_t>(&value)) return static_cast<int64_t>(*v);
    return default_val;
}

uint64_t MetadataValue::as_uint(uint64_t default_val) const {
    if (const auto* v = std::get_if<uint8_t>(&value)) return *v;
    if (const auto* v = std::get_if<uint16_t>(&value)) return *v;
    if (const auto* v = std::get_if<uint32_t>(&value)) return *v;
    if (const auto* v = std::get_if<uint64_t>(&value)) return *v;
    if (const auto* v = std::get_if<int8_t>(&value)) return *v >= 0 ? static_cast<uint64_t>(*v) : default_val;
    if (const auto* v = std::get_if<int16_t>(&value)) return *v >= 0 ? static_cast<uint64_t>(*v) : default_val;
    if (const auto* v = std::get_if<int32_t>(&value)) return *v >= 0 ? static_cast<uint64_t>(*v) : default_val;
    if (const auto* v = std::get_if<int64_t>(&value)) return *v >= 0 ? static_cast<uint64_t>(*v) : default_val;
    return default_val;
}

float MetadataValue::as_float(float default_val) const {
    if (const auto* v = std::get_if<float>(&value)) return *v;
    if (const auto* v = std::get_if<double>(&value)) return static_cast<float>(*v);
    if (const auto* v = std::get_if<int32_t>(&value)) return static_cast<float>(*v);
    if (const auto* v = std::get_if<uint32_t>(&value)) return static_cast<float>(*v);
    return default_val;
}

bool MetadataValue::as_bool(bool default_val) const {
    if (const auto* v = std::get_if<bool>(&value)) return *v;
    return default_val;
}

struct MmapShard {
    std::filesystem::path path;
    int fd = -1;
    size_t file_size = 0;
    const std::byte* mmap_data = nullptr;
    const std::byte* data_blob = nullptr;
    uint64_t data_blob_offset = 0;

    MmapShard() = default;
    ~MmapShard() { close(); }

    MmapShard(MmapShard&& other) noexcept
        : path(std::move(other.path)), fd(other.fd), file_size(other.file_size),
          mmap_data(other.mmap_data), data_blob(other.data_blob),
          data_blob_offset(other.data_blob_offset) {
        other.fd = -1;
        other.mmap_data = nullptr;
        other.data_blob = nullptr;
    }

    MmapShard& operator=(MmapShard&& other) noexcept {
        if (this != &other) {
            close();
            path = std::move(other.path);
            fd = other.fd;
            file_size = other.file_size;
            mmap_data = other.mmap_data;
            data_blob = other.data_blob;
            data_blob_offset = other.data_blob_offset;
            other.fd = -1;
            other.mmap_data = nullptr;
            other.data_blob = nullptr;
        }
        return *this;
    }

    void open_and_map(const std::filesystem::path& p) {
        close();
        path = p;
        fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open GGUF file: " + p.string() + ": " + std::strerror(errno));
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("Failed to stat GGUF file: " + p.string());
        }
        file_size = static_cast<size_t>(st.st_size);
        if (file_size < 24) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("GGUF file too small: " + p.string());
        }
        void* addr = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("Failed to mmap GGUF file: " + p.string() + ": " + std::strerror(errno));
        }
        mmap_data = static_cast<const std::byte*>(addr);
    }

    void close() noexcept {
        if (mmap_data != nullptr) {
            ::munmap(const_cast<void*>(static_cast<const void*>(mmap_data)), file_size);
            mmap_data = nullptr;
            data_blob = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }
};

struct GGUFFile::Impl {
    uint32_t version = 0;
    uint64_t total_tensor_count = 0;
    uint64_t kv_count = 0;
    uint64_t alignment = kDefaultAlignment;

    std::vector<MmapShard> shards;
    std::unordered_map<std::string, MetadataValue> metadata;
    std::vector<TensorDescriptor> tensors;
    std::unordered_map<std::string, size_t> tensor_index;

    void parse_single_or_split(const std::filesystem::path& initial_path);
    void parse_shard_tensors(size_t shard_idx, BinaryReader& reader, uint64_t shard_tensor_count);
};

void GGUFFile::Impl::parse_single_or_split(const std::filesystem::path& initial_path) {
    // 1. Open primary shard
    MmapShard primary;
    primary.open_and_map(initial_path);

    BinaryReader reader(primary.mmap_data, primary.file_size);
    uint32_t magic = reader.read<uint32_t>();
    if (magic != kGgufMagic && magic != 0x47475546) {
        throw std::runtime_error("Invalid GGUF magic in " + initial_path.string() + " (got " + std::to_string(magic) + ")");
    }
    version = reader.read<uint32_t>();
    if (version < 2 || version > 3) {
        throw std::runtime_error("Unsupported GGUF version " + std::to_string(version) + " in " + initial_path.string());
    }

    uint64_t primary_tensor_count = reader.read<uint64_t>();
    kv_count = reader.read<uint64_t>();

    // 2. Read metadata KV pairs
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = reader.read_string();
        auto val_type = static_cast<MetadataValueType>(reader.read<uint32_t>());
        metadata[key] = read_metadata_value(reader, val_type);
    }

    if (auto it = metadata.find("general.alignment"); it != metadata.end()) {
        alignment = it->second.as_uint(kDefaultAlignment);
    }

    // Check for split files
    uint32_t split_count = 1;
    uint32_t split_no = 0;
    if (auto it = metadata.find("split.count"); it != metadata.end()) {
        split_count = static_cast<uint32_t>(it->second.as_uint(1));
    }
    if (auto it = metadata.find("split.no"); it != metadata.end()) {
        split_no = static_cast<uint32_t>(it->second.as_uint(0));
    }

    shards.push_back(std::move(primary));

    if (split_count > 1) {
        // Resolve all shard filenames
        // Pattern: ...-00001-of-00002.gguf or ...-00000-of-00002.gguf
        std::string filename = initial_path.filename().string();
        std::regex split_regex(R"((.*)-(\d+)-of-(\d+)\.gguf)");
        std::smatch match;

        if (std::regex_match(filename, match, split_regex)) {
            std::string stem = match[1].str();
            int width = static_cast<int>(match[2].length());
            bool zero_indexed = (split_no == 0 && match[2].str() == std::string(width, '0'));

            // The primary is shard split_no. We need to load all other shards.
            std::vector<std::filesystem::path> shard_paths(split_count);
            for (uint32_t s = 0; s < split_count; ++s) {
                uint32_t file_idx = zero_indexed ? s : (s + 1);
                char buf[64];
                std::snprintf(buf, sizeof(buf), "-%0*u-of-%0*u.gguf", width, file_idx, width, split_count);
                shard_paths[s] = initial_path.parent_path() / (stem + buf);
            }

            // Replace shards[0] with shard 0 if primary was not shard 0
            shards.clear();
            shards.resize(split_count);
            for (size_t s = 0; s < split_count; ++s) {
                shards[s].open_and_map(shard_paths[s]);
            }
        } else {
            // Unsplit format or name doesn't match standard regex, keep single primary
            split_count = 1;
        }
    }

    // Now parse tensor descriptors from all shards
    total_tensor_count = 0;
    for (size_t s = 0; s < shards.size(); ++s) {
        BinaryReader shard_reader(shards[s].mmap_data, shards[s].file_size);
        shard_reader.read<uint32_t>(); // magic
        shard_reader.read<uint32_t>(); // ver
        uint64_t s_tensors = shard_reader.read<uint64_t>();
        uint64_t s_kv = shard_reader.read<uint64_t>();

        // Skip metadata in secondary shards
        for (uint64_t i = 0; i < s_kv; ++i) {
            uint64_t klen = shard_reader.read<uint64_t>();
            shard_reader.skip(klen);
            auto val_type = static_cast<MetadataValueType>(shard_reader.read<uint32_t>());
            // discard value
            (void)read_metadata_value(shard_reader, val_type);
        }

        parse_shard_tensors(s, shard_reader, s_tensors);
    }
}

void GGUFFile::Impl::parse_shard_tensors(size_t shard_idx, BinaryReader& reader, uint64_t shard_tensor_count) {
    size_t start_tensor_idx = tensors.size();
    tensors.reserve(start_tensor_idx + shard_tensor_count);

    for (uint64_t i = 0; i < shard_tensor_count; ++i) {
        TensorDescriptor desc;
        desc.name = reader.read_string();
        uint32_t ndims = reader.read<uint32_t>();
        desc.raw_dims.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d) {
            desc.raw_dims[d] = reader.read<uint64_t>();
        }
        // GGUF dims are column-major (ne0, ne1, ...).
        // Reverse for row-major shape: [rows, cols]
        desc.shape.resize(ndims);
        for (uint32_t d = 0; d < ndims; ++d) {
            desc.shape[d] = desc.raw_dims[ndims - 1 - d];
        }

        desc.ggml_type = static_cast<int>(reader.read<uint32_t>());
        desc.offset = reader.read<uint64_t>();
        desc.shard_index = shard_idx;

        int64_t ne = 1;
        for (uint64_t d : desc.raw_dims) {
            ne *= static_cast<int64_t>(d);
        }
        desc.size_bytes = ninfer_ggml_row_size(desc.ggml_type, ne);

        tensors.push_back(std::move(desc));
    }

    // The data blob begins aligned to `alignment` after tensor descriptors in this shard
    uint64_t header_end_offset = reader.pos();
    uint64_t data_blob_offset = align_up(header_end_offset, alignment);
    shards[shard_idx].data_blob_offset = data_blob_offset;
    shards[shard_idx].data_blob = shards[shard_idx].mmap_data + data_blob_offset;

    // Index tensor names
    for (size_t i = start_tensor_idx; i < tensors.size(); ++i) {
        tensor_index[tensors[i].name] = i;
    }
    total_tensor_count = tensors.size();
}

GGUFFile::GGUFFile(const std::filesystem::path& path) : impl_(std::make_unique<Impl>()) {
    impl_->parse_single_or_split(path);
}

GGUFFile::~GGUFFile() = default;
GGUFFile::GGUFFile(GGUFFile&&) noexcept = default;
GGUFFile& GGUFFile::operator=(GGUFFile&&) noexcept = default;

uint32_t GGUFFile::version() const noexcept { return impl_->version; }
uint64_t GGUFFile::tensor_count() const noexcept { return impl_->total_tensor_count; }
uint64_t GGUFFile::kv_count() const noexcept { return impl_->kv_count; }
size_t GGUFFile::shard_count() const noexcept { return impl_->shards.size(); }

const std::unordered_map<std::string, MetadataValue>& GGUFFile::metadata() const noexcept {
    return impl_->metadata;
}

const MetadataValue* GGUFFile::find_metadata(std::string_view key) const noexcept {
    auto it = impl_->metadata.find(std::string(key));
    if (it != impl_->metadata.end()) {
        return &it->second;
    }
    return nullptr;
}

ModelConfig GGUFFile::extract_config() const {
    ModelConfig cfg;
    if (const auto* arch = find_metadata("general.architecture")) {
        cfg.architecture = arch->as_string();
    }
    if (const auto* name = find_metadata("general.name")) {
        cfg.name = name->as_string();
    }

    std::string prefix = cfg.architecture.empty() ? "qwen35" : cfg.architecture;

    auto get_u32 = [&](std::string_view suffix, uint32_t def = 0) -> uint32_t {
        if (const auto* val = find_metadata(prefix + "." + std::string(suffix))) {
            return static_cast<uint32_t>(val->as_uint(def));
        }
        return def;
    };

    auto get_f32 = [&](std::string_view suffix, float def = 0.0f) -> float {
        if (const auto* val = find_metadata(prefix + "." + std::string(suffix))) {
            return val->as_float(def);
        }
        return def;
    };

    cfg.block_count = get_u32("block_count", 0);
    cfg.embedding_length = get_u32("embedding_length", 0);
    cfg.feed_forward_length = get_u32("feed_forward_length", 0);
    cfg.head_count = get_u32("attention.head_count", 0);
    cfg.head_count_kv = get_u32("attention.head_count_kv", cfg.head_count);
    cfg.key_length = get_u32("attention.key_length", 0);
    cfg.value_length = get_u32("attention.value_length", cfg.key_length);
    cfg.rope_freq_base = get_f32("rope.freq_base", 10000.0f);
    cfg.layer_norm_rms_epsilon = get_f32("attention.layer_norm_rms_epsilon", 1e-6f);

    // SSM/GDN specifics
    cfg.ssm_time_step_rank = get_u32("ssm.time_step_rank", 0);
    cfg.ssm_inner_size = get_u32("ssm.inner_size", 0);
    cfg.full_attention_interval = get_u32("full_attention_interval", 0);
    cfg.nextn_predict_layers = get_u32("nextn_predict_layers", 0);
    cfg.is_hybrid_ssm = (cfg.ssm_inner_size > 0 || cfg.ssm_time_step_rank > 0);

    // Vocab size
    if (const auto* tokens = find_metadata("tokenizer.ggml.tokens")) {
        if (const auto* arr = std::get_if<MetadataArray>(&tokens->value)) {
            cfg.vocab_size = static_cast<uint32_t>(arr->size());
        }
    }
    if (cfg.vocab_size == 0) {
        cfg.vocab_size = get_u32("vocab_size", 0);
    }

    return cfg;
}

const std::vector<TensorDescriptor>& GGUFFile::tensors() const noexcept {
    return impl_->tensors;
}

const TensorDescriptor* GGUFFile::find_tensor(std::string_view name) const noexcept {
    auto it = impl_->tensor_index.find(std::string(name));
    if (it != impl_->tensor_index.end()) {
        return &impl_->tensors[it->second];
    }
    return nullptr;
}

std::span<const std::byte> GGUFFile::tensor_data(const TensorDescriptor& desc) const {
    if (desc.shard_index >= impl_->shards.size()) {
        throw std::runtime_error("Invalid shard index in tensor descriptor: " + desc.name);
    }
    const auto& shard = impl_->shards[desc.shard_index];
    if (shard.data_blob == nullptr) {
        throw std::runtime_error("Shard data blob is not mapped for tensor: " + desc.name);
    }
    size_t total_offset = shard.data_blob_offset + desc.offset;
    if (total_offset + desc.size_bytes > shard.file_size) {
        throw std::runtime_error("Tensor data out of bounds for tensor: " + desc.name);
    }
    return std::span<const std::byte>(shard.data_blob + desc.offset, desc.size_bytes);
}

std::span<const std::byte> GGUFFile::tensor_data(std::string_view name) const {
    const auto* desc = find_tensor(name);
    if (!desc) {
        throw std::runtime_error("Tensor not found: " + std::string(name));
    }
    return tensor_data(*desc);
}

} // namespace ninfer::gguf
