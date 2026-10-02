#include "converter.h"
#include "dequant.h"
#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <thread>
#include <vector>

namespace ninfer::gguf {
namespace {

using Json = nlohmann::json;

constexpr uint64_t kPlaneAlignment = 256;
constexpr uint64_t kKAlignment = 128;
constexpr uint64_t kPayloadAlignment = 4096;

uint64_t align_up(uint64_t val, uint64_t align) {
    return (val + align - 1) / align * align;
}

uint16_t float_to_fp16(float val) {
    uint32_t x;
    std::memcpy(&x, &val, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant = (mant | 0x800000) >> (1 - exp);
        return static_cast<uint16_t>(sign | ((mant + 0x1000) >> 13));
    } else if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    return static_cast<uint16_t>(sign | (exp << 10) | ((mant + 0x1000) >> 13));
}

float fp16_to_float(uint16_t val) {
    uint32_t sign = (val & 0x8000) << 16;
    uint32_t exp = (val & 0x7c00) >> 10;
    uint32_t mant = (val & 0x03ff) << 13;
    if (exp == 0) {
        if (mant == 0) {
            float f;
            std::memcpy(&f, &sign, 4);
            return f;
        }
        while (!(mant & 0x00800000)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= 0x007fffff;
    } else if (exp == 31) {
        uint32_t res = sign | 0x7f800000 | mant;
        float f;
        std::memcpy(&f, &res, 4);
        return f;
    }
    uint32_t res = sign | ((exp + 127 - 15) << 23) | mant;
    float f;
    std::memcpy(&f, &res, 4);
    return f;
}

uint16_t float_to_bf16(float val) {
    uint32_t x;
    std::memcpy(&x, &val, 4);
    uint32_t lsb = (x >> 16) & 1;
    uint32_t bias = 0x7fff + lsb;
    x += bias;
    return static_cast<uint16_t>(x >> 16);
}

struct RowSplitSpec {
    int bits;
    int group_size;
    int qmin;
    int qmax;
};

RowSplitSpec get_row_split_spec(std::string_view format) {
    if (format == "Q4G64_F16S") return {4, 64, -8, 7};
    if (format == "Q5G64_F16S") return {5, 64, -16, 15};
    if (format == "Q6G64_F16S") return {6, 64, -32, 31};
    if (format == "W8G32_F16S") return {8, 32, -127, 127};
    throw std::runtime_error("Unsupported row-split format: " + std::string(format));
}

struct PlannedGeometry {
    uint64_t n;
    uint64_t k;
    uint64_t k_pad;
    uint64_t groups_per_row;
    uint64_t low_bytes_per_group;
    uint64_t high_bytes_per_group;
    uint64_t low_plane_bytes;
    uint64_t high_plane_offset;
    uint64_t high_plane_bytes;
    uint64_t scale_plane_offset;
    uint64_t scale_plane_bytes;
    uint64_t total_bytes;
};

PlannedGeometry compute_geometry(std::string_view format, uint64_t n, uint64_t k) {
    auto spec = get_row_split_spec(format);
    PlannedGeometry g{};
    g.n = n;
    g.k = k;
    g.k_pad = align_up(k, kKAlignment);
    g.groups_per_row = g.k_pad / spec.group_size;
    g.low_bytes_per_group = 32;
    g.high_bytes_per_group = (spec.bits == 5) ? 8 : (spec.bits == 6 ? 16 : 0);
    uint64_t total_groups = g.n * g.groups_per_row;
    g.low_plane_bytes = total_groups * g.low_bytes_per_group;
    g.high_plane_offset = align_up(g.low_plane_bytes, kPlaneAlignment);
    g.high_plane_bytes = total_groups * g.high_bytes_per_group;
    uint64_t aligned_high = align_up(g.high_plane_bytes, kPlaneAlignment);
    g.scale_plane_offset = g.high_plane_offset + aligned_high;
    g.scale_plane_bytes = total_groups * 2;
    g.total_bytes = g.scale_plane_offset + g.scale_plane_bytes;
    return g;
}

std::vector<std::byte> quantize_and_pack_row_split(
    const float* weights, uint64_t n, uint64_t k, std::string_view format, int num_threads) {

    auto spec = get_row_split_spec(format);
    auto g = compute_geometry(format, n, k);

    std::vector<std::byte> payload(g.total_bytes, std::byte{0});
    auto* low_plane = reinterpret_cast<uint8_t*>(payload.data());
    auto* high_plane = (g.high_plane_bytes > 0) ? reinterpret_cast<uint8_t*>(payload.data() + g.high_plane_offset) : nullptr;
    auto* scale_plane = reinterpret_cast<uint16_t*>(payload.data() + g.scale_plane_offset);

    auto worker = [&](uint64_t r_begin, uint64_t r_end) {
        for (uint64_t r = r_begin; r < r_end; ++r) {
            const float* row_weights = weights + r * k;
            for (uint64_t grp = 0; grp < g.groups_per_row; ++grp) {
                uint64_t g_idx = r * g.groups_per_row + grp;
                uint64_t col_start = grp * spec.group_size;

                float max_abs = 0.0f;
                for (int i = 0; i < spec.group_size; ++i) {
                    uint64_t col = col_start + i;
                    float w = (col < k) ? row_weights[col] : 0.0f;
                    float abs_w = std::fabs(w);
                    if (abs_w > max_abs) max_abs = abs_w;
                }

                float raw_scale = max_abs / static_cast<float>(spec.qmax);
                uint16_t s16 = float_to_fp16(raw_scale);
                if (s16 == 0 && max_abs > 0.0f) {
                    s16 = 0x0001; // min positive subnormal
                }
                scale_plane[g_idx] = s16;

                float actual_scale = fp16_to_float(s16);
                float recip = (actual_scale > 0.0f) ? (1.0f / actual_scale) : 0.0f;

                uint8_t* grp_low = low_plane + g_idx * 32;
                uint8_t* grp_high = high_plane ? (high_plane + g_idx * g.high_bytes_per_group) : nullptr;

                for (int i = 0; i < spec.group_size; ++i) {
                    uint64_t col = col_start + i;
                    float w = (col < k) ? row_weights[col] : 0.0f;
                    int code = static_cast<int>(std::round(w * recip));
                    if (code < spec.qmin) code = spec.qmin;
                    if (code > spec.qmax) code = spec.qmax;

                    if (spec.bits == 8) {
                        grp_low[i] = static_cast<uint8_t>(static_cast<int8_t>(code));
                    } else if (spec.bits == 4) {
                        uint8_t nibble = static_cast<uint8_t>(code) & 0x0F;
                        if ((i & 1) == 0) {
                            grp_low[i / 2] = nibble;
                        } else {
                            grp_low[i / 2] |= (nibble << 4);
                        }
                    } else if (spec.bits == 5) {
                        uint8_t nibble = static_cast<uint8_t>(code) & 0x0F;
                        if ((i & 1) == 0) {
                            grp_low[i / 2] = nibble;
                        } else {
                            grp_low[i / 2] |= (nibble << 4);
                        }
                        uint8_t hbit = (static_cast<uint8_t>(code) >> 4) & 1;
                        grp_high[i / 8] |= (hbit << (i % 8));
                    }
                }
            }
        }
    };

    if (num_threads <= 1 || n < 16) {
        worker(0, n);
    } else {
        std::vector<std::thread> pool;
        uint64_t chunk = (n + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            uint64_t begin = t * chunk;
            uint64_t end = std::min(n, begin + chunk);
            if (begin < end) {
                pool.emplace_back(worker, begin, end);
            }
        }
        for (auto& th : pool) th.join();
    }

    return payload;
}

std::vector<std::byte> float_to_bf16_bytes(const float* weights, size_t count) {
    std::vector<std::byte> out(count * sizeof(uint16_t));
    auto* dst = reinterpret_cast<uint16_t*>(out.data());
    for (size_t i = 0; i < count; ++i) {
        dst[i] = float_to_bf16(weights[i]);
    }
    return out;
}

std::vector<std::byte> float_to_fp32_bytes(const float* weights, size_t count) {
    std::vector<std::byte> out(count * sizeof(float));
    std::memcpy(out.data(), weights, count * sizeof(float));
    return out;
}

void dequantize_gguf_tensor_to_floats(const GGUFFile& gguf, const TensorDescriptor& desc, float* out_floats) {
    auto raw_data = gguf.tensor_data(desc);
    int64_t total_elements = 1;
    for (uint64_t d : desc.raw_dims) total_elements *= static_cast<int64_t>(d);

    int res = ninfer_ggml_dequantize_row(desc.ggml_type, raw_data.data(), out_floats, total_elements);
    if (res != 0) {
        throw std::runtime_error("Dequantization failed for tensor " + desc.name +
                                 " (type " + std::string(ninfer_ggml_type_name(desc.ggml_type)) + ")");
    }
}

std::vector<float> load_and_dequantize(const GGUFFile& gguf, std::string_view name) {
    const auto* desc = gguf.find_tensor(name);
    if (!desc) {
        throw std::runtime_error("Required tensor not found in GGUF: " + std::string(name));
    }
    int64_t total_elements = 1;
    for (uint64_t d : desc->raw_dims) total_elements *= static_cast<int64_t>(d);

    std::vector<float> buf(total_elements);
    dequantize_gguf_tensor_to_floats(gguf, *desc, buf.data());
    return buf;
}

struct StoredItem {
    std::string name;
    std::string kind; // "tensor" or "resource"
    std::string format;
    std::string layout;
    std::string encoding;
    std::vector<uint64_t> shape;
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::vector<std::byte> data;
};

std::vector<std::byte> read_file_bytes(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Failed to read file: " + p.string());
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::byte> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

} // namespace

void convert_gguf_to_ninfer(const ConvertOptions& options) {
    if (options.input_gguf.empty() || options.output_ninfer.empty()) {
        throw std::invalid_argument("Input and output paths must not be empty");
    }

    if (options.progress_callback) {
        options.progress_callback("Opening GGUF file...", 0.0f);
    }

    GGUFFile gguf(options.input_gguf);
    ModelConfig cfg = gguf.extract_config();

    std::string model_id = (cfg.embedding_length == 5120) ? "qwen3.8-27b" : "qwen3.6-27b";
    std::string weights_id = options.profile.empty() ? "groupwise-int" : options.profile;

    std::vector<StoredItem> items;

    // 1. Add Frontend Resources
    std::filesystem::path res_dir = options.resources_dir;
    if (res_dir.empty()) {
        res_dir = "resources/qwen3_8_27b";
    }

    static const std::array<const char*, 6> kResourceFiles = {
        "frontend/tokenizer.json",
        "frontend/tokenizer_config.json",
        "frontend/chat_template.jinja",
        "frontend/generation_config.json",
        "frontend/preprocessor_config.json",
        "frontend/video_preprocessor_config.json",
    };

    for (const char* rname : kResourceFiles) {
        std::filesystem::path rpath = res_dir / rname;
        std::vector<std::byte> rdata;
        if (std::filesystem::exists(rpath)) {
            rdata = read_file_bytes(rpath);
        } else if (std::string_view(rname) == "frontend/chat_template.jinja") {
            if (const auto* tmpl = gguf.find_metadata("tokenizer.chat_template")) {
                std::string s = tmpl->as_string();
                rdata.resize(s.size());
                std::memcpy(rdata.data(), s.data(), s.size());
            }
        }
        if (rdata.empty()) {
            // Provide placeholder minimal JSON
            std::string fallback = "{}";
            rdata.resize(fallback.size());
            std::memcpy(rdata.data(), fallback.data(), fallback.size());
        }

        StoredItem item;
        item.name = rname;
        item.kind = "resource";
        item.encoding = "raw-bytes-v1";
        item.bytes = rdata.size();
        item.data = std::move(rdata);
        items.push_back(std::move(item));
    }

    // 2. Vocabulary endpoints: token_embedding and output_head
    if (options.progress_callback) {
        options.progress_callback("Converting token_embedding and output_head...", 0.05f);
    }

    // token_embedding
    {
        auto embd_floats = load_and_dequantize(gguf, "token_embd.weight");
        uint64_t rows = 248320;
        uint64_t cols = 5120;
        StoredItem item;
        item.name = "text/token_embedding";
        item.kind = "tensor";
        item.format = "W8G32_F16S";
        item.layout = "row-split-k128-v1";
        item.shape = {rows, cols};
        item.data = quantize_and_pack_row_split(embd_floats.data(), rows, cols, item.format, options.threads);
        item.bytes = item.data.size();
        items.push_back(std::move(item));
    }

    // output.weight -> text/output_head
    {
        auto out_floats = load_and_dequantize(gguf, "output.weight");
        uint64_t rows = 248320;
        uint64_t cols = 5120;
        StoredItem item;
        item.name = "text/output_head";
        item.kind = "tensor";
        item.format = "W8G32_F16S";
        item.layout = "row-split-k128-v1";
        item.shape = {rows, cols};
        item.data = quantize_and_pack_row_split(out_floats.data(), rows, cols, item.format, options.threads);
        item.bytes = item.data.size();
        items.push_back(std::move(item));
    }

    // final_norm
    {
        auto norm_floats = load_and_dequantize(gguf, "output_norm.weight");
        StoredItem item;
        item.name = "text/final_norm";
        item.kind = "tensor";
        item.format = "BF16";
        item.layout = "contiguous-le-v1";
        item.shape = {5120};
        item.data = float_to_bf16_bytes(norm_floats.data(), norm_floats.size());
        item.bytes = item.data.size();
        items.push_back(std::move(item));
    }

    // 3. Convert 64 Text Layers
    for (int layer = 0; layer < 64; ++layer) {
        if (options.progress_callback && (layer % 8 == 0 || layer == 63)) {
            float prog = 0.10f + 0.80f * (static_cast<float>(layer) / 64.0f);
            options.progress_callback("Converting layer " + std::to_string(layer) + " / 64...", prog);
        }

        std::string src_pfx = "blk." + std::to_string(layer) + ".";
        std::string dst_pfx = "text/layers/" + std::to_string(layer) + "/";

        bool is_full = (layer >= 3 && (layer - 3) % 4 == 0);

        // input_norm
        {
            auto f = load_and_dequantize(gguf, src_pfx + "attn_norm.weight");
            StoredItem item;
            item.name = dst_pfx + "input_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {5120};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        if (is_full) {
            // Full attention layer
            // attn_q contains query (6144) + gate (6144)
            auto q_all = load_and_dequantize(gguf, src_pfx + "attn_q.weight");
            auto k = load_and_dequantize(gguf, src_pfx + "attn_k.weight");
            auto v = load_and_dequantize(gguf, src_pfx + "attn_v.weight");

            // query_key = concat(query[6144, 5120], k[1024, 5120]) -> [7168, 5120]
            {
                std::vector<float> qk(7168 * 5120);
                std::memcpy(qk.data(), q_all.data(), 6144 * 5120 * sizeof(float));
                std::memcpy(qk.data() + 6144 * 5120, k.data(), 1024 * 5120 * sizeof(float));

                StoredItem item;
                item.name = dst_pfx + "attention/query_key";
                item.kind = "tensor";
                item.format = "Q4G64_F16S";
                item.layout = "row-split-k128-v1";
                item.shape = {7168, 5120};
                item.data = quantize_and_pack_row_split(qk.data(), 7168, 5120, item.format, options.threads);
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // gate_value = concat(gate[6144, 5120], v[1024, 5120]) -> [7168, 5120]
            {
                std::vector<float> gv(7168 * 5120);
                std::memcpy(gv.data(), q_all.data() + 6144 * 5120, 6144 * 5120 * sizeof(float));
                std::memcpy(gv.data() + 6144 * 5120, v.data(), 1024 * 5120 * sizeof(float));

                StoredItem item;
                item.name = dst_pfx + "attention/gate_value";
                item.kind = "tensor";
                item.format = "Q5G64_F16S";
                item.layout = "row-split-k128-v1";
                item.shape = {7168, 5120};
                item.data = quantize_and_pack_row_split(gv.data(), 7168, 5120, item.format, options.threads);
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // query_norm
            {
                auto f = load_and_dequantize(gguf, src_pfx + "attn_q_norm.weight");
                StoredItem item;
                item.name = dst_pfx + "attention/query_norm";
                item.kind = "tensor";
                item.format = "BF16";
                item.layout = "contiguous-le-v1";
                item.shape = {256};
                item.data = float_to_bf16_bytes(f.data(), f.size());
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // key_norm
            {
                auto f = load_and_dequantize(gguf, src_pfx + "attn_k_norm.weight");
                StoredItem item;
                item.name = dst_pfx + "attention/key_norm";
                item.kind = "tensor";
                item.format = "BF16";
                item.layout = "contiguous-le-v1";
                item.shape = {256};
                item.data = float_to_bf16_bytes(f.data(), f.size());
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // output [5120, 6144]
            {
                auto f = load_and_dequantize(gguf, src_pfx + "attn_output.weight");
                StoredItem item;
                item.name = dst_pfx + "attention/output";
                item.kind = "tensor";
                item.format = "Q5G64_F16S";
                item.layout = "row-split-k128-v1";
                item.shape = {5120, 6144};
                item.data = quantize_and_pack_row_split(f.data(), 5120, 6144, item.format, options.threads);
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

        } else {
            // GDN / SSM layer
            // a_log [48]
            {
                auto f = load_and_dequantize(gguf, src_pfx + "ssm_a");
                StoredItem item;
                item.name = dst_pfx + "gdn/a_log";
                item.kind = "tensor";
                item.format = "FP32";
                item.layout = "contiguous-le-v1";
                item.shape = {48};
                item.data = float_to_fp32_bytes(f.data(), f.size());
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // dt_bias [48]
            {
                auto f = load_and_dequantize(gguf, src_pfx + "ssm_dt.bias");
                StoredItem item;
                item.name = dst_pfx + "gdn/dt_bias";
                item.kind = "tensor";
                item.format = "FP32";
                item.layout = "contiguous-le-v1";
                item.shape = {48};
                item.data = float_to_fp32_bytes(f.data(), f.size());
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // convolution [4, 10240]
            {
                auto f = load_and_dequantize(gguf, src_pfx + "ssm_conv1d.weight");
                StoredItem item;
                item.name = dst_pfx + "gdn/convolution";
                item.kind = "tensor";
                item.format = "BF16";
                item.layout = "contiguous-le-v1";
                item.shape = {4, 10240};
                item.data = float_to_bf16_bytes(f.data(), f.size());
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // a_projection [48, 5120]
            {
                auto f = load_and_dequantize(gguf, src_pfx + "ssm_alpha.weight");
                StoredItem item;
                item.name = dst_pfx + "gdn/a_projection";
                item.kind = "tensor";
                item.format = "BF16";
                item.layout = "contiguous-le-v1";
                item.shape = {48, 5120};
                item.data = float_to_bf16_bytes(f.data(), f.size());
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // b_projection [48, 5120]
            {
                auto f = load_and_dequantize(gguf, src_pfx + "ssm_beta.weight");
                StoredItem item;
                item.name = dst_pfx + "gdn/b_projection";
                item.kind = "tensor";
                item.format = "BF16";
                item.layout = "contiguous-le-v1";
                item.shape = {48, 5120};
                item.data = float_to_bf16_bytes(f.data(), f.size());
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // attn_qkv [10240, 5120] and attn_gate [6144, 5120]
            auto qkv = load_and_dequantize(gguf, src_pfx + "attn_qkv.weight");
            auto gate = load_and_dequantize(gguf, src_pfx + "attn_gate.weight");

            // gdn/query_key = q(2048) + k(2048) -> [4096, 5120]
            {
                std::vector<float> qk(4096 * 5120);
                std::memcpy(qk.data(), qkv.data(), 4096 * 5120 * sizeof(float));

                StoredItem item;
                item.name = dst_pfx + "gdn/query_key";
                item.kind = "tensor";
                item.format = "Q4G64_F16S";
                item.layout = "row-split-k128-v1";
                item.shape = {4096, 5120};
                item.data = quantize_and_pack_row_split(qk.data(), 4096, 5120, item.format, options.threads);
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // gdn/value_z = v(6144) + gate(6144) -> [12288, 5120]
            {
                std::vector<float> vz(12288 * 5120);
                std::memcpy(vz.data(), qkv.data() + 4096 * 5120, 6144 * 5120 * sizeof(float));
                std::memcpy(vz.data() + 6144 * 5120, gate.data(), 6144 * 5120 * sizeof(float));

                StoredItem item;
                item.name = dst_pfx + "gdn/value_z";
                item.kind = "tensor";
                item.format = "Q5G64_F16S";
                item.layout = "row-split-k128-v1";
                item.shape = {12288, 5120};
                item.data = quantize_and_pack_row_split(vz.data(), 12288, 5120, item.format, options.threads);
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // gdn/norm [128]
            {
                auto f = load_and_dequantize(gguf, src_pfx + "ssm_norm.weight");
                StoredItem item;
                item.name = dst_pfx + "gdn/norm";
                item.kind = "tensor";
                item.format = "BF16";
                item.layout = "contiguous-le-v1";
                item.shape = {128};
                item.data = float_to_bf16_bytes(f.data(), f.size());
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }

            // gdn/output [5120, 6144]
            {
                auto f = load_and_dequantize(gguf, src_pfx + "ssm_out.weight");
                StoredItem item;
                item.name = dst_pfx + "gdn/output";
                item.kind = "tensor";
                item.format = "Q5G64_F16S";
                item.layout = "row-split-k128-v1";
                item.shape = {5120, 6144};
                item.data = quantize_and_pack_row_split(f.data(), 5120, 6144, item.format, options.threads);
                item.bytes = item.data.size();
                items.push_back(std::move(item));
            }
        }

        // post_attention_norm [5120]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "post_attention_norm.weight");
            StoredItem item;
            item.name = dst_pfx + "post_attention_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {5120};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // MLP: ffn_gate [17408, 5120] and ffn_up [17408, 5120] -> mlp/gate_up [34816, 5120]
        {
            auto gate = load_and_dequantize(gguf, src_pfx + "ffn_gate.weight");
            auto up = load_and_dequantize(gguf, src_pfx + "ffn_up.weight");
            std::vector<float> gate_up(34816 * 5120);
            std::memcpy(gate_up.data(), gate.data(), 17408 * 5120 * sizeof(float));
            std::memcpy(gate_up.data() + 17408 * 5120, up.data(), 17408 * 5120 * sizeof(float));

            StoredItem item;
            item.name = dst_pfx + "mlp/gate_up";
            item.kind = "tensor";
            item.format = "Q4G64_F16S";
            item.layout = "row-split-k128-v1";
            item.shape = {34816, 5120};
            item.data = quantize_and_pack_row_split(gate_up.data(), 34816, 5120, item.format, options.threads);
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mlp/down [5120, 17408]
        {
            auto down = load_and_dequantize(gguf, src_pfx + "ffn_down.weight");
            StoredItem item;
            item.name = dst_pfx + "mlp/down";
            item.kind = "tensor";
            item.format = "Q5G64_F16S";
            item.layout = "row-split-k128-v1";
            item.shape = {5120, 17408};
            item.data = quantize_and_pack_row_split(down.data(), 5120, 17408, item.format, options.threads);
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }
    }

    // 4. Draft Head (required by 27B binding contract for validate_only)
    {
        StoredItem d_item;
        d_item.name = "text/draft_head";
        d_item.kind = "tensor";
        d_item.format = "Q4G64_F16S";
        d_item.layout = "row-split-k128-v1";
        d_item.shape = {131072, 5120};
        auto g = compute_geometry(d_item.format, 131072, 5120);
        d_item.data.resize(g.total_bytes, std::byte{0});
        d_item.bytes = d_item.data.size();
        items.push_back(std::move(d_item));

        StoredItem id_item;
        id_item.name = "text/draft_head_token_ids";
        id_item.kind = "tensor";
        id_item.format = "I32";
        id_item.layout = "contiguous-le-v1";
        id_item.shape = {131072};
        id_item.data.resize(131072 * sizeof(int32_t));
        auto* ids = reinterpret_cast<int32_t*>(id_item.data.data());
        for (int32_t i = 0; i < 131072; ++i) ids[i] = i;
        id_item.bytes = id_item.data.size();
        items.push_back(std::move(id_item));
    }

    // 5. MTP layer (if present in GGUF as blk.64.*)
    const auto* mtp_eh = gguf.find_tensor("blk.64.nextn.eh_proj.weight");
    if (mtp_eh != nullptr) {
        if (options.progress_callback) {
            options.progress_callback("Converting MTP layer...", 0.92f);
        }

        std::string src_pfx = "blk.64.";

        // mtp/input_projection [5120, 10240]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "nextn.eh_proj.weight");
            StoredItem item;
            item.name = "mtp/input_projection";
            item.kind = "tensor";
            item.format = "W8G32_F16S";
            item.layout = "row-split-k128-v1";
            item.shape = {5120, 10240};
            item.data = quantize_and_pack_row_split(f.data(), 5120, 10240, item.format, options.threads);
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/embedding_norm [5120]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "nextn.enorm.weight");
            StoredItem item;
            item.name = "mtp/embedding_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {5120};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/hidden_norm [5120]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "nextn.hnorm.weight");
            StoredItem item;
            item.name = "mtp/hidden_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {5120};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/layer/input_norm [5120]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "attn_norm.weight");
            StoredItem item;
            item.name = "mtp/layer/input_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {5120};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/layer/attention/query_key_gate_value [14336, 5120]
        {
            auto q = load_and_dequantize(gguf, src_pfx + "attn_q.weight");
            auto k = load_and_dequantize(gguf, src_pfx + "attn_k.weight");
            auto v = load_and_dequantize(gguf, src_pfx + "attn_v.weight");
            // q has query(6144) + gate(6144)
            std::vector<float> fused(14336 * 5120);
            std::memcpy(fused.data(), q.data(), 6144 * 5120 * sizeof(float));
            std::memcpy(fused.data() + 6144 * 5120, k.data(), 1024 * 5120 * sizeof(float));
            std::memcpy(fused.data() + 7168 * 5120, q.data() + 6144 * 5120, 6144 * 5120 * sizeof(float));
            std::memcpy(fused.data() + 13312 * 5120, v.data(), 1024 * 5120 * sizeof(float));

            StoredItem item;
            item.name = "mtp/layer/attention/query_key_gate_value";
            item.kind = "tensor";
            item.format = "W8G32_F16S";
            item.layout = "row-split-k128-v1";
            item.shape = {14336, 5120};
            item.data = quantize_and_pack_row_split(fused.data(), 14336, 5120, item.format, options.threads);
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/layer/attention/query_norm [256]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "attn_q_norm.weight");
            StoredItem item;
            item.name = "mtp/layer/attention/query_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {256};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/layer/attention/key_norm [256]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "attn_k_norm.weight");
            StoredItem item;
            item.name = "mtp/layer/attention/key_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {256};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/layer/attention/output [5120, 6144]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "attn_output.weight");
            StoredItem item;
            item.name = "mtp/layer/attention/output";
            item.kind = "tensor";
            item.format = "W8G32_F16S";
            item.layout = "row-split-k128-v1";
            item.shape = {5120, 6144};
            item.data = quantize_and_pack_row_split(f.data(), 5120, 6144, item.format, options.threads);
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/layer/post_attention_norm [5120]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "post_attention_norm.weight");
            StoredItem item;
            item.name = "mtp/layer/post_attention_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {5120};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/layer/mlp/gate_up [34816, 5120]
        {
            auto gate = load_and_dequantize(gguf, src_pfx + "ffn_gate.weight");
            auto up = load_and_dequantize(gguf, src_pfx + "ffn_up.weight");
            std::vector<float> gate_up(34816 * 5120);
            std::memcpy(gate_up.data(), gate.data(), 17408 * 5120 * sizeof(float));
            std::memcpy(gate_up.data() + 17408 * 5120, up.data(), 17408 * 5120 * sizeof(float));

            StoredItem item;
            item.name = "mtp/layer/mlp/gate_up";
            item.kind = "tensor";
            item.format = "W8G32_F16S";
            item.layout = "row-split-k128-v1";
            item.shape = {34816, 5120};
            item.data = quantize_and_pack_row_split(gate_up.data(), 34816, 5120, item.format, options.threads);
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/layer/mlp/down [5120, 17408]
        {
            auto down = load_and_dequantize(gguf, src_pfx + "ffn_down.weight");
            StoredItem item;
            item.name = "mtp/layer/mlp/down";
            item.kind = "tensor";
            item.format = "W8G32_F16S";
            item.layout = "row-split-k128-v1";
            item.shape = {5120, 17408};
            item.data = quantize_and_pack_row_split(down.data(), 5120, 17408, item.format, options.threads);
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }

        // mtp/final_norm [5120]
        {
            auto f = load_and_dequantize(gguf, src_pfx + "nextn.shared_head_norm.weight");
            StoredItem item;
            item.name = "mtp/final_norm";
            item.kind = "tensor";
            item.format = "BF16";
            item.layout = "contiguous-le-v1";
            item.shape = {5120};
            item.data = float_to_bf16_bytes(f.data(), f.size());
            item.bytes = item.data.size();
            items.push_back(std::move(item));
        }
    }

    // 6. Write .ninfer Container
    if (options.progress_callback) {
        options.progress_callback("Writing .ninfer container to disk...", 0.96f);
    }

    // Compute aligned payload offsets for all items
    uint64_t current_payload_offset = 0;
    for (auto& item : items) {
        uint64_t alignment = (item.kind == "tensor") ? kPlaneAlignment : 1;
        current_payload_offset = align_up(current_payload_offset, alignment);
        item.offset = current_payload_offset;
        current_payload_offset += item.bytes;
    }

    // Build JSON metadata
    Json json_dir;
    json_dir["identity"] = {
        {"model_id", model_id},
        {"weights_id", weights_id}
    };
    Json obj_list = Json::array();
    for (const auto& item : items) {
        Json obj;
        obj["name"] = item.name;
        obj["kind"] = item.kind;
        obj["offset"] = item.offset;
        obj["bytes"] = item.bytes;
        if (item.kind == "tensor") {
            obj["format"] = item.format;
            obj["layout"] = item.layout;
            obj["shape"] = item.shape;
        } else {
            obj["encoding"] = item.encoding;
        }
        obj_list.push_back(std::move(obj));
    }
    json_dir["objects"] = std::move(obj_list);

    std::string json_str = json_dir.dump();
    uint64_t json_bytes = json_str.size();
    uint64_t payload_start = align_up(16 + json_bytes, kPayloadAlignment);

    // Open output file
    std::filesystem::create_directories(options.output_ninfer.parent_path());
    std::ofstream out(options.output_ninfer, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to create output file: " + options.output_ninfer.string());
    }

    // Magic (8 bytes): "NINFER\0\2"
    static constexpr std::array<char, 8> kMagic = {'N', 'I', 'N', 'F', 'E', 'R', 0, 2};
    out.write(kMagic.data(), 8);

    // JSON length (8 bytes)
    out.write(reinterpret_cast<const char*>(&json_bytes), 8);

    // JSON body
    out.write(json_str.data(), json_bytes);

    // Zero-pad up to payload_start
    size_t written_so_far = 16 + json_bytes;
    if (payload_start > written_so_far) {
        std::vector<char> pad(payload_start - written_so_far, 0);
        out.write(pad.data(), pad.size());
    }

    // Write all item payloads
    for (const auto& item : items) {
        // Pad to item.offset
        uint64_t cur_pos = static_cast<uint64_t>(out.tellp()) - payload_start;
        if (item.offset > cur_pos) {
            std::vector<char> pad(item.offset - cur_pos, 0);
            out.write(pad.data(), pad.size());
        }
        out.write(reinterpret_cast<const char*>(item.data.data()), item.bytes);
    }

    out.flush();
    if (!out) {
        throw std::runtime_error("Write failure while writing " + options.output_ninfer.string());
    }

    if (options.progress_callback) {
        options.progress_callback("Conversion completed successfully!", 1.0f);
    }
}

} // namespace ninfer::gguf
