#include "dequant.h"

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "GGML ABORT at %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

static inline float fp16_to_fp32(uint16_t h) {
    return GGML_FP16_TO_FP32(h);
}

static inline float bf16_to_fp32(uint16_t b) {
    ggml_bf16_t bf;
    bf.bits = b;
    return GGML_BF16_TO_FP32(bf);
}

static inline uint16_t fp32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    // Round to nearest even
    uint32_t lsb = (x >> 16) & 1;
    uint32_t bias = 0x7fff + lsb;
    x += bias;
    return static_cast<uint16_t>(x >> 16);
}

int ninfer_ggml_dequantize_row(int ggml_type, const void* src, float* dst, int64_t n_elements) {
    if (!src || !dst || n_elements <= 0) return -1;

    switch (ggml_type) {
    case GGML_TYPE_F32:
        std::memcpy(dst, src, n_elements * sizeof(float));
        return 0;
    case GGML_TYPE_F16: {
        const auto* s = static_cast<const uint16_t*>(src);
        for (int64_t i = 0; i < n_elements; ++i) dst[i] = fp16_to_fp32(s[i]);
        return 0;
    }
    case GGML_TYPE_BF16: {
        const auto* s = static_cast<const uint16_t*>(src);
        for (int64_t i = 0; i < n_elements; ++i) dst[i] = bf16_to_fp32(s[i]);
        return 0;
    }
    case GGML_TYPE_Q4_0:
        dequantize_row_q4_0(static_cast<const block_q4_0*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q4_1:
        dequantize_row_q4_1(static_cast<const block_q4_1*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q5_0:
        dequantize_row_q5_0(static_cast<const block_q5_0*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q5_1:
        dequantize_row_q5_1(static_cast<const block_q5_1*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q8_0:
        dequantize_row_q8_0(static_cast<const block_q8_0*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q2_K:
        dequantize_row_q2_K(static_cast<const block_q2_K*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q3_K:
        dequantize_row_q3_K(static_cast<const block_q3_K*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q4_K:
        dequantize_row_q4_K(static_cast<const block_q4_K*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q5_K:
        dequantize_row_q5_K(static_cast<const block_q5_K*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q6_K:
        dequantize_row_q6_K(static_cast<const block_q6_K*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_Q8_K:
        dequantize_row_q8_K(static_cast<const block_q8_K*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ2_XXS:
        dequantize_row_iq2_xxs(static_cast<const block_iq2_xxs*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ2_XS:
        dequantize_row_iq2_xs(static_cast<const block_iq2_xs*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ2_S:
        dequantize_row_iq2_s(static_cast<const block_iq2_s*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ3_XXS:
        dequantize_row_iq3_xxs(static_cast<const block_iq3_xxs*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ3_S:
        dequantize_row_iq3_s(static_cast<const block_iq3_s*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ1_S:
        dequantize_row_iq1_s(static_cast<const block_iq1_s*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ1_M:
        dequantize_row_iq1_m(static_cast<const block_iq1_m*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ4_NL:
        dequantize_row_iq4_nl(static_cast<const block_iq4_nl*>(src), dst, n_elements);
        return 0;
    case GGML_TYPE_IQ4_XS:
        dequantize_row_iq4_xs(static_cast<const block_iq4_xs*>(src), dst, n_elements);
        return 0;
    default:
        return -1;
    }
}

int ninfer_ggml_dequantize_to_bf16(int ggml_type, const void* src, uint16_t* dst_bf16, int64_t n_elements) {
    if (!src || !dst_bf16 || n_elements <= 0) return -1;

    if (ggml_type == GGML_TYPE_BF16) {
        std::memcpy(dst_bf16, src, n_elements * sizeof(uint16_t));
        return 0;
    }

    constexpr int64_t kChunk = 4096;
    float temp[kChunk];

    for (int64_t offset = 0; offset < n_elements; offset += kChunk) {
        int64_t count = std::min(kChunk, n_elements - offset);
        int64_t bs = ninfer_ggml_blck_size(ggml_type);
        size_t ts = ninfer_ggml_type_size(ggml_type);
        assert(count % bs == 0);
        size_t byte_offset = (offset / bs) * ts;
        const void* chunk_src = static_cast<const char*>(src) + byte_offset;

        int res = ninfer_ggml_dequantize_row(ggml_type, chunk_src, temp, count);
        if (res != 0) return res;

        for (int64_t i = 0; i < count; ++i) {
            dst_bf16[offset + i] = fp32_to_bf16(temp[i]);
        }
    }
    return 0;
}

const char* ninfer_ggml_type_name(int ggml_type) {
    switch (ggml_type) {
    case GGML_TYPE_F32: return "F32";
    case GGML_TYPE_F16: return "F16";
    case GGML_TYPE_Q4_0: return "Q4_0";
    case GGML_TYPE_Q4_1: return "Q4_1";
    case GGML_TYPE_Q5_0: return "Q5_0";
    case GGML_TYPE_Q5_1: return "Q5_1";
    case GGML_TYPE_Q8_0: return "Q8_0";
    case GGML_TYPE_Q8_1: return "Q8_1";
    case GGML_TYPE_Q2_K: return "Q2_K";
    case GGML_TYPE_Q3_K: return "Q3_K";
    case GGML_TYPE_Q4_K: return "Q4_K";
    case GGML_TYPE_Q5_K: return "Q5_K";
    case GGML_TYPE_Q6_K: return "Q6_K";
    case GGML_TYPE_Q8_K: return "Q8_K";
    case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
    case GGML_TYPE_IQ2_XS: return "IQ2_XS";
    case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
    case GGML_TYPE_IQ1_S: return "IQ1_S";
    case GGML_TYPE_IQ4_NL: return "IQ4_NL";
    case GGML_TYPE_IQ3_S: return "IQ3_S";
    case GGML_TYPE_IQ2_S: return "IQ2_S";
    case GGML_TYPE_IQ4_XS: return "IQ4_XS";
    case GGML_TYPE_I8: return "I8";
    case GGML_TYPE_I16: return "I16";
    case GGML_TYPE_I32: return "I32";
    case GGML_TYPE_I64: return "I64";
    case GGML_TYPE_F64: return "F64";
    case GGML_TYPE_IQ1_M: return "IQ1_M";
    case GGML_TYPE_BF16: return "BF16";
    case GGML_TYPE_Q1_0: return "Q1_0";
    case GGML_TYPE_Q2_0: return "Q2_0";
    case GGML_TYPE_TQ1_0: return "TQ1_0";
    case GGML_TYPE_TQ2_0: return "TQ2_0";
    case GGML_TYPE_MXFP4: return "MXFP4";
    case GGML_TYPE_NVFP4: return "NVFP4";
    default: return "UNKNOWN";
    }
}

int64_t ninfer_ggml_blck_size(int ggml_type) {
    switch (ggml_type) {
    case GGML_TYPE_F32:
    case GGML_TYPE_F16:
    case GGML_TYPE_BF16:
    case GGML_TYPE_I8:
    case GGML_TYPE_I16:
    case GGML_TYPE_I32:
    case GGML_TYPE_I64:
    case GGML_TYPE_F64:
        return 1;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q1_0:
    case GGML_TYPE_Q2_0:
        return 32;
    case GGML_TYPE_MXFP4:
        return 32;
    case GGML_TYPE_NVFP4:
        return 32;
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_K:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ1_S:
    case GGML_TYPE_IQ4_NL:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ4_XS:
    case GGML_TYPE_IQ1_M:
    case GGML_TYPE_TQ1_0:
    case GGML_TYPE_TQ2_0:
        return 256;
    default:
        return 1;
    }
}

size_t ninfer_ggml_type_size(int ggml_type) {
    switch (ggml_type) {
    case GGML_TYPE_F32: return sizeof(float);
    case GGML_TYPE_F16: return sizeof(uint16_t);
    case GGML_TYPE_BF16: return sizeof(uint16_t);
    case GGML_TYPE_I8: return sizeof(int8_t);
    case GGML_TYPE_I16: return sizeof(int16_t);
    case GGML_TYPE_I32: return sizeof(int32_t);
    case GGML_TYPE_I64: return sizeof(int64_t);
    case GGML_TYPE_F64: return sizeof(double);
    case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
    case GGML_TYPE_Q4_1: return sizeof(block_q4_1);
    case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
    case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
    case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
    case GGML_TYPE_Q8_1: return sizeof(block_q8_1);
    case GGML_TYPE_Q1_0: return sizeof(block_q1_0);
    case GGML_TYPE_Q2_0: return sizeof(block_q2_0);
    case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
    case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
    case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
    case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
    case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
    case GGML_TYPE_Q8_K: return sizeof(block_q8_K);
    case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
    case GGML_TYPE_IQ2_XS: return sizeof(block_iq2_xs);
    case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
    case GGML_TYPE_IQ1_S: return sizeof(block_iq1_s);
    case GGML_TYPE_IQ4_NL: return sizeof(block_iq4_nl);
    case GGML_TYPE_IQ3_S: return sizeof(block_iq3_s);
    case GGML_TYPE_IQ2_S: return sizeof(block_iq2_s);
    case GGML_TYPE_IQ4_XS: return sizeof(block_iq4_xs);
    case GGML_TYPE_IQ1_M: return sizeof(block_iq1_m);
    case GGML_TYPE_TQ1_0: return sizeof(block_tq1_0);
    case GGML_TYPE_TQ2_0: return sizeof(block_tq2_0);
    case GGML_TYPE_MXFP4: return sizeof(block_mxfp4);
    case GGML_TYPE_NVFP4: return sizeof(block_nvfp4);
    default: return 0;
    }
}

size_t ninfer_ggml_row_size(int ggml_type, int64_t n_elements) {
    int64_t bs = ninfer_ggml_blck_size(ggml_type);
    size_t ts = ninfer_ggml_type_size(ggml_type);
    assert(n_elements % bs == 0);
    return (n_elements / bs) * ts;
}

// Stubs for ggml-quants.c references:
size_t ggml_type_size(enum ggml_type type) { return ninfer_ggml_type_size(type); }
size_t ggml_row_size(enum ggml_type type, int64_t ne) { return ninfer_ggml_row_size(type, ne); }
const char* ggml_type_name(enum ggml_type type) { return ninfer_ggml_type_name(type); }

} // extern "C"
