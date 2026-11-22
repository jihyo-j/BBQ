#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace adcbq {

inline void validate_sign_packing_dimension(int dim) {
    if (dim <= 0 || dim % 8 != 0) {
        throw std::invalid_argument("BQ sign packing requires a positive dimension divisible by 8");
    }
}

inline void pack_sign_bits_row_scalar(const float* src, int dim, uint8_t* dst) {
    for (int j = 0; j < dim; j += 8) {
        uint8_t packed = 0;
        for (int bit = 0; bit < 8; ++bit) {
            if (src[j + bit] > 0.0f) {
                packed |= static_cast<uint8_t>(1u << (7 - bit));
            }
        }
        dst[j / 8] = packed;
    }
}

#if defined(__AVX512F__)
// AVX-512 comparison masks number lanes from the least-significant bit, while
// pgvector's binary representation stores the first dimension in each byte's
// most-significant bit. Reverse the bits independently within every byte.
inline uint64_t reverse_bits_in_each_byte(uint64_t value) {
    value = ((value & UINT64_C(0x5555555555555555)) << 1) |
            ((value >> 1) & UINT64_C(0x5555555555555555));
    value = ((value & UINT64_C(0x3333333333333333)) << 2) |
            ((value >> 2) & UINT64_C(0x3333333333333333));
    value = ((value & UINT64_C(0x0f0f0f0f0f0f0f0f)) << 4) |
            ((value >> 4) & UINT64_C(0x0f0f0f0f0f0f0f0f));
    return value;
}

inline void pack_sign_bits_row_avx512(const float* src, int dim, uint8_t* dst) {
    const __m512 zero = _mm512_setzero_ps();
    int j = 0;
    for (; j + 64 <= dim; j += 64) {
        const uint64_t mask0 = static_cast<uint16_t>(_mm512_cmp_ps_mask(
                _mm512_loadu_ps(src + j), zero, _CMP_GT_OQ));
        const uint64_t mask1 = static_cast<uint16_t>(_mm512_cmp_ps_mask(
                _mm512_loadu_ps(src + j + 16), zero, _CMP_GT_OQ));
        const uint64_t mask2 = static_cast<uint16_t>(_mm512_cmp_ps_mask(
                _mm512_loadu_ps(src + j + 32), zero, _CMP_GT_OQ));
        const uint64_t mask3 = static_cast<uint16_t>(_mm512_cmp_ps_mask(
                _mm512_loadu_ps(src + j + 48), zero, _CMP_GT_OQ));
        const uint64_t lane_mask = mask0 | (mask1 << 16) | (mask2 << 32) | (mask3 << 48);
        const uint64_t packed = reverse_bits_in_each_byte(lane_mask);
        std::memcpy(dst + j / 8, &packed, sizeof(packed));
    }

    // Handle dimensions divisible by 8 but not by 64 without touching bytes
    // outside the destination code.
    pack_sign_bits_row_scalar(src + j, dim - j, dst + j / 8);
}
#endif

inline void pack_sign_bits_row(const float* src, int dim, uint8_t* dst) {
#if defined(__AVX512F__)
    pack_sign_bits_row_avx512(src, dim, dst);
#else
    pack_sign_bits_row_scalar(src, dim, dst);
#endif
}

inline void pack_sign_bits_scalar(
        const float* x,
        size_t n,
        int dim,
        std::vector<uint8_t>& codes) {
    validate_sign_packing_dimension(dim);
    const size_t code_size = static_cast<size_t>(dim) / 8;
    codes.resize(n * code_size);
    for (size_t row = 0; row < n; ++row) {
        pack_sign_bits_row_scalar(
                x + row * static_cast<size_t>(dim),
                dim,
                codes.data() + row * code_size);
    }
}

inline void pack_sign_bits(
        const float* x,
        size_t n,
        int dim,
        std::vector<uint8_t>& codes) {
    validate_sign_packing_dimension(dim);
    const size_t code_size = static_cast<size_t>(dim) / 8;
    codes.resize(n * code_size);
    for (size_t row = 0; row < n; ++row) {
        pack_sign_bits_row(
                x + row * static_cast<size_t>(dim),
                dim,
                codes.data() + row * code_size);
    }
}

}  // namespace adcbq
