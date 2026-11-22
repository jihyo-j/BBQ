#include "adcbq_sign_packing.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
#if !defined(__AVX512F__)
#error "Compile this test with AVX-512 enabled"
#endif

    std::mt19937 rng(20261019);
    std::uniform_real_distribution<float> distribution(-10.0f, 10.0f);
    const std::vector<int> dimensions = {8, 16, 56, 64, 72, 120, 128, 768, 1024, 2560, 4096};

    for (int dim : dimensions) {
        constexpr size_t rows = 257;
        std::vector<float> input(rows * static_cast<size_t>(dim));
        for (float& value : input) {
            value = distribution(rng);
        }

        input[0] = 1.0f;
        input[1] = -1.0f;
        input[2] = std::numeric_limits<float>::infinity();
        input[3] = -std::numeric_limits<float>::infinity();
        input[4] = 0.0f;
        input[5] = -0.0f;
        input[6] = std::numeric_limits<float>::quiet_NaN();
        input[7] = 2.0f;

        std::vector<uint8_t> scalar;
        std::vector<uint8_t> avx512;
        adcbq::pack_sign_bits_scalar(input.data(), rows, dim, scalar);
        adcbq::pack_sign_bits(input.data(), rows, dim, avx512);
        if (scalar != avx512) {
            throw std::runtime_error("AVX-512 output differs from scalar output at dim=" +
                                     std::to_string(dim));
        }
        if (avx512.front() != UINT8_C(0xA1)) {
            throw std::runtime_error("MSB-first bit order or x > 0 semantics changed");
        }
    }

    std::cout << "PASS: AVX-512 BQ packing is byte-identical to scalar for all test cases\n";
    return 0;
}
