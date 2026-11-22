#include "qdrant_tq_ffi/qdrant_tq_ffi.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

struct IndexDeleter {
    void operator()(QdrantTqIndex* value) const { qdrant_tq_free(value); }
};

struct QueryDeleter {
    void operator()(QdrantTqQuery* value) const { qdrant_tq_query_free(value); }
};

using IndexPtr = std::unique_ptr<QdrantTqIndex, IndexDeleter>;
using QueryPtr = std::unique_ptr<QdrantTqQuery, QueryDeleter>;

std::string last_error() {
    const char* value = qdrant_tq_last_error();
    return value && *value ? value : "unknown Qdrant TurboQuant error";
}

float read_float(const uint8_t* src) {
    float value = 0.0f;
    std::memcpy(&value, src, sizeof(value));
    return value;
}

std::vector<float> read_fbin_prefix(
        const std::string& path,
        size_t wanted_rows,
        int& dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open fbin: " + path);
    int32_t rows = 0;
    int32_t file_dim = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&file_dim), sizeof(file_dim));
    if (!input || rows <= 0 || file_dim <= 0) {
        throw std::runtime_error("invalid fbin header: " + path);
    }
    const size_t take = std::min(wanted_rows, static_cast<size_t>(rows));
    std::vector<float> result(take * static_cast<size_t>(file_dim));
    input.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!input) throw std::runtime_error("short fbin payload: " + path);
    dim = file_dim;
    return result;
}

std::vector<float> read_npy_prefix(
        const std::string& path,
        size_t wanted_rows,
        int expected_dim) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open npy: " + path);

    uint8_t prefix[10] = {};
    input.read(reinterpret_cast<char*>(prefix), sizeof(prefix));
    if (!input || std::memcmp(prefix, "\x93NUMPY", 6) != 0) {
        throw std::runtime_error("invalid npy header: " + path);
    }
    size_t header_len = 0;
    if (prefix[6] == 1) {
        header_len = static_cast<size_t>(prefix[8]) |
                (static_cast<size_t>(prefix[9]) << 8);
    } else {
        uint8_t extra[2] = {};
        input.read(reinterpret_cast<char*>(extra), sizeof(extra));
        if (!input) throw std::runtime_error("short npy v2 header: " + path);
        header_len = static_cast<size_t>(prefix[8]) |
                (static_cast<size_t>(prefix[9]) << 8) |
                (static_cast<size_t>(extra[0]) << 16) |
                (static_cast<size_t>(extra[1]) << 24);
    }
    std::string header(header_len, '\0');
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!input || header.find("<f4") == std::string::npos) {
        throw std::runtime_error("validator expects little-endian float32 npy: " + path);
    }
    const std::string dim_token = ", " + std::to_string(expected_dim) + ")";
    if (header.find(dim_token) == std::string::npos) {
        throw std::runtime_error("unexpected npy shape: " + header);
    }

    std::vector<float> result(wanted_rows * static_cast<size_t>(expected_dim));
    input.read(
            reinterpret_cast<char*>(result.data()),
            static_cast<std::streamsize>(result.size() * sizeof(float)));
    if (!input) throw std::runtime_error("short npy payload: " + path);
    return result;
}

struct ErrorStats {
    uint64_t comparisons = 0;
    uint64_t failures = 0;
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    float max_rel = 0.0f;

    void add(float actual, float expected, float abs_limit = 2e-4f) {
        const float abs_error = std::abs(actual - expected);
        const float rel_error = abs_error / std::max(1.0f, std::abs(expected));
        ++comparisons;
        sum_abs += abs_error;
        max_abs = std::max(max_abs, abs_error);
        max_rel = std::max(max_rel, rel_error);
        if (!std::isfinite(actual) || !std::isfinite(expected) || abs_error > abs_limit) {
            ++failures;
        }
    }

    double mean_abs() const {
        return comparisons ? sum_abs / static_cast<double>(comparisons) : 0.0;
    }
};

struct DirectScorer {
    const QdrantTqIndex* index = nullptr;
    const uint8_t* codes = nullptr;
    size_t code_size = 0;
    size_t code_bytes = 0;
    size_t padded_dim = 0;
    size_t query_plane_len = 0;
    int dim = 0;
    QueryPtr query;
    std::vector<int8_t> query_planes;
    float postprocess_scale = 0.0f;
    int64_t bias_correction = 0;
    float ec_correction = 0.0f;
    float query_l2 = 0.0f;

    DirectScorer(
            const QdrantTqIndex* index_in,
            const uint8_t* codes_in,
            size_t code_size_in,
            int dim_in)
            : index(index_in),
              codes(codes_in),
              code_size(code_size_in),
              code_bytes((static_cast<size_t>(dim_in) + 7) / 8),
              padded_dim(code_bytes * 8),
              query_plane_len(((code_bytes + 63) / 64) * 64),
              dim(dim_in),
              query_planes(2 * 8 * query_plane_len) {
        if (code_size < code_bytes + 2 * sizeof(float)) {
            throw std::runtime_error("unexpected TQ1+ code stride");
        }
    }

    void set_query(const float* values) {
        query.reset(qdrant_tq_query_new(index, values, static_cast<size_t>(dim)));
        if (!query) throw std::runtime_error(last_error());
        if (!qdrant_tq_query_export_bits1_wide_planes(
                    query.get(),
                    query_planes.data(),
                    query_planes.size(),
                    &postprocess_scale,
                    &bias_correction,
                    &ec_correction,
                    &query_l2)) {
            throw std::runtime_error("failed to export official Qdrant TQ1+ query");
        }
    }

#if defined(__x86_64__)
    __attribute__((target("avx512f,avx512bw,avx512vnni")))
    static int64_t raw_dot_avx512_vnni(
            const int8_t* planes,
            size_t plane_len,
            const uint8_t* code,
            size_t code_len) {
        alignas(16) static constexpr uint8_t codebook_bytes[16] = {
                0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const __m128i codebook_128 = _mm_load_si128(
                reinterpret_cast<const __m128i*>(codebook_bytes));
        const __m512i codebook = _mm512_broadcast_i32x4(codebook_128);
        const __m512i mask = _mm512_set1_epi8(1);
        __m512i low0 = _mm512_setzero_si512();
        __m512i low1 = _mm512_setzero_si512();
        __m512i high0 = _mm512_setzero_si512();
        __m512i high1 = _mm512_setzero_si512();
        alignas(64) uint8_t tail[64] = {};
        const size_t blocks = (code_len + 63) / 64;
        for (size_t block = 0; block < blocks; ++block) {
            const size_t offset = block * 64;
            const size_t remaining = code_len - offset;
            const uint8_t* source = code + offset;
            if (remaining < 64) {
                std::memset(tail, 0, sizeof(tail));
                std::memcpy(tail, source, remaining);
                source = tail;
            }
            __m512i shifted = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(source));
            for (size_t k = 0; k < 8; ++k) {
                const __m512i values = _mm512_shuffle_epi8(
                        codebook, _mm512_and_si512(shifted, mask));
                const __m512i query_low = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(planes + k * plane_len + offset));
                const __m512i query_high = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(planes + (8 + k) * plane_len + offset));
                if ((k & 1) == 0) {
                    low0 = _mm512_dpbusd_epi32(low0, values, query_low);
                    high0 = _mm512_dpbusd_epi32(high0, values, query_high);
                } else {
                    low1 = _mm512_dpbusd_epi32(low1, values, query_low);
                    high1 = _mm512_dpbusd_epi32(high1, values, query_high);
                }
                shifted = _mm512_srli_epi16(shifted, 1);
            }
        }
        const int64_t low = static_cast<int64_t>(
                _mm512_reduce_add_epi32(_mm512_add_epi32(low0, low1)));
        const int64_t high = static_cast<int64_t>(
                _mm512_reduce_add_epi32(_mm512_add_epi32(high0, high1)));
        return low + 256 * high;
    }
#endif

    int64_t raw_dot(const uint8_t* code) const {
#if defined(__x86_64__)
        if (__builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vnni")) {
            return raw_dot_avx512_vnni(
                    query_planes.data(), query_plane_len, code, code_bytes);
        }
#endif
        throw std::runtime_error("validator requires AVX-512 VNNI like the production scorer");
    }

    float score(uint32_t id) const {
        const uint8_t* code = codes + static_cast<size_t>(id) * code_size;
        const int64_t centered_raw = raw_dot(code) - bias_correction;
        const float reconstructed_dot =
                postprocess_scale * static_cast<float>(centered_raw);
        const float dot = reconstructed_dot + ec_correction;
        const float vector_scale = read_float(code + code_bytes);
        return -(dot * vector_scale);
    }

    float official(uint32_t id) const {
        return qdrant_tq_score(index, query.get(), id);
    }

    float symmetric(uint32_t left_id, uint32_t right_id) const {
        const uint8_t* left = codes + static_cast<size_t>(left_id) * code_size;
        const uint8_t* right = codes + static_cast<size_t>(right_id) * code_size;
        int hamming = 0;
        for (size_t byte = 0; byte < code_bytes; ++byte) {
            hamming += __builtin_popcount(
                    static_cast<unsigned>(left[byte] ^ right[byte]));
        }
        constexpr float centroid_sq = 0.7978846f * 0.7978846f;
        const float raw = centroid_sq * static_cast<float>(
                static_cast<int64_t>(padded_dim) - 2LL * hamming);
        return -(raw * read_float(left + code_bytes) * read_float(right + code_bytes));
    }
};

std::vector<uint32_t> top_ids(
        const std::vector<uint32_t>& ids,
        const std::vector<float>& scores,
        size_t top) {
    std::vector<size_t> order(ids.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(
            order.begin(),
            order.begin() + static_cast<std::ptrdiff_t>(top),
            order.end(),
            [&](size_t left, size_t right) {
                if (scores[left] != scores[right]) return scores[left] < scores[right];
                return ids[left] < ids[right];
            });
    std::vector<uint32_t> result;
    result.reserve(top);
    for (size_t i = 0; i < top; ++i) result.push_back(ids[order[i]]);
    return result;
}

size_t set_overlap(const std::vector<uint32_t>& left, const std::vector<uint32_t>& right) {
    std::unordered_set<uint32_t> values(left.begin(), left.end());
    size_t overlap = 0;
    for (uint32_t value : right) overlap += values.count(value);
    return overlap;
}

uint32_t spread_id(uint64_t value, uint32_t rows) {
    return static_cast<uint32_t>((value * 11400714819323198485ull) % rows);
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 8) {
            std::cerr << "usage: " << argv[0]
                      << " BASE.tqdata BASE.meta QUERY.fbin NEW.npy"
                         " [new_rows=4096] [query_rows=16] [candidate_rows_each=4096]\n";
            return 2;
        }
        const std::string code_path = argv[1];
        const std::string meta_path = argv[2];
        const std::string query_path = argv[3];
        const std::string new_path = argv[4];
        const size_t new_rows = argc >= 6 ? std::stoull(argv[5]) : 4096;
        const size_t query_rows = argc >= 7 ? std::stoull(argv[6]) : 16;
        const size_t candidates_each = argc >= 8 ? std::stoull(argv[7]) : 4096;

        IndexPtr index(qdrant_tq_load(code_path.c_str(), meta_path.c_str()));
        if (!index) throw std::runtime_error(last_error());
        const size_t code_size = qdrant_tq_code_size(index.get());
        size_t base_bytes = 0;
        qdrant_tq_codes(index.get(), &base_bytes);
        if (code_size == 0 || base_bytes % code_size != 0) {
            throw std::runtime_error("invalid base TQ code storage");
        }
        const uint32_t base_rows = static_cast<uint32_t>(base_bytes / code_size);

        int dim = 0;
        std::vector<float> workload_queries = read_fbin_prefix(query_path, query_rows, dim);
        std::vector<float> new_vectors = read_npy_prefix(new_path, new_rows, dim);
        if (!qdrant_tq_append(index.get(), new_vectors.data(), new_rows, dim, 8)) {
            throw std::runtime_error(last_error());
        }
        size_t all_bytes = 0;
        const uint8_t* codes = qdrant_tq_codes(index.get(), &all_bytes);
        const uint32_t all_rows = static_cast<uint32_t>(all_bytes / code_size);
        if (!codes || all_rows != base_rows + new_rows) {
            throw std::runtime_error("unexpected appended TQ code count");
        }

        std::vector<uint32_t> candidate_ids;
        candidate_ids.reserve(2 * candidates_each);
        for (size_t i = 0; i < candidates_each; ++i) {
            candidate_ids.push_back(spread_id(i + 1, base_rows));
        }
        for (size_t i = 0; i < std::min(candidates_each, new_rows); ++i) {
            candidate_ids.push_back(base_rows + static_cast<uint32_t>(i));
        }

        std::vector<float> all_queries = workload_queries;
        const size_t new_query_rows = std::min(query_rows, new_rows);
        all_queries.insert(
                all_queries.end(),
                new_vectors.begin(),
                new_vectors.begin() + static_cast<std::ptrdiff_t>(new_query_rows * dim));
        const size_t total_queries = all_queries.size() / static_cast<size_t>(dim);

        DirectScorer scorer(index.get(), codes, code_size, dim);
        ErrorStats asymmetric;
        uint64_t top60_set_hits = 0;
        uint64_t top60_rank_hits = 0;
        for (size_t qi = 0; qi < total_queries; ++qi) {
            scorer.set_query(all_queries.data() + qi * static_cast<size_t>(dim));
            std::vector<float> direct_scores(candidate_ids.size());
            std::vector<float> official_scores(candidate_ids.size());
            for (size_t ci = 0; ci < candidate_ids.size(); ++ci) {
                direct_scores[ci] = scorer.score(candidate_ids[ci]);
                official_scores[ci] = scorer.official(candidate_ids[ci]);
                asymmetric.add(direct_scores[ci], official_scores[ci]);
            }
            const size_t top = std::min<size_t>(60, candidate_ids.size());
            const auto direct_top = top_ids(candidate_ids, direct_scores, top);
            const auto official_top = top_ids(candidate_ids, official_scores, top);
            top60_set_hits += set_overlap(direct_top, official_top);
            for (size_t i = 0; i < top; ++i) {
                top60_rank_hits += direct_top[i] == official_top[i];
            }
        }

        ErrorStats symmetric;
        constexpr size_t pairs_per_group = 50000;
        for (int group = 0; group < 3; ++group) {
            for (size_t sample = 0; sample < pairs_per_group; ++sample) {
                uint32_t left = 0;
                uint32_t right = 0;
                if (group == 0) {
                    left = spread_id(2 * sample + 1, base_rows);
                    right = spread_id(2 * sample + 2, base_rows);
                } else if (group == 1) {
                    left = spread_id(sample + 7, base_rows);
                    right = base_rows + static_cast<uint32_t>((sample * 17 + 3) % new_rows);
                } else {
                    left = base_rows + static_cast<uint32_t>((sample * 13 + 1) % new_rows);
                    right = base_rows + static_cast<uint32_t>((sample * 29 + 2) % new_rows);
                }
                symmetric.add(
                        scorer.symmetric(left, right),
                        qdrant_tq_score_internal(index.get(), left, right));
            }
        }

        uint64_t symmetric_top24_set_hits = 0;
        uint64_t symmetric_top24_rank_hits = 0;
        constexpr size_t anchors = 64;
        const size_t symmetric_candidates = std::min<size_t>(512, candidate_ids.size());
        for (size_t anchor_idx = 0; anchor_idx < anchors; ++anchor_idx) {
            const uint32_t anchor = anchor_idx % 2 == 0
                    ? spread_id(anchor_idx + 101, base_rows)
                    : base_rows + static_cast<uint32_t>(anchor_idx % new_rows);
            std::vector<uint32_t> ids(candidate_ids.begin(), candidate_ids.begin() + symmetric_candidates);
            std::vector<float> direct_scores(ids.size());
            std::vector<float> official_scores(ids.size());
            for (size_t i = 0; i < ids.size(); ++i) {
                direct_scores[i] = scorer.symmetric(anchor, ids[i]);
                official_scores[i] = qdrant_tq_score_internal(index.get(), anchor, ids[i]);
            }
            const size_t top = std::min<size_t>(24, ids.size());
            const auto direct_top = top_ids(ids, direct_scores, top);
            const auto official_top = top_ids(ids, official_scores, top);
            symmetric_top24_set_hits += set_overlap(direct_top, official_top);
            for (size_t i = 0; i < top; ++i) {
                symmetric_top24_rank_hits += direct_top[i] == official_top[i];
            }
        }

        const uint64_t top60_total = total_queries * std::min<size_t>(60, candidate_ids.size());
        const uint64_t top24_total = anchors * std::min<size_t>(24, symmetric_candidates);
        std::cout << std::setprecision(10)
                  << "base_rows=" << base_rows
                  << " appended_v2_rows=" << new_rows
                  << " dim=" << dim
                  << " code_size=" << code_size << '\n'
                  << "asymmetric_comparisons=" << asymmetric.comparisons
                  << " max_abs_error=" << asymmetric.max_abs
                  << " mean_abs_error=" << asymmetric.mean_abs()
                  << " failures=" << asymmetric.failures << '\n'
                  << "asymmetric_top60_set_match=" << top60_set_hits << '/' << top60_total
                  << " asymmetric_top60_rank_match=" << top60_rank_hits << '/' << top60_total << '\n'
                  << "symmetric_comparisons=" << symmetric.comparisons
                  << " max_abs_error=" << symmetric.max_abs
                  << " mean_abs_error=" << symmetric.mean_abs()
                  << " failures=" << symmetric.failures << '\n'
                  << "symmetric_top24_set_match=" << symmetric_top24_set_hits << '/' << top24_total
                  << " symmetric_top24_rank_match=" << symmetric_top24_rank_hits << '/' << top24_total << '\n';

        const bool passed = asymmetric.failures == 0 && symmetric.failures == 0 &&
                top60_set_hits == top60_total && top60_rank_hits == top60_total &&
                symmetric_top24_set_hits == top24_total && symmetric_top24_rank_hits == top24_total;
        std::cout << "RESULT=" << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
