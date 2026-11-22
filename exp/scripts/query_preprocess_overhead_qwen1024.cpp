#include <faiss/IndexPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <sched.h>

#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "qdrant_tq_ffi.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kDim = 1024;
constexpr size_t kMrqBits = 320;
constexpr size_t kMrqClusters = 4096;
constexpr int kMrqQueryBits = 4;
constexpr size_t kPqM = 256;
constexpr size_t kPqKsub = 256;

volatile double g_sink = 0.0;

struct Matrix {
    size_t rows = 0;
    size_t dim = 0;
    std::vector<float> values;

    const float* row(size_t i) const { return values.data() + i * dim; }
};

struct Args {
    std::string dataset;
    std::string query;
    std::string mrq_pca;
    std::string mrq_mean;
    std::string mrq_rotation;
    std::string mrq_centroids;
    std::string opq_index;
    std::string tq_data;
    std::string tq_meta;
    std::string output;
    int cpu = 20;
    int rounds = 9;
    size_t fixed_iterations = 6000;
    size_t mrq_iterations = 512;
    bool tq_only = false;
};

struct Stats {
    double median_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    std::vector<double> samples;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + key);
            return argv[i];
        };
        if (key == "--dataset") args.dataset = next();
        else if (key == "--query") args.query = next();
        else if (key == "--mrq-pca") args.mrq_pca = next();
        else if (key == "--mrq-mean") args.mrq_mean = next();
        else if (key == "--mrq-rotation") args.mrq_rotation = next();
        else if (key == "--mrq-centroids") args.mrq_centroids = next();
        else if (key == "--opq-index") args.opq_index = next();
        else if (key == "--tq-data") args.tq_data = next();
        else if (key == "--tq-meta") args.tq_meta = next();
        else if (key == "--output") args.output = next();
        else if (key == "--cpu") args.cpu = std::stoi(next());
        else if (key == "--rounds") args.rounds = std::stoi(next());
        else if (key == "--fixed-iterations") args.fixed_iterations = std::stoull(next());
        else if (key == "--mrq-iterations") args.mrq_iterations = std::stoull(next());
        else if (key == "--tq-only") args.tq_only = true;
        else throw std::runtime_error("unknown argument: " + key);
    }
    if (args.dataset.empty() || args.query.empty() || args.mrq_pca.empty() ||
        args.mrq_mean.empty() || args.mrq_rotation.empty() ||
        args.mrq_centroids.empty() || args.opq_index.empty() ||
        args.tq_data.empty() || args.tq_meta.empty() || args.output.empty()) {
        throw std::runtime_error(
            "required: --dataset --query --mrq-pca --mrq-mean --mrq-rotation "
            "--mrq-centroids --opq-index --tq-data --tq-meta --output"
        );
    }
    if (args.rounds < 3 || args.fixed_iterations == 0 || args.mrq_iterations == 0) {
        throw std::runtime_error("invalid benchmark settings");
    }
    return args;
}

void pin_cpu(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        throw std::runtime_error("sched_setaffinity failed");
    }
}

Matrix read_fbin(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open fbin: " + path);
    int32_t rows = 0;
    int32_t dim = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!input || rows <= 0 || dim <= 0) throw std::runtime_error("bad fbin header");
    Matrix matrix;
    matrix.rows = static_cast<size_t>(rows);
    matrix.dim = static_cast<size_t>(dim);
    matrix.values.resize(matrix.rows * matrix.dim);
    input.read(
        reinterpret_cast<char*>(matrix.values.data()),
        static_cast<std::streamsize>(matrix.values.size() * sizeof(float))
    );
    if (!input) throw std::runtime_error("short fbin payload: " + path);
    return matrix;
}

Matrix read_fvecs(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open fvecs: " + path);
    const std::streamsize bytes = input.tellg();
    input.seekg(0);
    int32_t dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!input || dim <= 0) throw std::runtime_error("bad fvecs header: " + path);
    const size_t row_bytes = (static_cast<size_t>(dim) + 1) * sizeof(float);
    if (bytes <= 0 || static_cast<size_t>(bytes) % row_bytes != 0) {
        throw std::runtime_error("bad fvecs size: " + path);
    }
    Matrix matrix;
    matrix.rows = static_cast<size_t>(bytes) / row_bytes;
    matrix.dim = static_cast<size_t>(dim);
    matrix.values.resize(matrix.rows * matrix.dim);
    input.seekg(0);
    for (size_t i = 0; i < matrix.rows; ++i) {
        int32_t row_dim = 0;
        input.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        if (row_dim != dim) throw std::runtime_error("ragged fvecs: " + path);
        input.read(
            reinterpret_cast<char*>(matrix.values.data() + i * matrix.dim),
            static_cast<std::streamsize>(matrix.dim * sizeof(float))
        );
    }
    if (!input) throw std::runtime_error("short fvecs payload: " + path);
    return matrix;
}

template <typename Function>
Stats benchmark(size_t iterations, int rounds, Function&& function) {
    const size_t warmup = std::min<size_t>(iterations, 256);
    for (size_t i = 0; i < warmup; ++i) function(i);

    Stats stats;
    stats.samples.reserve(static_cast<size_t>(rounds));
    for (int round = 0; round < rounds; ++round) {
        const auto start = Clock::now();
        for (size_t i = 0; i < iterations; ++i) function(i);
        const auto stop = Clock::now();
        const double elapsed_us =
            std::chrono::duration<double, std::micro>(stop - start).count();
        stats.samples.push_back(elapsed_us / static_cast<double>(iterations));
    }
    std::vector<double> sorted = stats.samples;
    std::sort(sorted.begin(), sorted.end());
    stats.median_us = sorted[sorted.size() / 2];
    stats.min_us = sorted.front();
    stats.max_us = sorted.back();
    return stats;
}

class AdcBqQueryPrep {
   public:
    explicit AdcBqQueryPrep(size_t dim)
        : dim_(dim),
          code_size_((dim + 7) / 8),
          lut_i16_(code_size_ * 32),
          lut_hacc_(code_size_ * 64) {}

    void set_query(const float* query) {
        const float scale = int16_lut_scale(query);

        const size_t codebooks = (dim_ + 3) / 4;
        for (size_t codebook = 0; codebook < codebooks; ++codebook) {
            const size_t base = codebook * 4;
            float delta[4] = {};
            float base_score = 0.0f;
            for (size_t bit = 0; bit < 4 && base + bit < dim_; ++bit) {
                const float value = query[base + bit];
                base_score -= value;
                delta[bit] = 2.0f * value;
            }
            float row[16];
            row[0] = base_score;
            row[1] = row[0] + delta[3];
            row[2] = row[0] + delta[2];
            row[3] = row[2] + delta[3];
            row[4] = row[0] + delta[1];
            row[5] = row[4] + delta[3];
            row[6] = row[4] + delta[2];
            row[7] = row[6] + delta[3];
            row[8] = row[0] + delta[0];
            row[9] = row[8] + delta[3];
            row[10] = row[8] + delta[2];
            row[11] = row[10] + delta[3];
            row[12] = row[8] + delta[1];
            row[13] = row[12] + delta[3];
            row[14] = row[12] + delta[2];
            row[15] = row[14] + delta[3];
            int16_t* output = lut_i16_.data() + codebook * 16;
            for (size_t entry = 0; entry < 16; ++entry) {
                const float scaled = std::round(row[entry] * scale);
                output[entry] = static_cast<int16_t>(
                    std::max(-32767.0f, std::min(32767.0f, scaled))
                );
            }
        }

        for (size_t codebook = 0; codebook < codebooks; ++codebook) {
            const size_t group = codebook / 4;
            const size_t lane = codebook % 4;
            uint8_t* low = lut_hacc_.data() + group * 128 + lane * 16;
            uint8_t* high = low + 64;
            const int16_t* row = lut_i16_.data() + codebook * 16;
            for (size_t entry = 0; entry < 16; ++entry) {
                const uint16_t biased = static_cast<uint16_t>(
                    static_cast<int32_t>(row[entry]) + 32768
                );
                low[entry] = static_cast<uint8_t>(biased & 0xffu);
                high[entry] = static_cast<uint8_t>(biased >> 8);
            }
        }
        g_sink += lut_hacc_[17];
    }

    void set_calibrated_query(const float* query) {
        set_query(query);
        // CalibratedThreshold converts tau to the quantized distance width
        // with a second scale scan after set_query().
        calibration_scale_scan(query);
    }

    void calibration_scale_scan(const float* query) {
        g_sink += int16_lut_scale(query);
    }

   private:
    float int16_lut_scale(const float* query) const {
        float max_abs = 0.0f;
        for (size_t base = 0; base < dim_; base += 4) {
            float sum = 0.0f;
            for (size_t bit = 0; bit < 4 && base + bit < dim_; ++bit) {
                sum += std::abs(query[base + bit]);
            }
            max_abs = std::max(max_abs, sum);
        }
        return max_abs > 0.0f ? 32767.0f / max_abs : 1.0f;
    }

    size_t dim_;
    size_t code_size_;
    std::vector<int16_t> lut_i16_;
    std::vector<uint8_t> lut_hacc_;
};

void matrix_row_multiply(
    const float* vector,
    const float* matrix,
    size_t input_dim,
    size_t output_dim,
    float* output
) {
    // Match the free RESQ mul(A, B) used by res_search.cpp: zero the result,
    // then traverse the input dimension outside and the contiguous output
    // dimension inside.  The inner loop is the production omp-simd loop.
    std::fill(output, output + output_dim, 0.0f);
    for (size_t row = 0; row < input_dim; ++row) {
        const float value = vector[row];
        const float* matrix_row = matrix + row * output_dim;
#pragma omp simd
        for (size_t column = 0; column < output_dim; ++column) {
            output[column] += value * matrix_row[column];
        }
    }
}

void mrq_transform(
    const float* query,
    const Matrix& pca,
    const Matrix& mean,
    const Matrix& rotation,
    float* projected,
    float* randomized
) {
    matrix_row_multiply(query, pca.values.data(), kDim, kDim, projected);
    for (size_t i = 0; i < kDim; ++i) projected[i] -= mean.values[i];
    matrix_row_multiply(
        projected, rotation.values.data(), kMrqBits, kMrqBits, randomized
    );
}

void mrq_pack_lut(const uint8_t* byte_query, uint8_t* lut) {
    static constexpr uint8_t positions[16] = {
        3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3,
    };
    for (size_t block = 0; block < kMrqBits / 4; ++block) {
        lut[0] = 0;
        for (int code = 1; code < 16; ++code) {
            const int low = code & -code;
            lut[code] = static_cast<uint8_t>(
                lut[code - low] + byte_query[positions[code]]
            );
        }
        lut += 16;
        byte_query += 4;
    }
}

float mrq_sqr_dist_320(const float* lhs, const float* rhs) {
    __m256 sum = _mm256_setzero_ps();
    for (size_t i = 0; i < kMrqBits; i += 16) {
        __m256 left = _mm256_loadu_ps(lhs + i);
        __m256 right = _mm256_loadu_ps(rhs + i);
        __m256 diff = _mm256_sub_ps(left, right);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        left = _mm256_loadu_ps(lhs + i + 8);
        right = _mm256_loadu_ps(rhs + i + 8);
        diff = _mm256_sub_ps(left, right);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    return std::accumulate(lanes, lanes + 8, 0.0f);
}

class MrqRoutingPrep {
   public:
    explicit MrqRoutingPrep(const Matrix& centroids) : centroids_(centroids) {
        if (centroids_.rows != kMrqClusters || centroids_.dim != kMrqBits) {
            throw std::runtime_error("MRQ centroid shape mismatch");
        }
    }

    void route(const float* randomized_query, size_t probes) const {
        using Result = std::pair<float, uint32_t>;
        std::vector<Result> centroid_dist(centroids_.rows);
        for (size_t cluster = 0; cluster < centroids_.rows; ++cluster) {
            centroid_dist[cluster].first = mrq_sqr_dist_320(
                randomized_query, centroids_.row(cluster)
            );
            centroid_dist[cluster].second = static_cast<uint32_t>(cluster);
        }
        const size_t take = std::min(probes, centroid_dist.size());
        std::partial_sort(
            centroid_dist.begin(),
            centroid_dist.begin() + static_cast<std::ptrdiff_t>(take),
            centroid_dist.end()
        );
        g_sink += centroid_dist[take - 1].first + centroid_dist[0].second;
    }

   private:
    const Matrix& centroids_;
};

class MrqProbePrep {
   public:
    MrqProbePrep(const Matrix& centroids, std::vector<float> dither)
        : centroids_(centroids), dither_(std::move(dither)) {
        if (centroids_.rows != kMrqClusters || centroids_.dim != kMrqBits) {
            throw std::runtime_error("MRQ centroid shape mismatch");
        }
        if (dither_.size() != kMrqBits) throw std::runtime_error("bad dither");
    }

    void prepare(const float* randomized_query, size_t query_id, size_t probes) {
        for (size_t probe = 0; probe < probes; ++probe) {
            const size_t cluster =
                (query_id * 1315423911ull + probe * 2654435761ull) % centroids_.rows;
            const float* centroid = centroids_.row(cluster);
            float low = std::numeric_limits<float>::infinity();
            float high = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < kMrqBits; ++i) {
                const float residual = randomized_query[i] - centroid[i];
                low = std::min(low, residual);
                high = std::max(high, residual);
            }
            const float width = (high - low) / ((1 << kMrqQueryBits) - 1);
            const float inverse_width = width > 0.0f ? 1.0f / width : 0.0f;
            uint64_t sum = 0;
            for (size_t i = 0; i < kMrqBits; ++i) {
                const float residual = randomized_query[i] - centroid[i];
                const uint8_t quantized = static_cast<uint8_t>(
                    (residual - low) * inverse_width + dither_[i]
                );
                byte_query_[i] = quantized;
                sum += quantized;
            }
            mrq_pack_lut(byte_query_.data(), lut_.data());
            // fast_scan/res_scan scale every freshly built LUT in place.
            for (uint8_t& entry : lut_) entry = static_cast<uint8_t>(entry * 2);
            g_sink += static_cast<double>(sum + lut_[probe % lut_.size()]);
        }
    }

   private:
    const Matrix& centroids_;
    std::vector<float> dither_;
    alignas(64) std::array<uint8_t, kMrqBits> byte_query_{};
    alignas(64) std::array<uint8_t, (kMrqBits / 4) * 16> lut_{};
};

void write_row(
    std::ofstream& output,
    const Args& args,
    const std::string& method,
    const std::string& stage,
    size_t parameter,
    size_t output_bytes,
    const std::string& notes,
    const Stats& stats
) {
    output << args.dataset << ',' << method << ',' << stage << ',' << parameter << ','
           << std::setprecision(10) << stats.median_us << ',' << stats.min_us << ','
           << stats.max_us << ',' << output_bytes << ',' << args.cpu << ','
           << args.rounds << ',' << notes << '\n';
    output.flush();
    std::cout << method << " / " << stage;
    if (parameter) std::cout << " / " << parameter;
    std::cout << ": " << std::fixed << std::setprecision(3)
              << stats.median_us << " us/query\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        pin_cpu(args.cpu);

        const Matrix queries = read_fbin(args.query);
        const Matrix pca = read_fvecs(args.mrq_pca);
        const Matrix mean = read_fvecs(args.mrq_mean);
        const Matrix rotation = read_fvecs(args.mrq_rotation);
        const Matrix mrq_centroids = read_fvecs(args.mrq_centroids);
        if (queries.dim != kDim || pca.rows != kDim || pca.dim != kDim ||
            mean.rows < 1 || mean.dim != kDim || rotation.rows != kMrqBits ||
            rotation.dim != kMrqBits) {
            throw std::runtime_error("input shape mismatch");
        }

        std::ofstream output(args.output);
        if (!output) throw std::runtime_error("cannot create output: " + args.output);
        output << "dataset,method,stage,parameter,median_us_per_query,min_us_per_query,"
                  "max_us_per_query,output_bytes,cpu,rounds,notes\n";

        const size_t nq = queries.rows;
        const size_t fixed_iterations = std::max(args.fixed_iterations, nq);

        if (!args.tq_only) {
        AdcBqQueryPrep adc(kDim);
        write_row(
            output,
            args,
            "ADC-BQ int16 nibble",
            "lut_build_pack",
            0,
            8192,
            "production set_query: scale+int16 4D LUT+hacc byte pack",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                adc.set_query(queries.row(i % nq));
            })
        );
        write_row(
            output,
            args,
            "ADC-BQ int16 nibble",
            "calibration_tau_scale_scan",
            0,
            0,
            "second int16_lut_scale scan used to convert calibrated tau to distance width",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                adc.calibration_scale_scan(queries.row(i % nq));
            })
        );
        write_row(
            output,
            args,
            "ADC-BQ int16 nibble",
            "method_total",
            0,
            8192,
            "calibrated production: set_query scale+int16 4D LUT+hacc pack+second tau-width scale scan; excludes common normalization",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                adc.set_calibrated_query(queries.row(i % nq));
            })
        );

        faiss::ProductQuantizer pq(kDim, kPqM, 8);
        for (size_t sub = 0; sub < kPqM; ++sub) {
            for (size_t code = 0; code < kPqKsub; ++code) {
                const float* source = queries.row(code % nq) + sub * 4;
                float* centroid = pq.get_centroids(sub, code);
                for (size_t j = 0; j < 4; ++j) {
                    centroid[j] = source[j] + static_cast<float>((code + j) % 7) * 1e-4f;
                }
            }
        }
        std::vector<float> pq_lut(kPqM * kPqKsub);
        write_row(
            output,
            args,
            "PQ M256x8 dsub4",
            "method_total",
            0,
            pq_lut.size() * sizeof(float),
            "Faiss ProductQuantizer::compute_inner_prod_table; 256 4D subquantizers x 256 centroids",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                pq.compute_inner_prod_table(queries.row(i % nq), pq_lut.data());
                g_sink += pq_lut[(i * 17) % pq_lut.size()];
            })
        );

        // Match the production Faiss codec configurations used by
        // faiss_exact_matched_budget_compare.cpp.  SQ query preparation is
        // deliberately just FlatCodesDistanceComputer::set_query(): SQ4
        // decodes each database code while scoring and builds no query LUT.
        faiss::IndexScalarQuantizer sq4(
            kDim, faiss::ScalarQuantizer::QT_4bit, faiss::METRIC_INNER_PRODUCT
        );
        sq4.train(static_cast<faiss::idx_t>(nq), queries.values.data());
        std::unique_ptr<faiss::FlatCodesDistanceComputer> sq4_dc(
            sq4.get_FlatCodesDistanceComputer()
        );
        const size_t sq_iterations = std::max<size_t>(1000000, fixed_iterations * 100);
        write_row(
            output,
            args,
            "SQ4",
            "method_total",
            0,
            0,
            "Faiss IndexScalarQuantizer FlatCodesDistanceComputer::set_query pointer bind; no query LUT",
            benchmark(sq_iterations, args.rounds, [&](size_t i) {
                sq4_dc->set_query(queries.row(i % nq));
                asm volatile("" ::: "memory");
            })
        );

        // Load the exact production OPQ8-M128 index rather than synthesizing
        // a rotation or codebook.  The trained MSMARCO transform is also used
        // for DBpedia because this experiment measures only compute cost; all
        // inputs have the same dimension and the values do not change the
        // dense rotation/LUT operation count.
        std::unique_ptr<faiss::Index> opq_owned(faiss::read_index(args.opq_index.c_str()));
        auto* opq = dynamic_cast<faiss::IndexPreTransform*>(opq_owned.get());
        auto* pq128 = opq ? dynamic_cast<faiss::IndexPQ*>(opq->index) : nullptr;
        if (!opq || !pq128 || opq->chain.size() != 1 || opq->d != kDim ||
            pq128->pq.M != 128 || pq128->pq.nbits != 8) {
            throw std::runtime_error("unexpected OPQ8-M128 index structure");
        }
        std::unique_ptr<faiss::FlatCodesDistanceComputer> pq128_dc(
            pq128->get_FlatCodesDistanceComputer()
        );
        write_row(
            output,
            args,
            "PQ M128x8 dsub8",
            "method_total",
            0,
            128 * 256 * sizeof(float),
            "Faiss IndexPQ production set_query using trained M128x8 codebook from OPQ index; rotation excluded",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                pq128_dc->set_query(queries.row(i % nq));
                asm volatile("" ::: "memory");
            })
        );

        write_row(
            output,
            args,
            "OPQ M128x8",
            "opq_rotation",
            0,
            kDim * sizeof(float),
            "Faiss IndexPreTransform::apply_chain using trained production OPQMatrix",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                const float* query = queries.row(i % nq);
                const float* transformed = opq->apply_chain(1, query);
                g_sink += transformed[(i * 29) % kDim];
                if (transformed != query) delete[] transformed;
            })
        );
        std::vector<float> opq_transformed_queries(nq * kDim);
        for (size_t query_id = 0; query_id < nq; ++query_id) {
            const float* query = queries.row(query_id);
            const float* transformed = opq->apply_chain(1, query);
            std::memcpy(
                opq_transformed_queries.data() + query_id * kDim,
                transformed,
                kDim * sizeof(float)
            );
            if (transformed != query) delete[] transformed;
        }
        std::unique_ptr<faiss::FlatCodesDistanceComputer> opq_lut_dc(
            pq128->get_FlatCodesDistanceComputer()
        );
        write_row(
            output,
            args,
            "OPQ M128x8",
            "pq_lut_build_isolated",
            0,
            128 * 256 * sizeof(float),
            "Faiss M128x8 PQ set_query on pre-rotated production OPQ queries; rotation excluded",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                opq_lut_dc->set_query(
                    opq_transformed_queries.data() + (i % nq) * kDim
                );
                asm volatile("" ::: "memory");
            })
        );
        std::unique_ptr<faiss::FlatCodesDistanceComputer> opq_pq_dc(
            pq128->get_FlatCodesDistanceComputer()
        );
        write_row(
            output,
            args,
            "OPQ M128x8",
            "method_total",
            0,
            kDim * sizeof(float) + 128 * 256 * sizeof(float),
            "production apply_chain allocation+trained OPQ rotation+Faiss M128x8 PQ LUT set_query",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                const float* query = queries.row(i % nq);
                const float* transformed = opq->apply_chain(1, query);
                opq_pq_dc->set_query(transformed);
                asm volatile("" ::: "memory");
                if (transformed != query) delete[] transformed;
            })
        );
        }

        // Canonical Qdrant TurboQuant query path: TQ+ Bits1 with padded
        // Hadamard rotation and its widened 16-bit asymmetric query.  Build a
        // small timing-only model from the workload's query set when no cache
        // exists.  Model fitting/encoding is outside the timed region.
        QdrantTqIndex* tq_index = nullptr;
        {
            std::ifstream tq_data_input(args.tq_data, std::ios::binary);
            std::ifstream tq_meta_input(args.tq_meta);
            if (tq_data_input.good() && tq_meta_input.good()) {
                tq_index = qdrant_tq_load(args.tq_data.c_str(), args.tq_meta.c_str());
            } else {
                tq_index = qdrant_tq_build(
                    queries.values.data(),
                    nq,
                    kDim,
                    0,
                    1,
                    args.tq_data.c_str(),
                    args.tq_meta.c_str()
                );
            }
        }
        if (!tq_index) {
            throw std::runtime_error(
                std::string("Qdrant TurboQuant setup failed: ") + qdrant_tq_last_error()
            );
        }
        write_row(
            output,
            args,
            "TurboQuant TQ+ 1b",
            "encode_query_only",
            0,
            0,
            "Qdrant official encode_query plus query object allocation/free; scorer plane export excluded",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                QdrantTqQuery* tq_query = qdrant_tq_query_new(
                    tq_index, queries.row(i % nq), kDim
                );
                if (!tq_query) {
                    throw std::runtime_error(qdrant_tq_last_error());
                }
                asm volatile("" : : "r"(tq_query) : "memory");
                qdrant_tq_query_free(tq_query);
            })
        );
        const size_t tq_code_bytes = (kDim + 7) / 8;
        const size_t tq_query_plane_len = ((tq_code_bytes + 63) / 64) * 64;
        std::vector<int8_t> tq_query_planes(2 * 8 * tq_query_plane_len);
        float tq_postprocess_scale = 0.0f;
        int64_t tq_bias_correction = 0;
        float tq_ec_correction = 0.0f;
        float tq_query_l2 = 0.0f;
        write_row(
            output,
            args,
            "TurboQuant TQ+ 1b",
            "method_total",
            0,
            tq_query_planes.size(),
            "production set_query: Qdrant encode_query padded Hadamard+TQ+ correction then export_bits1_wide_planes for the AVX-512 scorer; includes object allocation/free",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                QdrantTqQuery* tq_query = qdrant_tq_query_new(
                    tq_index, queries.row(i % nq), kDim
                );
                if (!tq_query) {
                    throw std::runtime_error(qdrant_tq_last_error());
                }
                if (!qdrant_tq_query_export_bits1_wide_planes(
                        tq_query,
                        tq_query_planes.data(),
                        tq_query_planes.size(),
                        &tq_postprocess_scale,
                        &tq_bias_correction,
                        &tq_ec_correction,
                        &tq_query_l2)) {
                    qdrant_tq_query_free(tq_query);
                    throw std::runtime_error(
                        "failed to export Qdrant TQ1+ wide query planes"
                    );
                }
                g_sink += tq_postprocess_scale + tq_ec_correction + tq_query_l2 +
                    static_cast<double>(tq_bias_correction & 1) +
                    tq_query_planes[i % tq_query_planes.size()];
                qdrant_tq_query_free(tq_query);
            })
        );
        qdrant_tq_free(tq_index);

        if (args.tq_only) {
            std::cerr << "checksum=" << std::setprecision(17) << g_sink << '\n';
            return 0;
        }

        const size_t padded_dim = 1024;
        std::unique_ptr<rabitqlib::Rotator<float>> rotator(
            rabitqlib::choose_rotator<float>(
                kDim, rabitqlib::RotatorType::FhtKacRotator, padded_dim
            )
        );
        const auto query_config = rabitqlib::quant::faster_config(
            padded_dim, rabitqlib::SplitSingleQuery<float>::kNumBits
        );
        std::vector<float> rotated_queries(nq * padded_dim);
        for (size_t i = 0; i < nq; ++i) {
            rotator->rotate(queries.row(i), rotated_queries.data() + i * padded_dim);
        }
        std::vector<float> rq_centroids(16 * padded_dim);
        for (size_t cluster = 0; cluster < 16; ++cluster) {
            std::memcpy(
                rq_centroids.data() + cluster * padded_dim,
                rotated_queries.data() + (cluster % nq) * padded_dim,
                padded_dim * sizeof(float)
            );
        }
        std::vector<float> rotated_scratch(padded_dim);
        const Stats rq_rotate = benchmark(
            fixed_iterations, args.rounds, [&](size_t i) {
                rotator->rotate(queries.row(i % nq), rotated_scratch.data());
                g_sink += rotated_scratch[(i * 13) % padded_dim];
            }
        );
        write_row(
            output, args, "RaBitQ-1b C16", "fht_rotation", 0,
            padded_dim * sizeof(float), "production FhtKacRotator", rq_rotate
        );
        const Stats rq_quant = benchmark(
            fixed_iterations, args.rounds, [&](size_t i) {
                const float* rotated = rotated_queries.data() + (i % nq) * padded_dim;
                rabitqlib::SplitSingleQuery<float> wrapper(
                    rotated, padded_dim, 0, query_config, rabitqlib::METRIC_IP
                );
                g_sink += wrapper.delta() + wrapper.k1xsumq();
            }
        );
        write_row(
            output, args, "RaBitQ-1b C16", "query_quantize_pack", 0, 512,
            "production SplitSingleQuery 4-bit query quantization+bit-plane pack", rq_quant
        );
        auto rq_metadata_fn = [&](const float* rotated) {
            std::vector<float> q_to_centroids(16);
            q_to_centroids.resize(32);
            for (size_t cluster = 0; cluster < 16; ++cluster) {
                const float* centroid = rq_centroids.data() + cluster * padded_dim;
                float ip = 0.0f;
                float l2 = 0.0f;
                for (size_t j = 0; j < padded_dim; ++j) ip += rotated[j] * centroid[j];
                for (size_t j = 0; j < padded_dim; ++j) {
                    const float diff = rotated[j] - centroid[j];
                    l2 += diff * diff;
                }
                q_to_centroids[cluster] = ip;
                q_to_centroids[cluster + 16] = std::sqrt(l2);
            }
            g_sink += q_to_centroids[3] + q_to_centroids[19];
        };
        const Stats rq_metadata = benchmark(
            fixed_iterations, args.rounds, [&](size_t i) {
                rq_metadata_fn(rotated_queries.data() + (i % nq) * padded_dim);
            }
        );
        write_row(
            output, args, "RaBitQ-1b C16", "c16_metadata", 16, 128,
            "production IP path: dot+L2-to-centroid for 16 centroids", rq_metadata
        );
        write_row(
            output,
            args,
            "RaBitQ-1b C16",
            "method_total",
            0,
            padded_dim * sizeof(float) + 512 + 128,
            "FHT rotation+SplitSingleQuery+C16 metadata and production temporary allocations",
            benchmark(fixed_iterations, args.rounds, [&](size_t i) {
                const size_t query_id = i % nq;
                std::vector<float> query_rotated(padded_dim);
                rotator->rotate(queries.row(query_id), query_rotated.data());
                rabitqlib::SplitSingleQuery<float> wrapper(
                    query_rotated.data(),
                    padded_dim,
                    0,
                    query_config,
                    rabitqlib::METRIC_IP
                );
                rq_metadata_fn(query_rotated.data());
                g_sink += wrapper.delta();
            })
        );

        std::vector<float> projected_scratch(kDim);
        std::vector<float> randomized_scratch(kMrqBits);
        const Stats mrq_transform_stats = benchmark(
            fixed_iterations, args.rounds, [&](size_t i) {
                mrq_transform(
                    queries.row(i % nq),
                    pca,
                    mean,
                    rotation,
                    projected_scratch.data(),
                    randomized_scratch.data()
                );
                g_sink += projected_scratch[(i * 7) % kDim]
                    + randomized_scratch[(i * 11) % kMrqBits];
            }
        );
        write_row(
            output,
            args,
            "MRQ B320 C4096",
            "dense_transform",
            0,
            (kDim + kMrqBits) * sizeof(float),
            "RESQ production contiguous SIMD loop: 1024x1024 PCA+mean subtraction+320x320 rotation",
            mrq_transform_stats
        );

        std::vector<float> projected_queries(nq * kDim);
        std::vector<float> randomized_queries(nq * kMrqBits);
        for (size_t i = 0; i < nq; ++i) {
            mrq_transform(
                queries.row(i),
                pca,
                mean,
                rotation,
                projected_queries.data() + i * kDim,
                randomized_queries.data() + i * kMrqBits
            );
        }
        std::vector<float> dither(kMrqBits);
        for (size_t i = 0; i < kMrqBits; ++i) {
            dither[i] = static_cast<float>((i * 2654435761ull) & 0xffffu) / 65536.0f;
        }
        MrqProbePrep mrq_probe(mrq_centroids, std::move(dither));
        MrqRoutingPrep mrq_routing(mrq_centroids);
        const std::array<size_t, 11> probes = {
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
        };
        for (const size_t probe_count : probes) {
            const Stats routing_stats = benchmark(
                args.mrq_iterations, args.rounds, [&](size_t i) {
                    const size_t query_id = i % nq;
                    mrq_routing.route(
                        randomized_queries.data() + query_id * kMrqBits,
                        probe_count
                    );
                }
            );
            write_row(
                output,
                args,
                "MRQ B320 C4096",
                "ivf_centroid_routing",
                probe_count,
                kMrqClusters * kMrqBits * sizeof(float),
                "production C4096 B320 L2 centroid scan+partial_sort(nprobe); includes temporary vector allocation",
                routing_stats
            );
            const Stats probe_stats = benchmark(
                args.mrq_iterations, args.rounds, [&](size_t i) {
                    const size_t query_id = i % nq;
                    mrq_probe.prepare(
                        randomized_queries.data() + query_id * kMrqBits,
                        query_id,
                        probe_count
                    );
                }
            );
            write_row(
                output,
                args,
                "MRQ B320 C4096",
                "probe_residual_luts",
                probe_count,
                probe_count * (kMrqBits / 4) * 16,
                "per-probe range+4-bit residual quantization+1280-byte LUT+scanner x2 scaling; excludes IVF centroid routing",
                probe_stats
            );
        }

        std::cerr << "checksum=" << std::setprecision(17) << g_sink << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
