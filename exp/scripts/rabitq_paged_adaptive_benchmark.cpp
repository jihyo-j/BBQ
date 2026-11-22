#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef O_DIRECT
#define O_DIRECT 0
#endif

#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string dataset;
    std::string train;
    std::string query;
    std::string gt;
    std::string index;
    std::string paged_index;
    std::string fp_path;
    std::string out;
    int threads = 24;
    size_t cache_mib = 1;
    uint64_t cache_bytes = 0;
    int rounds = 3;
    int k = 10;
    size_t max_queries = 0;
    bool rebuild_paged_index = false;
    bool prepare_only = false;
    bool code_only = false;
    bool index_page_async = false;
    unsigned index_page_qd = 48;
    int colocation_iterations = 0;
    int colocation_partition_size = 64;
    std::vector<int> efs{64, 128, 256, 512, 1024};
};

std::string need_value(int argc, char** argv, int& i, const char* name) {
    if (++i >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
    return argv[i];
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--dataset") args.dataset = need_value(argc, argv, i, "--dataset");
        else if (option == "--train") args.train = need_value(argc, argv, i, "--train");
        else if (option == "--query") args.query = need_value(argc, argv, i, "--query");
        else if (option == "--gt") args.gt = need_value(argc, argv, i, "--gt");
        else if (option == "--index") args.index = need_value(argc, argv, i, "--index");
        else if (option == "--paged-index") args.paged_index = need_value(argc, argv, i, "--paged-index");
        else if (option == "--fp-path") args.fp_path = need_value(argc, argv, i, "--fp-path");
        else if (option == "--out") args.out = need_value(argc, argv, i, "--out");
        else if (option == "--threads") args.threads = std::stoi(need_value(argc, argv, i, "--threads"));
        else if (option == "--cache-mib") args.cache_mib = std::stoull(need_value(argc, argv, i, "--cache-mib"));
        else if (option == "--cache-bytes") args.cache_bytes = std::stoull(need_value(argc, argv, i, "--cache-bytes"));
        else if (option == "--rounds") args.rounds = std::stoi(need_value(argc, argv, i, "--rounds"));
        else if (option == "--k") args.k = std::stoi(need_value(argc, argv, i, "--k"));
        else if (option == "--max-queries") args.max_queries = std::stoull(need_value(argc, argv, i, "--max-queries"));
        else if (option == "--colocation-iterations") args.colocation_iterations = std::stoi(need_value(argc, argv, i, "--colocation-iterations"));
        else if (option == "--colocation-partition-size") args.colocation_partition_size = std::stoi(need_value(argc, argv, i, "--colocation-partition-size"));
        else if (option == "--rebuild-paged-index") args.rebuild_paged_index = true;
        else if (option == "--prepare-only") args.prepare_only = true;
        else if (option == "--code-only") args.code_only = true;
        else if (option == "--index-page-async") args.index_page_async = true;
        else if (option == "--index-page-qd") args.index_page_qd = static_cast<unsigned>(
            std::stoul(need_value(argc, argv, i, "--index-page-qd"))
        );
        else if (option == "--efs") {
            args.efs.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                args.efs.push_back(std::stoi(argv[++i]));
            }
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    if (
        args.dataset.empty() || args.train.empty() || args.query.empty() ||
        args.gt.empty() || args.index.empty() || args.paged_index.empty() ||
        args.out.empty() || args.threads <= 0 ||
        (!args.code_only && args.cache_mib == 0) ||
        (args.code_only && args.cache_bytes == 0) ||
        args.rounds <= 0 || args.k <= 0 || args.efs.empty() ||
        args.index_page_qd == 0 ||
        args.colocation_iterations < 0 || args.colocation_partition_size <= 0 ||
        args.colocation_partition_size > 64
    ) {
        throw std::invalid_argument("missing or invalid required arguments");
    }
    if (args.code_only && args.index_page_async) {
        throw std::invalid_argument("code-only paging currently uses matched synchronous O_DIRECT");
    }
    if (args.fp_path.empty()) args.fp_path = args.train;
    return args;
}

template <typename T>
struct Matrix {
    int32_t rows = 0;
    int32_t cols = 0;
    std::vector<T> data;
};

template <typename T>
Matrix<T> read_matrix(const std::string& path, size_t max_rows = 0) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open matrix: " + path);
    Matrix<T> matrix;
    input.read(reinterpret_cast<char*>(&matrix.rows), sizeof(matrix.rows));
    input.read(reinterpret_cast<char*>(&matrix.cols), sizeof(matrix.cols));
    if (!input || matrix.rows <= 0 || matrix.cols <= 0) {
        throw std::runtime_error("bad matrix header: " + path);
    }
    if (max_rows > 0 && max_rows < static_cast<size_t>(matrix.rows)) {
        matrix.rows = static_cast<int32_t>(max_rows);
    }
    const size_t count = static_cast<size_t>(matrix.rows) * matrix.cols;
    matrix.data.resize(count);
    input.read(
        reinterpret_cast<char*>(matrix.data.data()),
        static_cast<std::streamsize>(count * sizeof(T))
    );
    if (!input) throw std::runtime_error("short matrix payload: " + path);
    return matrix;
}

void normalize_rows(Matrix<float>& matrix) {
    for (int32_t i = 0; i < matrix.rows; ++i) {
        float* row = matrix.data.data() + static_cast<size_t>(i) * matrix.cols;
        double norm2 = 0.0;
        for (int32_t j = 0; j < matrix.cols; ++j) {
            norm2 += static_cast<double>(row[j]) * row[j];
        }
        const float inverse = static_cast<float>(1.0 / std::sqrt(std::max(norm2, 1e-30)));
        for (int32_t j = 0; j < matrix.cols; ++j) row[j] *= inverse;
    }
}

class FbinODirectReader {
    int fd_ = -1;
    uint64_t data_offset_ = 8;

   public:
    struct ThreadBuffer {
        void* data = nullptr;
        size_t capacity = 0;
        ~ThreadBuffer() { std::free(data); }
        void ensure(size_t bytes) {
            if (capacity >= bytes) return;
            std::free(data);
            data = nullptr;
            if (posix_memalign(&data, 4096, bytes) != 0) throw std::bad_alloc();
            capacity = bytes;
        }
    };

    int32_t n = 0;
    int32_t d = 0;
    size_t row_bytes = 0;
    std::atomic<uint64_t> read_bytes{0};
    std::atomic<uint64_t> read_requests{0};

    explicit FbinODirectReader(const std::string& path) {
        int metadata_fd = ::open(path.c_str(), O_RDONLY);
        if (metadata_fd < 0) throw std::runtime_error("cannot open fbin: " + path);
        int32_t header[2]{};
        if (::pread(metadata_fd, header, sizeof(header), 0) != sizeof(header)) {
            ::close(metadata_fd);
            throw std::runtime_error("cannot read fbin header: " + path);
        }
        n = header[0];
        d = header[1];
        row_bytes = static_cast<size_t>(d) * sizeof(float);
        struct stat file_status {};
        if (::fstat(metadata_fd, &file_status) != 0 || file_status.st_size < 0) {
            ::close(metadata_fd);
            throw std::runtime_error("cannot stat fbin: " + path);
        }
        const uint64_t payload_bytes =
            static_cast<uint64_t>(n) * static_cast<uint64_t>(row_bytes);
        const uint64_t file_bytes = static_cast<uint64_t>(file_status.st_size);
        if (file_bytes == 4096ULL + payload_bytes) {
            data_offset_ = 4096;
        } else if (file_bytes == 8ULL + payload_bytes) {
            data_offset_ = 8;
        } else {
            ::close(metadata_fd);
            throw std::runtime_error("unexpected fbin file size: " + path);
        }
        ::close(metadata_fd);
        fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd_ < 0) {
            throw std::runtime_error(
                "O_DIRECT open failed: " + path + ": " + std::string(std::strerror(errno))
            );
        }
    }

    ~FbinODirectReader() {
        if (fd_ >= 0) ::close(fd_);
    }

    const float* read_row(
        size_t row,
        ThreadBuffer& buffer,
        size_t* submitted_bytes_out = nullptr
    ) {
        if (row >= static_cast<size_t>(n)) throw std::out_of_range("fbin row");
        const uint64_t logical_offset = data_offset_ + row * row_bytes;
        const uint64_t aligned_offset = logical_offset & ~uint64_t(4095);
        const size_t prefix = static_cast<size_t>(logical_offset - aligned_offset);
        const size_t needed = prefix + row_bytes;
        const size_t submitted = (needed + 4095U) & ~size_t(4095U);
        buffer.ensure(submitted);
        size_t done = 0;
        while (done < submitted) {
            const size_t requested = submitted - done;
            const ssize_t got = ::pread(
                fd_,
                static_cast<char*>(buffer.data) + done,
                requested,
                static_cast<off_t>(aligned_offset + done)
            );
            if (got < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(
                    "FP O_DIRECT read failed: " + std::string(std::strerror(errno))
                );
            }
            if (got == 0) {
                if (done < needed) throw std::runtime_error("short FP O_DIRECT read");
                std::memset(static_cast<char*>(buffer.data) + done, 0, submitted - done);
                break;
            }
            done += static_cast<size_t>(got);
            // The final aligned request may extend beyond the unpadded fbin EOF.
            // Accept a short read once the complete requested row is resident.
            if (static_cast<size_t>(got) < requested) {
                if (done < needed) throw std::runtime_error("short FP O_DIRECT read");
                std::memset(static_cast<char*>(buffer.data) + done, 0, submitted - done);
                break;
            }
        }
        read_bytes.fetch_add(submitted, std::memory_order_relaxed);
        read_requests.fetch_add(1, std::memory_order_relaxed);
        if (submitted_bytes_out != nullptr) *submitted_bytes_out = submitted;
        return reinterpret_cast<const float*>(
            static_cast<const char*>(buffer.data) + prefix
        );
    }
};

double percentile(std::vector<size_t> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double rank = p * static_cast<double>(values.size() - 1);
    const size_t low = static_cast<size_t>(std::floor(rank));
    const size_t high = static_cast<size_t>(std::ceil(rank));
    const double fraction = rank - low;
    return values[low] * (1.0 - fraction) + values[high] * fraction;
}

double recall_at_k(
    const std::vector<std::vector<std::pair<float, rabitqlib::PID>>>& results,
    const Matrix<int32_t>& gt,
    int k
) {
    uint64_t hits = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        for (int j = 0; j < k && j < static_cast<int>(results[i].size()); ++j) {
            const auto id = static_cast<int32_t>(results[i][static_cast<size_t>(j)].second);
            const int32_t* truth = gt.data.data() + i * gt.cols;
            for (int t = 0; t < k; ++t) {
                if (truth[t] == id) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / (results.size() * static_cast<size_t>(k));
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
        std::max<size_t>(values.size(), 1);
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        Matrix<float> queries = read_matrix<float>(args.query, args.max_queries);
        Matrix<int32_t> gt = read_matrix<int32_t>(args.gt, args.max_queries);
        if (queries.rows != gt.rows || queries.cols <= 0 || gt.cols < args.k) {
            throw std::runtime_error("query/ground-truth shape mismatch");
        }
        normalize_rows(queries);

        rabitqlib::hnsw::HierarchicalNSW index;
        const auto load_start = Clock::now();
        index.load(args.index.c_str());
        const double load_seconds = std::chrono::duration<double>(
            Clock::now() - load_start
        ).count();
        if (index.nbits() != 1 || index.dimension() != static_cast<size_t>(queries.cols)) {
            throw std::runtime_error("incompatible RaBitQ index");
        }
        const uint64_t upper_graph_bytes = index.upper_graph_bytes();
        const auto build = args.code_only
            ? index.prepare_1bit_paged_code_index(
                args.paged_index, args.rebuild_paged_index
            )
            : index.prepare_1bit_paged_index(
                args.paged_index,
                args.rebuild_paged_index,
                args.colocation_iterations,
                args.colocation_partition_size
            );
        if (args.prepare_only) {
            fs::path output_path(args.out);
            if (!output_path.parent_path().empty()) {
                fs::create_directories(output_path.parent_path());
            }
            std::ofstream output(args.out, std::ios::trunc);
            if (!output) throw std::runtime_error("cannot create prepare-only CSV");
            output
                << "dataset,page_bytes,paged_index_prepare_seconds,paged_index_status,"
                << "paged_index_file_bytes,paged_index_data_pages,upper_graph_bytes,"
                << "upper_graph_paged,resident_upper_graph_bytes,colocated,"
                << "colocation_iterations,colocation_partition_size,"
                << "same_page_edge_ratio,paged_storage_scope,paged_index_path\n"
                << args.dataset << ',' << rabitqlib::hnsw::kRabitqPagedIndexPageBytes
                << ',' << std::setprecision(12) << build.seconds << ','
                << (build.loaded ? "loaded" : "built") << ',' << build.file_bytes
                << ',' << build.data_pages << ',' << upper_graph_bytes
                << (args.code_only ? ",false," : ",true,")
                << (args.code_only ? upper_graph_bytes : 0) << ','
                << (build.colocated ? "true" : "false") << ','
                << build.colocation_iterations << ','
                << build.colocation_partition_size << ','
                << build.same_page_edge_ratio << ','
                << (args.code_only ? "code_only" : "graph_and_code") << ','
                << args.paged_index << '\n';
            return 0;
        }
        if (args.code_only) {
            index.release_codes_keep_graph_for_paged_search();
        } else {
            index.release_level0_memory_for_paged_search();
        }

        fs::path output_path(args.out);
        if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
        std::ofstream output(args.out, std::ios::trunc);
        if (!output) throw std::runtime_error("cannot create output CSV");
        output << std::setprecision(12);
        output
            << "dataset,method,nq,nb,dim,M,num_clusters,efConstruction,efSearch,k,rounds,search_threads,"
            << "recall_at_k,qps,total_seconds_mean,fp_reads_mean,fp_reads_min,fp_reads_max,"
            << "fp_reads_p95,fp_reads_p99,fp_odirect_bytes_mean,fp_odirect_requests_mean,"
            << "fp_odirect_bytes_per_query,fp_odirect_requests_per_query,"
            << "fp_physical_4k_reads_per_query,fp_physical_4k_reads_mean,"
            << "fp_physical_4k_reads_min,fp_physical_4k_reads_max,"
            << "fp_physical_4k_reads_p95,fp_physical_4k_reads_p99,"
            << "index_page_bytes,index_page_cache_mib,index_page_cache_bytes,"
            << "index_page_async,index_page_qd,index_async_batches_mean,"
            << "index_async_requests_mean,"
            << "index_async_cached_nodes_scored_while_inflight_mean,"
            << "index_async_completed_nodes_scored_while_inflight_mean,"
            << "index_async_completions_ready_before_wait_mean,"
            << "index_async_ready_ratio,index_async_wait_seconds_sum_mean,"
            << "index_data_pages,upper_graph_bytes,upper_graph_paged,"
            << "resident_upper_graph_bytes,index_page_accesses_mean,index_page_hits_mean,"
            << "index_page_misses_mean,index_page_evictions_mean,index_page_hit_ratio,"
            << "index_page_read_bytes_mean,index_page_read_bytes_per_query,"
            << "index_page_requests_per_query,total_physical_4k_reads_per_query,"
            << "index_load_seconds,paged_index_prepare_seconds,"
            << "paged_index_status,paged_index_file_bytes,colocated,"
            << "colocation_iterations,colocation_partition_size,"
            << "same_page_edge_ratio,paged_storage_scope,paged_index_path,index_path,fp_path\n";

        for (const int ef : args.efs) {
            std::vector<double> seconds;
            std::vector<double> recalls;
            std::vector<double> fp_means;
            std::vector<double> fp_mins;
            std::vector<double> fp_maxs;
            std::vector<double> fp_p95s;
            std::vector<double> fp_p99s;
            std::vector<double> fp_bytes;
            std::vector<double> fp_requests;
            std::vector<double> fp_page_means;
            std::vector<double> fp_page_mins;
            std::vector<double> fp_page_maxs;
            std::vector<double> fp_page_p95s;
            std::vector<double> fp_page_p99s;
            std::vector<double> page_accesses;
            std::vector<double> page_hits;
            std::vector<double> page_misses;
            std::vector<double> page_evictions;
            std::vector<double> page_read_bytes;
            std::vector<double> async_batches;
            std::vector<double> async_requests;
            std::vector<double> async_cached_nodes;
            std::vector<double> async_completed_nodes;
            std::vector<double> async_ready_completions;
            std::vector<double> async_wait_seconds;
            uint64_t data_pages = 0;

            for (int round = 0; round < args.rounds; ++round) {
                FbinODirectReader reader(args.fp_path);
                if (reader.d != queries.cols) throw std::runtime_error("train/query dim mismatch");
                std::vector<size_t> fp_pages_by_query(
                    static_cast<size_t>(queries.rows), 0
                );
                auto exact_distance = [&](size_t query_id, rabitqlib::PID label) -> float {
                    thread_local FbinODirectReader::ThreadBuffer buffer;
                    size_t submitted_bytes = 0;
                    const float* row = reader.read_row(
                        static_cast<size_t>(label), buffer, &submitted_bytes
                    );
                    fp_pages_by_query[query_id] += submitted_bytes / 4096;
                    const float* query = queries.data.data() + query_id * queries.cols;
                    double dot = 0.0;
                    double norm2 = 0.0;
                    for (int32_t j = 0; j < queries.cols; ++j) {
                        const double value = row[j];
                        dot += value * query[j];
                        norm2 += value * value;
                    }
                    return static_cast<float>(
                        1.0 - dot / std::sqrt(std::max(norm2, 1e-30))
                    );
                };
                const auto search_start = Clock::now();
                auto result = args.code_only
                    ? index.search_1bit_adaptive_fp_code_paged_callback(
                        queries.data.data(),
                        static_cast<size_t>(reader.n),
                        static_cast<size_t>(queries.rows),
                        static_cast<size_t>(args.k),
                        static_cast<size_t>(ef),
                        static_cast<size_t>(args.threads),
                        args.paged_index,
                        args.cache_bytes,
                        exact_distance
                    )
                    : index.search_1bit_adaptive_fp_paged_callback(
                        queries.data.data(),
                        static_cast<size_t>(reader.n),
                        static_cast<size_t>(queries.rows),
                        static_cast<size_t>(args.k),
                        static_cast<size_t>(ef),
                        static_cast<size_t>(args.threads),
                        args.paged_index,
                        args.cache_mib,
                        exact_distance,
                        args.index_page_async,
                        args.index_page_qd
                    );
                const double elapsed = std::chrono::duration<double>(
                    Clock::now() - search_start
                ).count();
                const double recall = recall_at_k(result.results, gt, args.k);
                const auto [minimum, maximum] = std::minmax_element(
                    result.fp_read_count.begin(), result.fp_read_count.end()
                );
                const auto [page_minimum, page_maximum] = std::minmax_element(
                    fp_pages_by_query.begin(), fp_pages_by_query.end()
                );
                const double fp_mean = std::accumulate(
                    result.fp_read_count.begin(), result.fp_read_count.end(), 0.0
                ) / result.fp_read_count.size();
                const double fp_page_mean = std::accumulate(
                    fp_pages_by_query.begin(), fp_pages_by_query.end(), 0.0
                ) / fp_pages_by_query.size();
                seconds.push_back(elapsed);
                recalls.push_back(recall);
                fp_means.push_back(fp_mean);
                fp_mins.push_back(*minimum);
                fp_maxs.push_back(*maximum);
                fp_p95s.push_back(percentile(result.fp_read_count, 0.95));
                fp_p99s.push_back(percentile(result.fp_read_count, 0.99));
                fp_bytes.push_back(reader.read_bytes.load());
                fp_requests.push_back(reader.read_requests.load());
                fp_page_means.push_back(fp_page_mean);
                fp_page_mins.push_back(*page_minimum);
                fp_page_maxs.push_back(*page_maximum);
                fp_page_p95s.push_back(percentile(fp_pages_by_query, 0.95));
                fp_page_p99s.push_back(percentile(fp_pages_by_query, 0.99));
                page_accesses.push_back(result.page_cache.accesses);
                page_hits.push_back(result.page_cache.hits);
                page_misses.push_back(result.page_cache.misses);
                page_evictions.push_back(result.page_cache.evictions);
                page_read_bytes.push_back(result.page_cache.read_bytes);
                async_batches.push_back(result.page_cache.async_batches);
                async_requests.push_back(result.page_cache.async_requests);
                async_cached_nodes.push_back(
                    result.page_cache.async_cached_nodes_scored_while_inflight
                );
                async_completed_nodes.push_back(
                    result.page_cache.async_completed_nodes_scored_while_inflight
                );
                async_ready_completions.push_back(
                    result.page_cache.async_completions_ready_before_wait
                );
                async_wait_seconds.push_back(
                    result.page_cache.async_wait_seconds
                );
                data_pages = result.page_cache.data_pages;
                std::cout
                    << args.dataset << " ef=" << ef << " round=" << (round + 1)
                    << "/" << args.rounds << " recall=" << recall
                    << " qps=" << queries.rows / elapsed
                    << " fp_reads=" << fp_mean
                    << " fp_4k_reads=" << fp_page_mean
                    << " page_hit_ratio="
                    << static_cast<double>(result.page_cache.hits) /
                        std::max<uint64_t>(result.page_cache.accesses, 1)
                    << " page_read_bytes=" << result.page_cache.read_bytes
                    << " graph_async=" << (args.index_page_async ? "true" : "false")
                    << " async_requests=" << result.page_cache.async_requests
                    << std::endl;
            }

            const double mean_seconds = mean(seconds);
            const double mean_page_accesses = mean(page_accesses);
            const double mean_page_hits = mean(page_hits);
            const double mean_page_misses = mean(page_misses);
            const double mean_page_read_bytes = mean(page_read_bytes);
            output
                << args.dataset << ','
                << (args.code_only
                    ? "RaBitQ-1bit-adaptive-FP-code-paged4k"
                    : (build.colocated
                    ? (args.index_page_async
                        ? "RaBitQ-1bit-adaptive-FP-paged4k-coloc-graph-async"
                        : "RaBitQ-1bit-adaptive-FP-paged4k-coloc")
                    : (args.index_page_async
                        ? "RaBitQ-1bit-adaptive-FP-paged4k-baseline-graph-async"
                        : "RaBitQ-1bit-adaptive-FP-paged4k-baseline"))) << ','
                << queries.rows << ',' << index.max_elements() << ',' << queries.cols << ','
                << index.M() << ',' << index.num_clusters() << ','
                << index.ef_construction() << ',' << ef << ',' << args.k << ','
                << args.rounds << ',' << args.threads << ',' << mean(recalls) << ','
                << queries.rows / mean_seconds << ',' << mean_seconds << ','
                << mean(fp_means) << ',' << mean(fp_mins) << ',' << mean(fp_maxs) << ','
                << mean(fp_p95s) << ',' << mean(fp_p99s) << ',' << mean(fp_bytes) << ','
                << mean(fp_requests) << ',' << mean(fp_bytes) / queries.rows << ','
                << mean(fp_requests) / queries.rows << ','
                << mean(fp_bytes) / queries.rows / 4096.0 << ','
                << mean(fp_page_means) << ',' << mean(fp_page_mins) << ','
                << mean(fp_page_maxs) << ',' << mean(fp_page_p95s) << ','
                << mean(fp_page_p99s) << ','
                << rabitqlib::hnsw::kRabitqPagedIndexPageBytes << ','
                << (args.code_only
                    ? static_cast<double>(args.cache_bytes) / (1024.0 * 1024.0)
                    : static_cast<double>(args.cache_mib)) << ','
                << (args.code_only
                    ? args.cache_bytes
                    : static_cast<uint64_t>(args.cache_mib) * 1024ULL * 1024ULL) << ','
                << (args.index_page_async ? "true" : "false") << ','
                << args.index_page_qd << ',' << mean(async_batches) << ','
                << mean(async_requests) << ',' << mean(async_cached_nodes) << ','
                << mean(async_completed_nodes) << ','
                << mean(async_ready_completions) << ','
                << mean(async_ready_completions) /
                    std::max(mean(async_requests), 1.0) << ','
                << mean(async_wait_seconds) << ','
                << data_pages << ',' << upper_graph_bytes << ','
                << (args.code_only ? "false" : "true") << ','
                << (args.code_only ? upper_graph_bytes : 0) << ','
                << mean_page_accesses << ',' << mean_page_hits << ','
                << mean_page_misses << ',' << mean(page_evictions) << ','
                << mean_page_hits / std::max(mean_page_accesses, 1.0) << ','
                << mean_page_read_bytes << ',' << mean_page_read_bytes / queries.rows << ','
                << mean_page_misses / queries.rows << ','
                << (mean_page_read_bytes + mean(fp_bytes)) / queries.rows / 4096.0 << ','
                << load_seconds << ','
                << build.seconds << ',' << (build.loaded ? "loaded" : "built") << ','
                << build.file_bytes << ','
                << (build.colocated ? "true" : "false") << ','
                << build.colocation_iterations << ','
                << build.colocation_partition_size << ','
                << build.same_page_edge_ratio << ','
                << (args.code_only ? "code_only" : "graph_and_code") << ','
                << args.paged_index << ',' << args.index << ',' << args.fp_path << '\n';
            output.flush();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
