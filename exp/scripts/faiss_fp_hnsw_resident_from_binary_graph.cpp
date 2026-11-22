#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct Args {
    std::string dataset;
    std::string train;
    std::string query;
    std::string gt;
    std::string graph;
    std::string out;
    int m = 24;
    int ef_construction = 200;
    int k = 10;
    int rounds = 5;
    int threads = 24;
    int add_batch_size = 4096;
    std::vector<int> efs{32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " --dataset NAME --train X.fbin --query Q.fbin --gt G.ibin"
              << " --graph ADCBQ_BINARY_HNSW.faiss --out O.csv [--efs ...]"
              << " [--m 24] [--ef-construction 200] [--k 10] [--rounds 5]"
              << " [--threads 24] [--add-batch-size 4096]\n";
    std::exit(2);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (flag == "--dataset") args.dataset = need("--dataset");
        else if (flag == "--train") args.train = need("--train");
        else if (flag == "--query") args.query = need("--query");
        else if (flag == "--gt") args.gt = need("--gt");
        else if (flag == "--graph") args.graph = need("--graph");
        else if (flag == "--out") args.out = need("--out");
        else if (flag == "--m") args.m = std::stoi(need("--m"));
        else if (flag == "--ef-construction") args.ef_construction = std::stoi(need("--ef-construction"));
        else if (flag == "--k") args.k = std::stoi(need("--k"));
        else if (flag == "--rounds") args.rounds = std::stoi(need("--rounds"));
        else if (flag == "--threads") args.threads = std::stoi(need("--threads"));
        else if (flag == "--add-batch-size") args.add_batch_size = std::stoi(need("--add-batch-size"));
        else if (flag == "--efs") {
            args.efs.clear();
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                args.efs.push_back(std::stoi(argv[++i]));
            }
        } else {
            throw std::runtime_error("unknown argument: " + flag);
        }
    }
    if (args.dataset.empty() || args.train.empty() || args.query.empty() ||
        args.gt.empty() || args.graph.empty() || args.out.empty() || args.efs.empty()) {
        usage(argv[0]);
    }
    if (args.k <= 0 || args.rounds <= 0 || args.threads <= 0 || args.add_batch_size <= 0) {
        throw std::runtime_error("k, rounds, threads, and add-batch-size must be positive");
    }
    return args;
}

struct FbinMmap {
    int fd = -1;
    void* mapping = MAP_FAILED;
    size_t bytes = 0;
    int32_t n = 0;
    int32_t d = 0;
    const float* data = nullptr;

    explicit FbinMmap(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open failed: " + path);
        int32_t shape[2];
        if (::pread(fd, shape, sizeof(shape), 0) != static_cast<ssize_t>(sizeof(shape))) {
            throw std::runtime_error("bad fbin header: " + path);
        }
        n = shape[0];
        d = shape[1];
        if (n <= 0 || d <= 0) throw std::runtime_error("invalid fbin shape: " + path);
        bytes = 8 + static_cast<size_t>(n) * d * sizeof(float);
        struct stat st {};
        if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) != bytes) {
            throw std::runtime_error("unexpected fbin size: " + path);
        }
        mapping = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        data = reinterpret_cast<const float*>(static_cast<const char*>(mapping) + 8);
    }

    ~FbinMmap() {
        release();
    }

    void release() {
        if (mapping != MAP_FAILED) ::munmap(mapping, bytes);
        if (fd >= 0) ::close(fd);
        mapping = MAP_FAILED;
        fd = -1;
        data = nullptr;
    }

    const float* row(size_t i) const { return data + i * static_cast<size_t>(d); }
};

std::vector<float> read_fbin(const std::string& path, int32_t& n, int32_t& d) {
    FbinMmap source(path);
    n = source.n;
    d = source.d;
    std::vector<float> values(static_cast<size_t>(n) * d);
    std::memcpy(values.data(), source.data, values.size() * sizeof(float));
    return values;
}

std::vector<int32_t> read_ibin(const std::string& path, int32_t& n, int32_t& d) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("open failed: " + path);
    input.read(reinterpret_cast<char*>(&n), sizeof(n));
    input.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!input || n <= 0 || d <= 0) throw std::runtime_error("bad ibin header: " + path);
    std::vector<int32_t> values(static_cast<size_t>(n) * d);
    input.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(int32_t));
    if (!input) throw std::runtime_error("short ibin: " + path);
    return values;
}

void normalize_rows(float* values, int64_t rows, int d) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        float* row = values + static_cast<size_t>(i) * d;
        double squared = 0.0;
#pragma omp simd reduction(+ : squared)
        for (int j = 0; j < d; ++j) squared += static_cast<double>(row[j]) * row[j];
        const float inv = 1.0f / static_cast<float>(std::sqrt(std::max(squared, 1e-30)));
#pragma omp simd
        for (int j = 0; j < d; ++j) row[j] *= inv;
    }
}

std::unique_ptr<faiss::IndexBinaryHNSW> load_graph(const std::string& path) {
    faiss::IndexBinary* raw = faiss::read_index_binary(path.c_str());
    auto* graph = dynamic_cast<faiss::IndexBinaryHNSW*>(raw);
    if (!graph) {
        delete raw;
        throw std::runtime_error("not an IndexBinaryHNSW: " + path);
    }
    return std::unique_ptr<faiss::IndexBinaryHNSW>(graph);
}

double recall_at_k(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<int32_t>& gt,
        int nq,
        int gt_dim,
        int k) {
    int64_t hits = 0;
#pragma omp parallel for reduction(+ : hits) schedule(static)
    for (int qi = 0; qi < nq; ++qi) {
        for (int j = 0; j < k; ++j) {
            const faiss::idx_t id = labels[static_cast<size_t>(qi) * k + j];
            for (int g = 0; g < k; ++g) {
                if (id == gt[static_cast<size_t>(qi) * gt_dim + g]) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(nq * k);
}

uint64_t current_rss_bytes() {
    std::ifstream status("/proc/self/status");
    std::string key;
    while (status >> key) {
        if (key == "VmRSS:") {
            uint64_t kib = 0;
            std::string unit;
            status >> kib >> unit;
            return kib * 1024;
        }
        std::string rest;
        std::getline(status, rest);
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        omp_set_num_threads(args.threads);

        FbinMmap train(args.train);
        const int32_t nb = train.n;
        const int32_t dim = train.d;
        int32_t nq = 0;
        int32_t query_dim = 0;
        std::vector<float> queries = read_fbin(args.query, nq, query_dim);
        int32_t gt_n = 0;
        int32_t gt_dim = 0;
        std::vector<int32_t> gt = read_ibin(args.gt, gt_n, gt_dim);
        if (query_dim != dim || gt_n != nq || gt_dim < args.k) {
            throw std::runtime_error("train/query/GT shape mismatch");
        }
        normalize_rows(queries.data(), nq, query_dim);

        const auto prepare_start = std::chrono::steady_clock::now();
        auto graph = load_graph(args.graph);
        if (graph->ntotal != nb) throw std::runtime_error("graph/train row mismatch");

        faiss::IndexHNSWFlat fp(dim, args.m, faiss::METRIC_INNER_PRODUCT);
        fp.hnsw.efConstruction = args.ef_construction;
        for (int64_t begin = 0; begin < nb; begin += args.add_batch_size) {
            const int64_t end = std::min<int64_t>(nb, begin + args.add_batch_size);
            std::vector<float> block(static_cast<size_t>(end - begin) * dim);
            std::memcpy(
                    block.data(),
                    train.row(static_cast<size_t>(begin)),
                    block.size() * sizeof(float));
            normalize_rows(block.data(), end - begin, dim);
            fp.storage->add(end - begin, block.data());
            if (end == nb || end % 100000 == 0) {
                std::cout << "  resident FP32 rows " << end << "/" << nb << std::endl;
            }
        }
        fp.ntotal = fp.storage->ntotal;
        fp.hnsw = graph->hnsw;
        fp.hnsw.efConstruction = args.ef_construction;
        graph.reset();
        train.release();
        ::malloc_trim(0);
        const double prepare_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - prepare_start).count();
        const uint64_t rss_bytes = current_rss_bytes();
        const uint64_t vector_bytes = static_cast<uint64_t>(nb) * dim * sizeof(float);
        std::cout << "FP-HNSW ready with copied topology and process-owned full vectors: ntotal="
                  << fp.ntotal << " vector_bytes=" << vector_bytes
                  << " rss_bytes=" << rss_bytes
                  << " prepare_seconds=" << prepare_seconds << std::endl;

        fs::create_directories(fs::path(args.out).parent_path());
        std::ofstream csv(args.out);
        if (!csv) throw std::runtime_error("cannot open output: " + args.out);
        csv << "dataset,method,k,nq,nb,dim,M,efConstruction,efSearch,rounds,threads,"
            << "search_seconds_mean,search_seconds_min,search_seconds_max,recall_at_k,qps,qps_median,"
            << "distance_evaluations_mean,hops_mean,"
            << "full_vector_storage,resident_vector_bytes,process_rss_bytes,prepare_seconds,"
            << "graph_template_bytes,graph_template_path\n";

        omp_set_num_threads(args.threads);
        std::vector<float> distances(static_cast<size_t>(nq) * args.k);
        std::vector<faiss::idx_t> labels(static_cast<size_t>(nq) * args.k);
        for (int ef : args.efs) {
            fp.hnsw.efSearch = ef;
            std::vector<double> elapsed_values;
            elapsed_values.reserve(args.rounds);
            uint64_t distance_evaluations = 0;
            uint64_t hops = 0;
            for (int round = 0; round < args.rounds; ++round) {
                faiss::hnsw_stats.reset();
                const auto start = std::chrono::steady_clock::now();
                fp.search(nq, queries.data(), args.k, distances.data(), labels.data());
                const double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start).count();
                distance_evaluations += faiss::hnsw_stats.ndis;
                hops += faiss::hnsw_stats.nhops;
                elapsed_values.push_back(elapsed);
                std::cout << "ef=" << ef << " round=" << round + 1 << "/" << args.rounds
                          << " qps=" << nq / elapsed
                          << " distance_evaluations/query="
                          << static_cast<double>(faiss::hnsw_stats.ndis) / nq
                          << std::endl;
            }
            const double mean = std::accumulate(
                    elapsed_values.begin(), elapsed_values.end(), 0.0) /
                    elapsed_values.size();
            const auto [minimum, maximum] = std::minmax_element(
                    elapsed_values.begin(), elapsed_values.end());
            std::vector<double> sorted_qps;
            sorted_qps.reserve(elapsed_values.size());
            for (double elapsed : elapsed_values) {
                sorted_qps.push_back(nq / elapsed);
            }
            std::sort(sorted_qps.begin(), sorted_qps.end());
            const double qps_median = sorted_qps.size() % 2 == 0
                    ? 0.5 * (sorted_qps[sorted_qps.size() / 2 - 1] +
                             sorted_qps[sorted_qps.size() / 2])
                    : sorted_qps[sorted_qps.size() / 2];
            const double recall = recall_at_k(labels, gt, nq, gt_dim, args.k);
            csv << args.dataset << ",FP-HNSW," << args.k << ',' << nq << ',' << nb
                << ',' << dim << ',' << args.m << ',' << args.ef_construction << ','
                << ef << ',' << args.rounds << ',' << args.threads << ',' << mean << ','
                << *minimum << ',' << *maximum << ',' << recall << ',' << nq / mean
                << ',' << qps_median
                << ',' << static_cast<double>(distance_evaluations) /
                                   (static_cast<double>(nq) * args.rounds)
                << ',' << static_cast<double>(hops) /
                                   (static_cast<double>(nq) * args.rounds)
                << ",process_owned_ram," << vector_bytes << ',' << rss_bytes << ','
                << prepare_seconds << ',' << fs::file_size(args.graph) << ','
                << args.graph << '\n';
            csv.flush();
            std::cout << "FINAL ef=" << ef << " recall=" << recall
                      << " qps=" << nq / mean << std::endl;
        }
        std::cout << "WROTE " << args.out << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
