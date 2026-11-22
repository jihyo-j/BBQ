#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/scalar_quantizer/training.h>
#include <faiss/index_io.h>
#include <faiss/utils/random.h>

#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "qdrant_tq_ffi/qdrant_tq_ffi.h"

#include <cblas.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

enum class Method { PQ2, SQ4, TQProd2, BQ1, SymBQ1, RotBQ1, RaBitQ1, QdrantTQ1 };
enum class Metric { IP, L2 };

struct Args {
    std::string dataset;
    std::string train;
    std::string query;
    std::string gt;
    std::string graph;
    std::string index;
    std::string meta;
    std::string reference_meta;
    std::string rabitq_centroids;
    std::string out;
    Method method = Method::PQ2;
    Metric metric = Metric::IP;
    int m = 24;
    int ef_construction = 200;
    std::vector<int> efs{64, 128, 256, 512, 1024};
    int k = 10;
    int max_queries = 1000;
    int64_t max_base = 0;
    int rounds = 3;
    int build_threads = 24;
    int search_threads = 1;
    int add_batch_size = 4096;
    int train_size = 100000;
    int rotation_seed = 20261020;
    int qjl_seed = 67890;
    bool rebuild = false;
    bool rebuild_graph = false;
    bool native_bq_graph = false;
    bool native_quantized_graph = false;
};

const char* method_name(Method method) {
    switch (method) {
        case Method::PQ2: return "Faiss-PQ2-LUT256";
        case Method::SQ4: return "Faiss-SQ4";
        case Method::TQProd2: return "TurboQuant-prod2";
        case Method::BQ1: return "ADC-BQ1";
        case Method::SymBQ1: return "Symmetric-BQ1";
        case Method::RotBQ1: return "Rotated-Symmetric-BQ1";
        case Method::RaBitQ1: return "RaBitQ-1b-c16";
        case Method::QdrantTQ1: return "Qdrant-TQ1-Plus-C++";
    }
    return "unknown";
}

const char* metric_name(Metric metric) {
    return metric == Metric::IP ? "ip" : "l2";
}

Method parse_method(const std::string& value) {
    if (value == "pq2" || value == "pq") return Method::PQ2;
    if (value == "sq4" || value == "sq") return Method::SQ4;
    if (value == "tq2" || value == "tqprod2" || value == "turboquant") {
        return Method::TQProd2;
    }
    if (value == "bq1" || value == "adcbq") return Method::BQ1;
    if (value == "symbq1" || value == "symbq" || value == "bq") return Method::SymBQ1;
    if (value == "rotbq1" || value == "rotbq") return Method::RotBQ1;
    if (value == "rabitq1" || value == "rq1" || value == "rabitq") {
        return Method::RaBitQ1;
    }
    if (value == "qdranttq1" || value == "qdrant-tq1" || value == "tq1") {
        return Method::QdrantTQ1;
    }
    throw std::runtime_error(
            "--method must be pq2|sq4|tqprod2|bq1|symbq1|rotbq1|rabitq1|qdranttq1");
}

Metric parse_metric(const std::string& value) {
    if (value == "ip" || value == "cosine") return Metric::IP;
    if (value == "l2") return Metric::L2;
    throw std::runtime_error("--metric must be ip|l2");
}

[[noreturn]] void usage(const char* argv0) {
    std::cerr
            << "Usage: " << argv0
            << " --dataset NAME --method pq2|sq4|tqprod2|bq1|symbq1|rotbq1|rabitq1|qdranttq1"
            << " --metric ip|l2"
            << " --train X.fbin --query Q.fbin --gt G.ibin --graph GRAPH.faiss"
            << " --index INDEX.faiss --out OUT.csv [--meta META.bin] [--efs ...]"
            << " [--reference-meta TQ_REFERENCE_META.bin]"
            << " [--rabitq-centroids C16.fbin]"
            << " [--m 24] [--ef-construction 200] [--build-threads 24]"
            << " [--search-threads 1] [--rebuild] [--rebuild-graph]"
            << " [--native-bq-graph] [--native-quantized-graph]\n";
    std::exit(2);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (a == "--dataset") args.dataset = need("--dataset");
        else if (a == "--method") args.method = parse_method(need("--method"));
        else if (a == "--metric") args.metric = parse_metric(need("--metric"));
        else if (a == "--train") args.train = need("--train");
        else if (a == "--query") args.query = need("--query");
        else if (a == "--gt") args.gt = need("--gt");
        else if (a == "--graph") args.graph = need("--graph");
        else if (a == "--index") args.index = need("--index");
        else if (a == "--meta") args.meta = need("--meta");
        else if (a == "--reference-meta") args.reference_meta = need("--reference-meta");
        else if (a == "--rabitq-centroids") {
            args.rabitq_centroids = need("--rabitq-centroids");
        }
        else if (a == "--out") args.out = need("--out");
        else if (a == "--m") args.m = std::stoi(need("--m"));
        else if (a == "--ef-construction") args.ef_construction = std::stoi(need("--ef-construction"));
        else if (a == "--k") args.k = std::stoi(need("--k"));
        else if (a == "--max-queries") args.max_queries = std::stoi(need("--max-queries"));
        else if (a == "--max-base") args.max_base = std::stoll(need("--max-base"));
        else if (a == "--rounds") args.rounds = std::stoi(need("--rounds"));
        else if (a == "--build-threads") args.build_threads = std::stoi(need("--build-threads"));
        else if (a == "--search-threads") args.search_threads = std::stoi(need("--search-threads"));
        else if (a == "--add-batch-size") args.add_batch_size = std::stoi(need("--add-batch-size"));
        else if (a == "--train-size") args.train_size = std::stoi(need("--train-size"));
        else if (a == "--rotation-seed") args.rotation_seed = std::stoi(need("--rotation-seed"));
        else if (a == "--qjl-seed") args.qjl_seed = std::stoi(need("--qjl-seed"));
        else if (a == "--rebuild") args.rebuild = true;
        else if (a == "--rebuild-graph") args.rebuild_graph = true;
        else if (a == "--native-bq-graph") args.native_bq_graph = true;
        else if (a == "--native-quantized-graph") args.native_quantized_graph = true;
        else if (a == "--efs") {
            args.efs.clear();
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                args.efs.push_back(std::stoi(argv[++i]));
            }
        } else if (a == "--help" || a == "-h") {
            usage(argv[0]);
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (args.dataset.empty() || args.train.empty() || args.query.empty() || args.gt.empty() ||
        (!args.native_bq_graph && !args.native_quantized_graph && args.graph.empty()) ||
        args.index.empty() || args.out.empty()) {
        usage(argv[0]);
    }
    if (args.native_bq_graph &&
        args.method != Method::SymBQ1 && args.method != Method::RotBQ1) {
        throw std::runtime_error(
                "--native-bq-graph only supports symbq1 or rotbq1");
    }
    if (args.native_quantized_graph &&
        args.method != Method::PQ2 && args.method != Method::SQ4) {
        throw std::runtime_error(
                "--native-quantized-graph only supports pq2 or sq4");
    }
    if (args.native_bq_graph && args.native_quantized_graph) {
        throw std::runtime_error(
                "choose only one native graph construction mode");
    }
    if ((args.method == Method::TQProd2 || args.method == Method::RotBQ1 ||
         args.method == Method::RaBitQ1 ||
         args.method == Method::QdrantTQ1) &&
        args.meta.empty()) {
        args.meta = args.index + (args.method == Method::TQProd2 ? ".tqmeta"
                                    : args.method == Method::RotBQ1 ? ".rotbqmeta"
                                    : args.method == Method::RaBitQ1 ? ".rqmeta"
                                                                    : ".json");
    }
    return args;
}

void make_parent(const std::string& path) {
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

struct FbinMmap {
    int fd = -1;
    void* mapping = MAP_FAILED;
    size_t mapping_size = 0;
    int32_t file_n = 0;
    int32_t n = 0;
    int32_t d = 0;
    size_t data_offset = 8;
    const float* data = nullptr;

    FbinMmap(const std::string& path, int64_t max_rows = 0) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open failed: " + path);
        int32_t header[2];
        if (::pread(fd, header, sizeof(header), 0) != static_cast<ssize_t>(sizeof(header))) {
            throw std::runtime_error("bad fbin header: " + path);
        }
        file_n = header[0];
        d = header[1];
        if (file_n <= 0 || d <= 0) {
            throw std::runtime_error("invalid fbin shape: " + path);
        }
        n = max_rows > 0 ? static_cast<int32_t>(std::min<int64_t>(file_n, max_rows)) : file_n;
        struct stat st {};
        if (::fstat(fd, &st) != 0 || st.st_size < 0) {
            throw std::runtime_error("fstat failed: " + path);
        }
        mapping_size = static_cast<size_t>(st.st_size);
        const size_t payload_bytes =
                static_cast<size_t>(file_n) * static_cast<size_t>(d) * sizeof(float);
        if (mapping_size == 8ull + payload_bytes) {
            data_offset = 8;
        } else if (mapping_size == 4096ull + payload_bytes) {
            data_offset = 4096;
        } else {
            throw std::runtime_error("unexpected fbin size: " + path);
        }
        mapping = ::mmap(nullptr, mapping_size, PROT_READ, MAP_SHARED, fd, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
        data = reinterpret_cast<const float*>(
                static_cast<const char*>(mapping) + data_offset);
    }

    ~FbinMmap() {
        if (mapping != MAP_FAILED) ::munmap(mapping, mapping_size);
        if (fd >= 0) ::close(fd);
    }

    const float* row(size_t i) const { return data + i * static_cast<size_t>(d); }
};

std::vector<int32_t> read_ibin(const std::string& path, int max_queries, int32_t& rows, int32_t& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    int32_t file_rows = 0;
    in.read(reinterpret_cast<char*>(&file_rows), sizeof(file_rows));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    rows = max_queries > 0 ? std::min(file_rows, max_queries) : file_rows;
    std::vector<int32_t> result(static_cast<size_t>(rows) * dim);
    in.read(reinterpret_cast<char*>(result.data()), result.size() * sizeof(int32_t));
    if (!in) throw std::runtime_error("bad ibin payload: " + path);
    return result;
}

void normalize_rows(float* x, int64_t n, int d) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        float* row = x + static_cast<size_t>(i) * d;
        double ss = 0.0;
        for (int j = 0; j < d; ++j) ss += static_cast<double>(row[j]) * row[j];
        const float inv = 1.0f / static_cast<float>(std::sqrt(std::max(ss, 1e-30)));
        for (int j = 0; j < d; ++j) row[j] *= inv;
    }
}

std::vector<float> copy_block(const FbinMmap& input, int64_t start, int64_t rows, bool normalize) {
    std::vector<float> result(static_cast<size_t>(rows) * input.d);
    std::memcpy(result.data(), input.row(start), result.size() * sizeof(float));
    if (normalize) normalize_rows(result.data(), rows, input.d);
    return result;
}

std::vector<float> read_queries(const FbinMmap& input, int max_queries, bool normalize) {
    const int n = max_queries > 0 ? std::min(input.n, max_queries) : input.n;
    return copy_block(input, 0, n, normalize);
}

faiss::MetricType faiss_metric(Metric metric) {
    return metric == Metric::IP ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
}

std::unique_ptr<faiss::IndexBinaryHNSW> load_binary_hnsw(const std::string& path) {
    faiss::IndexBinary* raw = faiss::read_index_binary(path.c_str());
    auto* hnsw = dynamic_cast<faiss::IndexBinaryHNSW*>(raw);
    if (!hnsw) {
        delete raw;
        throw std::runtime_error("not an IndexBinaryHNSW: " + path);
    }
    return std::unique_ptr<faiss::IndexBinaryHNSW>(hnsw);
}

std::unique_ptr<faiss::IndexBinaryHNSW> build_or_load_fp_graph(
        const Args& args,
        const FbinMmap& train,
        double& graph_seconds,
        std::string& status) {
    const auto start = std::chrono::steady_clock::now();
    if (fs::exists(args.graph) && !args.rebuild_graph) {
        auto graph = load_binary_hnsw(args.graph);
        if (graph->ntotal != train.n) throw std::runtime_error("graph ntotal mismatch");
        graph_seconds = seconds_since(start);
        status = "loaded";
        return graph;
    }

    std::cout << "Building common FP32 HNSW graph: n=" << train.n << " d=" << train.d
              << " M=" << args.m << " efC=" << args.ef_construction
              << " metric=" << metric_name(args.metric) << std::endl;
    faiss::IndexHNSWFlat fp(train.d, args.m, faiss_metric(args.metric));
    fp.hnsw.efConstruction = args.ef_construction;
    omp_set_num_threads(args.build_threads);
    for (int64_t begin = 0; begin < train.n; begin += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(train.n, begin + args.add_batch_size);
        auto block = copy_block(train, begin, end - begin, args.metric == Metric::IP);
        fp.add(end - begin, block.data());
        std::cout << "  FP graph " << end << "/" << train.n << std::endl;
    }

    auto* storage = new faiss::IndexBinaryFlat(8);
    std::vector<uint8_t> zeros(static_cast<size_t>(train.n), 0);
    storage->add(train.n, zeros.data());
    auto graph = std::make_unique<faiss::IndexBinaryHNSW>(storage, args.m);
    graph->own_fields = true;
    graph->ntotal = storage->ntotal;
    graph->hnsw = fp.hnsw;
    graph->hnsw.efConstruction = args.ef_construction;
    make_parent(args.graph);
    faiss::write_index_binary(graph.get(), args.graph.c_str());
    graph_seconds = seconds_since(start);
    status = "built";
    std::cout << "Saved common graph -> " << args.graph << std::endl;
    return graph;
}

void copy_graph(const faiss::IndexBinaryHNSW& source, faiss::IndexHNSW& destination) {
    destination.hnsw = source.hnsw;
    destination.hnsw.efConstruction = source.hnsw.efConstruction;
    destination.hnsw.efSearch = source.hnsw.efSearch;
}

void copy_graph(const faiss::IndexBinaryHNSW& source, faiss::IndexBinaryHNSW& destination) {
    destination.hnsw = source.hnsw;
    destination.hnsw.efConstruction = source.hnsw.efConstruction;
    destination.hnsw.efSearch = source.hnsw.efSearch;
}

std::vector<float> training_sample(const FbinMmap& train, const Args& args) {
    const int64_t n = std::min<int64_t>(train.n, args.train_size);
    return copy_block(train, 0, n, args.metric == Metric::IP);
}

std::unique_ptr<faiss::IndexHNSW> build_or_load_native(
        const Args& args,
        const FbinMmap& train,
        const faiss::IndexBinaryHNSW& graph,
        double& setup_seconds,
        double& encode_seconds,
        std::string& status) {
    if (fs::exists(args.index) && !args.rebuild) {
        const auto start = std::chrono::steady_clock::now();
        faiss::Index* raw = faiss::read_index(args.index.c_str());
        auto* index = dynamic_cast<faiss::IndexHNSW*>(raw);
        if (!index) {
            delete raw;
            throw std::runtime_error("not an IndexHNSW: " + args.index);
        }
        setup_seconds = seconds_since(start);
        encode_seconds = 0.0;
        status = "loaded";
        return std::unique_ptr<faiss::IndexHNSW>(index);
    }

    const auto setup_start = std::chrono::steady_clock::now();
    std::unique_ptr<faiss::IndexHNSW> index;
    if (args.method == Method::PQ2) {
        if (train.d % 4 != 0) throw std::runtime_error("PQ2 requires dimension divisible by four");
        index = std::make_unique<faiss::IndexHNSWPQ>(
                train.d, train.d / 4, args.m, 8, faiss_metric(args.metric));
    } else if (args.method == Method::SQ4) {
        index = std::make_unique<faiss::IndexHNSWSQ>(
                train.d, faiss::ScalarQuantizer::QT_4bit, args.m, faiss_metric(args.metric));
    } else {
        throw std::runtime_error("native builder only supports PQ2/SQ4");
    }
    auto sample = training_sample(train, args);
    omp_set_num_threads(args.build_threads);
    index->train(sample.size() / train.d, sample.data());
    setup_seconds = seconds_since(setup_start);

    const auto encode_start = std::chrono::steady_clock::now();
    for (int64_t begin = 0; begin < train.n; begin += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(train.n, begin + args.add_batch_size);
        auto block = copy_block(train, begin, end - begin, args.metric == Metric::IP);
        index->storage->add(end - begin, block.data());
        std::cout << "  " << method_name(args.method) << " encoded " << end << "/" << train.n
                  << std::endl;
    }
    index->ntotal = index->storage->ntotal;
    copy_graph(graph, *index);
    encode_seconds = seconds_since(encode_start);
    make_parent(args.index);
    faiss::write_index(index.get(), args.index.c_str());
    status = "built";
    return index;
}

std::unique_ptr<faiss::IndexHNSW> build_or_load_native_quantized_graph(
        const Args& args,
        const FbinMmap& train,
        double& setup_seconds,
        double& encode_seconds,
        double& graph_seconds,
        std::string& graph_status,
        std::string& index_status) {
    if (args.method != Method::PQ2 && args.method != Method::SQ4) {
        throw std::runtime_error(
                "native quantized graph builder only supports PQ2/SQ4");
    }

    if (fs::exists(args.index) && !args.rebuild && !args.rebuild_graph) {
        const auto load_start = std::chrono::steady_clock::now();
        faiss::Index* raw = faiss::read_index(args.index.c_str());
        auto* index = dynamic_cast<faiss::IndexHNSW*>(raw);
        if (!index) {
            delete raw;
            throw std::runtime_error("not an IndexHNSW: " + args.index);
        }
        if (index->ntotal != train.n) {
            delete index;
            throw std::runtime_error("native quantized graph ntotal mismatch");
        }
        graph_seconds = seconds_since(load_start);
        setup_seconds = 0.0;
        encode_seconds = 0.0;
        graph_status = "loaded-quantized";
        index_status = "loaded";
        return std::unique_ptr<faiss::IndexHNSW>(index);
    }

    const auto setup_start = std::chrono::steady_clock::now();
    std::unique_ptr<faiss::IndexHNSW> index;
    // Faiss PQ's symmetric code-distance table is always L2. For normalized
    // cosine/IP data, build the topology with that well-defined quantized-L2
    // distance, then switch only the query traversal scorer back to IP.
    const Metric topology_metric =
            args.metric == Metric::IP ? Metric::L2 : args.metric;
    if (args.method == Method::PQ2) {
        if (train.d % 4 != 0) {
            throw std::runtime_error("PQ2 requires dimension divisible by four");
        }
        index = std::make_unique<faiss::IndexHNSWPQ>(
                train.d, train.d / 4, args.m, 8, faiss_metric(topology_metric));
    } else {
        index = std::make_unique<faiss::IndexHNSWSQ>(
                train.d,
                faiss::ScalarQuantizer::QT_4bit,
                args.m,
                faiss_metric(topology_metric));
    }
    auto sample = training_sample(train, args);
    omp_set_num_threads(args.build_threads);
    index->train(sample.size() / train.d, sample.data());
    setup_seconds = seconds_since(setup_start);

    std::cout << "Building native " << method_name(args.method)
              << " HNSW graph: n=" << train.n << " d=" << train.d
              << " M=" << args.m << " efC=" << args.ef_construction
              << std::endl;
    const auto graph_start = std::chrono::steady_clock::now();
    index->hnsw.efConstruction = args.ef_construction;
    index->retain_locks = true;
    for (int64_t begin = 0; begin < train.n; begin += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(train.n, begin + args.add_batch_size);
        auto block = copy_block(
                train, begin, end - begin, args.metric == Metric::IP);
        index->add(end - begin, block.data());
        std::cout << "  native " << method_name(args.method) << " graph "
                  << end << "/" << train.n << std::endl;
    }
    index->retain_locks = false;
    if (args.metric == Metric::IP) {
        index->metric_type = faiss::METRIC_INNER_PRODUCT;
        index->storage->metric_type = faiss::METRIC_INNER_PRODUCT;
    }
    make_parent(args.index);
    faiss::write_index(index.get(), args.index.c_str());
    graph_seconds = seconds_since(graph_start);
    // Faiss encodes each incoming block inside IndexHNSW::add, so encoding
    // cannot be timed independently from native graph construction here.
    encode_seconds = 0.0;
    graph_status = args.method == Method::PQ2
            ? (args.metric == Metric::IP ? "built-pq-cosine" : "built-pq")
            : (args.metric == Metric::IP ? "built-sq-cosine" : "built-sq");
    index_status = "built";
    return index;
}

inline void set_packed_bits(uint8_t* output, size_t bit_offset, unsigned bits, uint8_t value) {
    const size_t byte = bit_offset >> 3;
    const unsigned shift = bit_offset & 7u;
    uint16_t word = output[byte];
    if (shift + bits > 8) word |= static_cast<uint16_t>(output[byte + 1]) << 8;
    const uint16_t mask = static_cast<uint16_t>(((1u << bits) - 1u) << shift);
    word = static_cast<uint16_t>((word & ~mask) | ((static_cast<uint16_t>(value) << shift) & mask));
    output[byte] = static_cast<uint8_t>(word & 0xffu);
    if (shift + bits > 8) output[byte + 1] = static_cast<uint8_t>(word >> 8);
}

inline uint8_t get_packed_bits(const uint8_t* input, size_t bit_offset, unsigned bits) {
    const size_t byte = bit_offset >> 3;
    const unsigned shift = bit_offset & 7u;
    uint16_t word = input[byte];
    if (shift + bits > 8) word |= static_cast<uint16_t>(input[byte + 1]) << 8;
    return static_cast<uint8_t>((word >> shift) & ((1u << bits) - 1u));
}

void pack_signs(const float* values, int d, uint8_t* output) {
    std::memset(output, 0, static_cast<size_t>(d + 7) / 8);
    for (int j = 0; j < d; ++j) {
        if (values[j] >= 0.0f) output[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
    }
}

struct TQMeta {
    int32_t d = 0;
    int32_t mse_bits = 1;
    int32_t rotation_seed = 0;
    int32_t qjl_seed = 0;
    std::vector<float> centroids;
    std::vector<float> rotation;
    std::vector<float> projection;
};

void save_tq_meta(const TQMeta& meta, const std::string& path) {
    make_parent(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("open failed: " + path);
    const char magic[8] = {'T', 'Q', 'P', 'R', 'O', 'D', '2', '\0'};
    out.write(magic, sizeof(magic));
    const int32_t header[5] = {
            meta.d, meta.mse_bits, meta.rotation_seed, meta.qjl_seed,
            static_cast<int32_t>(meta.centroids.size())};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(meta.centroids.data()), meta.centroids.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(meta.rotation.data()), meta.rotation.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(meta.projection.data()), meta.projection.size() * sizeof(float));
    if (!out) throw std::runtime_error("write failed: " + path);
}

TQMeta load_tq_meta(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    char magic[8];
    int32_t header[5];
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || std::memcmp(magic, "TQPROD2", 7) != 0) {
        throw std::runtime_error("bad TurboQuant metadata: " + path);
    }
    TQMeta meta;
    meta.d = header[0];
    meta.mse_bits = header[1];
    meta.rotation_seed = header[2];
    meta.qjl_seed = header[3];
    meta.centroids.resize(header[4]);
    meta.rotation.resize(static_cast<size_t>(meta.d) * meta.d);
    meta.projection.resize(static_cast<size_t>(meta.d) * meta.d);
    in.read(reinterpret_cast<char*>(meta.centroids.data()), meta.centroids.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(meta.rotation.data()), meta.rotation.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(meta.projection.data()), meta.projection.size() * sizeof(float));
    if (!in) throw std::runtime_error("truncated TurboQuant metadata: " + path);
    return meta;
}

TQMeta make_tq_meta(const Args& args, int d) {
    TQMeta meta;
    meta.d = d;
    meta.rotation_seed = args.rotation_seed;
    meta.qjl_seed = args.qjl_seed;
    std::vector<float> trained;
    faiss::scalar_quantizer::train_TurboQuantMSE(d, 1, trained);
    meta.centroids.assign(trained.begin(), trained.begin() + 2);
    faiss::RandomRotationMatrix rotation(d, d);
    rotation.init(args.rotation_seed);
    meta.rotation = rotation.A;
    meta.projection.resize(static_cast<size_t>(d) * d);
    faiss::float_randn(meta.projection.data(), meta.projection.size(), args.qjl_seed);
    return meta;
}

struct RaBitQMeta {
    int32_t d = 0;
    int32_t padded_dim = 0;
    int32_t num_clusters = 16;
    int32_t rotation_seed = 0;
    Metric metric = Metric::IP;
    std::vector<float> rotated_centroids;
    std::vector<char> rotator_state;
    std::unique_ptr<rabitqlib::Rotator<float>> rotator;
};

struct RotatedBQMeta {
    int32_t d = 0;
    int32_t padded_dim = 0;
    int32_t rotation_seed = 0;
    std::vector<char> rotator_state;
    std::unique_ptr<rabitqlib::Rotator<float>> rotator;
};

std::unique_ptr<rabitqlib::Rotator<float>> make_rabitq_rotator(
        int d,
        int padded_dim,
        const std::vector<char>& state) {
    auto rotator = std::unique_ptr<rabitqlib::Rotator<float>>(
            rabitqlib::choose_rotator<float>(
                    static_cast<size_t>(d),
                    rabitqlib::RotatorType::FhtKacRotator,
                    static_cast<size_t>(padded_dim)));
    if (state.size() != rotator->dump_bytes()) {
        throw std::runtime_error("RaBitQ rotator state size mismatch");
    }
    rotator->load(state.data());
    return rotator;
}

void save_rotated_bq_meta(const RotatedBQMeta& meta, const std::string& path) {
    make_parent(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("open failed: " + path);
    const char magic[8] = {'R', 'O', 'T', 'B', 'Q', '1', '\0', '\0'};
    const int32_t header[4] = {
            meta.d,
            meta.padded_dim,
            meta.rotation_seed,
            static_cast<int32_t>(meta.rotator_state.size())};
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(meta.rotator_state.data(), meta.rotator_state.size());
    if (!out) throw std::runtime_error("write failed: " + path);
}

RotatedBQMeta load_rotated_bq_meta(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    char magic[8];
    int32_t header[4];
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || std::memcmp(magic, "ROTBQ1", 6) != 0 || header[0] <= 0 ||
        header[1] <= 0 || header[3] <= 0) {
        throw std::runtime_error("bad rotated BQ metadata: " + path);
    }
    RotatedBQMeta meta;
    meta.d = header[0];
    meta.padded_dim = header[1];
    meta.rotation_seed = header[2];
    meta.rotator_state.resize(header[3]);
    in.read(meta.rotator_state.data(), meta.rotator_state.size());
    if (!in) throw std::runtime_error("truncated rotated BQ metadata: " + path);
    meta.rotator = make_rabitq_rotator(meta.d, meta.padded_dim, meta.rotator_state);
    return meta;
}

RotatedBQMeta make_rotated_bq_meta(const Args& args, int d) {
    RotatedBQMeta meta;
    meta.d = d;
    meta.padded_dim = static_cast<int32_t>(
            rabitqlib::round_up_to_multiple(static_cast<size_t>(d), 64));
    meta.rotation_seed = args.rotation_seed;
    meta.rotator = std::unique_ptr<rabitqlib::Rotator<float>>(
            rabitqlib::choose_rotator<float>(
                    static_cast<size_t>(d),
                    rabitqlib::RotatorType::FhtKacRotator,
                    static_cast<size_t>(meta.padded_dim)));
    meta.rotator_state.resize(meta.rotator->dump_bytes());
    std::mt19937 generator(static_cast<uint32_t>(args.rotation_seed));
    std::uniform_int_distribution<int> distribution(0, 255);
    for (char& value : meta.rotator_state) {
        value = static_cast<char>(distribution(generator));
    }
    meta.rotator->load(meta.rotator_state.data());
    return meta;
}

std::vector<float> read_fbin_matrix(
        const std::string& path, int expected_rows, int expected_dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    int32_t rows = 0;
    int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || rows != expected_rows || dim != expected_dim) {
        throw std::runtime_error("unexpected centroid shape: " + path);
    }
    std::vector<float> result(static_cast<size_t>(rows) * dim);
    in.read(reinterpret_cast<char*>(result.data()), result.size() * sizeof(float));
    if (!in) throw std::runtime_error("truncated centroid file: " + path);
    return result;
}

void save_rabitq_meta(const RaBitQMeta& meta, const std::string& path) {
    make_parent(path);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("open failed: " + path);
    const char magic[8] = {'R', 'A', 'B', 'I', 'T', 'Q', '1', '\0'};
    out.write(magic, sizeof(magic));
    const int32_t header[6] = {
            meta.d,
            meta.padded_dim,
            meta.num_clusters,
            meta.rotation_seed,
            static_cast<int32_t>(meta.metric),
            static_cast<int32_t>(meta.rotator_state.size())};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(
            reinterpret_cast<const char*>(meta.rotated_centroids.data()),
            meta.rotated_centroids.size() * sizeof(float));
    out.write(meta.rotator_state.data(), meta.rotator_state.size());
    if (!out) throw std::runtime_error("write failed: " + path);
}

RaBitQMeta load_rabitq_meta(const std::string& path, Metric expected_metric) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    char magic[8];
    int32_t header[6];
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || std::memcmp(magic, "RABITQ1", 7) != 0) {
        throw std::runtime_error("bad RaBitQ metadata: " + path);
    }
    RaBitQMeta meta;
    meta.d = header[0];
    meta.padded_dim = header[1];
    meta.num_clusters = header[2];
    meta.rotation_seed = header[3];
    meta.metric = static_cast<Metric>(header[4]);
    if (meta.metric != expected_metric || meta.d <= 0 || meta.padded_dim <= 0 ||
        meta.num_clusters <= 0 || header[5] <= 0) {
        throw std::runtime_error("RaBitQ metadata configuration mismatch: " + path);
    }
    meta.rotated_centroids.resize(
            static_cast<size_t>(meta.num_clusters) * meta.padded_dim);
    meta.rotator_state.resize(header[5]);
    in.read(
            reinterpret_cast<char*>(meta.rotated_centroids.data()),
            meta.rotated_centroids.size() * sizeof(float));
    in.read(meta.rotator_state.data(), meta.rotator_state.size());
    if (!in) throw std::runtime_error("truncated RaBitQ metadata: " + path);
    meta.rotator = make_rabitq_rotator(meta.d, meta.padded_dim, meta.rotator_state);
    return meta;
}

RaBitQMeta make_rabitq_meta(
        const Args& args,
        int d,
        const std::vector<float>& centroids) {
    RaBitQMeta meta;
    meta.d = d;
    meta.padded_dim = static_cast<int32_t>(
            rabitqlib::round_up_to_multiple(static_cast<size_t>(d), 64));
    meta.num_clusters = 16;
    meta.rotation_seed = args.rotation_seed;
    meta.metric = args.metric;
    meta.rotator = std::unique_ptr<rabitqlib::Rotator<float>>(
            rabitqlib::choose_rotator<float>(
                    static_cast<size_t>(d),
                    rabitqlib::RotatorType::FhtKacRotator,
                    static_cast<size_t>(meta.padded_dim)));
    meta.rotator_state.resize(meta.rotator->dump_bytes());
    std::mt19937 generator(static_cast<uint32_t>(args.rotation_seed));
    std::uniform_int_distribution<int> distribution(0, 255);
    for (char& value : meta.rotator_state) {
        value = static_cast<char>(distribution(generator));
    }
    meta.rotator->load(meta.rotator_state.data());
    meta.rotated_centroids.resize(
            static_cast<size_t>(meta.num_clusters) * meta.padded_dim);
    for (int cluster = 0; cluster < meta.num_clusters; ++cluster) {
        meta.rotator->rotate(
                centroids.data() + static_cast<size_t>(cluster) * d,
                meta.rotated_centroids.data() +
                        static_cast<size_t>(cluster) * meta.padded_dim);
    }
    return meta;
}

size_t sign_bytes(int d) { return static_cast<size_t>(d + 7) / 8; }
size_t tq_mse_bytes(int d) { return sign_bytes(d); }
size_t tq_code_size(int d, Metric metric) {
    return tq_mse_bytes(d) + sign_bytes(d) +
            (metric == Metric::IP ? sizeof(float) : 2 * sizeof(float));
}
size_t bq_code_size(int d, Metric metric) {
    return sign_bytes(d) + (metric == Metric::IP ? 0 : sizeof(float));
}
size_t symmetric_bq_code_size(int d) { return sign_bytes(d); }
size_t rabitq_payload_size(int padded_dim) {
    return rabitqlib::BinDataMap<float>::data_bytes(static_cast<size_t>(padded_dim));
}
size_t rabitq_code_size(int padded_dim) { return rabitq_payload_size(padded_dim) + 1; }

void write_float(uint8_t* dst, float value) { std::memcpy(dst, &value, sizeof(value)); }
float read_float(const uint8_t* src) {
    float value;
    std::memcpy(&value, src, sizeof(value));
    return value;
}

void normalize_with_norms(std::vector<float>& block, int64_t rows, int d, std::vector<float>& norms) {
    norms.resize(rows);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        float* x = block.data() + static_cast<size_t>(i) * d;
        double ss = 0.0;
        for (int j = 0; j < d; ++j) ss += static_cast<double>(x[j]) * x[j];
        const float norm = static_cast<float>(std::sqrt(std::max(ss, 1e-30)));
        norms[i] = norm;
        const float inv = 1.0f / norm;
        for (int j = 0; j < d; ++j) x[j] *= inv;
    }
}

void encode_tq_block(
        const float* source,
        int64_t rows,
        int d,
        const TQMeta& meta,
        Metric metric,
        std::vector<uint8_t>& output) {
    std::vector<float> unit(source, source + static_cast<size_t>(rows) * d);
    std::vector<float> norms;
    normalize_with_norms(unit, rows, d, norms);
    std::vector<float> rotated(static_cast<size_t>(rows) * d);
    std::vector<float> reconstructed_rot(static_cast<size_t>(rows) * d);
    std::vector<float> reconstructed(static_cast<size_t>(rows) * d);
    std::vector<float> residual(static_cast<size_t>(rows) * d);
    std::vector<float> projected(static_cast<size_t>(rows) * d);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rows, d, d, 1.0f, unit.data(), d, meta.rotation.data(), d,
                0.0f, rotated.data(), d);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        for (int j = 0; j < d; ++j) {
            const float value = rotated[static_cast<size_t>(i) * d + j];
            const int best = value >= 0.0f ? 1 : 0;
            reconstructed_rot[static_cast<size_t>(i) * d + j] = meta.centroids[best];
        }
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, d, d, 1.0f, reconstructed_rot.data(), d, meta.rotation.data(), d,
                0.0f, reconstructed.data(), d);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows * static_cast<int64_t>(d); ++i) {
        residual[i] = unit[i] - reconstructed[i];
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rows, d, d, 1.0f, residual.data(), d, meta.projection.data(), d,
                0.0f, projected.data(), d);

    const size_t mse_bytes = tq_mse_bytes(d);
    const size_t qjl_bytes = sign_bytes(d);
    const size_t code_size = tq_code_size(d, metric);
    output.assign(static_cast<size_t>(rows) * code_size, 0);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        uint8_t* code = output.data() + static_cast<size_t>(i) * code_size;
        double residual_ss = 0.0;
        for (int j = 0; j < d; ++j) {
            const float value = rotated[static_cast<size_t>(i) * d + j];
            if (value >= 0.0f) code[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
            const float r = residual[static_cast<size_t>(i) * d + j];
            residual_ss += static_cast<double>(r) * r;
        }
        pack_signs(projected.data() + static_cast<size_t>(i) * d, d, code + mse_bytes);
        write_float(code + mse_bytes + qjl_bytes, static_cast<float>(std::sqrt(residual_ss)));
        if (metric == Metric::L2) {
            write_float(code + mse_bytes + qjl_bytes + sizeof(float), norms[i]);
        }
    }
}

void encode_bq_block(
        const float* source,
        int64_t rows,
        int d,
        Metric metric,
        std::vector<uint8_t>& output) {
    const size_t bits_bytes = sign_bytes(d);
    const size_t code_size = bq_code_size(d, metric);
    output.assign(static_cast<size_t>(rows) * code_size, 0);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        const float* x = source + static_cast<size_t>(i) * d;
        uint8_t* code = output.data() + static_cast<size_t>(i) * code_size;
        double abs_sum = 0.0;
        for (int j = 0; j < d; ++j) {
            if (x[j] >= 0.0f) code[j >> 3] |= static_cast<uint8_t>(1u << (j & 7));
            abs_sum += std::abs(static_cast<double>(x[j]));
        }
        if (metric == Metric::L2) {
            write_float(code + bits_bytes, static_cast<float>(abs_sum / d));
        }
    }
}

void encode_symmetric_bq_block(
        const float* source,
        int64_t rows,
        int d,
        std::vector<uint8_t>& output) {
    const size_t code_size = symmetric_bq_code_size(d);
    output.assign(static_cast<size_t>(rows) * code_size, 0);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < rows; ++i) {
        pack_signs(
                source + static_cast<size_t>(i) * d,
                d,
                output.data() + static_cast<size_t>(i) * code_size);
    }
}

void encode_rotated_symmetric_bq_block(
        const float* source,
        int64_t rows,
        int d,
        const RotatedBQMeta& meta,
        std::vector<uint8_t>& output) {
    const size_t code_size = symmetric_bq_code_size(meta.padded_dim);
    output.assign(static_cast<size_t>(rows) * code_size, 0);
#pragma omp parallel
    {
        std::vector<float> rotated(static_cast<size_t>(meta.padded_dim));
#pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            meta.rotator->rotate(
                    source + static_cast<size_t>(i) * d, rotated.data());
            pack_signs(
                    rotated.data(),
                    meta.padded_dim,
                    output.data() + static_cast<size_t>(i) * code_size);
        }
    }
}

float vector_dot(const float* left, const float* right, int d) {
    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
    for (int j = 0; j < d; ++j) sum += left[j] * right[j];
    return sum;
}

float vector_l2_squared(const float* left, const float* right, int d) {
    float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
    for (int j = 0; j < d; ++j) {
        const float delta = left[j] - right[j];
        sum += delta * delta;
    }
    return sum;
}

uint8_t nearest_rabitq_centroid(
        const float* vector,
        const std::vector<float>& centroids,
        int num_clusters,
        int d,
        Metric metric) {
    int best = 0;
    float best_value = metric == Metric::IP
            ? -std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::infinity();
    for (int cluster = 0; cluster < num_clusters; ++cluster) {
        const float* centroid = centroids.data() + static_cast<size_t>(cluster) * d;
        const float value = metric == Metric::IP
                ? vector_dot(vector, centroid, d)
                : vector_l2_squared(vector, centroid, d);
        if ((metric == Metric::IP && value > best_value) ||
            (metric == Metric::L2 && value < best_value)) {
            best = cluster;
            best_value = value;
        }
    }
    return static_cast<uint8_t>(best);
}

void encode_rabitq_block(
        const float* source,
        int64_t rows,
        int d,
        const std::vector<float>& centroids,
        const RaBitQMeta& meta,
        std::vector<uint8_t>& output) {
    const size_t payload_size = rabitq_payload_size(meta.padded_dim);
    const size_t code_size = rabitq_code_size(meta.padded_dim);
    output.assign(static_cast<size_t>(rows) * code_size, 0);
    const rabitqlib::MetricType rq_metric = meta.metric == Metric::IP
            ? rabitqlib::METRIC_IP
            : rabitqlib::METRIC_L2;
#pragma omp parallel
    {
        std::vector<float> prepared(static_cast<size_t>(d));
        std::vector<float> rotated(static_cast<size_t>(meta.padded_dim));
#pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            const float* input = source + static_cast<size_t>(i) * d;
            if (meta.metric == Metric::IP) {
                double norm2 = 0.0;
                for (int j = 0; j < d; ++j) {
                    norm2 += static_cast<double>(input[j]) * input[j];
                }
                const float inv = 1.0f /
                        static_cast<float>(std::sqrt(std::max(norm2, 1e-30)));
                for (int j = 0; j < d; ++j) prepared[j] = input[j] * inv;
            } else {
                std::memcpy(prepared.data(), input, static_cast<size_t>(d) * sizeof(float));
            }
            const uint8_t cluster = nearest_rabitq_centroid(
                    prepared.data(), centroids, meta.num_clusters, d, meta.metric);
            meta.rotator->rotate(prepared.data(), rotated.data());
            uint8_t* code = output.data() + static_cast<size_t>(i) * code_size;
            rabitqlib::quant::quantize_compact_one_bit(
                    rotated.data(),
                    meta.rotated_centroids.data() +
                            static_cast<size_t>(cluster) * meta.padded_dim,
                    static_cast<size_t>(meta.padded_dim),
                    reinterpret_cast<char*>(code),
                    rq_metric);
            code[payload_size] = cluster;
        }
    }
}

std::unique_ptr<faiss::IndexBinaryHNSW> build_or_load_custom(
        const Args& args,
        const FbinMmap& train,
        const faiss::IndexBinaryHNSW& graph,
        TQMeta* tq_meta,
        RotatedBQMeta* rotated_bq_meta,
        RaBitQMeta* rabitq_meta,
        double& setup_seconds,
        double& encode_seconds,
        std::string& status) {
    if (fs::exists(args.index) && !args.rebuild) {
        const auto start = std::chrono::steady_clock::now();
        auto index = load_binary_hnsw(args.index);
        if (args.method == Method::TQProd2) *tq_meta = load_tq_meta(args.meta);
        if (args.method == Method::RotBQ1) {
            *rotated_bq_meta = load_rotated_bq_meta(args.meta);
            if (rotated_bq_meta->d != train.d) {
                throw std::runtime_error("rotated BQ metadata dimension mismatch");
            }
        }
        if (args.method == Method::RaBitQ1) {
            *rabitq_meta = load_rabitq_meta(args.meta, args.metric);
            if (rabitq_meta->d != train.d) {
                throw std::runtime_error("RaBitQ metadata dimension mismatch");
            }
        }
        setup_seconds = seconds_since(start);
        encode_seconds = 0.0;
        status = "loaded";
        return index;
    }

    const auto setup_start = std::chrono::steady_clock::now();
    size_t code_size;
    std::vector<float> rabitq_centroids;
    if (args.method == Method::TQProd2) {
        if (!args.reference_meta.empty()) {
            *tq_meta = load_tq_meta(args.reference_meta);
            if (tq_meta->d != train.d || tq_meta->mse_bits != 1) {
                throw std::runtime_error("TurboQuant reference metadata does not match d/bit-width");
            }
        } else {
            *tq_meta = make_tq_meta(args, train.d);
        }
        save_tq_meta(*tq_meta, args.meta);
        code_size = tq_code_size(train.d, args.metric);
    } else if (args.method == Method::BQ1) {
        code_size = bq_code_size(train.d, args.metric);
    } else if (args.method == Method::SymBQ1) {
        code_size = symmetric_bq_code_size(train.d);
    } else if (args.method == Method::RotBQ1) {
        *rotated_bq_meta = make_rotated_bq_meta(args, train.d);
        save_rotated_bq_meta(*rotated_bq_meta, args.meta);
        code_size = symmetric_bq_code_size(rotated_bq_meta->padded_dim);
    } else if (args.method == Method::RaBitQ1) {
        if (args.rabitq_centroids.empty()) {
            throw std::runtime_error("RaBitQ build requires --rabitq-centroids");
        }
        constexpr int num_clusters = 16;
        rabitq_centroids = read_fbin_matrix(
                args.rabitq_centroids, num_clusters, train.d);
        *rabitq_meta = make_rabitq_meta(args, train.d, rabitq_centroids);
        save_rabitq_meta(*rabitq_meta, args.meta);
        code_size = rabitq_code_size(rabitq_meta->padded_dim);
    } else {
        throw std::runtime_error(
                "custom builder only supports TQProd2/BQ1/SymBQ1/RotBQ1/RaBitQ1");
    }
    setup_seconds = seconds_since(setup_start);

    auto* storage = new faiss::IndexBinaryFlat(code_size * 8);
    const auto encode_start = std::chrono::steady_clock::now();
    omp_set_num_threads(args.build_threads);
    for (int64_t begin = 0; begin < train.n; begin += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(train.n, begin + args.add_batch_size);
        std::vector<uint8_t> codes;
        if (args.method == Method::TQProd2) {
            encode_tq_block(
                    train.row(begin), end - begin, train.d, *tq_meta, args.metric, codes);
        } else if (args.method == Method::RaBitQ1) {
            encode_rabitq_block(
                    train.row(begin),
                    end - begin,
                    train.d,
                    rabitq_centroids,
                    *rabitq_meta,
                    codes);
        } else if (args.method == Method::BQ1) {
            encode_bq_block(train.row(begin), end - begin, train.d, args.metric, codes);
        } else if (args.method == Method::RotBQ1) {
            encode_rotated_symmetric_bq_block(
                    train.row(begin), end - begin, train.d, *rotated_bq_meta, codes);
        } else {
            encode_symmetric_bq_block(train.row(begin), end - begin, train.d, codes);
        }
        storage->add(end - begin, codes.data());
        std::cout << "  " << method_name(args.method) << " encoded " << end << "/" << train.n
                  << std::endl;
    }
    auto index = std::make_unique<faiss::IndexBinaryHNSW>(storage, args.m);
    index->own_fields = true;
    index->ntotal = storage->ntotal;
    copy_graph(graph, *index);
    encode_seconds = seconds_since(encode_start);
    make_parent(args.index);
    faiss::write_index_binary(index.get(), args.index.c_str());
    status = "built";
    return index;
}

std::unique_ptr<faiss::IndexBinaryHNSW> build_or_load_native_bq_graph(
        const Args& args,
        const FbinMmap& train,
        RotatedBQMeta* rotated_bq_meta,
        double& setup_seconds,
        double& encode_seconds,
        double& graph_seconds,
        std::string& graph_status,
        std::string& index_status) {
    if (args.method != Method::SymBQ1 && args.method != Method::RotBQ1) {
        throw std::runtime_error(
                "native BQ graph builder only supports SymBQ1/RotBQ1");
    }

    if (fs::exists(args.index) && !args.rebuild && !args.rebuild_graph) {
        const auto load_start = std::chrono::steady_clock::now();
        auto index = load_binary_hnsw(args.index);
        if (index->ntotal != train.n) {
            throw std::runtime_error("native BQ graph ntotal mismatch");
        }
        if (args.method == Method::RotBQ1) {
            *rotated_bq_meta = load_rotated_bq_meta(args.meta);
            if (rotated_bq_meta->d != train.d) {
                throw std::runtime_error("rotated BQ metadata dimension mismatch");
            }
        }
        graph_seconds = seconds_since(load_start);
        setup_seconds = 0.0;
        encode_seconds = 0.0;
        graph_status = "loaded-bq";
        index_status = "loaded";
        return index;
    }

    const auto setup_start = std::chrono::steady_clock::now();
    size_t code_size = 0;
    if (args.method == Method::RotBQ1) {
        *rotated_bq_meta = make_rotated_bq_meta(args, train.d);
        save_rotated_bq_meta(*rotated_bq_meta, args.meta);
        code_size = symmetric_bq_code_size(rotated_bq_meta->padded_dim);
    } else {
        code_size = symmetric_bq_code_size(train.d);
    }
    setup_seconds = seconds_since(setup_start);

    std::cout << "Encoding DB for native BQ HNSW: n=" << train.n
              << " code_bytes=" << code_size << std::endl;
    const auto encode_start = std::chrono::steady_clock::now();
    std::vector<uint8_t> codes;
    omp_set_num_threads(args.build_threads);
    if (args.method == Method::RotBQ1) {
        encode_rotated_symmetric_bq_block(
                train.data, train.n, train.d, *rotated_bq_meta, codes);
    } else {
        encode_symmetric_bq_block(train.data, train.n, train.d, codes);
    }
    encode_seconds = seconds_since(encode_start);

    std::cout << "Building native BQ HNSW graph: n=" << train.n
              << " bits=" << code_size * 8 << " M=" << args.m
              << " efC=" << args.ef_construction << std::endl;
    const auto graph_start = std::chrono::steady_clock::now();
    auto* storage = new faiss::IndexBinaryFlat(code_size * 8);
    auto index = std::make_unique<faiss::IndexBinaryHNSW>(storage, args.m);
    index->own_fields = true;
    index->hnsw.efConstruction = args.ef_construction;
    index->retain_locks = true;
    for (int64_t begin = 0; begin < train.n; begin += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(train.n, begin + args.add_batch_size);
        index->add(
                end - begin,
                codes.data() + static_cast<size_t>(begin) * code_size);
        std::cout << "  native BQ graph " << end << "/" << train.n << std::endl;
    }
    index->retain_locks = false;
    make_parent(args.index);
    faiss::write_index_binary(index.get(), args.index.c_str());
    graph_seconds = seconds_since(graph_start);
    graph_status = "built-bq";
    index_status = "built";
    return index;
}

const uint8_t* binary_codes(const faiss::IndexBinaryHNSW& index) {
    auto* storage = dynamic_cast<faiss::IndexBinaryFlat*>(index.storage);
    if (!storage) throw std::runtime_error("binary HNSW storage is not flat");
    return storage->xb.data();
}

void make_sign_lut(const float* query, int d, std::vector<float>& lut) {
    const size_t bytes = sign_bytes(d);
    lut.assign(bytes * 32, 0.0f);
    for (size_t byte = 0; byte < bytes; ++byte) {
        for (int half = 0; half < 2; ++half) {
            for (int pattern = 0; pattern < 16; ++pattern) {
                float sum = 0.0f;
                for (int bit = 0; bit < 4; ++bit) {
                    const int j = static_cast<int>(byte) * 8 + half * 4 + bit;
                    if (j >= d) break;
                    sum += ((pattern >> bit) & 1) ? query[j] : -query[j];
                }
                lut[byte * 32 + half * 16 + pattern] = sum;
            }
        }
    }
}

float sign_dot(const uint8_t* code, size_t bytes, const std::vector<float>& lut) {
    float sum = 0.0f;
    for (size_t i = 0; i < bytes; ++i) {
        const uint8_t value = code[i];
        const float* row = lut.data() + i * 32;
        sum += row[value & 15] + row[16 + (value >> 4)];
    }
    return sum;
}

struct BQDistanceComputer : faiss::DistanceComputer {
    const uint8_t* codes;
    size_t code_size;
    int d;
    Metric metric;
    size_t bits_bytes;
    std::vector<float> lut;
    float query_norm2 = 0.0f;

    BQDistanceComputer(const uint8_t* codes_in, size_t code_size_in, int d_in, Metric metric_in)
            : codes(codes_in), code_size(code_size_in), d(d_in), metric(metric_in),
              bits_bytes(sign_bytes(d_in)) {}

    void set_query(const float* query) override {
        make_sign_lut(query, d, lut);
        query_norm2 = 0.0f;
        for (int j = 0; j < d; ++j) query_norm2 += query[j] * query[j];
    }

    float distance_to_code(const uint8_t* code) {
        const float sign_score = sign_dot(code, bits_bytes, lut);
        if (metric == Metric::IP) return -sign_score;
        const float alpha = read_float(code + bits_bytes);
        const float dot = alpha * sign_score;
        return query_norm2 + d * alpha * alpha - 2.0f * dot;
    }

    float operator()(faiss::idx_t i) override { return distance_to_code(codes + i * code_size); }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint8_t* a = codes + i * code_size;
        const uint8_t* b = codes + j * code_size;
        int hamming = 0;
        for (size_t k = 0; k < bits_bytes; ++k) hamming += __builtin_popcount(a[k] ^ b[k]);
        return static_cast<float>(hamming);
    }
};

struct SymmetricBQDistanceComputer : faiss::DistanceComputer {
    const uint8_t* codes;
    size_t code_size;
    int d;
    std::vector<uint8_t> query_code;

    SymmetricBQDistanceComputer(const uint8_t* codes_in, size_t code_size_in, int d_in)
            : codes(codes_in), code_size(code_size_in), d(d_in), query_code(code_size_in) {}

    void set_query(const float* query) override {
        pack_signs(query, d, query_code.data());
    }

    float distance_to_code(const uint8_t* code) const {
        size_t byte = 0;
        int hamming = 0;
        for (; byte + sizeof(uint64_t) <= code_size; byte += sizeof(uint64_t)) {
            uint64_t left = 0;
            uint64_t right = 0;
            std::memcpy(&left, query_code.data() + byte, sizeof(left));
            std::memcpy(&right, code + byte, sizeof(right));
            hamming += __builtin_popcountll(left ^ right);
        }
        for (; byte < code_size; ++byte) {
            hamming += __builtin_popcount(
                    static_cast<unsigned>(query_code[byte] ^ code[byte]));
        }
        return static_cast<float>(hamming);
    }

    float operator()(faiss::idx_t i) override {
        return distance_to_code(codes + static_cast<size_t>(i) * code_size);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint8_t* left = codes + static_cast<size_t>(i) * code_size;
        const uint8_t* right = codes + static_cast<size_t>(j) * code_size;
        int hamming = 0;
        size_t byte = 0;
        for (; byte + sizeof(uint64_t) <= code_size; byte += sizeof(uint64_t)) {
            uint64_t a = 0;
            uint64_t b = 0;
            std::memcpy(&a, left + byte, sizeof(a));
            std::memcpy(&b, right + byte, sizeof(b));
            hamming += __builtin_popcountll(a ^ b);
        }
        for (; byte < code_size; ++byte) {
            hamming += __builtin_popcount(static_cast<unsigned>(left[byte] ^ right[byte]));
        }
        return static_cast<float>(hamming);
    }
};

struct RotatedSymmetricBQDistanceComputer : SymmetricBQDistanceComputer {
    const RotatedBQMeta& meta;
    std::vector<float> rotated_query;

    RotatedSymmetricBQDistanceComputer(
            const uint8_t* codes_in,
            size_t code_size_in,
            const RotatedBQMeta& meta_in)
            : SymmetricBQDistanceComputer(
                      codes_in, code_size_in, meta_in.padded_dim),
              meta(meta_in),
              rotated_query(static_cast<size_t>(meta_in.padded_dim)) {}

    void set_query(const float* query) override {
        meta.rotator->rotate(query, rotated_query.data());
        SymmetricBQDistanceComputer::set_query(rotated_query.data());
    }
};

struct RaBitQDistanceComputer : faiss::DistanceComputer {
    const uint8_t* codes;
    size_t code_size;
    size_t payload_size;
    int d;
    const RaBitQMeta& meta;
    std::vector<float> prepared_query;
    std::vector<float> rotated_query;
    std::vector<float> g_add;
    float k1xsumq = 0.0f;

    RaBitQDistanceComputer(
            const uint8_t* codes_in,
            size_t code_size_in,
            int d_in,
            const RaBitQMeta& meta_in)
            : codes(codes_in),
              code_size(code_size_in),
              payload_size(rabitq_payload_size(meta_in.padded_dim)),
              d(d_in),
              meta(meta_in),
              prepared_query(d_in),
              rotated_query(meta_in.padded_dim),
              g_add(meta_in.num_clusters) {}

    void set_query(const float* query) override {
        if (meta.metric == Metric::IP) {
            double norm2 = 0.0;
            for (int j = 0; j < d; ++j) {
                norm2 += static_cast<double>(query[j]) * query[j];
            }
            const float inv = 1.0f /
                    static_cast<float>(std::sqrt(std::max(norm2, 1e-30)));
            for (int j = 0; j < d; ++j) prepared_query[j] = query[j] * inv;
        } else {
            std::memcpy(
                    prepared_query.data(), query, static_cast<size_t>(d) * sizeof(float));
        }
        meta.rotator->rotate(prepared_query.data(), rotated_query.data());
        k1xsumq = -0.5f * std::accumulate(
                rotated_query.begin(), rotated_query.end(), 0.0f);
        for (int cluster = 0; cluster < meta.num_clusters; ++cluster) {
            const float* centroid = meta.rotated_centroids.data() +
                    static_cast<size_t>(cluster) * meta.padded_dim;
            g_add[cluster] = meta.metric == Metric::IP
                    ? -vector_dot(rotated_query.data(), centroid, meta.padded_dim)
                    : vector_l2_squared(rotated_query.data(), centroid, meta.padded_dim);
        }
    }

    float distance_to_code(const uint8_t* code_bytes) {
        const uint8_t cluster = code_bytes[payload_size];
        if (cluster >= meta.num_clusters) {
            return std::numeric_limits<float>::infinity();
        }
        rabitqlib::ConstBinDataMap<float> code(
                reinterpret_cast<const char*>(code_bytes),
                static_cast<size_t>(meta.padded_dim));
        const float query_code_ip = rabitqlib::mask_ip_x0_q(
                rotated_query.data(), code.bin_code(), static_cast<size_t>(meta.padded_dim));
        return code.f_add() + g_add[cluster] +
                code.f_rescale() * (query_code_ip + k1xsumq);
    }

    float operator()(faiss::idx_t i) override {
        return distance_to_code(codes + static_cast<size_t>(i) * code_size);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint8_t* left = codes + static_cast<size_t>(i) * code_size;
        const uint8_t* right = codes + static_cast<size_t>(j) * code_size;
        int hamming = 0;
        for (int byte = 0; byte < meta.padded_dim / 8; ++byte) {
            hamming += __builtin_popcount(
                    static_cast<unsigned>(left[byte] ^ right[byte]));
        }
        return static_cast<float>(hamming);
    }
};

struct TQDistanceComputer : faiss::DistanceComputer {
    const uint8_t* codes;
    size_t code_size;
    int d;
    Metric metric;
    const TQMeta& meta;
    size_t mse_bytes;
    size_t qjl_bytes;
    std::vector<float> unit_query;
    std::vector<float> rotated_query;
    std::vector<float> projected_query;
    std::vector<float> mse_lut;
    std::vector<float> qjl_lut;
    float query_norm = 1.0f;

    TQDistanceComputer(
            const uint8_t* codes_in,
            size_t code_size_in,
            int d_in,
            Metric metric_in,
            const TQMeta& meta_in)
            : codes(codes_in), code_size(code_size_in), d(d_in), metric(metric_in), meta(meta_in),
              mse_bytes(tq_mse_bytes(d_in)), qjl_bytes(sign_bytes(d_in)), unit_query(d_in),
              rotated_query(d_in), projected_query(d_in) {}

    void set_query(const float* query) override {
        double ss = 0.0;
        for (int j = 0; j < d; ++j) ss += static_cast<double>(query[j]) * query[j];
        query_norm = static_cast<float>(std::sqrt(std::max(ss, 1e-30)));
        for (int j = 0; j < d; ++j) unit_query[j] = query[j] / query_norm;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, d, d, 1.0f,
                    meta.rotation.data(), d, unit_query.data(), 1, 0.0f, rotated_query.data(), 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, d, d, 1.0f,
                    meta.projection.data(), d, unit_query.data(), 1, 0.0f, projected_query.data(), 1);
        make_sign_lut(rotated_query.data(), d, mse_lut);
        make_sign_lut(projected_query.data(), d, qjl_lut);
    }

    float distance_to_code(const uint8_t* code) {
        const float mse_score = meta.centroids[1] * sign_dot(code, mse_bytes, mse_lut);
        const uint8_t* qjl = code + mse_bytes;
        const float gamma = read_float(qjl + qjl_bytes);
        const float source_norm = metric == Metric::IP
                ? 1.0f
                : read_float(qjl + qjl_bytes + sizeof(float));
        constexpr float sqrt_pi_over_2 = 1.2533141373155001f;
        const float qjl_score = (sqrt_pi_over_2 / d) * gamma * sign_dot(qjl, qjl_bytes, qjl_lut);
        const float unit_ip = mse_score + qjl_score;
        const float ip = query_norm * source_norm * unit_ip;
        if (metric == Metric::IP) return -ip;
        return query_norm * query_norm + source_norm * source_norm - 2.0f * ip;
    }

    float operator()(faiss::idx_t i) override { return distance_to_code(codes + i * code_size); }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint8_t* a = codes + i * code_size;
        const uint8_t* b = codes + j * code_size;
        int hamming = 0;
        for (size_t k = 0; k < mse_bytes + qjl_bytes; ++k) hamming += __builtin_popcount(a[k] ^ b[k]);
        return static_cast<float>(hamming);
    }
};

struct QdrantTqIndexDeleter {
    void operator()(QdrantTqIndex* index) const { qdrant_tq_free(index); }
};

using QdrantTqIndexPtr = std::unique_ptr<QdrantTqIndex, QdrantTqIndexDeleter>;

std::string qdrant_tq_error() {
    const char* message = qdrant_tq_last_error();
    return message && *message ? message : "unknown Qdrant TurboQuant error";
}

QdrantTqIndexPtr build_or_load_qdrant_tq(
        const Args& args,
        const FbinMmap& train,
        double& setup_seconds,
        double& encode_seconds,
        std::string& status) {
    make_parent(args.index);
    make_parent(args.meta);
    if (fs::exists(args.index) && fs::exists(args.meta) && !args.rebuild) {
        const auto start = std::chrono::steady_clock::now();
        QdrantTqIndex* raw = qdrant_tq_load(args.index.c_str(), args.meta.c_str());
        if (!raw) throw std::runtime_error(qdrant_tq_error());
        setup_seconds = seconds_since(start);
        encode_seconds = 0.0;
        status = "loaded";
        return QdrantTqIndexPtr(raw);
    }

    const auto start = std::chrono::steady_clock::now();
    QdrantTqIndex* raw = qdrant_tq_build(
            train.data,
            static_cast<size_t>(train.n),
            static_cast<size_t>(train.d),
            args.metric == Metric::IP ? 0 : 1,
            static_cast<size_t>(args.build_threads),
            args.index.c_str(),
            args.meta.c_str());
    if (!raw) throw std::runtime_error(qdrant_tq_error());
    setup_seconds = 0.0;
    encode_seconds = seconds_since(start);
    status = "built";
    return QdrantTqIndexPtr(raw);
}

struct QdrantTqDirectDistanceComputer : faiss::DistanceComputer {
    const QdrantTqIndex* index;
    const uint8_t* codes;
    size_t code_size;
    size_t code_bytes;
    size_t padded_dim;
    size_t query_plane_len;
    int d;
    Metric metric;
    QdrantTqQuery* query = nullptr;
    std::vector<int8_t> query_planes;
    float postprocess_scale = 0.0f;
    int64_t bias_correction = 0;
    float ec_correction = 0.0f;
    float query_l2 = 0.0f;

    QdrantTqDirectDistanceComputer(
            const QdrantTqIndex* index_in,
            const uint8_t* codes_in,
            size_t code_size_in,
            int d_in,
            Metric metric_in)
            : index(index_in),
              codes(codes_in),
              code_size(code_size_in),
              code_bytes((static_cast<size_t>(d_in) + 7) / 8),
              padded_dim(code_bytes * 8),
              query_plane_len(((code_bytes + 63) / 64) * 64),
              d(d_in),
              metric(metric_in),
              query_planes(2 * 8 * query_plane_len) {}

    ~QdrantTqDirectDistanceComputer() override { qdrant_tq_query_free(query); }

    void set_query(const float* values) override {
        qdrant_tq_query_free(query);
        query = qdrant_tq_query_new(index, values, static_cast<size_t>(d));
        if (!query) throw std::runtime_error(qdrant_tq_error());
        if (!qdrant_tq_query_export_bits1_wide_planes(
                    query,
                    query_planes.data(),
                    query_planes.size(),
                    &postprocess_scale,
                    &bias_correction,
                    &ec_correction,
                    &query_l2)) {
            throw std::runtime_error("failed to export Qdrant TQ1+ wide query");
        }
    }

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
                std::memcpy(tail, source, remaining);
                source = tail;
            }
            __m512i shifted = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(source));
            for (size_t k = 0; k < 8; ++k) {
                const __m512i values = _mm512_shuffle_epi8(
                        codebook, _mm512_and_si512(shifted, mask));
                const __m512i query_low = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(
                                planes + k * plane_len + offset));
                const __m512i query_high = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(
                                planes + (8 + k) * plane_len + offset));
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

    int64_t raw_dot(const uint8_t* code) const {
#if defined(__x86_64__)
        if (__builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vnni")) {
            return raw_dot_avx512_vnni(
                    query_planes.data(), query_plane_len, code, code_bytes);
        }
#endif
        throw std::runtime_error("Qdrant TQ1+ direct scorer requires AVX-512 VNNI");
    }

    float distance_to_code(const uint8_t* code) const {
        const int64_t centered_raw = raw_dot(code) - bias_correction;
        const float reconstructed_dot =
                postprocess_scale * static_cast<float>(centered_raw);
        const float dot = reconstructed_dot + ec_correction;
        const float vector_scale = read_float(code + code_bytes);
        if (metric == Metric::IP) {
            return -(dot * vector_scale);
        }
        const float vector_l2 = read_float(code + code_bytes + sizeof(float));
        return query_l2 * query_l2 + vector_l2 * vector_l2 -
                2.0f * dot * vector_scale;
    }

    float operator()(faiss::idx_t i) override {
        return distance_to_code(codes + static_cast<size_t>(i) * code_size);
    }

    void distances_batch_4(
            const faiss::idx_t idx0,
            const faiss::idx_t idx1,
            const faiss::idx_t idx2,
            const faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        dis0 = distance_to_code(codes + static_cast<size_t>(idx0) * code_size);
        dis1 = distance_to_code(codes + static_cast<size_t>(idx1) * code_size);
        dis2 = distance_to_code(codes + static_cast<size_t>(idx2) * code_size);
        dis3 = distance_to_code(codes + static_cast<size_t>(idx3) * code_size);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint8_t* left = codes + static_cast<size_t>(i) * code_size;
        const uint8_t* right = codes + static_cast<size_t>(j) * code_size;
        int hamming = 0;
        for (size_t byte = 0; byte < code_bytes; ++byte) {
            hamming += __builtin_popcount(
                    static_cast<unsigned>(left[byte] ^ right[byte]));
        }
        constexpr float centroid_sq = 0.7978846f * 0.7978846f;
        const float raw_dot = centroid_sq *
                static_cast<float>(static_cast<int64_t>(padded_dim) - 2LL * hamming);
        const float left_scale = read_float(left + code_bytes);
        const float right_scale = read_float(right + code_bytes);
        if (metric == Metric::IP) {
            return -(raw_dot * left_scale * right_scale);
        }
        const float left_l2 = read_float(left + code_bytes + sizeof(float));
        const float right_l2 = read_float(right + code_bytes + sizeof(float));
        return left_l2 * left_l2 + right_l2 * right_l2 -
                2.0f * raw_dot * left_scale * right_scale;
    }

    float ffi_reference(faiss::idx_t i) const {
        return qdrant_tq_score(index, query, static_cast<uint32_t>(i));
    }
};

void validate_qdrant_tq_direct_scorer(
        const QdrantTqIndex* index,
        const uint8_t* codes,
        size_t code_size,
        int64_t rows,
        int d,
        Metric metric,
        const std::vector<float>& queries,
        int nq) {
    QdrantTqDirectDistanceComputer scorer(index, codes, code_size, d, metric);
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    const int query_count = std::min(nq, 8);
    for (int qi = 0; qi < query_count; ++qi) {
        scorer.set_query(queries.data() + static_cast<size_t>(qi) * d);
        for (int sample = 0; sample < 64; ++sample) {
            const faiss::idx_t id = static_cast<faiss::idx_t>(
                    (static_cast<uint64_t>(sample) * 11400714819323198485ull + qi * 17ull) %
                    static_cast<uint64_t>(rows));
            const float direct = scorer(id);
            const float reference = scorer.ffi_reference(id);
            const float abs_error = std::abs(direct - reference);
            const float rel_error = abs_error / std::max(1.0f, std::abs(reference));
            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, rel_error);
            if (abs_error > 2e-4f && rel_error > 2e-5f) {
                throw std::runtime_error(
                        "C++ Qdrant TQ1+ scorer does not match the Rust reference");
            }
        }
    }
    std::cout << "Validated C++ Qdrant TQ1+ scorer against Rust: max_abs_error="
              << max_abs_error << " max_rel_error=" << max_rel_error << std::endl;
}

template <typename DistanceComputer>
void search_custom(
        const faiss::IndexBinaryHNSW& index,
        const float* queries,
        int nq,
        int d,
        int k,
        int ef,
        int threads,
        const std::function<std::unique_ptr<DistanceComputer>()>& make_dc,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& labels) {
    distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    labels.assign(static_cast<size_t>(nq) * k, -1);
    faiss::SearchParametersHNSW params;
    params.efSearch = ef;
    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;
    RH block_result(nq, distances.data(), labels.data(), k);
    omp_set_num_threads(threads);
#pragma omp parallel
    {
        faiss::VisitedTable visited(index.ntotal);
        auto computer = make_dc();
        RH::SingleResultHandler result(block_result);
#pragma omp for schedule(static)
        for (int i = 0; i < nq; ++i) {
            result.begin(i);
            computer->set_query(queries + static_cast<size_t>(i) * d);
            index.hnsw.search(*computer, nullptr, result, visited, &params);
            result.end();
        }
    }
}

double recall_at_k(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<int32_t>& gt,
        int nq,
        int gt_dim,
        int k) {
    int64_t hits = 0;
    for (int i = 0; i < nq; ++i) {
        for (int p = 0; p < k; ++p) {
            const faiss::idx_t id = labels[static_cast<size_t>(i) * k + p];
            for (int g = 0; g < k; ++g) {
                if (id == gt[static_cast<size_t>(i) * gt_dim + g]) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(nq * k);
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        FbinMmap train(args.train, args.max_base);
        FbinMmap query_mmap(args.query);
        if (train.d != query_mmap.d) throw std::runtime_error("train/query dimension mismatch");
        omp_set_num_threads(args.build_threads);
        std::vector<float> queries = read_queries(query_mmap, args.max_queries, args.metric == Metric::IP);
        const int nq = queries.size() / train.d;
        int32_t gt_rows = 0;
        int32_t gt_dim = 0;
        const std::vector<int32_t> gt = read_ibin(args.gt, nq, gt_rows, gt_dim);
        if (gt_rows != nq || gt_dim < args.k) throw std::runtime_error("ground truth shape mismatch");

        std::cout << "Dataset=" << args.dataset << " method=" << method_name(args.method)
                  << " metric=" << metric_name(args.metric) << " base=(" << train.n << ',' << train.d
                  << ") query=" << nq << std::endl;

        double graph_seconds = 0.0;
        std::string graph_status;
        std::unique_ptr<faiss::IndexBinaryHNSW> graph;
        if (!args.native_bq_graph && !args.native_quantized_graph) {
            graph = build_or_load_fp_graph(args, train, graph_seconds, graph_status);
        }

        double setup_seconds = 0.0;
        double encode_seconds = 0.0;
        std::string index_status;
        std::unique_ptr<faiss::IndexHNSW> native_index;
        std::unique_ptr<faiss::IndexBinaryHNSW> custom_index;
        QdrantTqIndexPtr qdrant_tq_index;
        TQMeta tq_meta;
        RotatedBQMeta rotated_bq_meta;
        RaBitQMeta rabitq_meta;
        if (args.native_bq_graph) {
            custom_index = build_or_load_native_bq_graph(
                    args,
                    train,
                    &rotated_bq_meta,
                    setup_seconds,
                    encode_seconds,
                    graph_seconds,
                    graph_status,
                    index_status);
        } else if (args.native_quantized_graph) {
            native_index = build_or_load_native_quantized_graph(
                    args,
                    train,
                    setup_seconds,
                    encode_seconds,
                    graph_seconds,
                    graph_status,
                    index_status);
        } else if (args.method == Method::PQ2 || args.method == Method::SQ4) {
            native_index = build_or_load_native(
                    args, train, *graph, setup_seconds, encode_seconds, index_status);
        } else if (args.method == Method::QdrantTQ1) {
            qdrant_tq_index = build_or_load_qdrant_tq(
                    args, train, setup_seconds, encode_seconds, index_status);
        } else {
            custom_index = build_or_load_custom(
                    args,
                    train,
                    *graph,
                    &tq_meta,
                    &rotated_bq_meta,
                    &rabitq_meta,
                    setup_seconds,
                    encode_seconds,
                    index_status);
        }

        size_t code_bytes = 0;
        const uint8_t* qdrant_tq_codes_ptr = nullptr;
        size_t qdrant_tq_codes_bytes = 0;
        if (native_index) {
            auto* flat_codes = dynamic_cast<faiss::IndexFlatCodes*>(native_index->storage);
            if (!flat_codes) throw std::runtime_error("native storage is not IndexFlatCodes");
            code_bytes = flat_codes->code_size;
        } else if (qdrant_tq_index) {
            code_bytes = qdrant_tq_code_size(qdrant_tq_index.get());
            qdrant_tq_codes_ptr = qdrant_tq_codes(
                    qdrant_tq_index.get(), &qdrant_tq_codes_bytes);
            if (!qdrant_tq_codes_ptr ||
                qdrant_tq_codes_bytes != static_cast<size_t>(train.n) * code_bytes) {
                throw std::runtime_error("invalid Qdrant TQ1+ code buffer");
            }
            validate_qdrant_tq_direct_scorer(
                    qdrant_tq_index.get(),
                    qdrant_tq_codes_ptr,
                    code_bytes,
                    train.n,
                    train.d,
                    args.metric,
                    queries,
                    nq);
        } else {
            code_bytes = custom_index->code_size;
        }
        const uintmax_t index_bytes = args.method == Method::QdrantTQ1
                ? fs::file_size(args.graph) + fs::file_size(args.index)
                : fs::file_size(args.index);
        const std::string reported_graph_path =
                (args.native_bq_graph || args.native_quantized_graph)
                ? args.index
                : args.graph;

        make_parent(args.out);
        std::ofstream csv(args.out);
        csv << "dataset,method,metric,bits_per_dim,k,nq,nb,dim,M,efConstruction,efSearch,"
            << "rounds,build_threads,search_threads,graph_status,graph_seconds,index_status,"
            << "setup_seconds,encode_seconds,code_bytes_per_vector,index_bytes,search_seconds_mean,"
            << "recall_at_10,qps,index_path,graph_path,meta_path,reference_meta_path\n";

        for (int ef : args.efs) {
            std::vector<double> recalls;
            std::vector<double> elapsed_values;
            std::vector<double> qps_values;
            for (int round = 0; round < args.rounds; ++round) {
                std::vector<float> distances;
                std::vector<faiss::idx_t> labels;
                const auto start = std::chrono::steady_clock::now();
                if (native_index) {
                    native_index->hnsw.efSearch = ef;
                    omp_set_num_threads(args.search_threads);
                    distances.resize(static_cast<size_t>(nq) * args.k);
                    labels.resize(static_cast<size_t>(nq) * args.k);
                    native_index->search(nq, queries.data(), args.k, distances.data(), labels.data());
                } else if (args.method == Method::QdrantTQ1) {
                    search_custom<QdrantTqDirectDistanceComputer>(
                            *graph,
                            queries.data(),
                            nq,
                            train.d,
                            args.k,
                            ef,
                            args.search_threads,
                            [&]() {
                                return std::make_unique<QdrantTqDirectDistanceComputer>(
                                        qdrant_tq_index.get(),
                                        qdrant_tq_codes_ptr,
                                        code_bytes,
                                        train.d,
                                        args.metric);
                            },
                            distances,
                            labels);
                } else if (args.method == Method::BQ1) {
                    const uint8_t* codes = binary_codes(*custom_index);
                    search_custom<BQDistanceComputer>(
                            *custom_index, queries.data(), nq, train.d, args.k, ef,
                            args.search_threads,
                            [&]() {
                                return std::make_unique<BQDistanceComputer>(
                                        codes, custom_index->code_size, train.d, args.metric);
                            },
                            distances, labels);
                } else if (args.method == Method::SymBQ1) {
                    const uint8_t* codes = binary_codes(*custom_index);
                    search_custom<SymmetricBQDistanceComputer>(
                            *custom_index, queries.data(), nq, train.d, args.k, ef,
                            args.search_threads,
                            [&]() {
                                return std::make_unique<SymmetricBQDistanceComputer>(
                                        codes, custom_index->code_size, train.d);
                            },
                            distances, labels);
                } else if (args.method == Method::RotBQ1) {
                    const uint8_t* codes = binary_codes(*custom_index);
                    search_custom<RotatedSymmetricBQDistanceComputer>(
                            *custom_index, queries.data(), nq, train.d, args.k, ef,
                            args.search_threads,
                            [&]() {
                                return std::make_unique<RotatedSymmetricBQDistanceComputer>(
                                        codes, custom_index->code_size, rotated_bq_meta);
                            },
                            distances, labels);
                } else if (args.method == Method::RaBitQ1) {
                    const uint8_t* codes = binary_codes(*custom_index);
                    search_custom<RaBitQDistanceComputer>(
                            *custom_index,
                            queries.data(),
                            nq,
                            train.d,
                            args.k,
                            ef,
                            args.search_threads,
                            [&]() {
                                return std::make_unique<RaBitQDistanceComputer>(
                                        codes,
                                        custom_index->code_size,
                                        train.d,
                                        rabitq_meta);
                            },
                            distances,
                            labels);
                } else {
                    const uint8_t* codes = binary_codes(*custom_index);
                    search_custom<TQDistanceComputer>(
                            *custom_index, queries.data(), nq, train.d, args.k, ef,
                            args.search_threads,
                            [&]() {
                                return std::make_unique<TQDistanceComputer>(
                                        codes, custom_index->code_size, train.d, args.metric, tq_meta);
                            },
                            distances, labels);
                }
                const double elapsed = seconds_since(start);
                const double recall = recall_at_k(labels, gt, nq, gt_dim, args.k);
                const double qps = nq / elapsed;
                recalls.push_back(recall);
                elapsed_values.push_back(elapsed);
                qps_values.push_back(qps);
                std::cout << "  ef=" << ef << " round=" << round + 1 << '/' << args.rounds
                          << " recall@" << args.k << '=' << recall << " qps=" << qps << std::endl;
            }
            const auto mean = [](const std::vector<double>& values) {
                return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
            };
            const int bits_per_dim =
                    (args.method == Method::BQ1 || args.method == Method::SymBQ1 ||
                     args.method == Method::RotBQ1 ||
                     args.method == Method::RaBitQ1 ||
                     args.method == Method::QdrantTQ1)
                    ? 1
                    : args.method == Method::SQ4                  ? 4
                                                                  : 2;
            csv << args.dataset << ',' << method_name(args.method) << ',' << metric_name(args.metric)
                << ',' << bits_per_dim << ',' << args.k << ',' << nq << ',' << train.n << ',' << train.d
                << ',' << args.m << ',' << args.ef_construction << ',' << ef << ',' << args.rounds
                << ',' << args.build_threads << ',' << args.search_threads << ',' << graph_status
                << ',' << graph_seconds << ',' << index_status << ',' << setup_seconds << ','
                << encode_seconds << ',' << code_bytes << ',' << index_bytes << ','
                << mean(elapsed_values) << ',' << mean(recalls) << ',' << mean(qps_values) << ','
                << args.index << ',' << reported_graph_path << ',' << args.meta << ','
                << args.reference_meta << '\n';
        }
        std::cout << "Wrote " << args.out << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
