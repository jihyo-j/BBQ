#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <faiss/utils/Heap.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/index_io.h>

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <liburing.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include "adcbq_sign_packing.hpp"

// The development FAISS fork adds two optional per-vector payloads to IndexBinaryFlat (`xflags`,
// `block_scales`) for experiments outside the BBQ paper. BBQ never fills them; it only has to keep them in
// step when it permutes or releases the code storage. Detect them at compile time so this file builds
// against upstream FAISS as well as the fork.
#include <type_traits>
#include <utility>
namespace bbq_compat {
template <class T, class = void>
struct has_fork_payloads : std::false_type {};
template <class T>
struct has_fork_payloads<T, std::void_t<decltype(std::declval<T&>().xflags),
                                        decltype(std::declval<T&>().block_scales),
                                        decltype(std::declval<T&>().block_scale_size)>>
        : std::true_type {};

template <class Flat, class Reorder>
void reorder_fork_payloads(Flat& flat, size_t code_size, Reorder&& reorder) {
    if constexpr (has_fork_payloads<Flat>::value) {
        if (flat.xflags.size() > 0) reorder(flat.xflags, code_size);
        if (flat.block_scales.size() > 0) reorder(flat.block_scales, static_cast<size_t>(flat.block_scale_size));
    }
}

template <class Flat>
void release_fork_payloads(Flat& flat) {
    if constexpr (has_fork_payloads<Flat>::value) {
        flat.xflags = decltype(flat.xflags)();
        flat.block_scales = decltype(flat.block_scales)();
    }
}
}  // namespace bbq_compat


#ifndef O_DIRECT
#define O_DIRECT 0
#endif

namespace fs = std::filesystem;

constexpr size_t kFpDataAlignment = 4096;
constexpr size_t kLegacyFbinHeaderBytes = 2 * sizeof(int32_t);

enum class AdcMode {
    Fp32,
    Fp32Direct,
    Int16,
    Int8,
};

AdcMode parse_adc_mode(const std::string& value) {
    if (value == "fp32") return AdcMode::Fp32;
    if (value == "fp32-direct" || value == "direct") return AdcMode::Fp32Direct;
    if (value == "int16") return AdcMode::Int16;
    if (value == "int8") return AdcMode::Int8;
    throw std::runtime_error("--adc-mode must be one of: fp32, fp32-direct, int16, int8");
}

const char* adc_mode_name(AdcMode mode) {
    switch (mode) {
        case AdcMode::Fp32:
            return "fp32";
        case AdcMode::Fp32Direct:
            return "fp32-direct";
        case AdcMode::Int16:
            return "int16";
        case AdcMode::Int8:
            return "int8";
    }
    return "unknown";
}

enum class LutMode {
    Byte,
    Nibble,
};

LutMode parse_lut_mode(const std::string& value) {
    if (value == "byte") return LutMode::Byte;
    if (value == "nibble") return LutMode::Nibble;
    throw std::runtime_error("--lut-mode must be one of: byte, nibble");
}

const char* lut_mode_name(LutMode mode) {
    switch (mode) {
        case LutMode::Byte:
            return "byte";
        case LutMode::Nibble:
            return "nibble";
    }
    return "unknown";
}


enum class SearchImpl {
    Faiss,
    Inline,
};

enum class GraphBuildMode {
    Binary,
    Fp,
};

enum class DistanceMode {
    AdcBQ,
    XFlagBQ,
    AdcXFlagBQ,
};

SearchImpl parse_search_impl(const std::string& value) {
    if (value == "faiss") return SearchImpl::Faiss;
    if (value == "inline") return SearchImpl::Inline;
    throw std::runtime_error("--search-impl must be one of: faiss, inline");
}

const char* search_impl_name(SearchImpl impl) {
    switch (impl) {
        case SearchImpl::Faiss:
            return "faiss";
        case SearchImpl::Inline:
            return "inline";
    }
    return "unknown";
}

GraphBuildMode parse_graph_build_mode(const std::string& value) {
    if (value == "binary") return GraphBuildMode::Binary;
    if (value == "fp") return GraphBuildMode::Fp;
    throw std::runtime_error("--graph-build must be one of: binary, fp");
}

const char* graph_build_mode_name(GraphBuildMode mode) {
    switch (mode) {
        case GraphBuildMode::Binary:
            return "binary";
        case GraphBuildMode::Fp:
            return "fp";
    }
    return "unknown";
}

DistanceMode parse_distance_mode(const std::string& value) {
    if (value == "adcbq" || value == "adc-bq") return DistanceMode::AdcBQ;
    if (value == "xflag" || value == "xflag-bq") return DistanceMode::XFlagBQ;
    if (value == "adc-xflag" || value == "adcxflag" || value == "adc-xflag-bq") {
        return DistanceMode::AdcXFlagBQ;
    }
    throw std::runtime_error("--distance-mode must be one of: adcbq, xflag, adc-xflag");
}

const char* distance_mode_name(DistanceMode mode) {
    switch (mode) {
        case DistanceMode::AdcBQ:
            return "adcbq";
        case DistanceMode::XFlagBQ:
            return "xflag";
        case DistanceMode::AdcXFlagBQ:
            return "adc-xflag";
    }
    return "unknown";
}

enum class RerankDotMode {
    Fp64,
    Fp32,
    Int8,
};

RerankDotMode parse_rerank_dot_mode(const std::string& value) {
    if (value == "fp64") return RerankDotMode::Fp64;
    if (value == "fp32") return RerankDotMode::Fp32;
    if (value == "int8") return RerankDotMode::Int8;
    throw std::runtime_error("--rerank-dot must be one of: fp64, fp32, int8");
}

const char* rerank_dot_mode_name(RerankDotMode mode) {
    switch (mode) {
        case RerankDotMode::Fp64:
            return "fp64";
        case RerankDotMode::Fp32:
            return "fp32";
        case RerankDotMode::Int8:
            return "int8";
    }
    return "unknown";
}

enum class FpRerankSource {
    Ram,
    Mmap,
    Pread,
    ODirect,
    ODirectUring,
};

FpRerankSource parse_fp_rerank_source(const std::string& value) {
    if (value == "ram" || value == "resident") return FpRerankSource::Ram;
    if (value == "mmap") return FpRerankSource::Mmap;
    if (value == "pread") return FpRerankSource::Pread;
    if (value == "odirect" || value == "o_direct") return FpRerankSource::ODirect;
    if (value == "odirect-uring" || value == "o_direct_uring" || value == "uring") {
        return FpRerankSource::ODirectUring;
    }
    throw std::runtime_error("--fp-rerank-source must be one of: ram, mmap, pread, odirect, odirect-uring");
}

const char* fp_rerank_source_name(FpRerankSource source) {
    switch (source) {
        case FpRerankSource::Ram:
            return "ram";
        case FpRerankSource::Mmap:
            return "mmap";
        case FpRerankSource::Pread:
            return "pread";
        case FpRerankSource::ODirect:
            return "odirect";
        case FpRerankSource::ODirectUring:
            return "odirect-uring";
    }
    return "unknown";
}

enum class FpRerankNormMode {
    Precompute,
    OnFetch,
};

enum class FpRerankMetric {
    Cosine,
    InnerProduct,
};

FpRerankMetric parse_fp_rerank_metric(const std::string& value) {
    if (value == "cosine") return FpRerankMetric::Cosine;
    if (value == "ip" || value == "inner-product" || value == "inner_product") {
        return FpRerankMetric::InnerProduct;
    }
    throw std::runtime_error("--fp-rerank-metric must be one of: cosine, ip");
}

const char* fp_rerank_metric_name(FpRerankMetric metric) {
    switch (metric) {
        case FpRerankMetric::Cosine:
            return "cosine";
        case FpRerankMetric::InnerProduct:
            return "ip";
    }
    return "unknown";
}

FpRerankNormMode parse_fp_rerank_norm_mode(const std::string& value) {
    if (value == "precompute") return FpRerankNormMode::Precompute;
    if (value == "onfetch" || value == "on_fetch") return FpRerankNormMode::OnFetch;
    throw std::runtime_error("--fp-rerank-norms must be one of: precompute, onfetch");
}

const char* fp_rerank_norm_mode_name(FpRerankNormMode mode) {
    switch (mode) {
        case FpRerankNormMode::Precompute:
            return "precompute";
        case FpRerankNormMode::OnFetch:
            return "onfetch";
    }
    return "unknown";
}

enum class CodeLayout {
    Vector,
    Block32,
};

CodeLayout parse_code_layout(const std::string& value) {
    if (value == "vector") return CodeLayout::Vector;
    if (value == "block32") return CodeLayout::Block32;
    throw std::runtime_error("--code-layout must be one of: vector, block32");
}

const char* code_layout_name(CodeLayout layout) {
    switch (layout) {
        case CodeLayout::Vector:
            return "vector";
        case CodeLayout::Block32:
            return "block32";
    }
    return "unknown";
}

struct Args {
    std::string dataset;
    std::string train;
    std::string query;
    std::string gt;
    std::string index;
    std::string out;
    std::string stats_out;
    std::string candidates_out;
    std::string final_labels_out;
    std::string per_query_metrics_out;
    std::string xflag_code_path;
    int m = 24;
    int ef_construction = 200;
    std::vector<int> efs;
    int rerank_n = 50;
    // Number of approximate candidates returned by HNSW before exact-FP
    // refinement.  Zero preserves the historical behavior (pool == rerank_n).
    int candidate_pool_size = 0;
    int adaptive_rerank_min = 0;
    int k = 10;
    int max_queries = 1000;
    int rounds = 3;
    int build_threads = 48;
    int search_threads = 1;
    int prefetch_lookahead = 16;
    int add_batch_size = 50000;
    int colocation_iterations = 0;
    int colocation_partition_size = 64;
    size_t colocation_io_merge_bytes = 0;
    std::string colocation_map_path;
    bool rebuild_colocation = false;
    AdcMode adc_mode = AdcMode::Fp32;
    LutMode lut_mode = LutMode::Byte;
    RerankDotMode rerank_dot_mode = RerankDotMode::Fp64;
    FpRerankSource fp_rerank_source = FpRerankSource::Mmap;
    FpRerankNormMode fp_rerank_norm_mode = FpRerankNormMode::Precompute;
    FpRerankMetric fp_rerank_metric = FpRerankMetric::Cosine;
    bool fp_rerank_cache = false;
    size_t fp_rerank_cache_rows = 131072;
    size_t fp_rerank_page_cache_mib = 0;
    bool fp_rerank_overlap_queries = false;
    bool fp_rerank_overlap_cache_hits = false;
    int fp_rerank_qd = 80;
    std::string fp_rerank_path;
    size_t index_page_cache_mib = 0;
    bool index_page_async = false;
    int index_page_qd = 48;
    int index_page_async_sync_miss_max = 0;
    std::string paged_index_path;
    bool rebuild_paged_index = false;
    std::string mmap_index_path;
    bool rebuild_mmap_index = false;
    bool mmap_drop_cache_before_round = false;
    CodeLayout code_layout = CodeLayout::Vector;
    SearchImpl search_impl = SearchImpl::Faiss;
    GraphBuildMode graph_build_mode = GraphBuildMode::Binary;
    DistanceMode distance_mode = DistanceMode::AdcBQ;
    float xflag_threshold = -1.0f;
    double xflag_quantile = 0.7;
    int xflag_sample_rows = 50000;
    bool rebuild = false;
    bool no_rerank = false;
    bool candidate_only = false;
    bool strict_query_e2e = false;
    bool adaptive_rerank = false;
    float adaptive_rerank_alpha = 0.5f;
    bool calibrated_threshold_rerank = false;
    float calibrated_threshold_rerank_tau = 0.0f;
    bool no_check_relative_distance = false;
    bool unbounded_queue = false;
    bool early_abandon = false;
    bool ordered_early_abandon = false;
    int ordered_early_abandon_probe_bytes = 16;
    bool adaptive_cfr = false;
    bool adaptive_cfr_paper_bucket = false;
    float adaptive_cfr_tau = 0.6f;
    int adaptive_cfr_tmin_pops = 25;
    int adaptive_cfr_bucket_count = 4;
    std::vector<float> adaptive_cfr_gammas;
    bool query_preprocess_only = false;
    bool query_preprocess_include_calibration = false;
    size_t query_preprocess_iterations = 6000;
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
            << "Usage: " << argv0 << " --dataset NAME --train X.fbin --query Q.fbin "
            << "--gt G.ibin --index I.faiss --out O.csv [--efs ...] "
            << "[--adc-mode fp32|fp32-direct|int16|int8] [--lut-mode byte|nibble] "
            << "[--rerank-dot fp64|fp32|int8] [--fp-rerank-source ram|mmap|pread|odirect|odirect-uring] "
            << "[--fp-rerank-metric cosine|ip] "
            << "[--fp-rerank-path PATH] "
            << "[--fp-rerank-cache] [--fp-rerank-cache-rows N] "
            << "[--fp-rerank-page-cache-mib N] "
            << "[--fp-rerank-overlap-queries] [--fp-rerank-overlap-cache-hits] [--fp-rerank-qd N] "
            << "[--index-page-cache-mib N] [--index-page-async] [--index-page-qd N] "
            << "[--paged-index-path PATH] [--rebuild-paged-index] "
            << "[--mmap-index-path PATH] [--rebuild-mmap-index] [--mmap-drop-cache-before-round] "
            << "[--code-layout vector|block32] [--search-impl faiss|inline] [--graph-build binary|fp] "
            << "[--distance-mode adcbq|xflag|adc-xflag] [--xflag-code-path PATH] "
            << "[--xflag-threshold T] [--xflag-quantile Q] [--xflag-sample-rows N] "
            << "[--stats-out O.csv] [--candidates-out O.csv] "
            << "[--final-labels-out O.csv] "
            << "[--per-query-metrics-out O.csv] "
            << "[--candidate-pool-size N] [--rerank-n N] "
            << "[--colocation-iterations N] [--colocation-partition-size N] "
            << "[--colocation-io-merge-bytes N] "
            << "[--colocation-map-path PATH] [--rebuild-colocation] "
            << "[--candidate-only] "
            << "[--strict-query-e2e] "
            << "[--adaptive-rerank] [--adaptive-rerank-min N] [--adaptive-rerank-alpha A] "
            << "[--calibrated-threshold-rerank-tau T] "
            << "[--adaptive-cfr] [--adaptive-cfr-tau T] [--adaptive-cfr-tmin-pops N] "
            << "[--adaptive-cfr-paper-bucket] [--adaptive-cfr-bucket-count N] [--adaptive-cfr-gammas ...] "
            << "[--prefetch-lookahead N] [--no-rerank] [--no-check-relative-distance] "
            << "[--unbounded-queue] [--early-abandon] [--ordered-early-abandon] [--ordered-early-abandon-probe-bytes N] "
            << "[--query-preprocess-only] [--query-preprocess-include-calibration] "
            << "[--query-preprocess-iterations N]\n";
    std::exit(2);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (a == "--dataset") args.dataset = need_value("--dataset");
        else if (a == "--train") args.train = need_value("--train");
        else if (a == "--query") args.query = need_value("--query");
        else if (a == "--gt") args.gt = need_value("--gt");
        else if (a == "--index") args.index = need_value("--index");
        else if (a == "--out") args.out = need_value("--out");
        else if (a == "--stats-out") args.stats_out = need_value("--stats-out");
        else if (a == "--candidates-out") args.candidates_out = need_value("--candidates-out");
        else if (a == "--final-labels-out") args.final_labels_out = need_value("--final-labels-out");
        else if (a == "--per-query-metrics-out") args.per_query_metrics_out = need_value("--per-query-metrics-out");
        else if (a == "--xflag-code-path") args.xflag_code_path = need_value("--xflag-code-path");
        else if (a == "--m") args.m = std::stoi(need_value("--m"));
        else if (a == "--ef-construction") args.ef_construction = std::stoi(need_value("--ef-construction"));
        else if (a == "--rerank-n") args.rerank_n = std::stoi(need_value("--rerank-n"));
        else if (a == "--candidate-pool-size") args.candidate_pool_size = std::stoi(need_value("--candidate-pool-size"));
        else if (a == "--adaptive-rerank-min") args.adaptive_rerank_min = std::stoi(need_value("--adaptive-rerank-min"));
        else if (a == "--k") args.k = std::stoi(need_value("--k"));
        else if (a == "--max-queries") args.max_queries = std::stoi(need_value("--max-queries"));
        else if (a == "--rounds") args.rounds = std::stoi(need_value("--rounds"));
        else if (a == "--build-threads") args.build_threads = std::stoi(need_value("--build-threads"));
        else if (a == "--search-threads") args.search_threads = std::stoi(need_value("--search-threads"));
        else if (a == "--prefetch-lookahead") args.prefetch_lookahead = std::stoi(need_value("--prefetch-lookahead"));
        else if (a == "--add-batch-size") args.add_batch_size = std::stoi(need_value("--add-batch-size"));
        else if (a == "--colocation-iterations") args.colocation_iterations = std::stoi(need_value("--colocation-iterations"));
        else if (a == "--colocation-partition-size") args.colocation_partition_size = std::stoi(need_value("--colocation-partition-size"));
        else if (a == "--colocation-io-merge-bytes") args.colocation_io_merge_bytes = static_cast<size_t>(std::stoull(need_value("--colocation-io-merge-bytes")));
        else if (a == "--colocation-map-path") args.colocation_map_path = need_value("--colocation-map-path");
        else if (a == "--rebuild-colocation") args.rebuild_colocation = true;
        else if (a == "--adc-mode") args.adc_mode = parse_adc_mode(need_value("--adc-mode"));
        else if (a == "--lut-mode") args.lut_mode = parse_lut_mode(need_value("--lut-mode"));
        else if (a == "--rerank-dot") args.rerank_dot_mode = parse_rerank_dot_mode(need_value("--rerank-dot"));
        else if (a == "--fp-rerank-source") args.fp_rerank_source = parse_fp_rerank_source(need_value("--fp-rerank-source"));
        else if (a == "--fp-rerank-path") args.fp_rerank_path = need_value("--fp-rerank-path");
        else if (a == "--fp-rerank-norms") args.fp_rerank_norm_mode = parse_fp_rerank_norm_mode(need_value("--fp-rerank-norms"));
        else if (a == "--fp-rerank-metric") args.fp_rerank_metric = parse_fp_rerank_metric(need_value("--fp-rerank-metric"));
        else if (a == "--fp-rerank-cache") args.fp_rerank_cache = true;
        else if (a == "--fp-rerank-cache-rows") args.fp_rerank_cache_rows = static_cast<size_t>(std::stoull(need_value("--fp-rerank-cache-rows")));
        else if (a == "--fp-rerank-page-cache-mib") args.fp_rerank_page_cache_mib = static_cast<size_t>(std::stoull(need_value("--fp-rerank-page-cache-mib")));
        else if (a == "--fp-rerank-overlap-queries" || a == "--fp-rerank-overlap-search") args.fp_rerank_overlap_queries = true;
        else if (a == "--fp-rerank-overlap-cache-hits") args.fp_rerank_overlap_cache_hits = true;
        else if (a == "--fp-rerank-qd") args.fp_rerank_qd = std::stoi(need_value("--fp-rerank-qd"));
        else if (a == "--index-page-cache-mib") args.index_page_cache_mib = static_cast<size_t>(std::stoull(need_value("--index-page-cache-mib")));
        else if (a == "--index-page-async") args.index_page_async = true;
        else if (a == "--index-page-qd") args.index_page_qd = std::stoi(need_value("--index-page-qd"));
        else if (a == "--index-page-async-sync-miss-max") args.index_page_async_sync_miss_max = std::stoi(need_value("--index-page-async-sync-miss-max"));
        else if (a == "--paged-index-path") args.paged_index_path = need_value("--paged-index-path");
        else if (a == "--rebuild-paged-index") args.rebuild_paged_index = true;
        else if (a == "--mmap-index-path") args.mmap_index_path = need_value("--mmap-index-path");
        else if (a == "--rebuild-mmap-index") args.rebuild_mmap_index = true;
        else if (a == "--mmap-drop-cache-before-round") args.mmap_drop_cache_before_round = true;
        else if (a == "--code-layout") args.code_layout = parse_code_layout(need_value("--code-layout"));
        else if (a == "--search-impl") args.search_impl = parse_search_impl(need_value("--search-impl"));
        else if (a == "--graph-build") args.graph_build_mode = parse_graph_build_mode(need_value("--graph-build"));
        else if (a == "--distance-mode") args.distance_mode = parse_distance_mode(need_value("--distance-mode"));
        else if (a == "--xflag-threshold") args.xflag_threshold = std::stof(need_value("--xflag-threshold"));
        else if (a == "--xflag-quantile") args.xflag_quantile = std::stod(need_value("--xflag-quantile"));
        else if (a == "--xflag-sample-rows") args.xflag_sample_rows = std::stoi(need_value("--xflag-sample-rows"));
        else if (a == "--rebuild") args.rebuild = true;
        else if (a == "--no-rerank") args.no_rerank = true;
        else if (a == "--candidate-only") args.candidate_only = true;
        else if (a == "--strict-query-e2e") args.strict_query_e2e = true;
        else if (a == "--adaptive-rerank") args.adaptive_rerank = true;
        else if (a == "--adaptive-rerank-alpha") args.adaptive_rerank_alpha = std::stof(need_value("--adaptive-rerank-alpha"));
        else if (a == "--calibrated-threshold-rerank-tau") {
            args.calibrated_threshold_rerank = true;
            args.calibrated_threshold_rerank_tau =
                    std::stof(need_value("--calibrated-threshold-rerank-tau"));
        }
        else if (a == "--no-check-relative-distance") args.no_check_relative_distance = true;
        else if (a == "--unbounded-queue") args.unbounded_queue = true;
        else if (a == "--early-abandon") args.early_abandon = true;
        else if (a == "--ordered-early-abandon") args.ordered_early_abandon = true;
        else if (a == "--ordered-early-abandon-probe-bytes") args.ordered_early_abandon_probe_bytes = std::stoi(need_value("--ordered-early-abandon-probe-bytes"));
        else if (a == "--adaptive-cfr") args.adaptive_cfr = true;
        else if (a == "--adaptive-cfr-tau") args.adaptive_cfr_tau = std::stof(need_value("--adaptive-cfr-tau"));
        else if (a == "--adaptive-cfr-tmin-pops") args.adaptive_cfr_tmin_pops = std::stoi(need_value("--adaptive-cfr-tmin-pops"));
        else if (a == "--adaptive-cfr-paper-bucket") args.adaptive_cfr_paper_bucket = true;
        else if (a == "--adaptive-cfr-bucket-count") args.adaptive_cfr_bucket_count = std::stoi(need_value("--adaptive-cfr-bucket-count"));
        else if (a == "--query-preprocess-only") args.query_preprocess_only = true;
        else if (a == "--query-preprocess-include-calibration") {
            args.query_preprocess_include_calibration = true;
        }
        else if (a == "--query-preprocess-iterations") {
            args.query_preprocess_iterations = static_cast<size_t>(
                    std::stoull(need_value("--query-preprocess-iterations")));
        }
        else if (a == "--adaptive-cfr-gammas") {
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                args.adaptive_cfr_gammas.push_back(std::stof(argv[++i]));
            }
        }
        else if (a == "--efs") {
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                args.efs.push_back(std::stoi(argv[++i]));
            }
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }
    if (args.efs.empty()) args.efs = {128, 256, 512};
    if (args.candidate_pool_size == 0) args.candidate_pool_size = args.rerank_n;
    if (args.query_preprocess_only) {
        if (args.dataset.empty() || args.query.empty() || args.out.empty()) {
            throw std::runtime_error(
                    "--query-preprocess-only requires --dataset --query --out");
        }
        if (args.rounds < 3 || args.query_preprocess_iterations == 0) {
            throw std::runtime_error("invalid query preprocessing benchmark settings");
        }
        return args;
    }
    if (args.dataset.empty() || args.train.empty() || args.query.empty() ||
        args.gt.empty() || args.index.empty() || args.out.empty()) {
        usage(argv[0]);
    }
    if ((!args.no_rerank || args.candidate_only) && args.candidate_pool_size < args.k) {
        throw std::runtime_error("--candidate-pool-size must be >= --k");
    }
    if (!args.no_rerank && !args.adaptive_rerank &&
        !args.calibrated_threshold_rerank && args.rerank_n < args.k) {
        throw std::runtime_error("--rerank-n must be >= --k");
    }
    if (!args.no_rerank && !args.adaptive_rerank &&
        !args.calibrated_threshold_rerank &&
        args.rerank_n > args.candidate_pool_size) {
        throw std::runtime_error("--rerank-n must be <= --candidate-pool-size");
    }
    if (args.candidate_only && !args.no_rerank) {
        throw std::runtime_error("--candidate-only must be combined with --no-rerank");
    }
    if (args.candidate_only && args.candidates_out.empty()) {
        throw std::runtime_error("--candidate-only requires --candidates-out");
    }
    if (args.candidate_only &&
        (args.fp_rerank_overlap_queries || args.fp_rerank_overlap_cache_hits ||
         args.adaptive_rerank || args.calibrated_threshold_rerank)) {
        throw std::runtime_error(
                "--candidate-only cannot be combined with rerank adaptation or overlap");
    }
    if (args.adaptive_rerank && args.calibrated_threshold_rerank) {
        throw std::runtime_error(
                "heuristic adaptive rerank and calibrated threshold rerank are mutually exclusive");
    }
    if (!args.per_query_metrics_out.empty()) {
        if (args.no_rerank || args.candidate_only || args.fp_rerank_overlap_queries ||
            args.search_impl != SearchImpl::Inline ||
            args.distance_mode != DistanceMode::AdcBQ ||
            !args.mmap_index_path.empty() || args.index_page_cache_mib > 0 ||
            args.rerank_dot_mode == RerankDotMode::Int8) {
            throw std::runtime_error(
                    "--per-query-metrics-out currently requires resident inline ADC-BQ "
                    "search with non-overlapped FP32/FP64 reranking");
        }
    }
    if (args.strict_query_e2e) {
        if (args.per_query_metrics_out.empty()) {
            throw std::runtime_error(
                    "--strict-query-e2e requires --per-query-metrics-out");
        }
        if (args.search_threads != 1 || args.fp_rerank_overlap_queries ||
            args.adaptive_rerank || args.rerank_dot_mode == RerankDotMode::Int8 ||
            args.fp_rerank_source != FpRerankSource::ODirectUring ||
            args.fp_rerank_norm_mode != FpRerankNormMode::OnFetch ||
            args.fp_rerank_cache || args.fp_rerank_page_cache_mib > 0 ||
            !args.candidates_out.empty()) {
            throw std::runtime_error(
                    "--strict-query-e2e currently requires T=1, O_DIRECT io_uring, "
                    "on-fetch norms, no FP cache, fixed/calibrated reranking, and no "
                    "candidate CSV output");
        }
    }
    if (args.calibrated_threshold_rerank) {
        if (args.no_rerank) {
            throw std::runtime_error(
                    "--calibrated-threshold-rerank-tau cannot be used with --no-rerank");
        }
        if (!std::isfinite(args.calibrated_threshold_rerank_tau)) {
            throw std::runtime_error("calibrated threshold rerank tau must be finite");
        }
        if (args.adc_mode == AdcMode::Int8) {
            throw std::runtime_error(
                    "calibrated threshold rerank currently supports fp32 or int16 ADC");
        }
    }
    if (args.colocation_iterations < 0) {
        throw std::runtime_error("--colocation-iterations must be >= 0");
    }
    if (args.colocation_partition_size <= 0) {
        throw std::runtime_error("--colocation-partition-size must be > 0");
    }
    if (args.colocation_partition_size > 64) {
        throw std::runtime_error("--colocation-partition-size must be <= 64 for page-aware reorder");
    }
    if (args.colocation_io_merge_bytes > 0 && args.colocation_io_merge_bytes % 4096 != 0) {
        throw std::runtime_error("--colocation-io-merge-bytes must be 0 or a multiple of 4096");
    }
    if (args.colocation_iterations == 0 &&
        (!args.colocation_map_path.empty() || args.rebuild_colocation)) {
        throw std::runtime_error("colocation paths/rebuild require --colocation-iterations > 0");
    }
    if (args.fp_rerank_page_cache_mib > 0 && args.fp_rerank_source != FpRerankSource::ODirectUring) {
        throw std::runtime_error("--fp-rerank-page-cache-mib currently requires --fp-rerank-source odirect-uring");
    }
    if (args.fp_rerank_page_cache_mib > 0 && args.fp_rerank_cache) {
        throw std::runtime_error("row cache and page cache cannot be enabled together");
    }
    if (args.fp_rerank_qd <= 0) {
        throw std::runtime_error("--fp-rerank-qd must be > 0");
    }
    if (args.fp_rerank_overlap_queries) {
        if (args.no_rerank) {
            throw std::runtime_error("--fp-rerank-overlap-queries requires FP reranking");
        }
        if (args.fp_rerank_source != FpRerankSource::ODirectUring) {
            throw std::runtime_error("--fp-rerank-overlap-queries requires --fp-rerank-source odirect-uring");
        }
        if (args.search_impl != SearchImpl::Inline || args.distance_mode != DistanceMode::AdcBQ) {
            throw std::runtime_error("--fp-rerank-overlap-queries currently requires inline ADC-BQ search");
        }
        if (args.rerank_dot_mode == RerankDotMode::Int8) {
            throw std::runtime_error("--fp-rerank-overlap-queries requires FP32 or FP64 rerank dot products");
        }
        if (args.adaptive_rerank) {
            throw std::runtime_error("--fp-rerank-overlap-queries does not yet support adaptive rerank counts");
        }
        if (args.fp_rerank_page_cache_mib > 0) {
            throw std::runtime_error("--fp-rerank-overlap-queries cannot be combined with the FP page cache");
        }
        if (args.fp_rerank_cache && !args.fp_rerank_overlap_cache_hits) {
            throw std::runtime_error(
                    "FP row cache with --fp-rerank-overlap-queries also requires "
                    "--fp-rerank-overlap-cache-hits (hybrid overlap mode)");
        }
        if (args.index_page_cache_mib > 0 || !args.mmap_index_path.empty()) {
            throw std::runtime_error("--fp-rerank-overlap-queries currently requires the HNSW graph and BQ codes in RAM");
        }
        const int max_query_reads = args.calibrated_threshold_rerank
                ? args.candidate_pool_size
                : args.rerank_n;
        if (args.fp_rerank_qd < max_query_reads) {
            throw std::runtime_error("--fp-rerank-qd must cover the maximum one-batch query reads");
        }
    }
    if (args.fp_rerank_overlap_cache_hits) {
        if (args.no_rerank) {
            throw std::runtime_error("--fp-rerank-overlap-cache-hits requires FP reranking");
        }
        if (args.fp_rerank_source != FpRerankSource::ODirectUring) {
            throw std::runtime_error("--fp-rerank-overlap-cache-hits requires --fp-rerank-source odirect-uring");
        }
        if (!args.fp_rerank_cache || args.fp_rerank_page_cache_mib > 0) {
            throw std::runtime_error("--fp-rerank-overlap-cache-hits requires the FP row cache and cannot use the FP page cache");
        }
        if (args.rerank_dot_mode == RerankDotMode::Int8) {
            throw std::runtime_error("--fp-rerank-overlap-cache-hits requires FP32 or FP64 rerank dot products");
        }
        const int max_query_reads =
                (args.adaptive_rerank || args.calibrated_threshold_rerank)
                ? args.candidate_pool_size
                : args.rerank_n;
        if (args.fp_rerank_qd < max_query_reads) {
            throw std::runtime_error("--fp-rerank-qd must cover the maximum one-batch cache-miss I/O");
        }
    }
    if (args.index_page_cache_mib > 0 && args.search_impl != SearchImpl::Inline) {
        throw std::runtime_error("--index-page-cache-mib requires --search-impl inline");
    }
    if (args.index_page_cache_mib > 0 && args.distance_mode != DistanceMode::AdcBQ) {
        throw std::runtime_error("paged index cache currently supports only --distance-mode adcbq");
    }
    if (args.index_page_async && args.index_page_cache_mib == 0) {
        throw std::runtime_error("--index-page-async requires --index-page-cache-mib");
    }
    if (args.index_page_qd <= 0) {
        throw std::runtime_error("--index-page-qd must be > 0");
    }
    if (args.index_page_async_sync_miss_max < 0 ||
        args.index_page_async_sync_miss_max > 64) {
        throw std::runtime_error(
                "--index-page-async-sync-miss-max must be between 0 and 64");
    }
    if (args.index_page_async_sync_miss_max > 0 && !args.index_page_async) {
        throw std::runtime_error(
                "--index-page-async-sync-miss-max requires --index-page-async");
    }
    if (args.rebuild_paged_index && args.index_page_cache_mib == 0) {
        throw std::runtime_error("--rebuild-paged-index requires --index-page-cache-mib");
    }
    if (!args.mmap_index_path.empty() && args.index_page_cache_mib > 0) {
        throw std::runtime_error("mmap index and explicit index page cache are mutually exclusive");
    }
    if (!args.mmap_index_path.empty() && args.search_impl != SearchImpl::Inline) {
        throw std::runtime_error("--mmap-index-path requires --search-impl inline");
    }
    if (!args.mmap_index_path.empty() && args.distance_mode != DistanceMode::AdcBQ) {
        throw std::runtime_error("mmap index currently supports only --distance-mode adcbq");
    }
    if (!args.mmap_index_path.empty() && args.code_layout != CodeLayout::Vector) {
        throw std::runtime_error("mmap index currently supports only --code-layout vector");
    }
    if (args.rebuild_mmap_index && args.mmap_index_path.empty()) {
        throw std::runtime_error("--rebuild-mmap-index requires --mmap-index-path");
    }
    if (args.mmap_drop_cache_before_round && args.mmap_index_path.empty()) {
        throw std::runtime_error("--mmap-drop-cache-before-round requires --mmap-index-path");
    }
    if (args.adaptive_rerank) {
        if (args.no_rerank) {
            throw std::runtime_error("--adaptive-rerank cannot be used with --no-rerank");
        }
        if (args.adaptive_rerank_min == 0) args.adaptive_rerank_min = args.k;
        if (args.adaptive_rerank_min < args.k ||
            args.adaptive_rerank_min > args.candidate_pool_size) {
            throw std::runtime_error(
                    "--adaptive-rerank-min must be in [--k, --candidate-pool-size]");
        }
        if (args.adaptive_rerank_alpha < 0.0f || args.adaptive_rerank_alpha > 1.0f) {
            throw std::runtime_error("--adaptive-rerank-alpha must be in [0, 1]");
        }
    }
    if (args.adc_mode == AdcMode::Int8 && args.lut_mode != LutMode::Nibble) {
        throw std::runtime_error("--adc-mode int8 requires --lut-mode nibble");
    }
    if (args.prefetch_lookahead < 0) {
        throw std::runtime_error("--prefetch-lookahead must be >= 0");
    }
    if (args.ordered_early_abandon) {
        args.early_abandon = true;
        if (args.ordered_early_abandon_probe_bytes <= 0) {
            throw std::runtime_error("--ordered-early-abandon-probe-bytes must be > 0");
        }
        if (args.adc_mode != AdcMode::Fp32 || args.lut_mode != LutMode::Nibble) {
            throw std::runtime_error("--ordered-early-abandon currently requires --adc-mode fp32 --lut-mode nibble");
        }
    }
    if (args.adaptive_cfr) {
        if (args.search_impl != SearchImpl::Inline) {
            throw std::runtime_error("--adaptive-cfr requires --search-impl inline");
        }
        if (args.unbounded_queue) {
            throw std::runtime_error("--adaptive-cfr requires bounded queue");
        }
        if (!std::isfinite(args.adaptive_cfr_tau) || args.adaptive_cfr_tau <= 0.0f) {
            throw std::runtime_error("--adaptive-cfr-tau must be finite and > 0");
        }
        if (args.adaptive_cfr_tmin_pops < 16) {
            throw std::runtime_error("--adaptive-cfr-tmin-pops must be >= 16");
        }
        if (args.adaptive_cfr_bucket_count < 2 || args.adaptive_cfr_bucket_count > 8) {
            throw std::runtime_error("--adaptive-cfr-bucket-count must be in [2, 8]");
        }
        if (args.adaptive_cfr_paper_bucket) {
            if (static_cast<int>(args.adaptive_cfr_gammas.size()) != args.adaptive_cfr_bucket_count - 1) {
                throw std::runtime_error("--adaptive-cfr-gammas must contain bucket_count - 1 values");
            }
            float prev = -std::numeric_limits<float>::infinity();
            for (float gamma : args.adaptive_cfr_gammas) {
                if (!std::isfinite(gamma) || gamma < 0.0f || gamma > 1.0f || gamma < prev) {
                    throw std::runtime_error("--adaptive-cfr-gammas must be finite, monotone, and in [0, 1]");
                }
                prev = gamma;
            }
        }
    }
    if (args.rerank_dot_mode == RerankDotMode::Int8 && args.fp_rerank_source != FpRerankSource::Mmap) {
        throw std::runtime_error("--fp-rerank-source applies only to FP rerank modes, not --rerank-dot int8");
    }
    if (args.rerank_dot_mode == RerankDotMode::Int8 &&
        args.fp_rerank_metric != FpRerankMetric::Cosine) {
        throw std::runtime_error("--fp-rerank-metric ip requires FP32 or FP64 rerank dot products");
    }
    if (args.distance_mode == DistanceMode::XFlagBQ ||
        args.distance_mode == DistanceMode::AdcXFlagBQ) {
        if (args.search_impl != SearchImpl::Inline) {
            throw std::runtime_error("xflag distance modes require --search-impl inline");
        }
        if (args.unbounded_queue) {
            throw std::runtime_error("xflag distance modes support only bounded queue");
        }
        if (args.xflag_threshold < 0.0f &&
            (args.xflag_quantile < 0.0 || args.xflag_quantile > 1.0)) {
            throw std::runtime_error("--xflag-quantile must be in [0, 1]");
        }
        if (args.xflag_sample_rows <= 0) {
            throw std::runtime_error("--xflag-sample-rows must be > 0");
        }
    }
    return args;
}

struct FbinMmap {
    std::string path;
    int fd = -1;
    void* mapping = MAP_FAILED;
    size_t mapping_size = 0;
    int32_t n = 0;
    int32_t d = 0;
    size_t data_offset = kLegacyFbinHeaderBytes;
    const float* data = nullptr;
    std::unique_ptr<float[]> resident_data;

    explicit FbinMmap(const std::string& path_in) : path(path_in) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("open failed: " + path);
        int32_t header[2];
        if (::pread(fd, header, sizeof(header), 0) != static_cast<ssize_t>(sizeof(header))) {
            throw std::runtime_error("bad fbin header: " + path);
        }
        n = header[0];
        d = header[1];
        if (n <= 0 || d <= 0) throw std::runtime_error("invalid fbin shape: " + path);
        struct stat st {};
        if (::fstat(fd, &st) != 0 || st.st_size < 0) {
            throw std::runtime_error("fstat failed: " + path);
        }
        mapping_size = static_cast<size_t>(st.st_size);
        const size_t payload_bytes = static_cast<size_t>(n) * static_cast<size_t>(d) * sizeof(float);
        const size_t legacy_size = kLegacyFbinHeaderBytes + payload_bytes;
        const size_t aligned_size = kFpDataAlignment + payload_bytes;
        if (mapping_size == aligned_size) {
            data_offset = kFpDataAlignment;
        } else if (mapping_size == legacy_size) {
            data_offset = kLegacyFbinHeaderBytes;
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

    const float* row(size_t i) const {
        return data + i * static_cast<size_t>(d);
    }

    size_t materialize_resident() {
        if (resident_data) {
            return static_cast<size_t>(n) * static_cast<size_t>(d) * sizeof(float);
        }
        const size_t values = static_cast<size_t>(n) * static_cast<size_t>(d);
        resident_data = std::make_unique<float[]>(values);
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            std::memcpy(
                    resident_data.get() + static_cast<size_t>(i) * d,
                    row(static_cast<size_t>(i)),
                    static_cast<size_t>(d) * sizeof(float));
        }
        if (mapping != MAP_FAILED) {
            ::munmap(mapping, mapping_size);
            mapping = MAP_FAILED;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        data = resident_data.get();
        return values * sizeof(float);
    }
};

std::vector<float> read_queries(const FbinMmap& mmap, int max_queries) {
    const int nq = max_queries > 0 ? std::min<int>(mmap.n, max_queries) : mmap.n;
    std::vector<float> out(static_cast<size_t>(nq) * mmap.d);
    std::memcpy(out.data(), mmap.data, out.size() * sizeof(float));
    return out;
}

std::vector<int32_t> read_ibin(const std::string& path, int max_queries, int32_t& n, int32_t& dim) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in) throw std::runtime_error("bad ibin header: " + path);
    const int rows = max_queries > 0 ? std::min<int>(n, max_queries) : n;
    std::vector<int32_t> out(static_cast<size_t>(rows) * dim);
    in.read(reinterpret_cast<char*>(out.data()), out.size() * sizeof(int32_t));
    if (!in) throw std::runtime_error("bad ibin payload: " + path);
    n = rows;
    return out;
}

void normalize_rows_inplace(float* data, int64_t n, int dim) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        float* row = data + static_cast<size_t>(i) * dim;
        double ss = 0.0;
        for (int j = 0; j < dim; ++j) {
            ss += static_cast<double>(row[j]) * row[j];
        }
        const float inv_norm = 1.0f / static_cast<float>(std::sqrt(std::max(ss, 1e-30)));
        for (int j = 0; j < dim; ++j) {
            row[j] *= inv_norm;
        }
    }
}

std::vector<float> normalized_train_block(const FbinMmap& train, int64_t start, int64_t rows) {
    std::vector<float> block(static_cast<size_t>(rows) * train.d);
    std::memcpy(block.data(), train.row(static_cast<size_t>(start)), block.size() * sizeof(float));
    normalize_rows_inplace(block.data(), rows, train.d);
    return block;
}


constexpr size_t kCodeAlignment = 64;

size_t round_up_to_alignment(size_t value, size_t alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

struct AlignedPackedCodes {
    uint8_t* data = nullptr;
    size_t n = 0;
    size_t code_size = 0;
    size_t stride = 0;
    size_t bytes = 0;

    AlignedPackedCodes() = default;

    AlignedPackedCodes(const uint8_t* src, size_t n_in, size_t code_size_in)
            : n(n_in),
              code_size(code_size_in),
              stride(round_up_to_alignment(code_size_in, kCodeAlignment)),
              bytes(n_in * stride) {
        if (n == 0 || code_size == 0) return;
        if (stride < code_size || bytes / stride != n) {
            throw std::runtime_error("aligned code storage size overflow");
        }
        void* ptr = nullptr;
        if (::posix_memalign(&ptr, kCodeAlignment, bytes) != 0) {
            throw std::runtime_error("posix_memalign failed for aligned code storage");
        }
        data = static_cast<uint8_t*>(ptr);
        if (stride == code_size) {
#pragma omp parallel for schedule(static)
            for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
                std::memcpy(data + static_cast<size_t>(row) * stride,
                            src + static_cast<size_t>(row) * code_size,
                            code_size);
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
                uint8_t* dst = data + static_cast<size_t>(row) * stride;
                std::memcpy(dst, src + static_cast<size_t>(row) * code_size, code_size);
                std::memset(dst + code_size, 0, stride - code_size);
            }
        }
    }

    ~AlignedPackedCodes() {
        std::free(data);
    }

    AlignedPackedCodes(const AlignedPackedCodes&) = delete;
    AlignedPackedCodes& operator=(const AlignedPackedCodes&) = delete;

    AlignedPackedCodes(AlignedPackedCodes&& other) noexcept
            : data(other.data),
              n(other.n),
              code_size(other.code_size),
              stride(other.stride),
              bytes(other.bytes) {
        other.data = nullptr;
        other.n = 0;
        other.code_size = 0;
        other.stride = 0;
        other.bytes = 0;
    }

    AlignedPackedCodes& operator=(AlignedPackedCodes&& other) noexcept {
        if (this != &other) {
            std::free(data);
            data = other.data;
            n = other.n;
            code_size = other.code_size;
            stride = other.stride;
            bytes = other.bytes;
            other.data = nullptr;
            other.n = 0;
            other.code_size = 0;
            other.stride = 0;
            other.bytes = 0;
        }
        return *this;
    }
};

struct Block32GroupMajorCodes {
    static constexpr size_t kBlockRows = 32;

    uint8_t* data = nullptr;
    size_t n = 0;
    size_t code_size = 0;
    size_t blocks = 0;
    size_t block_stride = 0;
    size_t bytes = 0;

    Block32GroupMajorCodes() = default;

    Block32GroupMajorCodes(const uint8_t* src, size_t n_in, size_t code_size_in)
            : n(n_in),
              code_size(code_size_in),
              blocks((n_in + kBlockRows - 1) / kBlockRows),
              block_stride(code_size_in * kBlockRows),
              bytes(blocks * block_stride) {
        if (n == 0 || code_size == 0) return;
        if (block_stride / kBlockRows != code_size || (blocks != 0 && bytes / blocks != block_stride)) {
            throw std::runtime_error("block32 code storage size overflow");
        }
        void* ptr = nullptr;
        if (::posix_memalign(&ptr, kCodeAlignment, bytes) != 0) {
            throw std::runtime_error("posix_memalign failed for block32 code storage");
        }
        data = static_cast<uint8_t*>(ptr);
        std::memset(data, 0, bytes);
#pragma omp parallel for schedule(static)
        for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
            const size_t r = static_cast<size_t>(row);
            const size_t block = r / kBlockRows;
            const size_t lane = r % kBlockRows;
            const uint8_t* src_row = src + r * code_size;
            uint8_t* dst_block = data + block * block_stride;
            for (size_t byte = 0; byte < code_size; ++byte) {
                dst_block[byte * kBlockRows + lane] = src_row[byte];
            }
        }
    }

    ~Block32GroupMajorCodes() {
        std::free(data);
    }

    Block32GroupMajorCodes(const Block32GroupMajorCodes&) = delete;
    Block32GroupMajorCodes& operator=(const Block32GroupMajorCodes&) = delete;

    Block32GroupMajorCodes(Block32GroupMajorCodes&& other) noexcept
            : data(other.data),
              n(other.n),
              code_size(other.code_size),
              blocks(other.blocks),
              block_stride(other.block_stride),
              bytes(other.bytes) {
        other.data = nullptr;
        other.n = 0;
        other.code_size = 0;
        other.blocks = 0;
        other.block_stride = 0;
        other.bytes = 0;
    }

    Block32GroupMajorCodes& operator=(Block32GroupMajorCodes&& other) noexcept {
        if (this != &other) {
            std::free(data);
            data = other.data;
            n = other.n;
            code_size = other.code_size;
            blocks = other.blocks;
            block_stride = other.block_stride;
            bytes = other.bytes;
            other.data = nullptr;
            other.n = 0;
            other.code_size = 0;
            other.blocks = 0;
            other.block_stride = 0;
            other.bytes = 0;
        }
        return *this;
    }

    const uint8_t* byte_ptr(faiss::idx_t id, size_t byte) const {
        const size_t r = static_cast<size_t>(id);
        return data + (r / kBlockRows) * block_stride + byte * kBlockRows + (r % kBlockRows);
    }

    uint8_t byte_at(faiss::idx_t id, size_t byte) const {
        return *byte_ptr(id, byte);
    }
};

//candidate를 bq 1bit로 만들기
void pack_pgvector_sign_bits(const float* x, size_t n, int dim, std::vector<uint8_t>& codes) {
    adcbq::pack_sign_bits(x, n, dim, codes);
}

void pack_pgvector_xflag_bits(
        const float* x,
        size_t n,
        int dim,
        float threshold,
        std::vector<uint8_t>& flags) {
    const size_t code_size = static_cast<size_t>(dim) / 8;
    flags.assign(n * code_size, 0);
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
        const float* src = x + static_cast<size_t>(row) * static_cast<size_t>(dim);
        uint8_t* dst = flags.data() + static_cast<size_t>(row) * code_size;
        for (int j = 0; j < dim; ++j) {
            if (std::abs(src[j]) < threshold) {
                dst[j / 8] |= static_cast<uint8_t>(1u << (7 - (j % 8)));
            }
        }
    }
}

float compute_xflag_threshold_quantile(
        const FbinMmap& train,
        double quantile,
        int sample_rows) {
    if (train.n <= 0 || train.d <= 0) {
        throw std::runtime_error("empty train data for xflag threshold");
    }
    const int64_t rows = std::min<int64_t>(train.n, sample_rows);
    std::vector<float> values;
    values.reserve(static_cast<size_t>(rows) * static_cast<size_t>(train.d));
    for (int64_t r = 0; r < rows; ++r) {
        const int64_t src_row = rows == 1 ? 0 : (r * static_cast<int64_t>(train.n - 1)) / (rows - 1);
        const float* row = train.row(static_cast<size_t>(src_row));
        for (int j = 0; j < train.d; ++j) {
            values.push_back(std::abs(row[j]));
        }
    }
    if (values.empty()) return 0.0f;
    const double clamped = std::max(0.0, std::min(1.0, quantile));
    const size_t kth = std::min(
            values.size() - 1,
            static_cast<size_t>(std::llround(clamped * static_cast<double>(values.size() - 1))));
    std::nth_element(values.begin(), values.begin() + kth, values.end());
    return values[kth];
}

std::string resolved_xflag_code_path(const Args& args) {
    return args.xflag_code_path.empty() ? args.index + ".xflag" : args.xflag_code_path;
}

struct XFlagCodeFile {
    float threshold = -1.0f;
    double quantile = 0.0;
    int32_t sample_rows = 0;
    std::vector<uint8_t> codes;
};

void write_xflag_code_file(
        const std::string& path,
        const std::vector<uint8_t>& codes,
        uint64_t n,
        uint32_t dim,
        uint32_t code_size,
        float threshold,
        double quantile,
        int32_t sample_rows) {
    const fs::path out_path(path);
    if (!out_path.parent_path().empty()) {
        fs::create_directories(out_path.parent_path());
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("open failed for xflag code write: " + path);
    const char magic[8] = {'X', 'F', 'B', 'Q', '0', '0', '0', '1'};
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(&n), sizeof(n));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&code_size), sizeof(code_size));
    out.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
    out.write(reinterpret_cast<const char*>(&quantile), sizeof(quantile));
    out.write(reinterpret_cast<const char*>(&sample_rows), sizeof(sample_rows));
    out.write(reinterpret_cast<const char*>(codes.data()), static_cast<std::streamsize>(codes.size()));
    if (!out) throw std::runtime_error("bad xflag code write: " + path);
}

XFlagCodeFile read_xflag_code_file(
        const std::string& path,
        uint64_t expected_n,
        uint32_t expected_dim,
        uint32_t expected_code_size) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed for xflag code read: " + path);
    char magic[8];
    uint64_t n = 0;
    uint32_t dim = 0;
    uint32_t code_size = 0;
    XFlagCodeFile out;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    in.read(reinterpret_cast<char*>(&code_size), sizeof(code_size));
    in.read(reinterpret_cast<char*>(&out.threshold), sizeof(out.threshold));
    in.read(reinterpret_cast<char*>(&out.quantile), sizeof(out.quantile));
    in.read(reinterpret_cast<char*>(&out.sample_rows), sizeof(out.sample_rows));
    const char expected_magic[8] = {'X', 'F', 'B', 'Q', '0', '0', '0', '1'};
    if (!in || std::memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        throw std::runtime_error("bad xflag code header: " + path);
    }
    if (n != expected_n || dim != expected_dim || code_size != expected_code_size) {
        throw std::runtime_error("xflag code shape does not match index/train: " + path);
    }
    const uint64_t payload_size = n * static_cast<uint64_t>(code_size);
    if (payload_size / std::max<uint32_t>(code_size, 1) != n) {
        throw std::runtime_error("xflag code payload size overflow: " + path);
    }
    out.codes.resize(static_cast<size_t>(payload_size));
    in.read(reinterpret_cast<char*>(out.codes.data()), static_cast<std::streamsize>(out.codes.size()));
    if (!in) throw std::runtime_error("bad xflag code payload: " + path);
    return out;
}


struct PackedAsymBQDistanceComputer : faiss::DistanceComputer {
    using CodeResolver = const uint8_t* (*)(const void*, faiss::idx_t);
    const uint8_t* codes = nullptr;
    const uint8_t* const* indirect_codes = nullptr;
    const void* code_resolver_context = nullptr;
    CodeResolver code_resolver = nullptr;
    const uint8_t* block32_codes = nullptr;
    size_t code_size = 0;
    size_t code_stride = 0;
    size_t block32_block_stride = 0;
    int dim = 0;
    AdcMode mode = AdcMode::Fp32;
    LutMode lut_mode = LutMode::Byte;
    bool use_early_abandon = false;
    const float* query_direct = nullptr;
    float query_direct_sum = 0.0f;
    std::vector<float> lut_fp32;
    std::vector<int16_t> lut_i16;
    // Byte-split, four-codebook-grouped copy of the int16 nibble LUT.  It is
    // consumed by the AVX-512 pshufb kernel used by the IVF scanner.  Entries
    // are biased by 32768 so the exact signed int16 sum can be reconstructed
    // with unsigned byte arithmetic and a single constant subtraction.
    std::vector<uint8_t> lut_i16_hacc;
    std::vector<int8_t> lut_i8;
    std::vector<uint8_t> lut_u8;
    bool use_ordered_early_abandon = false;
    size_t ordered_early_abandon_probe_bytes = 16;
    std::vector<uint16_t> byte_order;
    std::vector<float> suffix_abs_by_byte;

    PackedAsymBQDistanceComputer(
            const uint8_t* codes_in,
            size_t code_size_in,
            size_t code_stride_in,
            int dim_in,
            AdcMode mode_in,
            LutMode lut_mode_in,
            bool early_abandon_in = false,
            bool ordered_early_abandon_in = false,
            size_t ordered_probe_bytes_in = 16,
            const uint8_t* block32_codes_in = nullptr,
            size_t block32_block_stride_in = 0)
            : codes(codes_in),
              block32_codes(block32_codes_in),
              code_size(code_size_in),
              code_stride(code_stride_in == 0 ? code_size_in : code_stride_in),
              block32_block_stride(block32_block_stride_in),
              dim(dim_in),
              mode(mode_in),
              lut_mode(lut_mode_in),
              use_early_abandon(early_abandon_in),
              use_ordered_early_abandon(ordered_early_abandon_in),
              ordered_early_abandon_probe_bytes(ordered_probe_bytes_in),
              lut_fp32(mode_in == AdcMode::Fp32 ? lut_entries(code_size_in, lut_mode_in) : 0),
              lut_i16(mode_in == AdcMode::Int16 ? lut_entries(code_size_in, lut_mode_in) : 0),
              lut_i16_hacc(
                      mode_in == AdcMode::Int16 && lut_mode_in == LutMode::Nibble
                              ? code_size_in * 64
                              : 0),
              lut_i8(0),
              lut_u8(mode_in == AdcMode::Int8 ? lut_entries(code_size_in, lut_mode_in) : 0),
              byte_order(ordered_early_abandon_in ? code_size_in : 0),
              suffix_abs_by_byte(early_abandon_in ? code_size_in + 1 : 0) {}

    static size_t lut_entries(size_t code_size, LutMode lut_mode) {
        return lut_mode == LutMode::Byte ? code_size * 256 : code_size * 32;
        // 256은 byte lut, 32는 nibble lut
    }

    const uint8_t* code_at(faiss::idx_t i) const {
        if (indirect_codes) return indirect_codes[static_cast<size_t>(i)];
        if (code_resolver) return code_resolver(code_resolver_context, i);
        return codes + static_cast<size_t>(i) * code_stride;
    }

    void set_indirect_codes(const uint8_t* const* pointers) {
        indirect_codes = pointers;
    }

    void set_code_resolver(CodeResolver resolver, const void* context) {
        code_resolver = resolver;
        code_resolver_context = context;
    }

    bool has_block32_codes() const {
        return block32_codes != nullptr && block32_block_stride != 0;
    }

    const uint8_t* block32_byte_ptr(faiss::idx_t i, size_t byte) const {
        const size_t r = static_cast<size_t>(i);
        return block32_codes + (r / 32) * block32_block_stride + byte * 32 + (r & 31);
    }

    uint8_t block32_byte_at(faiss::idx_t i, size_t byte) const {
        return *block32_byte_ptr(i, byte);
    }

    void set_query(const float* q) override {
        if (mode == AdcMode::Int8) {
            set_query_int8(q);
        } else if (mode == AdcMode::Int16) {
            set_query_int16(q);
        } else if (mode == AdcMode::Fp32Direct) {
            set_query_fp32_direct(q);
        } else {
            set_query_fp32(q);
        }
    }

    // 16개 lut만드는 로직
    template <typename T>
    static void build_nibble_row(T* row, const T* delta, T base_score) {
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
    }

    void build_suffix_abs_by_byte(const float* q) {
        if (!use_early_abandon) return;
        if (suffix_abs_by_byte.size() != code_size + 1) {
            suffix_abs_by_byte.resize(code_size + 1);
        }
        std::fill(suffix_abs_by_byte.begin(), suffix_abs_by_byte.end(), 0.0f);

        std::vector<std::pair<float, uint16_t>> weighted_bytes;
        if (use_ordered_early_abandon) {
            weighted_bytes.reserve(code_size);
            if (byte_order.size() != code_size) byte_order.resize(code_size);
        }

        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            const int remaining = std::max(0, std::min(8, dim - base));
            float abs_sum = 0.0f;
            for (int bit = 0; bit < remaining; ++bit) {
                abs_sum += std::abs(q[base + bit]);
            }
            if (use_ordered_early_abandon) {
                weighted_bytes.emplace_back(abs_sum, static_cast<uint16_t>(byte));
            } else {
                suffix_abs_by_byte[byte] = abs_sum;
            }
        }

        if (use_ordered_early_abandon) {
            std::stable_sort(
                    weighted_bytes.begin(),
                    weighted_bytes.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            for (size_t rank = 0; rank < weighted_bytes.size(); ++rank) {
                suffix_abs_by_byte[rank] = weighted_bytes[rank].first;
                byte_order[rank] = weighted_bytes[rank].second;
            }
        }

        for (size_t byte = code_size; byte-- > 0;) {
            suffix_abs_by_byte[byte] += suffix_abs_by_byte[byte + 1];
        }
    }

    void set_query_fp32_byte(const float* q) {
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            const int remaining = std::min(8, dim - base);
            float delta[8] = {};
            float base_score = 0.0f;
            for (int bit = 0; bit < remaining; ++bit) {
                const float v = q[base + bit];
                base_score -= v;
                delta[bit] = 2.0f * v;
            }

            float* row = lut_fp32.data() + byte * 256;
            row[0] = base_score;
            for (int code = 1; code < 256; ++code) {
                const int low = code & -code;
                const int bit = 7 - __builtin_ctz(static_cast<unsigned>(low));
                row[code] = row[code ^ low] + delta[bit];
            }
        }
    }

    // nibble lut만드는 코드
    void set_query_fp32_nibble(const float* q) {
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            float* row = lut_fp32.data() + byte * 32;
            for (int half = 0; half < 2; ++half) {
                const int offset = half * 4;
                const int remaining = std::min(4, dim - base - offset);
                float delta[4] = {};
                float base_score = 0.0f;
                for (int bit = 0; bit < remaining; ++bit) {
                    const float v = q[base + offset + bit];
                    base_score -= v; // base code = 0000 이니깐 -1
                    delta[bit] = 2.0f * v;
                }
                build_nibble_row(row + half * 16, delta, base_score);
            }
        }
    }

    void set_query_fp32(const float* q) {
        if (lut_mode == LutMode::Nibble) {
            set_query_fp32_nibble(q);
        } else {
            set_query_fp32_byte(q);
        }
        build_suffix_abs_by_byte(q);
    }

    void set_query_fp32_direct(const float* q) {
        query_direct = q;
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        int i = 0;
        for (; i + 4 <= dim; i += 4) {
            s0 += q[i + 0];
            s1 += q[i + 1];
            s2 += q[i + 2];
            s3 += q[i + 3];
        }
        float total = s0 + s1 + s2 + s3;
        for (; i < dim; ++i) total += q[i];
        query_direct_sum = total;
    }

    static uint8_t reverse_bits8(uint8_t x) {
        x = static_cast<uint8_t>(((x & 0xf0u) >> 4) | ((x & 0x0fu) << 4));
        x = static_cast<uint8_t>(((x & 0xccu) >> 2) | ((x & 0x33u) << 2));
        x = static_cast<uint8_t>(((x & 0xaau) >> 1) | ((x & 0x55u) << 1));
        return x;
    }

#if defined(ADCBQ_FORCE_FP32_DIRECT_SCALAR) && defined(__GNUC__)
    __attribute__((noinline, optimize("no-tree-vectorize")))
#endif
    float distance_to_code_fp32_direct_scalar(const uint8_t* code) const {
        const float* q = query_direct;
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        size_t byte = 0;
        for (; byte + 4 <= code_size; byte += 4) {
            uint8_t c = code[byte + 0];
            const int base0 = static_cast<int>((byte + 0) * 8);
            s0 += (c & 0x80u) ? q[base0 + 0] : 0.0f;
            s0 += (c & 0x40u) ? q[base0 + 1] : 0.0f;
            s0 += (c & 0x20u) ? q[base0 + 2] : 0.0f;
            s0 += (c & 0x10u) ? q[base0 + 3] : 0.0f;
            s0 += (c & 0x08u) ? q[base0 + 4] : 0.0f;
            s0 += (c & 0x04u) ? q[base0 + 5] : 0.0f;
            s0 += (c & 0x02u) ? q[base0 + 6] : 0.0f;
            s0 += (c & 0x01u) ? q[base0 + 7] : 0.0f;

            c = code[byte + 1];
            const int base1 = static_cast<int>((byte + 1) * 8);
            s1 += (c & 0x80u) ? q[base1 + 0] : 0.0f;
            s1 += (c & 0x40u) ? q[base1 + 1] : 0.0f;
            s1 += (c & 0x20u) ? q[base1 + 2] : 0.0f;
            s1 += (c & 0x10u) ? q[base1 + 3] : 0.0f;
            s1 += (c & 0x08u) ? q[base1 + 4] : 0.0f;
            s1 += (c & 0x04u) ? q[base1 + 5] : 0.0f;
            s1 += (c & 0x02u) ? q[base1 + 6] : 0.0f;
            s1 += (c & 0x01u) ? q[base1 + 7] : 0.0f;

            c = code[byte + 2];
            const int base2 = static_cast<int>((byte + 2) * 8);
            s2 += (c & 0x80u) ? q[base2 + 0] : 0.0f;
            s2 += (c & 0x40u) ? q[base2 + 1] : 0.0f;
            s2 += (c & 0x20u) ? q[base2 + 2] : 0.0f;
            s2 += (c & 0x10u) ? q[base2 + 3] : 0.0f;
            s2 += (c & 0x08u) ? q[base2 + 4] : 0.0f;
            s2 += (c & 0x04u) ? q[base2 + 5] : 0.0f;
            s2 += (c & 0x02u) ? q[base2 + 6] : 0.0f;
            s2 += (c & 0x01u) ? q[base2 + 7] : 0.0f;

            c = code[byte + 3];
            const int base3 = static_cast<int>((byte + 3) * 8);
            s3 += (c & 0x80u) ? q[base3 + 0] : 0.0f;
            s3 += (c & 0x40u) ? q[base3 + 1] : 0.0f;
            s3 += (c & 0x20u) ? q[base3 + 2] : 0.0f;
            s3 += (c & 0x10u) ? q[base3 + 3] : 0.0f;
            s3 += (c & 0x08u) ? q[base3 + 4] : 0.0f;
            s3 += (c & 0x04u) ? q[base3 + 5] : 0.0f;
            s3 += (c & 0x02u) ? q[base3 + 6] : 0.0f;
            s3 += (c & 0x01u) ? q[base3 + 7] : 0.0f;
        }
        float selected = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            const uint8_t c = code[byte];
            const int base = static_cast<int>(byte * 8);
            selected += (c & 0x80u) ? q[base + 0] : 0.0f;
            selected += (c & 0x40u) ? q[base + 1] : 0.0f;
            selected += (c & 0x20u) ? q[base + 2] : 0.0f;
            selected += (c & 0x10u) ? q[base + 3] : 0.0f;
            selected += (c & 0x08u) ? q[base + 4] : 0.0f;
            selected += (c & 0x04u) ? q[base + 5] : 0.0f;
            selected += (c & 0x02u) ? q[base + 6] : 0.0f;
            selected += (c & 0x01u) ? q[base + 7] : 0.0f;
        }
        return -(2.0f * selected - query_direct_sum);
    }

#if defined(__AVX512F__)
    float distance_to_code_fp32_direct_avx512(const uint8_t* code) const {
        const float* q = query_direct;
        __m512 selected_acc = _mm512_setzero_ps();
        size_t byte = 0;
        int base = 0;
        for (; byte + 2 <= code_size; byte += 2, base += 16) {
            const __mmask16 mask = static_cast<__mmask16>(
                    static_cast<uint16_t>(reverse_bits8(code[byte])) |
                    (static_cast<uint16_t>(reverse_bits8(code[byte + 1])) << 8));
            selected_acc = _mm512_add_ps(selected_acc, _mm512_maskz_loadu_ps(mask, q + base));
        }
        float selected = _mm512_reduce_add_ps(selected_acc);
        for (; byte < code_size; ++byte, base += 8) {
            const uint8_t c = code[byte];
            selected += (c & 0x80u) ? q[base + 0] : 0.0f;
            selected += (c & 0x40u) ? q[base + 1] : 0.0f;
            selected += (c & 0x20u) ? q[base + 2] : 0.0f;
            selected += (c & 0x10u) ? q[base + 3] : 0.0f;
            selected += (c & 0x08u) ? q[base + 4] : 0.0f;
            selected += (c & 0x04u) ? q[base + 5] : 0.0f;
            selected += (c & 0x02u) ? q[base + 6] : 0.0f;
            selected += (c & 0x01u) ? q[base + 7] : 0.0f;
        }
        return -(2.0f * selected - query_direct_sum);
    }
#endif

    float distance_to_code_fp32_direct(const uint8_t* code) const {
#if defined(ADCBQ_FORCE_FP32_DIRECT_SCALAR)
        return distance_to_code_fp32_direct_scalar(code);
#elif defined(__AVX512F__)
        return distance_to_code_fp32_direct_avx512(code);
#else
        return distance_to_code_fp32_direct_scalar(code);
#endif
    }

    static int16_t quantize_lut_entry_i16(float value, float scale) {
        const float scaled = std::round(value * scale);
        const float clipped = std::max(-32767.0f, std::min(32767.0f, scaled));
        return static_cast<int16_t>(clipped);
    }

    float int16_lut_scale(const float* q) const {
        const int block_size = lut_mode == LutMode::Byte ? 8 : 4;
        float max_abs = 0.0f;
        for (int base = 0; base < dim; base += block_size) {
            const int remaining = std::min(block_size, dim - base);
            float abs_sum = 0.0f;
            for (int bit = 0; bit < remaining; ++bit) {
                abs_sum += std::abs(q[base + bit]);
            }
            max_abs = std::max(max_abs, abs_sum);
        }
        return max_abs > 0.0f ? 32767.0f / max_abs : 1.0f;
    }

    void set_query_int16_byte(const float* q) {
        const float scale = int16_lut_scale(q);
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            const int remaining = std::min(8, dim - base);
            float delta[8] = {};
            float base_score = 0.0f;
            for (int bit = 0; bit < remaining; ++bit) {
                const float v = q[base + bit];
                base_score -= v;
                delta[bit] = 2.0f * v;
            }

            float row[256];
            row[0] = base_score;
            for (int code = 1; code < 256; ++code) {
                const int low = code & -code;
                const int bit = 7 - __builtin_ctz(static_cast<unsigned>(low));
                row[code] = row[code ^ low] + delta[bit];
            }
            int16_t* output = lut_i16.data() + byte * 256;
            for (int code = 0; code < 256; ++code) {
                output[code] = quantize_lut_entry_i16(row[code], scale);
            }
        }
    }

    void set_query_int16_nibble_scalar(const float* q) {
        const float scale = int16_lut_scale(q);
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            int16_t* output = lut_i16.data() + byte * 32;
            for (int half = 0; half < 2; ++half) {
                const int offset = half * 4;
                const int remaining = std::min(4, dim - base - offset);
                float delta[4] = {};
                float base_score = 0.0f;
                for (int bit = 0; bit < remaining; ++bit) {
                    const float v = q[base + offset + bit];
                    base_score -= v;
                    delta[bit] = 2.0f * v;
                }
                float row[16];
                build_nibble_row(row, delta, base_score);
                for (int code = 0; code < 16; ++code) {
                    output[half * 16 + code] = quantize_lut_entry_i16(row[code], scale);
                }
            }
        }
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
    static __m256i quantize_nibble_row_i16_avx512(
            const float* row,
            float scale) {
        const __m512 values = _mm512_loadu_ps(row);
        const __m512 scaled = _mm512_mul_ps(values, _mm512_set1_ps(scale));

        // Match std::round exactly: add copysign(0.5, x), then truncate.
        // The default AVX conversion rounds ties to even, which would change
        // LUT entries at half-integers and is therefore not used here.
        const __m512i scaled_bits = _mm512_castps_si512(scaled);
        const __m512i sign_bits = _mm512_and_si512(
                scaled_bits, _mm512_set1_epi32(static_cast<int>(0x80000000u)));
        const __m512i signed_half_bits = _mm512_or_si512(
                sign_bits, _mm512_set1_epi32(0x3f000000));
        const __m512 adjusted = _mm512_add_ps(
                scaled, _mm512_castsi512_ps(signed_half_bits));
        __m512i rounded = _mm512_cvttps_epi32(adjusted);
        rounded = _mm512_max_epi32(rounded, _mm512_set1_epi32(-32767));
        rounded = _mm512_min_epi32(rounded, _mm512_set1_epi32(32767));
        return _mm512_cvtsepi32_epi16(rounded);
    }

    void set_query_int16_nibble_avx512(const float* q) {
        const float scale = int16_lut_scale(q);
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            int16_t* output = lut_i16.data() + byte * 32;
            for (int half = 0; half < 2; ++half) {
                const int offset = half * 4;
                const int remaining = std::min(4, dim - base - offset);
                float delta[4] = {};
                float base_score = 0.0f;
                for (int bit = 0; bit < remaining; ++bit) {
                    const float v = q[base + offset + bit];
                    base_score -= v;
                    delta[bit] = 2.0f * v;
                }
                alignas(64) float row[16];
                build_nibble_row(row, delta, base_score);
                const __m256i quantized =
                        quantize_nibble_row_i16_avx512(row, scale);
                _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(output + half * 16),
                        quantized);
            }
        }
    }
#endif

    void set_query_int16_nibble(const float* q) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__)
        set_query_int16_nibble_avx512(q);
#else
        set_query_int16_nibble_scalar(q);
#endif
    }

    void pack_int16_nibble_hacc_scalar() {
        const size_t num_codebooks = code_size * 2;
        for (size_t codebook = 0; codebook < num_codebooks; ++codebook) {
            const size_t group = codebook / 4;
            const size_t lane = codebook % 4;
            uint8_t* low = lut_i16_hacc.data() + group * 128 + lane * 16;
            uint8_t* high = low + 64;
            const int16_t* row = lut_i16.data() + codebook * 16;
            for (size_t entry = 0; entry < 16; ++entry) {
                const uint16_t biased = static_cast<uint16_t>(
                        static_cast<int32_t>(row[entry]) + 32768);
                low[entry] = static_cast<uint8_t>(biased & 0xffu);
                high[entry] = static_cast<uint8_t>(biased >> 8);
            }
        }
    }

#if defined(__AVX512BW__) && defined(__AVX512VL__)
    void pack_int16_nibble_hacc_avx512() {
        const size_t num_codebooks = code_size * 2;
        const __m256i sign_bias =
                _mm256_set1_epi16(static_cast<int16_t>(0x8000u));
        for (size_t codebook = 0; codebook < num_codebooks; ++codebook) {
            const size_t group = codebook / 4;
            const size_t lane = codebook % 4;
            uint8_t* low = lut_i16_hacc.data() + group * 128 + lane * 16;
            uint8_t* high = low + 64;
            const int16_t* row = lut_i16.data() + codebook * 16;
            const __m256i values = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(row));
            const __m256i biased = _mm256_xor_si256(values, sign_bias);
            const __m128i low_bytes = _mm256_cvtepi16_epi8(biased);
            const __m128i high_bytes = _mm256_cvtepi16_epi8(
                    _mm256_srli_epi16(biased, 8));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(low), low_bytes);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(high), high_bytes);
        }
    }
#endif

    void pack_int16_nibble_hacc() {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
        pack_int16_nibble_hacc_avx512();
#else
        pack_int16_nibble_hacc_scalar();
#endif
    }

    void set_query_int16_scalar_reference(const float* q) {
        if (lut_mode == LutMode::Nibble) {
            set_query_int16_nibble_scalar(q);
            const size_t num_codebooks = code_size * 2;
            if (lut_i16_hacc.size() != num_codebooks * 32) {
                lut_i16_hacc.resize(num_codebooks * 32);
            }
            pack_int16_nibble_hacc_scalar();
        } else {
            set_query_int16_byte(q);
        }
    }

    void set_query_int16(const float* q) {
        if (lut_mode == LutMode::Nibble) {
            set_query_int16_nibble(q);
            // Pack four consecutive 4-D codebooks into 128 bytes: the first
            // 64 bytes contain the low byte of each biased uint16 row and the
            // next 64 contain the high byte.  This matches one AVX-512
            // pshufb group and is built once per query, not once per list.
            const size_t num_codebooks = code_size * 2;
            if (lut_i16_hacc.size() != num_codebooks * 32) {
                lut_i16_hacc.resize(num_codebooks * 32);
            }
            pack_int16_nibble_hacc();
        } else {
            set_query_int16_byte(q);
        }
    }

    static float nibble_score(const float* q, int base, int remaining, int code) {
        float score = 0.0f;
        for (int bit = 0; bit < remaining; ++bit) {
            const bool one = (code & (1 << (3 - bit))) != 0;
            const float v = q[base + bit];
            score += one ? v : -v;
        }
        return score;
    }

    void set_query_int8(const float* q) {
        float max_abs = 0.0f;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            for (int half = 0; half < 2; ++half) {
                const int offset = half * 4;
                const int remaining = std::max(0, std::min(4, dim - base - offset));
                float abs_sum = 0.0f;
                for (int bit = 0; bit < remaining; ++bit) {
                    abs_sum += std::abs(q[base + offset + bit]);
                }
                max_abs = std::max(max_abs, abs_sum);
            }
        }

        const float scale = max_abs > 0.0f ? 127.0f / max_abs : 1.0f;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            uint8_t* row = lut_u8.data() + byte * 32;
            for (int half = 0; half < 2; ++half) {
                const int offset = half * 4;
                const int remaining = std::max(0, std::min(4, dim - base - offset));
                float delta[4] = {};
                float base_score = 0.0f;
                for (int bit = 0; bit < remaining; ++bit) {
                    const float v = q[base + offset + bit] * scale;
                    base_score -= v;
                    delta[bit] = 2.0f * v;
                }
                float scores[16];
                build_nibble_row(scores, delta, base_score);
                for (int code = 0; code < 16; ++code) {
                    const float scaled = std::round(scores[code]);
                    const int32_t v = static_cast<int32_t>(std::max(-127.0f, std::min(127.0f, scaled)));
                    row[half * 16 + code] = static_cast<uint8_t>(v + 128);
                }
            }
        }
    }

    float distance_to_code_fp32_byte(const uint8_t* code) const {
        const float* lut = lut_fp32.data();
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        size_t byte = 0;
        for (; byte + 16 <= code_size; byte += 16) {
            __builtin_prefetch(code + byte + 64, 0, 1);
            const float* row = lut + byte * 256;
            s0 += row[code[byte + 0]];
            s1 += row[256 + code[byte + 1]];
            s2 += row[512 + code[byte + 2]];
            s3 += row[768 + code[byte + 3]];
            s0 += row[1024 + code[byte + 4]];
            s1 += row[1280 + code[byte + 5]];
            s2 += row[1536 + code[byte + 6]];
            s3 += row[1792 + code[byte + 7]];
            s0 += row[2048 + code[byte + 8]];
            s1 += row[2304 + code[byte + 9]];
            s2 += row[2560 + code[byte + 10]];
            s3 += row[2816 + code[byte + 11]];
            s0 += row[3072 + code[byte + 12]];
            s1 += row[3328 + code[byte + 13]];
            s2 += row[3584 + code[byte + 14]];
            s3 += row[3840 + code[byte + 15]];
        }
        float ip = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            ip += lut[byte * 256 + code[byte]];
        }
        return -ip;
    }

    //candidate 하나 거리 계산 로직
    float distance_to_code_fp32_nibble(const uint8_t* code) const {
        const float* lut = lut_fp32.data();
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        size_t byte = 0;
        for (; byte + 8 <= code_size; byte += 8) {
            __builtin_prefetch(code + byte + 64, 0, 1);
            const float* row = lut + byte * 32;
            uint8_t c = code[byte + 0];
            s0 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = code[byte + 1];
            s1 += row[32 + (c >> 4)] + row[48 + (c & 0x0f)];
            c = code[byte + 2];
            s2 += row[64 + (c >> 4)] + row[80 + (c & 0x0f)];
            c = code[byte + 3];
            s3 += row[96 + (c >> 4)] + row[112 + (c & 0x0f)];
            c = code[byte + 4];
            s0 += row[128 + (c >> 4)] + row[144 + (c & 0x0f)];
            c = code[byte + 5];
            s1 += row[160 + (c >> 4)] + row[176 + (c & 0x0f)];
            c = code[byte + 6];
            s2 += row[192 + (c >> 4)] + row[208 + (c & 0x0f)];
            c = code[byte + 7];
            s3 += row[224 + (c >> 4)] + row[240 + (c & 0x0f)];
        }
        float ip = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            const uint8_t c = code[byte];
            const float* row = lut + byte * 32;
            ip += row[c >> 4] + row[16 + (c & 0x0f)];
        }
        return -ip;
    }

    float distance_to_code_fp32_nibble_ordered_pruned(const uint8_t* code, float max_distance) const {
        if (!use_ordered_early_abandon || !std::isfinite(max_distance) ||
            suffix_abs_by_byte.empty() || byte_order.empty()) {
            return distance_to_code_fp32_nibble_pruned(code, max_distance);
        }
        const float target_ip = -max_distance;
        const float* lut = lut_fp32.data();
        float ip = 0.0f;
        const size_t probe_bytes = std::min(code_size, ordered_early_abandon_probe_bytes);
        for (size_t rank = 0; rank < probe_bytes; ++rank) {
            const size_t byte = byte_order[rank];
            const uint8_t c = code[byte];
            const float* row = lut + byte * 32;
            ip += row[c >> 4] + row[16 + (c & 0x0f)];
            if (ip + suffix_abs_by_byte[rank + 1] < target_ip) {
                return max_distance;
            }
        }
        if (probe_bytes == code_size) {
            return -ip;
        }
        return distance_to_code_fp32_nibble(code);
    }

    float distance_to_code_fp32_nibble_pruned(const uint8_t* code, float max_distance) const {
        if (!use_early_abandon || !std::isfinite(max_distance) || suffix_abs_by_byte.empty()) {
            return distance_to_code_fp32_nibble(code);
        }
        const float target_ip = -max_distance;
        const float* lut = lut_fp32.data();
        const float* suffix = suffix_abs_by_byte.data();
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        size_t byte = 0;
        for (; byte + 8 <= code_size; byte += 8) {
            __builtin_prefetch(code + byte + 64, 0, 1);
            const float* row = lut + byte * 32;
            uint8_t c = code[byte + 0];
            s0 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = code[byte + 1];
            s1 += row[32 + (c >> 4)] + row[48 + (c & 0x0f)];
            c = code[byte + 2];
            s2 += row[64 + (c >> 4)] + row[80 + (c & 0x0f)];
            c = code[byte + 3];
            s3 += row[96 + (c >> 4)] + row[112 + (c & 0x0f)];
            c = code[byte + 4];
            s0 += row[128 + (c >> 4)] + row[144 + (c & 0x0f)];
            c = code[byte + 5];
            s1 += row[160 + (c >> 4)] + row[176 + (c & 0x0f)];
            c = code[byte + 6];
            s2 += row[192 + (c >> 4)] + row[208 + (c & 0x0f)];
            c = code[byte + 7];
            s3 += row[224 + (c >> 4)] + row[240 + (c & 0x0f)];
            if (s0 + s1 + s2 + s3 + suffix[byte + 8] < target_ip) {
                return max_distance;
            }
        }
        float ip = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            const uint8_t c = code[byte];
            const float* row = lut + byte * 32;
            ip += row[c >> 4] + row[16 + (c & 0x0f)];
            if (ip + suffix[byte + 1] < target_ip) {
                return max_distance;
            }
        }
        return -ip;
    }

    float distance_to_code_int16_byte_scalar(const uint8_t* code) const {
        const int16_t* lut = lut_i16.data();
        int32_t s0 = 0;
        int32_t s1 = 0;
        int32_t s2 = 0;
        int32_t s3 = 0;
        size_t byte = 0;
        for (; byte + 16 <= code_size; byte += 16) {
            __builtin_prefetch(code + byte + 64, 0, 1);
            const int16_t* row = lut + byte * 256;
            s0 += static_cast<int32_t>(row[code[byte + 0]]);
            s1 += static_cast<int32_t>(row[256 + code[byte + 1]]);
            s2 += static_cast<int32_t>(row[512 + code[byte + 2]]);
            s3 += static_cast<int32_t>(row[768 + code[byte + 3]]);
            s0 += static_cast<int32_t>(row[1024 + code[byte + 4]]);
            s1 += static_cast<int32_t>(row[1280 + code[byte + 5]]);
            s2 += static_cast<int32_t>(row[1536 + code[byte + 6]]);
            s3 += static_cast<int32_t>(row[1792 + code[byte + 7]]);
            s0 += static_cast<int32_t>(row[2048 + code[byte + 8]]);
            s1 += static_cast<int32_t>(row[2304 + code[byte + 9]]);
            s2 += static_cast<int32_t>(row[2560 + code[byte + 10]]);
            s3 += static_cast<int32_t>(row[2816 + code[byte + 11]]);
            s0 += static_cast<int32_t>(row[3072 + code[byte + 12]]);
            s1 += static_cast<int32_t>(row[3328 + code[byte + 13]]);
            s2 += static_cast<int32_t>(row[3584 + code[byte + 14]]);
            s3 += static_cast<int32_t>(row[3840 + code[byte + 15]]);
        }
        int32_t ip = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            ip += static_cast<int32_t>(lut[byte * 256 + code[byte]]);
        }
        return -static_cast<float>(ip);
    }

    static inline int32_t lookup_i16_nibble(const int16_t* row, uint8_t c) {
        return static_cast<int32_t>(row[c >> 4]) +
                static_cast<int32_t>(row[16 + (c & 0x0f)]);
    }

    float distance_to_code_int16_nibble(const uint8_t* code) const {
        const int16_t* lut = lut_i16.data();
        int32_t s0 = 0;
        int32_t s1 = 0;
        int32_t s2 = 0;
        int32_t s3 = 0;
        size_t byte = 0;
        for (; byte + 8 <= code_size; byte += 8) {
            __builtin_prefetch(code + byte + 64, 0, 1);
            const int16_t* row = lut + byte * 32;
            uint8_t c = code[byte + 0];
            s0 += lookup_i16_nibble(row, c);
            c = code[byte + 1];
            s1 += lookup_i16_nibble(row + 32, c);
            c = code[byte + 2];
            s2 += lookup_i16_nibble(row + 64, c);
            c = code[byte + 3];
            s3 += lookup_i16_nibble(row + 96, c);
            c = code[byte + 4];
            s0 += lookup_i16_nibble(row + 128, c);
            c = code[byte + 5];
            s1 += lookup_i16_nibble(row + 160, c);
            c = code[byte + 6];
            s2 += lookup_i16_nibble(row + 192, c);
            c = code[byte + 7];
            s3 += lookup_i16_nibble(row + 224, c);
        }
        int32_t ip = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            const uint8_t c = code[byte];
            const int16_t* row = lut + byte * 32;
            ip += lookup_i16_nibble(row, c);
        }
        return -static_cast<float>(ip);
    }

    float distance_to_code_int8_nibble(const uint8_t* code) const {
        const uint8_t* lut = lut_u8.data();
        int32_t s0 = 0;
        int32_t s1 = 0;
        int32_t s2 = 0;
        int32_t s3 = 0;
        size_t byte = 0;
        for (; byte + 8 <= code_size; byte += 8) {
            __builtin_prefetch(code + byte + 64, 0, 1);
            const uint8_t* row = lut + byte * 32;
            uint8_t c = code[byte + 0];
            s0 += static_cast<int32_t>(row[c >> 4]) + static_cast<int32_t>(row[16 + (c & 0x0f)]);
            c = code[byte + 1];
            s1 += static_cast<int32_t>(row[32 + (c >> 4)]) + static_cast<int32_t>(row[48 + (c & 0x0f)]);
            c = code[byte + 2];
            s2 += static_cast<int32_t>(row[64 + (c >> 4)]) + static_cast<int32_t>(row[80 + (c & 0x0f)]);
            c = code[byte + 3];
            s3 += static_cast<int32_t>(row[96 + (c >> 4)]) + static_cast<int32_t>(row[112 + (c & 0x0f)]);
            c = code[byte + 4];
            s0 += static_cast<int32_t>(row[128 + (c >> 4)]) + static_cast<int32_t>(row[144 + (c & 0x0f)]);
            c = code[byte + 5];
            s1 += static_cast<int32_t>(row[160 + (c >> 4)]) + static_cast<int32_t>(row[176 + (c & 0x0f)]);
            c = code[byte + 6];
            s2 += static_cast<int32_t>(row[192 + (c >> 4)]) + static_cast<int32_t>(row[208 + (c & 0x0f)]);
            c = code[byte + 7];
            s3 += static_cast<int32_t>(row[224 + (c >> 4)]) + static_cast<int32_t>(row[240 + (c & 0x0f)]);
        }
        int32_t ip = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            const uint8_t c = code[byte];
            const uint8_t* row = lut + byte * 32;
            ip += static_cast<int32_t>(row[c >> 4]) + static_cast<int32_t>(row[16 + (c & 0x0f)]);
        }
        return -static_cast<float>(ip);
    }

#ifdef __AVX2__
    float distance_to_code_int16_byte_avx2(const uint8_t* code) const {
        return distance_to_code_int16_byte_scalar(code);
    }
#endif

    float distance_to_code(const uint8_t* code) const {
        if (mode == AdcMode::Fp32Direct) {
            return distance_to_code_fp32_direct(code);
        }
        if (mode == AdcMode::Int8) {
            return distance_to_code_int8_nibble(code);
        }
        if (mode == AdcMode::Int16) {
            if (lut_mode == LutMode::Nibble) return distance_to_code_int16_nibble(code);
#ifdef __AVX2__
            return distance_to_code_int16_byte_avx2(code);
#else
            return distance_to_code_int16_byte_scalar(code);
#endif
        }
        if (lut_mode == LutMode::Nibble) return distance_to_code_fp32_nibble(code);
        return distance_to_code_fp32_byte(code);
    }

    float distance_idx(faiss::idx_t i) const {
        if (has_block32_codes() && mode == AdcMode::Fp32 && lut_mode == LutMode::Nibble) {
            return distance_idx_fp32_nibble_block32(i);
        }
        return distance_to_code(code_at(i));
    }

    float distance_idx_pruned(faiss::idx_t i, float max_distance) const {
        if (i < 0) return std::numeric_limits<float>::max();
        if (has_block32_codes() && mode == AdcMode::Fp32 && lut_mode == LutMode::Nibble) {
            return distance_idx_fp32_nibble_block32(i);
        }
        if (mode == AdcMode::Fp32 && lut_mode == LutMode::Nibble) {
            if (use_ordered_early_abandon) {
                return distance_to_code_fp32_nibble_ordered_pruned(code_at(i), max_distance);
            }
            return distance_to_code_fp32_nibble_pruned(
                    code_at(i),
                    max_distance);
        }
        return distance_idx(i);
    }

    float operator()(faiss::idx_t i) override {
        return distance_idx(i);
    }

    float safe_distance(faiss::idx_t i) {
        if (i < 0) return std::numeric_limits<float>::max();
        return distance_idx(i);
    }

    void batch4_fp32_byte(
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {
        const float* lut = lut_fp32.data();
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const float* row = lut + byte * 256;
            s0 += row[c0[byte]];
            s1 += row[c1[byte]];
            s2 += row[c2[byte]];
            s3 += row[c3[byte]];
        }
        dis0 = -s0;
        dis1 = -s1;
        dis2 = -s2;
        dis3 = -s3;
    }

    void batch4_fp32_nibble(
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {
        const float* lut = lut_fp32.data();
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const float* row = lut + byte * 32;
            uint8_t c = c0[byte];
            s0 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c1[byte];
            s1 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c2[byte];
            s2 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c3[byte];
            s3 += row[c >> 4] + row[16 + (c & 0x0f)];
        }
        dis0 = -s0;
        dis1 = -s1;
        dis2 = -s2;
        dis3 = -s3;
    }

    void batch4_int16_byte(
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {
        const int16_t* lut = lut_i16.data();
        int32_t s0 = 0;
        int32_t s1 = 0;
        int32_t s2 = 0;
        int32_t s3 = 0;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int16_t* row = lut + byte * 256;
            s0 += static_cast<int32_t>(row[c0[byte]]);
            s1 += static_cast<int32_t>(row[c1[byte]]);
            s2 += static_cast<int32_t>(row[c2[byte]]);
            s3 += static_cast<int32_t>(row[c3[byte]]);
        }
        dis0 = -static_cast<float>(s0);
        dis1 = -static_cast<float>(s1);
        dis2 = -static_cast<float>(s2);
        dis3 = -static_cast<float>(s3);
    }

    void batch4_int16_nibble(
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {
        const int16_t* lut = lut_i16.data();
        int32_t s0 = 0;
        int32_t s1 = 0;
        int32_t s2 = 0;
        int32_t s3 = 0;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int16_t* row = lut + byte * 32;
            uint8_t c = c0[byte];
            s0 += lookup_i16_nibble(row, c);
            c = c1[byte];
            s1 += lookup_i16_nibble(row, c);
            c = c2[byte];
            s2 += lookup_i16_nibble(row, c);
            c = c3[byte];
            s3 += lookup_i16_nibble(row, c);
        }
        dis0 = -static_cast<float>(s0);
        dis1 = -static_cast<float>(s1);
        dis2 = -static_cast<float>(s2);
        dis3 = -static_cast<float>(s3);
    }

    void batch4_int8_nibble(
            const uint8_t* c0,
            const uint8_t* c1,
            const uint8_t* c2,
            const uint8_t* c3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) const {
#if defined(__SSSE3__) && defined(__AVX2__)
        const uint8_t* lut = lut_u8.data();
        const __m128i ones = _mm_set1_epi8(1);
        __m128i acc16 = _mm_setzero_si128();
        __m256i acc32 = _mm256_setzero_si256();
        int block_count = 0;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const uint8_t* row = lut + byte * 32;
            const __m128i lut_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
            const __m128i lut_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 16));
            const uint8_t b0 = c0[byte];
            const uint8_t b1 = c1[byte];
            const uint8_t b2 = c2[byte];
            const uint8_t b3 = c3[byte];
            const int hi_idx = static_cast<int>(
                    (b0 >> 4) | ((b1 >> 4) << 8) |
                    ((b2 >> 4) << 16) | ((b3 >> 4) << 24));
            const int lo_idx = static_cast<int>(
                    (b0 & 0x0f) | ((b1 & 0x0f) << 8) |
                    ((b2 & 0x0f) << 16) | ((b3 & 0x0f) << 24));
            const __m128i hi_vals = _mm_shuffle_epi8(lut_hi, _mm_cvtsi32_si128(hi_idx));
            const __m128i lo_vals = _mm_shuffle_epi8(lut_lo, _mm_cvtsi32_si128(lo_idx));
            const __m128i pair_vals = _mm_unpacklo_epi8(hi_vals, lo_vals);
            acc16 = _mm_add_epi16(acc16, _mm_maddubs_epi16(pair_vals, ones));
            if (++block_count == 64) {
                acc32 = _mm256_add_epi32(acc32, _mm256_cvtepu16_epi32(acc16));
                acc16 = _mm_setzero_si128();
                block_count = 0;
            }
        }
        if (block_count > 0) {
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepu16_epi32(acc16));
        }
        alignas(32) int32_t lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc32);
        dis0 = -static_cast<float>(lanes[0]);
        dis1 = -static_cast<float>(lanes[1]);
        dis2 = -static_cast<float>(lanes[2]);
        dis3 = -static_cast<float>(lanes[3]);
#else
        int32_t s0 = 0;
        int32_t s1 = 0;
        int32_t s2 = 0;
        int32_t s3 = 0;
        const uint8_t* lut = lut_u8.data();
        for (size_t byte = 0; byte < code_size; ++byte) {
            const uint8_t* row = lut + byte * 32;
            uint8_t c = c0[byte];
            s0 += static_cast<int32_t>(row[c >> 4]) + static_cast<int32_t>(row[16 + (c & 0x0f)]);
            c = c1[byte];
            s1 += static_cast<int32_t>(row[c >> 4]) + static_cast<int32_t>(row[16 + (c & 0x0f)]);
            c = c2[byte];
            s2 += static_cast<int32_t>(row[c >> 4]) + static_cast<int32_t>(row[16 + (c & 0x0f)]);
            c = c3[byte];
            s3 += static_cast<int32_t>(row[c >> 4]) + static_cast<int32_t>(row[16 + (c & 0x0f)]);
        }
        dis0 = -static_cast<float>(s0);
        dis1 = -static_cast<float>(s1);
        dis2 = -static_cast<float>(s2);
        dis3 = -static_cast<float>(s3);
#endif
    }

    void prefetch_code(const uint8_t* code) const {
        __builtin_prefetch(code, 0, 1);
    }

#if defined(__SSSE3__)
    static __m128i lookup8_u8_nibble_pair_sums(
            const uint8_t* row,
            uint8_t b0,
            uint8_t b1,
            uint8_t b2,
            uint8_t b3,
            uint8_t b4,
            uint8_t b5,
            uint8_t b6,
            uint8_t b7,
            __m128i ones) {
        const __m128i lut_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
        const __m128i lut_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 16));
        const uint64_t hi_idx =
                static_cast<uint64_t>(b0 >> 4) |
                (static_cast<uint64_t>(b1 >> 4) << 8) |
                (static_cast<uint64_t>(b2 >> 4) << 16) |
                (static_cast<uint64_t>(b3 >> 4) << 24) |
                (static_cast<uint64_t>(b4 >> 4) << 32) |
                (static_cast<uint64_t>(b5 >> 4) << 40) |
                (static_cast<uint64_t>(b6 >> 4) << 48) |
                (static_cast<uint64_t>(b7 >> 4) << 56);
        const uint64_t lo_idx =
                static_cast<uint64_t>(b0 & 0x0f) |
                (static_cast<uint64_t>(b1 & 0x0f) << 8) |
                (static_cast<uint64_t>(b2 & 0x0f) << 16) |
                (static_cast<uint64_t>(b3 & 0x0f) << 24) |
                (static_cast<uint64_t>(b4 & 0x0f) << 32) |
                (static_cast<uint64_t>(b5 & 0x0f) << 40) |
                (static_cast<uint64_t>(b6 & 0x0f) << 48) |
                (static_cast<uint64_t>(b7 & 0x0f) << 56);
        const __m128i hi_vals = _mm_shuffle_epi8(
                lut_hi, _mm_cvtsi64_si128(static_cast<long long>(hi_idx)));
        const __m128i lo_vals = _mm_shuffle_epi8(
                lut_lo, _mm_cvtsi64_si128(static_cast<long long>(lo_idx)));
        const __m128i pair_vals = _mm_unpacklo_epi8(hi_vals, lo_vals);
        return _mm_maddubs_epi16(pair_vals, ones);
    }
#endif


#if defined(__SSSE3__)
    static void lookup16_u8_nibble_pair_sums(
            const uint8_t* row,
            uint8_t b0,
            uint8_t b1,
            uint8_t b2,
            uint8_t b3,
            uint8_t b4,
            uint8_t b5,
            uint8_t b6,
            uint8_t b7,
            uint8_t b8,
            uint8_t b9,
            uint8_t b10,
            uint8_t b11,
            uint8_t b12,
            uint8_t b13,
            uint8_t b14,
            uint8_t b15,
            __m128i ones,
            __m128i low_nibble_mask,
            __m128i high_nibble_mask,
            __m128i& sums_lo,
            __m128i& sums_hi) {
        const __m128i lut_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
        const __m128i lut_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 16));
        const __m128i code = _mm_setr_epi8(
                static_cast<char>(b0), static_cast<char>(b1),
                static_cast<char>(b2), static_cast<char>(b3),
                static_cast<char>(b4), static_cast<char>(b5),
                static_cast<char>(b6), static_cast<char>(b7),
                static_cast<char>(b8), static_cast<char>(b9),
                static_cast<char>(b10), static_cast<char>(b11),
                static_cast<char>(b12), static_cast<char>(b13),
                static_cast<char>(b14), static_cast<char>(b15));
        const __m128i hi_idx = _mm_srli_epi16(_mm_and_si128(code, high_nibble_mask), 4);
        const __m128i lo_idx = _mm_and_si128(code, low_nibble_mask);
        const __m128i hi_vals = _mm_shuffle_epi8(lut_hi, hi_idx);
        const __m128i lo_vals = _mm_shuffle_epi8(lut_lo, lo_idx);
        sums_lo = _mm_maddubs_epi16(_mm_unpacklo_epi8(hi_vals, lo_vals), ones);
        sums_hi = _mm_maddubs_epi16(_mm_unpackhi_epi8(hi_vals, lo_vals), ones);
    }
#endif

    void batch8_int8_nibble(
            const faiss::idx_t* ids,
            float* dis_out) const {
#if defined(__SSSE3__) && defined(__SSE4_1__) && defined(__AVX2__)
        const uint8_t* c0 = code_at(ids[0]);
        const uint8_t* c1 = code_at(ids[1]);
        const uint8_t* c2 = code_at(ids[2]);
        const uint8_t* c3 = code_at(ids[3]);
        const uint8_t* c4 = code_at(ids[4]);
        const uint8_t* c5 = code_at(ids[5]);
        const uint8_t* c6 = code_at(ids[6]);
        const uint8_t* c7 = code_at(ids[7]);
        prefetch_code(c0);
        prefetch_code(c1);
        prefetch_code(c2);
        prefetch_code(c3);
        prefetch_code(c4);
        prefetch_code(c5);
        prefetch_code(c6);
        prefetch_code(c7);

        const uint8_t* lut = lut_u8.data();
        const __m128i ones = _mm_set1_epi8(1);
        __m256i acc32 = _mm256_setzero_si256();
        size_t byte = 0;
        for (; byte + 64 <= code_size; byte += 64) {
            __m128i acc16_0 = _mm_setzero_si128();
            __m128i acc16_1 = _mm_setzero_si128();
            __m128i acc16_2 = _mm_setzero_si128();
            __m128i acc16_3 = _mm_setzero_si128();
            for (size_t off = 0; off < 64; off += 4) {
                const size_t b = byte + off;
                const uint8_t* row = lut + b * 32;
                acc16_0 = _mm_add_epi16(acc16_0, lookup8_u8_nibble_pair_sums(
                        row,
                        c0[b + 0], c1[b + 0], c2[b + 0], c3[b + 0],
                        c4[b + 0], c5[b + 0], c6[b + 0], c7[b + 0],
                        ones));
                acc16_1 = _mm_add_epi16(acc16_1, lookup8_u8_nibble_pair_sums(
                        row + 32,
                        c0[b + 1], c1[b + 1], c2[b + 1], c3[b + 1],
                        c4[b + 1], c5[b + 1], c6[b + 1], c7[b + 1],
                        ones));
                acc16_2 = _mm_add_epi16(acc16_2, lookup8_u8_nibble_pair_sums(
                        row + 64,
                        c0[b + 2], c1[b + 2], c2[b + 2], c3[b + 2],
                        c4[b + 2], c5[b + 2], c6[b + 2], c7[b + 2],
                        ones));
                acc16_3 = _mm_add_epi16(acc16_3, lookup8_u8_nibble_pair_sums(
                        row + 96,
                        c0[b + 3], c1[b + 3], c2[b + 3], c3[b + 3],
                        c4[b + 3], c5[b + 3], c6[b + 3], c7[b + 3],
                        ones));
            }
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepu16_epi32(acc16_0));
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepu16_epi32(acc16_1));
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepu16_epi32(acc16_2));
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepu16_epi32(acc16_3));
        }
        __m128i tail16 = _mm_setzero_si128();
        int tail_count = 0;
        for (; byte < code_size; ++byte) {
            tail16 = _mm_add_epi16(tail16, lookup8_u8_nibble_pair_sums(
                    lut + byte * 32,
                    c0[byte], c1[byte], c2[byte], c3[byte],
                    c4[byte], c5[byte], c6[byte], c7[byte],
                    ones));
            if (++tail_count == 64) {
                acc32 = _mm256_add_epi32(acc32, _mm256_cvtepu16_epi32(tail16));
                tail16 = _mm_setzero_si128();
                tail_count = 0;
            }
        }
        if (tail_count > 0) {
            acc32 = _mm256_add_epi32(acc32, _mm256_cvtepu16_epi32(tail16));
        }
        alignas(32) int32_t lanes[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc32);
        dis_out[0] = -static_cast<float>(lanes[0]);
        dis_out[1] = -static_cast<float>(lanes[1]);
        dis_out[2] = -static_cast<float>(lanes[2]);
        dis_out[3] = -static_cast<float>(lanes[3]);
        dis_out[4] = -static_cast<float>(lanes[4]);
        dis_out[5] = -static_cast<float>(lanes[5]);
        dis_out[6] = -static_cast<float>(lanes[6]);
        dis_out[7] = -static_cast<float>(lanes[7]);
#else
        distances_batch_4(ids[0], ids[1], ids[2], ids[3], dis_out[0], dis_out[1], dis_out[2], dis_out[3]);
        distances_batch_4(ids[4], ids[5], ids[6], ids[7], dis_out[4], dis_out[5], dis_out[6], dis_out[7]);
#endif
    }


    void batch16_int8_nibble(
            const faiss::idx_t* ids,
            float* dis_out) const {
#if defined(__SSSE3__) && defined(__SSE4_1__) && defined(__AVX2__)
        const uint8_t* c[16];
        for (int i = 0; i < 16; ++i) {
            c[i] = code_at(ids[i]);
            prefetch_code(c[i]);
        }

        const uint8_t* lut = lut_u8.data();
        const __m128i ones = _mm_set1_epi8(1);
        const __m128i low_nibble_mask = _mm_set1_epi8(0x0f);
        const __m128i high_nibble_mask = _mm_set1_epi8(static_cast<char>(0xf0));
        __m256i acc32_lo = _mm256_setzero_si256();
        __m256i acc32_hi = _mm256_setzero_si256();
        size_t byte = 0;
        for (; byte + 64 <= code_size; byte += 64) {
            __m128i acc16_lo0 = _mm_setzero_si128();
            __m128i acc16_lo1 = _mm_setzero_si128();
            __m128i acc16_lo2 = _mm_setzero_si128();
            __m128i acc16_lo3 = _mm_setzero_si128();
            __m128i acc16_hi0 = _mm_setzero_si128();
            __m128i acc16_hi1 = _mm_setzero_si128();
            __m128i acc16_hi2 = _mm_setzero_si128();
            __m128i acc16_hi3 = _mm_setzero_si128();
            for (size_t off = 0; off < 64; off += 4) {
                const size_t b = byte + off;
                const uint8_t* row = lut + b * 32;
                __m128i lo;
                __m128i hi;
                lookup16_u8_nibble_pair_sums(
                        row,
                        c[0][b + 0], c[1][b + 0], c[2][b + 0], c[3][b + 0],
                        c[4][b + 0], c[5][b + 0], c[6][b + 0], c[7][b + 0],
                        c[8][b + 0], c[9][b + 0], c[10][b + 0], c[11][b + 0],
                        c[12][b + 0], c[13][b + 0], c[14][b + 0], c[15][b + 0],
                        ones,
                        low_nibble_mask,
                        high_nibble_mask,
                        lo,
                        hi);
                acc16_lo0 = _mm_add_epi16(acc16_lo0, lo);
                acc16_hi0 = _mm_add_epi16(acc16_hi0, hi);
                lookup16_u8_nibble_pair_sums(
                        row + 32,
                        c[0][b + 1], c[1][b + 1], c[2][b + 1], c[3][b + 1],
                        c[4][b + 1], c[5][b + 1], c[6][b + 1], c[7][b + 1],
                        c[8][b + 1], c[9][b + 1], c[10][b + 1], c[11][b + 1],
                        c[12][b + 1], c[13][b + 1], c[14][b + 1], c[15][b + 1],
                        ones,
                        low_nibble_mask,
                        high_nibble_mask,
                        lo,
                        hi);
                acc16_lo1 = _mm_add_epi16(acc16_lo1, lo);
                acc16_hi1 = _mm_add_epi16(acc16_hi1, hi);
                lookup16_u8_nibble_pair_sums(
                        row + 64,
                        c[0][b + 2], c[1][b + 2], c[2][b + 2], c[3][b + 2],
                        c[4][b + 2], c[5][b + 2], c[6][b + 2], c[7][b + 2],
                        c[8][b + 2], c[9][b + 2], c[10][b + 2], c[11][b + 2],
                        c[12][b + 2], c[13][b + 2], c[14][b + 2], c[15][b + 2],
                        ones,
                        low_nibble_mask,
                        high_nibble_mask,
                        lo,
                        hi);
                acc16_lo2 = _mm_add_epi16(acc16_lo2, lo);
                acc16_hi2 = _mm_add_epi16(acc16_hi2, hi);
                lookup16_u8_nibble_pair_sums(
                        row + 96,
                        c[0][b + 3], c[1][b + 3], c[2][b + 3], c[3][b + 3],
                        c[4][b + 3], c[5][b + 3], c[6][b + 3], c[7][b + 3],
                        c[8][b + 3], c[9][b + 3], c[10][b + 3], c[11][b + 3],
                        c[12][b + 3], c[13][b + 3], c[14][b + 3], c[15][b + 3],
                        ones,
                        low_nibble_mask,
                        high_nibble_mask,
                        lo,
                        hi);
                acc16_lo3 = _mm_add_epi16(acc16_lo3, lo);
                acc16_hi3 = _mm_add_epi16(acc16_hi3, hi);
            }
            acc32_lo = _mm256_add_epi32(acc32_lo, _mm256_cvtepu16_epi32(acc16_lo0));
            acc32_lo = _mm256_add_epi32(acc32_lo, _mm256_cvtepu16_epi32(acc16_lo1));
            acc32_lo = _mm256_add_epi32(acc32_lo, _mm256_cvtepu16_epi32(acc16_lo2));
            acc32_lo = _mm256_add_epi32(acc32_lo, _mm256_cvtepu16_epi32(acc16_lo3));
            acc32_hi = _mm256_add_epi32(acc32_hi, _mm256_cvtepu16_epi32(acc16_hi0));
            acc32_hi = _mm256_add_epi32(acc32_hi, _mm256_cvtepu16_epi32(acc16_hi1));
            acc32_hi = _mm256_add_epi32(acc32_hi, _mm256_cvtepu16_epi32(acc16_hi2));
            acc32_hi = _mm256_add_epi32(acc32_hi, _mm256_cvtepu16_epi32(acc16_hi3));
        }

        __m128i tail_lo = _mm_setzero_si128();
        __m128i tail_hi = _mm_setzero_si128();
        int tail_count = 0;
        for (; byte < code_size; ++byte) {
            __m128i lo;
            __m128i hi;
            lookup16_u8_nibble_pair_sums(
                    lut + byte * 32,
                    c[0][byte], c[1][byte], c[2][byte], c[3][byte],
                    c[4][byte], c[5][byte], c[6][byte], c[7][byte],
                    c[8][byte], c[9][byte], c[10][byte], c[11][byte],
                    c[12][byte], c[13][byte], c[14][byte], c[15][byte],
                    ones,
                    low_nibble_mask,
                    high_nibble_mask,
                    lo,
                    hi);
            tail_lo = _mm_add_epi16(tail_lo, lo);
            tail_hi = _mm_add_epi16(tail_hi, hi);
            if (++tail_count == 64) {
                acc32_lo = _mm256_add_epi32(acc32_lo, _mm256_cvtepu16_epi32(tail_lo));
                acc32_hi = _mm256_add_epi32(acc32_hi, _mm256_cvtepu16_epi32(tail_hi));
                tail_lo = _mm_setzero_si128();
                tail_hi = _mm_setzero_si128();
                tail_count = 0;
            }
        }
        if (tail_count > 0) {
            acc32_lo = _mm256_add_epi32(acc32_lo, _mm256_cvtepu16_epi32(tail_lo));
            acc32_hi = _mm256_add_epi32(acc32_hi, _mm256_cvtepu16_epi32(tail_hi));
        }

        alignas(32) int32_t lanes_lo[8];
        alignas(32) int32_t lanes_hi[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_lo), acc32_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_hi), acc32_hi);
        for (int i = 0; i < 8; ++i) {
            dis_out[i] = -static_cast<float>(lanes_lo[i]);
            dis_out[i + 8] = -static_cast<float>(lanes_hi[i]);
        }
#else
        batch8_int8_nibble(ids, dis_out);
        batch8_int8_nibble(ids + 8, dis_out + 8);
#endif
    }


#if defined(__AVX2__)
    static void fastscan32_accumulate_u8_chunk_avx2(
            const uint8_t* packed_codes,
            const uint8_t* packed_lut,
            size_t packed_len,
            uint16_t* result) {
        __m256i c, lo, hi, lut, res_lo, res_hi;
        const __m256i low_mask = _mm256_set1_epi8(0x0f);
        __m256i accu0 = _mm256_setzero_si256();
        __m256i accu1 = _mm256_setzero_si256();
        __m256i accu2 = _mm256_setzero_si256();
        __m256i accu3 = _mm256_setzero_si256();

        for (size_t i = 0; i < packed_len; i += 64) {
            c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed_codes + i));
            lut = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed_lut + i));
            lo = _mm256_and_si256(c, low_mask);
            hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
            res_lo = _mm256_shuffle_epi8(lut, lo);
            res_hi = _mm256_shuffle_epi8(lut, hi);
            accu0 = _mm256_add_epi16(accu0, res_lo);
            accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
            accu2 = _mm256_add_epi16(accu2, res_hi);
            accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

            c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed_codes + i + 32));
            lut = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(packed_lut + i + 32));
            lo = _mm256_and_si256(c, low_mask);
            hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);
            res_lo = _mm256_shuffle_epi8(lut, lo);
            res_hi = _mm256_shuffle_epi8(lut, hi);
            accu0 = _mm256_add_epi16(accu0, res_lo);
            accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
            accu2 = _mm256_add_epi16(accu2, res_hi);
            accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
        }

        accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
        const __m256i dis0 = _mm256_add_epi16(
                _mm256_permute2f128_si256(accu0, accu1, 0x21),
                _mm256_blend_epi32(accu0, accu1, 0xF0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result), dis0);

        accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
        const __m256i dis1 = _mm256_add_epi16(
                _mm256_permute2f128_si256(accu2, accu3, 0x21),
                _mm256_blend_epi32(accu2, accu3, 0xF0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result + 16), dis1);
    }
#endif

    void batch32_int8_nibble_fastscan(
            const faiss::idx_t* ids,
            float* dis_out) const {
#if defined(__SSSE3__) && defined(__AVX2__)
        static constexpr size_t kFastScanMaxCodeSize = 1024;
        static constexpr int kPerm0[16] = {
                0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
        static constexpr size_t kChunkCodeBytes = 64;
        if (code_size == 0 || code_size > kFastScanMaxCodeSize || (code_size % kChunkCodeBytes) != 0) {
            batch16_int8_nibble(ids, dis_out);
            batch16_int8_nibble(ids + 16, dis_out + 16);
            return;
        }

        const uint8_t* c[32];
        for (int lane = 0; lane < 32; ++lane) {
            c[lane] = code_at(ids[lane]);
            prefetch_code(c[lane]);
        }

        alignas(32) uint8_t packed[kFastScanMaxCodeSize * 32];
        for (size_t byte = 0; byte < code_size; ++byte) {
            uint8_t* dst = packed + byte * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t hi0 = static_cast<uint8_t>(c[kPerm0[j]][byte] >> 4);
                const uint8_t hi1 = static_cast<uint8_t>(c[kPerm0[j] + 16][byte] >> 4);
                const uint8_t lo0 = static_cast<uint8_t>(c[kPerm0[j]][byte] & 0x0f);
                const uint8_t lo1 = static_cast<uint8_t>(c[kPerm0[j] + 16][byte] & 0x0f);
                dst[j] = static_cast<uint8_t>(hi0 | (hi1 << 4));
                dst[j + 16] = static_cast<uint8_t>(lo0 | (lo1 << 4));
            }
        }

        int32_t acc[32] = {};
        alignas(32) uint16_t partial[32];
        const uint8_t* lut = lut_u8.data();
        for (size_t byte = 0; byte < code_size; byte += kChunkCodeBytes) {
            fastscan32_accumulate_u8_chunk_avx2(
                    packed + byte * 32,
                    lut + byte * 32,
                    kChunkCodeBytes * 32,
                    partial);
            for (int lane = 0; lane < 32; ++lane) {
                acc[lane] += static_cast<int32_t>(partial[lane]);
            }
        }
        for (int lane = 0; lane < 32; ++lane) {
            dis_out[lane] = -static_cast<float>(acc[lane]);
        }
#else
        batch16_int8_nibble(ids, dis_out);
        batch16_int8_nibble(ids + 16, dis_out + 16);
#endif
    }

    void distances_batch_32(const faiss::idx_t* ids, float* dis_out) {
        if (mode == AdcMode::Int8 && lut_mode == LutMode::Nibble) {
            batch32_int8_nibble_fastscan(ids, dis_out);
            return;
        }
        distances_batch_16(ids, dis_out);
        distances_batch_16(ids + 16, dis_out + 16);
    }

    bool batch_all_neighbors() const { return false; }

    void distances_batch_n(
            const faiss::idx_t* ids,
            int count,
            float* distances) {
        for (int i = 0; i < count; ++i) distances[i] = distance_idx(ids[i]);
    }

#if defined(__AVX2__)
    static __m256 lookup16_ps_avx2(
            const float* row,
            __m256i idx,
            __m256i idx7) {
        const __m256 lo = _mm256_loadu_ps(row);
        const __m256 hi = _mm256_loadu_ps(row + 8);
        const __m256i low_idx = _mm256_and_si256(idx, idx7);
        const __m256 lo_vals = _mm256_permutevar8x32_ps(lo, low_idx);
        const __m256 hi_vals = _mm256_permutevar8x32_ps(hi, low_idx);
        const __m256 hi_mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx, idx7));
        return _mm256_blendv_ps(lo_vals, hi_vals, hi_mask);
    }

    static __m256 lookup_byte8_fp32_nibble_avx2(
            const float* row,
            uint8_t b0,
            uint8_t b1,
            uint8_t b2,
            uint8_t b3,
            uint8_t b4,
            uint8_t b5,
            uint8_t b6,
            uint8_t b7,
            __m256i idx7,
            __m256i low_nibble_mask) {
        const __m128i code8 = _mm_setr_epi8(
                static_cast<char>(b0), static_cast<char>(b1),
                static_cast<char>(b2), static_cast<char>(b3),
                static_cast<char>(b4), static_cast<char>(b5),
                static_cast<char>(b6), static_cast<char>(b7),
                0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i code32 = _mm256_cvtepu8_epi32(code8);
        const __m256i hi_idx = _mm256_srli_epi32(code32, 4);
        const __m256i lo_idx = _mm256_and_si256(code32, low_nibble_mask);
        const __m256 hi_vals = lookup16_ps_avx2(row, hi_idx, idx7);
        const __m256 lo_vals = lookup16_ps_avx2(row + 16, lo_idx, idx7);
        return _mm256_add_ps(hi_vals, lo_vals);
    }

    static __m256 lookup_packed_byte8_fp32_nibble_avx2(
            const float* row,
            __m128i code8,
            __m256i idx7,
            __m256i low_nibble_mask) {
        const __m256i code32 = _mm256_cvtepu8_epi32(code8);
        const __m256i hi_idx = _mm256_srli_epi32(code32, 4);
        const __m256i lo_idx = _mm256_and_si256(code32, low_nibble_mask);
        const __m256 hi_vals = lookup16_ps_avx2(row, hi_idx, idx7);
        const __m256 lo_vals = lookup16_ps_avx2(row + 16, lo_idx, idx7);
        return _mm256_add_ps(hi_vals, lo_vals);
    }

    static __m256i lookup16_i32_avx2(
            const int32_t* row,
            __m256i idx,
            __m256i idx7) {
        const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row));
        const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 8));
        const __m256i low_idx = _mm256_and_si256(idx, idx7);
        const __m256i lo_vals = _mm256_permutevar8x32_epi32(lo, low_idx);
        const __m256i hi_vals = _mm256_permutevar8x32_epi32(hi, low_idx);
        const __m256i hi_mask = _mm256_cmpgt_epi32(idx, idx7);
        return _mm256_blendv_epi8(lo_vals, hi_vals, hi_mask);
    }

    static __m256i lookup_byte8_i32_nibble_avx2(
            const int32_t* row,
            uint8_t b0,
            uint8_t b1,
            uint8_t b2,
            uint8_t b3,
            uint8_t b4,
            uint8_t b5,
            uint8_t b6,
            uint8_t b7,
            __m256i idx7,
            __m256i low_nibble_mask) {
        const __m128i code8 = _mm_setr_epi8(
                static_cast<char>(b0), static_cast<char>(b1),
                static_cast<char>(b2), static_cast<char>(b3),
                static_cast<char>(b4), static_cast<char>(b5),
                static_cast<char>(b6), static_cast<char>(b7),
                0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i code32 = _mm256_cvtepu8_epi32(code8);
        const __m256i hi_idx = _mm256_srli_epi32(code32, 4);
        const __m256i lo_idx = _mm256_and_si256(code32, low_nibble_mask);
        const __m256i hi_vals = lookup16_i32_avx2(row, hi_idx, idx7);
        const __m256i lo_vals = lookup16_i32_avx2(row + 16, lo_idx, idx7);
        return _mm256_add_epi32(hi_vals, lo_vals);
    }
#endif

    float distance_idx_fp32_nibble_block32(faiss::idx_t id) const {
        const float* lut = lut_fp32.data();
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        size_t byte = 0;
        for (; byte + 8 <= code_size; byte += 8) {
            const float* row = lut + byte * 32;
            uint8_t c = block32_byte_at(id, byte + 0);
            s0 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = block32_byte_at(id, byte + 1);
            s1 += row[32 + (c >> 4)] + row[48 + (c & 0x0f)];
            c = block32_byte_at(id, byte + 2);
            s2 += row[64 + (c >> 4)] + row[80 + (c & 0x0f)];
            c = block32_byte_at(id, byte + 3);
            s3 += row[96 + (c >> 4)] + row[112 + (c & 0x0f)];
            c = block32_byte_at(id, byte + 4);
            s0 += row[128 + (c >> 4)] + row[144 + (c & 0x0f)];
            c = block32_byte_at(id, byte + 5);
            s1 += row[160 + (c >> 4)] + row[176 + (c & 0x0f)];
            c = block32_byte_at(id, byte + 6);
            s2 += row[192 + (c >> 4)] + row[208 + (c & 0x0f)];
            c = block32_byte_at(id, byte + 7);
            s3 += row[224 + (c >> 4)] + row[240 + (c & 0x0f)];
        }
        float ip = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            const uint8_t c = block32_byte_at(id, byte);
            const float* row = lut + byte * 32;
            ip += row[c >> 4] + row[16 + (c & 0x0f)];
        }
        return -ip;
    }

    void batch8_fp32_nibble_block32(
            const faiss::idx_t* ids,
            float* dis_out) const {
        const float* lut = lut_fp32.data();
#if defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const __m256i idx7 = _mm256_set1_epi32(7);
        const __m256i low_nibble_mask = _mm256_set1_epi32(0x0f);
        size_t byte = 0;
        for (; byte + 4 <= code_size; byte += 4) {
            const float* row = lut + byte * 32;
            acc0 = _mm256_add_ps(acc0, lookup_byte8_fp32_nibble_avx2(
                    row,
                    block32_byte_at(ids[0], byte + 0), block32_byte_at(ids[1], byte + 0),
                    block32_byte_at(ids[2], byte + 0), block32_byte_at(ids[3], byte + 0),
                    block32_byte_at(ids[4], byte + 0), block32_byte_at(ids[5], byte + 0),
                    block32_byte_at(ids[6], byte + 0), block32_byte_at(ids[7], byte + 0),
                    idx7, low_nibble_mask));
            acc1 = _mm256_add_ps(acc1, lookup_byte8_fp32_nibble_avx2(
                    row + 32,
                    block32_byte_at(ids[0], byte + 1), block32_byte_at(ids[1], byte + 1),
                    block32_byte_at(ids[2], byte + 1), block32_byte_at(ids[3], byte + 1),
                    block32_byte_at(ids[4], byte + 1), block32_byte_at(ids[5], byte + 1),
                    block32_byte_at(ids[6], byte + 1), block32_byte_at(ids[7], byte + 1),
                    idx7, low_nibble_mask));
            acc2 = _mm256_add_ps(acc2, lookup_byte8_fp32_nibble_avx2(
                    row + 64,
                    block32_byte_at(ids[0], byte + 2), block32_byte_at(ids[1], byte + 2),
                    block32_byte_at(ids[2], byte + 2), block32_byte_at(ids[3], byte + 2),
                    block32_byte_at(ids[4], byte + 2), block32_byte_at(ids[5], byte + 2),
                    block32_byte_at(ids[6], byte + 2), block32_byte_at(ids[7], byte + 2),
                    idx7, low_nibble_mask));
            acc3 = _mm256_add_ps(acc3, lookup_byte8_fp32_nibble_avx2(
                    row + 96,
                    block32_byte_at(ids[0], byte + 3), block32_byte_at(ids[1], byte + 3),
                    block32_byte_at(ids[2], byte + 3), block32_byte_at(ids[3], byte + 3),
                    block32_byte_at(ids[4], byte + 3), block32_byte_at(ids[5], byte + 3),
                    block32_byte_at(ids[6], byte + 3), block32_byte_at(ids[7], byte + 3),
                    idx7, low_nibble_mask));
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        for (; byte < code_size; ++byte) {
            acc = _mm256_add_ps(acc, lookup_byte8_fp32_nibble_avx2(
                    lut + byte * 32,
                    block32_byte_at(ids[0], byte), block32_byte_at(ids[1], byte),
                    block32_byte_at(ids[2], byte), block32_byte_at(ids[3], byte),
                    block32_byte_at(ids[4], byte), block32_byte_at(ids[5], byte),
                    block32_byte_at(ids[6], byte), block32_byte_at(ids[7], byte),
                    idx7, low_nibble_mask));
        }
        _mm256_storeu_ps(dis_out, _mm256_sub_ps(_mm256_setzero_ps(), acc));
#else
        for (int lane = 0; lane < 8; ++lane) {
            dis_out[lane] = distance_idx_fp32_nibble_block32(ids[lane]);
        }
#endif
    }

    void batch8_fp32_nibble(
            const faiss::idx_t* ids,
            float* dis_out) const {
        const uint8_t* c0 = code_at(ids[0]);
        const uint8_t* c1 = code_at(ids[1]);
        const uint8_t* c2 = code_at(ids[2]);
        const uint8_t* c3 = code_at(ids[3]);
        const uint8_t* c4 = code_at(ids[4]);
        const uint8_t* c5 = code_at(ids[5]);
        const uint8_t* c6 = code_at(ids[6]);
        const uint8_t* c7 = code_at(ids[7]);
        prefetch_code(c0);
        prefetch_code(c1);
        prefetch_code(c2);
        prefetch_code(c3);
        prefetch_code(c4);
        prefetch_code(c5);
        prefetch_code(c6);
        prefetch_code(c7);
        const float* lut = lut_fp32.data();
#if defined(__AVX2__) && defined(ADCBQ_USE_BATCH8_GATHER_BUFFER)
        static constexpr size_t kGatherMaxCodeSize = 1024;
        if (code_size <= kGatherMaxCodeSize) {
            alignas(64) uint8_t gathered[kGatherMaxCodeSize][8];
            for (size_t byte = 0; byte < code_size; ++byte) {
                gathered[byte][0] = c0[byte];
                gathered[byte][1] = c1[byte];
                gathered[byte][2] = c2[byte];
                gathered[byte][3] = c3[byte];
                gathered[byte][4] = c4[byte];
                gathered[byte][5] = c5[byte];
                gathered[byte][6] = c6[byte];
                gathered[byte][7] = c7[byte];
            }
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            const __m256i idx7 = _mm256_set1_epi32(7);
            const __m256i low_nibble_mask = _mm256_set1_epi32(0x0f);
            size_t byte = 0;
            for (; byte + 4 <= code_size; byte += 4) {
                const float* row = lut + byte * 32;
                acc0 = _mm256_add_ps(acc0, lookup_packed_byte8_fp32_nibble_avx2(
                        row,
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gathered[byte + 0])),
                        idx7,
                        low_nibble_mask));
                acc1 = _mm256_add_ps(acc1, lookup_packed_byte8_fp32_nibble_avx2(
                        row + 32,
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gathered[byte + 1])),
                        idx7,
                        low_nibble_mask));
                acc2 = _mm256_add_ps(acc2, lookup_packed_byte8_fp32_nibble_avx2(
                        row + 64,
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gathered[byte + 2])),
                        idx7,
                        low_nibble_mask));
                acc3 = _mm256_add_ps(acc3, lookup_packed_byte8_fp32_nibble_avx2(
                        row + 96,
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gathered[byte + 3])),
                        idx7,
                        low_nibble_mask));
            }
            __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            for (; byte < code_size; ++byte) {
                acc = _mm256_add_ps(acc, lookup_packed_byte8_fp32_nibble_avx2(
                        lut + byte * 32,
                        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(gathered[byte])),
                        idx7,
                        low_nibble_mask));
            }
            _mm256_storeu_ps(dis_out, _mm256_sub_ps(_mm256_setzero_ps(), acc));
            return;
        }
#endif
#if defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const __m256i idx7 = _mm256_set1_epi32(7);
        const __m256i low_nibble_mask = _mm256_set1_epi32(0x0f);
        size_t byte = 0;
        for (; byte + 4 <= code_size; byte += 4) {
            const float* row = lut + byte * 32;
            acc0 = _mm256_add_ps(acc0, lookup_byte8_fp32_nibble_avx2(
                    row,
                    c0[byte + 0], c1[byte + 0], c2[byte + 0], c3[byte + 0],
                    c4[byte + 0], c5[byte + 0], c6[byte + 0], c7[byte + 0],
                    idx7,
                    low_nibble_mask));
            acc1 = _mm256_add_ps(acc1, lookup_byte8_fp32_nibble_avx2(
                    row + 32,
                    c0[byte + 1], c1[byte + 1], c2[byte + 1], c3[byte + 1],
                    c4[byte + 1], c5[byte + 1], c6[byte + 1], c7[byte + 1],
                    idx7,
                    low_nibble_mask));
            acc2 = _mm256_add_ps(acc2, lookup_byte8_fp32_nibble_avx2(
                    row + 64,
                    c0[byte + 2], c1[byte + 2], c2[byte + 2], c3[byte + 2],
                    c4[byte + 2], c5[byte + 2], c6[byte + 2], c7[byte + 2],
                    idx7,
                    low_nibble_mask));
            acc3 = _mm256_add_ps(acc3, lookup_byte8_fp32_nibble_avx2(
                    row + 96,
                    c0[byte + 3], c1[byte + 3], c2[byte + 3], c3[byte + 3],
                    c4[byte + 3], c5[byte + 3], c6[byte + 3], c7[byte + 3],
                    idx7,
                    low_nibble_mask));
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        for (; byte < code_size; ++byte) {
            acc = _mm256_add_ps(acc, lookup_byte8_fp32_nibble_avx2(
                    lut + byte * 32,
                    c0[byte], c1[byte], c2[byte], c3[byte],
                    c4[byte], c5[byte], c6[byte], c7[byte],
                    idx7,
                    low_nibble_mask));
        }
        _mm256_storeu_ps(dis_out, _mm256_sub_ps(_mm256_setzero_ps(), acc));
#else
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        float s4 = 0.0f;
        float s5 = 0.0f;
        float s6 = 0.0f;
        float s7 = 0.0f;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const float* row = lut + byte * 32;
            uint8_t c = c0[byte];
            s0 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c1[byte];
            s1 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c2[byte];
            s2 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c3[byte];
            s3 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c4[byte];
            s4 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c5[byte];
            s5 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c6[byte];
            s6 += row[c >> 4] + row[16 + (c & 0x0f)];
            c = c7[byte];
            s7 += row[c >> 4] + row[16 + (c & 0x0f)];
        }
        dis_out[0] = -s0;
        dis_out[1] = -s1;
        dis_out[2] = -s2;
        dis_out[3] = -s3;
        dis_out[4] = -s4;
        dis_out[5] = -s5;
        dis_out[6] = -s6;
        dis_out[7] = -s7;
#endif
    }


    void batch8_int16_nibble_avx2(
            const faiss::idx_t* ids,
            float* dis_out) const {
        const uint8_t* c0 = code_at(ids[0]);
        const uint8_t* c1 = code_at(ids[1]);
        const uint8_t* c2 = code_at(ids[2]);
        const uint8_t* c3 = code_at(ids[3]);
        const uint8_t* c4 = code_at(ids[4]);
        const uint8_t* c5 = code_at(ids[5]);
        const uint8_t* c6 = code_at(ids[6]);
        const uint8_t* c7 = code_at(ids[7]);
        prefetch_code(c0);
        prefetch_code(c1);
        prefetch_code(c2);
        prefetch_code(c3);
        prefetch_code(c4);
        prefetch_code(c5);
        prefetch_code(c6);
        prefetch_code(c7);
        const int16_t* lut = lut_i16.data();
        int32_t s0 = 0;
        int32_t s1 = 0;
        int32_t s2 = 0;
        int32_t s3 = 0;
        int32_t s4 = 0;
        int32_t s5 = 0;
        int32_t s6 = 0;
        int32_t s7 = 0;
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int16_t* row = lut + byte * 32;
            uint8_t c = c0[byte];
            s0 += lookup_i16_nibble(row, c);
            c = c1[byte];
            s1 += lookup_i16_nibble(row, c);
            c = c2[byte];
            s2 += lookup_i16_nibble(row, c);
            c = c3[byte];
            s3 += lookup_i16_nibble(row, c);
            c = c4[byte];
            s4 += lookup_i16_nibble(row, c);
            c = c5[byte];
            s5 += lookup_i16_nibble(row, c);
            c = c6[byte];
            s6 += lookup_i16_nibble(row, c);
            c = c7[byte];
            s7 += lookup_i16_nibble(row, c);
        }
        dis_out[0] = -static_cast<float>(s0);
        dis_out[1] = -static_cast<float>(s1);
        dis_out[2] = -static_cast<float>(s2);
        dis_out[3] = -static_cast<float>(s3);
        dis_out[4] = -static_cast<float>(s4);
        dis_out[5] = -static_cast<float>(s5);
        dis_out[6] = -static_cast<float>(s6);
        dis_out[7] = -static_cast<float>(s7);
    }

#if defined(__AVX512BW__) && defined(__AVX512VL__)
    static void accumulate_byte16_i16_nibble_avx512(
            const int16_t* row,
            uint8_t b0,
            uint8_t b1,
            uint8_t b2,
            uint8_t b3,
            uint8_t b4,
            uint8_t b5,
            uint8_t b6,
            uint8_t b7,
            uint8_t b8,
            uint8_t b9,
            uint8_t b10,
            uint8_t b11,
            uint8_t b12,
            uint8_t b13,
            uint8_t b14,
            uint8_t b15,
            const __m256i low_nibble_mask,
            __m256i& acc_lo,
            __m256i& acc_hi) {
        const __m128i code8 = _mm_setr_epi8(
                static_cast<char>(b0), static_cast<char>(b1),
                static_cast<char>(b2), static_cast<char>(b3),
                static_cast<char>(b4), static_cast<char>(b5),
                static_cast<char>(b6), static_cast<char>(b7),
                static_cast<char>(b8), static_cast<char>(b9),
                static_cast<char>(b10), static_cast<char>(b11),
                static_cast<char>(b12), static_cast<char>(b13),
                static_cast<char>(b14), static_cast<char>(b15));
        const __m256i code16 = _mm256_cvtepu8_epi16(code8);
        const __m256i hi_idx = _mm256_srli_epi16(code16, 4);
        const __m256i lo_idx = _mm256_and_si256(code16, low_nibble_mask);
        const __m256i lut_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row));
        const __m256i lut_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 16));
        const __m256i hi_vals = _mm256_permutexvar_epi16(hi_idx, lut_hi);
        const __m256i lo_vals = _mm256_permutexvar_epi16(lo_idx, lut_lo);
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi_vals)));
        acc_lo = _mm256_add_epi32(acc_lo, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo_vals)));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi_vals, 1)));
        acc_hi = _mm256_add_epi32(acc_hi, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo_vals, 1)));
    }

    void batch16_int16_nibble_avx512(
            const faiss::idx_t* ids,
            float* dis_out) const {
        const uint8_t* c0 = code_at(ids[0]);
        const uint8_t* c1 = code_at(ids[1]);
        const uint8_t* c2 = code_at(ids[2]);
        const uint8_t* c3 = code_at(ids[3]);
        const uint8_t* c4 = code_at(ids[4]);
        const uint8_t* c5 = code_at(ids[5]);
        const uint8_t* c6 = code_at(ids[6]);
        const uint8_t* c7 = code_at(ids[7]);
        const uint8_t* c8 = code_at(ids[8]);
        const uint8_t* c9 = code_at(ids[9]);
        const uint8_t* c10 = code_at(ids[10]);
        const uint8_t* c11 = code_at(ids[11]);
        const uint8_t* c12 = code_at(ids[12]);
        const uint8_t* c13 = code_at(ids[13]);
        const uint8_t* c14 = code_at(ids[14]);
        const uint8_t* c15 = code_at(ids[15]);
        prefetch_code(c0);
        prefetch_code(c1);
        prefetch_code(c2);
        prefetch_code(c3);
        prefetch_code(c4);
        prefetch_code(c5);
        prefetch_code(c6);
        prefetch_code(c7);
        prefetch_code(c8);
        prefetch_code(c9);
        prefetch_code(c10);
        prefetch_code(c11);
        prefetch_code(c12);
        prefetch_code(c13);
        prefetch_code(c14);
        prefetch_code(c15);
        const int16_t* lut = lut_i16.data();
        __m256i acc_lo0 = _mm256_setzero_si256();
        __m256i acc_hi0 = _mm256_setzero_si256();
        __m256i acc_lo1 = _mm256_setzero_si256();
        __m256i acc_hi1 = _mm256_setzero_si256();
        __m256i acc_lo2 = _mm256_setzero_si256();
        __m256i acc_hi2 = _mm256_setzero_si256();
        __m256i acc_lo3 = _mm256_setzero_si256();
        __m256i acc_hi3 = _mm256_setzero_si256();
        const __m256i low_nibble_mask = _mm256_set1_epi16(0x0f);
        size_t byte = 0;
        for (; byte + 4 <= code_size; byte += 4) {
            const int16_t* row = lut + byte * 32;
            accumulate_byte16_i16_nibble_avx512(
                    row,
                    c0[byte + 0], c1[byte + 0], c2[byte + 0], c3[byte + 0],
                    c4[byte + 0], c5[byte + 0], c6[byte + 0], c7[byte + 0],
                    c8[byte + 0], c9[byte + 0], c10[byte + 0], c11[byte + 0],
                    c12[byte + 0], c13[byte + 0], c14[byte + 0], c15[byte + 0],
                    low_nibble_mask,
                    acc_lo0,
                    acc_hi0);
            accumulate_byte16_i16_nibble_avx512(
                    row + 32,
                    c0[byte + 1], c1[byte + 1], c2[byte + 1], c3[byte + 1],
                    c4[byte + 1], c5[byte + 1], c6[byte + 1], c7[byte + 1],
                    c8[byte + 1], c9[byte + 1], c10[byte + 1], c11[byte + 1],
                    c12[byte + 1], c13[byte + 1], c14[byte + 1], c15[byte + 1],
                    low_nibble_mask,
                    acc_lo1,
                    acc_hi1);
            accumulate_byte16_i16_nibble_avx512(
                    row + 64,
                    c0[byte + 2], c1[byte + 2], c2[byte + 2], c3[byte + 2],
                    c4[byte + 2], c5[byte + 2], c6[byte + 2], c7[byte + 2],
                    c8[byte + 2], c9[byte + 2], c10[byte + 2], c11[byte + 2],
                    c12[byte + 2], c13[byte + 2], c14[byte + 2], c15[byte + 2],
                    low_nibble_mask,
                    acc_lo2,
                    acc_hi2);
            accumulate_byte16_i16_nibble_avx512(
                    row + 96,
                    c0[byte + 3], c1[byte + 3], c2[byte + 3], c3[byte + 3],
                    c4[byte + 3], c5[byte + 3], c6[byte + 3], c7[byte + 3],
                    c8[byte + 3], c9[byte + 3], c10[byte + 3], c11[byte + 3],
                    c12[byte + 3], c13[byte + 3], c14[byte + 3], c15[byte + 3],
                    low_nibble_mask,
                    acc_lo3,
                    acc_hi3);
        }
        __m256i acc_lo = _mm256_add_epi32(
                _mm256_add_epi32(acc_lo0, acc_lo1),
                _mm256_add_epi32(acc_lo2, acc_lo3));
        __m256i acc_hi = _mm256_add_epi32(
                _mm256_add_epi32(acc_hi0, acc_hi1),
                _mm256_add_epi32(acc_hi2, acc_hi3));
        for (; byte < code_size; ++byte) {
            accumulate_byte16_i16_nibble_avx512(
                    lut + byte * 32,
                    c0[byte], c1[byte], c2[byte], c3[byte],
                    c4[byte], c5[byte], c6[byte], c7[byte],
                    c8[byte], c9[byte], c10[byte], c11[byte],
                    c12[byte], c13[byte], c14[byte], c15[byte],
                    low_nibble_mask,
                    acc_lo,
                    acc_hi);
        }
        alignas(32) int32_t lanes_lo[8];
        alignas(32) int32_t lanes_hi[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_lo), acc_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes_hi), acc_hi);
        for (int i = 0; i < 8; ++i) {
            dis_out[i] = -static_cast<float>(lanes_lo[i]);
            dis_out[i + 8] = -static_cast<float>(lanes_hi[i]);
        }
    }
#endif

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(ADCBQ_USE_AVX512_BATCH16)
    static __m512 lookup_byte16_fp32_nibble_avx512(
            const float* row,
            uint8_t b0,
            uint8_t b1,
            uint8_t b2,
            uint8_t b3,
            uint8_t b4,
            uint8_t b5,
            uint8_t b6,
            uint8_t b7,
            uint8_t b8,
            uint8_t b9,
            uint8_t b10,
            uint8_t b11,
            uint8_t b12,
            uint8_t b13,
            uint8_t b14,
            uint8_t b15,
            __m512i low_nibble_mask) {
        const __m128i code8 = _mm_setr_epi8(
                static_cast<char>(b0), static_cast<char>(b1),
                static_cast<char>(b2), static_cast<char>(b3),
                static_cast<char>(b4), static_cast<char>(b5),
                static_cast<char>(b6), static_cast<char>(b7),
                static_cast<char>(b8), static_cast<char>(b9),
                static_cast<char>(b10), static_cast<char>(b11),
                static_cast<char>(b12), static_cast<char>(b13),
                static_cast<char>(b14), static_cast<char>(b15));
        const __m512i code32 = _mm512_cvtepu8_epi32(code8);
        const __m512i hi_idx = _mm512_srli_epi32(code32, 4);
        const __m512i lo_idx = _mm512_and_si512(code32, low_nibble_mask);
        const __m512 lut_hi = _mm512_loadu_ps(row);
        const __m512 lut_lo = _mm512_loadu_ps(row + 16);
        return _mm512_add_ps(
                _mm512_permutexvar_ps(hi_idx, lut_hi),
                _mm512_permutexvar_ps(lo_idx, lut_lo));
    }

    void batch16_fp32_nibble_avx512(
            const faiss::idx_t* ids,
            float* dis_out) const {
        const uint8_t* c0 = code_at(ids[0]);
        const uint8_t* c1 = code_at(ids[1]);
        const uint8_t* c2 = code_at(ids[2]);
        const uint8_t* c3 = code_at(ids[3]);
        const uint8_t* c4 = code_at(ids[4]);
        const uint8_t* c5 = code_at(ids[5]);
        const uint8_t* c6 = code_at(ids[6]);
        const uint8_t* c7 = code_at(ids[7]);
        const uint8_t* c8 = code_at(ids[8]);
        const uint8_t* c9 = code_at(ids[9]);
        const uint8_t* c10 = code_at(ids[10]);
        const uint8_t* c11 = code_at(ids[11]);
        const uint8_t* c12 = code_at(ids[12]);
        const uint8_t* c13 = code_at(ids[13]);
        const uint8_t* c14 = code_at(ids[14]);
        const uint8_t* c15 = code_at(ids[15]);
        prefetch_code(c0);
        prefetch_code(c1);
        prefetch_code(c2);
        prefetch_code(c3);
        prefetch_code(c4);
        prefetch_code(c5);
        prefetch_code(c6);
        prefetch_code(c7);
        prefetch_code(c8);
        prefetch_code(c9);
        prefetch_code(c10);
        prefetch_code(c11);
        prefetch_code(c12);
        prefetch_code(c13);
        prefetch_code(c14);
        prefetch_code(c15);

        const float* lut = lut_fp32.data();
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();
        const __m512i low_nibble_mask = _mm512_set1_epi32(0x0f);
        size_t byte = 0;
        for (; byte + 4 <= code_size; byte += 4) {
            const float* row = lut + byte * 32;
            acc0 = _mm512_add_ps(acc0, lookup_byte16_fp32_nibble_avx512(
                    row,
                    c0[byte + 0], c1[byte + 0], c2[byte + 0], c3[byte + 0],
                    c4[byte + 0], c5[byte + 0], c6[byte + 0], c7[byte + 0],
                    c8[byte + 0], c9[byte + 0], c10[byte + 0], c11[byte + 0],
                    c12[byte + 0], c13[byte + 0], c14[byte + 0], c15[byte + 0],
                    low_nibble_mask));
            acc1 = _mm512_add_ps(acc1, lookup_byte16_fp32_nibble_avx512(
                    row + 32,
                    c0[byte + 1], c1[byte + 1], c2[byte + 1], c3[byte + 1],
                    c4[byte + 1], c5[byte + 1], c6[byte + 1], c7[byte + 1],
                    c8[byte + 1], c9[byte + 1], c10[byte + 1], c11[byte + 1],
                    c12[byte + 1], c13[byte + 1], c14[byte + 1], c15[byte + 1],
                    low_nibble_mask));
            acc2 = _mm512_add_ps(acc2, lookup_byte16_fp32_nibble_avx512(
                    row + 64,
                    c0[byte + 2], c1[byte + 2], c2[byte + 2], c3[byte + 2],
                    c4[byte + 2], c5[byte + 2], c6[byte + 2], c7[byte + 2],
                    c8[byte + 2], c9[byte + 2], c10[byte + 2], c11[byte + 2],
                    c12[byte + 2], c13[byte + 2], c14[byte + 2], c15[byte + 2],
                    low_nibble_mask));
            acc3 = _mm512_add_ps(acc3, lookup_byte16_fp32_nibble_avx512(
                    row + 96,
                    c0[byte + 3], c1[byte + 3], c2[byte + 3], c3[byte + 3],
                    c4[byte + 3], c5[byte + 3], c6[byte + 3], c7[byte + 3],
                    c8[byte + 3], c9[byte + 3], c10[byte + 3], c11[byte + 3],
                    c12[byte + 3], c13[byte + 3], c14[byte + 3], c15[byte + 3],
                    low_nibble_mask));
        }
        __m512 acc = _mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3));
        for (; byte < code_size; ++byte) {
            acc = _mm512_add_ps(acc, lookup_byte16_fp32_nibble_avx512(
                    lut + byte * 32,
                    c0[byte], c1[byte], c2[byte], c3[byte],
                    c4[byte], c5[byte], c6[byte], c7[byte],
                    c8[byte], c9[byte], c10[byte], c11[byte],
                    c12[byte], c13[byte], c14[byte], c15[byte],
                    low_nibble_mask));
        }
        _mm512_storeu_ps(dis_out, _mm512_sub_ps(_mm512_setzero_ps(), acc));
    }

#endif

#if defined(__AVX512F__) && defined(__SSE2__)
    static inline __mmask16 direct_msb_mask16(__m128i v) {
#if defined(__AVX512BW__)
        return _mm_movepi8_mask(v);
#else
        return static_cast<__mmask16>(_mm_movemask_epi8(v));
#endif
    }

    void batch16_fp32_direct_avx512(
            const faiss::idx_t* ids,
            float* dis_out) const {
        const uint8_t* c0 = code_at(ids[0]);
        const uint8_t* c1 = code_at(ids[1]);
        const uint8_t* c2 = code_at(ids[2]);
        const uint8_t* c3 = code_at(ids[3]);
        const uint8_t* c4 = code_at(ids[4]);
        const uint8_t* c5 = code_at(ids[5]);
        const uint8_t* c6 = code_at(ids[6]);
        const uint8_t* c7 = code_at(ids[7]);
        const uint8_t* c8 = code_at(ids[8]);
        const uint8_t* c9 = code_at(ids[9]);
        const uint8_t* c10 = code_at(ids[10]);
        const uint8_t* c11 = code_at(ids[11]);
        const uint8_t* c12 = code_at(ids[12]);
        const uint8_t* c13 = code_at(ids[13]);
        const uint8_t* c14 = code_at(ids[14]);
        const uint8_t* c15 = code_at(ids[15]);
        prefetch_code(c0);
        prefetch_code(c1);
        prefetch_code(c2);
        prefetch_code(c3);
        prefetch_code(c4);
        prefetch_code(c5);
        prefetch_code(c6);
        prefetch_code(c7);
        prefetch_code(c8);
        prefetch_code(c9);
        prefetch_code(c10);
        prefetch_code(c11);
        prefetch_code(c12);
        prefetch_code(c13);
        prefetch_code(c14);
        prefetch_code(c15);

        const float* q = query_direct;
        __m512 selected0 = _mm512_setzero_ps();
        __m512 selected1 = _mm512_setzero_ps();
        __m512 selected2 = _mm512_setzero_ps();
        __m512 selected3 = _mm512_setzero_ps();
        __m512 selected4 = _mm512_setzero_ps();
        __m512 selected5 = _mm512_setzero_ps();
        __m512 selected6 = _mm512_setzero_ps();
        __m512 selected7 = _mm512_setzero_ps();
        for (size_t byte = 0; byte < code_size; ++byte) {
            const __m128i codev = _mm_setr_epi8(
                    static_cast<char>(c0[byte]), static_cast<char>(c1[byte]),
                    static_cast<char>(c2[byte]), static_cast<char>(c3[byte]),
                    static_cast<char>(c4[byte]), static_cast<char>(c5[byte]),
                    static_cast<char>(c6[byte]), static_cast<char>(c7[byte]),
                    static_cast<char>(c8[byte]), static_cast<char>(c9[byte]),
                    static_cast<char>(c10[byte]), static_cast<char>(c11[byte]),
                    static_cast<char>(c12[byte]), static_cast<char>(c13[byte]),
                    static_cast<char>(c14[byte]), static_cast<char>(c15[byte]));
            const int base = static_cast<int>(byte * 8);
#define ADCBQ_DIRECT_ADD_SHIFT(acc, bitpos, shift) \
            do { \
                const __m128i shifted = (shift) == 0 ? codev : _mm_slli_epi16(codev, (shift)); \
                const __mmask16 m = direct_msb_mask16(shifted); \
                acc = _mm512_mask_add_ps(acc, m, acc, _mm512_set1_ps(q[base + (bitpos)])); \
            } while (0)
            ADCBQ_DIRECT_ADD_SHIFT(selected0, 0, 0);
            ADCBQ_DIRECT_ADD_SHIFT(selected1, 1, 1);
            ADCBQ_DIRECT_ADD_SHIFT(selected2, 2, 2);
            ADCBQ_DIRECT_ADD_SHIFT(selected3, 3, 3);
            ADCBQ_DIRECT_ADD_SHIFT(selected4, 4, 4);
            ADCBQ_DIRECT_ADD_SHIFT(selected5, 5, 5);
            ADCBQ_DIRECT_ADD_SHIFT(selected6, 6, 6);
            ADCBQ_DIRECT_ADD_SHIFT(selected7, 7, 7);
#undef ADCBQ_DIRECT_ADD_SHIFT
        }
        const __m512 selected = _mm512_add_ps(
                _mm512_add_ps(_mm512_add_ps(selected0, selected1), _mm512_add_ps(selected2, selected3)),
                _mm512_add_ps(_mm512_add_ps(selected4, selected5), _mm512_add_ps(selected6, selected7)));
        const __m512 dist = _mm512_sub_ps(
                _mm512_set1_ps(query_direct_sum),
                _mm512_add_ps(selected, selected));
        _mm512_storeu_ps(dis_out, dist);
    }

    void batch8_fp32_direct_avx512(
            const faiss::idx_t* ids,
            float* dis_out) const {
        const uint8_t* c0 = code_at(ids[0]);
        const uint8_t* c1 = code_at(ids[1]);
        const uint8_t* c2 = code_at(ids[2]);
        const uint8_t* c3 = code_at(ids[3]);
        const uint8_t* c4 = code_at(ids[4]);
        const uint8_t* c5 = code_at(ids[5]);
        const uint8_t* c6 = code_at(ids[6]);
        const uint8_t* c7 = code_at(ids[7]);
        prefetch_code(c0);
        prefetch_code(c1);
        prefetch_code(c2);
        prefetch_code(c3);
        prefetch_code(c4);
        prefetch_code(c5);
        prefetch_code(c6);
        prefetch_code(c7);

        const float* q = query_direct;
        __m512 selected0 = _mm512_setzero_ps();
        __m512 selected1 = _mm512_setzero_ps();
        __m512 selected2 = _mm512_setzero_ps();
        __m512 selected3 = _mm512_setzero_ps();
        __m512 selected4 = _mm512_setzero_ps();
        __m512 selected5 = _mm512_setzero_ps();
        __m512 selected6 = _mm512_setzero_ps();
        __m512 selected7 = _mm512_setzero_ps();
        for (size_t byte = 0; byte < code_size; ++byte) {
            const __m128i codev = _mm_setr_epi8(
                    static_cast<char>(c0[byte]), static_cast<char>(c1[byte]),
                    static_cast<char>(c2[byte]), static_cast<char>(c3[byte]),
                    static_cast<char>(c4[byte]), static_cast<char>(c5[byte]),
                    static_cast<char>(c6[byte]), static_cast<char>(c7[byte]),
                    0, 0, 0, 0, 0, 0, 0, 0);
            const int base = static_cast<int>(byte * 8);
#define ADCBQ_DIRECT_ADD_SHIFT(acc, bitpos, shift) \
            do { \
                const __m128i shifted = (shift) == 0 ? codev : _mm_slli_epi16(codev, (shift)); \
                const __mmask16 m = static_cast<__mmask16>(direct_msb_mask16(shifted) & 0xff); \
                acc = _mm512_mask_add_ps(acc, m, acc, _mm512_set1_ps(q[base + (bitpos)])); \
            } while (0)
            ADCBQ_DIRECT_ADD_SHIFT(selected0, 0, 0);
            ADCBQ_DIRECT_ADD_SHIFT(selected1, 1, 1);
            ADCBQ_DIRECT_ADD_SHIFT(selected2, 2, 2);
            ADCBQ_DIRECT_ADD_SHIFT(selected3, 3, 3);
            ADCBQ_DIRECT_ADD_SHIFT(selected4, 4, 4);
            ADCBQ_DIRECT_ADD_SHIFT(selected5, 5, 5);
            ADCBQ_DIRECT_ADD_SHIFT(selected6, 6, 6);
            ADCBQ_DIRECT_ADD_SHIFT(selected7, 7, 7);
#undef ADCBQ_DIRECT_ADD_SHIFT
        }
        const __m512 selected = _mm512_add_ps(
                _mm512_add_ps(_mm512_add_ps(selected0, selected1), _mm512_add_ps(selected2, selected3)),
                _mm512_add_ps(_mm512_add_ps(selected4, selected5), _mm512_add_ps(selected6, selected7)));
        const __m512 dist = _mm512_sub_ps(
                _mm512_set1_ps(query_direct_sum),
                _mm512_add_ps(selected, selected));
        alignas(64) float lanes[16];
        _mm512_store_ps(lanes, dist);
        for (int i = 0; i < 8; ++i) dis_out[i] = lanes[i];
    }
#endif

    void distances_batch_16(const faiss::idx_t* ids, float* dis_out) {
        if (mode == AdcMode::Fp32Direct) {
#if !defined(ADCBQ_FORCE_FP32_DIRECT_SCALAR) && defined(__AVX512F__) && defined(__SSE2__)
            batch16_fp32_direct_avx512(ids, dis_out);
#else
            for (int i = 0; i < 16; ++i) {
                dis_out[i] = distance_to_code_fp32_direct(code_at(ids[i]));
            }
#endif
            return;
        }
        if (mode == AdcMode::Int8 && lut_mode == LutMode::Nibble) {
            batch16_int8_nibble(ids, dis_out);
            return;
        }
        if (mode == AdcMode::Fp32 && lut_mode == LutMode::Nibble) {
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(ADCBQ_USE_AVX512_BATCH16)
            if (has_block32_codes()) {
                batch8_fp32_nibble_block32(ids, dis_out);
                batch8_fp32_nibble_block32(ids + 8, dis_out + 8);
            } else {
                batch16_fp32_nibble_avx512(ids, dis_out);
            }
#else
            if (has_block32_codes()) {
                batch8_fp32_nibble_block32(ids, dis_out);
                batch8_fp32_nibble_block32(ids + 8, dis_out + 8);
            } else {
                batch8_fp32_nibble(ids, dis_out);
                batch8_fp32_nibble(ids + 8, dis_out + 8);
            }
#endif
            return;
        }
        if (mode == AdcMode::Int16 && lut_mode == LutMode::Nibble) {
#if defined(__AVX512BW__) && defined(__AVX512VL__)
            batch16_int16_nibble_avx512(ids, dis_out);
#else
            batch8_int16_nibble_avx2(ids, dis_out);
            batch8_int16_nibble_avx2(ids + 8, dis_out + 8);
#endif
            return;
        }
        distances_batch_8(ids, dis_out);
        distances_batch_8(ids + 8, dis_out + 8);
    }

    void distances_batch_8(const faiss::idx_t* ids, float* dis_out) {
        if (mode == AdcMode::Fp32Direct) {
#if !defined(ADCBQ_FORCE_FP32_DIRECT_SCALAR) && defined(__AVX512F__) && defined(__SSE2__)
            batch8_fp32_direct_avx512(ids, dis_out);
#else
            for (int i = 0; i < 8; ++i) {
                dis_out[i] = distance_to_code_fp32_direct(code_at(ids[i]));
            }
#endif
            return;
        }
        if (mode == AdcMode::Fp32 && lut_mode == LutMode::Nibble) {
            if (has_block32_codes()) batch8_fp32_nibble_block32(ids, dis_out);
            else batch8_fp32_nibble(ids, dis_out);
            return;
        }
        if (mode == AdcMode::Int8 && lut_mode == LutMode::Nibble) {
            batch8_int8_nibble(ids, dis_out);
            return;
        }
        if (mode == AdcMode::Int16 && lut_mode == LutMode::Nibble) {
            batch8_int16_nibble_avx2(ids, dis_out);
            return;
        }
        distances_batch_4(ids[0], ids[1], ids[2], ids[3], dis_out[0], dis_out[1], dis_out[2], dis_out[3]);
        distances_batch_4(ids[4], ids[5], ids[6], ids[7], dis_out[4], dis_out[5], dis_out[6], dis_out[7]);
    }

    void prefetch_idx(faiss::idx_t i) const {
        if (i >= 0) {
            prefetch_code(code_at(i));
        }
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
        if (idx0 < 0 || idx1 < 0 || idx2 < 0 || idx3 < 0) {
            dis0 = safe_distance(idx0);
            dis1 = safe_distance(idx1);
            dis2 = safe_distance(idx2);
            dis3 = safe_distance(idx3);
            return;
        }
        const uint8_t* c0 = code_at(idx0);
        const uint8_t* c1 = code_at(idx1);
        const uint8_t* c2 = code_at(idx2);
        const uint8_t* c3 = code_at(idx3);
        prefetch_code(c0);
        prefetch_code(c1);
        prefetch_code(c2);
        prefetch_code(c3);
        if (mode == AdcMode::Fp32Direct) {
            dis0 = distance_to_code_fp32_direct(c0);
            dis1 = distance_to_code_fp32_direct(c1);
            dis2 = distance_to_code_fp32_direct(c2);
            dis3 = distance_to_code_fp32_direct(c3);
        } else if (mode == AdcMode::Int8) {
            batch4_int8_nibble(c0, c1, c2, c3, dis0, dis1, dis2, dis3);
        } else if (mode == AdcMode::Int16) {
            if (lut_mode == LutMode::Nibble) {
                batch4_int16_nibble(c0, c1, c2, c3, dis0, dis1, dis2, dis3);
            } else {
                batch4_int16_byte(c0, c1, c2, c3, dis0, dis1, dis2, dis3);
            }
        } else {
            if (lut_mode == LutMode::Nibble) {
                batch4_fp32_nibble(c0, c1, c2, c3, dis0, dis1, dis2, dis3);
            } else {
                batch4_fp32_byte(c0, c1, c2, c3, dis0, dis1, dis2, dis3);
            }
        }
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint8_t* a = code_at(i);
        const uint8_t* b = code_at(j);
        int h = 0;
        for (size_t c = 0; c < code_size; ++c) {
            h += __builtin_popcount(static_cast<unsigned>(a[c] ^ b[c]));
        }
        return static_cast<float>(h);
    }
};


struct PackedXFlagBQDistanceComputer : faiss::DistanceComputer {
    const uint8_t* sign_codes = nullptr;
    const uint8_t* xflag_codes = nullptr;
    size_t code_size = 0;
    size_t sign_stride = 0;
    size_t xflag_stride = 0;
    int dim = 0;
    float threshold = 0.0f;
    std::vector<uint8_t> query_sign;
    std::vector<uint8_t> query_xflag;

    PackedXFlagBQDistanceComputer(
            const uint8_t* sign_codes_in,
            const uint8_t* xflag_codes_in,
            size_t code_size_in,
            size_t sign_stride_in,
            size_t xflag_stride_in,
            int dim_in,
            float threshold_in)
            : sign_codes(sign_codes_in),
              xflag_codes(xflag_codes_in),
              code_size(code_size_in),
              sign_stride(sign_stride_in == 0 ? code_size_in : sign_stride_in),
              xflag_stride(xflag_stride_in == 0 ? code_size_in : xflag_stride_in),
              dim(dim_in),
              threshold(threshold_in),
              query_sign(code_size_in, 0),
              query_xflag(code_size_in, 0) {}

    const uint8_t* sign_at(faiss::idx_t i) const {
        return sign_codes + static_cast<size_t>(i) * sign_stride;
    }

    const uint8_t* xflag_at(faiss::idx_t i) const {
        return xflag_codes + static_cast<size_t>(i) * xflag_stride;
    }

    void set_query(const float* q) override {
        std::fill(query_sign.begin(), query_sign.end(), 0);
        std::fill(query_xflag.begin(), query_xflag.end(), 0);
        for (int j = 0; j < dim; ++j) {
            const uint8_t mask = static_cast<uint8_t>(1u << (7 - (j % 8)));
            if (q[j] > 0.0f) {
                query_sign[static_cast<size_t>(j / 8)] |= mask;
            }
            if (std::abs(q[j]) < threshold) {
                query_xflag[static_cast<size_t>(j / 8)] |= mask;
            }
        }
    }

    static uint64_t load_u64(const uint8_t* p) {
        uint64_t v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }

    float distance_to_codes(const uint8_t* sign, const uint8_t* xflag) const {
        uint32_t kept_mismatch = 0;
        uint32_t weak_neutral = 0;
        size_t byte = 0;
        const uint8_t* qsign = query_sign.data();
        const uint8_t* qxflag = query_xflag.data();
        for (; byte + 8 <= code_size; byte += 8) {
            const uint64_t mismatch = load_u64(qsign + byte) ^ load_u64(sign + byte);
            const uint64_t neutral = load_u64(qxflag + byte) & load_u64(xflag + byte);
            kept_mismatch += static_cast<uint32_t>(__builtin_popcountll(mismatch & ~neutral));
            weak_neutral += static_cast<uint32_t>(__builtin_popcountll(neutral));
        }
        for (; byte < code_size; ++byte) {
            const uint8_t mismatch = static_cast<uint8_t>(qsign[byte] ^ sign[byte]);
            const uint8_t neutral = static_cast<uint8_t>(qxflag[byte] & xflag[byte]);
            kept_mismatch += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(mismatch & ~neutral)));
            weak_neutral += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(neutral)));
        }
        return static_cast<float>(2u * kept_mismatch + weak_neutral);
    }

    float distance_idx(faiss::idx_t i) const {
        if (i < 0) return std::numeric_limits<float>::max();
        return distance_to_codes(sign_at(i), xflag_at(i));
    }

    float distance_idx_pruned(faiss::idx_t i, float) const {
        return distance_idx(i);
    }

    float operator()(faiss::idx_t i) override {
        return distance_idx(i);
    }

    float safe_distance(faiss::idx_t i) const {
        return distance_idx(i);
    }

    void prefetch_idx(faiss::idx_t i) const {
        if (i >= 0) {
            __builtin_prefetch(sign_at(i), 0, 1);
            __builtin_prefetch(xflag_at(i), 0, 1);
        }
    }

    void distances_batch_16(const faiss::idx_t* ids, float* dis_out) {
        for (int i = 0; i < 16; ++i) {
            dis_out[i] = distance_idx(ids[i]);
        }
    }

    void distances_batch_32(const faiss::idx_t* ids, float* dis_out) {
        distances_batch_16(ids, dis_out);
        distances_batch_16(ids + 16, dis_out + 16);
    }

    bool batch_all_neighbors() const { return false; }

    void distances_batch_n(
            const faiss::idx_t* ids,
            int count,
            float* distances) {
        for (int i = 0; i < count; ++i) distances[i] = distance_idx(ids[i]);
    }

    void distances_batch_8(const faiss::idx_t* ids, float* dis_out) {
        for (int i = 0; i < 8; ++i) {
            dis_out[i] = distance_idx(ids[i]);
        }
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
        dis0 = safe_distance(idx0);
        dis1 = safe_distance(idx1);
        dis2 = safe_distance(idx2);
        dis3 = safe_distance(idx3);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint8_t* a = sign_at(i);
        const uint8_t* ax = xflag_at(i);
        const uint8_t* b = sign_at(j);
        const uint8_t* bx = xflag_at(j);
        uint32_t kept_mismatch = 0;
        uint32_t weak_neutral = 0;
        size_t byte = 0;
        for (; byte + 8 <= code_size; byte += 8) {
            const uint64_t mismatch = load_u64(a + byte) ^ load_u64(b + byte);
            const uint64_t neutral = load_u64(ax + byte) & load_u64(bx + byte);
            kept_mismatch += static_cast<uint32_t>(__builtin_popcountll(mismatch & ~neutral));
            weak_neutral += static_cast<uint32_t>(__builtin_popcountll(neutral));
        }
        for (; byte < code_size; ++byte) {
            const uint8_t mismatch = static_cast<uint8_t>(a[byte] ^ b[byte]);
            const uint8_t neutral = static_cast<uint8_t>(ax[byte] & bx[byte]);
            kept_mismatch += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(mismatch & ~neutral)));
            weak_neutral += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(neutral)));
        }
        return static_cast<float>(2u * kept_mismatch + weak_neutral);
    }
};


struct PackedAsymXFlagBQDistanceComputer : faiss::DistanceComputer {
    const uint8_t* sign_codes = nullptr;
    const uint8_t* xflag_codes = nullptr;
    size_t code_size = 0;
    size_t sign_stride = 0;
    size_t xflag_stride = 0;
    int dim = 0;
    std::vector<float> subset_lut;

    PackedAsymXFlagBQDistanceComputer(
            const uint8_t* sign_codes_in,
            const uint8_t* xflag_codes_in,
            size_t code_size_in,
            size_t sign_stride_in,
            size_t xflag_stride_in,
            int dim_in)
            : sign_codes(sign_codes_in),
              xflag_codes(xflag_codes_in),
              code_size(code_size_in),
              sign_stride(sign_stride_in == 0 ? code_size_in : sign_stride_in),
              xflag_stride(xflag_stride_in == 0 ? code_size_in : xflag_stride_in),
              dim(dim_in),
              subset_lut(code_size_in * 32, 0.0f) {}

    const uint8_t* sign_at(faiss::idx_t i) const {
        return sign_codes + static_cast<size_t>(i) * sign_stride;
    }

    const uint8_t* xflag_at(faiss::idx_t i) const {
        return xflag_codes + static_cast<size_t>(i) * xflag_stride;
    }

    static void build_subset_nibble_row(float* row, const float* q) {
        row[0] = 0.0f;
        row[1] = q[3];
        row[2] = q[2];
        row[3] = q[2] + q[3];
        row[4] = q[1];
        row[5] = q[1] + q[3];
        row[6] = q[1] + q[2];
        row[7] = q[1] + q[2] + q[3];
        row[8] = q[0];
        row[9] = q[0] + q[3];
        row[10] = q[0] + q[2];
        row[11] = q[0] + q[2] + q[3];
        row[12] = q[0] + q[1];
        row[13] = q[0] + q[1] + q[3];
        row[14] = q[0] + q[1] + q[2];
        row[15] = q[0] + q[1] + q[2] + q[3];
    }

    void set_query(const float* q) override {
        for (size_t byte = 0; byte < code_size; ++byte) {
            const int base = static_cast<int>(byte * 8);
            float* row = subset_lut.data() + byte * 32;
            build_subset_nibble_row(row, q + base);
            build_subset_nibble_row(row + 16, q + base + 4);
        }
    }

    static float adc_xflag_nibble_score(const float* row, uint8_t sign, uint8_t flag) {
        const uint8_t strong_or_weak_sign = sign & 0x0f;
        const uint8_t weak = flag & 0x0f;
        const uint8_t weak_sign = strong_or_weak_sign & weak;
        return 2.0f * row[strong_or_weak_sign] - row[15]
                - row[weak_sign] + 0.5f * row[weak];
    }

    float distance_to_codes(const uint8_t* sign, const uint8_t* xflag) const {
        const float* lut = subset_lut.data();
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        size_t byte = 0;
        for (; byte + 4 <= code_size; byte += 4) {
            __builtin_prefetch(sign + byte + 64, 0, 1);
            __builtin_prefetch(xflag + byte + 64, 0, 1);
            const float* row = lut + byte * 32;
            uint8_t sb = sign[byte + 0];
            uint8_t xb = xflag[byte + 0];
            s0 += adc_xflag_nibble_score(row, static_cast<uint8_t>(sb >> 4), static_cast<uint8_t>(xb >> 4));
            s0 += adc_xflag_nibble_score(row + 16, sb, xb);
            sb = sign[byte + 1];
            xb = xflag[byte + 1];
            s1 += adc_xflag_nibble_score(row + 32, static_cast<uint8_t>(sb >> 4), static_cast<uint8_t>(xb >> 4));
            s1 += adc_xflag_nibble_score(row + 48, sb, xb);
            sb = sign[byte + 2];
            xb = xflag[byte + 2];
            s2 += adc_xflag_nibble_score(row + 64, static_cast<uint8_t>(sb >> 4), static_cast<uint8_t>(xb >> 4));
            s2 += adc_xflag_nibble_score(row + 80, sb, xb);
            sb = sign[byte + 3];
            xb = xflag[byte + 3];
            s3 += adc_xflag_nibble_score(row + 96, static_cast<uint8_t>(sb >> 4), static_cast<uint8_t>(xb >> 4));
            s3 += adc_xflag_nibble_score(row + 112, sb, xb);
        }
        float score = s0 + s1 + s2 + s3;
        for (; byte < code_size; ++byte) {
            const float* row = lut + byte * 32;
            const uint8_t sb = sign[byte];
            const uint8_t xb = xflag[byte];
            score += adc_xflag_nibble_score(row, static_cast<uint8_t>(sb >> 4), static_cast<uint8_t>(xb >> 4));
            score += adc_xflag_nibble_score(row + 16, sb, xb);
        }
        return -score;
    }


#if defined(__AVX2__)
    static __m256 lookup16_ps_avx2(
            const float* row,
            __m256i idx,
            __m256i idx7) {
        const __m256 lo = _mm256_loadu_ps(row);
        const __m256 hi = _mm256_loadu_ps(row + 8);
        const __m256i low_idx = _mm256_and_si256(idx, idx7);
        const __m256 lo_vals = _mm256_permutevar8x32_ps(lo, low_idx);
        const __m256 hi_vals = _mm256_permutevar8x32_ps(hi, low_idx);
        const __m256 hi_mask = _mm256_castsi256_ps(_mm256_cmpgt_epi32(idx, idx7));
        return _mm256_blendv_ps(lo_vals, hi_vals, hi_mask);
    }

    static __m256 lookup8_adc_xflag_nibble_score_avx2(
            const float* row,
            __m256i sign_idx,
            __m256i flag_idx,
            __m256i idx7) {
        const __m256 sign_vals = lookup16_ps_avx2(row, sign_idx, idx7);
        const __m256 weak_sign_vals = lookup16_ps_avx2(row, _mm256_and_si256(sign_idx, flag_idx), idx7);
        const __m256 weak_vals = lookup16_ps_avx2(row, flag_idx, idx7);
        __m256 score = _mm256_add_ps(sign_vals, sign_vals);
        score = _mm256_sub_ps(score, _mm256_set1_ps(row[15]));
        score = _mm256_sub_ps(score, weak_sign_vals);
        score = _mm256_add_ps(score, _mm256_mul_ps(_mm256_set1_ps(0.5f), weak_vals));
        return score;
    }

    static __m256 lookup_byte8_adc_xflag_nibble_avx2(
            const float* row,
            uint8_t s0,
            uint8_t s1,
            uint8_t s2,
            uint8_t s3,
            uint8_t s4,
            uint8_t s5,
            uint8_t s6,
            uint8_t s7,
            uint8_t x0,
            uint8_t x1,
            uint8_t x2,
            uint8_t x3,
            uint8_t x4,
            uint8_t x5,
            uint8_t x6,
            uint8_t x7,
            __m256i idx7,
            __m256i low_nibble_mask) {
        const __m128i sign8 = _mm_setr_epi8(
                static_cast<char>(s0), static_cast<char>(s1),
                static_cast<char>(s2), static_cast<char>(s3),
                static_cast<char>(s4), static_cast<char>(s5),
                static_cast<char>(s6), static_cast<char>(s7),
                0, 0, 0, 0, 0, 0, 0, 0);
        const __m128i flag8 = _mm_setr_epi8(
                static_cast<char>(x0), static_cast<char>(x1),
                static_cast<char>(x2), static_cast<char>(x3),
                static_cast<char>(x4), static_cast<char>(x5),
                static_cast<char>(x6), static_cast<char>(x7),
                0, 0, 0, 0, 0, 0, 0, 0);
        const __m256i sign32 = _mm256_cvtepu8_epi32(sign8);
        const __m256i flag32 = _mm256_cvtepu8_epi32(flag8);
        const __m256i sign_hi = _mm256_srli_epi32(sign32, 4);
        const __m256i flag_hi = _mm256_srli_epi32(flag32, 4);
        const __m256i sign_lo = _mm256_and_si256(sign32, low_nibble_mask);
        const __m256i flag_lo = _mm256_and_si256(flag32, low_nibble_mask);
        return _mm256_add_ps(
                lookup8_adc_xflag_nibble_score_avx2(row, sign_hi, flag_hi, idx7),
                lookup8_adc_xflag_nibble_score_avx2(row + 16, sign_lo, flag_lo, idx7));
    }

    void batch8_adc_xflag_nibble_avx2(
            const faiss::idx_t* ids,
            float* dis_out) const {
        const uint8_t* s0 = sign_at(ids[0]);
        const uint8_t* s1 = sign_at(ids[1]);
        const uint8_t* s2 = sign_at(ids[2]);
        const uint8_t* s3 = sign_at(ids[3]);
        const uint8_t* s4 = sign_at(ids[4]);
        const uint8_t* s5 = sign_at(ids[5]);
        const uint8_t* s6 = sign_at(ids[6]);
        const uint8_t* s7 = sign_at(ids[7]);
        const uint8_t* x0 = xflag_at(ids[0]);
        const uint8_t* x1 = xflag_at(ids[1]);
        const uint8_t* x2 = xflag_at(ids[2]);
        const uint8_t* x3 = xflag_at(ids[3]);
        const uint8_t* x4 = xflag_at(ids[4]);
        const uint8_t* x5 = xflag_at(ids[5]);
        const uint8_t* x6 = xflag_at(ids[6]);
        const uint8_t* x7 = xflag_at(ids[7]);
        __builtin_prefetch(s0, 0, 1);
        __builtin_prefetch(s1, 0, 1);
        __builtin_prefetch(s2, 0, 1);
        __builtin_prefetch(s3, 0, 1);
        __builtin_prefetch(s4, 0, 1);
        __builtin_prefetch(s5, 0, 1);
        __builtin_prefetch(s6, 0, 1);
        __builtin_prefetch(s7, 0, 1);
        __builtin_prefetch(x0, 0, 1);
        __builtin_prefetch(x1, 0, 1);
        __builtin_prefetch(x2, 0, 1);
        __builtin_prefetch(x3, 0, 1);
        __builtin_prefetch(x4, 0, 1);
        __builtin_prefetch(x5, 0, 1);
        __builtin_prefetch(x6, 0, 1);
        __builtin_prefetch(x7, 0, 1);

        const float* lut = subset_lut.data();
        const __m256i idx7 = _mm256_set1_epi32(7);
        const __m256i low_nibble_mask = _mm256_set1_epi32(0x0f);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        size_t byte = 0;
        for (; byte + 4 <= code_size; byte += 4) {
            const float* row = lut + byte * 32;
            acc0 = _mm256_add_ps(acc0, lookup_byte8_adc_xflag_nibble_avx2(
                    row,
                    s0[byte + 0], s1[byte + 0], s2[byte + 0], s3[byte + 0],
                    s4[byte + 0], s5[byte + 0], s6[byte + 0], s7[byte + 0],
                    x0[byte + 0], x1[byte + 0], x2[byte + 0], x3[byte + 0],
                    x4[byte + 0], x5[byte + 0], x6[byte + 0], x7[byte + 0],
                    idx7,
                    low_nibble_mask));
            acc1 = _mm256_add_ps(acc1, lookup_byte8_adc_xflag_nibble_avx2(
                    row + 32,
                    s0[byte + 1], s1[byte + 1], s2[byte + 1], s3[byte + 1],
                    s4[byte + 1], s5[byte + 1], s6[byte + 1], s7[byte + 1],
                    x0[byte + 1], x1[byte + 1], x2[byte + 1], x3[byte + 1],
                    x4[byte + 1], x5[byte + 1], x6[byte + 1], x7[byte + 1],
                    idx7,
                    low_nibble_mask));
            acc2 = _mm256_add_ps(acc2, lookup_byte8_adc_xflag_nibble_avx2(
                    row + 64,
                    s0[byte + 2], s1[byte + 2], s2[byte + 2], s3[byte + 2],
                    s4[byte + 2], s5[byte + 2], s6[byte + 2], s7[byte + 2],
                    x0[byte + 2], x1[byte + 2], x2[byte + 2], x3[byte + 2],
                    x4[byte + 2], x5[byte + 2], x6[byte + 2], x7[byte + 2],
                    idx7,
                    low_nibble_mask));
            acc3 = _mm256_add_ps(acc3, lookup_byte8_adc_xflag_nibble_avx2(
                    row + 96,
                    s0[byte + 3], s1[byte + 3], s2[byte + 3], s3[byte + 3],
                    s4[byte + 3], s5[byte + 3], s6[byte + 3], s7[byte + 3],
                    x0[byte + 3], x1[byte + 3], x2[byte + 3], x3[byte + 3],
                    x4[byte + 3], x5[byte + 3], x6[byte + 3], x7[byte + 3],
                    idx7,
                    low_nibble_mask));
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        for (; byte < code_size; ++byte) {
            acc = _mm256_add_ps(acc, lookup_byte8_adc_xflag_nibble_avx2(
                    lut + byte * 32,
                    s0[byte], s1[byte], s2[byte], s3[byte],
                    s4[byte], s5[byte], s6[byte], s7[byte],
                    x0[byte], x1[byte], x2[byte], x3[byte],
                    x4[byte], x5[byte], x6[byte], x7[byte],
                    idx7,
                    low_nibble_mask));
        }
        _mm256_storeu_ps(dis_out, _mm256_sub_ps(_mm256_setzero_ps(), acc));
    }
#endif

    float distance_idx(faiss::idx_t i) const {
        if (i < 0) return std::numeric_limits<float>::max();
        return distance_to_codes(sign_at(i), xflag_at(i));
    }

    float distance_idx_pruned(faiss::idx_t i, float) const {
        return distance_idx(i);
    }

    float operator()(faiss::idx_t i) override {
        return distance_idx(i);
    }

    float safe_distance(faiss::idx_t i) const {
        return distance_idx(i);
    }

    void prefetch_idx(faiss::idx_t i) const {
        if (i >= 0) {
            __builtin_prefetch(sign_at(i), 0, 1);
            __builtin_prefetch(xflag_at(i), 0, 1);
        }
    }

    void distances_batch_16(const faiss::idx_t* ids, float* dis_out) {
#if defined(__AVX2__)
        batch8_adc_xflag_nibble_avx2(ids, dis_out);
        batch8_adc_xflag_nibble_avx2(ids + 8, dis_out + 8);
#else
        for (int i = 0; i < 16; ++i) {
            dis_out[i] = distance_idx(ids[i]);
        }
#endif
    }

    void distances_batch_32(const faiss::idx_t* ids, float* dis_out) {
        distances_batch_16(ids, dis_out);
        distances_batch_16(ids + 16, dis_out + 16);
    }

    bool batch_all_neighbors() const { return false; }

    void distances_batch_n(
            const faiss::idx_t* ids,
            int count,
            float* distances) {
        for (int i = 0; i < count; ++i) distances[i] = distance_idx(ids[i]);
    }

    void distances_batch_8(const faiss::idx_t* ids, float* dis_out) {
#if defined(__AVX2__)
        batch8_adc_xflag_nibble_avx2(ids, dis_out);
#else
        for (int i = 0; i < 8; ++i) {
            dis_out[i] = distance_idx(ids[i]);
        }
#endif
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
        dis0 = safe_distance(idx0);
        dis1 = safe_distance(idx1);
        dis2 = safe_distance(idx2);
        dis3 = safe_distance(idx3);
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const uint8_t* a = sign_at(i);
        const uint8_t* ax = xflag_at(i);
        const uint8_t* b = sign_at(j);
        const uint8_t* bx = xflag_at(j);
        uint32_t kept_mismatch = 0;
        uint32_t weak_neutral = 0;
        size_t byte = 0;
        for (; byte + 8 <= code_size; byte += 8) {
            const uint64_t mismatch = PackedXFlagBQDistanceComputer::load_u64(a + byte) ^
                    PackedXFlagBQDistanceComputer::load_u64(b + byte);
            const uint64_t neutral = PackedXFlagBQDistanceComputer::load_u64(ax + byte) &
                    PackedXFlagBQDistanceComputer::load_u64(bx + byte);
            kept_mismatch += static_cast<uint32_t>(__builtin_popcountll(mismatch & ~neutral));
            weak_neutral += static_cast<uint32_t>(__builtin_popcountll(neutral));
        }
        for (; byte < code_size; ++byte) {
            const uint8_t mismatch = static_cast<uint8_t>(a[byte] ^ b[byte]);
            const uint8_t neutral = static_cast<uint8_t>(ax[byte] & bx[byte]);
            kept_mismatch += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(mismatch & ~neutral)));
            weak_neutral += static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(neutral)));
        }
        return static_cast<float>(2u * kept_mismatch + weak_neutral);
    }
};

std::unique_ptr<faiss::IndexBinaryHNSW> build_binary_graph_index(const Args& args, const FbinMmap& train) {
    std::cout << "Building packed BinaryHNSW: n=" << train.n << ", dim=" << train.d
              << ", M=" << args.m << ", efC=" << args.ef_construction << std::endl;
    auto index = std::make_unique<faiss::IndexBinaryHNSW>(train.d, args.m);
    index->hnsw.efConstruction = args.ef_construction;
    omp_set_num_threads(args.build_threads);
    std::vector<uint8_t> codes;
    for (int64_t start = 0; start < train.n; start += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(train.n, start + args.add_batch_size);
        const int64_t rows = end - start;
        pack_pgvector_sign_bits(train.row(static_cast<size_t>(start)), static_cast<size_t>(rows), train.d, codes);
        index->add(rows, codes.data());
        std::cout << "  added packed codes " << end << "/" << train.n << std::endl;
    }
    return index;
}

std::unique_ptr<faiss::IndexBinaryHNSW> build_fp_graph_bq_storage_index(const Args& args, const FbinMmap& train) {
    std::cout << "Building FP-HNSW graph then storing packed BQ payload: n=" << train.n
              << ", dim=" << train.d << ", M=" << args.m
              << ", efC=" << args.ef_construction << std::endl;

    faiss::IndexHNSWFlat fp_index(train.d, args.m, faiss::METRIC_INNER_PRODUCT);
    fp_index.hnsw.efConstruction = args.ef_construction;
    fp_index.verbose = false;
    omp_set_num_threads(args.build_threads);
    for (int64_t start = 0; start < train.n; start += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(train.n, start + args.add_batch_size);
        const int64_t rows = end - start;
        std::vector<float> block = normalized_train_block(train, start, rows);
        fp_index.add(rows, block.data());
        std::cout << "  built FP graph vectors " << end << "/" << train.n << std::endl;
    }

    auto storage = new faiss::IndexBinaryFlat(train.d);
    std::vector<uint8_t> codes;
    for (int64_t start = 0; start < train.n; start += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(train.n, start + args.add_batch_size);
        const int64_t rows = end - start;
        pack_pgvector_sign_bits(train.row(static_cast<size_t>(start)), static_cast<size_t>(rows), train.d, codes);
        storage->add(rows, codes.data());
        std::cout << "  stored packed BQ codes " << end << "/" << train.n << std::endl;
    }

    auto index = std::make_unique<faiss::IndexBinaryHNSW>(storage, args.m);
    index->own_fields = true;
    index->ntotal = storage->ntotal;
    index->hnsw = fp_index.hnsw;
    index->hnsw.efConstruction = args.ef_construction;
    index->hnsw.efSearch = fp_index.hnsw.efSearch;
    return index;
}

std::unique_ptr<faiss::IndexBinaryHNSW> build_or_load_index(const Args& args, const FbinMmap& train) {
    if (train.d % 8 != 0) {
        throw std::runtime_error("this benchmark expects dim to be divisible by 8");
    }
    if (fs::exists(args.index) && !args.rebuild) {
        std::cout << "Loading packed BinaryHNSW: " << args.index << std::endl;
        faiss::IndexBinary* raw = faiss::read_index_binary(args.index.c_str());
        auto* hnsw = dynamic_cast<faiss::IndexBinaryHNSW*>(raw);
        if (!hnsw) {
            delete raw;
            throw std::runtime_error("index is not IndexBinaryHNSW: " + args.index);
        }
        if (hnsw->d != train.d) {
            delete hnsw;
            throw std::runtime_error("index dim does not match train dim");
        }
        return std::unique_ptr<faiss::IndexBinaryHNSW>(hnsw);
    }

    fs::create_directories(fs::path(args.index).parent_path());
    auto index = args.graph_build_mode == GraphBuildMode::Fp
            ? build_fp_graph_bq_storage_index(args, train)
            : build_binary_graph_index(args, train);
    faiss::write_index_binary(index.get(), args.index.c_str());
    std::cout << "Saved packed BinaryHNSW -> " << args.index
              << " (graph_build=" << graph_build_mode_name(args.graph_build_mode) << ")" << std::endl;
    return index;
}

const uint8_t* storage_codes(const faiss::IndexBinaryHNSW& index) {
    auto* flat = dynamic_cast<faiss::IndexBinaryFlat*>(index.storage);
    if (!flat) throw std::runtime_error("IndexBinaryHNSW storage is not IndexBinaryFlat");
    return flat->xb.data();
}

double seconds_since(std::chrono::steady_clock::time_point t0);

struct ColocationArtifacts {
    std::vector<uint32_t> physical_row_by_id;
    std::string map_path;
    double prepare_seconds = 0.0;
    double same_partition_edge_ratio = 0.0;
    bool loaded = false;
};

constexpr std::array<char, 8> kColocationMapMagic = {'A', 'D', 'C', 'L', 'D', 'G', '2', '\0'};

void for_each_level0_neighbor(
        const faiss::HNSW& hnsw,
        uint32_t node,
        const std::function<void(uint32_t)>& visitor) {
    size_t begin = 0;
    size_t end = 0;
    hnsw.neighbor_range(static_cast<faiss::HNSW::storage_idx_t>(node), 0, &begin, &end);
    for (size_t pos = begin; pos < end; ++pos) {
        const faiss::HNSW::storage_idx_t neighbor = hnsw.neighbors[pos];
        if (neighbor < 0) break;
        visitor(static_cast<uint32_t>(neighbor));
    }
}

std::vector<uint32_t> build_colocation_physical_rows(
        const faiss::IndexBinaryHNSW& index,
        int iterations,
        int partition_capacity,
        size_t index_record_bytes,
        double& same_partition_edge_ratio) {
    const size_t n = static_cast<size_t>(index.ntotal);
    if (n > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("colocation currently supports at most UINT32_MAX vectors");
    }
    if (n == 0) {
        same_partition_edge_ratio = 0.0;
        return {};
    }

    const size_t num_partitions =
            (n + static_cast<size_t>(partition_capacity) - 1) /
            static_cast<size_t>(partition_capacity);
    const uint32_t unassigned = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> assignment(n, unassigned);
    std::vector<int> sizes(num_partitions, 0);

    // Match pgvector's initialization: seed a partition with the next unassigned
    // node, then pull as many of its level-0 neighbors as capacity permits.
    size_t current_partition = 0;
    auto advance_partition = [&]() {
        while (current_partition < num_partitions &&
               sizes[current_partition] >= partition_capacity) {
            ++current_partition;
        }
        if (current_partition >= num_partitions) {
            throw std::runtime_error("no unfilled colocation partition remains");
        }
    };
    for (uint32_t node = 0; node < static_cast<uint32_t>(n); ++node) {
        if (assignment[node] != unassigned) continue;
        advance_partition();
        const uint32_t pid = static_cast<uint32_t>(current_partition);
        assignment[node] = pid;
        ++sizes[pid];
        for_each_level0_neighbor(index.hnsw, node, [&](uint32_t neighbor) {
            if (neighbor >= n || assignment[neighbor] != unassigned ||
                sizes[pid] >= partition_capacity) {
                return;
            }
            assignment[neighbor] = pid;
            ++sizes[pid];
        });
    }

    using HeapEntry = std::pair<int, uint32_t>;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<uint32_t> next_assignment(n, unassigned);
        std::vector<int> next_sizes(num_partitions, 0);
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
        for (uint32_t pid = 0; pid < static_cast<uint32_t>(num_partitions); ++pid) {
            heap.emplace(0, pid);
        }

        for (uint32_t node = 0; node < static_cast<uint32_t>(n); ++node) {
            std::vector<std::pair<uint32_t, int>> scores;
            scores.reserve(static_cast<size_t>(index.hnsw.nb_neighbors(0)));
            int best_score = 0;
            uint32_t best_partition = unassigned;
            for_each_level0_neighbor(index.hnsw, node, [&](uint32_t neighbor) {
                if (neighbor >= n) return;
                const uint32_t pid = assignment[neighbor];
                if (pid == unassigned || pid >= num_partitions) return;
                int score = 1;
                bool found = false;
                for (auto& item : scores) {
                    if (item.first == pid) {
                        score = ++item.second;
                        found = true;
                        break;
                    }
                }
                if (!found) scores.emplace_back(pid, score);
                if (score > best_score && next_sizes[pid] < partition_capacity) {
                    best_score = score;
                    best_partition = pid;
                }
            });

            if (best_partition == unassigned) {
                while (!heap.empty()) {
                    const auto [recorded_size, pid] = heap.top();
                    if (recorded_size != next_sizes[pid] || recorded_size >= partition_capacity) {
                        heap.pop();
                        continue;
                    }
                    best_partition = pid;
                    heap.pop();
                    break;
                }
                if (best_partition == unassigned) {
                    throw std::runtime_error("no unfilled colocation partition remains during LDG");
                }
            }

            next_assignment[node] = best_partition;
            ++next_sizes[best_partition];
            if (next_sizes[best_partition] < partition_capacity) {
                heap.emplace(next_sizes[best_partition], best_partition);
            }
        }
        assignment.swap(next_assignment);
        sizes.swap(next_sizes);
        std::cout << "  colocation LDG iteration " << (iteration + 1) << "/"
                  << iterations << std::endl;
    }

    std::vector<std::vector<uint32_t>> partitions(num_partitions);
    for (size_t pid = 0; pid < num_partitions; ++pid) {
        partitions[pid].reserve(static_cast<size_t>(sizes[pid]));
    }
    for (uint32_t node = 0; node < static_cast<uint32_t>(n); ++node) {
        partitions[assignment[node]].push_back(node);
    }

    // Match pgvector's page-aware pass after LDG. Within each <=64-node
    // partition, greedily place connected nodes in small physical page groups.
    constexpr size_t kPageGroupBudget = 4 * 8192;
    std::vector<int8_t> local_index(n, static_cast<int8_t>(-1));
    for (auto& partition : partitions) {
        if (partition.size() <= 1) continue;
        if (partition.size() > 64) {
            throw std::runtime_error("page-aware colocation partition exceeds 64 nodes");
        }
        for (size_t i = 0; i < partition.size(); ++i) {
            local_index[partition[i]] = static_cast<int8_t>(i);
        }
        std::vector<uint64_t> neighbor_masks(partition.size(), 0);
        for (size_t i = 0; i < partition.size(); ++i) {
            for_each_level0_neighbor(index.hnsw, partition[i], [&](uint32_t neighbor) {
                if (neighbor >= n) return;
                const int local = local_index[neighbor];
                if (local >= 0) neighbor_masks[i] |= uint64_t{1} << local;
            });
        }

        std::vector<uint32_t> ordered;
        ordered.reserve(partition.size());
        uint64_t placed_mask = 0;
        while (ordered.size() < partition.size()) {
            int seed = -1;
            int best_placed = -1;
            int best_degree = -1;
            for (size_t i = 0; i < partition.size(); ++i) {
                const uint64_t bit = uint64_t{1} << i;
                if (placed_mask & bit) continue;
                const int placed_score = __builtin_popcountll(neighbor_masks[i] & placed_mask);
                const int degree = __builtin_popcountll(neighbor_masks[i]);
                if (placed_score > best_placed ||
                    (placed_score == best_placed && degree > best_degree)) {
                    seed = static_cast<int>(i);
                    best_placed = placed_score;
                    best_degree = degree;
                }
            }
            if (seed < 0) break;

            size_t group_bytes = 0;
            uint64_t group_mask = 0;
            for (;;) {
                const uint64_t bit = uint64_t{1} << seed;
                ordered.push_back(partition[static_cast<size_t>(seed)]);
                placed_mask |= bit;
                group_mask |= bit;
                group_bytes += index_record_bytes;
                if (ordered.size() >= partition.size()) break;

                int next = -1;
                int best_group_score = 0;
                int best_all_placed_score = 0;
                for (size_t i = 0; i < partition.size(); ++i) {
                    const uint64_t candidate_bit = uint64_t{1} << i;
                    if (placed_mask & candidate_bit) continue;
                    if (group_bytes > 0 && group_bytes + index_record_bytes > kPageGroupBudget) continue;
                    const int group_score = __builtin_popcountll(neighbor_masks[i] & group_mask);
                    const int all_placed_score = __builtin_popcountll(neighbor_masks[i] & placed_mask);
                    if (group_score > best_group_score ||
                        (group_score == best_group_score &&
                         all_placed_score > best_all_placed_score)) {
                        next = static_cast<int>(i);
                        best_group_score = group_score;
                        best_all_placed_score = all_placed_score;
                    }
                }
                if (next < 0) break;
                seed = next;
            }
        }
        if (ordered.size() != partition.size()) {
            throw std::runtime_error("failed to page-order colocation partition");
        }
        for (uint32_t node : partition) local_index[node] = static_cast<int8_t>(-1);
        partition.swap(ordered);
    }

    std::vector<uint32_t> physical_row_by_id(n, unassigned);
    uint32_t physical_row = 0;
    for (const auto& partition : partitions) {
        for (uint32_t node : partition) {
            physical_row_by_id[node] = physical_row++;
        }
    }
    if (physical_row != n) {
        throw std::runtime_error("colocation mapping did not place every vector");
    }

    uint64_t edge_count = 0;
    uint64_t same_partition_edges = 0;
    for (uint32_t node = 0; node < static_cast<uint32_t>(n); ++node) {
        for_each_level0_neighbor(index.hnsw, node, [&](uint32_t neighbor) {
            if (neighbor >= n) return;
            ++edge_count;
            if (assignment[node] == assignment[neighbor]) ++same_partition_edges;
        });
    }
    same_partition_edge_ratio = edge_count == 0
            ? 0.0
            : static_cast<double>(same_partition_edges) / static_cast<double>(edge_count);
    return physical_row_by_id;
}

bool validate_aligned_fbin(const std::string& path, int32_t n, int32_t d) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    int32_t header[2] = {0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || header[0] != n || header[1] != d) return false;
    const uintmax_t expected = static_cast<uintmax_t>(kFpDataAlignment) +
            static_cast<uintmax_t>(n) *
            static_cast<uintmax_t>(d) * sizeof(float);
    std::error_code ec;
    const uintmax_t actual = fs::file_size(path, ec);
    return !ec && actual == expected;
}

void write_aligned_fbin(
        const std::string& path,
        const FbinMmap& train,
        const std::vector<uint32_t>* original_id_by_physical,
        const char* description) {
    const fs::path output_path(path);
    if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
    const std::string tmp_path = path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    std::ofstream out;
    out.rdbuf()->pubsetbuf(stream_buffer.data(), static_cast<std::streamsize>(stream_buffer.size()));
    out.open(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(std::string("failed to create ") + description + ": " + tmp_path);

    out.write(reinterpret_cast<const char*>(&train.n), sizeof(train.n));
    out.write(reinterpret_cast<const char*>(&train.d), sizeof(train.d));
    const std::array<char, kFpDataAlignment - kLegacyFbinHeaderBytes> padding{};
    out.write(padding.data(), static_cast<std::streamsize>(padding.size()));

    const size_t row_bytes = static_cast<size_t>(train.d) * sizeof(float);
    if (original_id_by_physical) {
        for (size_t physical = 0; physical < original_id_by_physical->size(); ++physical) {
            const uint32_t original = (*original_id_by_physical)[physical];
            out.write(
                    reinterpret_cast<const char*>(train.row(original)),
                    static_cast<std::streamsize>(row_bytes));
            if ((physical + 1) % 100000 == 0 || physical + 1 == original_id_by_physical->size()) {
                std::cout << "  wrote " << description << " rows " << (physical + 1) << "/"
                          << original_id_by_physical->size() << std::endl;
            }
        }
    } else {
        constexpr size_t target_chunk_bytes = 8 * 1024 * 1024;
        const size_t rows_per_chunk = std::max<size_t>(1, target_chunk_bytes / row_bytes);
        for (size_t first = 0; first < static_cast<size_t>(train.n); first += rows_per_chunk) {
            const size_t count = std::min(rows_per_chunk, static_cast<size_t>(train.n) - first);
            out.write(
                    reinterpret_cast<const char*>(train.row(first)),
                    static_cast<std::streamsize>(count * row_bytes));
        }
    }
    out.close();
    if (!out) throw std::runtime_error(std::string("failed to write ") + description + ": " + tmp_path);
    fs::rename(tmp_path, path);
}

void prepare_aligned_fp_store(const std::string& path, const FbinMmap& train) {
    if (validate_aligned_fbin(path, train.n, train.d)) {
        std::cout << "Loaded 4KB-aligned FP rerank store: " << path << std::endl;
        return;
    }
    if (path == train.path) {
        throw std::runtime_error("refusing to replace legacy --train file in place; use a separate --fp-rerank-path");
    }
    std::cout << "Building 4KB-aligned FP rerank store: " << path << std::endl;
    write_aligned_fbin(path, train, nullptr, "aligned FP");
    std::cout << "Saved 4KB-aligned FP rerank store: " << path << std::endl;
}

bool load_colocation_map(
        const std::string& path,
        int32_t n,
        int32_t d,
        int iterations,
        int partition_capacity,
        std::vector<uint32_t>& physical_row_by_id) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::array<char, 8> magic{};
    uint64_t stored_n = 0;
    uint32_t stored_d = 0;
    uint32_t stored_iterations = 0;
    uint32_t stored_capacity = 0;
    uint32_t reserved = 0;
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    in.read(reinterpret_cast<char*>(&stored_n), sizeof(stored_n));
    in.read(reinterpret_cast<char*>(&stored_d), sizeof(stored_d));
    in.read(reinterpret_cast<char*>(&stored_iterations), sizeof(stored_iterations));
    in.read(reinterpret_cast<char*>(&stored_capacity), sizeof(stored_capacity));
    in.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
    if (!in || magic != kColocationMapMagic || stored_n != static_cast<uint64_t>(n) ||
        stored_d != static_cast<uint32_t>(d) ||
        stored_iterations != static_cast<uint32_t>(iterations) ||
        stored_capacity != static_cast<uint32_t>(partition_capacity)) {
        return false;
    }
    physical_row_by_id.resize(static_cast<size_t>(n));
    in.read(
            reinterpret_cast<char*>(physical_row_by_id.data()),
            static_cast<std::streamsize>(physical_row_by_id.size() * sizeof(uint32_t)));
    if (!in) return false;

    std::vector<uint8_t> seen(static_cast<size_t>(n), 0);
    for (uint32_t physical : physical_row_by_id) {
        if (physical >= static_cast<uint32_t>(n) || seen[physical]) {
            throw std::runtime_error("invalid colocation permutation in " + path);
        }
        seen[physical] = 1;
    }
    return true;
}

void write_colocation_map(
        const std::string& path,
        int32_t n,
        int32_t d,
        int iterations,
        int partition_capacity,
        const std::vector<uint32_t>& physical_row_by_id) {
    const std::string tmp_path = path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to create colocation map: " + tmp_path);
    const uint64_t stored_n = static_cast<uint64_t>(n);
    const uint32_t stored_d = static_cast<uint32_t>(d);
    const uint32_t stored_iterations = static_cast<uint32_t>(iterations);
    const uint32_t stored_capacity = static_cast<uint32_t>(partition_capacity);
    const uint32_t reserved = 0;
    out.write(kColocationMapMagic.data(), static_cast<std::streamsize>(kColocationMapMagic.size()));
    out.write(reinterpret_cast<const char*>(&stored_n), sizeof(stored_n));
    out.write(reinterpret_cast<const char*>(&stored_d), sizeof(stored_d));
    out.write(reinterpret_cast<const char*>(&stored_iterations), sizeof(stored_iterations));
    out.write(reinterpret_cast<const char*>(&stored_capacity), sizeof(stored_capacity));
    out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    out.write(
            reinterpret_cast<const char*>(physical_row_by_id.data()),
            static_cast<std::streamsize>(physical_row_by_id.size() * sizeof(uint32_t)));
    out.close();
    if (!out) throw std::runtime_error("failed to write colocation map: " + tmp_path);
    fs::rename(tmp_path, path);
}

ColocationArtifacts prepare_colocation(
        const Args& args,
        const faiss::IndexBinaryHNSW& index,
        const FbinMmap& train) {
    ColocationArtifacts artifacts;
    if (args.colocation_iterations <= 0) return artifacts;

    artifacts.map_path = args.colocation_map_path.empty()
            ? args.index + ".coloc.map"
            : args.colocation_map_path;
    const fs::path map_path(artifacts.map_path);
    if (!map_path.parent_path().empty()) fs::create_directories(map_path.parent_path());

    const auto start = std::chrono::steady_clock::now();
    if (!args.rebuild_colocation &&
        load_colocation_map(
                artifacts.map_path,
                train.n,
                train.d,
                args.colocation_iterations,
                args.colocation_partition_size,
                artifacts.physical_row_by_id)) {
        artifacts.loaded = true;
        artifacts.prepare_seconds = seconds_since(start);
        std::cout << "Loaded neighbor-colocated index permutation: "
                  << artifacts.map_path << std::endl;
        return artifacts;
    }

    std::cout << "Building neighbor colocation with LDG iterations="
              << args.colocation_iterations << ", partition_size="
              << args.colocation_partition_size << std::endl;
    artifacts.physical_row_by_id = build_colocation_physical_rows(
            index,
            args.colocation_iterations,
            args.colocation_partition_size,
            static_cast<size_t>(index.code_size) +
                    static_cast<size_t>(index.hnsw.nb_neighbors(0)) *
                            sizeof(faiss::HNSW::storage_idx_t) +
                    32,
            artifacts.same_partition_edge_ratio);
    write_colocation_map(
            artifacts.map_path,
            train.n,
            train.d,
            args.colocation_iterations,
            args.colocation_partition_size,
            artifacts.physical_row_by_id);
    artifacts.prepare_seconds = seconds_since(start);
    std::cout << "Saved neighbor-colocated index permutation: " << artifacts.map_path
              << ", level0_same_partition_edge_ratio="
              << artifacts.same_partition_edge_ratio
              << ", seconds=" << artifacts.prepare_seconds << std::endl;
    return artifacts;
}

std::vector<uint32_t> apply_index_colocation(
        faiss::IndexBinaryHNSW& index,
        const std::vector<uint32_t>& colocated_id_by_original_id) {
    const size_t n = static_cast<size_t>(index.ntotal);
    if (colocated_id_by_original_id.size() != n) {
        throw std::runtime_error("index colocation permutation size mismatch");
    }

    std::vector<uint32_t> original_id_by_colocated_id(n);
    for (uint32_t original = 0; original < static_cast<uint32_t>(n); ++original) {
        const uint32_t colocated = colocated_id_by_original_id[original];
        if (colocated >= n) throw std::runtime_error("invalid index colocation id");
        original_id_by_colocated_id[colocated] = original;
    }

    auto* flat = dynamic_cast<faiss::IndexBinaryFlat*>(index.storage);
    if (!flat) throw std::runtime_error("index colocation requires IndexBinaryFlat storage");
    auto reorder_bytes = [&](faiss::MaybeOwnedVector<uint8_t>& values, size_t stride) {
        if (values.size() == 0) return;
        if (values.size() != n * stride) {
            throw std::runtime_error("index colocation storage size mismatch");
        }
        std::vector<uint8_t> reordered(values.size());
#pragma omp parallel for schedule(static)
        for (int64_t colocated = 0; colocated < static_cast<int64_t>(n); ++colocated) {
            const size_t original = original_id_by_colocated_id[static_cast<size_t>(colocated)];
            std::memcpy(
                    reordered.data() + static_cast<size_t>(colocated) * stride,
                    values.data() + original * stride,
                    stride);
        }
        values = faiss::MaybeOwnedVector<uint8_t>(std::move(reordered));
    };

    reorder_bytes(flat->xb, index.code_size);
    bbq_compat::reorder_fork_payloads(*flat, index.code_size, reorder_bytes);

    std::vector<faiss::idx_t> new_to_old(n);
    for (size_t colocated = 0; colocated < n; ++colocated) {
        new_to_old[colocated] = static_cast<faiss::idx_t>(original_id_by_colocated_id[colocated]);
    }
    index.hnsw.permute_entries(new_to_old.data());
    std::cout << "Applied index colocation to BQ codes and HNSW adjacency: "
              << n << " nodes" << std::endl;
    return original_id_by_colocated_id;
}

void restore_original_ids(
        std::vector<faiss::idx_t>& labels,
        const std::vector<uint32_t>& original_id_by_colocated_id) {
    if (original_id_by_colocated_id.empty()) return;
    for (faiss::idx_t& label : labels) {
        const faiss::idx_t colocated = label;
        if (colocated < 0) continue;
        if (static_cast<size_t>(colocated) >= original_id_by_colocated_id.size()) {
            throw std::runtime_error("search returned invalid colocated id");
        }
        label = static_cast<faiss::idx_t>(
                original_id_by_colocated_id[static_cast<size_t>(colocated)]);
    }
}

#ifndef ADCBQ_PAGED_INDEX_PAGE_BYTES
#define ADCBQ_PAGED_INDEX_PAGE_BYTES 4096
#endif
constexpr size_t kPagedIndexPageBytes = ADCBQ_PAGED_INDEX_PAGE_BYTES;
static_assert(
        kPagedIndexPageBytes == 4096 || kPagedIndexPageBytes == 8192 ||
                kPagedIndexPageBytes == 16384,
        "paged index page size must be 4KB, 8KB, or 16KB");
constexpr size_t kPagedNodeHeaderBytes = 64;
constexpr std::array<char, 8> kPagedIndexMagic = {'A', 'D', 'C', 'P', 'G', '0', '2', '\0'};

struct PagedIndexFileHeader {
    std::array<char, 8> magic{};
    uint32_t version = 1;
    uint32_t page_bytes = kPagedIndexPageBytes;
    uint64_t ntotal = 0;
    uint32_t dim = 0;
    uint32_t code_size = 0;
    uint32_t m = 0;
    int32_t entry_point = -1;
    int32_t max_level = -1;
    uint64_t neighbor_slots = 0;
    uint64_t locator_bytes = 0;
    uint64_t data_start_page = 0;
    uint64_t total_pages = 0;
};

struct PagedNodeLocator {
    uint32_t page_no = 0;
    uint16_t offset = 0;
    uint16_t record_bytes = 0;
};

struct alignas(64) PagedNodeRecordHeader {
    uint16_t levels = 0;
    uint16_t reserved16 = 0;
    uint32_t neighbor_slots = 0;
    std::array<uint8_t, kPagedNodeHeaderBytes - 8> reserved{};
};

static_assert(sizeof(PagedNodeLocator) == 8, "paged node locator must be compact");
static_assert(sizeof(PagedNodeRecordHeader) == kPagedNodeHeaderBytes, "paged node header must be 64B");
static_assert(sizeof(PagedIndexFileHeader) <= kPagedIndexPageBytes, "paged index header must fit one page");

size_t paged_node_record_bytes(size_t code_size, size_t neighbor_slots) {
    return round_up_to_alignment(
            kPagedNodeHeaderBytes + code_size +
                    neighbor_slots * sizeof(faiss::HNSW::storage_idx_t),
            64);
}

bool validate_paged_index_file(
        const std::string& path,
        const faiss::IndexBinaryHNSW& index,
        int m,
        PagedIndexFileHeader* header_out = nullptr) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    PagedIndexFileHeader header;
    const ssize_t got = ::pread(fd, &header, sizeof(header), 0);
    struct stat st {};
    const bool stat_ok = ::fstat(fd, &st) == 0 && st.st_size >= 0;
    ::close(fd);
    if (got != static_cast<ssize_t>(sizeof(header)) || !stat_ok) return false;
    if (header.magic != kPagedIndexMagic || header.version != 1 ||
        header.page_bytes != kPagedIndexPageBytes ||
        header.ntotal != static_cast<uint64_t>(index.ntotal) ||
        header.dim != static_cast<uint32_t>(index.d) ||
        header.code_size != static_cast<uint32_t>(index.code_size) ||
        header.m != static_cast<uint32_t>(m) ||
        header.entry_point != index.hnsw.entry_point ||
        header.max_level != index.hnsw.max_level ||
        header.neighbor_slots != index.hnsw.neighbors.size() ||
        header.locator_bytes != header.ntotal * sizeof(PagedNodeLocator) ||
        header.data_start_page < 1 || header.total_pages <= header.data_start_page ||
        static_cast<uint64_t>(st.st_size) != header.total_pages * kPagedIndexPageBytes) {
        return false;
    }
    if (header_out) *header_out = header;
    return true;
}

void write_paged_index_file(
        const std::string& path,
        const faiss::IndexBinaryHNSW& index,
        int m) {
    const auto* flat = dynamic_cast<const faiss::IndexBinaryFlat*>(index.storage);
    if (!flat) throw std::runtime_error("paged index requires IndexBinaryFlat storage");
    if (flat->xb.size() != static_cast<size_t>(index.ntotal) * index.code_size) {
        throw std::runtime_error("paged index BQ storage size mismatch");
    }

    PagedIndexFileHeader header;
    header.magic = kPagedIndexMagic;
    header.ntotal = static_cast<uint64_t>(index.ntotal);
    header.dim = static_cast<uint32_t>(index.d);
    header.code_size = static_cast<uint32_t>(index.code_size);
    header.m = static_cast<uint32_t>(m);
    header.entry_point = index.hnsw.entry_point;
    header.max_level = index.hnsw.max_level;
    header.neighbor_slots = index.hnsw.neighbors.size();
    header.locator_bytes = header.ntotal * sizeof(PagedNodeLocator);
    header.data_start_page = 1 +
            (header.locator_bytes + kPagedIndexPageBytes - 1) / kPagedIndexPageBytes;

    std::vector<PagedNodeLocator> locators(static_cast<size_t>(index.ntotal));
    uint64_t page_no = header.data_start_page;
    size_t page_offset = 0;
    for (size_t node = 0; node < static_cast<size_t>(index.ntotal); ++node) {
        const size_t neighbor_slots = index.hnsw.offsets[node + 1] - index.hnsw.offsets[node];
        const size_t record_bytes = paged_node_record_bytes(index.code_size, neighbor_slots);
        if (record_bytes > kPagedIndexPageBytes || record_bytes > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("HNSW node record does not fit the configured index page");
        }
        if (page_offset + record_bytes > kPagedIndexPageBytes) {
            ++page_no;
            page_offset = 0;
        }
        locators[node].page_no = static_cast<uint32_t>(page_no);
        locators[node].offset = static_cast<uint16_t>(page_offset);
        locators[node].record_bytes = static_cast<uint16_t>(record_bytes);
        page_offset += record_bytes;
    }
    header.total_pages = page_no + 1;

    const fs::path output_path(path);
    if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
    const std::string tmp_path = path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to create paged index: " + tmp_path);

    std::array<char, kPagedIndexPageBytes> page{};
    std::memcpy(page.data(), &header, sizeof(header));
    out.write(page.data(), static_cast<std::streamsize>(page.size()));
    out.write(
            reinterpret_cast<const char*>(locators.data()),
            static_cast<std::streamsize>(locators.size() * sizeof(PagedNodeLocator)));
    const uint64_t locator_end = kPagedIndexPageBytes + header.locator_bytes;
    const uint64_t data_start = header.data_start_page * kPagedIndexPageBytes;
    if (locator_end < data_start) {
        std::vector<char> padding(static_cast<size_t>(data_start - locator_end), 0);
        out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    }

    page.fill(0);
    uint32_t active_page = locators.empty()
            ? static_cast<uint32_t>(header.data_start_page)
            : locators[0].page_no;
    for (size_t node = 0; node < locators.size(); ++node) {
        const PagedNodeLocator locator = locators[node];
        if (locator.page_no != active_page) {
            out.write(page.data(), static_cast<std::streamsize>(page.size()));
            page.fill(0);
            active_page = locator.page_no;
        }
        PagedNodeRecordHeader record_header;
        record_header.levels = static_cast<uint16_t>(index.hnsw.levels[node]);
        record_header.neighbor_slots = static_cast<uint32_t>(
                index.hnsw.offsets[node + 1] - index.hnsw.offsets[node]);
        char* record = page.data() + locator.offset;
        std::memcpy(record, &record_header, sizeof(record_header));
        std::memcpy(
                record + kPagedNodeHeaderBytes,
                flat->xb.data() + node * index.code_size,
                index.code_size);
        std::memcpy(
                record + kPagedNodeHeaderBytes + index.code_size,
                index.hnsw.neighbors.data() + index.hnsw.offsets[node],
                static_cast<size_t>(record_header.neighbor_slots) *
                        sizeof(faiss::HNSW::storage_idx_t));
        if ((node + 1) % 100000 == 0 || node + 1 == locators.size()) {
            std::cout << "  wrote paged index nodes " << (node + 1) << "/"
                      << locators.size() << std::endl;
        }
    }
    if (!locators.empty()) out.write(page.data(), static_cast<std::streamsize>(page.size()));
    out.close();
    if (!out) throw std::runtime_error("failed to write paged index: " + tmp_path);
    fs::rename(tmp_path, path);
    std::cout << "Saved " << (kPagedIndexPageBytes / 1024)
              << "KB paged HNSW+BQ index: " << path
              << ", data_pages=" << (header.total_pages - header.data_start_page)
              << ", bytes=" << header.total_pages * kPagedIndexPageBytes << std::endl;
}

struct PagedIndexPrepareStats {
    bool loaded = false;
    double seconds = 0.0;
    uint64_t file_bytes = 0;
    uint64_t data_pages = 0;
};

PagedIndexPrepareStats prepare_paged_index_file(
        const std::string& path,
        const faiss::IndexBinaryHNSW& index,
        int m,
        bool rebuild) {
    const auto started = std::chrono::steady_clock::now();
    PagedIndexFileHeader header;
    PagedIndexPrepareStats stats;
    if (!rebuild && validate_paged_index_file(path, index, m, &header)) {
        stats.loaded = true;
        stats.seconds = seconds_since(started);
        stats.file_bytes = header.total_pages * kPagedIndexPageBytes;
        stats.data_pages = header.total_pages - header.data_start_page;
        std::cout << "Loaded " << (kPagedIndexPageBytes / 1024)
                  << "KB paged HNSW+BQ index: " << path << std::endl;
        return stats;
    }
    std::cout << "Building " << (kPagedIndexPageBytes / 1024)
              << "KB paged HNSW+BQ index: " << path << std::endl;
    write_paged_index_file(path, index, m);
    if (!validate_paged_index_file(path, index, m, &header)) {
        throw std::runtime_error("newly built paged index failed validation: " + path);
    }
    stats.seconds = seconds_since(started);
    stats.file_bytes = header.total_pages * kPagedIndexPageBytes;
    stats.data_pages = header.total_pages - header.data_start_page;
    return stats;
}

struct IndexPageCacheStats {
    uint64_t accesses = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t read_bytes = 0;
    uint64_t async_batches = 0;
    uint64_t async_requests = 0;
    uint64_t async_cached_nodes_scored_while_inflight = 0;
    uint64_t async_completed_nodes_scored_while_inflight = 0;
    uint64_t async_completions_ready_before_wait = 0;
    double async_wait_seconds = 0.0;
    uint64_t hybrid_sync_batches = 0;
    uint64_t hybrid_sync_requests = 0;
    double hybrid_sync_wait_seconds = 0.0;
    size_t resident_pages = 0;
    uint64_t data_pages = 0;
};

struct AlignedIndexPage {
    void* data = nullptr;

    AlignedIndexPage() {
        if (::posix_memalign(&data, kPagedIndexPageBytes, kPagedIndexPageBytes) != 0) {
            throw std::bad_alloc();
        }
    }

    ~AlignedIndexPage() {
        std::free(data);
    }

    AlignedIndexPage(const AlignedIndexPage&) = delete;
    AlignedIndexPage& operator=(const AlignedIndexPage&) = delete;
};

struct IndexUringThreadBuffer {
    io_uring ring{};
    bool initialized = false;
    unsigned depth = 0;

    IndexUringThreadBuffer() = default;
    IndexUringThreadBuffer(const IndexUringThreadBuffer&) = delete;
    IndexUringThreadBuffer& operator=(const IndexUringThreadBuffer&) = delete;

    ~IndexUringThreadBuffer() {
        if (initialized) io_uring_queue_exit(&ring);
    }

    io_uring& ensure(unsigned requested_depth) {
        requested_depth = std::max(2u, requested_depth);
        if (initialized && depth >= requested_depth) return ring;
        if (initialized) {
            io_uring_queue_exit(&ring);
            ring = {};
            initialized = false;
            depth = 0;
        }
        const int rc = io_uring_queue_init(requested_depth, &ring, 0);
        if (rc < 0) {
            throw std::runtime_error(
                    std::string("graph-page io_uring_queue_init failed: ") +
                    std::strerror(-rc));
        }
        initialized = true;
        depth = requested_depth;
        return ring;
    }
};

class PagedIndexPageCache {
    struct CacheEntry {
        std::shared_ptr<AlignedIndexPage> page;
        std::list<uint32_t>::iterator lru_position;
    };

    int fd_ = -1;
    size_t capacity_pages_ = 0;
    mutable std::mutex mutex_;
    std::unordered_map<uint32_t, CacheEntry> pages_;
    std::list<uint32_t> lru_;
    IndexPageCacheStats stats_;

public:
    PagedIndexPageCache(const std::string& path, size_t cache_mib, uint64_t data_pages) {
        capacity_pages_ = std::max<size_t>(
                1,
                (cache_mib * 1024ull * 1024ull) / kPagedIndexPageBytes);
        pages_.reserve(capacity_pages_ + capacity_pages_ / 4);
        stats_.data_pages = data_pages;
        fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (fd_ < 0) {
            throw std::runtime_error("failed to open paged index with O_DIRECT: " + path + ": " + std::strerror(errno));
        }
    }

    ~PagedIndexPageCache() {
        if (fd_ >= 0) ::close(fd_);
    }

    int fd() const { return fd_; }

    std::shared_ptr<AlignedIndexPage> lookup(uint32_t page_no) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.accesses;
        auto found = pages_.find(page_no);
        if (found != pages_.end()) {
            ++stats_.hits;
            lru_.splice(lru_.begin(), lru_, found->second.lru_position);
            return found->second.page;
        }
        ++stats_.misses;
        return {};
    }

    void lookup_batch(
            const uint32_t* page_nos,
            size_t count,
            std::shared_ptr<AlignedIndexPage>* pages,
            uint64_t coalesced_hits) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.accesses += coalesced_hits;
        stats_.hits += coalesced_hits;
        for (size_t i = 0; i < count; ++i) {
            ++stats_.accesses;
            auto found = pages_.find(page_nos[i]);
            if (found == pages_.end()) {
                ++stats_.misses;
                pages[i].reset();
                continue;
            }
            ++stats_.hits;
            lru_.splice(lru_.begin(), lru_, found->second.lru_position);
            pages[i] = found->second.page;
        }
    }

    void record_coalesced_hits(uint64_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.accesses += count;
        stats_.hits += count;
    }

    std::shared_ptr<AlignedIndexPage> install(
            uint32_t page_no,
            std::shared_ptr<AlignedIndexPage> page) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.read_bytes += kPagedIndexPageBytes;
        auto found = pages_.find(page_no);
        if (found != pages_.end()) {
            lru_.splice(lru_.begin(), lru_, found->second.lru_position);
            return found->second.page;
        }

        if (pages_.size() >= capacity_pages_) {
            const uint32_t victim = lru_.back();
            lru_.pop_back();
            pages_.erase(victim);
            ++stats_.evictions;
        }
        lru_.push_front(page_no);
        pages_.emplace(page_no, CacheEntry{page, lru_.begin()});
        return page;
    }

    void record_async_batch(
            uint64_t requests,
            uint64_t cached_nodes_scored_while_inflight,
            uint64_t completed_nodes_scored_while_inflight,
            uint64_t completions_ready_before_wait,
            double wait_seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.async_batches;
        stats_.async_requests += requests;
        stats_.async_cached_nodes_scored_while_inflight +=
                cached_nodes_scored_while_inflight;
        stats_.async_completed_nodes_scored_while_inflight +=
                completed_nodes_scored_while_inflight;
        stats_.async_completions_ready_before_wait +=
                completions_ready_before_wait;
        stats_.async_wait_seconds += wait_seconds;
    }

    void record_hybrid_sync_batch(uint64_t requests, double wait_seconds) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.hybrid_sync_batches;
        stats_.hybrid_sync_requests += requests;
        stats_.hybrid_sync_wait_seconds += wait_seconds;
    }

    std::shared_ptr<AlignedIndexPage> get(uint32_t page_no) {
        if (auto cached = lookup(page_no)) return cached;

        // Do not hold the global LRU mutex across O_DIRECT.  At T>1 this lets
        // independent query workers overlap graph-page reads.  A concurrent
        // fill of the same page may cause a redundant physical read; the
        // second lookup below keeps a single cached copy while read_bytes
        // still reports every request actually submitted to the device.
        auto page = std::make_shared<AlignedIndexPage>();
        size_t done = 0;
        const off_t offset = static_cast<off_t>(
                static_cast<uint64_t>(page_no) * kPagedIndexPageBytes);
        while (done < kPagedIndexPageBytes) {
            const ssize_t got = ::pread(
                    fd_,
                    static_cast<char*>(page->data) + done,
                    kPagedIndexPageBytes - done,
                    offset + static_cast<off_t>(done));
            if (got < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("paged index O_DIRECT read failed: ") + std::strerror(errno));
            }
            if (got == 0) throw std::runtime_error("short paged index read");
            done += static_cast<size_t>(got);
        }
        return install(page_no, std::move(page));
    }

    IndexPageCacheStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        IndexPageCacheStats copy = stats_;
        copy.resident_pages = pages_.size();
        return copy;
    }
};

struct PagedNodeView {
    std::shared_ptr<AlignedIndexPage> owner;
    const PagedNodeRecordHeader* header = nullptr;
    const uint8_t* code = nullptr;
    const faiss::HNSW::storage_idx_t* neighbors = nullptr;
};

class PagedHnswIndex {
    std::string path_;
    PagedIndexFileHeader header_;
    std::vector<PagedNodeLocator> locators_;
    std::unique_ptr<PagedIndexPageCache> cache_;

public:
    PagedHnswIndex(const std::string& path, size_t cache_mib) : path_(path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("failed to open paged index metadata: " + path);
        if (::pread(fd, &header_, sizeof(header_), 0) != static_cast<ssize_t>(sizeof(header_)) ||
            header_.magic != kPagedIndexMagic || header_.page_bytes != kPagedIndexPageBytes) {
            ::close(fd);
            throw std::runtime_error("invalid paged index header: " + path);
        }
        locators_.resize(static_cast<size_t>(header_.ntotal));
        const size_t locator_bytes = locators_.size() * sizeof(PagedNodeLocator);
        if (::pread(
                    fd,
                    locators_.data(),
                    locator_bytes,
                    static_cast<off_t>(kPagedIndexPageBytes)) !=
            static_cast<ssize_t>(locator_bytes)) {
            ::close(fd);
            throw std::runtime_error("failed to read paged index locators: " + path);
        }
        ::close(fd);
        cache_ = std::make_unique<PagedIndexPageCache>(
                path,
                cache_mib,
                header_.total_pages - header_.data_start_page);
    }

    size_t size() const { return static_cast<size_t>(header_.ntotal); }
    int dim() const { return static_cast<int>(header_.dim); }
    size_t code_size() const { return static_cast<size_t>(header_.code_size); }
    int m() const { return static_cast<int>(header_.m); }
    faiss::HNSW::storage_idx_t entry_point() const { return header_.entry_point; }
    int max_level() const { return header_.max_level; }

    const PagedNodeLocator& locator(size_t id) const {
        if (id >= locators_.size()) {
            throw std::runtime_error("paged index node id out of range");
        }
        return locators_[id];
    }

    std::shared_ptr<AlignedIndexPage> lookup_page(uint32_t page_no) {
        return cache_->lookup(page_no);
    }

    void lookup_pages_batch(
            const uint32_t* page_nos,
            size_t count,
            std::shared_ptr<AlignedIndexPage>* pages,
            uint64_t coalesced_hits) {
        cache_->lookup_batch(page_nos, count, pages, coalesced_hits);
    }

    void record_coalesced_page_hits(uint64_t count) {
        cache_->record_coalesced_hits(count);
    }

    std::shared_ptr<AlignedIndexPage> install_page(
            uint32_t page_no,
            std::shared_ptr<AlignedIndexPage> page) {
        return cache_->install(page_no, std::move(page));
    }

    int page_fd() const { return cache_->fd(); }

    void record_async_batch(
            uint64_t requests,
            uint64_t cached_nodes_scored_while_inflight,
            uint64_t completed_nodes_scored_while_inflight,
            uint64_t completions_ready_before_wait,
            double wait_seconds) {
        cache_->record_async_batch(
                requests,
                cached_nodes_scored_while_inflight,
                completed_nodes_scored_while_inflight,
                completions_ready_before_wait,
                wait_seconds);
    }

    void record_hybrid_sync_batch(uint64_t requests, double wait_seconds) {
        cache_->record_hybrid_sync_batch(requests, wait_seconds);
    }

    PagedNodeView node_from_page(
            size_t id,
            std::shared_ptr<AlignedIndexPage> page) const {
        PagedNodeView view = node_from_page_borrowed(id, page.get());
        view.owner = std::move(page);
        return view;
    }

    PagedNodeView node_from_page_borrowed(
            size_t id,
            const AlignedIndexPage* page) const {
        if (page == nullptr) {
            throw std::runtime_error("null paged-index page");
        }
        const PagedNodeLocator& location = locator(id);
        const char* record =
                static_cast<const char*>(page->data) + location.offset;
        const auto* record_header =
                reinterpret_cast<const PagedNodeRecordHeader*>(record);
        const size_t minimum_bytes = kPagedNodeHeaderBytes + code_size() +
                static_cast<size_t>(record_header->neighbor_slots) *
                        sizeof(faiss::HNSW::storage_idx_t);
        if (minimum_bytes > location.record_bytes ||
            static_cast<size_t>(location.offset) + location.record_bytes >
                    kPagedIndexPageBytes) {
            throw std::runtime_error("corrupt paged index node record");
        }
        PagedNodeView view;
        view.header = record_header;
        view.code = reinterpret_cast<const uint8_t*>(
                record + kPagedNodeHeaderBytes);
        view.neighbors =
                reinterpret_cast<const faiss::HNSW::storage_idx_t*>(
                        record + kPagedNodeHeaderBytes + code_size());
        return view;
    }

    PagedNodeView node(size_t id) {
        const PagedNodeLocator& location = locator(id);
        return node_from_page(id, cache_->get(location.page_no));
    }

    std::pair<size_t, size_t> neighbor_range(const PagedNodeView& node, int level) const {
        if (level < 0 || level >= node.header->levels) return {0, 0};
        const size_t begin = level == 0
                ? 0
                : static_cast<size_t>(2 * m() + (level - 1) * m());
        const size_t end = begin + static_cast<size_t>(level == 0 ? 2 * m() : m());
        if (end > node.header->neighbor_slots) {
            throw std::runtime_error("corrupt paged index neighbor range");
        }
        return {begin, end};
    }

    IndexPageCacheStats cache_stats() const { return cache_->stats(); }
};

constexpr size_t kMmapIndexHeaderBytes = 4096;
constexpr std::array<char, 8> kMmapIndexMagic = {'A', 'D', 'C', 'M', 'M', '0', '1', '\0'};

struct MmapIndexFileHeader {
    std::array<char, 8> magic{};
    uint32_t version = 1;
    uint32_t header_bytes = kMmapIndexHeaderBytes;
    uint64_t ntotal = 0;
    uint32_t dim = 0;
    uint32_t code_size = 0;
    uint32_t m = 0;
    uint32_t colocation_iterations = 0;
    uint32_t colocation_partition_size = 0;
    int32_t entry_point = -1;
    int32_t max_level = -1;
    uint64_t neighbor_slots = 0;
    uint64_t locator_offset = kMmapIndexHeaderBytes;
    uint64_t locator_bytes = 0;
    uint64_t data_offset = 0;
    uint64_t total_bytes = 0;
};

static_assert(sizeof(MmapIndexFileHeader) <= kMmapIndexHeaderBytes,
              "mmap index header must fit its fixed header area");

bool validate_mmap_index_file(
        const std::string& path,
        int32_t n,
        int32_t d,
        int m,
        int colocation_iterations,
        int colocation_partition_size,
        MmapIndexFileHeader* header_out = nullptr) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    MmapIndexFileHeader header;
    const ssize_t got = ::pread(fd, &header, sizeof(header), 0);
    struct stat st {};
    const bool stat_ok = ::fstat(fd, &st) == 0 && st.st_size >= 0;
    ::close(fd);
    if (got != static_cast<ssize_t>(sizeof(header)) || !stat_ok) return false;
    const uint64_t expected_locator_bytes =
            (static_cast<uint64_t>(n) + 1) * sizeof(uint64_t);
    if (header.magic != kMmapIndexMagic || header.version != 1 ||
        header.header_bytes != kMmapIndexHeaderBytes ||
        header.ntotal != static_cast<uint64_t>(n) ||
        header.dim != static_cast<uint32_t>(d) ||
        header.code_size != static_cast<uint32_t>(d / 8) ||
        header.m != static_cast<uint32_t>(m) ||
        header.colocation_iterations != static_cast<uint32_t>(colocation_iterations) ||
        header.colocation_partition_size != static_cast<uint32_t>(colocation_partition_size) ||
        header.locator_offset != kMmapIndexHeaderBytes ||
        header.locator_bytes != expected_locator_bytes ||
        header.data_offset < header.locator_offset + header.locator_bytes ||
        header.data_offset % 4096 != 0 ||
        header.total_bytes <= header.data_offset ||
        static_cast<uint64_t>(st.st_size) != header.total_bytes) {
        return false;
    }
    if (header_out) *header_out = header;
    return true;
}

void write_mmap_index_file(
        const std::string& path,
        const faiss::IndexBinaryHNSW& index,
        int m,
        int colocation_iterations,
        int colocation_partition_size) {
    const auto* flat = dynamic_cast<const faiss::IndexBinaryFlat*>(index.storage);
    if (!flat) throw std::runtime_error("mmap index requires IndexBinaryFlat storage");
    if (flat->xb.size() != static_cast<size_t>(index.ntotal) * index.code_size) {
        throw std::runtime_error("mmap index BQ storage size mismatch");
    }

    MmapIndexFileHeader header;
    header.magic = kMmapIndexMagic;
    header.ntotal = static_cast<uint64_t>(index.ntotal);
    header.dim = static_cast<uint32_t>(index.d);
    header.code_size = static_cast<uint32_t>(index.code_size);
    header.m = static_cast<uint32_t>(m);
    header.colocation_iterations = static_cast<uint32_t>(colocation_iterations);
    header.colocation_partition_size = static_cast<uint32_t>(colocation_partition_size);
    header.entry_point = index.hnsw.entry_point;
    header.max_level = index.hnsw.max_level;
    header.neighbor_slots = index.hnsw.neighbors.size();
    header.locator_bytes = (header.ntotal + 1) * sizeof(uint64_t);
    header.data_offset = round_up_to_alignment(
            static_cast<size_t>(header.locator_offset + header.locator_bytes),
            static_cast<size_t>(4096));

    std::vector<uint64_t> offsets(static_cast<size_t>(index.ntotal) + 1);
    uint64_t cursor = header.data_offset;
    for (size_t node = 0; node < static_cast<size_t>(index.ntotal); ++node) {
        offsets[node] = cursor;
        const size_t neighbor_slots = index.hnsw.offsets[node + 1] - index.hnsw.offsets[node];
        cursor += paged_node_record_bytes(index.code_size, neighbor_slots);
    }
    offsets.back() = cursor;
    header.total_bytes = round_up_to_alignment(static_cast<size_t>(cursor), static_cast<size_t>(4096));

    const fs::path output_path(path);
    if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
    const std::string tmp_path = path + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    std::vector<char> stream_buffer(8 * 1024 * 1024);
    std::ofstream out;
    out.rdbuf()->pubsetbuf(stream_buffer.data(), static_cast<std::streamsize>(stream_buffer.size()));
    out.open(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("failed to create mmap index: " + tmp_path);

    std::array<char, kMmapIndexHeaderBytes> header_area{};
    std::memcpy(header_area.data(), &header, sizeof(header));
    out.write(header_area.data(), static_cast<std::streamsize>(header_area.size()));
    out.write(
            reinterpret_cast<const char*>(offsets.data()),
            static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
    const uint64_t locator_end = header.locator_offset + header.locator_bytes;
    if (locator_end < header.data_offset) {
        std::vector<char> padding(static_cast<size_t>(header.data_offset - locator_end), 0);
        out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    }

    const std::array<char, 64> zero_padding{};
    for (size_t node = 0; node < static_cast<size_t>(index.ntotal); ++node) {
        PagedNodeRecordHeader record_header;
        record_header.levels = static_cast<uint16_t>(index.hnsw.levels[node]);
        record_header.neighbor_slots = static_cast<uint32_t>(
                index.hnsw.offsets[node + 1] - index.hnsw.offsets[node]);
        out.write(
                reinterpret_cast<const char*>(&record_header),
                static_cast<std::streamsize>(sizeof(record_header)));
        out.write(
                reinterpret_cast<const char*>(flat->xb.data() + node * index.code_size),
                static_cast<std::streamsize>(index.code_size));
        const size_t neighbor_bytes = static_cast<size_t>(record_header.neighbor_slots) *
                sizeof(faiss::HNSW::storage_idx_t);
        out.write(
                reinterpret_cast<const char*>(
                        index.hnsw.neighbors.data() + index.hnsw.offsets[node]),
                static_cast<std::streamsize>(neighbor_bytes));
        const size_t payload_bytes = sizeof(record_header) + index.code_size + neighbor_bytes;
        const size_t record_bytes = static_cast<size_t>(offsets[node + 1] - offsets[node]);
        const size_t padding_bytes = record_bytes - payload_bytes;
        if (padding_bytes > 0) {
            out.write(zero_padding.data(), static_cast<std::streamsize>(padding_bytes));
        }
        if ((node + 1) % 100000 == 0 || node + 1 == static_cast<size_t>(index.ntotal)) {
            std::cout << "  wrote mmap index nodes " << (node + 1) << "/"
                      << index.ntotal << std::endl;
        }
    }
    const uint64_t trailing_bytes = header.total_bytes - cursor;
    if (trailing_bytes > 0) {
        std::vector<char> padding(static_cast<size_t>(trailing_bytes), 0);
        out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
    }
    out.close();
    if (!out) throw std::runtime_error("failed to write mmap index: " + tmp_path);
    fs::rename(tmp_path, path);
    std::cout << "Saved native mmap HNSW+BQ index: " << path
              << ", bytes=" << header.total_bytes << std::endl;
}

void prepare_mmap_index_file(
        const std::string& path,
        const faiss::IndexBinaryHNSW& index,
        int m,
        int colocation_iterations,
        int colocation_partition_size,
        bool rebuild) {
    MmapIndexFileHeader header;
    if (!rebuild && validate_mmap_index_file(
                            path,
                            static_cast<int32_t>(index.ntotal),
                            index.d,
                            m,
                            colocation_iterations,
                            colocation_partition_size,
                            &header) &&
        header.entry_point == index.hnsw.entry_point &&
        header.max_level == index.hnsw.max_level &&
        header.neighbor_slots == index.hnsw.neighbors.size()) {
        std::cout << "Loaded native mmap HNSW+BQ index: " << path << std::endl;
        return;
    }
    std::cout << "Building native mmap HNSW+BQ index: " << path << std::endl;
    write_mmap_index_file(
            path,
            index,
            m,
            colocation_iterations,
            colocation_partition_size);
}

struct MmapNodeView {
    const PagedNodeRecordHeader* header = nullptr;
    const uint8_t* code = nullptr;
    const faiss::HNSW::storage_idx_t* neighbors = nullptr;
};

class MmapHnswIndex {
    std::string path_;
    int fd_ = -1;
    void* mapping_ = MAP_FAILED;
    size_t mapping_size_ = 0;
    const MmapIndexFileHeader* header_ = nullptr;
    const uint64_t* offsets_ = nullptr;
    std::vector<const char*> record_ptrs_;
    std::vector<const uint8_t*> code_ptrs_;
    size_t os_page_bytes_ = 4096;

public:
    MmapHnswIndex(
            const std::string& path,
            int32_t n,
            int32_t d,
            int m,
            int colocation_iterations,
            int colocation_partition_size)
            : path_(path) {
        MmapIndexFileHeader validated;
        if (!validate_mmap_index_file(
                    path,
                    n,
                    d,
                    m,
                    colocation_iterations,
                    colocation_partition_size,
                    &validated)) {
            throw std::runtime_error("invalid native mmap index: " + path);
        }
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open native mmap index: " + path);
        mapping_size_ = static_cast<size_t>(validated.total_bytes);
        mapping_ = ::mmap(nullptr, mapping_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (mapping_ == MAP_FAILED) {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("failed to mmap native index: " + path + ": " + std::strerror(errno));
        }
        header_ = static_cast<const MmapIndexFileHeader*>(mapping_);
        offsets_ = reinterpret_cast<const uint64_t*>(
                static_cast<const char*>(mapping_) + header_->locator_offset);
        record_ptrs_.resize(static_cast<size_t>(header_->ntotal));
        code_ptrs_.resize(static_cast<size_t>(header_->ntotal));
        for (size_t id = 0; id < record_ptrs_.size(); ++id) {
            record_ptrs_[id] = static_cast<const char*>(mapping_) + offsets_[id];
            code_ptrs_[id] = reinterpret_cast<const uint8_t*>(
                    record_ptrs_[id] + kPagedNodeHeaderBytes);
        }
        const long page_size = ::sysconf(_SC_PAGESIZE);
        if (page_size > 0) os_page_bytes_ = static_cast<size_t>(page_size);
        ::madvise(mapping_, mapping_size_, MADV_RANDOM);
    }

    ~MmapHnswIndex() {
        if (mapping_ != MAP_FAILED) ::munmap(mapping_, mapping_size_);
        if (fd_ >= 0) ::close(fd_);
    }

    MmapHnswIndex(const MmapHnswIndex&) = delete;
    MmapHnswIndex& operator=(const MmapHnswIndex&) = delete;

    size_t size() const { return static_cast<size_t>(header_->ntotal); }
    int dim() const { return static_cast<int>(header_->dim); }
    size_t code_size() const { return static_cast<size_t>(header_->code_size); }
    int m() const { return static_cast<int>(header_->m); }
    faiss::HNSW::storage_idx_t entry_point() const { return header_->entry_point; }
    int max_level() const { return header_->max_level; }
    size_t file_bytes() const { return mapping_size_; }
    size_t os_page_bytes() const { return os_page_bytes_; }
    const std::string& path() const { return path_; }

    const char* record_ptr(size_t id) const {
        if (id >= size()) throw std::runtime_error("native mmap index node id out of range");
        return record_ptrs_[id];
    }

    const uint8_t* code_ptr(size_t id) const {
        if (id >= size()) throw std::runtime_error("native mmap index node id out of range");
        return code_ptrs_[id];
    }

    const uint8_t* const* code_pointer_table() const { return code_ptrs_.data(); }

    MmapNodeView node(size_t id) const {
        const char* record = record_ptr(id);
        const auto* record_header = reinterpret_cast<const PagedNodeRecordHeader*>(record);
        const size_t expected_neighbor_slots = record_header->levels == 0
                ? 0
                : static_cast<size_t>(2 * m() + (record_header->levels - 1) * m());
        if (record_header->neighbor_slots != expected_neighbor_slots) {
            throw std::runtime_error("corrupt native mmap index node record");
        }
        MmapNodeView view;
        view.header = record_header;
        view.code = reinterpret_cast<const uint8_t*>(record + kPagedNodeHeaderBytes);
        view.neighbors = reinterpret_cast<const faiss::HNSW::storage_idx_t*>(
                record + kPagedNodeHeaderBytes + code_size());
        return view;
    }

    std::pair<size_t, size_t> neighbor_range(const MmapNodeView& node, int level) const {
        if (level < 0 || level >= node.header->levels) return {0, 0};
        const size_t begin = level == 0
                ? 0
                : static_cast<size_t>(2 * m() + (level - 1) * m());
        const size_t end = begin + static_cast<size_t>(level == 0 ? 2 * m() : m());
        if (end > node.header->neighbor_slots) {
            throw std::runtime_error("corrupt native mmap index neighbor range");
        }
        return {begin, end};
    }

    void prefetch(size_t id) const {
        if (id < size()) __builtin_prefetch(record_ptr(id), 0, 1);
    }

    size_t resident_pages() const {
        const size_t pages = (mapping_size_ + os_page_bytes_ - 1) / os_page_bytes_;
        std::vector<unsigned char> residency(pages);
        if (::mincore(mapping_, mapping_size_, residency.data()) != 0) {
            throw std::runtime_error("mincore failed for native mmap index: " +
                                     std::string(std::strerror(errno)));
        }
        return static_cast<size_t>(std::count_if(
                residency.begin(), residency.end(), [](unsigned char value) {
                    return (value & 1u) != 0;
                }));
    }

    void drop_file_cache() {
        if (::madvise(mapping_, mapping_size_, MADV_DONTNEED) != 0) {
            throw std::runtime_error("MADV_DONTNEED failed for native mmap index: " +
                                     std::string(std::strerror(errno)));
        }
        const int advice_result = ::posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
        if (advice_result != 0) {
            throw std::runtime_error("POSIX_FADV_DONTNEED failed for native mmap index: " +
                                     std::string(std::strerror(advice_result)));
        }
        ::madvise(mapping_, mapping_size_, MADV_RANDOM);
    }
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

struct AdcSearchStats {
    uint64_t adaptive_effective_ef = 0;
    uint64_t adaptive_shrunk = 0;
    uint64_t adaptive_early_stopped = 0;
    uint64_t adaptive_classified = 0;
    // DiskANN's `hops` counts expanded adjacency lists and `cmps` counts
    // routing-distance evaluations.  Keep the same totals here so the two
    // implementations can be matched by actual search work rather than by
    // the numeric efSearch/L knobs.
    uint64_t upper_hop_count = 0;
    uint64_t upper_distance_eval_count = 0;
    uint64_t base_pop_count = 0;
    uint64_t base_expand_count = 0;
    uint64_t base_neighbor_scan_count = 0;
    uint64_t base_visited_count = 0;
    uint64_t base_duplicate_visit_count = 0;
    uint64_t base_candidate_insert_count = 0;
    uint64_t base_result_insert_count = 0;
    uint64_t base_distance_eval_count = 0;
    uint64_t nhops = 0;
    uint64_t ndis = 0;
};

struct AdcAdaptiveCfrConfig {
    bool enabled = false;
    bool paper_bucket_mode = false;
    float tau = 0.6f;
    int tmin_pops = 25;
    int bucket_count = 4;
    std::vector<float> gammas;
};

inline int resolve_adcbq_cfr_bucket(const AdcAdaptiveCfrConfig& config, float ratio) {
    const int bucket_count = std::max(2, config.bucket_count);
    for (int i = 0; i + 1 < bucket_count; ++i) {
        if (ratio <= config.gammas[static_cast<size_t>(i)]) return i;
    }
    return bucket_count - 1;
}

inline int resolve_adcbq_cfr_ef(int configured_ef, int k, const AdcAdaptiveCfrConfig& config, float classify_ratio) {
    if (config.paper_bucket_mode) {
        const int bucket = resolve_adcbq_cfr_bucket(config, classify_ratio);
        if (bucket + 1 >= config.bucket_count) return configured_ef;
        return std::max(k, std::max(1, configured_ef * (bucket + 1) / config.bucket_count));
    }
    return std::max(k, configured_ef / 2);
}

inline bool minimax_push_finite(
        faiss::MinimaxHeap& heap,
        faiss::MinimaxHeap::storage_idx_t i,
        float v) {
    using HC = faiss::MinimaxHeap::HC;
    if (heap.k == heap.n) {
        if (v >= heap.dis[0]) {
            return false;
        }
        if (heap.ids[0] != -1) {
            --heap.nvalid;
        }
        faiss::heap_pop<HC>(heap.k--, heap.dis.data(), heap.ids.data());
    }
    faiss::heap_push<HC>(++heap.k, heap.dis.data(), heap.ids.data(), v, i);
    ++heap.nvalid;
    return true;
}

inline int minimax_count_below_simd(const faiss::MinimaxHeap& heap, float thresh) {
    const float* dis = heap.dis.data();
    const int n = heap.k;
    int count = 0;
    int i = 0;
#if defined(__AVX512F__)
    const __m512 t = _mm512_set1_ps(thresh);
    for (; i + 16 <= n; i += 16) {
        const __mmask16 mask = _mm512_cmp_ps_mask(_mm512_loadu_ps(dis + i), t, _CMP_LT_OQ);
        count += __builtin_popcount(static_cast<unsigned>(mask));
    }
#elif defined(__AVX2__)
    const __m256 t = _mm256_set1_ps(thresh);
    for (; i + 8 <= n; i += 8) {
        const __m256 cmp = _mm256_cmp_ps(_mm256_loadu_ps(dis + i), t, _CMP_LT_OQ);
        count += __builtin_popcount(static_cast<unsigned>(_mm256_movemask_ps(cmp)));
    }
#endif
    for (; i < n; ++i) {
        count += dis[i] < thresh ? 1 : 0;
    }
    return count;
}



struct HnswNeighborSpan {
    std::shared_ptr<AlignedIndexPage> owner;
    const faiss::HNSW::storage_idx_t* data = nullptr;
    size_t size = 0;
};

class RamHnswGraphAccessor {
    const faiss::HNSW& hnsw_;

public:
    explicit RamHnswGraphAccessor(const faiss::HNSW& hnsw) : hnsw_(hnsw) {}
    faiss::HNSW::storage_idx_t entry_point() const { return hnsw_.entry_point; }
    int max_level() const { return hnsw_.max_level; }

    HnswNeighborSpan neighbors(faiss::HNSW::storage_idx_t node, int level) {
        size_t begin = 0;
        size_t end = 0;
        hnsw_.neighbor_range(node, level, &begin, &end);
        return HnswNeighborSpan{
                {},
                hnsw_.neighbors.data() + begin,
                end - begin};
    }
};

class PagedHnswGraphAccessor {
    PagedHnswIndex& index_;

public:
    explicit PagedHnswGraphAccessor(PagedHnswIndex& index) : index_(index) {}
    faiss::HNSW::storage_idx_t entry_point() const { return index_.entry_point(); }
    int max_level() const { return index_.max_level(); }

    HnswNeighborSpan neighbors(faiss::HNSW::storage_idx_t node, int level) {
        PagedNodeView view = index_.node(static_cast<size_t>(node));
        const auto [begin, end] = index_.neighbor_range(view, level);
        HnswNeighborSpan span;
        span.owner = std::move(view.owner);
        span.data = view.neighbors + begin;
        span.size = end - begin;
        return span;
    }
};

class MmapHnswGraphAccessor {
    const MmapHnswIndex& index_;

public:
    explicit MmapHnswGraphAccessor(const MmapHnswIndex& index) : index_(index) {}
    faiss::HNSW::storage_idx_t entry_point() const { return index_.entry_point(); }
    int max_level() const { return index_.max_level(); }

    HnswNeighborSpan neighbors(faiss::HNSW::storage_idx_t node, int level) {
        MmapNodeView view = index_.node(static_cast<size_t>(node));
        const auto [begin, end] = index_.neighbor_range(view, level);
        return HnswNeighborSpan{{}, view.neighbors + begin, end - begin};
    }
};

template <typename GraphAccessorT, typename DistanceComputerT>
void greedy_update_nearest_packed(
        GraphAccessorT& graph,
        DistanceComputerT& dis,
        int level,
        faiss::HNSW::storage_idx_t& nearest,
        float& d_nearest,
        AdcSearchStats* stats) {
    for (;;) {
        if (stats) stats->upper_hop_count++;
        const faiss::HNSW::storage_idx_t prev_nearest = nearest;
        HnswNeighborSpan neighbor_span = graph.neighbors(nearest, level);

        int n_buffered = 0;
        faiss::idx_t buffered_ids[16];
        auto update_with_candidate = [&](faiss::idx_t idx, float cand_dis) {
            if (cand_dis < d_nearest) {
                nearest = static_cast<faiss::HNSW::storage_idx_t>(idx);
                d_nearest = cand_dis;
            }
        };

        for (size_t j = 0; j < neighbor_span.size; ++j) {
            const faiss::HNSW::storage_idx_t v = neighbor_span.data[j];
            if (v < 0) break;
            if (stats) stats->upper_distance_eval_count++;
            buffered_ids[n_buffered++] = v;
            if (n_buffered == 16) {
                float d[16];
                dis.distances_batch_16(buffered_ids, d);
                for (int bi = 0; bi < 16; ++bi) {
                    update_with_candidate(buffered_ids[bi], d[bi]);
                }
                n_buffered = 0;
            }
        }
        int tail = 0;
        for (; tail + 8 <= n_buffered; tail += 8) {
            float d[8];
            dis.distances_batch_8(buffered_ids + tail, d);
            for (int bi = 0; bi < 8; ++bi) {
                update_with_candidate(buffered_ids[tail + bi], d[bi]);
            }
        }
        for (; tail < n_buffered; ++tail) {
            update_with_candidate(buffered_ids[tail], dis.distance_idx(buffered_ids[tail]));
        }
        if (nearest == prev_nearest) return;
    }
}

struct CalibratedThresholdOutput {
    int anchor_k = 0;
    float distance_width = 0.0f;
    float* anchor_heap = nullptr;
    int* selected_count = nullptr;
};

// Keep only the candidates inside the calibrated distance threshold without
// sorting the complete efSearch result heap. The HNSW result arrays are a
// max-heap at this point. A small max-heap finds the anchor_k-th distance in
// O(ef log(anchor_k)); a second linear pass compacts matching candidates.
// This preserves the selected candidate set while avoiding heap_reorder's
// O(ef log(ef)) full ordering cost.
void compact_calibrated_threshold_results(
        int result_k,
        float* heap_dis,
        faiss::idx_t* heap_ids,
        CalibratedThresholdOutput& output) {
    if (output.anchor_k <= 0 || output.anchor_k > result_k ||
        !output.anchor_heap || !output.selected_count) {
        throw std::runtime_error("invalid calibrated threshold output state");
    }

    std::fill(
            output.anchor_heap,
            output.anchor_heap + output.anchor_k,
            std::numeric_limits<float>::infinity());
    std::make_heap(output.anchor_heap, output.anchor_heap + output.anchor_k);

    int valid = 0;
    for (int i = 0; i < result_k; ++i) {
        if (heap_ids[i] < 0 || !std::isfinite(heap_dis[i])) continue;
        ++valid;
        if (heap_dis[i] < output.anchor_heap[0]) {
            std::pop_heap(output.anchor_heap, output.anchor_heap + output.anchor_k);
            output.anchor_heap[output.anchor_k - 1] = heap_dis[i];
            std::push_heap(output.anchor_heap, output.anchor_heap + output.anchor_k);
        }
    }

    const float cutoff = valid <= output.anchor_k
            ? std::numeric_limits<float>::infinity()
            : output.anchor_heap[0] + output.distance_width;
    int write = 0;
    for (int read = 0; read < result_k; ++read) {
        const faiss::idx_t id = heap_ids[read];
        const float distance = heap_dis[read];
        if (id < 0 || !std::isfinite(distance) || distance > cutoff) continue;
        heap_ids[write] = id;
        heap_dis[write] = distance;
        ++write;
    }
    std::fill(heap_ids + write, heap_ids + result_k, faiss::idx_t{-1});
    std::fill(
            heap_dis + write,
            heap_dis + result_k,
            std::numeric_limits<float>::infinity());
    *output.selected_count = write;
}

// inline search path 추가
template <typename GraphAccessorT, typename DistanceComputerT>
void search_one_packed_inline_graph(
        GraphAccessorT& graph,
        DistanceComputerT& dis,
        faiss::VisitedTable& vt,
        int k,
        int candidate_ef,
        int stop_ef,
        bool check_relative_distance,
        bool early_abandon,
        int prefetch_lookahead,
        const AdcAdaptiveCfrConfig* adaptive_cfr,
        faiss::MinimaxHeap& candidates,
        float* heap_dis,
        faiss::idx_t* heap_ids,
        AdcSearchStats* stats,
        CalibratedThresholdOutput* calibrated_output = nullptr) {
    using C = faiss::HNSW::C;
    faiss::heap_heapify<C>(k, heap_dis, heap_ids);

    faiss::HNSW::storage_idx_t nearest = graph.entry_point();
    if (nearest < 0) {
        if (calibrated_output) {
            compact_calibrated_threshold_results(
                    k, heap_dis, heap_ids, *calibrated_output);
        } else {
            faiss::heap_reorder<C>(k, heap_dis, heap_ids);
        }
        vt.advance();
        return;
    }

    float d_nearest = dis.distance_idx(nearest);
    for (int level = graph.max_level(); level >= 1; --level) {
        greedy_update_nearest_packed(graph, dis, level, nearest, d_nearest, stats);
    }

    const bool adaptive_enabled = adaptive_cfr != nullptr && adaptive_cfr->enabled;
    const int configured_stop_ef = std::max(stop_ef, k);
    int effective_stop_ef = adaptive_enabled ? std::max(k, configured_stop_ef / 2) : configured_stop_ef;
    candidates.n = std::max(candidate_ef, k);
    candidates.clear();
    candidates.n = std::min(candidates.n, effective_stop_ef);
    if (stats) stats->base_distance_eval_count++;
    if (minimax_push_finite(candidates, nearest, d_nearest) && stats) {
        stats->base_candidate_insert_count++;
    }

    float threshold = heap_dis[0];
    int nres = 0;
    for (int i = 0; i < candidates.size(); ++i) {
        const faiss::idx_t v = candidates.ids[i];
        const float d = candidates.dis[i];
        if (d < threshold) {
            faiss::heap_replace_top<C>(k, heap_dis, heap_ids, d, v);
            threshold = heap_dis[0];
            nres += 1;
            if (stats) stats->base_result_insert_count++;
        }
        vt.set(static_cast<size_t>(v));
        if (stats) stats->base_visited_count++;
    }

    int nstep = 0;
    static constexpr int kCfrClassifyStart = 4;
    static constexpr int kCfrClassifyEnd = 16;
    static constexpr float kCfrEmaDecay = 0.8f;
    static constexpr float kCfrEmaUpdate = 1.0f - kCfrEmaDecay;
    int full_pop_count = 0;
    bool classification_evaluated = false;
    bool ef_was_shrunk = false;
    bool early_stopped = false;
    float best_seen_dist = d_nearest;
    float smoothed_ratio = std::numeric_limits<float>::quiet_NaN();
    float classify_ratio_sum = 0.0f;
    int classify_ratio_count = 0;
    while (candidates.size() > 0) {
        float d0 = 0.0f;
        const int v0 = candidates.pop_min(&d0);
        if (stats) stats->base_pop_count++;
        if (check_relative_distance) {
            const int n_dis_below = minimax_count_below_simd(candidates, d0);
            if (n_dis_below >= effective_stop_ef) break;
        }
        if (stats) stats->base_expand_count++;

        HnswNeighborSpan neighbor_span = graph.neighbors(v0, 0);

        int counter = 0;
        faiss::idx_t saved_j[64];
        const bool batch_all_neighbors =
                dis.batch_all_neighbors() &&
                !(early_abandon && candidates.k == candidates.n);
        threshold = heap_dis[0];

        auto add_to_heap = [&](faiss::idx_t idx, float cand_dis) {
            if (cand_dis < best_seen_dist) best_seen_dist = cand_dis;
            if (cand_dis < threshold) {
                faiss::heap_replace_top<C>(k, heap_dis, heap_ids, cand_dis, idx);
                threshold = heap_dis[0];
                nres += 1;
                if (stats) stats->base_result_insert_count++;
            }
            if (minimax_push_finite(candidates, static_cast<faiss::MinimaxHeap::storage_idx_t>(idx), cand_dis) && stats) {
                stats->base_candidate_insert_count++;
            }
        };

        for (size_t j = 0; j < neighbor_span.size; ++j) {
            const int v1 = neighbor_span.data[j];
            if (v1 < 0) break;
            if (stats) stats->base_neighbor_scan_count++;
            if (prefetch_lookahead > 0) {
                const size_t prefetch_j = j + static_cast<size_t>(prefetch_lookahead);
                if (prefetch_j < neighbor_span.size) {
                    const int pv = neighbor_span.data[prefetch_j];
                    if (pv >= 0) {
                        vt.prefetch(static_cast<size_t>(pv));
                        dis.prefetch_idx(pv);
                    }
                }
            }
            if (vt.set(static_cast<size_t>(v1))) {
                if (stats) stats->base_visited_count++;
                saved_j[counter++] = v1;
                dis.prefetch_idx(v1);
                if (!batch_all_neighbors && counter == 32) {
                    float d[32];
                    if (stats) stats->base_distance_eval_count += 32;
                    if (early_abandon && candidates.k == candidates.n) {
                        const float prune_distance = candidates.dis[0];
                        for (int bi = 0; bi < 32; ++bi) {
                            d[bi] = dis.distance_idx_pruned(saved_j[bi], prune_distance);
                        }
                    } else {
                        dis.distances_batch_32(saved_j, d);
                    }
                    for (int bi = 0; bi < 32; ++bi) {
                        add_to_heap(saved_j[bi], d[bi]);
                    }
                    counter = 0;
                }
                if (counter == 64) {
                    float d[64];
                    if (stats) stats->base_distance_eval_count += 64;
                    dis.distances_batch_n(saved_j, 64, d);
                    for (int bi = 0; bi < 64; ++bi) {
                        add_to_heap(saved_j[bi], d[bi]);
                    }
                    counter = 0;
                }
            } else if (stats) {
                stats->base_duplicate_visit_count++;
            }
        }
        if (batch_all_neighbors && counter > 0) {
            float d[64];
            if (stats) stats->base_distance_eval_count += counter;
            dis.distances_batch_n(saved_j, counter, d);
            for (int bi = 0; bi < counter; ++bi) {
                add_to_heap(saved_j[bi], d[bi]);
            }
            counter = 0;
        }
        int tail = 0;
        for (; tail + 16 <= counter; tail += 16) {
            float d[16];
            if (stats) stats->base_distance_eval_count += 16;
            if (early_abandon && candidates.k == candidates.n) {
                const float prune_distance = candidates.dis[0];
                for (int bi = 0; bi < 16; ++bi) {
                    d[bi] = dis.distance_idx_pruned(saved_j[tail + bi], prune_distance);
                }
            } else {
                dis.distances_batch_16(saved_j + tail, d);
            }
            for (int bi = 0; bi < 16; ++bi) {
                add_to_heap(saved_j[tail + bi], d[bi]);
            }
        }
        for (; tail + 8 <= counter; tail += 8) {
            float d[8];
            if (stats) stats->base_distance_eval_count += 8;
            if (early_abandon && candidates.k == candidates.n) {
                const float prune_distance = candidates.dis[0];
                for (int bi = 0; bi < 8; ++bi) {
                    d[bi] = dis.distance_idx_pruned(saved_j[tail + bi], prune_distance);
                }
            } else {
                dis.distances_batch_8(saved_j + tail, d);
            }
            for (int bi = 0; bi < 8; ++bi) {
                add_to_heap(saved_j[tail + bi], d[bi]);
            }
        }
        for (; tail < counter; ++tail) {
            if (stats) stats->base_distance_eval_count++;
            if (early_abandon && candidates.k == candidates.n) {
                add_to_heap(saved_j[tail], dis.distance_idx_pruned(saved_j[tail], candidates.dis[0]));
            } else {
                add_to_heap(saved_j[tail], dis.distance_idx(saved_j[tail]));
            }
        }

        nstep += 1;
        if (adaptive_enabled && candidates.k == candidates.n) {
            full_pop_count++;
            const float worst_frontier = candidates.dis[0];
            const float denom = std::max(worst_frontier - best_seen_dist, 1e-6f);
            float ratio = (d0 - best_seen_dist) / denom;
            ratio = std::min(1.0f, std::max(0.0f, ratio));
            if (std::isnan(smoothed_ratio)) {
                smoothed_ratio = ratio;
            } else {
                smoothed_ratio = kCfrEmaDecay * smoothed_ratio + kCfrEmaUpdate * ratio;
            }
            if (full_pop_count >= kCfrClassifyStart && full_pop_count <= kCfrClassifyEnd) {
                classify_ratio_sum += smoothed_ratio;
                classify_ratio_count++;
                if (!classification_evaluated && full_pop_count == kCfrClassifyEnd) {
                    classification_evaluated = true;
                    const float classify_ratio_mean = classify_ratio_sum / static_cast<float>(std::max(1, classify_ratio_count));
                    int target_ef = configured_stop_ef;
                    if (classify_ratio_mean <= adaptive_cfr->tau) {
                        const float normalized_ratio = classify_ratio_mean / std::max(adaptive_cfr->tau, 1e-6f);
                        target_ef = resolve_adcbq_cfr_ef(configured_stop_ef, k, *adaptive_cfr, normalized_ratio);
                    }
                    if (target_ef != effective_stop_ef) {
                        effective_stop_ef = target_ef;
                        candidates.n = std::min(std::max(candidate_ef, k), std::max(effective_stop_ef, candidates.k));
                        ef_was_shrunk = effective_stop_ef < configured_stop_ef;
                    }
                }
            }
        }
        if (!check_relative_distance && nstep > effective_stop_ef) break;
    }

    if (stats) {
        stats->adaptive_effective_ef = static_cast<uint64_t>(std::max(effective_stop_ef, 0));
        stats->adaptive_shrunk = ef_was_shrunk ? 1 : 0;
        stats->adaptive_early_stopped = early_stopped ? 1 : 0;
        stats->adaptive_classified = classification_evaluated ? 1 : 0;
        stats->nhops = stats->upper_hop_count + stats->base_expand_count;
        stats->ndis = stats->upper_distance_eval_count + stats->base_distance_eval_count;
    }

    if (calibrated_output) {
        compact_calibrated_threshold_results(
                k, heap_dis, heap_ids, *calibrated_output);
    } else {
        faiss::heap_reorder<C>(k, heap_dis, heap_ids);
    }
    vt.advance();
}

template <typename DistanceComputerT>
void search_one_packed_inline(
        const faiss::IndexBinaryHNSW& index,
        DistanceComputerT& dis,
        faiss::VisitedTable& vt,
        int k,
        int candidate_ef,
        int stop_ef,
        bool check_relative_distance,
        bool early_abandon,
        int prefetch_lookahead,
        const AdcAdaptiveCfrConfig* adaptive_cfr,
        faiss::MinimaxHeap& candidates,
        float* heap_dis,
        faiss::idx_t* heap_ids,
        AdcSearchStats* stats,
        CalibratedThresholdOutput* calibrated_output = nullptr) {
    RamHnswGraphAccessor graph(index.hnsw);
    search_one_packed_inline_graph(
            graph,
            dis,
            vt,
            k,
            candidate_ef,
            stop_ef,
            check_relative_distance,
            early_abandon,
            prefetch_lookahead,
            adaptive_cfr,
            candidates,
            heap_dis,
            heap_ids,
            stats,
            calibrated_output);
}

struct PagedAsymBQDistanceComputer : faiss::DistanceComputer {
    PagedHnswIndex& index;
    PackedAsymBQDistanceComputer evaluator;
    bool async_graph_pages = false;
    unsigned async_queue_depth = 48;
    unsigned async_sync_miss_max = 0;
    IndexUringThreadBuffer async_buffer;

    PagedAsymBQDistanceComputer(
            PagedHnswIndex& index_in,
            int dim,
            AdcMode adc_mode,
            LutMode lut_mode,
            bool early_abandon,
            bool ordered_early_abandon,
            size_t ordered_probe_bytes,
            bool async_graph_pages_in,
            unsigned async_queue_depth_in,
            unsigned async_sync_miss_max_in)
            : index(index_in),
              evaluator(
                      nullptr,
                      index_in.code_size(),
                      index_in.code_size(),
                      dim,
                      adc_mode,
                      lut_mode,
                      early_abandon,
                      ordered_early_abandon,
                      ordered_probe_bytes),
              async_graph_pages(async_graph_pages_in),
              async_queue_depth(async_queue_depth_in),
              async_sync_miss_max(async_sync_miss_max_in) {}

    void set_query(const float* query) override {
        evaluator.set_query(query);
    }

    float distance_idx(faiss::idx_t id) {
        if (id < 0) return std::numeric_limits<float>::max();
        PagedNodeView node = index.node(static_cast<size_t>(id));
        return evaluator.distance_to_code(node.code);
    }

    float distance_idx_pruned(faiss::idx_t id, float /*max_distance*/) {
        return distance_idx(id);
    }

    float operator()(faiss::idx_t id) override {
        return distance_idx(id);
    }

    float symmetric_dis(faiss::idx_t, faiss::idx_t) override {
        throw std::runtime_error("symmetric distance is not supported by the paged ADC-BQ search path");
    }

    void prefetch_idx(faiss::idx_t /*id*/) const {
        // Do not turn a CPU prefetch hint into a synchronous SSD page access.
    }

    void distances_batch_async(
            const faiss::idx_t* ids,
            int count,
            float* distances) {
        if (count <= 0) return;
        static constexpr int kMaxBatch = 64;
        if (count > kMaxBatch) {
            throw std::runtime_error("graph-page async batch exceeds 64 nodes");
        }

        // These batches execute for every graph expansion, so keep the small
        // (<=64) lookup tables on the stack instead of allocating vectors and
        // unordered_maps in the hot path.
        std::array<int, kMaxBatch> missing_slot{};
        missing_slot.fill(-1);
        std::array<int, kMaxBatch> known_index_by_position{};
        known_index_by_position.fill(-1);
        std::array<uint32_t, kMaxBatch> known_pages{};
        std::array<int, kMaxBatch> known_missing_slot{};
        known_missing_slot.fill(-1);
        std::array<std::shared_ptr<AlignedIndexPage>, kMaxBatch> known_cached{};
        static constexpr size_t kPageHashSize = 128;
        std::array<int, kPageHashSize> page_hash{};
        page_hash.fill(-1);
        std::array<uint32_t, kMaxBatch> missing_pages{};
        std::array<std::shared_ptr<AlignedIndexPage>, kMaxBatch> submitted_pages{};
        size_t known_count = 0;
        size_t miss_count = 0;
        uint64_t coalesced_hits = 0;

        for (int i = 0; i < count; ++i) {
            if (ids[i] < 0) {
                distances[i] = std::numeric_limits<float>::max();
                continue;
            }
            const uint32_t page_no = index.locator(
                    static_cast<size_t>(ids[i])).page_no;
            size_t bucket =
                    (static_cast<size_t>(page_no) * 2654435761u) &
                    (kPageHashSize - 1);
            while (
                    page_hash[bucket] >= 0 &&
                    known_pages[static_cast<size_t>(page_hash[bucket])] != page_no) {
                bucket = (bucket + 1) & (kPageHashSize - 1);
            }
            if (page_hash[bucket] >= 0) {
                const size_t known = static_cast<size_t>(page_hash[bucket]);
                known_index_by_position[static_cast<size_t>(i)] =
                        static_cast<int>(known);
                ++coalesced_hits;
                continue;
            }
            known_pages[known_count] = page_no;
            page_hash[bucket] = static_cast<int>(known_count);
            known_index_by_position[static_cast<size_t>(i)] =
                    static_cast<int>(known_count);
            ++known_count;
        }

        index.lookup_pages_batch(
                known_pages.data(),
                known_count,
                known_cached.data(),
                coalesced_hits);
        for (size_t known = 0; known < known_count; ++known) {
            if (known_cached[known]) continue;
            const size_t slot = miss_count++;
            missing_pages[slot] = known_pages[known];
            submitted_pages[slot] = std::make_shared<AlignedIndexPage>();
            known_missing_slot[known] = static_cast<int>(slot);
        }
        for (int i = 0; i < count; ++i) {
            const int known = known_index_by_position[static_cast<size_t>(i)];
            if (known < 0) continue;
            missing_slot[static_cast<size_t>(i)] =
                    known_missing_slot[static_cast<size_t>(known)];
        }

        // A fully cached expansion never touches io_uring.  Keep this as an
        // explicit fast path so the common high-RAM case skips all async
        // setup, submission, and completion handling.
        if (miss_count == 0) {
            for (int i = 0; i < count; ++i) {
                if (ids[i] < 0) continue;
                const size_t known = static_cast<size_t>(
                        known_index_by_position[static_cast<size_t>(i)]);
                PagedNodeView node = index.node_from_page_borrowed(
                        static_cast<size_t>(ids[i]),
                        known_cached[known].get());
                distances[i] = evaluator.distance_to_code(node.code);
            }
            return;
        }

        // For a tiny miss batch, synchronous O_DIRECT pread is cheaper than
        // preparing and draining an io_uring batch.  Across T search threads
        // these reads still provide device-level concurrency.  Larger batches
        // retain query-internal async overlap and the requested queue depth.
        if (async_sync_miss_max > 0 && miss_count <= async_sync_miss_max) {
            const auto sync_started = std::chrono::steady_clock::now();
            for (size_t slot = 0; slot < miss_count; ++slot) {
                size_t done = 0;
                const off_t offset = static_cast<off_t>(
                        static_cast<uint64_t>(missing_pages[slot]) *
                        kPagedIndexPageBytes);
                while (done < kPagedIndexPageBytes) {
                    const ssize_t got = ::pread(
                            index.page_fd(),
                            static_cast<char*>(submitted_pages[slot]->data) + done,
                            kPagedIndexPageBytes - done,
                            offset + static_cast<off_t>(done));
                    if (got < 0) {
                        if (errno == EINTR) continue;
                        throw std::runtime_error(
                                std::string(
                                        "hybrid synchronous graph-page read failed: ") +
                                std::strerror(errno));
                    }
                    if (got == 0) {
                        throw std::runtime_error(
                                "short hybrid synchronous graph-page read");
                    }
                    done += static_cast<size_t>(got);
                }
                submitted_pages[slot] = index.install_page(
                        missing_pages[slot],
                        std::move(submitted_pages[slot]));
            }
            const double sync_wait_seconds = seconds_since(sync_started);
            index.record_hybrid_sync_batch(miss_count, sync_wait_seconds);
            for (int i = 0; i < count; ++i) {
                if (ids[i] < 0) continue;
                const int slot = missing_slot[static_cast<size_t>(i)];
                const size_t known = static_cast<size_t>(
                        known_index_by_position[static_cast<size_t>(i)]);
                const AlignedIndexPage* page =
                        slot >= 0
                        ? submitted_pages[static_cast<size_t>(slot)].get()
                        : known_cached[known].get();
                PagedNodeView node = index.node_from_page_borrowed(
                        static_cast<size_t>(ids[i]),
                        page);
                distances[i] = evaluator.distance_to_code(node.code);
            }
            return;
        }

        uint64_t cached_nodes_scored = 0;
        uint64_t completed_nodes_scored_while_inflight = 0;
        uint64_t ready_before_wait = 0;
        double wait_seconds = 0.0;
        if (miss_count > 0) {
            // Treat --index-page-qd as an actual per-query in-flight cap.
            // The previous implementation sized the ring with
            // max(QD, miss_count) and submitted every miss at once, so values
            // below the expansion width did not change the I/O concurrency.
            const size_t inflight_limit = std::min<size_t>(
                    miss_count,
                    std::max<size_t>(1, async_queue_depth));
            io_uring& ring = async_buffer.ensure(
                    static_cast<unsigned>(inflight_limit));

            auto prepare_read = [&](size_t slot) {
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) {
                    throw std::runtime_error(
                            "io_uring_get_sqe failed for graph page");
                }
                io_uring_prep_read(
                        sqe,
                        index.page_fd(),
                        submitted_pages[slot]->data,
                        static_cast<unsigned>(kPagedIndexPageBytes),
                        static_cast<off_t>(
                                static_cast<uint64_t>(missing_pages[slot]) *
                                kPagedIndexPageBytes));
                io_uring_sqe_set_data64(sqe, slot);
            };
            auto submit_count = [&](size_t requested) {
                size_t submitted = 0;
                while (submitted < requested) {
                    const int rc = io_uring_submit(&ring);
                    if (rc == -EINTR) continue;
                    if (rc <= 0) {
                        throw std::runtime_error(
                                std::string(
                                        "io_uring_submit failed for graph pages: ") +
                                (rc < 0 ? std::strerror(-rc) : "no progress"));
                    }
                    submitted += static_cast<size_t>(rc);
                }
            };

            size_t next_to_submit = 0;
            for (; next_to_submit < inflight_limit; ++next_to_submit) {
                prepare_read(next_to_submit);
            }
            submit_count(inflight_limit);

            auto submit_next = [&]() {
                if (next_to_submit >= miss_count) return;
                prepare_read(next_to_submit);
                submit_count(1);
                ++next_to_submit;
            };

            /*
             * Keep at most inflight_limit requests outstanding.  Each
             * completion immediately admits one pending miss, maintaining
             * device utilization without silently expanding the requested
             * queue depth to the whole neighbor batch.
             */
            // Turbocharging-style hybrid overlap: score cache-resident nodes
            // while SSD reads for cache misses are in flight.
            for (int i = 0; i < count; ++i) {
                const int known_value =
                        known_index_by_position[static_cast<size_t>(i)];
                if (known_value < 0) continue;
                const size_t known = static_cast<size_t>(known_value);
                if (!known_cached[known]) continue;
                PagedNodeView node = index.node_from_page_borrowed(
                        static_cast<size_t>(ids[i]),
                        known_cached[known].get());
                distances[i] = evaluator.distance_to_code(node.code);
                ++cached_nodes_scored;
            }

            size_t completed = 0;
            auto consume = [&](io_uring_cqe* cqe) {
                const size_t slot = static_cast<size_t>(
                        io_uring_cqe_get_data64(cqe));
                if (slot >= miss_count) {
                    io_uring_cqe_seen(&ring, cqe);
                    throw std::runtime_error(
                            "io_uring returned invalid graph-page request id");
                }
                if (cqe->res < 0) {
                    const int error = -cqe->res;
                    io_uring_cqe_seen(&ring, cqe);
                    throw std::runtime_error(
                            std::string("async graph-page O_DIRECT read failed: ") +
                            std::strerror(error));
                }
                if (static_cast<size_t>(cqe->res) != kPagedIndexPageBytes) {
                    io_uring_cqe_seen(&ring, cqe);
                    throw std::runtime_error("short async graph-page read");
                }
                io_uring_cqe_seen(&ring, cqe);
                submitted_pages[slot] = index.install_page(
                        missing_pages[slot], std::move(submitted_pages[slot]));
                // Score every node carried by this page immediately.  Other
                // page requests remain in flight while this CPU work runs.
                uint64_t nodes_on_completed_page = 0;
                for (int i = 0; i < count; ++i) {
                    if (missing_slot[static_cast<size_t>(i)] !=
                        static_cast<int>(slot)) {
                        continue;
                    }
                    PagedNodeView node = index.node_from_page_borrowed(
                            static_cast<size_t>(ids[i]),
                            submitted_pages[slot].get());
                    distances[i] = evaluator.distance_to_code(node.code);
                    ++nodes_on_completed_page;
                }
                if (completed + 1 < miss_count) {
                    completed_nodes_scored_while_inflight +=
                            nodes_on_completed_page;
                }
                ++completed;
                submit_next();
            };

            for (;;) {
                io_uring_cqe* cqe = nullptr;
                const int rc = io_uring_peek_cqe(&ring, &cqe);
                if (rc == -EAGAIN || cqe == nullptr) break;
                if (rc < 0) {
                    throw std::runtime_error(
                            std::string("io_uring_peek_cqe failed for graph page: ") +
                            std::strerror(-rc));
                }
                consume(cqe);
                ++ready_before_wait;
            }

            const auto wait_started = std::chrono::steady_clock::now();
            while (completed < miss_count) {
                io_uring_cqe* cqe = nullptr;
                const int rc = io_uring_wait_cqe(&ring, &cqe);
                if (rc == -EINTR) continue;
                if (rc < 0) {
                    throw std::runtime_error(
                            std::string("io_uring_wait_cqe failed for graph page: ") +
                            std::strerror(-rc));
                }
                consume(cqe);
            }
            wait_seconds = seconds_since(wait_started);

            index.record_async_batch(
                    miss_count,
                    cached_nodes_scored,
                    completed_nodes_scored_while_inflight,
                    ready_before_wait,
                    wait_seconds);
            return;
        }

        for (int i = 0; i < count; ++i) {
            if (ids[i] < 0) continue;
            const size_t known = static_cast<size_t>(
                    known_index_by_position[static_cast<size_t>(i)]);
            PagedNodeView node = index.node_from_page_borrowed(
                    static_cast<size_t>(ids[i]),
                    known_cached[known].get());
            distances[i] = evaluator.distance_to_code(node.code);
        }
    }

    void distances_batch_8(const faiss::idx_t* ids, float* distances) {
        if (async_graph_pages) {
            distances_batch_async(ids, 8, distances);
            return;
        }
        for (int i = 0; i < 8; ++i) distances[i] = distance_idx(ids[i]);
    }

    void distances_batch_16(const faiss::idx_t* ids, float* distances) {
        if (async_graph_pages) {
            distances_batch_async(ids, 16, distances);
            return;
        }
        for (int i = 0; i < 16; ++i) distances[i] = distance_idx(ids[i]);
    }

    void distances_batch_32(const faiss::idx_t* ids, float* distances) {
        if (async_graph_pages) {
            distances_batch_async(ids, 32, distances);
            return;
        }
        for (int i = 0; i < 32; ++i) distances[i] = distance_idx(ids[i]);
    }

    bool batch_all_neighbors() const { return async_graph_pages; }

    void distances_batch_n(
            const faiss::idx_t* ids,
            int count,
            float* distances) {
        if (async_graph_pages) {
            distances_batch_async(ids, count, distances);
            return;
        }
        for (int i = 0; i < count; ++i) distances[i] = distance_idx(ids[i]);
    }
};

struct MmapAsymBQDistanceComputer : faiss::DistanceComputer {
    const MmapHnswIndex& index;
    PackedAsymBQDistanceComputer evaluator;

    MmapAsymBQDistanceComputer(
            const MmapHnswIndex& index_in,
            int dim,
            AdcMode adc_mode,
            LutMode lut_mode,
            bool early_abandon,
            bool ordered_early_abandon,
            size_t ordered_probe_bytes)
            : index(index_in),
              evaluator(
                      nullptr,
                      index_in.code_size(),
                      index_in.code_size(),
                      dim,
                      adc_mode,
                      lut_mode,
                      early_abandon,
                      ordered_early_abandon,
                      ordered_probe_bytes) {
        evaluator.set_indirect_codes(index.code_pointer_table());
    }

    static const uint8_t* resolve_code(const void* context, faiss::idx_t id) {
        const auto* mmap_index = static_cast<const MmapHnswIndex*>(context);
        return mmap_index->code_ptr(static_cast<size_t>(id));
    }

    void set_query(const float* query) override { evaluator.set_query(query); }

    float distance_idx(faiss::idx_t id) {
        if (id < 0) return std::numeric_limits<float>::max();
        return evaluator.distance_idx(id);
    }

    float distance_idx_pruned(faiss::idx_t id, float /*max_distance*/) {
        return distance_idx(id);
    }

    float operator()(faiss::idx_t id) override { return distance_idx(id); }

    float symmetric_dis(faiss::idx_t, faiss::idx_t) override {
        throw std::runtime_error("symmetric distance is not supported by the mmap ADC-BQ search path");
    }

    void prefetch_idx(faiss::idx_t id) const {
        evaluator.prefetch_idx(id);
    }

    void distances_batch_8(const faiss::idx_t* ids, float* distances) {
        evaluator.distances_batch_8(ids, distances);
    }

    void distances_batch_16(const faiss::idx_t* ids, float* distances) {
        evaluator.distances_batch_16(ids, distances);
    }

    void distances_batch_32(const faiss::idx_t* ids, float* distances) {
        evaluator.distances_batch_32(ids, distances);
    }

    bool batch_all_neighbors() const { return false; }

    void distances_batch_n(
            const faiss::idx_t* ids,
            int count,
            float* distances) {
        for (int i = 0; i < count; ++i) distances[i] = distance_idx(ids[i]);
    }
};


void search_packed_asym_inline(
        const faiss::IndexBinaryHNSW& index,
        const float* queries,
        int nq,
        int dim,
        int k,
        int ef,
        int threads,
        AdcMode adc_mode,
        LutMode lut_mode,
        bool check_relative_distance,
        bool early_abandon,
        bool ordered_early_abandon,
        int ordered_early_abandon_probe_bytes,
        int prefetch_lookahead,
        const AdcAdaptiveCfrConfig* adaptive_cfr,
        const uint8_t* codes,
        size_t code_stride,
        const uint8_t* block32_codes,
        size_t block32_block_stride,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& labels,
        std::vector<AdcSearchStats>* query_stats,
        std::vector<double>* query_seconds = nullptr) {
    distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    labels.assign(static_cast<size_t>(nq) * k, -1);
    if (!codes) {
        codes = storage_codes(index);
        code_stride = index.code_size;
    }
    if (query_stats) {
        query_stats->assign(static_cast<size_t>(nq), AdcSearchStats{});
    }
    if (query_seconds) query_seconds->assign(static_cast<size_t>(nq), 0.0);

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        faiss::VisitedTable vt(index.ntotal, false);
        faiss::MinimaxHeap candidates(std::max(ef, k));
        PackedAsymBQDistanceComputer dis(
                codes,
                index.code_size,
                code_stride,
                dim,
                adc_mode,
                lut_mode,
                early_abandon,
                ordered_early_abandon,
                static_cast<size_t>(ordered_early_abandon_probe_bytes),
                block32_codes,
                block32_block_stride);
#pragma omp for schedule(static)
        for (int i = 0; i < nq; ++i) {
            const auto query_start = std::chrono::steady_clock::now();
            dis.set_query(queries + static_cast<size_t>(i) * dim);
            search_one_packed_inline(
                    index,
                    dis,
                    vt,
                    k,
                    std::max(ef, k),
                    ef,
                    check_relative_distance,
                    early_abandon,
                    prefetch_lookahead,
                    adaptive_cfr,
                    candidates,
                    distances.data() + static_cast<size_t>(i) * k,
                    labels.data() + static_cast<size_t>(i) * k,
                    query_stats ? &(*query_stats)[static_cast<size_t>(i)] : nullptr);
            if (query_seconds) {
                (*query_seconds)[static_cast<size_t>(i)] = seconds_since(query_start);
            }
        }
    }
}

void search_paged_asym_inline(
        const std::string& paged_index_path,
        size_t index_page_cache_mib,
        bool index_page_async,
        int index_page_qd,
        int index_page_async_sync_miss_max,
        const float* queries,
        int nq,
        int dim,
        int k,
        int ef,
        int threads,
        AdcMode adc_mode,
        LutMode lut_mode,
        bool check_relative_distance,
        bool early_abandon,
        bool ordered_early_abandon,
        int ordered_early_abandon_probe_bytes,
        int prefetch_lookahead,
        const AdcAdaptiveCfrConfig* adaptive_cfr,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& labels,
        std::vector<AdcSearchStats>* query_stats,
        IndexPageCacheStats* page_cache_stats) {
    PagedHnswIndex paged_index(paged_index_path, index_page_cache_mib);
    if (paged_index.dim() != dim) throw std::runtime_error("paged index/query dim mismatch");
    distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    labels.assign(static_cast<size_t>(nq) * k, -1);
    if (query_stats) query_stats->assign(static_cast<size_t>(nq), AdcSearchStats{});

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        faiss::VisitedTable vt(paged_index.size(), false);
        faiss::MinimaxHeap candidates(std::max(ef, k));
        PagedAsymBQDistanceComputer dis(
                paged_index,
                dim,
                adc_mode,
                lut_mode,
                early_abandon,
                ordered_early_abandon,
                static_cast<size_t>(ordered_early_abandon_probe_bytes),
                index_page_async,
                static_cast<unsigned>(index_page_qd),
                static_cast<unsigned>(index_page_async_sync_miss_max));
        PagedHnswGraphAccessor graph(paged_index);
#pragma omp for schedule(static)
        for (int i = 0; i < nq; ++i) {
            dis.set_query(queries + static_cast<size_t>(i) * dim);
            search_one_packed_inline_graph(
                    graph,
                    dis,
                    vt,
                    k,
                    std::max(ef, k),
                    ef,
                    check_relative_distance,
                    early_abandon,
                    prefetch_lookahead,
                    adaptive_cfr,
                    candidates,
                    distances.data() + static_cast<size_t>(i) * k,
                    labels.data() + static_cast<size_t>(i) * k,
                    query_stats ? &(*query_stats)[static_cast<size_t>(i)] : nullptr);
        }
    }
    if (page_cache_stats) *page_cache_stats = paged_index.cache_stats();
}

void search_mmap_asym_inline(
        const MmapHnswIndex& mmap_index,
        const float* queries,
        int nq,
        int dim,
        int k,
        int ef,
        int threads,
        AdcMode adc_mode,
        LutMode lut_mode,
        bool check_relative_distance,
        bool early_abandon,
        bool ordered_early_abandon,
        int ordered_early_abandon_probe_bytes,
        int prefetch_lookahead,
        const AdcAdaptiveCfrConfig* adaptive_cfr,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& labels,
        std::vector<AdcSearchStats>* query_stats) {
    if (mmap_index.dim() != dim) throw std::runtime_error("mmap index/query dim mismatch");
    distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    labels.assign(static_cast<size_t>(nq) * k, -1);
    if (query_stats) query_stats->assign(static_cast<size_t>(nq), AdcSearchStats{});

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        faiss::VisitedTable vt(mmap_index.size(), false);
        faiss::MinimaxHeap candidates(std::max(ef, k));
        MmapAsymBQDistanceComputer dis(
                mmap_index,
                dim,
                adc_mode,
                lut_mode,
                early_abandon,
                ordered_early_abandon,
                static_cast<size_t>(ordered_early_abandon_probe_bytes));
        MmapHnswGraphAccessor graph(mmap_index);
#pragma omp for schedule(static)
        for (int i = 0; i < nq; ++i) {
            dis.set_query(queries + static_cast<size_t>(i) * dim);
            search_one_packed_inline_graph(
                    graph,
                    dis,
                    vt,
                    k,
                    std::max(ef, k),
                    ef,
                    check_relative_distance,
                    early_abandon,
                    prefetch_lookahead,
                    adaptive_cfr,
                    candidates,
                    distances.data() + static_cast<size_t>(i) * k,
                    labels.data() + static_cast<size_t>(i) * k,
                    query_stats ? &(*query_stats)[static_cast<size_t>(i)] : nullptr);
        }
    }
}


void search_packed_xflag_inline(
        const faiss::IndexBinaryHNSW& index,
        const float* queries,
        int nq,
        int dim,
        int k,
        int ef,
        int threads,
        bool check_relative_distance,
        bool early_abandon,
        int prefetch_lookahead,
        const uint8_t* sign_codes,
        size_t sign_code_stride,
        const uint8_t* xflag_codes,
        size_t xflag_code_stride,
        float xflag_threshold,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& labels,
        std::vector<AdcSearchStats>* query_stats) {
    if (!xflag_codes) {
        throw std::runtime_error("xflag search requires xflag codes");
    }
    distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    labels.assign(static_cast<size_t>(nq) * k, -1);
    if (!sign_codes) {
        sign_codes = storage_codes(index);
        sign_code_stride = index.code_size;
    }
    if (query_stats) {
        query_stats->assign(static_cast<size_t>(nq), AdcSearchStats{});
    }

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        faiss::VisitedTable vt(index.ntotal, false);
        faiss::MinimaxHeap candidates(std::max(ef, k));
        PackedXFlagBQDistanceComputer dis(
                sign_codes,
                xflag_codes,
                index.code_size,
                sign_code_stride,
                xflag_code_stride,
                dim,
                xflag_threshold);
        const AdcAdaptiveCfrConfig* adaptive_cfr = nullptr;
#pragma omp for schedule(static)
        for (int i = 0; i < nq; ++i) {
            dis.set_query(queries + static_cast<size_t>(i) * dim);
            search_one_packed_inline(
                    index,
                    dis,
                    vt,
                    k,
                    std::max(ef, k),
                    ef,
                    check_relative_distance,
                    early_abandon,
                    prefetch_lookahead,
                    adaptive_cfr,
                    candidates,
                    distances.data() + static_cast<size_t>(i) * k,
                    labels.data() + static_cast<size_t>(i) * k,
                    query_stats ? &(*query_stats)[static_cast<size_t>(i)] : nullptr);
        }
    }
}


void search_packed_adcxflag_inline(
        const faiss::IndexBinaryHNSW& index,
        const float* queries,
        int nq,
        int dim,
        int k,
        int ef,
        int threads,
        bool check_relative_distance,
        bool early_abandon,
        int prefetch_lookahead,
        const uint8_t* sign_codes,
        size_t sign_code_stride,
        const uint8_t* xflag_codes,
        size_t xflag_code_stride,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& labels,
        std::vector<AdcSearchStats>* query_stats) {
    if (!xflag_codes) {
        throw std::runtime_error("adc-xflag search requires xflag codes");
    }
    distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    labels.assign(static_cast<size_t>(nq) * k, -1);
    if (!sign_codes) {
        sign_codes = storage_codes(index);
        sign_code_stride = index.code_size;
    }
    if (query_stats) {
        query_stats->assign(static_cast<size_t>(nq), AdcSearchStats{});
    }

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        faiss::VisitedTable vt(index.ntotal, false);
        faiss::MinimaxHeap candidates(std::max(ef, k));
        PackedAsymXFlagBQDistanceComputer dis(
                sign_codes,
                xflag_codes,
                index.code_size,
                sign_code_stride,
                xflag_code_stride,
                dim);
        const AdcAdaptiveCfrConfig* adaptive_cfr = nullptr;
#pragma omp for schedule(static)
        for (int i = 0; i < nq; ++i) {
            dis.set_query(queries + static_cast<size_t>(i) * dim);
            search_one_packed_inline(
                    index,
                    dis,
                    vt,
                    k,
                    std::max(ef, k),
                    ef,
                    check_relative_distance,
                    early_abandon,
                    prefetch_lookahead,
                    adaptive_cfr,
                    candidates,
                    distances.data() + static_cast<size_t>(i) * k,
                    labels.data() + static_cast<size_t>(i) * k,
                    query_stats ? &(*query_stats)[static_cast<size_t>(i)] : nullptr);
        }
    }
}

void search_packed_asym(
        const faiss::IndexBinaryHNSW& index,
        const float* queries,
        int nq,
        int dim,
        int k,
        int ef,
        int threads,
        AdcMode adc_mode,
        LutMode lut_mode,
        SearchImpl search_impl,
        bool check_relative_distance,
        bool bounded_queue,
        bool early_abandon,
        bool ordered_early_abandon,
        int ordered_early_abandon_probe_bytes,
        int prefetch_lookahead,
        const AdcAdaptiveCfrConfig* adaptive_cfr,
        const uint8_t* inline_codes,
        size_t inline_code_stride,
        const uint8_t* block32_codes,
        size_t block32_block_stride,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& labels,
        std::vector<AdcSearchStats>* query_stats = nullptr,
        std::vector<double>* query_seconds = nullptr) {
    if (search_impl == SearchImpl::Inline) {
        if (!bounded_queue) {
            throw std::runtime_error("inline search supports only bounded queue");
        }
        search_packed_asym_inline(
                index,
                queries,
                nq,
                dim,
                k,
                ef,
                threads,
                adc_mode,
                lut_mode,
                check_relative_distance,
                early_abandon,
                ordered_early_abandon,
                ordered_early_abandon_probe_bytes,
                prefetch_lookahead,
                adaptive_cfr,
                inline_codes,
                inline_code_stride,
                block32_codes,
                block32_block_stride,
                distances,
                labels,
                query_stats,
                query_seconds);
        return;
    }

    distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    labels.assign(static_cast<size_t>(nq) * k, -1);
    if (query_stats) {
        throw std::runtime_error("--stats-out requires --search-impl inline");
    }
    if (query_seconds) query_seconds->assign(static_cast<size_t>(nq), 0.0);
    faiss::SearchParametersHNSW params;
    params.efSearch = ef;
    params.check_relative_distance = check_relative_distance;
    params.bounded_queue = bounded_queue;
    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;
    RH bres(nq, distances.data(), labels.data(), k);
    const uint8_t* codes = storage_codes(index);

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        faiss::VisitedTable vt(index.ntotal);
        PackedAsymBQDistanceComputer dis(codes, index.code_size, index.code_size, dim, adc_mode, lut_mode, false);
        RH::SingleResultHandler res(bres);
#pragma omp for schedule(static)
        for (int i = 0; i < nq; ++i) {
            const auto query_start = std::chrono::steady_clock::now();
            res.begin(i);
            dis.set_query(queries + static_cast<size_t>(i) * dim);
            index.hnsw.search(dis, nullptr, res, vt, &params);
            res.end();
            if (query_seconds) {
                (*query_seconds)[static_cast<size_t>(i)] = seconds_since(query_start);
            }
        }
    }
}

std::vector<double> compute_query_norms(
        const float* queries,
        int nq,
        int dim,
        int threads) {
    std::vector<double> norms(nq, 0.0);
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = queries + static_cast<size_t>(qi) * dim;
        double ss = 0.0;
#pragma omp simd reduction(+ : ss)
        for (int j = 0; j < dim; ++j) {
            ss += static_cast<double>(q[j]) * q[j];
        }
        norms[qi] = std::sqrt(std::max(ss, 1e-30));
    }
    return norms;
}

std::vector<double> compute_train_norms(const FbinMmap& train, int threads) {
    std::vector<double> norms(train.n, 0.0);
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < train.n; ++i) {
        const float* x = train.row(static_cast<size_t>(i));
        double ss = 0.0;
#pragma omp simd reduction(+ : ss)
        for (int j = 0; j < train.d; ++j) {
            ss += static_cast<double>(x[j]) * x[j];
        }
        norms[static_cast<size_t>(i)] = std::sqrt(std::max(ss, 1e-30));
    }
    return norms;
}

void prefetch_float_row(const float* x, int dim) {
    const char* p = reinterpret_cast<const char*>(x);
    const size_t bytes = static_cast<size_t>(dim) * sizeof(float);
    __builtin_prefetch(p, 0, 1);
    if (bytes > 2048) __builtin_prefetch(p + 2048, 0, 1);
    if (bytes > 4096) __builtin_prefetch(p + 4096, 0, 1);
    if (bytes > 8192) __builtin_prefetch(p + 8192, 0, 1);
    if (bytes > 12288) __builtin_prefetch(p + 12288, 0, 1);
}

double dot_product_fp64(const float* x, const float* q, int dim) {
    double dot = 0.0;
#pragma omp simd reduction(+ : dot)
    for (int j = 0; j < dim; ++j) {
        dot += static_cast<double>(x[j]) * q[j];
    }
    return dot;
}

float dot_product_fp32(const float* x, const float* q, int dim) {
#ifdef __AVX2__
    __m256 s0 = _mm256_setzero_ps();
    __m256 s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps();
    __m256 s3 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 32 <= dim; j += 32) {
        const __m256 x0 = _mm256_loadu_ps(x + j + 0);
        const __m256 q0 = _mm256_loadu_ps(q + j + 0);
        const __m256 x1 = _mm256_loadu_ps(x + j + 8);
        const __m256 q1 = _mm256_loadu_ps(q + j + 8);
        const __m256 x2 = _mm256_loadu_ps(x + j + 16);
        const __m256 q2 = _mm256_loadu_ps(q + j + 16);
        const __m256 x3 = _mm256_loadu_ps(x + j + 24);
        const __m256 q3 = _mm256_loadu_ps(q + j + 24);
#ifdef __FMA__
        s0 = _mm256_fmadd_ps(x0, q0, s0);
        s1 = _mm256_fmadd_ps(x1, q1, s1);
        s2 = _mm256_fmadd_ps(x2, q2, s2);
        s3 = _mm256_fmadd_ps(x3, q3, s3);
#else
        s0 = _mm256_add_ps(s0, _mm256_mul_ps(x0, q0));
        s1 = _mm256_add_ps(s1, _mm256_mul_ps(x1, q1));
        s2 = _mm256_add_ps(s2, _mm256_mul_ps(x2, q2));
        s3 = _mm256_add_ps(s3, _mm256_mul_ps(x3, q3));
#endif
    }
    const __m256 s01 = _mm256_add_ps(s0, s1);
    const __m256 s23 = _mm256_add_ps(s2, s3);
    const __m256 sum = _mm256_add_ps(s01, s23);
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float dot = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
            lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; j < dim; ++j) {
        dot += x[j] * q[j];
    }
    return dot;
#else
    float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
    for (int j = 0; j < dim; ++j) {
        dot += x[j] * q[j];
    }
    return dot;
#endif
}

std::string method_name(const Args& args);

uint64_t read_proc_self_read_bytes() {
    std::ifstream in("/proc/self/io");
    std::string key;
    uint64_t value = 0;
    while (in >> key >> value) {
        if (key == "read_bytes:") return value;
    }
    return 0;
}

struct ProcessMemoryIoSnapshot {
    uint64_t minor_faults = 0;
    uint64_t major_faults = 0;
    uint64_t read_bytes = 0;
};

ProcessMemoryIoSnapshot process_memory_io_snapshot() {
    struct rusage usage {};
    if (::getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed: " + std::string(std::strerror(errno)));
    }
    ProcessMemoryIoSnapshot snapshot;
    snapshot.minor_faults = static_cast<uint64_t>(usage.ru_minflt);
    snapshot.major_faults = static_cast<uint64_t>(usage.ru_majflt);
    snapshot.read_bytes = read_proc_self_read_bytes();
    return snapshot;
}

struct FpRerankThreadBuffer {
    std::vector<float> row;
    std::vector<const float*> batch_rows;
    std::vector<float> page_cache_rows;
    void* direct_ptr = nullptr;
    size_t direct_capacity = 0;
    void* batch_direct_ptr = nullptr;
    size_t batch_direct_capacity = 0;
    io_uring uring{};
    bool uring_initialized = false;
    unsigned uring_depth = 0;

    FpRerankThreadBuffer() = default;
    FpRerankThreadBuffer(const FpRerankThreadBuffer&) = delete;
    FpRerankThreadBuffer& operator=(const FpRerankThreadBuffer&) = delete;

    ~FpRerankThreadBuffer() {
        if (uring_initialized) io_uring_queue_exit(&uring);
        if (direct_ptr) std::free(direct_ptr);
        if (batch_direct_ptr) std::free(batch_direct_ptr);
    }

    io_uring& ensure_uring(unsigned requested_depth) {
        requested_depth = std::max(2u, requested_depth);
        if (uring_initialized && uring_depth >= requested_depth) return uring;
        if (uring_initialized) {
            io_uring_queue_exit(&uring);
            uring = {};
            uring_initialized = false;
            uring_depth = 0;
        }
        const int rc = io_uring_queue_init(requested_depth, &uring, 0);
        if (rc < 0) {
            throw std::runtime_error(std::string("io_uring_queue_init failed: ") +
                    std::strerror(-rc));
        }
        uring_initialized = true;
        uring_depth = requested_depth;
        return uring;
    }

    void ensure_row(size_t floats) {
        if (row.size() < floats) row.resize(floats);
    }

    void ensure_direct(size_t bytes) {
        bytes = round_up_to_alignment(bytes, 4096);
        if (direct_capacity >= bytes) return;
        if (direct_ptr) {
            std::free(direct_ptr);
            direct_ptr = nullptr;
            direct_capacity = 0;
        }
        if (::posix_memalign(&direct_ptr, 4096, bytes) != 0) {
            throw std::runtime_error("posix_memalign failed for O_DIRECT rerank buffer");
        }
        direct_capacity = bytes;
    }

    void ensure_batch_direct(size_t bytes) {
        bytes = round_up_to_alignment(bytes, 4096);
        if (batch_direct_capacity >= bytes) return;
        if (batch_direct_ptr) {
            std::free(batch_direct_ptr);
            batch_direct_ptr = nullptr;
            batch_direct_capacity = 0;
        }
        if (::posix_memalign(&batch_direct_ptr, 4096, bytes) != 0) {
            throw std::runtime_error("posix_memalign failed for O_DIRECT batch rerank buffer");
        }
        batch_direct_capacity = bytes;
    }

    void ensure_page_cache_rows(size_t floats) {
        if (page_cache_rows.size() < floats) page_cache_rows.resize(floats);
    }
};

constexpr size_t kFpPageCachePageBytes = 4096;
using FpPageCachePage = std::array<char, kFpPageCachePageBytes>;

struct FpUringRowRequest {
    size_t group_index = 0;
    size_t delta = 0;
};

struct FpUringIoGroup {
    off_t aligned_off = 0;
    size_t read_bytes = 0;
    size_t need_bytes = 0;
    size_t buffer_offset = 0;
};

struct FpUringBatch {
    std::vector<faiss::idx_t> ids;
    std::vector<FpUringRowRequest> row_requests;
    std::vector<FpUringIoGroup> groups;
    std::vector<const float*> rows;
    int outstanding = 0;
    bool active = false;

    void clear() {
        ids.clear();
        row_requests.clear();
        groups.clear();
        rows.clear();
        outstanding = 0;
        active = false;
    }
};

struct FpUringFinishStats {
    uint64_t completions_ready_before_wait = 0;
    double wait_seconds = 0.0;
};

struct FpRerankReader {
    const FbinMmap& train;
    const std::vector<uint32_t>* physical_row_by_id = nullptr;
    FpRerankSource source;
    bool cache_enabled = false;
    size_t cache_limit_rows = 0;
    size_t page_cache_limit_pages = 0;
    size_t io_merge_max_bytes = 0;
    int fd = -1;
    size_t row_bytes = 0;
    size_t row_cache_capacity = 0;
    mutable std::unique_ptr<std::atomic<int64_t>[]> row_cache_slot_by_id;
    mutable std::unique_ptr<float[]> row_cache_storage;
    mutable std::atomic<size_t> row_cache_rows{0};
    mutable std::mutex page_cache_mutex;
    mutable std::unordered_map<uint64_t, std::shared_ptr<FpPageCachePage>> page_cache;
    mutable std::deque<uint64_t> page_cache_fifo;
    mutable std::atomic<uint64_t> cache_hits{0};
    mutable std::atomic<uint64_t> cache_misses{0};
    mutable std::atomic<uint64_t> direct_row_requests{0};
    mutable std::atomic<uint64_t> direct_io_requests{0};
    mutable std::atomic<uint64_t> direct_logical_bytes{0};
    mutable std::atomic<uint64_t> direct_submitted_bytes{0};
    mutable std::atomic<uint64_t> page_cache_accesses{0};
    mutable std::atomic<uint64_t> page_cache_hits{0};
    mutable std::atomic<uint64_t> page_cache_misses{0};
    mutable std::atomic<uint64_t> page_cache_evictions{0};

    FpRerankReader(
            const FbinMmap& train_in,
            FpRerankSource source_in,
            bool cache_enabled_in,
            size_t cache_limit_rows_in,
            size_t page_cache_mib_in,
            const std::vector<uint32_t>* physical_row_by_id_in,
            size_t io_merge_max_bytes_in)
            : train(train_in),
              physical_row_by_id(physical_row_by_id_in),
              source(source_in),
              cache_enabled(cache_enabled_in),
              cache_limit_rows(cache_limit_rows_in),
              page_cache_limit_pages(
                      page_cache_mib_in == 0
                              ? 0
                              : std::max<size_t>(
                                        1,
                                        (page_cache_mib_in * 1024ull * 1024ull) /
                                                kFpPageCachePageBytes)),
              io_merge_max_bytes(io_merge_max_bytes_in),
              row_bytes(static_cast<size_t>(train_in.d) * sizeof(float)) {
        if (cache_enabled) {
            row_cache_capacity = cache_limit_rows > 0
                    ? std::min(cache_limit_rows, static_cast<size_t>(train.n))
                    : static_cast<size_t>(train.n);
            row_cache_slot_by_id.reset(
                    new std::atomic<int64_t>[static_cast<size_t>(train.n)]);
            for (size_t id = 0; id < static_cast<size_t>(train.n); ++id) {
                row_cache_slot_by_id[id].store(-1, std::memory_order_relaxed);
            }
            row_cache_storage.reset(
                    new float[row_cache_capacity * static_cast<size_t>(train.d)]);
        }
        if (source == FpRerankSource::Pread ||
            source == FpRerankSource::ODirect ||
            source == FpRerankSource::ODirectUring) {
            int flags = O_RDONLY;
            if (source == FpRerankSource::ODirect ||
                source == FpRerankSource::ODirectUring) {
                flags |= O_DIRECT;
            }
            fd = ::open(train.path.c_str(), flags);
            if (fd < 0) {
                throw std::runtime_error(std::string("open failed for FP rerank source ") +
                        fp_rerank_source_name(source) + ": " + train.path + ": " + std::strerror(errno));
            }
        }
    }

    FpRerankReader(const FpRerankReader&) = delete;
    FpRerankReader& operator=(const FpRerankReader&) = delete;

    ~FpRerankReader() {
        if (fd >= 0) ::close(fd);
    }

    const float* cache_get(size_t id) const {
        if (!cache_enabled) return nullptr;
        if (id >= static_cast<size_t>(train.n)) {
            throw std::runtime_error("FP row-cache id is out of range");
        }
        const int64_t slot =
                row_cache_slot_by_id[id].load(std::memory_order_acquire);
        if (slot < 0) return nullptr;
        cache_hits.fetch_add(1, std::memory_order_relaxed);
        return row_cache_storage.get() +
                static_cast<size_t>(slot) * static_cast<size_t>(train.d);
    }

    const float* cache_put(size_t id, const float* row_ptr) const {
        if (!cache_enabled) return nullptr;
        if (id >= static_cast<size_t>(train.n)) {
            throw std::runtime_error("FP row-cache id is out of range");
        }
        std::atomic<int64_t>& state = row_cache_slot_by_id[id];
        int64_t current = state.load(std::memory_order_acquire);
        if (current >= 0) {
            return row_cache_storage.get() +
                    static_cast<size_t>(current) * static_cast<size_t>(train.d);
        }

        // -1 means absent and -2 means another worker is publishing this row.
        // A racing reader treats -2 as a miss rather than waiting, so lookup is
        // lock-free and an SSD completion never blocks behind a cache writer.
        int64_t expected = -1;
        if (!state.compare_exchange_strong(
                    expected, -2, std::memory_order_acq_rel, std::memory_order_acquire)) {
            if (expected >= 0) {
                return row_cache_storage.get() +
                        static_cast<size_t>(expected) * static_cast<size_t>(train.d);
            }
            return nullptr;
        }

        size_t slot = row_cache_rows.load(std::memory_order_relaxed);
        while (slot < row_cache_capacity &&
               !row_cache_rows.compare_exchange_weak(
                       slot, slot + 1,
                       std::memory_order_relaxed,
                       std::memory_order_relaxed)) {
        }
        if (slot >= row_cache_capacity) {
            state.store(-1, std::memory_order_release);
            return nullptr;
        }
        float* destination = row_cache_storage.get() +
                slot * static_cast<size_t>(train.d);
        std::memcpy(destination, row_ptr, row_bytes);
        state.store(static_cast<int64_t>(slot), std::memory_order_release);
        return destination;
    }

    uint64_t cache_hit_count() const {
        return cache_hits.load(std::memory_order_relaxed);
    }

    uint64_t cache_miss_count() const {
        return cache_misses.load(std::memory_order_relaxed);
    }

    size_t cache_size() const {
        if (!cache_enabled) return 0;
        return std::min(
                row_cache_rows.load(std::memory_order_relaxed),
                row_cache_capacity);
    }

    std::shared_ptr<FpPageCachePage> page_cache_get(uint64_t page_no) const {
        if (page_cache_limit_pages == 0) return {};
        std::lock_guard<std::mutex> lock(page_cache_mutex);
        const auto it = page_cache.find(page_no);
        return it == page_cache.end() ? std::shared_ptr<FpPageCachePage>{} : it->second;
    }

    std::shared_ptr<FpPageCachePage> page_cache_put(
            uint64_t page_no,
            const char* page_data) const {
        auto value = std::make_shared<FpPageCachePage>();
        std::memcpy(value->data(), page_data, kFpPageCachePageBytes);
        std::lock_guard<std::mutex> lock(page_cache_mutex);
        const auto existing = page_cache.find(page_no);
        if (existing != page_cache.end()) return existing->second;
        while (page_cache.size() >= page_cache_limit_pages && !page_cache_fifo.empty()) {
            const uint64_t victim = page_cache_fifo.front();
            page_cache_fifo.pop_front();
            if (page_cache.erase(victim) > 0) {
                page_cache_evictions.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
        page_cache.emplace(page_no, value);
        page_cache_fifo.push_back(page_no);
        return value;
    }

    size_t page_cache_size() const {
        if (page_cache_limit_pages == 0) return 0;
        std::lock_guard<std::mutex> lock(page_cache_mutex);
        return page_cache.size();
    }

    uint64_t page_cache_access_count() const {
        return page_cache_accesses.load(std::memory_order_relaxed);
    }

    uint64_t page_cache_hit_count() const {
        return page_cache_hits.load(std::memory_order_relaxed);
    }

    uint64_t page_cache_miss_count() const {
        return page_cache_misses.load(std::memory_order_relaxed);
    }

    uint64_t page_cache_eviction_count() const {
        return page_cache_evictions.load(std::memory_order_relaxed);
    }

    uint64_t direct_row_request_count() const {
        return direct_row_requests.load(std::memory_order_relaxed);
    }

    uint64_t direct_io_request_count() const {
        return direct_io_requests.load(std::memory_order_relaxed);
    }

    uint64_t direct_logical_byte_count() const {
        return direct_logical_bytes.load(std::memory_order_relaxed);
    }

    uint64_t direct_submitted_byte_count() const {
        return direct_submitted_bytes.load(std::memory_order_relaxed);
    }

    size_t physical_row(size_t id) const {
        if (!physical_row_by_id) return id;
        if (id >= physical_row_by_id->size()) {
            throw std::runtime_error("FP rerank id exceeds colocation map");
        }
        return static_cast<size_t>((*physical_row_by_id)[id]);
    }

    off_t row_offset(size_t id) const {
        return static_cast<off_t>(train.data_offset + physical_row(id) * row_bytes);
    }

    void read_full(void* dst, size_t bytes, off_t offset) const {
        char* out = static_cast<char*>(dst);
        size_t done = 0;
        while (done < bytes) {
            const ssize_t got = ::pread(fd, out + done, bytes - done, offset + static_cast<off_t>(done));
            if (got < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("pread failed for FP rerank: ") + std::strerror(errno));
            }
            if (got == 0) throw std::runtime_error("short pread for FP rerank");
            done += static_cast<size_t>(got);
        }
    }

    void read_direct_at_least(void* dst, size_t read_bytes, off_t offset, size_t need_bytes) const {
        char* out = static_cast<char*>(dst);
        size_t done = 0;
        while (done < read_bytes) {
            const ssize_t got = ::pread(fd, out + done, read_bytes - done, offset + static_cast<off_t>(done));
            if (got < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("O_DIRECT pread failed for FP rerank: ") + std::strerror(errno));
            }
            if (got == 0) break;
            done += static_cast<size_t>(got);
            if (done >= need_bytes) break;
        }
        if (done < need_bytes) throw std::runtime_error("short O_DIRECT pread for FP rerank");
    }

    const float* row(size_t id, FpRerankThreadBuffer& buffer) const {
        if (source == FpRerankSource::Ram || source == FpRerankSource::Mmap) {
            return train.row(physical_row(id));
        }
        if (const float* cached = cache_get(id)) return cached;
        if (cache_enabled) cache_misses.fetch_add(1, std::memory_order_relaxed);
        const off_t off = row_offset(id);
        if (source == FpRerankSource::Pread) {
            buffer.ensure_row(static_cast<size_t>(train.d));
            read_full(buffer.row.data(), row_bytes, off);
            if (const float* inserted = cache_put(id, buffer.row.data())) return inserted;
            return buffer.row.data();
        }

        static constexpr size_t kAlign = 4096;
        const off_t aligned_off = static_cast<off_t>((static_cast<uint64_t>(off) / kAlign) * kAlign);
        const size_t delta = static_cast<size_t>(off - aligned_off);
        const size_t need = delta + row_bytes;
        const size_t direct_read = round_up_to_alignment(need, kAlign);
        buffer.ensure_direct(direct_read);
        read_direct_at_least(buffer.direct_ptr, direct_read, aligned_off, need);
        const float* result = reinterpret_cast<const float*>(static_cast<const char*>(buffer.direct_ptr) + delta);
        if (const float* inserted = cache_put(id, result)) return inserted;
        return result;
    }

    void submit_rows_uring_async(
            const faiss::idx_t* ids,
            int count,
            int queue_depth,
            FpRerankThreadBuffer& buffer,
            FpUringBatch& batch) const {
        if (source != FpRerankSource::ODirectUring) {
            throw std::runtime_error("async FP batch requires the odirect-uring source");
        }
        if (page_cache_limit_pages > 0) {
            throw std::runtime_error("async FP batch does not support the page cache");
        }
        if (batch.active) {
            throw std::runtime_error("cannot submit a second FP batch while one is active");
        }

        batch.clear();
        if (count <= 0) return;
        batch.ids.assign(ids, ids + count);
        batch.row_requests.resize(static_cast<size_t>(count));
        batch.groups.reserve(static_cast<size_t>(count));

        static constexpr size_t kAlign = 4096;
        uint64_t logical_bytes = 0;
        for (int i = 0; i < count; ++i) {
            const off_t off = row_offset(static_cast<size_t>(ids[i]));
            const off_t aligned_off = static_cast<off_t>(
                    (static_cast<uint64_t>(off) / kAlign) * kAlign);
            const size_t delta = static_cast<size_t>(off - aligned_off);
            const size_t need = delta + row_bytes;
            const size_t direct_read = round_up_to_alignment(need, kAlign);
            logical_bytes += direct_read;
            const uint64_t request_end = static_cast<uint64_t>(aligned_off) + direct_read;
            const uint64_t needed_end = static_cast<uint64_t>(off) + row_bytes;

            const uint64_t previous_end = batch.groups.empty()
                    ? 0
                    : static_cast<uint64_t>(batch.groups.back().aligned_off) +
                            batch.groups.back().read_bytes;
            const size_t merged_bytes = batch.groups.empty()
                    ? direct_read
                    : static_cast<size_t>(std::max(request_end, previous_end) -
                            static_cast<uint64_t>(batch.groups.back().aligned_off));
            const bool merge_with_previous = physical_row_by_id && io_merge_max_bytes > 0 &&
                    !batch.groups.empty() && static_cast<uint64_t>(aligned_off) <= previous_end &&
                    merged_bytes <= io_merge_max_bytes;
            if (!merge_with_previous) {
                FpUringIoGroup group;
                group.aligned_off = aligned_off;
                group.read_bytes = direct_read;
                group.need_bytes = need;
                batch.groups.push_back(group);
            } else {
                FpUringIoGroup& group = batch.groups.back();
                group.read_bytes = std::max(
                        group.read_bytes,
                        static_cast<size_t>(request_end -
                                static_cast<uint64_t>(group.aligned_off)));
                group.need_bytes = std::max(
                        group.need_bytes,
                        static_cast<size_t>(needed_end -
                                static_cast<uint64_t>(group.aligned_off)));
            }
            FpUringRowRequest& request = batch.row_requests[static_cast<size_t>(i)];
            request.group_index = batch.groups.size() - 1;
            request.delta = static_cast<size_t>(
                    static_cast<uint64_t>(off) -
                    static_cast<uint64_t>(batch.groups.back().aligned_off));
        }

        size_t total_buffer_bytes = 0;
        for (FpUringIoGroup& group : batch.groups) {
            group.buffer_offset = total_buffer_bytes;
            total_buffer_bytes += group.read_bytes;
        }
        const int group_count = static_cast<int>(batch.groups.size());
        if (queue_depth < group_count) {
            throw std::runtime_error("FP io_uring queue depth is smaller than the submitted batch");
        }

        buffer.ensure_batch_direct(total_buffer_bytes);
        char* base = static_cast<char*>(buffer.batch_direct_ptr);
        io_uring& ring = buffer.ensure_uring(static_cast<unsigned>(queue_depth));
        for (int gi = 0; gi < group_count; ++gi) {
            const FpUringIoGroup& group = batch.groups[static_cast<size_t>(gi)];
            io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (!sqe) throw std::runtime_error("io_uring_get_sqe failed for async FP batch");
            io_uring_prep_read(
                    sqe,
                    fd,
                    base + group.buffer_offset,
                    static_cast<unsigned>(group.read_bytes),
                    group.aligned_off);
            io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(gi));
        }
        const int submitted = io_uring_submit(&ring);
        if (submitted != group_count) {
            throw std::runtime_error("io_uring_submit submitted fewer async FP requests than expected");
        }

        direct_row_requests.fetch_add(static_cast<uint64_t>(count), std::memory_order_relaxed);
        direct_io_requests.fetch_add(static_cast<uint64_t>(group_count), std::memory_order_relaxed);
        direct_logical_bytes.fetch_add(logical_bytes, std::memory_order_relaxed);
        direct_submitted_bytes.fetch_add(total_buffer_bytes, std::memory_order_relaxed);
        batch.outstanding = group_count;
        batch.active = true;
    }

    uint64_t poll_rows_uring_async(
            FpRerankThreadBuffer& buffer,
            FpUringBatch& batch) const {
        if (!batch.active) return 0;
        io_uring& ring = buffer.ensure_uring(buffer.uring_depth);
        uint64_t completed = 0;
        while (batch.outstanding > 0) {
            io_uring_cqe* cqe = nullptr;
            const int rc = io_uring_peek_cqe(&ring, &cqe);
            if (rc == -EAGAIN) break;
            if (rc < 0) {
                throw std::runtime_error(std::string("io_uring_peek_cqe failed: ") +
                        std::strerror(-rc));
            }
            const int group_index = static_cast<int>(io_uring_cqe_get_data64(cqe));
            if (group_index < 0 || group_index >= static_cast<int>(batch.groups.size())) {
                io_uring_cqe_seen(&ring, cqe);
                throw std::runtime_error("io_uring returned an invalid async FP request id");
            }
            const FpUringIoGroup& group = batch.groups[static_cast<size_t>(group_index)];
            if (cqe->res < 0) {
                const int err = -cqe->res;
                io_uring_cqe_seen(&ring, cqe);
                throw std::runtime_error(std::string("async FP O_DIRECT read failed: ") +
                        std::strerror(err));
            }
            if (static_cast<size_t>(cqe->res) < group.need_bytes) {
                io_uring_cqe_seen(&ring, cqe);
                throw std::runtime_error("short async FP O_DIRECT read");
            }
            io_uring_cqe_seen(&ring, cqe);
            --batch.outstanding;
            ++completed;
        }
        return completed;
    }

    FpUringFinishStats finish_rows_uring_async(
            FpRerankThreadBuffer& buffer,
            FpUringBatch& batch) const {
        FpUringFinishStats stats;
        if (!batch.active) return stats;
        io_uring& ring = buffer.ensure_uring(buffer.uring_depth);
        char* base = static_cast<char*>(buffer.batch_direct_ptr);

        stats.completions_ready_before_wait = poll_rows_uring_async(buffer, batch);

        auto consume = [&](io_uring_cqe* cqe) {
            const int group_index = static_cast<int>(io_uring_cqe_get_data64(cqe));
            if (group_index < 0 || group_index >= static_cast<int>(batch.groups.size())) {
                io_uring_cqe_seen(&ring, cqe);
                throw std::runtime_error("io_uring returned an invalid async FP request id");
            }
            const FpUringIoGroup& group = batch.groups[static_cast<size_t>(group_index)];
            if (cqe->res < 0) {
                const int err = -cqe->res;
                io_uring_cqe_seen(&ring, cqe);
                throw std::runtime_error(std::string("async FP O_DIRECT read failed: ") +
                        std::strerror(err));
            }
            if (static_cast<size_t>(cqe->res) < group.need_bytes) {
                io_uring_cqe_seen(&ring, cqe);
                throw std::runtime_error("short async FP O_DIRECT read");
            }
            io_uring_cqe_seen(&ring, cqe);
            --batch.outstanding;
        };

        const auto wait_start = std::chrono::steady_clock::now();
        while (batch.outstanding > 0) {
            io_uring_cqe* cqe = nullptr;
            const int rc = io_uring_wait_cqe(&ring, &cqe);
            if (rc < 0) {
                throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") +
                        std::strerror(-rc));
            }
            consume(cqe);
        }
        stats.wait_seconds = seconds_since(wait_start);

        batch.rows.resize(batch.ids.size());
        for (size_t i = 0; i < batch.ids.size(); ++i) {
            const FpUringRowRequest& request = batch.row_requests[i];
            const FpUringIoGroup& group = batch.groups[request.group_index];
            batch.rows[i] = reinterpret_cast<const float*>(
                    base + group.buffer_offset + request.delta);
        }
        batch.active = false;
        return stats;
    }

    void rows_uring_page_cached(
            const faiss::idx_t* ids,
            int count,
            FpRerankThreadBuffer& buffer,
            std::vector<const float*>& rows) const {
        struct RowSpan {
            uint64_t offset = 0;
            uint64_t first_page = 0;
            uint64_t last_page = 0;
        };

        rows.resize(static_cast<size_t>(count));
        if (count <= 0) return;

        std::vector<RowSpan> spans(static_cast<size_t>(count));
        std::unordered_map<uint64_t, std::shared_ptr<FpPageCachePage>> available_pages;
        std::unordered_map<uint64_t, size_t> pending_page_index;
        std::vector<uint64_t> missing_pages;
        available_pages.reserve(static_cast<size_t>(count) * 2);
        pending_page_index.reserve(static_cast<size_t>(count) * 2);
        missing_pages.reserve(static_cast<size_t>(count) * 2);

        uint64_t accesses = 0;
        uint64_t hits = 0;
        uint64_t misses = 0;
        for (int i = 0; i < count; ++i) {
            const uint64_t offset = static_cast<uint64_t>(
                    row_offset(static_cast<size_t>(ids[i])));
            RowSpan& span = spans[static_cast<size_t>(i)];
            span.offset = offset;
            span.first_page = offset / kFpPageCachePageBytes;
            span.last_page = (offset + row_bytes - 1) / kFpPageCachePageBytes;
            for (uint64_t page_no = span.first_page; page_no <= span.last_page; ++page_no) {
                ++accesses;
                if (available_pages.find(page_no) != available_pages.end() ||
                    pending_page_index.find(page_no) != pending_page_index.end()) {
                    ++hits;
                    continue;
                }
                if (auto cached = page_cache_get(page_no)) {
                    available_pages.emplace(page_no, std::move(cached));
                    ++hits;
                    continue;
                }
                pending_page_index.emplace(page_no, missing_pages.size());
                missing_pages.push_back(page_no);
                ++misses;
            }
        }
        page_cache_accesses.fetch_add(accesses, std::memory_order_relaxed);
        page_cache_hits.fetch_add(hits, std::memory_order_relaxed);
        page_cache_misses.fetch_add(misses, std::memory_order_relaxed);

        const size_t miss_count = missing_pages.size();
        const size_t submitted_bytes = miss_count * kFpPageCachePageBytes;
        direct_row_requests.fetch_add(accesses, std::memory_order_relaxed);
        direct_io_requests.fetch_add(miss_count, std::memory_order_relaxed);
        direct_logical_bytes.fetch_add(
                accesses * kFpPageCachePageBytes,
                std::memory_order_relaxed);
        direct_submitted_bytes.fetch_add(submitted_bytes, std::memory_order_relaxed);

        if (miss_count > 0) {
            buffer.ensure_batch_direct(submitted_bytes);
            char* base = static_cast<char*>(buffer.batch_direct_ptr);
            const uint64_t file_bytes = static_cast<uint64_t>(train.mapping_size);

            const int qd = std::max<int>(2, static_cast<int>(miss_count));
            io_uring& ring = buffer.ensure_uring(static_cast<unsigned>(qd));
            int rc = 0;

            try {
                for (size_t mi = 0; mi < miss_count; ++mi) {
                    io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                    if (!sqe) throw std::runtime_error("io_uring_get_sqe failed");
                    const uint64_t offset = missing_pages[mi] * kFpPageCachePageBytes;
                    io_uring_prep_read(
                            sqe,
                            fd,
                            base + mi * kFpPageCachePageBytes,
                            static_cast<unsigned>(kFpPageCachePageBytes),
                            static_cast<off_t>(offset));
                    io_uring_sqe_set_data64(sqe, mi);
                }

                const int submitted = io_uring_submit(&ring);
                if (submitted != static_cast<int>(miss_count)) {
                    throw std::runtime_error("io_uring_submit submitted fewer page requests than expected");
                }

                for (size_t done = 0; done < miss_count; ++done) {
                    io_uring_cqe* cqe = nullptr;
                    rc = io_uring_wait_cqe(&ring, &cqe);
                    if (rc < 0) {
                        throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") +
                                std::strerror(-rc));
                    }
                    const size_t miss_index = static_cast<size_t>(io_uring_cqe_get_data64(cqe));
                    if (miss_index >= miss_count) {
                        io_uring_cqe_seen(&ring, cqe);
                        throw std::runtime_error("io_uring returned invalid page request id");
                    }
                    if (cqe->res < 0) {
                        const int err = -cqe->res;
                        io_uring_cqe_seen(&ring, cqe);
                        throw std::runtime_error(std::string("io_uring page read failed: ") +
                                std::strerror(err));
                    }
                    const uint64_t offset = missing_pages[miss_index] * kFpPageCachePageBytes;
                    const size_t need_bytes = static_cast<size_t>(
                            std::min<uint64_t>(kFpPageCachePageBytes, file_bytes - offset));
                    if (static_cast<size_t>(cqe->res) < need_bytes) {
                        io_uring_cqe_seen(&ring, cqe);
                        throw std::runtime_error("short io_uring page read for FP rerank");
                    }
                    io_uring_cqe_seen(&ring, cqe);
                }

                for (size_t mi = 0; mi < miss_count; ++mi) {
                    const uint64_t page_no = missing_pages[mi];
                    auto page = page_cache_put(
                            page_no,
                            base + mi * kFpPageCachePageBytes);
                    available_pages.emplace(page_no, std::move(page));
                }
            } catch (...) {
                throw;
            }
        }

        buffer.ensure_page_cache_rows(
                static_cast<size_t>(count) * static_cast<size_t>(train.d));
        for (int i = 0; i < count; ++i) {
            char* destination = reinterpret_cast<char*>(
                    buffer.page_cache_rows.data() + static_cast<size_t>(i) * train.d);
            uint64_t cursor = spans[static_cast<size_t>(i)].offset;
            size_t remaining = row_bytes;
            while (remaining > 0) {
                const uint64_t page_no = cursor / kFpPageCachePageBytes;
                const size_t page_delta = static_cast<size_t>(cursor % kFpPageCachePageBytes);
                const size_t chunk = std::min(remaining, kFpPageCachePageBytes - page_delta);
                const auto page_it = available_pages.find(page_no);
                if (page_it == available_pages.end()) {
                    throw std::runtime_error("FP rerank page missing after io_uring completion");
                }
                std::memcpy(
                        destination + (row_bytes - remaining),
                        page_it->second->data() + page_delta,
                        chunk);
                cursor += chunk;
                remaining -= chunk;
            }
            rows[static_cast<size_t>(i)] =
                    buffer.page_cache_rows.data() + static_cast<size_t>(i) * train.d;
        }
    }

    double rows_uring(
            const faiss::idx_t* ids,
            int count,
            FpRerankThreadBuffer& buffer,
            std::vector<const float*>& rows) const {
        if (source != FpRerankSource::ODirectUring) {
            throw std::runtime_error("rows_uring called for non-uring source");
        }
        if (page_cache_limit_pages > 0) {
            rows_uring_page_cached(ids, count, buffer, rows);
            return 0.0;
        }
        rows.resize(static_cast<size_t>(count));
        if (count <= 0) return 0.0;

        std::vector<int> miss_positions;
        miss_positions.reserve(static_cast<size_t>(count));
        if (cache_enabled) {
            for (int i = 0; i < count; ++i) {
                const size_t id = static_cast<size_t>(ids[i]);
                if (const float* cached = cache_get(id)) {
                    rows[static_cast<size_t>(i)] = cached;
                } else {
                    cache_misses.fetch_add(1, std::memory_order_relaxed);
                    miss_positions.push_back(i);
                    rows[static_cast<size_t>(i)] = nullptr;
                }
            }
            if (miss_positions.empty()) return 0.0;
        } else {
            miss_positions.resize(static_cast<size_t>(count));
            for (int i = 0; i < count; ++i) miss_positions[static_cast<size_t>(i)] = i;
        }

        static constexpr size_t kAlign = 4096;
        const int miss_count = static_cast<int>(miss_positions.size());
        struct RowRequest {
            size_t group_index = 0;
            size_t delta = 0;
        };
        struct IoGroup {
            off_t aligned_off = 0;
            size_t read_bytes = 0;
            size_t need_bytes = 0;
            size_t buffer_offset = 0;
        };
        std::vector<RowRequest> row_requests(static_cast<size_t>(miss_count));
        std::vector<IoGroup> groups;
        groups.reserve(static_cast<size_t>(miss_count));

        uint64_t logical_bytes = 0;
        // Candidate IDs arrive sorted by physical row. For a colocated store,
        // merge overlapping or adjacent aligned ranges, but cap each merged read
        // so io_uring retains enough queue depth. Without colocation (or with a
        // zero cap), preserve the original one-request-per-row baseline.
        for (int mi = 0; mi < miss_count; ++mi) {
            const int original_i = miss_positions[static_cast<size_t>(mi)];
            const off_t off = row_offset(static_cast<size_t>(ids[original_i]));
            const off_t aligned_off = static_cast<off_t>(
                    (static_cast<uint64_t>(off) / kAlign) * kAlign);
            const size_t delta = static_cast<size_t>(off - aligned_off);
            const size_t need = delta + row_bytes;
            const size_t direct_read = round_up_to_alignment(need, kAlign);
            logical_bytes += direct_read;
            const uint64_t request_end = static_cast<uint64_t>(aligned_off) + direct_read;
            const uint64_t needed_end = static_cast<uint64_t>(off) + row_bytes;

            const uint64_t previous_end = groups.empty()
                    ? 0
                    : static_cast<uint64_t>(groups.back().aligned_off) + groups.back().read_bytes;
            const size_t merged_bytes = groups.empty()
                    ? direct_read
                    : static_cast<size_t>(std::max(request_end, previous_end) -
                            static_cast<uint64_t>(groups.back().aligned_off));
            const bool merge_with_previous = physical_row_by_id && io_merge_max_bytes > 0 &&
                    !groups.empty() && static_cast<uint64_t>(aligned_off) <= previous_end &&
                    merged_bytes <= io_merge_max_bytes;
            if (!merge_with_previous) {
                IoGroup group;
                group.aligned_off = aligned_off;
                group.read_bytes = direct_read;
                group.need_bytes = need;
                groups.push_back(group);
            } else {
                IoGroup& group = groups.back();
                group.read_bytes = std::max(
                        group.read_bytes,
                        static_cast<size_t>(request_end - static_cast<uint64_t>(group.aligned_off)));
                group.need_bytes = std::max(
                        group.need_bytes,
                        static_cast<size_t>(needed_end - static_cast<uint64_t>(group.aligned_off)));
            }
            RowRequest& request = row_requests[static_cast<size_t>(mi)];
            request.group_index = groups.size() - 1;
            request.delta = static_cast<size_t>(
                    static_cast<uint64_t>(off) -
                    static_cast<uint64_t>(groups.back().aligned_off));
        }

        size_t total_buffer_bytes = 0;
        for (IoGroup& group : groups) {
            group.buffer_offset = total_buffer_bytes;
            total_buffer_bytes += group.read_bytes;
        }
        direct_row_requests.fetch_add(static_cast<uint64_t>(miss_count), std::memory_order_relaxed);
        direct_io_requests.fetch_add(static_cast<uint64_t>(groups.size()), std::memory_order_relaxed);
        direct_logical_bytes.fetch_add(logical_bytes, std::memory_order_relaxed);
        direct_submitted_bytes.fetch_add(total_buffer_bytes, std::memory_order_relaxed);
        buffer.ensure_batch_direct(total_buffer_bytes);
        char* base = static_cast<char*>(buffer.batch_direct_ptr);

        const int group_count = static_cast<int>(groups.size());
        const int qd = std::max(2, group_count);
        io_uring& ring = buffer.ensure_uring(static_cast<unsigned>(qd));
        int rc = 0;
        double io_wait_seconds = 0.0;

        try {
            for (int gi = 0; gi < group_count; ++gi) {
                const IoGroup& group = groups[static_cast<size_t>(gi)];
                io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                if (!sqe) throw std::runtime_error("io_uring_get_sqe failed");
                io_uring_prep_read(
                        sqe,
                        fd,
                        base + group.buffer_offset,
                        static_cast<unsigned>(group.read_bytes),
                        group.aligned_off);
                io_uring_sqe_set_data64(sqe, static_cast<uint64_t>(gi));
            }

            const auto submit_started = std::chrono::steady_clock::now();
            int submitted = io_uring_submit(&ring);
            io_wait_seconds += seconds_since(submit_started);
            if (submitted != group_count) {
                throw std::runtime_error("io_uring_submit submitted fewer requests than expected");
            }

            for (int done = 0; done < group_count; ++done) {
                io_uring_cqe* cqe = nullptr;
                const auto wait_started = std::chrono::steady_clock::now();
                rc = io_uring_wait_cqe(&ring, &cqe);
                io_wait_seconds += seconds_since(wait_started);
                if (rc < 0) {
                    throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") + std::strerror(-rc));
                }
                const int group_index = static_cast<int>(io_uring_cqe_get_data64(cqe));
                if (group_index < 0 || group_index >= group_count) {
                    io_uring_cqe_seen(&ring, cqe);
                    throw std::runtime_error("io_uring returned invalid request id");
                }
                const IoGroup& group = groups[static_cast<size_t>(group_index)];
                if (cqe->res < 0) {
                    const int err = -cqe->res;
                    io_uring_cqe_seen(&ring, cqe);
                    throw std::runtime_error(std::string("io_uring O_DIRECT read failed: ") + std::strerror(err));
                }
                if (static_cast<size_t>(cqe->res) < group.need_bytes) {
                    io_uring_cqe_seen(&ring, cqe);
                    throw std::runtime_error("short io_uring O_DIRECT read for FP rerank");
                }
                io_uring_cqe_seen(&ring, cqe);
            }

            for (int mi = 0; mi < miss_count; ++mi) {
                const int original_i = miss_positions[static_cast<size_t>(mi)];
                const RowRequest& request = row_requests[static_cast<size_t>(mi)];
                const IoGroup& group = groups[request.group_index];
                const float* result = reinterpret_cast<const float*>(
                        base + group.buffer_offset + request.delta);
                if (const float* inserted = cache_put(static_cast<size_t>(ids[original_i]), result)) {
                    rows[static_cast<size_t>(original_i)] = inserted;
                } else {
                    rows[static_cast<size_t>(original_i)] = result;
                }
            }
        } catch (...) {
            throw;
        }
        return io_wait_seconds;
    }

    void prefetch(size_t id) const {
        if (source == FpRerankSource::Ram || source == FpRerankSource::Mmap) {
            prefetch_float_row(train.row(physical_row(id)), train.d);
        }
    }
};

float score_fp_rerank_row(
        const float* x,
        const float* q,
        int dim,
        RerankDotMode dot_mode,
        FpRerankMetric metric,
        FpRerankNormMode norm_mode,
        double precomputed_norm) {
    const double dot = dot_mode == RerankDotMode::Fp32
            ? static_cast<double>(dot_product_fp32(x, q, dim))
            : dot_product_fp64(x, q, dim);
    if (metric == FpRerankMetric::InnerProduct) {
        return static_cast<float>(dot);
    }
    double norm = precomputed_norm;
    if (norm_mode == FpRerankNormMode::OnFetch) {
        double ss = 0.0;
#pragma omp simd reduction(+ : ss)
        for (int j = 0; j < dim; ++j) {
            ss += static_cast<double>(x[j]) * x[j];
        }
        norm = std::sqrt(std::max(ss, 1e-30));
    }
    return static_cast<float>(dot / norm);
}

struct CounterSummary {
    double mean = 0.0;
    double p50 = 0.0;
    double p95 = 0.0;
    double max = 0.0;
};

CounterSummary summarize_counter(std::vector<uint64_t> values) {
    CounterSummary out;
    if (values.empty()) return out;
    std::sort(values.begin(), values.end());
    const long double sum = std::accumulate(values.begin(), values.end(), 0.0L);
    out.mean = static_cast<double>(sum / values.size());
    auto percentile = [&](double pct) -> double {
        if (values.size() == 1) return static_cast<double>(values[0]);
        const double pos = pct * static_cast<double>(values.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(lo + 1, values.size() - 1);
        const double frac = pos - static_cast<double>(lo);
        return static_cast<double>(values[lo]) * (1.0 - frac) + static_cast<double>(values[hi]) * frac;
    };
    out.p50 = percentile(0.50);
    out.p95 = percentile(0.95);
    out.max = static_cast<double>(values.back());
    return out;
}

std::vector<uint64_t> collect_counter(
        const std::vector<AdcSearchStats>& stats,
        uint64_t AdcSearchStats::* member) {
    std::vector<uint64_t> values;
    values.reserve(stats.size());
    for (const auto& s : stats) values.push_back(s.*member);
    return values;
}

void write_counter_columns(std::ostream& os, const CounterSummary& summary) {
    os << ',' << summary.mean << ',' << summary.p50 << ',' << summary.p95 << ',' << summary.max;
}

void write_stats_header(std::ostream& os) {
    os << "dataset,method,k,nq,nb,dim,M,efConstruction,efSearch,rerank_enabled,rerank_n,rounds,"
       << "search_seconds_mean,rerank_seconds_mean,total_seconds_mean,recall_at_k,qps";
    const char* names[] = {
        "adaptive_effective_ef",
        "adaptive_shrunk",
        "adaptive_early_stopped",
        "adaptive_classified",
        "upper_hop_count",
        "upper_distance_eval_count",
        "base_pop_count",
        "base_expand_count",
        "base_neighbor_scan_count",
        "base_visited_count",
        "base_duplicate_visit_count",
        "base_candidate_insert_count",
        "base_result_insert_count",
        "base_distance_eval_count",
        "nhops",
        "ndis",
    };
    for (const char* name : names) {
        os << ',' << name << "_mean"
           << ',' << name << "_p50"
           << ',' << name << "_p95"
           << ',' << name << "_max";
    }
    os << "\n";
}

void write_adcbq_stats_row(
        std::ostream& os,
        const Args& args,
        int nq,
        int nb,
        int dim,
        int ef,
        double search_mean,
        double rerank_mean,
        double total_mean,
        double recall_mean,
        double qps_mean,
        const std::vector<AdcSearchStats>& stats) {
    os << args.dataset << ',' << method_name(args) << ','
       << args.k << ',' << nq << ',' << nb << ',' << dim << ',' << args.m << ','
       << args.ef_construction << ',' << ef << ',' << (args.no_rerank ? "false" : "true")
       << ',' << (args.no_rerank ? 0 : args.rerank_n) << ',' << args.rounds
       << ',' << search_mean << ',' << rerank_mean << ',' << total_mean
       << ',' << recall_mean << ',' << qps_mean;
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::adaptive_effective_ef)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::adaptive_shrunk)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::adaptive_early_stopped)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::adaptive_classified)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::upper_hop_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::upper_distance_eval_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::base_pop_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::base_expand_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::base_neighbor_scan_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::base_visited_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::base_duplicate_visit_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::base_candidate_insert_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::base_result_insert_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::base_distance_eval_count)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::nhops)));
    write_counter_columns(os, summarize_counter(collect_counter(stats, &AdcSearchStats::ndis)));
    os << "\n";
}


struct Int8RerankIndex {
    int64_t n = 0;
    int dim = 0;
    std::vector<int8_t> codes;
    std::vector<float> scales;
    std::vector<float> inv_norms;

    size_t bytes() const {
        return codes.size() * sizeof(int8_t) +
                scales.size() * sizeof(float) +
                inv_norms.size() * sizeof(float);
    }

    const int8_t* row(size_t i) const {
        return codes.data() + i * static_cast<size_t>(dim);
    }
};

void prefetch_int8_rerank_row(const Int8RerankIndex& index, faiss::idx_t id) {
    const size_t row_id = static_cast<size_t>(id);
    __builtin_prefetch(index.row(row_id), 0, 1);
    __builtin_prefetch(index.scales.data() + row_id, 0, 1);
    __builtin_prefetch(index.inv_norms.data() + row_id, 0, 1);
}

inline int8_t quantize_i8_value(float v, float inv_scale) {
    int q = static_cast<int>(std::lrintf(v * inv_scale));
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return static_cast<int8_t>(q);
}

Int8RerankIndex build_int8_rerank_index(const FbinMmap& train, int threads) {
    Int8RerankIndex out;
    out.n = train.n;
    out.dim = train.d;
    out.codes.resize(static_cast<size_t>(train.n) * static_cast<size_t>(train.d));
    out.scales.resize(static_cast<size_t>(train.n));
    out.inv_norms.resize(static_cast<size_t>(train.n));

    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < train.n; ++i) {
        const float* x = train.row(static_cast<size_t>(i));
        float max_abs = 0.0f;
        double ss = 0.0;
        for (int j = 0; j < train.d; ++j) {
            const float v = x[j];
            max_abs = std::max(max_abs, std::abs(v));
            ss += static_cast<double>(v) * v;
        }
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        const float inv_scale = 1.0f / scale;
        out.scales[static_cast<size_t>(i)] = scale;
        out.inv_norms[static_cast<size_t>(i)] =
                1.0f / static_cast<float>(std::sqrt(std::max(ss, 1e-30)));
        int8_t* code = out.codes.data() + static_cast<size_t>(i) * static_cast<size_t>(train.d);
        for (int j = 0; j < train.d; ++j) {
            code[j] = quantize_i8_value(x[j], inv_scale);
        }
    }
    return out;
}

void quantize_query_i8(const float* q, int dim, std::vector<int8_t>& out) {
    out.resize(static_cast<size_t>(dim));
    float max_abs = 0.0f;
    for (int j = 0; j < dim; ++j) max_abs = std::max(max_abs, std::abs(q[j]));
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
    const float inv_scale = 1.0f / scale;
    for (int j = 0; j < dim; ++j) out[static_cast<size_t>(j)] = quantize_i8_value(q[j], inv_scale);
}

int32_t dot_product_i8_scalar(const int8_t* x, const int8_t* q, int dim) {
    int32_t dot = 0;
#pragma omp simd reduction(+ : dot)
    for (int j = 0; j < dim; ++j) {
        dot += static_cast<int32_t>(x[j]) * static_cast<int32_t>(q[j]);
    }
    return dot;
}

int32_t dot_product_i8(const int8_t* x, const int8_t* q, int dim) {
#ifdef __AVX2__
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    int j = 0;
    for (; j + 32 <= dim; j += 32) {
        const __m128i xlo8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + j));
        const __m128i qlo8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + j));
        const __m128i xhi8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x + j + 16));
        const __m128i qhi8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + j + 16));
        const __m256i xlo16 = _mm256_cvtepi8_epi16(xlo8);
        const __m256i qlo16 = _mm256_cvtepi8_epi16(qlo8);
        const __m256i xhi16 = _mm256_cvtepi8_epi16(xhi8);
        const __m256i qhi16 = _mm256_cvtepi8_epi16(qhi8);
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(xlo16, qlo16));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(xhi16, qhi16));
    }
    __m256i acc = _mm256_add_epi32(acc0, acc1);
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    int32_t dot = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
            lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; j < dim; ++j) {
        dot += static_cast<int32_t>(x[j]) * static_cast<int32_t>(q[j]);
    }
    return dot;
#else
    return dot_product_i8_scalar(x, q, dim);
#endif
}


struct AdaptiveRerankSelection {
    std::vector<faiss::idx_t> labels;
    std::vector<int> counts;
    std::vector<double> selection_seconds;
    double mean_count = 0.0;
    int min_count = 0;
    double p95_count = 0.0;
    double p99_count = 0.0;
    int max_count = 0;
};

AdaptiveRerankSelection summarize_rerank_counts(const std::vector<int>& counts) {
    AdaptiveRerankSelection out;
    if (counts.empty()) return out;
    out.counts = counts;
    out.min_count = *std::min_element(counts.begin(), counts.end());
    out.max_count = *std::max_element(counts.begin(), counts.end());
    out.mean_count = static_cast<double>(
            std::accumulate(counts.begin(), counts.end(), int64_t{0})) /
            static_cast<double>(counts.size());
    std::vector<int> sorted_counts = counts;
    std::sort(sorted_counts.begin(), sorted_counts.end());
    auto percentile = [&](double q) -> double {
        const double position = q * static_cast<double>(sorted_counts.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(position));
        const size_t hi = static_cast<size_t>(std::ceil(position));
        const double fraction = position - static_cast<double>(lo);
        return static_cast<double>(sorted_counts[lo]) * (1.0 - fraction) +
                static_cast<double>(sorted_counts[hi]) * fraction;
    };
    out.p95_count = percentile(0.95);
    out.p99_count = percentile(0.99);
    return out;
}

AdaptiveRerankSelection select_fixed_rerank_candidates(
        const std::vector<faiss::idx_t>& labels,
        int nq,
        int candidate_k,
        int rerank_n) {
    AdaptiveRerankSelection out;
    out.labels = labels;
    std::vector<int> counts(static_cast<size_t>(nq), 0);
    std::vector<double> selection_seconds(static_cast<size_t>(nq), 0.0);
    for (int qi = 0; qi < nq; ++qi) {
        const auto selection_start = std::chrono::steady_clock::now();
        const size_t base = static_cast<size_t>(qi) * candidate_k;
        int valid = 0;
        while (valid < candidate_k && out.labels[base + static_cast<size_t>(valid)] >= 0) {
            ++valid;
        }
        const int take = std::min(valid, rerank_n);
        counts[static_cast<size_t>(qi)] = take;
        for (int ci = take; ci < candidate_k; ++ci) {
            out.labels[base + static_cast<size_t>(ci)] = -1;
        }
        selection_seconds[static_cast<size_t>(qi)] = seconds_since(selection_start);
    }
    AdaptiveRerankSelection stats = summarize_rerank_counts(counts);
    stats.labels = std::move(out.labels);
    stats.selection_seconds = std::move(selection_seconds);
    return stats;
}

int calibrated_threshold_rerank_count(
        const faiss::idx_t* labels,
        const float* distances,
        int candidate_k,
        int k,
        float tau,
        float distance_scale) {
    int valid = 0;
    while (valid < candidate_k && labels[valid] >= 0 &&
           std::isfinite(distances[valid])) {
        ++valid;
    }
    if (valid <= 0) return 0;
    const int min_take = std::min(k, valid);
    if (valid <= min_take) return valid;
    const float cutoff = distances[min_take - 1] + tau * distance_scale;
    int take = min_take;
    while (take < valid && distances[take] <= cutoff) ++take;
    return take;
}

AdaptiveRerankSelection select_calibrated_threshold_rerank_candidates(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<float>& distances,
        const float* queries,
        int nq,
        int dim,
        int candidate_k,
        int k,
        float tau,
        AdcMode adc_mode,
        LutMode lut_mode) {
    AdaptiveRerankSelection out;
    out.labels = labels;
    std::vector<int> counts(static_cast<size_t>(nq), 0);
    std::vector<double> selection_seconds(static_cast<size_t>(nq), 0.0);
    PackedAsymBQDistanceComputer scale_computer(
            nullptr,
            static_cast<size_t>((dim + 7) / 8),
            static_cast<size_t>((dim + 7) / 8),
            dim,
            adc_mode,
            lut_mode);
    for (int qi = 0; qi < nq; ++qi) {
        const auto selection_start = std::chrono::steady_clock::now();
        const size_t base = static_cast<size_t>(qi) * candidate_k;
        const float distance_scale = adc_mode == AdcMode::Int16
                ? scale_computer.int16_lut_scale(
                          queries + static_cast<size_t>(qi) * dim)
                : 1.0f;
        const int take = calibrated_threshold_rerank_count(
                labels.data() + base,
                distances.data() + base,
                candidate_k,
                k,
                tau,
                distance_scale);
        counts[static_cast<size_t>(qi)] = take;
        for (int ci = take; ci < candidate_k; ++ci) {
            out.labels[base + static_cast<size_t>(ci)] = -1;
        }
        selection_seconds[static_cast<size_t>(qi)] = seconds_since(selection_start);
    }
    AdaptiveRerankSelection stats = summarize_rerank_counts(counts);
    stats.labels = std::move(out.labels);
    stats.selection_seconds = std::move(selection_seconds);
    return stats;
}

AdaptiveRerankSelection select_adaptive_rerank_candidates(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<float>& distances,
        int nq,
        int candidate_k,
        int k,
        int min_rerank,
        float alpha) {
    AdaptiveRerankSelection out;
    out.labels = labels;
    std::vector<int> counts(static_cast<size_t>(nq), 0);
    std::vector<double> selection_seconds(static_cast<size_t>(nq), 0.0);

    for (int qi = 0; qi < nq; ++qi) {
        const auto selection_start = std::chrono::steady_clock::now();
        const size_t base = static_cast<size_t>(qi) * candidate_k;
        int valid = 0;
        while (valid < candidate_k) {
            const faiss::idx_t id = labels[base + valid];
            const float d = distances[base + valid];
            if (id < 0 || !std::isfinite(d)) break;
            ++valid;
        }

        int take = std::min(valid, candidate_k);
        if (take > 0) {
            const int min_take = std::min(std::max(k, min_rerank), take);
            if (alpha < 1.0f && take > min_take) {
                const float anchor = distances[base + min_take - 1];
                const float tail = distances[base + take - 1];
                const float width = std::max(0.0f, tail - anchor);
                const float cutoff = anchor + alpha * width;
                take = min_take;
                while (take < valid && distances[base + take] <= cutoff) {
                    ++take;
                }
            }
        }

        for (int ci = take; ci < candidate_k; ++ci) {
            out.labels[base + ci] = -1;
        }
        counts[static_cast<size_t>(qi)] = take;
        selection_seconds[static_cast<size_t>(qi)] = seconds_since(selection_start);
    }

    AdaptiveRerankSelection stats = summarize_rerank_counts(counts);
    stats.labels = std::move(out.labels);
    stats.selection_seconds = std::move(selection_seconds);
    return stats;
}

int gather_sorted_candidate_ids(
        const std::vector<faiss::idx_t>& labels,
        size_t label_base,
        int candidate_k,
        faiss::idx_t* ids,
        const std::vector<uint32_t>* physical_row_by_id = nullptr) {
    int count = 0;
    for (int ci = 0; ci < candidate_k; ++ci) {
        const faiss::idx_t id = labels[label_base + ci];
        if (id >= 0) ids[count++] = id;
    }
    if (physical_row_by_id) {
        std::sort(ids, ids + count, [&](faiss::idx_t lhs, faiss::idx_t rhs) {
            const uint32_t lhs_row = (*physical_row_by_id)[static_cast<size_t>(lhs)];
            const uint32_t rhs_row = (*physical_row_by_id)[static_cast<size_t>(rhs)];
            return lhs_row < rhs_row || (lhs_row == rhs_row && lhs < rhs);
        });
    } else {
        std::sort(ids, ids + count);
    }
    return count;
}

std::vector<faiss::idx_t> rerank_int8_ram(
        const Int8RerankIndex& int8_index,
        const float* queries,
        int nq,
        const std::vector<faiss::idx_t>& labels,
        int candidate_k,
        int k) {
    std::vector<faiss::idx_t> out(static_cast<size_t>(nq) * k, -1);
    constexpr int prefetch_ahead = 4;
    constexpr int max_stack_k = 64;
    constexpr int max_stack_candidates = 256;

    auto better = [](float lhs_score, faiss::idx_t lhs_id, float rhs_score, faiss::idx_t rhs_id) {
        return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_id < rhs_id);
    };

#pragma omp parallel for schedule(static)
    for (int qi = 0; qi < nq; ++qi) {
        const float* q = queries + static_cast<size_t>(qi) * int8_index.dim;
        std::vector<int8_t> q_code;
        quantize_query_i8(q, int8_index.dim, q_code);
        const size_t label_base = static_cast<size_t>(qi) * candidate_k;
        std::array<faiss::idx_t, max_stack_candidates> sorted_ids_stack;
        std::vector<faiss::idx_t> sorted_ids_heap;
        faiss::idx_t* sorted_ids = sorted_ids_stack.data();
        if (candidate_k > max_stack_candidates) {
            sorted_ids_heap.resize(static_cast<size_t>(candidate_k));
            sorted_ids = sorted_ids_heap.data();
        }
        const int sorted_count = gather_sorted_candidate_ids(labels, label_base, candidate_k, sorted_ids);

        if (k <= max_stack_k) {
            std::array<float, max_stack_k> top_scores;
            std::array<faiss::idx_t, max_stack_k> top_ids;
            for (int j = 0; j < k; ++j) {
                top_scores[j] = -std::numeric_limits<float>::infinity();
                top_ids[j] = std::numeric_limits<faiss::idx_t>::max();
            }
            for (int ci = 0; ci < std::min(sorted_count, prefetch_ahead); ++ci) {
                prefetch_int8_rerank_row(int8_index, sorted_ids[ci]);
            }
            for (int ci = 0; ci < sorted_count; ++ci) {
                const int next = ci + prefetch_ahead;
                if (next < sorted_count) {
                    prefetch_int8_rerank_row(int8_index, sorted_ids[next]);
                }
                const faiss::idx_t id = sorted_ids[ci];
                const int32_t dot = dot_product_i8(
                        int8_index.row(static_cast<size_t>(id)),
                        q_code.data(),
                        int8_index.dim);
                const float score = static_cast<float>(dot) *
                        int8_index.scales[static_cast<size_t>(id)] *
                        int8_index.inv_norms[static_cast<size_t>(id)];
                if (!better(score, id, top_scores[k - 1], top_ids[k - 1])) continue;
                int pos = k - 1;
                while (pos > 0 && better(score, id, top_scores[pos - 1], top_ids[pos - 1])) {
                    top_scores[pos] = top_scores[pos - 1];
                    top_ids[pos] = top_ids[pos - 1];
                    --pos;
                }
                top_scores[pos] = score;
                top_ids[pos] = id;
            }
            for (int j = 0; j < k; ++j) {
                if (top_ids[j] != std::numeric_limits<faiss::idx_t>::max()) {
                    out[static_cast<size_t>(qi) * k + j] = top_ids[j];
                }
            }
        } else {
            std::vector<std::pair<float, faiss::idx_t>> scored;
            scored.reserve(sorted_count);
            for (int ci = 0; ci < sorted_count; ++ci) {
                const faiss::idx_t id = sorted_ids[ci];
                const int32_t dot = dot_product_i8(
                        int8_index.row(static_cast<size_t>(id)),
                        q_code.data(),
                        int8_index.dim);
                const float score = static_cast<float>(dot) *
                        int8_index.scales[static_cast<size_t>(id)] *
                        int8_index.inv_norms[static_cast<size_t>(id)];
                scored.emplace_back(score, id);
            }
            const int take = std::min<int>(k, scored.size());
            std::partial_sort(
                    scored.begin(),
                    scored.begin() + take,
                    scored.end(),
                    [](const auto& a, const auto& b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second < b.second;
                    });
            for (int j = 0; j < take; ++j) {
                out[static_cast<size_t>(qi) * k + j] = scored[j].second;
            }
        }
    }
    return out;
}

struct FpRerankCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    size_t rows = 0;
    size_t bytes = 0;
    uint64_t direct_rows = 0;
    uint64_t direct_requests = 0;
    uint64_t direct_logical_bytes = 0;
    uint64_t direct_submitted_bytes = 0;
    size_t page_cache_pages = 0;
    size_t page_cache_bytes = 0;
    uint64_t page_cache_accesses = 0;
    uint64_t page_cache_hits = 0;
    uint64_t page_cache_misses = 0;
    uint64_t page_cache_evictions = 0;
};

struct FpIntraQueryPipelineStats {
    uint64_t queries = 0;
    uint64_t queries_with_hit_miss_overlap = 0;
    uint64_t cache_hit_rows_scored_while_io = 0;
    uint64_t io_requests = 0;
    uint64_t completions_ready_after_hit_scoring = 0;
    double io_wait_seconds = 0.0;
    double hit_score_seconds = 0.0;
    std::vector<double> per_query_io_wait_seconds;
    std::vector<double> per_query_rerank_seconds;
};

struct StrictQueryPipelineResult {
    std::vector<float> distances;
    std::vector<faiss::idx_t> candidates;
    std::vector<faiss::idx_t> final_labels;
    std::vector<int> selected_counts;
    std::vector<double> search_seconds;
    std::vector<double> selection_seconds;
    std::vector<double> io_wait_seconds;
    std::vector<double> rerank_seconds;
    std::vector<double> end_to_end_seconds;
    FpRerankCacheStats cache_stats;
    FpIntraQueryPipelineStats intra_query_stats;
    AdaptiveRerankSelection count_summary;
    double wall_seconds = 0.0;
};

StrictQueryPipelineResult search_rerank_strict_query_local(
        const faiss::IndexBinaryHNSW& index,
        const FbinMmap& fp_store,
        const float* queries,
        int nq,
        int dim,
        int candidate_k,
        int k,
        int ef,
        AdcMode adc_mode,
        LutMode lut_mode,
        bool check_relative_distance,
        bool early_abandon,
        bool ordered_early_abandon,
        int ordered_early_abandon_probe_bytes,
        int prefetch_lookahead,
        const AdcAdaptiveCfrConfig* adaptive_cfr,
        const uint8_t* codes,
        size_t code_stride,
        const uint8_t* block32_codes,
        size_t block32_block_stride,
        bool calibrated_threshold,
        float calibrated_tau,
        int fixed_rerank_n,
        RerankDotMode dot_mode,
        FpRerankMetric rerank_metric,
        int io_queue_depth,
        const std::vector<uint32_t>* original_id_by_colocated_id,
        const std::vector<uint32_t>* physical_row_by_id,
        size_t colocation_io_merge_bytes) {
    if (!codes) {
        codes = storage_codes(index);
        code_stride = index.code_size;
    }

    StrictQueryPipelineResult result;
    const size_t candidate_rows = static_cast<size_t>(nq) * candidate_k;
    result.distances.assign(candidate_rows, std::numeric_limits<float>::infinity());
    result.candidates.assign(candidate_rows, -1);
    result.final_labels.assign(static_cast<size_t>(nq) * k, -1);
    result.selected_counts.assign(static_cast<size_t>(nq), 0);
    result.search_seconds.assign(static_cast<size_t>(nq), 0.0);
    result.selection_seconds.assign(static_cast<size_t>(nq), 0.0);
    result.io_wait_seconds.assign(static_cast<size_t>(nq), 0.0);
    result.rerank_seconds.assign(static_cast<size_t>(nq), 0.0);
    result.end_to_end_seconds.assign(static_cast<size_t>(nq), 0.0);

    faiss::VisitedTable visited(index.ntotal, false);
    faiss::MinimaxHeap candidate_heap(std::max(ef, candidate_k));
    PackedAsymBQDistanceComputer distance_computer(
            codes,
            index.code_size,
            code_stride,
            dim,
            adc_mode,
            lut_mode,
            early_abandon,
            ordered_early_abandon,
            static_cast<size_t>(ordered_early_abandon_probe_bytes),
            block32_codes,
            block32_block_stride);
    FpRerankReader reader(
            fp_store,
            FpRerankSource::ODirectUring,
            false,
            0,
            0,
            physical_row_by_id,
            colocation_io_merge_bytes);
    FpRerankThreadBuffer fp_buffer;
    std::vector<faiss::idx_t> sorted_ids(static_cast<size_t>(candidate_k));
    const auto wall_start = std::chrono::steady_clock::now();
    uint64_t ready_completions = 0;

    auto better = [](float lhs_score, faiss::idx_t lhs_id,
                     float rhs_score, faiss::idx_t rhs_id) {
        return lhs_score > rhs_score ||
                (lhs_score == rhs_score && lhs_id < rhs_id);
    };

    for (int qi = 0; qi < nq; ++qi) {
        const auto query_start = std::chrono::steady_clock::now();
        const float* query = queries + static_cast<size_t>(qi) * dim;
        const size_t candidate_base = static_cast<size_t>(qi) * candidate_k;

        const auto search_start = std::chrono::steady_clock::now();
        distance_computer.set_query(query);
        search_one_packed_inline(
                index,
                distance_computer,
                visited,
                candidate_k,
                std::max(ef, candidate_k),
                ef,
                check_relative_distance,
                early_abandon,
                prefetch_lookahead,
                adaptive_cfr,
                candidate_heap,
                result.distances.data() + candidate_base,
                result.candidates.data() + candidate_base,
                nullptr);
        result.search_seconds[static_cast<size_t>(qi)] = seconds_since(search_start);

        const auto selection_start = std::chrono::steady_clock::now();
        if (original_id_by_colocated_id) {
            for (int ci = 0; ci < candidate_k; ++ci) {
                faiss::idx_t& id = result.candidates[candidate_base + ci];
                if (id >= 0) {
                    id = (*original_id_by_colocated_id)[static_cast<size_t>(id)];
                }
            }
        }
        int take = 0;
        if (calibrated_threshold) {
            const float distance_scale = adc_mode == AdcMode::Int16
                    ? distance_computer.int16_lut_scale(query)
                    : 1.0f;
            take = calibrated_threshold_rerank_count(
                    result.candidates.data() + candidate_base,
                    result.distances.data() + candidate_base,
                    candidate_k,
                    k,
                    calibrated_tau,
                    distance_scale);
        } else {
            int valid = 0;
            while (valid < candidate_k &&
                   result.candidates[candidate_base + valid] >= 0) {
                ++valid;
            }
            take = std::min(valid, fixed_rerank_n);
        }
        result.selected_counts[static_cast<size_t>(qi)] = take;
        for (int ci = take; ci < candidate_k; ++ci) {
            result.candidates[candidate_base + ci] = -1;
        }
        result.selection_seconds[static_cast<size_t>(qi)] =
                seconds_since(selection_start);

        const auto rerank_start = std::chrono::steady_clock::now();
        const int sorted_count = gather_sorted_candidate_ids(
                result.candidates,
                candidate_base,
                candidate_k,
                sorted_ids.data(),
                physical_row_by_id);
        FpUringBatch batch;
        reader.submit_rows_uring_async(
                sorted_ids.data(),
                sorted_count,
                io_queue_depth,
                fp_buffer,
                batch);
        const FpUringFinishStats finish =
                reader.finish_rows_uring_async(fp_buffer, batch);
        ready_completions += finish.completions_ready_before_wait;
        result.io_wait_seconds[static_cast<size_t>(qi)] = finish.wait_seconds;

        std::vector<float> top_scores(
                static_cast<size_t>(k),
                -std::numeric_limits<float>::infinity());
        std::vector<faiss::idx_t> top_ids(
                static_cast<size_t>(k),
                std::numeric_limits<faiss::idx_t>::max());
        for (int ci = 0; ci < sorted_count; ++ci) {
            const faiss::idx_t id = sorted_ids[static_cast<size_t>(ci)];
            const float score = score_fp_rerank_row(
                    batch.rows[static_cast<size_t>(ci)],
                    query,
                    dim,
                    dot_mode,
                    rerank_metric,
                    FpRerankNormMode::OnFetch,
                    1.0);
            if (!better(score, id, top_scores.back(), top_ids.back())) continue;
            int pos = k - 1;
            while (pos > 0 && better(
                    score, id,
                    top_scores[static_cast<size_t>(pos - 1)],
                    top_ids[static_cast<size_t>(pos - 1)])) {
                top_scores[static_cast<size_t>(pos)] =
                        top_scores[static_cast<size_t>(pos - 1)];
                top_ids[static_cast<size_t>(pos)] =
                        top_ids[static_cast<size_t>(pos - 1)];
                --pos;
            }
            top_scores[static_cast<size_t>(pos)] = score;
            top_ids[static_cast<size_t>(pos)] = id;
        }
        for (int rank = 0; rank < k; ++rank) {
            if (top_ids[static_cast<size_t>(rank)] !=
                std::numeric_limits<faiss::idx_t>::max()) {
                result.final_labels[static_cast<size_t>(qi) * k + rank] =
                        top_ids[static_cast<size_t>(rank)];
            }
        }
        result.rerank_seconds[static_cast<size_t>(qi)] =
                seconds_since(rerank_start);
        result.end_to_end_seconds[static_cast<size_t>(qi)] =
                seconds_since(query_start);
    }
    result.wall_seconds = seconds_since(wall_start);
    result.count_summary = summarize_rerank_counts(result.selected_counts);

    result.cache_stats.hits = reader.cache_hit_count();
    result.cache_stats.misses = reader.cache_miss_count();
    result.cache_stats.rows = reader.cache_size();
    result.cache_stats.bytes =
            reader.cache_size() * static_cast<size_t>(fp_store.d) * sizeof(float);
    result.cache_stats.direct_rows = reader.direct_row_request_count();
    result.cache_stats.direct_requests = reader.direct_io_request_count();
    result.cache_stats.direct_logical_bytes = reader.direct_logical_byte_count();
    result.cache_stats.direct_submitted_bytes = reader.direct_submitted_byte_count();
    result.cache_stats.page_cache_pages = reader.page_cache_size();
    result.cache_stats.page_cache_bytes =
            reader.page_cache_size() * kFpPageCachePageBytes;
    result.cache_stats.page_cache_accesses = reader.page_cache_access_count();
    result.cache_stats.page_cache_hits = reader.page_cache_hit_count();
    result.cache_stats.page_cache_misses = reader.page_cache_miss_count();
    result.cache_stats.page_cache_evictions = reader.page_cache_eviction_count();

    result.intra_query_stats.queries = static_cast<uint64_t>(nq);
    result.intra_query_stats.io_requests = result.cache_stats.direct_requests;
    result.intra_query_stats.completions_ready_after_hit_scoring = ready_completions;
    result.intra_query_stats.io_wait_seconds = std::accumulate(
            result.io_wait_seconds.begin(), result.io_wait_seconds.end(), 0.0);
    result.intra_query_stats.per_query_io_wait_seconds = result.io_wait_seconds;
    result.intra_query_stats.per_query_rerank_seconds = result.rerank_seconds;
    return result;
}

std::vector<faiss::idx_t> rerank_cosine(
        const FbinMmap& fp_store,
        const std::vector<double>& train_norms,
        const float* queries,
        int nq,
        const std::vector<faiss::idx_t>& labels,
        int candidate_k,
        int k,
        RerankDotMode dot_mode,
        FpRerankMetric rerank_metric,
        FpRerankSource fp_source,
        FpRerankNormMode norm_mode,
        bool fp_cache,
        size_t fp_cache_rows,
        size_t fp_page_cache_mib,
        bool overlap_cache_hits,
        int io_queue_depth,
        const std::vector<uint32_t>* physical_row_by_id,
        size_t colocation_io_merge_bytes,
        FpRerankCacheStats* cache_stats,
        FpIntraQueryPipelineStats* intra_query_stats) {
    std::vector<faiss::idx_t> out(static_cast<size_t>(nq) * k, -1);
    constexpr int prefetch_ahead = 4;
    constexpr int max_stack_k = 64;
    constexpr int max_stack_candidates = 256;

    if (rerank_metric == FpRerankMetric::Cosine &&
        norm_mode == FpRerankNormMode::Precompute &&
        train_norms.size() != static_cast<size_t>(fp_store.n)) {
        throw std::runtime_error("precomputed train norms are required for FP rerank norm mode precompute");
    }

    FpRerankReader reader(
            fp_store,
            fp_source,
            fp_cache,
            fp_cache_rows,
            fp_page_cache_mib,
            physical_row_by_id,
            colocation_io_merge_bytes);

    std::vector<uint64_t> intra_query_overlap(static_cast<size_t>(nq), 0);
    std::vector<uint64_t> intra_hit_rows(static_cast<size_t>(nq), 0);
    std::vector<uint64_t> intra_io_requests(static_cast<size_t>(nq), 0);
    std::vector<uint64_t> intra_ready_completions(static_cast<size_t>(nq), 0);
    std::vector<double> intra_wait_seconds(static_cast<size_t>(nq), 0.0);
    std::vector<double> intra_hit_score_seconds(static_cast<size_t>(nq), 0.0);
    std::vector<double> rerank_total_seconds(static_cast<size_t>(nq), 0.0);

    auto better = [](float lhs_score, faiss::idx_t lhs_id, float rhs_score, faiss::idx_t rhs_id) {
        return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_id < rhs_id);
    };

#pragma omp parallel
    {
        FpRerankThreadBuffer fp_buffer;
#pragma omp for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            const auto query_rerank_start = std::chrono::steady_clock::now();
            const float* q = queries + static_cast<size_t>(qi) * fp_store.d;
            const size_t label_base = static_cast<size_t>(qi) * candidate_k;
            std::array<faiss::idx_t, max_stack_candidates> sorted_ids_stack;
            std::vector<faiss::idx_t> sorted_ids_heap;
            faiss::idx_t* sorted_ids = sorted_ids_stack.data();
            if (candidate_k > max_stack_candidates) {
                sorted_ids_heap.resize(static_cast<size_t>(candidate_k));
                sorted_ids = sorted_ids_heap.data();
            }
            const int sorted_count = gather_sorted_candidate_ids(
                    labels, label_base, candidate_k, sorted_ids, physical_row_by_id);

            if (fp_source == FpRerankSource::ODirectUring) {
                if (overlap_cache_hits) {
                    std::vector<faiss::idx_t> hit_ids;
                    std::vector<const float*> hit_rows;
                    std::vector<faiss::idx_t> miss_ids;
                    hit_ids.reserve(static_cast<size_t>(sorted_count));
                    hit_rows.reserve(static_cast<size_t>(sorted_count));
                    miss_ids.reserve(static_cast<size_t>(sorted_count));
                    for (int ci = 0; ci < sorted_count; ++ci) {
                        const faiss::idx_t id = sorted_ids[ci];
                        if (const float* cached = reader.cache_get(static_cast<size_t>(id))) {
                            hit_ids.push_back(id);
                            hit_rows.push_back(cached);
                        } else {
                            reader.cache_misses.fetch_add(1, std::memory_order_relaxed);
                            miss_ids.push_back(id);
                        }
                    }

                    FpUringBatch miss_batch;
                    if (!miss_ids.empty()) {
                        reader.submit_rows_uring_async(
                                miss_ids.data(),
                                static_cast<int>(miss_ids.size()),
                                io_queue_depth,
                                fp_buffer,
                                miss_batch);
                    }
                    intra_io_requests[static_cast<size_t>(qi)] = miss_batch.groups.size();

                    std::vector<std::pair<float, faiss::idx_t>> scored;
                    scored.reserve(static_cast<size_t>(sorted_count));
                    const auto hit_score_start = std::chrono::steady_clock::now();
                    for (size_t hi = 0; hi < hit_ids.size(); ++hi) {
                        const faiss::idx_t id = hit_ids[hi];
                        const double norm = norm_mode == FpRerankNormMode::Precompute
                                ? train_norms[static_cast<size_t>(id)]
                                : 1.0;
                        scored.emplace_back(
                                score_fp_rerank_row(
                                        hit_rows[hi],
                                        q,
                                        fp_store.d,
                                        dot_mode,
                                        rerank_metric,
                                        norm_mode,
                                        norm),
                                id);
                    }
                    intra_hit_score_seconds[static_cast<size_t>(qi)] =
                            seconds_since(hit_score_start);

                    FpUringFinishStats finish_stats;
                    if (miss_batch.active) {
                        finish_stats = reader.finish_rows_uring_async(fp_buffer, miss_batch);
                    }
                    intra_ready_completions[static_cast<size_t>(qi)] =
                            finish_stats.completions_ready_before_wait;
                    intra_wait_seconds[static_cast<size_t>(qi)] = finish_stats.wait_seconds;
                    if (!hit_ids.empty() && !miss_ids.empty()) {
                        intra_query_overlap[static_cast<size_t>(qi)] = 1;
                        intra_hit_rows[static_cast<size_t>(qi)] = hit_ids.size();
                    }

                    for (size_t mi = 0; mi < miss_batch.ids.size(); ++mi) {
                        const faiss::idx_t id = miss_batch.ids[mi];
                        const float* row = miss_batch.rows[mi];
                        if (const float* inserted = reader.cache_put(static_cast<size_t>(id), row)) {
                            row = inserted;
                        }
                        const double norm = norm_mode == FpRerankNormMode::Precompute
                                ? train_norms[static_cast<size_t>(id)]
                                : 1.0;
                        scored.emplace_back(
                                score_fp_rerank_row(
                                        row,
                                        q,
                                        fp_store.d,
                                        dot_mode,
                                        rerank_metric,
                                        norm_mode,
                                        norm),
                                id);
                    }

                    const int take = std::min<int>(k, scored.size());
                    std::partial_sort(
                            scored.begin(),
                            scored.begin() + take,
                            scored.end(),
                            [](const auto& a, const auto& b) {
                                if (a.first != b.first) return a.first > b.first;
                                return a.second < b.second;
                            });
                    for (int rank = 0; rank < take; ++rank) {
                        out[static_cast<size_t>(qi) * k + rank] =
                                scored[static_cast<size_t>(rank)].second;
                    }
                    rerank_total_seconds[static_cast<size_t>(qi)] =
                            seconds_since(query_rerank_start);
                    continue;
                }

                intra_wait_seconds[static_cast<size_t>(qi)] = reader.rows_uring(
                        sorted_ids, sorted_count, fp_buffer, fp_buffer.batch_rows);
                auto score_pos = [&](int ci) -> float {
                    const faiss::idx_t id = sorted_ids[ci];
                    const float* x = fp_buffer.batch_rows[static_cast<size_t>(ci)];
                    const double norm = norm_mode == FpRerankNormMode::Precompute
                            ? train_norms[static_cast<size_t>(id)]
                            : 1.0;
                    return score_fp_rerank_row(
                            x, q, fp_store.d, dot_mode, rerank_metric, norm_mode, norm);
                };

                if (k <= max_stack_k) {
                    std::array<float, max_stack_k> top_scores;
                    std::array<faiss::idx_t, max_stack_k> top_ids;
                    for (int j = 0; j < k; ++j) {
                        top_scores[j] = -std::numeric_limits<float>::infinity();
                        top_ids[j] = std::numeric_limits<faiss::idx_t>::max();
                    }
                    for (int ci = 0; ci < sorted_count; ++ci) {
                        const faiss::idx_t id = sorted_ids[ci];
                        const float score = score_pos(ci);
                        if (!better(score, id, top_scores[k - 1], top_ids[k - 1])) continue;

                        int pos = k - 1;
                        while (pos > 0 && better(score, id, top_scores[pos - 1], top_ids[pos - 1])) {
                            top_scores[pos] = top_scores[pos - 1];
                            top_ids[pos] = top_ids[pos - 1];
                            --pos;
                        }
                        top_scores[pos] = score;
                        top_ids[pos] = id;
                    }
                    for (int j = 0; j < k; ++j) {
                        if (top_ids[j] != std::numeric_limits<faiss::idx_t>::max()) {
                            out[static_cast<size_t>(qi) * k + j] = top_ids[j];
                        }
                    }
                } else {
                    std::vector<std::pair<float, faiss::idx_t>> scored;
                    scored.reserve(sorted_count);
                    for (int ci = 0; ci < sorted_count; ++ci) {
                        scored.emplace_back(score_pos(ci), sorted_ids[ci]);
                    }
                    const int take = std::min<int>(k, scored.size());
                    std::partial_sort(
                            scored.begin(),
                            scored.begin() + take,
                            scored.end(),
                            [](const auto& a, const auto& b) {
                                if (a.first != b.first) return a.first > b.first;
                                return a.second < b.second;
                            });
                    for (int j = 0; j < take; ++j) {
                        out[static_cast<size_t>(qi) * k + j] = scored[j].second;
                    }
                }
                rerank_total_seconds[static_cast<size_t>(qi)] =
                        seconds_since(query_rerank_start);
                continue;
            }

            auto score_id = [&](faiss::idx_t id) -> float {
                const float* x = reader.row(static_cast<size_t>(id), fp_buffer);
                const double norm = norm_mode == FpRerankNormMode::Precompute
                        ? train_norms[static_cast<size_t>(id)]
                        : 1.0;
                return score_fp_rerank_row(
                        x, q, fp_store.d, dot_mode, rerank_metric, norm_mode, norm);
            };

            if (k <= max_stack_k) {
                std::array<float, max_stack_k> top_scores;
                std::array<faiss::idx_t, max_stack_k> top_ids;
                for (int j = 0; j < k; ++j) {
                    top_scores[j] = -std::numeric_limits<float>::infinity();
                    top_ids[j] = std::numeric_limits<faiss::idx_t>::max();
                }

                for (int ci = 0; ci < std::min(sorted_count, prefetch_ahead); ++ci) {
                    reader.prefetch(static_cast<size_t>(sorted_ids[ci]));
                }

                for (int ci = 0; ci < sorted_count; ++ci) {
                    const int next = ci + prefetch_ahead;
                    if (next < sorted_count) {
                        reader.prefetch(static_cast<size_t>(sorted_ids[next]));
                    }

                    const faiss::idx_t id = sorted_ids[ci];
                    const float score = score_id(id);
                    if (!better(score, id, top_scores[k - 1], top_ids[k - 1])) continue;

                    int pos = k - 1;
                    while (pos > 0 && better(score, id, top_scores[pos - 1], top_ids[pos - 1])) {
                        top_scores[pos] = top_scores[pos - 1];
                        top_ids[pos] = top_ids[pos - 1];
                        --pos;
                    }
                    top_scores[pos] = score;
                    top_ids[pos] = id;
                }
                for (int j = 0; j < k; ++j) {
                    if (top_ids[j] != std::numeric_limits<faiss::idx_t>::max()) {
                        out[static_cast<size_t>(qi) * k + j] = top_ids[j];
                    }
                }
            } else {
                std::vector<std::pair<float, faiss::idx_t>> scored;
                scored.reserve(sorted_count);
                for (int ci = 0; ci < sorted_count; ++ci) {
                    const faiss::idx_t id = sorted_ids[ci];
                    scored.emplace_back(score_id(id), id);
                }
                const int take = std::min<int>(k, scored.size());
                std::partial_sort(
                        scored.begin(),
                        scored.begin() + take,
                        scored.end(),
                        [](const auto& a, const auto& b) {
                            if (a.first != b.first) return a.first > b.first;
                            return a.second < b.second;
                        });
                for (int j = 0; j < take; ++j) {
                    out[static_cast<size_t>(qi) * k + j] = scored[j].second;
                }
            }
            rerank_total_seconds[static_cast<size_t>(qi)] =
                    seconds_since(query_rerank_start);
        }
    }
    if (cache_stats) {
        cache_stats->hits = reader.cache_hit_count();
        cache_stats->misses = reader.cache_miss_count();
        cache_stats->rows = reader.cache_size();
        cache_stats->bytes = reader.cache_size() * static_cast<size_t>(fp_store.d) * sizeof(float);
        cache_stats->direct_rows = reader.direct_row_request_count();
        cache_stats->direct_requests = reader.direct_io_request_count();
        cache_stats->direct_logical_bytes = reader.direct_logical_byte_count();
        cache_stats->direct_submitted_bytes = reader.direct_submitted_byte_count();
        cache_stats->page_cache_pages = reader.page_cache_size();
        cache_stats->page_cache_bytes =
                reader.page_cache_size() * kFpPageCachePageBytes;
        cache_stats->page_cache_accesses = reader.page_cache_access_count();
        cache_stats->page_cache_hits = reader.page_cache_hit_count();
        cache_stats->page_cache_misses = reader.page_cache_miss_count();
        cache_stats->page_cache_evictions = reader.page_cache_eviction_count();
    }
    if (intra_query_stats) {
        intra_query_stats->queries = static_cast<uint64_t>(nq);
        intra_query_stats->queries_with_hit_miss_overlap = std::accumulate(
                intra_query_overlap.begin(), intra_query_overlap.end(), uint64_t{0});
        intra_query_stats->cache_hit_rows_scored_while_io = std::accumulate(
                intra_hit_rows.begin(), intra_hit_rows.end(), uint64_t{0});
        intra_query_stats->io_requests = std::accumulate(
                intra_io_requests.begin(), intra_io_requests.end(), uint64_t{0});
        intra_query_stats->completions_ready_after_hit_scoring = std::accumulate(
                intra_ready_completions.begin(), intra_ready_completions.end(), uint64_t{0});
        intra_query_stats->io_wait_seconds = std::accumulate(
                intra_wait_seconds.begin(), intra_wait_seconds.end(), 0.0);
        intra_query_stats->hit_score_seconds = std::accumulate(
                intra_hit_score_seconds.begin(), intra_hit_score_seconds.end(), 0.0);
        intra_query_stats->per_query_io_wait_seconds = std::move(intra_wait_seconds);
        intra_query_stats->per_query_rerank_seconds = std::move(rerank_total_seconds);
    }
    return out;
}

struct FpInterQueryPipelineStats {
    uint64_t queries = 0;
    uint64_t queries_overlapped_with_search = 0;
    uint64_t io_requests = 0;
    uint64_t io_requests_overlapped_with_search = 0;
    uint64_t completions_ready_after_search = 0;
    double io_wait_seconds = 0.0;
    double score_seconds = 0.0;
};

std::vector<faiss::idx_t> search_rerank_interquery_pipeline(
        const faiss::IndexBinaryHNSW& index,
        const FbinMmap& fp_store,
        const std::vector<double>& train_norms,
        const float* queries,
        int nq,
        int dim,
        int candidate_k,
        int k,
        int ef,
        int threads,
        AdcMode adc_mode,
        LutMode lut_mode,
        bool check_relative_distance,
        bool early_abandon,
        bool ordered_early_abandon,
        int ordered_early_abandon_probe_bytes,
        int prefetch_lookahead,
        const AdcAdaptiveCfrConfig* adaptive_cfr,
        const uint8_t* codes,
        size_t code_stride,
        const uint8_t* block32_codes,
        size_t block32_block_stride,
        RerankDotMode dot_mode,
        FpRerankMetric rerank_metric,
        FpRerankNormMode norm_mode,
        bool cache_enabled,
        size_t cache_limit_rows,
        bool overlap_cache_hits,
        int fixed_rerank_n,
        bool calibrated_threshold_rerank,
        float calibrated_threshold_tau,
        int io_queue_depth,
        const std::vector<uint32_t>* original_id_by_colocated_id,
        const std::vector<uint32_t>* physical_row_by_id,
        size_t colocation_io_merge_bytes,
        std::vector<float>& distances,
        std::vector<faiss::idx_t>& candidates_out,
        std::vector<AdcSearchStats>* query_stats,
        FpRerankCacheStats* cache_stats,
        FpInterQueryPipelineStats* pipeline_stats,
        FpIntraQueryPipelineStats* intra_query_stats,
        AdaptiveRerankSelection* rerank_count_stats) {
    if (rerank_metric == FpRerankMetric::Cosine &&
        norm_mode == FpRerankNormMode::Precompute &&
        train_norms.size() != static_cast<size_t>(fp_store.n)) {
        throw std::runtime_error("precomputed train norms are required for pipelined FP rerank");
    }
    if (!codes) {
        codes = storage_codes(index);
        code_stride = index.code_size;
    }

    distances.assign(static_cast<size_t>(nq) * candidate_k, 0.0f);
    candidates_out.assign(static_cast<size_t>(nq) * candidate_k, -1);
    std::vector<faiss::idx_t> final_labels(static_cast<size_t>(nq) * k, -1);
    if (query_stats) query_stats->assign(static_cast<size_t>(nq), AdcSearchStats{});

    std::vector<uint64_t> request_counts(static_cast<size_t>(nq), 0);
    std::vector<uint64_t> overlapped_request_counts(static_cast<size_t>(nq), 0);
    std::vector<uint64_t> ready_counts(static_cast<size_t>(nq), 0);
    std::vector<uint8_t> overlapped(static_cast<size_t>(nq), 0);
    std::vector<double> wait_seconds(static_cast<size_t>(nq), 0.0);
    std::vector<double> score_seconds(static_cast<size_t>(nq), 0.0);
    std::vector<uint64_t> intra_query_overlap(static_cast<size_t>(nq), 0);
    std::vector<uint64_t> intra_hit_rows(static_cast<size_t>(nq), 0);
    std::vector<uint64_t> intra_io_requests(static_cast<size_t>(nq), 0);
    std::vector<uint64_t> intra_ready_completions(static_cast<size_t>(nq), 0);
    std::vector<double> intra_wait_seconds(static_cast<size_t>(nq), 0.0);
    std::vector<double> intra_hit_score_seconds(static_cast<size_t>(nq), 0.0);
    const int fixed_selected_count = std::min(candidate_k, fixed_rerank_n);
    std::vector<int> selected_counts(
            static_cast<size_t>(nq),
            calibrated_threshold_rerank ? 0 : fixed_selected_count);

    FpRerankReader reader(
            fp_store,
            FpRerankSource::ODirectUring,
            cache_enabled,
            cache_limit_rows,
            0,
            physical_row_by_id,
            colocation_io_merge_bytes);

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        faiss::VisitedTable vt(index.ntotal, false);
        faiss::MinimaxHeap search_candidates(std::max(ef, candidate_k));
        PackedAsymBQDistanceComputer dis(
                codes,
                index.code_size,
                code_stride,
                dim,
                adc_mode,
                lut_mode,
                early_abandon,
                ordered_early_abandon,
                static_cast<size_t>(ordered_early_abandon_probe_bytes),
                block32_codes,
                block32_block_stride);
        FpRerankThreadBuffer fp_buffer;
        FpUringBatch pending_batch;
        std::vector<std::pair<float, faiss::idx_t>> pending_scores;
        std::vector<float> calibrated_anchor_heap(
                calibrated_threshold_rerank ? static_cast<size_t>(k) : 0);
        int pending_query = -1;

        auto finish_and_score_pending = [&](bool was_overlapped) {
            if (pending_query < 0) return;
            const FpUringFinishStats finish_stats =
                    reader.finish_rows_uring_async(fp_buffer, pending_batch);
            const size_t pending_index = static_cast<size_t>(pending_query);
            const bool io_was_overlapped =
                    was_overlapped && request_counts[pending_index] > 0;
            if (io_was_overlapped) {
                ready_counts[static_cast<size_t>(pending_query)] =
                        intra_ready_completions[pending_index] +
                        finish_stats.completions_ready_before_wait;
                overlapped_request_counts[static_cast<size_t>(pending_query)] =
                        request_counts[static_cast<size_t>(pending_query)];
            }
            wait_seconds[static_cast<size_t>(pending_query)] = finish_stats.wait_seconds;
            intra_wait_seconds[pending_index] = finish_stats.wait_seconds;
            overlapped[static_cast<size_t>(pending_query)] = io_was_overlapped ? 1 : 0;

            const auto score_start = std::chrono::steady_clock::now();
            const float* q = queries + static_cast<size_t>(pending_query) * dim;
            pending_scores.reserve(pending_scores.size() + pending_batch.ids.size());
            for (size_t ci = 0; ci < pending_batch.ids.size(); ++ci) {
                const faiss::idx_t id = pending_batch.ids[ci];
                const float* row = pending_batch.rows[ci];
                if (cache_enabled) {
                    if (const float* inserted = reader.cache_put(static_cast<size_t>(id), row)) {
                        row = inserted;
                    }
                }
                const double norm = norm_mode == FpRerankNormMode::Precompute
                        ? train_norms[static_cast<size_t>(id)]
                        : 1.0;
                const float score = score_fp_rerank_row(
                        row, q, dim, dot_mode, rerank_metric, norm_mode, norm);
                pending_scores.emplace_back(score, id);
            }
            const int take = std::min<int>(k, pending_scores.size());
            std::partial_sort(
                    pending_scores.begin(),
                    pending_scores.begin() + take,
                    pending_scores.end(),
                    [](const auto& lhs, const auto& rhs) {
                        if (lhs.first != rhs.first) return lhs.first > rhs.first;
                        return lhs.second < rhs.second;
                    });
            for (int rank = 0; rank < take; ++rank) {
                final_labels[static_cast<size_t>(pending_query) * k + rank] =
                        pending_scores[static_cast<size_t>(rank)].second;
            }
            score_seconds[static_cast<size_t>(pending_query)] =
                    intra_hit_score_seconds[pending_index] + seconds_since(score_start);
            pending_scores.clear();
            pending_batch.clear();
            pending_query = -1;
        };

#pragma omp for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            const float* query = queries + static_cast<size_t>(qi) * dim;
            dis.set_query(query);
            int selected_count = calibrated_threshold_rerank
                    ? candidate_k
                    : fixed_selected_count;
            CalibratedThresholdOutput calibrated_output;
            CalibratedThresholdOutput* calibrated_output_ptr = nullptr;
            if (calibrated_threshold_rerank) {
                const float distance_scale = adc_mode == AdcMode::Int16
                        ? dis.int16_lut_scale(query)
                        : 1.0f;
                calibrated_output.anchor_k = k;
                calibrated_output.distance_width =
                        calibrated_threshold_tau * distance_scale;
                calibrated_output.anchor_heap = calibrated_anchor_heap.data();
                calibrated_output.selected_count = &selected_count;
                calibrated_output_ptr = &calibrated_output;
            }
            search_one_packed_inline(
                    index,
                    dis,
                    vt,
                    candidate_k,
                    std::max(ef, candidate_k),
                    ef,
                    check_relative_distance,
                    early_abandon,
                    prefetch_lookahead,
                    adaptive_cfr,
                    search_candidates,
                    distances.data() + static_cast<size_t>(qi) * candidate_k,
                    candidates_out.data() + static_cast<size_t>(qi) * candidate_k,
                    query_stats ? &(*query_stats)[static_cast<size_t>(qi)] : nullptr,
                    calibrated_output_ptr);

            if (original_id_by_colocated_id && !original_id_by_colocated_id->empty()) {
                for (int ci = 0; ci < candidate_k; ++ci) {
                    faiss::idx_t& id = candidates_out[
                            static_cast<size_t>(qi) * candidate_k + ci];
                    if (id < 0) continue;
                    id = static_cast<faiss::idx_t>(
                            (*original_id_by_colocated_id)[static_cast<size_t>(id)]);
                }
            }

            // The previous query's exact FP reads have been in flight during
            // this query's complete HNSW traversal.  Only now do we wait for
            // any tail and score it, before reusing the per-thread DMA buffer.
            finish_and_score_pending(true);

            if (calibrated_threshold_rerank) {
                selected_counts[static_cast<size_t>(qi)] = selected_count;
            }

            std::vector<faiss::idx_t> sorted_ids(static_cast<size_t>(selected_count));
            const int sorted_count = gather_sorted_candidate_ids(
                    candidates_out,
                    static_cast<size_t>(qi) * candidate_k,
                    selected_count,
                    sorted_ids.data(),
                    physical_row_by_id);
            sorted_ids.resize(static_cast<size_t>(sorted_count));

            if (overlap_cache_hits) {
                std::vector<faiss::idx_t> hit_ids;
                std::vector<const float*> hit_rows;
                std::vector<faiss::idx_t> miss_ids;
                hit_ids.reserve(static_cast<size_t>(sorted_count));
                hit_rows.reserve(static_cast<size_t>(sorted_count));
                miss_ids.reserve(static_cast<size_t>(sorted_count));
                for (const faiss::idx_t id : sorted_ids) {
                    if (const float* cached = reader.cache_get(static_cast<size_t>(id))) {
                        hit_ids.push_back(id);
                        hit_rows.push_back(cached);
                    } else {
                        reader.cache_misses.fetch_add(1, std::memory_order_relaxed);
                        miss_ids.push_back(id);
                    }
                }

                if (!miss_ids.empty()) {
                    reader.submit_rows_uring_async(
                            miss_ids.data(),
                            static_cast<int>(miss_ids.size()),
                            io_queue_depth,
                            fp_buffer,
                            pending_batch);
                }
                const size_t query_index = static_cast<size_t>(qi);
                intra_io_requests[query_index] = pending_batch.groups.size();

                pending_scores.clear();
                pending_scores.reserve(static_cast<size_t>(sorted_count));
                const float* q = queries + query_index * dim;
                const auto hit_score_start = std::chrono::steady_clock::now();
                for (size_t hi = 0; hi < hit_ids.size(); ++hi) {
                    const faiss::idx_t id = hit_ids[hi];
                    const double norm = norm_mode == FpRerankNormMode::Precompute
                            ? train_norms[static_cast<size_t>(id)]
                            : 1.0;
                    pending_scores.emplace_back(
                            score_fp_rerank_row(
                                    hit_rows[hi], q, dim, dot_mode,
                                    rerank_metric, norm_mode, norm),
                            id);
                }
                intra_hit_score_seconds[query_index] = seconds_since(hit_score_start);
                if (pending_batch.active) {
                    intra_ready_completions[query_index] =
                            reader.poll_rows_uring_async(fp_buffer, pending_batch);
                }
                if (!hit_ids.empty() && !miss_ids.empty()) {
                    intra_query_overlap[query_index] = 1;
                    intra_hit_rows[query_index] = hit_ids.size();
                }
            } else {
                reader.submit_rows_uring_async(
                        sorted_ids.data(),
                        sorted_count,
                        io_queue_depth,
                        fp_buffer,
                        pending_batch);
            }
            request_counts[static_cast<size_t>(qi)] = pending_batch.groups.size();
            pending_query = qi;
        }

        // The final query owned by each worker has no following HNSW search to
        // hide behind, so this is the only unavoidable pipeline drain.
        finish_and_score_pending(false);
    }

    if (cache_stats) {
        cache_stats->hits = reader.cache_hit_count();
        cache_stats->misses = reader.cache_miss_count();
        cache_stats->rows = reader.cache_size();
        cache_stats->bytes = reader.cache_size() * static_cast<size_t>(fp_store.d) * sizeof(float);
        cache_stats->direct_rows = reader.direct_row_request_count();
        cache_stats->direct_requests = reader.direct_io_request_count();
        cache_stats->direct_logical_bytes = reader.direct_logical_byte_count();
        cache_stats->direct_submitted_bytes = reader.direct_submitted_byte_count();
        cache_stats->page_cache_pages = 0;
        cache_stats->page_cache_bytes = 0;
        cache_stats->page_cache_accesses = 0;
        cache_stats->page_cache_hits = 0;
        cache_stats->page_cache_misses = 0;
        cache_stats->page_cache_evictions = 0;
    }
    if (pipeline_stats) {
        pipeline_stats->queries = static_cast<uint64_t>(nq);
        pipeline_stats->queries_overlapped_with_search =
                std::accumulate(overlapped.begin(), overlapped.end(), uint64_t{0});
        pipeline_stats->io_requests =
                std::accumulate(request_counts.begin(), request_counts.end(), uint64_t{0});
        pipeline_stats->io_requests_overlapped_with_search =
                std::accumulate(
                        overlapped_request_counts.begin(),
                        overlapped_request_counts.end(),
                        uint64_t{0});
        pipeline_stats->completions_ready_after_search =
                std::accumulate(ready_counts.begin(), ready_counts.end(), uint64_t{0});
        pipeline_stats->io_wait_seconds =
                std::accumulate(wait_seconds.begin(), wait_seconds.end(), 0.0);
        pipeline_stats->score_seconds =
                std::accumulate(score_seconds.begin(), score_seconds.end(), 0.0);
    }
    if (intra_query_stats) {
        intra_query_stats->queries = static_cast<uint64_t>(nq);
        intra_query_stats->queries_with_hit_miss_overlap = std::accumulate(
                intra_query_overlap.begin(), intra_query_overlap.end(), uint64_t{0});
        intra_query_stats->cache_hit_rows_scored_while_io = std::accumulate(
                intra_hit_rows.begin(), intra_hit_rows.end(), uint64_t{0});
        intra_query_stats->io_requests = std::accumulate(
                intra_io_requests.begin(), intra_io_requests.end(), uint64_t{0});
        intra_query_stats->completions_ready_after_hit_scoring = std::accumulate(
                intra_ready_completions.begin(), intra_ready_completions.end(), uint64_t{0});
        intra_query_stats->io_wait_seconds = std::accumulate(
                intra_wait_seconds.begin(), intra_wait_seconds.end(), 0.0);
        intra_query_stats->hit_score_seconds = std::accumulate(
                intra_hit_score_seconds.begin(), intra_hit_score_seconds.end(), 0.0);
    }
    if (rerank_count_stats) {
        *rerank_count_stats = summarize_rerank_counts(std::move(selected_counts));
    }
    return final_labels;
}

double recall_at_k(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<int32_t>& gt,
        int nq,
        int gt_dim,
        int k) {
    int64_t hits = 0;
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

std::string method_name(const Args& args) {
    if (args.distance_mode == DistanceMode::XFlagBQ ||
        args.distance_mode == DistanceMode::AdcXFlagBQ) {
        std::string name = args.distance_mode == DistanceMode::XFlagBQ
                ? "FAISS-PackedXFlagBQ-HalfCreditAND-2BitStored"
                : "FAISS-PackedAsymXFlagBQ-FP32Query-2BitStored";
        if (args.search_impl == SearchImpl::Inline) {
            name += "-InlineVectorVisitedSimdHeap-NativeSignCodes-AlignedXFlagCodes";
            name += "-PrefetchLA" + std::to_string(args.prefetch_lookahead);
        }
        name += args.graph_build_mode == GraphBuildMode::Fp ? "-FpGraphHNSW" : "-BinaryGraphHNSW";
        if (args.adaptive_rerank) {
            name += "-AdaptiveRerankMin" + std::to_string(args.adaptive_rerank_min) +
                    "Alpha" + std::to_string(args.adaptive_rerank_alpha);
        }
        if (args.calibrated_threshold_rerank) {
            name += "-CalibratedThresholdTau" +
                    std::to_string(args.calibrated_threshold_rerank_tau);
        }
        name += args.no_rerank ? "-NoRerank" : "+FP32-Rerank";
        return name;
    }

    std::string name = "FAISS-PackedAsymBQ";
    if (args.adc_mode == AdcMode::Fp32Direct) name += "-FP32Direct";
    if (args.adc_mode == AdcMode::Int16) name += "-Int16LUT";
    if (args.adc_mode == AdcMode::Int8) name += "-Int8QuickADC";
    if (args.lut_mode == LutMode::Nibble) name += "-Nibble";
    if (!args.no_rerank && args.rerank_dot_mode == RerankDotMode::Fp32) name += "-RerankFP32Dot";
    if (!args.no_rerank && args.rerank_dot_mode == RerankDotMode::Int8) name += "-RerankInt8RAM";
    if (args.search_impl == SearchImpl::Inline) {
        if (!args.mmap_index_path.empty()) {
            name += "-NativeMmap4KB";
            if (args.mmap_drop_cache_before_round) name += "-ColdEachRound";
        } else if (args.index_page_cache_mib > 0) {
            name += "-Paged" + std::to_string(kPagedIndexPageBytes / 1024) +
                    "KB-IndexCache" +
                    std::to_string(args.index_page_cache_mib) + "MiB";
            if (args.index_page_async) {
                name += "-GraphAsyncQD" + std::to_string(args.index_page_qd);
                if (args.index_page_async_sync_miss_max > 0) {
                    name += "-SyncMissLE" +
                            std::to_string(args.index_page_async_sync_miss_max);
                }
            }
        } else {
            name += "-InlineVectorVisitedSimdHeap-NativeCodes";
        }
        name += "-PrefetchLA" + std::to_string(args.prefetch_lookahead);
    }
    if (args.early_abandon) name += "-EarlyAbandon";
    if (args.ordered_early_abandon) name += "-OrderedBytes";
    if (args.adaptive_cfr) {
        name += "-AdaptiveCFRtau" + std::to_string(args.adaptive_cfr_tau);
        if (args.adaptive_cfr_paper_bucket) name += "Bucket" + std::to_string(args.adaptive_cfr_bucket_count);
    }
    if (args.colocation_iterations > 0) {
        name += "-ColocLDG" + std::to_string(args.colocation_iterations) +
                "P" + std::to_string(args.colocation_partition_size);
    }
    if (args.fp_rerank_page_cache_mib > 0) {
        name += "-PageCache" + std::to_string(args.fp_rerank_page_cache_mib) + "MiB";
    }
    if (args.fp_rerank_cache) {
        name += "-RowCache" + std::to_string(args.fp_rerank_cache_rows);
    }
    if (args.fp_rerank_overlap_queries && args.fp_rerank_overlap_cache_hits) {
        name += "-HybridCacheHitInterQueryIoOverlapQD" + std::to_string(args.fp_rerank_qd);
    } else if (args.fp_rerank_overlap_queries) {
        name += "-InterQueryIoOverlapQD" + std::to_string(args.fp_rerank_qd);
    } else if (args.fp_rerank_overlap_cache_hits) {
        name += "-IntraQueryCacheHitIoOverlapQD" + std::to_string(args.fp_rerank_qd);
    }
    name += args.graph_build_mode == GraphBuildMode::Fp ? "-FpGraphHNSW" : "-BinaryGraphHNSW";
    if (args.adaptive_rerank) {
        name += "-AdaptiveRerankMin" + std::to_string(args.adaptive_rerank_min) +
                "Alpha" + std::to_string(args.adaptive_rerank_alpha);
    }
    if (args.calibrated_threshold_rerank) {
        name += "-CalibratedThresholdTau" +
                std::to_string(args.calibrated_threshold_rerank_tau);
    }
    if (args.candidate_only) name += "-CandidateOnly";
    name += args.no_rerank ? "-NoRerank" : "+FP32-Rerank";
    return name;
}

void run_query_preprocess_only(const Args& args) {
    FbinMmap query_mmap(args.query);
    const std::vector<float> queries = read_queries(query_mmap, args.max_queries);
    const size_t nq = queries.size() / static_cast<size_t>(query_mmap.d);
    if (nq == 0) throw std::runtime_error("empty query set");

    const size_t code_size = (static_cast<size_t>(query_mmap.d) + 7) / 8;
    PackedAsymBQDistanceComputer dis(
            nullptr,
            code_size,
            code_size,
            query_mmap.d,
            AdcMode::Int16,
            LutMode::Nibble);
    PackedAsymBQDistanceComputer scalar_reference(
            nullptr,
            code_size,
            code_size,
            query_mmap.d,
            AdcMode::Int16,
            LutMode::Nibble);
    for (size_t query_id = 0; query_id < nq; ++query_id) {
        const float* query = queries.data() + query_id * query_mmap.d;
        dis.set_query(query);
        scalar_reference.set_query_int16_scalar_reference(query);
        if (dis.lut_i16 != scalar_reference.lut_i16 ||
            dis.lut_i16_hacc != scalar_reference.lut_i16_hacc) {
            throw std::runtime_error(
                    "optimized ADC-BQ LUT differs from scalar reference at query " +
                    std::to_string(query_id));
        }
    }
    std::cout << "Validated optimized ADC-BQ LUTs against scalar reference for "
              << nq << " queries" << std::endl;
    const size_t iterations = std::max(args.query_preprocess_iterations, nq);
    const size_t warmup = std::min<size_t>(iterations, 256);
    const float calibration_tau = args.calibrated_threshold_rerank
            ? args.calibrated_threshold_rerank_tau
            : 1.5f;
    for (size_t i = 0; i < warmup; ++i) {
        const float* query = queries.data() + (i % nq) * query_mmap.d;
        dis.set_query(query);
        if (args.query_preprocess_include_calibration) {
            const float distance_width = calibration_tau * dis.int16_lut_scale(query);
            asm volatile(
                    "" : : "r"(dis.lut_i16_hacc.data()), "x"(distance_width) : "memory");
        } else {
            asm volatile("" : : "r"(dis.lut_i16_hacc.data()) : "memory");
        }
    }

    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(args.rounds));
    for (int round = 1; round <= args.rounds; ++round) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            const float* query = queries.data() + (i % nq) * query_mmap.d;
            dis.set_query(query);
            if (args.query_preprocess_include_calibration) {
                const float distance_width = calibration_tau * dis.int16_lut_scale(query);
                asm volatile(
                        "" : : "r"(dis.lut_i16_hacc.data()), "x"(distance_width) : "memory");
            } else {
                asm volatile("" : : "r"(dis.lut_i16_hacc.data()) : "memory");
            }
        }
        const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        samples.push_back(seconds * 1e6 / static_cast<double>(iterations));
    }
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];

    std::ofstream output(args.out);
    if (!output) throw std::runtime_error("cannot create output: " + args.out);
    output << "dataset,method,stage,median_us_per_query,min_us_per_query,"
              "max_us_per_query,iterations,rounds,notes\n";
    output << args.dataset
           << (args.query_preprocess_include_calibration
                       ? ",ADC-BQ calibrated optimized int16 nibble,method_total,"
                       : ",ADC-BQ optimized int16 nibble,method_total,")
           << std::setprecision(10) << median << ',' << sorted.front() << ','
           << sorted.back() << ',' << iterations << ',' << args.rounds
           << (args.query_preprocess_include_calibration
                       ? ",exact set_query plus calibrated-threshold query scale and tau multiply\n"
                       : ",exact PackedAsymBQDistanceComputer::set_query; calibration threshold scan excluded\n");
    std::cout << args.dataset << " optimized ADC-BQ "
              << (args.query_preprocess_include_calibration
                          ? "set_query + calibrated threshold scale: "
                          : "set_query: ")
              << std::fixed << std::setprecision(3) << median << " us/query"
              << std::endl;
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.query_preprocess_only) {
            run_query_preprocess_only(args);
            return 0;
        }
        FbinMmap train(args.train);
        FbinMmap query_mmap(args.query);
        if (train.d != query_mmap.d) throw std::runtime_error("query dim mismatch");
        const std::vector<float> queries = read_queries(query_mmap, args.max_queries);
        const int nq = static_cast<int>(queries.size() / train.d);
        int32_t gt_n = 0;
        int32_t gt_dim = 0;
        const std::vector<int32_t> gt = read_ibin(args.gt, nq, gt_n, gt_dim);
        if (gt_n != nq) throw std::runtime_error("GT row count mismatch");

        std::cout << "Dataset=" << args.dataset << ", train=(" << train.n << ", " << train.d
                  << "), queries=(" << nq << ", " << train.d << "), GT=(" << gt_n
                  << ", " << gt_dim << "), distance_mode=" << distance_mode_name(args.distance_mode)
                  << ", adc_mode=" << adc_mode_name(args.adc_mode)
                  << ", lut_mode=" << lut_mode_name(args.lut_mode)
                  << ", rerank_dot=" << rerank_dot_mode_name(args.rerank_dot_mode)
                  << ", fp_rerank_metric=" << fp_rerank_metric_name(args.fp_rerank_metric)
                  << ", fp_rerank_source=" << fp_rerank_source_name(args.fp_rerank_source)
                  << ", fp_rerank_path=" << (args.fp_rerank_path.empty() ? args.train : args.fp_rerank_path)
                  << ", fp_rerank_norms=" << fp_rerank_norm_mode_name(args.fp_rerank_norm_mode)
                  << ", fp_rerank_cache=" << args.fp_rerank_cache
                  << ", fp_rerank_cache_rows=" << (args.fp_rerank_cache ? args.fp_rerank_cache_rows : 0)
                  << ", fp_rerank_page_cache_mib=" << args.fp_rerank_page_cache_mib
                  << ", index_page_cache_mib=" << args.index_page_cache_mib
                  << ", index_page_async=" << args.index_page_async
                  << ", index_page_qd=" << args.index_page_qd
                  << ", mmap_index_path=" << args.mmap_index_path
                  << ", mmap_drop_cache_before_round=" << args.mmap_drop_cache_before_round
                  << ", code_layout=" << code_layout_name(args.code_layout)
                  << ", search_impl=" << search_impl_name(args.search_impl)
                  << ", graph_build=" << graph_build_mode_name(args.graph_build_mode)
                  << ", check_relative_distance=" << (!args.no_check_relative_distance)
                  << ", bounded_queue=" << (!args.unbounded_queue)
                  << ", prefetch_lookahead=" << args.prefetch_lookahead
                  << ", early_abandon=" << args.early_abandon
                  << ", ordered_early_abandon=" << args.ordered_early_abandon
                  << ", colocation_iterations=" << args.colocation_iterations
                  << ", colocation_partition_size=" << args.colocation_partition_size
                  << ", colocation_io_merge_bytes=" << args.colocation_io_merge_bytes
                  << ", fp_rerank_overlap_queries=" << args.fp_rerank_overlap_queries
                  << ", fp_rerank_overlap_cache_hits=" << args.fp_rerank_overlap_cache_hits
                  << ", fp_rerank_qd=" << args.fp_rerank_qd
                  << ", calibrated_threshold_rerank="
                  << args.calibrated_threshold_rerank
                  << ", calibrated_threshold_tau="
                  << args.calibrated_threshold_rerank_tau
                  << ", candidate_only=" << args.candidate_only
                  << std::endl;

        const auto build_start = std::chrono::steady_clock::now();
        std::unique_ptr<faiss::IndexBinaryHNSW> index;
        ColocationArtifacts colocation;
        std::vector<uint32_t> original_id_by_colocated_id;
        double colocation_apply_seconds = 0.0;
        const bool mmap_file_ready = !args.mmap_index_path.empty() &&
                !args.rebuild_mmap_index &&
                validate_mmap_index_file(
                        args.mmap_index_path,
                        train.n,
                        train.d,
                        args.m,
                        args.colocation_iterations,
                        args.colocation_partition_size);
        bool can_skip_faiss_load = mmap_file_ready;
        if (can_skip_faiss_load && args.colocation_iterations > 0) {
            colocation.map_path = args.colocation_map_path.empty()
                    ? args.index + ".coloc.map"
                    : args.colocation_map_path;
            const auto colocation_start = std::chrono::steady_clock::now();
            if (!load_colocation_map(
                        colocation.map_path,
                        train.n,
                        train.d,
                        args.colocation_iterations,
                        args.colocation_partition_size,
                        colocation.physical_row_by_id)) {
                can_skip_faiss_load = false;
                colocation = ColocationArtifacts{};
            } else {
                colocation.loaded = true;
                colocation.prepare_seconds = seconds_since(colocation_start);
                original_id_by_colocated_id.resize(static_cast<size_t>(train.n));
                for (uint32_t original = 0; original < static_cast<uint32_t>(train.n); ++original) {
                    original_id_by_colocated_id[colocation.physical_row_by_id[original]] = original;
                }
                std::cout << "Loaded neighbor-colocated index permutation without materializing FAISS: "
                          << colocation.map_path << std::endl;
            }
        }

        if (!can_skip_faiss_load) {
            index = build_or_load_index(args, train);
            std::cout << "Index ntotal=" << index->ntotal << ", code_size=" << index->code_size
                      << " bytes/vector" << std::endl;
            if (index->ntotal != train.n) {
                throw std::runtime_error("index ntotal does not match train rows");
            }
            colocation = prepare_colocation(args, *index, train);
            if (args.colocation_iterations > 0) {
                const auto colocation_apply_start = std::chrono::steady_clock::now();
                original_id_by_colocated_id =
                        apply_index_colocation(*index, colocation.physical_row_by_id);
                colocation_apply_seconds = seconds_since(colocation_apply_start);
            }
            if (!args.mmap_index_path.empty()) {
                prepare_mmap_index_file(
                        args.mmap_index_path,
                        *index,
                        args.m,
                        args.colocation_iterations,
                        args.colocation_partition_size,
                        args.rebuild_mmap_index);
            }
        } else {
            std::cout << "Using existing native mmap index without loading the FAISS heap index: "
                      << args.mmap_index_path << std::endl;
        }
        const double build_or_load_seconds = seconds_since(build_start);

        std::unique_ptr<MmapHnswIndex> mmap_index;
        if (!args.mmap_index_path.empty()) {
            mmap_index = std::make_unique<MmapHnswIndex>(
                    args.mmap_index_path,
                    train.n,
                    train.d,
                    args.m,
                    args.colocation_iterations,
                    args.colocation_partition_size);
            index.reset();
            std::cout << "Mapped native HNSW+BQ index without populating it: bytes="
                      << mmap_index->file_bytes() << ", os_page_bytes="
                      << mmap_index->os_page_bytes() << std::endl;
        }

        std::unique_ptr<FbinMmap> aligned_fp_store;
        const FbinMmap* fp_rerank_store = &train;
        if (!args.fp_rerank_path.empty()) {
            prepare_aligned_fp_store(args.fp_rerank_path, train);
            aligned_fp_store = std::make_unique<FbinMmap>(args.fp_rerank_path);
            if (aligned_fp_store->n != train.n || aligned_fp_store->d != train.d ||
                aligned_fp_store->data_offset != kFpDataAlignment) {
                throw std::runtime_error("aligned FP rerank store does not match train data");
            }
            fp_rerank_store = aligned_fp_store.get();
        }
        if (!args.no_rerank && !args.candidate_only &&
            args.rerank_dot_mode != RerankDotMode::Int8 &&
            args.fp_rerank_source == FpRerankSource::Ram) {
            FbinMmap* mutable_store = args.fp_rerank_path.empty()
                    ? &train
                    : aligned_fp_store.get();
            const auto resident_start = std::chrono::steady_clock::now();
            const size_t resident_bytes = mutable_store->materialize_resident();
            fp_rerank_store = mutable_store;
            std::cout << "Materialized full FP rerank vectors in process-owned RAM: bytes="
                      << resident_bytes << ", seconds="
                      << seconds_since(resident_start) << std::endl;
        }
        const std::vector<uint32_t>* physical_row_by_id = nullptr;

        std::string active_paged_index_path;
        PagedIndexPrepareStats paged_index_prepare_stats;
        if (args.index_page_cache_mib > 0) {
            active_paged_index_path = args.paged_index_path;
            if (active_paged_index_path.empty()) {
                active_paged_index_path = args.index;
                if (args.colocation_iterations > 0) {
                    active_paged_index_path += ".index_coloc_ldg" +
                            std::to_string(args.colocation_iterations) + "_p" +
                            std::to_string(args.colocation_partition_size);
                } else {
                    active_paged_index_path += ".index_baseline";
                }
                active_paged_index_path += ".paged" +
                        std::to_string(kPagedIndexPageBytes / 1024) + "k";
            }
            paged_index_prepare_stats = prepare_paged_index_file(
                    active_paged_index_path,
                    *index,
                    args.m,
                    args.rebuild_paged_index);

            auto* flat = dynamic_cast<faiss::IndexBinaryFlat*>(index->storage);
            if (!flat) throw std::runtime_error("paged index release requires IndexBinaryFlat");
            flat->xb = faiss::MaybeOwnedVector<uint8_t>();
            bbq_compat::release_fork_payloads(*flat);
            index->hnsw.neighbors = faiss::MaybeOwnedVector<faiss::HNSW::storage_idx_t>();
            index->hnsw.offsets.clear();
            index->hnsw.levels.clear();
            std::cout << "Released RAM BQ codes and HNSW adjacency; search will use paged index cache only"
                      << std::endl;
        }

        const uint8_t* inline_codes = nullptr;
        size_t inline_code_stride = static_cast<size_t>(train.d) / 8;
        if (args.search_impl == SearchImpl::Inline && args.index_page_cache_mib == 0 &&
            !mmap_index) {
            inline_codes = storage_codes(*index);
            inline_code_stride = index->code_size;
            std::cout << "Using native FAISS packed search codes without a duplicate: stride="
                      << inline_code_stride << " bytes/vector" << std::endl;
        }

        const uint8_t* inline_block32_codes = nullptr;
        size_t inline_block32_block_stride = 0;
        if (args.search_impl == SearchImpl::Inline &&
            args.code_layout == CodeLayout::Block32 &&
            args.index_page_cache_mib == 0 && !mmap_index) {
            std::cout << "WARNING: --code-layout block32 is deprecated and ignored; "
                      << "using the native FAISS vector layout without a duplicate"
                      << std::endl;
        }

        AlignedPackedCodes aligned_xflag_codes;
        const uint8_t* inline_xflag_codes = nullptr;
        size_t inline_xflag_code_stride = static_cast<size_t>(train.d) / 8;
        float active_xflag_threshold = args.xflag_threshold;
        double xflag_prep_seconds = 0.0;
        std::string xflag_code_path;
        std::string xflag_code_status;
        if (args.distance_mode == DistanceMode::XFlagBQ ||
            args.distance_mode == DistanceMode::AdcXFlagBQ) {
            if (static_cast<int64_t>(index->ntotal) != static_cast<int64_t>(train.n)) {
                throw std::runtime_error("xflag search requires index ntotal to match train rows");
            }
            const auto xflag_start = std::chrono::steady_clock::now();
            xflag_code_path = resolved_xflag_code_path(args);
            std::vector<uint8_t> xflag_codes;
            if (fs::exists(xflag_code_path) && !args.rebuild) {
                XFlagCodeFile loaded = read_xflag_code_file(
                        xflag_code_path,
                        static_cast<uint64_t>(index->ntotal),
                        static_cast<uint32_t>(train.d),
                        static_cast<uint32_t>(index->code_size));
                if (active_xflag_threshold >= 0.0f &&
                    std::abs(active_xflag_threshold - loaded.threshold) > 1e-6f) {
                    throw std::runtime_error(
                            "existing xflag code threshold differs; pass --rebuild or a different --xflag-code-path");
                }
                active_xflag_threshold = loaded.threshold;
                xflag_codes = std::move(loaded.codes);
                xflag_code_status = "loaded";
            } else {
                if (active_xflag_threshold < 0.0f) {
                    active_xflag_threshold = compute_xflag_threshold_quantile(
                            train,
                            args.xflag_quantile,
                            args.xflag_sample_rows);
                }
                omp_set_num_threads(args.build_threads);
                pack_pgvector_xflag_bits(
                        train.data,
                        static_cast<size_t>(train.n),
                        train.d,
                        active_xflag_threshold,
                        xflag_codes);
                write_xflag_code_file(
                        xflag_code_path,
                        xflag_codes,
                        static_cast<uint64_t>(index->ntotal),
                        static_cast<uint32_t>(train.d),
                        static_cast<uint32_t>(index->code_size),
                        active_xflag_threshold,
                        args.xflag_quantile,
                        args.xflag_sample_rows);
                xflag_code_status = "built";
            }
            aligned_xflag_codes = AlignedPackedCodes(
                    xflag_codes.data(),
                    static_cast<size_t>(train.n),
                    index->code_size);
            std::vector<uint8_t>().swap(xflag_codes);
            inline_xflag_codes = aligned_xflag_codes.data;
            inline_xflag_code_stride = aligned_xflag_codes.stride;
            xflag_prep_seconds = seconds_since(xflag_start);
            std::cout << "Prepared stored xflag codes: status=" << xflag_code_status
                      << ", path=" << xflag_code_path
                      << ", threshold=" << active_xflag_threshold
                      << ", quantile=" << args.xflag_quantile
                      << ", sample_rows=" << std::min<int>(train.n, args.xflag_sample_rows)
                      << ", stride=" << inline_xflag_code_stride
                      << " bytes/vector, total=" << aligned_xflag_codes.bytes
                      << " bytes, seconds=" << xflag_prep_seconds << std::endl;
        }

        double norm_precompute_seconds = 0.0;
        double int8_rerank_prep_seconds = 0.0;
        size_t int8_rerank_bytes = 0;
        std::vector<double> train_norms;
        Int8RerankIndex int8_rerank_index;
        if (!args.no_rerank && !args.candidate_only) {
            const auto norm_start = std::chrono::steady_clock::now();
            if (args.rerank_dot_mode == RerankDotMode::Int8) {
                int8_rerank_index = build_int8_rerank_index(train, args.build_threads);
                int8_rerank_prep_seconds = seconds_since(norm_start);
                int8_rerank_bytes = int8_rerank_index.bytes();
                norm_precompute_seconds = int8_rerank_prep_seconds;
                std::cout << "Prepared int8 RAM rerank index in " << int8_rerank_prep_seconds
                          << " seconds, bytes=" << int8_rerank_bytes << std::endl;
            } else if (args.fp_rerank_metric == FpRerankMetric::Cosine &&
                       args.fp_rerank_norm_mode == FpRerankNormMode::Precompute) {
                train_norms = compute_train_norms(train, args.build_threads);
                norm_precompute_seconds = seconds_since(norm_start);
                std::cout << "Precomputed FP rerank norms in " << norm_precompute_seconds
                          << " seconds" << std::endl;
            } else {
                norm_precompute_seconds = 0.0;
                if (args.fp_rerank_metric == FpRerankMetric::InnerProduct) {
                    std::cout << "Skipping FP rerank norms for inner-product refinement" << std::endl;
                } else {
                    std::cout << "Skipping FP rerank norm precompute; norms will be computed on fetched rows" << std::endl;
                }
            }
        }

        AdcAdaptiveCfrConfig adaptive_cfr_config;
        adaptive_cfr_config.enabled = args.adaptive_cfr;
        adaptive_cfr_config.paper_bucket_mode = args.adaptive_cfr_paper_bucket;
        adaptive_cfr_config.tau = args.adaptive_cfr_tau;
        adaptive_cfr_config.tmin_pops = args.adaptive_cfr_tmin_pops;
        adaptive_cfr_config.bucket_count = args.adaptive_cfr_bucket_count;
        adaptive_cfr_config.gammas = args.adaptive_cfr_gammas;
        const AdcAdaptiveCfrConfig* adaptive_cfr_ptr = args.adaptive_cfr ? &adaptive_cfr_config : nullptr;
        const bool collect_search_stats = !args.stats_out.empty();

        fs::create_directories(fs::path(args.out).parent_path());
        std::ofstream csv(args.out);
        csv << "dataset,method,distance_mode,adc_mode,lut_mode,rerank_dot,fp_rerank_metric,fp_rerank_source,fp_rerank_norms,graph_build,k,search_threads,nq,nb,dim,M,efConstruction,efSearch,rerank_enabled,rerank_n,candidate_pool_size,"
            << "adaptive_rerank_enabled,adaptive_rerank_min,adaptive_rerank_alpha,"
            << "calibrated_threshold_rerank_enabled,calibrated_threshold_rerank_tau,"
            << "adaptive_rerank_count_mean,adaptive_rerank_count_min,adaptive_rerank_count_p95,adaptive_rerank_count_p99,adaptive_rerank_count_max,rounds,"
            << "index_status,build_or_load_seconds,norm_precompute_seconds,int8_rerank_prep_seconds,int8_rerank_bytes,xflag_prep_seconds,search_seconds_mean,rerank_seconds_mean,total_seconds_mean,"
            << "rerank_read_bytes_mean,rerank_read_bytes_per_query,fp_rerank_cache_enabled,fp_rerank_cache_rows_limit,fp_rerank_cache_rows_mean,fp_rerank_cache_bytes_mean,"
            << "fp_rerank_cache_hits_mean,fp_rerank_cache_misses_mean,fp_page_cache_mib,fp_page_cache_pages_mean,fp_page_cache_bytes_mean,"
            << "fp_page_cache_accesses_mean,fp_page_cache_hits_mean,fp_page_cache_misses_mean,fp_page_cache_evictions_mean,fp_page_cache_hit_ratio,"
            << "index_page_cache_mib,index_page_bytes,index_page_cache_pages_mean,index_data_pages,index_page_accesses_mean,index_page_hits_mean,index_page_misses_mean,index_page_evictions_mean,index_page_hit_ratio,index_page_read_bytes_mean,"
            << "index_page_async,index_page_qd,index_page_async_sync_miss_max,index_hybrid_sync_batches_mean,index_hybrid_sync_requests_mean,index_hybrid_sync_wait_seconds_sum_mean,index_async_batches_mean,index_async_requests_mean,index_async_cached_nodes_scored_while_inflight_mean,index_async_completed_nodes_scored_while_inflight_mean,index_async_completions_ready_before_wait_mean,index_async_ready_ratio,index_async_wait_seconds_sum_mean,"
            << "paged_index_path,paged_index_prepare_seconds,paged_index_status,paged_index_file_bytes,paged_index_data_pages,"
            << "mmap_index_enabled,mmap_index_path,mmap_index_file_bytes,mmap_os_page_bytes,mmap_drop_cache_before_round,mmap_minor_faults_mean,mmap_major_faults_mean,mmap_index_read_bytes_mean,mmap_resident_pages_before_mean,mmap_resident_pages_after_mean,"
            << "recall_at_k,qps,qps_median,xflag_threshold,xflag_quantile,xflag_sample_rows,xflag_code_status,index_path,xflag_code_path,"
            << "colocation_iterations,colocation_partition_size,colocation_io_merge_bytes,colocation_prepare_seconds,colocation_apply_seconds,colocation_status,colocation_map_path,"
            << "direct_rows_mean,direct_requests_mean,direct_logical_bytes_mean,direct_submitted_bytes_mean,direct_request_reduction_ratio,direct_byte_reuse_ratio,"
            << "fp_rerank_overlap_queries,fp_rerank_qd,pipeline_overlapped_queries_mean,pipeline_io_requests_mean,pipeline_overlapped_io_requests_mean,pipeline_ready_completions_mean,pipeline_ready_ratio,pipeline_wait_seconds_sum_mean,pipeline_score_seconds_sum_mean,"
            << "fp_rerank_overlap_cache_hits,intra_query_overlapped_queries_mean,intra_query_hit_rows_mean,intra_query_io_requests_mean,intra_query_ready_completions_mean,intra_query_ready_ratio,intra_query_wait_seconds_sum_mean,intra_query_hit_score_seconds_sum_mean,fp_rerank_path\n";

        std::ofstream stats_csv;
        if (!args.stats_out.empty()) {
            fs::create_directories(fs::path(args.stats_out).parent_path());
            stats_csv.open(args.stats_out);
            write_stats_header(stats_csv);
        }

        std::ofstream candidates_csv;
        if (!args.candidates_out.empty()) {
            const fs::path candidates_path(args.candidates_out);
            if (!candidates_path.parent_path().empty()) {
                fs::create_directories(candidates_path.parent_path());
            }
            candidates_csv.open(candidates_path);
            if (!candidates_csv) {
                throw std::runtime_error("failed to open candidate output: " + args.candidates_out);
            }
            candidates_csv << std::setprecision(std::numeric_limits<float>::max_digits10);
            candidates_csv << "dataset,adc_mode,lut_mode,efSearch,round,query_id,rank,id,distance\n";
        }

        std::ofstream final_labels_csv;
        if (!args.final_labels_out.empty()) {
            const fs::path labels_path(args.final_labels_out);
            if (!labels_path.parent_path().empty()) {
                fs::create_directories(labels_path.parent_path());
            }
            final_labels_csv.open(labels_path);
            if (!final_labels_csv) {
                throw std::runtime_error(
                        "failed to open final-label output: " + args.final_labels_out);
            }
            final_labels_csv
                    << "dataset,efSearch,candidate_pool_size,round,query_id,rank,id\n";
        }

        std::ofstream per_query_metrics_csv;
        if (!args.per_query_metrics_out.empty()) {
            const fs::path metrics_path(args.per_query_metrics_out);
            if (!metrics_path.parent_path().empty()) {
                fs::create_directories(metrics_path.parent_path());
            }
            per_query_metrics_csv.open(metrics_path);
            if (!per_query_metrics_csv) {
                throw std::runtime_error(
                        "failed to open per-query metrics output: " +
                        args.per_query_metrics_out);
            }
            per_query_metrics_csv
                    << "dataset,efSearch,candidate_pool_size,policy,latency_mode,round,query_id,"
                    << "selected_fp_reads,search_us,selection_us,io_wait_us,"
                    << "rerank_us,end_to_end_us,recall_at_k\n";
        }

        for (int ef : args.efs) {
            std::vector<double> recalls;
            std::vector<double> search_secs;
            std::vector<double> rerank_secs;
            std::vector<double> total_secs;
            std::vector<double> qps_values;
            std::vector<double> rerank_read_bytes_values;
            std::vector<double> fp_cache_rows_values;
            std::vector<double> fp_cache_bytes_values;
            std::vector<double> fp_cache_hits_values;
            std::vector<double> fp_cache_misses_values;
            std::vector<double> fp_page_cache_pages_values;
            std::vector<double> fp_page_cache_bytes_values;
            std::vector<double> fp_page_cache_accesses_values;
            std::vector<double> fp_page_cache_hits_values;
            std::vector<double> fp_page_cache_misses_values;
            std::vector<double> fp_page_cache_evictions_values;
            std::vector<double> index_page_cache_pages_values;
            std::vector<double> index_data_pages_values;
            std::vector<double> index_page_accesses_values;
            std::vector<double> index_page_hits_values;
            std::vector<double> index_page_misses_values;
            std::vector<double> index_page_evictions_values;
            std::vector<double> index_page_read_bytes_values;
            std::vector<double> index_hybrid_sync_batches_values;
            std::vector<double> index_hybrid_sync_requests_values;
            std::vector<double> index_hybrid_sync_wait_seconds_values;
            std::vector<double> index_async_batches_values;
            std::vector<double> index_async_requests_values;
            std::vector<double> index_async_cached_nodes_values;
            std::vector<double> index_async_completed_nodes_values;
            std::vector<double> index_async_ready_completions_values;
            std::vector<double> index_async_wait_seconds_values;
            std::vector<double> mmap_minor_faults_values;
            std::vector<double> mmap_major_faults_values;
            std::vector<double> mmap_index_read_bytes_values;
            std::vector<double> mmap_resident_pages_before_values;
            std::vector<double> mmap_resident_pages_after_values;
            std::vector<double> direct_rows_values;
            std::vector<double> direct_requests_values;
            std::vector<double> direct_logical_bytes_values;
            std::vector<double> direct_submitted_bytes_values;
            std::vector<double> adaptive_count_means;
            std::vector<double> adaptive_count_mins;
            std::vector<double> adaptive_count_p95s;
            std::vector<double> adaptive_count_p99s;
            std::vector<double> adaptive_count_maxs;
            std::vector<double> pipeline_overlapped_queries_values;
            std::vector<double> pipeline_ready_completions_values;
            std::vector<double> pipeline_io_requests_values;
            std::vector<double> pipeline_overlapped_io_requests_values;
            std::vector<double> pipeline_wait_seconds_values;
            std::vector<double> pipeline_score_seconds_values;
            std::vector<double> intra_query_overlapped_queries_values;
            std::vector<double> intra_query_hit_rows_values;
            std::vector<double> intra_query_io_requests_values;
            std::vector<double> intra_query_ready_completions_values;
            std::vector<double> intra_query_wait_seconds_values;
            std::vector<double> intra_query_hit_score_seconds_values;
            std::vector<AdcSearchStats> stats_accum;
            for (int round = 0; round < args.rounds; ++round) {
                const int first_stage_k = args.candidate_only
                        ? args.candidate_pool_size
                        : (args.no_rerank ? args.k : args.candidate_pool_size);
                std::vector<float> distances;
                std::vector<faiss::idx_t> candidates;
                std::vector<double> query_search_seconds;
                std::vector<double> query_end_to_end_seconds;
                std::vector<faiss::idx_t> pipelined_final_labels;
                std::vector<AdcSearchStats> query_stats;
                FpRerankCacheStats pipelined_cache_stats;
                FpInterQueryPipelineStats pipelined_stats;
                FpIntraQueryPipelineStats pipelined_intra_query_stats;
                AdaptiveRerankSelection pipelined_rerank_count_stats;
                IndexPageCacheStats index_page_cache_stats;
                ProcessMemoryIoSnapshot mmap_before;
                size_t mmap_resident_pages_before = 0;
                if (mmap_index) {
                    if (args.mmap_drop_cache_before_round) mmap_index->drop_file_cache();
                    mmap_resident_pages_before = mmap_index->resident_pages();
                    mmap_before = process_memory_io_snapshot();
                }
                const uint64_t pipelined_read_bytes_before = args.fp_rerank_overlap_queries
                        ? read_proc_self_read_bytes()
                        : 0;
                auto t0 = std::chrono::steady_clock::now();
                if (args.strict_query_e2e) {
                    distances.assign(
                            static_cast<size_t>(nq) * first_stage_k,
                            std::numeric_limits<float>::infinity());
                    candidates.assign(static_cast<size_t>(nq) * first_stage_k, -1);
                    query_search_seconds.assign(static_cast<size_t>(nq), 0.0);
                } else if (args.fp_rerank_overlap_queries) {
                    pipelined_final_labels = search_rerank_interquery_pipeline(
                            *index,
                            *fp_rerank_store,
                            train_norms,
                            queries.data(),
                            nq,
                            train.d,
                            first_stage_k,
                            args.k,
                            ef,
                            args.search_threads,
                            args.adc_mode,
                            args.lut_mode,
                            !args.no_check_relative_distance,
                            args.early_abandon,
                            args.ordered_early_abandon,
                            args.ordered_early_abandon_probe_bytes,
                            args.prefetch_lookahead,
                            adaptive_cfr_ptr,
                            inline_codes,
                            inline_code_stride,
                            inline_block32_codes,
                            inline_block32_block_stride,
                            args.rerank_dot_mode,
                            args.fp_rerank_metric,
                            args.fp_rerank_norm_mode,
                            args.fp_rerank_cache,
                            args.fp_rerank_cache_rows,
                            args.fp_rerank_overlap_cache_hits,
                            args.rerank_n,
                            args.calibrated_threshold_rerank,
                            args.calibrated_threshold_rerank_tau,
                            args.fp_rerank_qd,
                            original_id_by_colocated_id.empty()
                                    ? nullptr
                                    : &original_id_by_colocated_id,
                            physical_row_by_id,
                            args.colocation_io_merge_bytes,
                            distances,
                            candidates,
                            collect_search_stats ? &query_stats : nullptr,
                            &pipelined_cache_stats,
                            &pipelined_stats,
                            &pipelined_intra_query_stats,
                            &pipelined_rerank_count_stats);
                } else if (args.distance_mode == DistanceMode::XFlagBQ) {
                    search_packed_xflag_inline(
                            *index,
                            queries.data(),
                            nq,
                            train.d,
                            first_stage_k,
                            ef,
                            args.search_threads,
                            !args.no_check_relative_distance,
                            args.early_abandon,
                            args.prefetch_lookahead,
                            inline_codes,
                            inline_code_stride,
                            inline_xflag_codes,
                            inline_xflag_code_stride,
                            active_xflag_threshold,
                            distances,
                            candidates,
                            collect_search_stats ? &query_stats : nullptr);
                } else if (args.distance_mode == DistanceMode::AdcXFlagBQ) {
                    search_packed_adcxflag_inline(
                            *index,
                            queries.data(),
                            nq,
                            train.d,
                            first_stage_k,
                            ef,
                            args.search_threads,
                            !args.no_check_relative_distance,
                            args.early_abandon,
                            args.prefetch_lookahead,
                            inline_codes,
                            inline_code_stride,
                            inline_xflag_codes,
                            inline_xflag_code_stride,
                            distances,
                            candidates,
                            collect_search_stats ? &query_stats : nullptr);
                } else {
                    if (mmap_index) {
                        search_mmap_asym_inline(
                                *mmap_index,
                                queries.data(),
                                nq,
                                train.d,
                                first_stage_k,
                                ef,
                                args.search_threads,
                                args.adc_mode,
                                args.lut_mode,
                                !args.no_check_relative_distance,
                                args.early_abandon,
                                args.ordered_early_abandon,
                                args.ordered_early_abandon_probe_bytes,
                                args.prefetch_lookahead,
                                adaptive_cfr_ptr,
                                distances,
                                candidates,
                                collect_search_stats ? &query_stats : nullptr);
                    } else if (args.index_page_cache_mib > 0) {
                        search_paged_asym_inline(
                                active_paged_index_path,
                                args.index_page_cache_mib,
                                args.index_page_async,
                                args.index_page_qd,
                                args.index_page_async_sync_miss_max,
                                queries.data(),
                                nq,
                                train.d,
                                first_stage_k,
                                ef,
                                args.search_threads,
                                args.adc_mode,
                                args.lut_mode,
                                !args.no_check_relative_distance,
                                args.early_abandon,
                                args.ordered_early_abandon,
                                args.ordered_early_abandon_probe_bytes,
                                args.prefetch_lookahead,
                                adaptive_cfr_ptr,
                                distances,
                                candidates,
                                collect_search_stats ? &query_stats : nullptr,
                                &index_page_cache_stats);
                    } else {
                        search_packed_asym(
                                *index,
                                queries.data(),
                                nq,
                                train.d,
                                first_stage_k,
                                ef,
                                args.search_threads,
                                args.adc_mode,
                                args.lut_mode,
                                args.search_impl,
                                !args.no_check_relative_distance,
                                !args.unbounded_queue,
                                args.early_abandon,
                                args.ordered_early_abandon,
                                args.ordered_early_abandon_probe_bytes,
                                args.prefetch_lookahead,
                                adaptive_cfr_ptr,
                                inline_codes,
                                inline_code_stride,
                                inline_block32_codes,
                                inline_block32_block_stride,
                                distances,
                                candidates,
                                collect_search_stats ? &query_stats : nullptr,
                                per_query_metrics_csv.is_open()
                                        ? &query_search_seconds
                                        : nullptr);
                    }
                }
                if (mmap_index) {
                    const ProcessMemoryIoSnapshot mmap_after = process_memory_io_snapshot();
                    const size_t mmap_resident_pages_after = mmap_index->resident_pages();
                    mmap_minor_faults_values.push_back(static_cast<double>(
                            mmap_after.minor_faults - mmap_before.minor_faults));
                    mmap_major_faults_values.push_back(static_cast<double>(
                            mmap_after.major_faults - mmap_before.major_faults));
                    mmap_index_read_bytes_values.push_back(static_cast<double>(
                            mmap_after.read_bytes - mmap_before.read_bytes));
                    mmap_resident_pages_before_values.push_back(
                            static_cast<double>(mmap_resident_pages_before));
                    mmap_resident_pages_after_values.push_back(
                            static_cast<double>(mmap_resident_pages_after));
                } else {
                    mmap_minor_faults_values.push_back(0.0);
                    mmap_major_faults_values.push_back(0.0);
                    mmap_index_read_bytes_values.push_back(0.0);
                    mmap_resident_pages_before_values.push_back(0.0);
                    mmap_resident_pages_after_values.push_back(0.0);
                }
                index_page_cache_pages_values.push_back(
                        static_cast<double>(index_page_cache_stats.resident_pages));
                index_data_pages_values.push_back(
                        static_cast<double>(index_page_cache_stats.data_pages));
                index_page_accesses_values.push_back(
                        static_cast<double>(index_page_cache_stats.accesses));
                index_page_hits_values.push_back(
                        static_cast<double>(index_page_cache_stats.hits));
                index_page_misses_values.push_back(
                        static_cast<double>(index_page_cache_stats.misses));
                index_page_evictions_values.push_back(
                        static_cast<double>(index_page_cache_stats.evictions));
                index_page_read_bytes_values.push_back(
                        static_cast<double>(index_page_cache_stats.read_bytes));
                index_hybrid_sync_batches_values.push_back(
                        static_cast<double>(index_page_cache_stats.hybrid_sync_batches));
                index_hybrid_sync_requests_values.push_back(
                        static_cast<double>(index_page_cache_stats.hybrid_sync_requests));
                index_hybrid_sync_wait_seconds_values.push_back(
                        index_page_cache_stats.hybrid_sync_wait_seconds);
                index_async_batches_values.push_back(
                        static_cast<double>(index_page_cache_stats.async_batches));
                index_async_requests_values.push_back(
                        static_cast<double>(index_page_cache_stats.async_requests));
                index_async_cached_nodes_values.push_back(static_cast<double>(
                        index_page_cache_stats.async_cached_nodes_scored_while_inflight));
                index_async_completed_nodes_values.push_back(static_cast<double>(
                        index_page_cache_stats.async_completed_nodes_scored_while_inflight));
                index_async_ready_completions_values.push_back(static_cast<double>(
                        index_page_cache_stats.async_completions_ready_before_wait));
                index_async_wait_seconds_values.push_back(
                        index_page_cache_stats.async_wait_seconds);
                if (!args.fp_rerank_overlap_queries && !args.strict_query_e2e) {
                    restore_original_ids(candidates, original_id_by_colocated_id);
                }
                if (collect_search_stats) {
                    stats_accum.insert(stats_accum.end(), query_stats.begin(), query_stats.end());
                }
                double search_s = seconds_since(t0);
                if (candidates_csv) {
                    for (size_t query_id = 0; query_id < nq; ++query_id) {
                        const size_t base = query_id * static_cast<size_t>(first_stage_k);
                        for (int rank = 0; rank < first_stage_k; ++rank) {
                            const size_t pos = base + static_cast<size_t>(rank);
                            candidates_csv
                                    << args.dataset << ','
                                    << adc_mode_name(args.adc_mode) << ','
                                    << lut_mode_name(args.lut_mode) << ','
                                    << ef << ',' << round << ',' << query_id << ',' << rank << ','
                                    << candidates[pos] << ',' << distances[pos] << '\n';
                        }
                    }
                    candidates_csv.flush();
                }
                std::vector<faiss::idx_t> final_labels;
                double rerank_s = 0.0;
                double strict_total_s = 0.0;
                FpRerankCacheStats fp_cache_stats;
                FpIntraQueryPipelineStats intra_query_stats;
                std::vector<int> query_selected_counts;
                std::vector<double> query_selection_seconds;
                double adaptive_count_mean = args.no_rerank ? 0.0 : static_cast<double>(args.rerank_n);
                double adaptive_count_min = adaptive_count_mean;
                double adaptive_count_p95 = adaptive_count_mean;
                double adaptive_count_p99 = adaptive_count_mean;
                double adaptive_count_max = adaptive_count_mean;
                if (args.candidate_only) {
                    final_labels.assign(static_cast<size_t>(nq) * args.k, -1);
                    for (int qi = 0; qi < nq; ++qi) {
                        for (int rank = 0; rank < args.k; ++rank) {
                            final_labels[static_cast<size_t>(qi) * args.k + rank] =
                                    candidates[static_cast<size_t>(qi) * first_stage_k + rank];
                        }
                    }
                } else if (args.no_rerank) {
                    final_labels = std::move(candidates);
                } else if (args.strict_query_e2e) {
                    const uint64_t read_bytes_before = read_proc_self_read_bytes();
                    StrictQueryPipelineResult strict =
                            search_rerank_strict_query_local(
                                    *index,
                                    *fp_rerank_store,
                                    queries.data(),
                                    nq,
                                    train.d,
                                    args.candidate_pool_size,
                                    args.k,
                                    ef,
                                    args.adc_mode,
                                    args.lut_mode,
                                    !args.no_check_relative_distance,
                                    args.early_abandon,
                                    args.ordered_early_abandon,
                                    args.ordered_early_abandon_probe_bytes,
                                    args.prefetch_lookahead,
                                    adaptive_cfr_ptr,
                                    inline_codes,
                                    inline_code_stride,
                                    inline_block32_codes,
                                    inline_block32_block_stride,
                                    args.calibrated_threshold_rerank,
                                    args.calibrated_threshold_rerank_tau,
                                    args.rerank_n,
                                    args.rerank_dot_mode,
                                    args.fp_rerank_metric,
                                    args.fp_rerank_qd,
                                    original_id_by_colocated_id.empty()
                                            ? nullptr
                                            : &original_id_by_colocated_id,
                                    physical_row_by_id,
                                    args.colocation_io_merge_bytes);
                    const uint64_t read_bytes_after = read_proc_self_read_bytes();
                    strict_total_s = strict.wall_seconds;
                    query_search_seconds = std::move(strict.search_seconds);
                    query_selection_seconds = std::move(strict.selection_seconds);
                    query_end_to_end_seconds = std::move(strict.end_to_end_seconds);
                    query_selected_counts = std::move(strict.selected_counts);
                    search_s = std::accumulate(
                            query_search_seconds.begin(),
                            query_search_seconds.end(),
                            0.0);
                    rerank_s = std::max(0.0, strict_total_s - search_s);
                    distances = std::move(strict.distances);
                    candidates = std::move(strict.candidates);
                    final_labels = std::move(strict.final_labels);
                    adaptive_count_mean = strict.count_summary.mean_count;
                    adaptive_count_min = strict.count_summary.min_count;
                    adaptive_count_p95 = strict.count_summary.p95_count;
                    adaptive_count_p99 = strict.count_summary.p99_count;
                    adaptive_count_max = strict.count_summary.max_count;
                    fp_cache_stats = strict.cache_stats;
                    intra_query_stats = std::move(strict.intra_query_stats);

                    rerank_read_bytes_values.push_back(static_cast<double>(
                            read_bytes_after - read_bytes_before));
                    fp_cache_rows_values.push_back(static_cast<double>(fp_cache_stats.rows));
                    fp_cache_bytes_values.push_back(static_cast<double>(fp_cache_stats.bytes));
                    fp_cache_hits_values.push_back(static_cast<double>(fp_cache_stats.hits));
                    fp_cache_misses_values.push_back(static_cast<double>(fp_cache_stats.misses));
                    fp_page_cache_pages_values.push_back(static_cast<double>(
                            fp_cache_stats.page_cache_pages));
                    fp_page_cache_bytes_values.push_back(static_cast<double>(
                            fp_cache_stats.page_cache_bytes));
                    fp_page_cache_accesses_values.push_back(static_cast<double>(
                            fp_cache_stats.page_cache_accesses));
                    fp_page_cache_hits_values.push_back(static_cast<double>(
                            fp_cache_stats.page_cache_hits));
                    fp_page_cache_misses_values.push_back(static_cast<double>(
                            fp_cache_stats.page_cache_misses));
                    fp_page_cache_evictions_values.push_back(static_cast<double>(
                            fp_cache_stats.page_cache_evictions));
                    direct_rows_values.push_back(static_cast<double>(
                            fp_cache_stats.direct_rows));
                    direct_requests_values.push_back(static_cast<double>(
                            fp_cache_stats.direct_requests));
                    direct_logical_bytes_values.push_back(static_cast<double>(
                            fp_cache_stats.direct_logical_bytes));
                    direct_submitted_bytes_values.push_back(static_cast<double>(
                            fp_cache_stats.direct_submitted_bytes));
                    intra_query_overlapped_queries_values.push_back(static_cast<double>(
                            intra_query_stats.queries_with_hit_miss_overlap));
                    intra_query_hit_rows_values.push_back(static_cast<double>(
                            intra_query_stats.cache_hit_rows_scored_while_io));
                    intra_query_io_requests_values.push_back(static_cast<double>(
                            intra_query_stats.io_requests));
                    intra_query_ready_completions_values.push_back(static_cast<double>(
                            intra_query_stats.completions_ready_after_hit_scoring));
                    intra_query_wait_seconds_values.push_back(
                            intra_query_stats.io_wait_seconds);
                    intra_query_hit_score_seconds_values.push_back(
                            intra_query_stats.hit_score_seconds);
                } else if (args.fp_rerank_overlap_queries) {
                    final_labels = std::move(pipelined_final_labels);
                    adaptive_count_mean = pipelined_rerank_count_stats.mean_count;
                    adaptive_count_min = pipelined_rerank_count_stats.min_count;
                    adaptive_count_p95 = pipelined_rerank_count_stats.p95_count;
                    adaptive_count_p99 = pipelined_rerank_count_stats.p99_count;
                    adaptive_count_max = pipelined_rerank_count_stats.max_count;
                    fp_cache_stats = pipelined_cache_stats;
                    const uint64_t read_bytes_after = read_proc_self_read_bytes();
                    rerank_read_bytes_values.push_back(static_cast<double>(
                            read_bytes_after - pipelined_read_bytes_before));
                    fp_cache_rows_values.push_back(static_cast<double>(fp_cache_stats.rows));
                    fp_cache_bytes_values.push_back(static_cast<double>(fp_cache_stats.bytes));
                    fp_cache_hits_values.push_back(static_cast<double>(fp_cache_stats.hits));
                    fp_cache_misses_values.push_back(static_cast<double>(fp_cache_stats.misses));
                    fp_page_cache_pages_values.push_back(0.0);
                    fp_page_cache_bytes_values.push_back(0.0);
                    fp_page_cache_accesses_values.push_back(0.0);
                    fp_page_cache_hits_values.push_back(0.0);
                    fp_page_cache_misses_values.push_back(0.0);
                    fp_page_cache_evictions_values.push_back(0.0);
                    direct_rows_values.push_back(static_cast<double>(fp_cache_stats.direct_rows));
                    direct_requests_values.push_back(static_cast<double>(fp_cache_stats.direct_requests));
                    direct_logical_bytes_values.push_back(static_cast<double>(fp_cache_stats.direct_logical_bytes));
                    direct_submitted_bytes_values.push_back(static_cast<double>(fp_cache_stats.direct_submitted_bytes));
                    pipeline_overlapped_queries_values.push_back(static_cast<double>(
                            pipelined_stats.queries_overlapped_with_search));
                    pipeline_ready_completions_values.push_back(static_cast<double>(
                            pipelined_stats.completions_ready_after_search));
                    pipeline_io_requests_values.push_back(static_cast<double>(
                            pipelined_stats.io_requests));
                    pipeline_overlapped_io_requests_values.push_back(static_cast<double>(
                            pipelined_stats.io_requests_overlapped_with_search));
                    pipeline_wait_seconds_values.push_back(pipelined_stats.io_wait_seconds);
                    pipeline_score_seconds_values.push_back(pipelined_stats.score_seconds);
                    intra_query_overlapped_queries_values.push_back(static_cast<double>(
                            pipelined_intra_query_stats.queries_with_hit_miss_overlap));
                    intra_query_hit_rows_values.push_back(static_cast<double>(
                            pipelined_intra_query_stats.cache_hit_rows_scored_while_io));
                    intra_query_io_requests_values.push_back(static_cast<double>(
                            pipelined_intra_query_stats.io_requests));
                    intra_query_ready_completions_values.push_back(static_cast<double>(
                            pipelined_intra_query_stats.completions_ready_after_hit_scoring));
                    intra_query_wait_seconds_values.push_back(
                            pipelined_intra_query_stats.io_wait_seconds);
                    intra_query_hit_score_seconds_values.push_back(
                            pipelined_intra_query_stats.hit_score_seconds);
                } else {
                    t0 = std::chrono::steady_clock::now();
                    std::vector<faiss::idx_t> adaptive_candidates;
                    const std::vector<faiss::idx_t>* rerank_candidates = &candidates;
                    if (args.adaptive_rerank) {
                        AdaptiveRerankSelection selection = select_adaptive_rerank_candidates(
                                candidates,
                                distances,
                                nq,
                                args.candidate_pool_size,
                                args.k,
                                args.adaptive_rerank_min,
                                args.adaptive_rerank_alpha);
                        adaptive_count_mean = selection.mean_count;
                        adaptive_count_min = static_cast<double>(selection.min_count);
                        adaptive_count_p95 = selection.p95_count;
                        adaptive_count_p99 = selection.p99_count;
                        adaptive_count_max = static_cast<double>(selection.max_count);
                        query_selected_counts = selection.counts;
                        query_selection_seconds = selection.selection_seconds;
                        adaptive_candidates = std::move(selection.labels);
                        rerank_candidates = &adaptive_candidates;
                    } else if (args.calibrated_threshold_rerank) {
                        AdaptiveRerankSelection selection =
                                select_calibrated_threshold_rerank_candidates(
                                        candidates,
                                        distances,
                                        queries.data(),
                                        nq,
                                        train.d,
                                        args.candidate_pool_size,
                                        args.k,
                                        args.calibrated_threshold_rerank_tau,
                                        args.adc_mode,
                                        args.lut_mode);
                        adaptive_count_mean = selection.mean_count;
                        adaptive_count_min = static_cast<double>(selection.min_count);
                        adaptive_count_p95 = selection.p95_count;
                        adaptive_count_p99 = selection.p99_count;
                        adaptive_count_max = static_cast<double>(selection.max_count);
                        query_selected_counts = selection.counts;
                        query_selection_seconds = selection.selection_seconds;
                        adaptive_candidates = std::move(selection.labels);
                        rerank_candidates = &adaptive_candidates;
                    } else {
                        AdaptiveRerankSelection selection =
                                select_fixed_rerank_candidates(
                                        candidates,
                                        nq,
                                        args.candidate_pool_size,
                                        args.rerank_n);
                        adaptive_count_mean = selection.mean_count;
                        adaptive_count_min = static_cast<double>(selection.min_count);
                        adaptive_count_p95 = selection.p95_count;
                        adaptive_count_p99 = selection.p99_count;
                        adaptive_count_max = static_cast<double>(selection.max_count);
                        query_selected_counts = selection.counts;
                        query_selection_seconds = selection.selection_seconds;
                        adaptive_candidates = std::move(selection.labels);
                        rerank_candidates = &adaptive_candidates;
                    }
                    const uint64_t read_bytes_before = read_proc_self_read_bytes();
                    if (args.rerank_dot_mode == RerankDotMode::Int8) {
                        final_labels = rerank_int8_ram(
                                int8_rerank_index,
                                queries.data(),
                                nq,
                                *rerank_candidates,
                                args.candidate_pool_size,
                                args.k);
                    } else {
                        final_labels = rerank_cosine(
                                *fp_rerank_store,
                                train_norms,
                                queries.data(),
                                nq,
                                *rerank_candidates,
                                args.candidate_pool_size,
                                args.k,
                                args.rerank_dot_mode,
                                args.fp_rerank_metric,
                                args.fp_rerank_source,
                                args.fp_rerank_norm_mode,
                                args.fp_rerank_cache,
                                args.fp_rerank_cache_rows,
                                args.fp_rerank_page_cache_mib,
                                args.fp_rerank_overlap_cache_hits,
                                args.fp_rerank_qd,
                                physical_row_by_id,
                                args.colocation_io_merge_bytes,
                                &fp_cache_stats,
                                &intra_query_stats);
                    }
                    const uint64_t read_bytes_after = read_proc_self_read_bytes();
                    rerank_read_bytes_values.push_back(static_cast<double>(read_bytes_after - read_bytes_before));
                    fp_cache_rows_values.push_back(static_cast<double>(fp_cache_stats.rows));
                    fp_cache_bytes_values.push_back(static_cast<double>(fp_cache_stats.bytes));
                    fp_cache_hits_values.push_back(static_cast<double>(fp_cache_stats.hits));
                    fp_cache_misses_values.push_back(static_cast<double>(fp_cache_stats.misses));
                    fp_page_cache_pages_values.push_back(static_cast<double>(fp_cache_stats.page_cache_pages));
                    fp_page_cache_bytes_values.push_back(static_cast<double>(fp_cache_stats.page_cache_bytes));
                    fp_page_cache_accesses_values.push_back(static_cast<double>(fp_cache_stats.page_cache_accesses));
                    fp_page_cache_hits_values.push_back(static_cast<double>(fp_cache_stats.page_cache_hits));
                    fp_page_cache_misses_values.push_back(static_cast<double>(fp_cache_stats.page_cache_misses));
                    fp_page_cache_evictions_values.push_back(static_cast<double>(fp_cache_stats.page_cache_evictions));
                    direct_rows_values.push_back(static_cast<double>(fp_cache_stats.direct_rows));
                    direct_requests_values.push_back(static_cast<double>(fp_cache_stats.direct_requests));
                    direct_logical_bytes_values.push_back(static_cast<double>(fp_cache_stats.direct_logical_bytes));
                    direct_submitted_bytes_values.push_back(static_cast<double>(fp_cache_stats.direct_submitted_bytes));
                    intra_query_overlapped_queries_values.push_back(static_cast<double>(
                            intra_query_stats.queries_with_hit_miss_overlap));
                    intra_query_hit_rows_values.push_back(static_cast<double>(
                            intra_query_stats.cache_hit_rows_scored_while_io));
                    intra_query_io_requests_values.push_back(static_cast<double>(
                            intra_query_stats.io_requests));
                    intra_query_ready_completions_values.push_back(static_cast<double>(
                            intra_query_stats.completions_ready_after_hit_scoring));
                    intra_query_wait_seconds_values.push_back(intra_query_stats.io_wait_seconds);
                    intra_query_hit_score_seconds_values.push_back(intra_query_stats.hit_score_seconds);
                    rerank_s = seconds_since(t0);
                }
                const double total_s = args.strict_query_e2e
                        ? strict_total_s
                        : search_s + rerank_s;
                const double recall = recall_at_k(final_labels, gt, nq, gt_dim, args.k);
                if (final_labels_csv.is_open()) {
                    for (int qi = 0; qi < nq; ++qi) {
                        for (int rank = 0; rank < args.k; ++rank) {
                            final_labels_csv
                                    << args.dataset << ',' << ef << ','
                                    << args.candidate_pool_size << ',' << round << ','
                                    << qi << ',' << rank << ','
                                    << final_labels[static_cast<size_t>(qi) * args.k + rank]
                                    << '\n';
                        }
                    }
                    final_labels_csv.flush();
                }
                if (per_query_metrics_csv.is_open()) {
                    const size_t expected = static_cast<size_t>(nq);
                    if (query_search_seconds.size() != expected ||
                        query_selected_counts.size() != expected ||
                        query_selection_seconds.size() != expected ||
                        intra_query_stats.per_query_io_wait_seconds.size() != expected ||
                        intra_query_stats.per_query_rerank_seconds.size() != expected ||
                        (args.strict_query_e2e &&
                         query_end_to_end_seconds.size() != expected)) {
                        throw std::runtime_error(
                                "per-query metric vector size mismatch");
                    }
                    const char* policy = args.calibrated_threshold_rerank
                            ? "calibrated_threshold"
                            : (args.adaptive_rerank ? "adaptive_heuristic" : "fixed_r");
                    for (int qi = 0; qi < nq; ++qi) {
                        int query_hits = 0;
                        for (int rank = 0; rank < args.k; ++rank) {
                            const faiss::idx_t id = final_labels[
                                    static_cast<size_t>(qi) * args.k + rank];
                            for (int gold_rank = 0; gold_rank < args.k; ++gold_rank) {
                                if (id == gt[static_cast<size_t>(qi) * gt_dim + gold_rank]) {
                                    ++query_hits;
                                    break;
                                }
                            }
                        }
                        const double search_us =
                                query_search_seconds[static_cast<size_t>(qi)] * 1e6;
                        const double selection_us =
                                query_selection_seconds[static_cast<size_t>(qi)] * 1e6;
                        const double io_wait_us =
                                intra_query_stats.per_query_io_wait_seconds[
                                        static_cast<size_t>(qi)] * 1e6;
                        const double rerank_us =
                                intra_query_stats.per_query_rerank_seconds[
                                        static_cast<size_t>(qi)] * 1e6;
                        const double end_to_end_us = args.strict_query_e2e
                                ? query_end_to_end_seconds[
                                          static_cast<size_t>(qi)] * 1e6
                                : search_us + selection_us + rerank_us;
                        per_query_metrics_csv
                                << args.dataset << ',' << ef << ','
                                << args.candidate_pool_size << ',' << policy << ','
                                << (args.strict_query_e2e
                                            ? "strict_query_wall"
                                            : "component_sum")
                                << ',' << round << ',' << qi << ','
                                << query_selected_counts[static_cast<size_t>(qi)] << ','
                                << search_us << ',' << selection_us << ','
                                << io_wait_us << ',' << rerank_us << ','
                                << end_to_end_us << ','
                                << static_cast<double>(query_hits) / args.k << '\n';
                    }
                    per_query_metrics_csv.flush();
                }
                const double qps = static_cast<double>(nq) / std::max(total_s, 1e-12);
                recalls.push_back(recall);
                search_secs.push_back(search_s);
                rerank_secs.push_back(rerank_s);
                total_secs.push_back(total_s);
                qps_values.push_back(qps);
                adaptive_count_means.push_back(adaptive_count_mean);
                adaptive_count_mins.push_back(adaptive_count_min);
                adaptive_count_p95s.push_back(adaptive_count_p95);
                adaptive_count_p99s.push_back(adaptive_count_p99);
                adaptive_count_maxs.push_back(adaptive_count_max);
                const char* search_label = args.distance_mode == DistanceMode::XFlagBQ
                        ? (args.no_rerank ? "XFlagBQ" : "XFlagBQ+rerank")
                        : (args.distance_mode == DistanceMode::AdcXFlagBQ
                                ? (args.no_rerank ? "AdcXFlagBQ" : "AdcXFlagBQ+rerank")
                                : (args.no_rerank ? "PackedAsymBQ" : "PackedAsymBQ+rerank"));
                std::cout << "  "
                          << search_label
                          << "(" << distance_mode_name(args.distance_mode)
                          << ",adc=" << adc_mode_name(args.adc_mode)
                          << ",lut=" << lut_mode_name(args.lut_mode)
                          << ",rerank_dot=" << rerank_dot_mode_name(args.rerank_dot_mode)
                          << ",fp_metric=" << fp_rerank_metric_name(args.fp_rerank_metric)
                          << ",fp_source=" << fp_rerank_source_name(args.fp_rerank_source)
                          << ",fp_norms=" << fp_rerank_norm_mode_name(args.fp_rerank_norm_mode) << ")"
                          << " ef=" << ef << " round=" << (round + 1)
                          << "/" << args.rounds << " recall@" << args.k << "=" << recall
                          << " qps=" << qps;
                if (index_page_cache_stats.accesses > 0) {
                    std::cout << " index_page_cache=" << index_page_cache_stats.resident_pages
                              << "pages hits=" << index_page_cache_stats.hits
                              << " misses=" << index_page_cache_stats.misses
                              << " hit_ratio="
                              << static_cast<double>(index_page_cache_stats.hits) /
                                      index_page_cache_stats.accesses
                              << " read_bytes=" << index_page_cache_stats.read_bytes
                              << " evictions=" << index_page_cache_stats.evictions;
                    if (index_page_cache_stats.async_batches > 0) {
                        std::cout
                              << " graph_async_batches="
                              << index_page_cache_stats.async_batches
                              << " graph_async_requests="
                              << index_page_cache_stats.async_requests
                              << " cached_nodes_while_inflight="
                              << index_page_cache_stats.async_cached_nodes_scored_while_inflight
                              << " completed_nodes_while_inflight="
                              << index_page_cache_stats.async_completed_nodes_scored_while_inflight
                              << " ready_before_wait="
                              << index_page_cache_stats.async_completions_ready_before_wait
                              << " async_wait_s="
                              << index_page_cache_stats.async_wait_seconds;
                    }
                }
                if (mmap_index) {
                    std::cout << " mmap_minor_faults=" << mmap_minor_faults_values.back()
                              << " mmap_major_faults=" << mmap_major_faults_values.back()
                              << " mmap_read_bytes=" << mmap_index_read_bytes_values.back()
                              << " mmap_resident_pages=" << mmap_resident_pages_before_values.back()
                              << "->" << mmap_resident_pages_after_values.back();
                }
                if (!args.no_rerank) {
                    std::cout << " rerank_read_bytes=" << rerank_read_bytes_values.back();
                    if (args.fp_rerank_cache && args.rerank_dot_mode != RerankDotMode::Int8) {
                        std::cout << " fp_cache_rows=" << fp_cache_stats.rows
                                  << " fp_cache_hits=" << fp_cache_stats.hits
                                  << " fp_cache_misses=" << fp_cache_stats.misses;
                    }
                    if (fp_cache_stats.page_cache_accesses > 0) {
                        std::cout << " fp_page_cache=" << fp_cache_stats.page_cache_pages
                                  << "pages hits=" << fp_cache_stats.page_cache_hits
                                  << " misses=" << fp_cache_stats.page_cache_misses
                                  << " hit_ratio="
                                  << static_cast<double>(fp_cache_stats.page_cache_hits) /
                                          fp_cache_stats.page_cache_accesses;
                    }
                    if (fp_cache_stats.direct_rows > 0) {
                        const double request_reduction = 1.0 -
                                static_cast<double>(fp_cache_stats.direct_requests) /
                                        fp_cache_stats.direct_rows;
                        const double byte_reuse = 1.0 -
                                static_cast<double>(fp_cache_stats.direct_submitted_bytes) /
                                        fp_cache_stats.direct_logical_bytes;
                        std::cout << " direct_requests=" << fp_cache_stats.direct_requests
                                  << "/" << fp_cache_stats.direct_rows
                                  << " request_reduction=" << request_reduction
                                  << " byte_reuse=" << byte_reuse;
                    }
                    if (args.fp_rerank_overlap_queries) {
                        const double ready_ratio = pipelined_stats.io_requests_overlapped_with_search > 0
                                ? static_cast<double>(pipelined_stats.completions_ready_after_search) /
                                        pipelined_stats.io_requests_overlapped_with_search
                                : 0.0;
                        std::cout << " pipeline_overlap_queries="
                                  << pipelined_stats.queries_overlapped_with_search
                                  << "/" << pipelined_stats.queries
                                  << " ready_after_next_search="
                                  << pipelined_stats.completions_ready_after_search
                                  << "/" << pipelined_stats.io_requests_overlapped_with_search
                                  << " ready_ratio=" << ready_ratio
                                  << " wait_seconds_sum=" << pipelined_stats.io_wait_seconds;
                    }
                }
                if (args.adaptive_rerank || args.calibrated_threshold_rerank) {
                    std::cout << " adaptive_rerank_mean=" << adaptive_count_mean
                              << " min=" << adaptive_count_min
                              << " p95=" << adaptive_count_p95
                              << " p99=" << adaptive_count_p99
                              << " max=" << adaptive_count_max;
                }
                std::cout << std::endl;
            }
            auto mean = [](const std::vector<double>& v) {
                return std::accumulate(v.begin(), v.end(), 0.0) / std::max<size_t>(v.size(), 1);
            };
            const double search_mean = mean(search_secs);
            const double rerank_mean = mean(rerank_secs);
            const double total_mean = mean(total_secs);
            const double recall_mean = mean(recalls);
            const double qps_mean = mean(qps_values);
            auto sorted_qps = qps_values;
            std::sort(sorted_qps.begin(), sorted_qps.end());
            const double qps_median = sorted_qps.empty()
                    ? 0.0
                    : (sorted_qps.size() % 2 == 0
                            ? 0.5 * (sorted_qps[sorted_qps.size() / 2 - 1] +
                                     sorted_qps[sorted_qps.size() / 2])
                            : sorted_qps[sorted_qps.size() / 2]);
            const double rerank_read_bytes_mean = mean(rerank_read_bytes_values);
            const double rerank_read_bytes_per_query = rerank_read_bytes_mean / std::max(nq, 1);
            const double fp_cache_rows_mean = mean(fp_cache_rows_values);
            const double fp_cache_bytes_mean = mean(fp_cache_bytes_values);
            const double fp_cache_hits_mean = mean(fp_cache_hits_values);
            const double fp_cache_misses_mean = mean(fp_cache_misses_values);
            const double fp_page_cache_pages_mean = mean(fp_page_cache_pages_values);
            const double fp_page_cache_bytes_mean = mean(fp_page_cache_bytes_values);
            const double fp_page_cache_accesses_mean = mean(fp_page_cache_accesses_values);
            const double fp_page_cache_hits_mean = mean(fp_page_cache_hits_values);
            const double fp_page_cache_misses_mean = mean(fp_page_cache_misses_values);
            const double fp_page_cache_evictions_mean = mean(fp_page_cache_evictions_values);
            const double fp_page_cache_hit_ratio = fp_page_cache_accesses_mean > 0.0
                    ? fp_page_cache_hits_mean / fp_page_cache_accesses_mean
                    : 0.0;
            const double index_page_cache_pages_mean = mean(index_page_cache_pages_values);
            const double index_data_pages_mean = mean(index_data_pages_values);
            const double index_page_accesses_mean = mean(index_page_accesses_values);
            const double index_page_hits_mean = mean(index_page_hits_values);
            const double index_page_misses_mean = mean(index_page_misses_values);
            const double index_page_evictions_mean = mean(index_page_evictions_values);
            const double index_page_read_bytes_mean = mean(index_page_read_bytes_values);
            const double index_page_hit_ratio = index_page_accesses_mean > 0.0
                    ? index_page_hits_mean / index_page_accesses_mean
                    : 0.0;
            const double index_hybrid_sync_batches_mean =
                    mean(index_hybrid_sync_batches_values);
            const double index_hybrid_sync_requests_mean =
                    mean(index_hybrid_sync_requests_values);
            const double index_hybrid_sync_wait_seconds_mean =
                    mean(index_hybrid_sync_wait_seconds_values);
            const double index_async_batches_mean = mean(index_async_batches_values);
            const double index_async_requests_mean = mean(index_async_requests_values);
            const double index_async_cached_nodes_mean = mean(index_async_cached_nodes_values);
            const double index_async_completed_nodes_mean =
                    mean(index_async_completed_nodes_values);
            const double index_async_ready_completions_mean =
                    mean(index_async_ready_completions_values);
            const double index_async_ready_ratio = index_async_requests_mean > 0.0
                    ? index_async_ready_completions_mean / index_async_requests_mean
                    : 0.0;
            const double index_async_wait_seconds_mean =
                    mean(index_async_wait_seconds_values);
            const double mmap_minor_faults_mean = mean(mmap_minor_faults_values);
            const double mmap_major_faults_mean = mean(mmap_major_faults_values);
            const double mmap_index_read_bytes_mean = mean(mmap_index_read_bytes_values);
            const double mmap_resident_pages_before_mean = mean(mmap_resident_pages_before_values);
            const double mmap_resident_pages_after_mean = mean(mmap_resident_pages_after_values);
            const double direct_rows_mean = mean(direct_rows_values);
            const double direct_requests_mean = mean(direct_requests_values);
            const double direct_logical_bytes_mean = mean(direct_logical_bytes_values);
            const double direct_submitted_bytes_mean = mean(direct_submitted_bytes_values);
            const double direct_request_reduction_ratio = direct_rows_mean > 0.0
                    ? 1.0 - direct_requests_mean / direct_rows_mean
                    : 0.0;
            const double direct_byte_reuse_ratio = direct_logical_bytes_mean > 0.0
                    ? 1.0 - direct_submitted_bytes_mean / direct_logical_bytes_mean
                    : 0.0;
            const double adaptive_count_mean = mean(adaptive_count_means);
            const double adaptive_count_min = mean(adaptive_count_mins);
            const double adaptive_count_p95 = mean(adaptive_count_p95s);
            const double adaptive_count_p99 = mean(adaptive_count_p99s);
            const double adaptive_count_max = mean(adaptive_count_maxs);
            const double pipeline_overlapped_queries_mean = mean(pipeline_overlapped_queries_values);
            const double pipeline_io_requests_mean = mean(pipeline_io_requests_values);
            const double pipeline_overlapped_io_requests_mean =
                    mean(pipeline_overlapped_io_requests_values);
            const double pipeline_ready_completions_mean = mean(pipeline_ready_completions_values);
            const double pipeline_ready_ratio = pipeline_overlapped_io_requests_mean > 0.0
                    ? pipeline_ready_completions_mean / pipeline_overlapped_io_requests_mean
                    : 0.0;
            const double pipeline_wait_seconds_mean = mean(pipeline_wait_seconds_values);
            const double pipeline_score_seconds_mean = mean(pipeline_score_seconds_values);
            const double intra_query_overlapped_queries_mean =
                    mean(intra_query_overlapped_queries_values);
            const double intra_query_hit_rows_mean = mean(intra_query_hit_rows_values);
            const double intra_query_io_requests_mean = mean(intra_query_io_requests_values);
            const double intra_query_ready_completions_mean =
                    mean(intra_query_ready_completions_values);
            const double intra_query_ready_ratio = intra_query_io_requests_mean > 0.0
                    ? intra_query_ready_completions_mean / intra_query_io_requests_mean
                    : 0.0;
            const double intra_query_wait_seconds_mean =
                    mean(intra_query_wait_seconds_values);
            const double intra_query_hit_score_seconds_mean =
                    mean(intra_query_hit_score_seconds_values);
            csv << args.dataset << ','
                << method_name(args) << ',' << distance_mode_name(args.distance_mode) << ','
                << adc_mode_name(args.adc_mode) << ',' << lut_mode_name(args.lut_mode) << ','
                << rerank_dot_mode_name(args.rerank_dot_mode) << ','
                << fp_rerank_metric_name(args.fp_rerank_metric) << ','
                << fp_rerank_source_name(args.fp_rerank_source) << ','
                << fp_rerank_norm_mode_name(args.fp_rerank_norm_mode) << ','
                << graph_build_mode_name(args.graph_build_mode) << ','
                << args.k << ',' << args.search_threads << ',' << nq << ',' << train.n << ',' << train.d << ',' << args.m << ','
                << args.ef_construction << ',' << ef << ',' << (args.no_rerank ? "false" : "true")
                << ',' << (args.no_rerank ? 0 : args.rerank_n) << ','
                << args.candidate_pool_size << ','
                << (args.adaptive_rerank ? "true" : "false") << ','
                << (args.adaptive_rerank ? args.adaptive_rerank_min : 0) << ','
                << (args.adaptive_rerank ? args.adaptive_rerank_alpha : 0.0f) << ','
                << (args.calibrated_threshold_rerank ? "true" : "false") << ','
                << (args.calibrated_threshold_rerank
                        ? args.calibrated_threshold_rerank_tau
                        : 0.0f) << ','
                << adaptive_count_mean << ',' << adaptive_count_min << ','
                << adaptive_count_p95 << ',' << adaptive_count_p99 << ','
                << adaptive_count_max << ','
                << args.rounds << ',' << (args.rebuild ? "built" : "loaded_or_built") << ','
                << build_or_load_seconds << ',' << norm_precompute_seconds << ','
                << int8_rerank_prep_seconds << ',' << int8_rerank_bytes << ','
                << xflag_prep_seconds << ',' << search_mean << ',' << rerank_mean << ','
                << total_mean << ',' << rerank_read_bytes_mean << ',' << rerank_read_bytes_per_query << ','
                << (args.fp_rerank_cache ? "true" : "false") << ','
                << (args.fp_rerank_cache ? args.fp_rerank_cache_rows : 0) << ','
                << fp_cache_rows_mean << ',' << fp_cache_bytes_mean << ','
                << fp_cache_hits_mean << ',' << fp_cache_misses_mean << ','
                << args.fp_rerank_page_cache_mib << ','
                << fp_page_cache_pages_mean << ',' << fp_page_cache_bytes_mean << ','
                << fp_page_cache_accesses_mean << ',' << fp_page_cache_hits_mean << ','
                << fp_page_cache_misses_mean << ',' << fp_page_cache_evictions_mean << ','
                << fp_page_cache_hit_ratio << ','
                << args.index_page_cache_mib << ',' << kPagedIndexPageBytes << ','
                << index_page_cache_pages_mean << ',' << index_data_pages_mean << ','
                << index_page_accesses_mean << ',' << index_page_hits_mean << ','
                << index_page_misses_mean << ',' << index_page_evictions_mean << ','
                << index_page_hit_ratio << ',' << index_page_read_bytes_mean << ','
                << (args.index_page_async ? "true" : "false") << ','
                << args.index_page_qd << ','
                << args.index_page_async_sync_miss_max << ','
                << index_hybrid_sync_batches_mean << ','
                << index_hybrid_sync_requests_mean << ','
                << index_hybrid_sync_wait_seconds_mean << ','
                << index_async_batches_mean << ','
                << index_async_requests_mean << ',' << index_async_cached_nodes_mean << ','
                << index_async_completed_nodes_mean << ','
                << index_async_ready_completions_mean << ',' << index_async_ready_ratio << ','
                << index_async_wait_seconds_mean << ','
                << active_paged_index_path << ','
                << paged_index_prepare_stats.seconds << ','
                << (active_paged_index_path.empty()
                        ? "disabled"
                        : (paged_index_prepare_stats.loaded ? "loaded" : "built")) << ','
                << paged_index_prepare_stats.file_bytes << ','
                << paged_index_prepare_stats.data_pages << ','
                << (mmap_index ? "true" : "false") << ','
                << (mmap_index ? mmap_index->path() : "") << ','
                << (mmap_index ? mmap_index->file_bytes() : 0) << ','
                << (mmap_index ? mmap_index->os_page_bytes() : 0) << ','
                << (args.mmap_drop_cache_before_round ? "true" : "false") << ','
                << mmap_minor_faults_mean << ',' << mmap_major_faults_mean << ','
                << mmap_index_read_bytes_mean << ','
                << mmap_resident_pages_before_mean << ',' << mmap_resident_pages_after_mean << ','
                << recall_mean << ',' << qps_mean << ',' << qps_median << ','
                << active_xflag_threshold << ',' << args.xflag_quantile << ',' << args.xflag_sample_rows << ','
                << xflag_code_status << ',' << args.index << ',' << xflag_code_path << ','
                << args.colocation_iterations << ',' << args.colocation_partition_size << ','
                << args.colocation_io_merge_bytes << ','
                << colocation.prepare_seconds << ','
                << colocation_apply_seconds << ','
                << (args.colocation_iterations > 0 ? (colocation.loaded ? "loaded" : "built") : "disabled") << ','
                << colocation.map_path << ','
                << direct_rows_mean << ',' << direct_requests_mean << ','
                << direct_logical_bytes_mean << ',' << direct_submitted_bytes_mean << ','
                << direct_request_reduction_ratio << ',' << direct_byte_reuse_ratio << ','
                << (args.fp_rerank_overlap_queries ? "true" : "false") << ','
                << args.fp_rerank_qd << ','
                << pipeline_overlapped_queries_mean << ',' << pipeline_io_requests_mean << ','
                << pipeline_overlapped_io_requests_mean << ','
                << pipeline_ready_completions_mean << ',' << pipeline_ready_ratio << ','
                << pipeline_wait_seconds_mean << ',' << pipeline_score_seconds_mean << ','
                << (args.fp_rerank_overlap_cache_hits ? "true" : "false") << ','
                << intra_query_overlapped_queries_mean << ','
                << intra_query_hit_rows_mean << ',' << intra_query_io_requests_mean << ','
                << intra_query_ready_completions_mean << ',' << intra_query_ready_ratio << ','
                << intra_query_wait_seconds_mean << ',' << intra_query_hit_score_seconds_mean << ','
                << fp_rerank_store->path << '\n';
            csv.flush();
            if (stats_csv.is_open()) {
                write_adcbq_stats_row(
                        stats_csv,
                        args,
                        nq,
                        train.n,
                        train.d,
                        ef,
                        search_mean,
                        rerank_mean,
                        total_mean,
                        recall_mean,
                        qps_mean,
                        stats_accum);
                stats_csv.flush();
            }
        }
        std::cout << "CSV saved -> " << args.out << std::endl;
        if (!args.stats_out.empty()) {
            std::cout << "Stats CSV saved -> " << args.stats_out << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
