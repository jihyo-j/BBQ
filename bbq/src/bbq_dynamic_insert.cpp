// Dynamic-insertion benchmark for the production ADC-BQ configuration.
//
// Graph: normalized FP32 inner-product HNSW
// Search payload: packed one-bit sign codes, Int16 nibble LUT ADC
// Final ranking: exact FP32 dot-product rerank over a fixed candidate count
// SSD mode: 4 KiB-aligned FP store, O_DIRECT + io_uring, with inter-query
//           overlap between the previous query's reads and the next HNSW walk.
//
// The benchmark compares cumulative online add against a fresh full rebuild at
// each checkpoint.  Dynamic labels are [base rows, selected insert rows].  Its
// online prototype appends raw FP32 payloads before publishing the matching BQ
// nodes into the same live HNSW. In ADC-BQ add mode, each incoming FP32 vector
// is the asymmetric query used to route over the resident BQ codes. Search takes
// a reader lock only while using the graph and its FP layout; each micro-batch is
// published under a writer lock.
// The append log can later be compacted into graph-locality order without
// rebuilding or merging a second graph.

#define main adcbq_hnsw_benchmark_main
#include "bbq_hnsw.cpp"
#undef main

#include <faiss/clone_index.h>
#include <faiss/utils/random.h>

#include <atomic>
#include <exception>
#include <iomanip>
#include <optional>
#include <regex>
#include <shared_mutex>
#include <thread>
#include <unordered_set>

#include "qdrant_tq_ffi/qdrant_tq_ffi.h"

namespace {

enum class DynamicGraphAddMode {
    Fp,
    BqHamming,
    AdcBq,
    Tq1,
};

DynamicGraphAddMode parse_dynamic_graph_add_mode(const std::string& value) {
    if (value == "fp") return DynamicGraphAddMode::Fp;
    if (value == "bq") return DynamicGraphAddMode::BqHamming;
    if (value == "adcbq") return DynamicGraphAddMode::AdcBq;
    if (value == "tq1") return DynamicGraphAddMode::Tq1;
    throw std::runtime_error("--online-graph-add must be fp, bq, adcbq, or tq1");
}

const char* dynamic_graph_add_mode_name(DynamicGraphAddMode mode) {
    switch (mode) {
        case DynamicGraphAddMode::Fp:
            return "fp";
        case DynamicGraphAddMode::BqHamming:
            return "bq";
        case DynamicGraphAddMode::AdcBq:
            return "adcbq";
        case DynamicGraphAddMode::Tq1:
            return "tq1";
    }
    throw std::runtime_error("invalid dynamic graph add mode");
}

struct DynamicArgs {
    std::string dataset = "msmarco-qwen1024-1m-to-5m";
    std::string base_train =
            "/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-1m_base.fbin";
    std::string larger_train =
            "/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-5m_base.fbin";
    std::string query =
            "/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-5m_query.fbin";
    std::string calibration_query;
    std::string base_gt =
            "/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/msmarco-1m_gt.ibin";
    std::string base_to_larger_ids;
    std::string selected_ids;
    std::string gt_dir;
    std::string out;
    std::string policy = "random";
    std::string fp_rerank_path;
    std::string fp_append_store_path;
    std::string fp_compacted_store_path;
    std::string tq_code_path;
    std::string tq_meta_path;
    std::string insert_emb_root;
    std::string insert_permutation;
    std::vector<int> insert_sizes = {1000000, 2000000, 4000000};
    std::vector<int> efs = {64, 128, 256, 512, 1024};
    int m = 24;
    int ef_construction = 200;
    int k = 10;
    int rerank_n = 80;
    int max_queries = 1000;
    int rounds = 1;
    int build_threads = 24;
    int add_threads = 24;
    int search_threads = 1;
    int add_batch_size = 50000;
    int io_queue_depth = 80;
    int fp_sort_iterations = 1;
    int fp_sort_partition_size = 64;
    int concurrent_search_ef = 256;
    double calibration_target = 0.99;
    FpRerankSource fp_source = FpRerankSource::ODirectUring;
    DynamicGraphAddMode online_graph_add = DynamicGraphAddMode::AdcBq;
    bool prepare_aligned_store = false;
    bool rebuild_fp_append_store = false;
    bool fp_durable_writes = false;
    bool search_during_add = false;
    bool overlap_queries = true;
    bool independent_online_checkpoints = false;
    bool freeze_online_calibration_from_base = false;
    bool skip_online = false;
    bool skip_rebuild = false;
};

[[noreturn]] void dynamic_usage(const char* argv0) {
    std::cerr
            << "Usage: " << argv0
            << " --base-to-larger-ids MAP.i64bin --selected-ids IDS.i64bin"
            << " --gt-dir DIR --out RESULT.csv [options]\n"
            << "  --fp-rerank-source mmap|odirect-uring\n"
            << "  --fp-rerank-path aligned.fbin --prepare-aligned-store\n"
            << "  --fp-append-store PATH [--rebuild-fp-append-store] [--fp-durable-writes]\n"
            << "  --fp-compact-final PATH [--fp-sort-iterations N]"
            << " [--fp-sort-partition-size N]\n"
            << "  --online-graph-add fp|bq|adcbq\n"
            << "  --tq-code-path BASE.tqdata --tq-meta-path BASE.meta\n"
            << "  --insert-emb-root CHUNK_DIR --insert-permutation PERM.npy\n"
            << "  --freeze-online-calibration-from-base\n"
            << "  --search-during-add [--concurrent-search-ef N]\n"
            << "  --insert-sizes 1000000 2000000 4000000\n"
            << "  --efs 64 128 256 512 1024\n";
    std::exit(2);
}

DynamicArgs parse_dynamic_args(int argc, char** argv) {
    DynamicArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (a == "--dataset") args.dataset = need("--dataset");
        else if (a == "--base-train") args.base_train = need("--base-train");
        else if (a == "--larger-train") args.larger_train = need("--larger-train");
        else if (a == "--query") args.query = need("--query");
        else if (a == "--calibration-query") args.calibration_query = need("--calibration-query");
        else if (a == "--base-gt") args.base_gt = need("--base-gt");
        else if (a == "--base-to-larger-ids") args.base_to_larger_ids = need("--base-to-larger-ids");
        else if (a == "--selected-ids") args.selected_ids = need("--selected-ids");
        else if (a == "--gt-dir") args.gt_dir = need("--gt-dir");
        else if (a == "--out") args.out = need("--out");
        else if (a == "--policy") args.policy = need("--policy");
        else if (a == "--fp-rerank-path") args.fp_rerank_path = need("--fp-rerank-path");
        else if (a == "--fp-append-store") args.fp_append_store_path = need("--fp-append-store");
        else if (a == "--fp-compact-final") args.fp_compacted_store_path = need("--fp-compact-final");
        else if (a == "--fp-rerank-source") args.fp_source = parse_fp_rerank_source(need("--fp-rerank-source"));
        else if (a == "--online-graph-add") args.online_graph_add = parse_dynamic_graph_add_mode(need("--online-graph-add"));
        else if (a == "--tq-code-path") args.tq_code_path = need("--tq-code-path");
        else if (a == "--tq-meta-path") args.tq_meta_path = need("--tq-meta-path");
        else if (a == "--insert-emb-root") args.insert_emb_root = need("--insert-emb-root");
        else if (a == "--insert-permutation") args.insert_permutation = need("--insert-permutation");
        else if (a == "--prepare-aligned-store") args.prepare_aligned_store = true;
        else if (a == "--rebuild-fp-append-store") args.rebuild_fp_append_store = true;
        else if (a == "--fp-durable-writes") args.fp_durable_writes = true;
        else if (a == "--search-during-add") args.search_during_add = true;
        else if (a == "--no-overlap") args.overlap_queries = false;
        else if (a == "--independent-online-checkpoints") args.independent_online_checkpoints = true;
        else if (a == "--freeze-online-calibration-from-base") args.freeze_online_calibration_from_base = true;
        else if (a == "--skip-online") args.skip_online = true;
        else if (a == "--skip-rebuild") args.skip_rebuild = true;
        else if (a == "--m") args.m = std::stoi(need("--m"));
        else if (a == "--ef-construction") args.ef_construction = std::stoi(need("--ef-construction"));
        else if (a == "--k") args.k = std::stoi(need("--k"));
        else if (a == "--rerank-n") args.rerank_n = std::stoi(need("--rerank-n"));
        else if (a == "--max-queries") args.max_queries = std::stoi(need("--max-queries"));
        else if (a == "--rounds") args.rounds = std::stoi(need("--rounds"));
        else if (a == "--build-threads") args.build_threads = std::stoi(need("--build-threads"));
        else if (a == "--add-threads") args.add_threads = std::stoi(need("--add-threads"));
        else if (a == "--search-threads") args.search_threads = std::stoi(need("--search-threads"));
        else if (a == "--add-batch-size") args.add_batch_size = std::stoi(need("--add-batch-size"));
        else if (a == "--io-queue-depth") args.io_queue_depth = std::stoi(need("--io-queue-depth"));
        else if (a == "--fp-sort-iterations") args.fp_sort_iterations = std::stoi(need("--fp-sort-iterations"));
        else if (a == "--fp-sort-partition-size") args.fp_sort_partition_size = std::stoi(need("--fp-sort-partition-size"));
        else if (a == "--concurrent-search-ef") args.concurrent_search_ef = std::stoi(need("--concurrent-search-ef"));
        else if (a == "--calibration-target") args.calibration_target = std::stod(need("--calibration-target"));
        else if (a == "--insert-sizes") {
            args.insert_sizes.clear();
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                args.insert_sizes.push_back(std::stoi(argv[++i]));
            }
        } else if (a == "--efs") {
            args.efs.clear();
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                args.efs.push_back(std::stoi(argv[++i]));
            }
        } else {
            throw std::runtime_error("unknown argument: " + a);
        }
    }

    if (args.base_to_larger_ids.empty() || args.selected_ids.empty() ||
        args.gt_dir.empty() || args.out.empty() || args.insert_sizes.empty() ||
        args.efs.empty()) {
        dynamic_usage(argv[0]);
    }
    if (args.policy != "random" && args.policy != "far") {
        throw std::runtime_error("--policy must be random or far");
    }
    if (args.k <= 0 || args.rerank_n < args.k || args.rounds <= 0 ||
        args.build_threads <= 0 || args.add_threads <= 0 ||
        args.search_threads <= 0 || args.add_batch_size <= 0) {
        throw std::runtime_error("invalid non-positive benchmark parameter");
    }
    if (!args.calibration_query.empty() &&
        (!(args.calibration_target > 0.0) || !(args.calibration_target < 1.0))) {
        throw std::runtime_error("--calibration-target must be strictly between zero and one");
    }
    if (args.freeze_online_calibration_from_base && args.calibration_query.empty()) {
        throw std::runtime_error(
                "--freeze-online-calibration-from-base requires --calibration-query");
    }
    if (args.fp_source != FpRerankSource::Mmap &&
        args.fp_source != FpRerankSource::ODirectUring) {
        throw std::runtime_error("dynamic runner supports mmap or odirect-uring rerank only");
    }
    if (args.fp_source == FpRerankSource::ODirectUring && args.fp_rerank_path.empty() &&
        args.fp_append_store_path.empty()) {
        throw std::runtime_error("odirect-uring requires --fp-rerank-path or --fp-append-store");
    }
    if (args.overlap_queries && args.fp_source != FpRerankSource::ODirectUring) {
        args.overlap_queries = false;
    }
    if (args.overlap_queries && args.io_queue_depth < args.rerank_n) {
        throw std::runtime_error("overlap mode requires --io-queue-depth >= --rerank-n");
    }
    if (args.online_graph_add == DynamicGraphAddMode::Tq1 &&
        (args.tq_code_path.empty() || args.tq_meta_path.empty())) {
        throw std::runtime_error("tq1 graph add requires --tq-code-path and --tq-meta-path");
    }
    if (args.insert_emb_root.empty() != args.insert_permutation.empty()) {
        throw std::runtime_error(
                "--insert-emb-root and --insert-permutation must be provided together");
    }
    if (!args.insert_emb_root.empty() &&
        args.online_graph_add != DynamicGraphAddMode::Tq1) {
        throw std::runtime_error("chunked insert source is currently supported only for tq1");
    }
    if (!args.insert_emb_root.empty() &&
        (args.fp_source != FpRerankSource::Mmap ||
         !args.fp_append_store_path.empty() ||
         !args.fp_compacted_store_path.empty() ||
         args.prepare_aligned_store)) {
        throw std::runtime_error(
                "chunked insert source requires mmap rerank without an FP append/compact store");
    }
    if (args.online_graph_add == DynamicGraphAddMode::Tq1 &&
        (args.overlap_queries || args.search_during_add)) {
        throw std::runtime_error(
                "tq1 dynamic mode currently requires --no-overlap and no concurrent search");
    }
    if (!args.fp_compacted_store_path.empty() && args.fp_append_store_path.empty()) {
        throw std::runtime_error("--fp-compact-final requires --fp-append-store");
    }
    if (!args.fp_compacted_store_path.empty() && args.skip_online) {
        throw std::runtime_error("--fp-compact-final requires the online run");
    }
    if (!args.fp_append_store_path.empty() && args.skip_online) {
        throw std::runtime_error("--fp-append-store requires the online run");
    }
    if (args.prepare_aligned_store && !args.fp_append_store_path.empty()) {
        throw std::runtime_error("--prepare-aligned-store and --fp-append-store are mutually exclusive");
    }
    if ((args.rebuild_fp_append_store || args.fp_durable_writes) &&
        args.fp_append_store_path.empty()) {
        throw std::runtime_error(
                "--rebuild-fp-append-store/--fp-durable-writes require --fp-append-store");
    }
    if (args.fp_sort_iterations < 0 || args.fp_sort_partition_size <= 0 ||
        args.fp_sort_partition_size > 64 || args.concurrent_search_ef <= 0) {
        throw std::runtime_error("invalid FP sorting configuration");
    }
    std::sort(args.insert_sizes.begin(), args.insert_sizes.end());
    args.insert_sizes.erase(
            std::unique(args.insert_sizes.begin(), args.insert_sizes.end()),
            args.insert_sizes.end());
    std::sort(args.efs.begin(), args.efs.end());
    args.efs.erase(std::unique(args.efs.begin(), args.efs.end()), args.efs.end());
    if (!args.calibration_query.empty() && args.overlap_queries &&
        args.io_queue_depth < args.efs.back()) {
        throw std::runtime_error(
                "per-ef calibrated overlap requires --io-queue-depth >= max efSearch");
    }
    return args;
}

std::vector<int64_t> dynamic_read_i64bin(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!in) throw std::runtime_error("bad i64bin header: " + path);
    std::vector<int64_t> ids(static_cast<size_t>(n));
    in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(int64_t)));
    if (!in) throw std::runtime_error("bad i64bin payload: " + path);
    return ids;
}

struct DynamicRowSource {
    virtual ~DynamicRowSource() = default;
    virtual int64_t rows() const = 0;
    virtual int dimension() const = 0;
    virtual const float* row(size_t id) const = 0;
};

struct DynamicFbinRowSource final : DynamicRowSource {
    const FbinMmap& source;
    explicit DynamicFbinRowSource(const FbinMmap& source_in) : source(source_in) {}
    int64_t rows() const override { return source.n; }
    int dimension() const override { return source.d; }
    const float* row(size_t id) const override { return source.row(id); }
};

struct DynamicNpyMmap {
    int fd = -1;
    void* mapping = MAP_FAILED;
    size_t mapping_bytes = 0;
    const uint8_t* payload = nullptr;
    std::vector<size_t> shape;
    std::string descr;

    explicit DynamicNpyMmap(const fs::path& path) {
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("open npy failed: " + path.string());
        struct stat st {};
        if (::fstat(fd, &st) != 0 || st.st_size < 16) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("invalid npy file: " + path.string());
        }
        mapping_bytes = static_cast<size_t>(st.st_size);
        mapping = ::mmap(nullptr, mapping_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapping == MAP_FAILED) {
            ::close(fd);
            fd = -1;
            throw std::runtime_error("mmap npy failed: " + path.string());
        }
        const auto* bytes = static_cast<const uint8_t*>(mapping);
        if (std::memcmp(bytes, "\x93NUMPY", 6) != 0) {
            throw std::runtime_error("bad npy magic: " + path.string());
        }
        const uint8_t major = bytes[6];
        size_t header_len = 0;
        size_t header_offset = 0;
        if (major == 1) {
            uint16_t value = 0;
            std::memcpy(&value, bytes + 8, sizeof(value));
            header_len = value;
            header_offset = 10;
        } else if (major == 2 || major == 3) {
            uint32_t value = 0;
            std::memcpy(&value, bytes + 8, sizeof(value));
            header_len = value;
            header_offset = 12;
        } else {
            throw std::runtime_error("unsupported npy version: " + path.string());
        }
        if (header_offset + header_len > mapping_bytes) {
            throw std::runtime_error("truncated npy header: " + path.string());
        }
        const std::string header(
                reinterpret_cast<const char*>(bytes + header_offset), header_len);
        std::smatch match;
        if (!std::regex_search(
                    header,
                    match,
                    std::regex("['\\\"]descr['\\\"]\\s*:\\s*['\\\"]([^'\\\"]+)['\\\"]"))) {
            throw std::runtime_error("npy descr is missing: " + path.string());
        }
        descr = match[1].str();
        if (!std::regex_search(
                    header,
                    match,
                    std::regex("['\\\"]shape['\\\"]\\s*:\\s*\\(([^)]*)\\)"))) {
            throw std::runtime_error("npy shape is missing: " + path.string());
        }
        const std::string shape_text = match[1].str();
        const std::regex number("[0-9]+");
        for (auto it = std::sregex_iterator(shape_text.begin(), shape_text.end(), number);
             it != std::sregex_iterator();
             ++it) {
            shape.push_back(static_cast<size_t>(std::stoull(it->str())));
        }
        payload = bytes + header_offset + header_len;
    }

    DynamicNpyMmap(const DynamicNpyMmap&) = delete;
    DynamicNpyMmap& operator=(const DynamicNpyMmap&) = delete;

    ~DynamicNpyMmap() {
        if (mapping != MAP_FAILED) ::munmap(mapping, mapping_bytes);
        if (fd >= 0) ::close(fd);
    }
};

struct DynamicChunkedCombinedSource final : DynamicRowSource {
    const FbinMmap& base;
    DynamicNpyMmap permutation;
    std::vector<std::unique_ptr<DynamicNpyMmap>> chunks;
    std::vector<int64_t> chunk_starts;
    std::vector<int64_t> chunk_ends;
    int64_t logical_new_rows = 0;
    int64_t source_rows = 0;

    DynamicChunkedCombinedSource(
            const FbinMmap& base_in,
            const fs::path& emb_root,
            const fs::path& permutation_path,
            int64_t required_new_rows)
            : base(base_in), permutation(permutation_path) {
        if (permutation.descr != "<i8" || permutation.shape.size() != 1 ||
            permutation.shape[0] < static_cast<size_t>(required_new_rows)) {
            throw std::runtime_error("unexpected insertion permutation layout");
        }
        const auto* perm = reinterpret_cast<const int64_t*>(permutation.payload);
        int64_t required_source_rows = 0;
        for (int64_t logical = 0; logical < required_new_rows; ++logical) {
            if (perm[logical] < 0) {
                throw std::runtime_error("negative row in insertion permutation");
            }
            required_source_rows = std::max(required_source_rows, perm[logical] + 1);
        }
        std::vector<fs::path> paths;
        const std::regex chunk_name(".*_c[0-9][0-9][0-9]\\.npy");
        for (const auto& entry : fs::recursive_directory_iterator(emb_root)) {
            if (entry.is_regular_file() &&
                std::regex_match(entry.path().filename().string(), chunk_name)) {
                paths.push_back(entry.path());
            }
        }
        std::sort(paths.begin(), paths.end());
        int64_t total = 0;
        for (const fs::path& path : paths) {
            if (total >= required_source_rows) break;
            auto chunk = std::make_unique<DynamicNpyMmap>(path);
            if ((chunk->descr != "<f4" && chunk->descr != "|f4") ||
                chunk->shape.size() != 2 ||
                chunk->shape[1] != static_cast<size_t>(base.d)) {
                throw std::runtime_error("unexpected embedding chunk layout: " + path.string());
            }
            const int64_t take = std::min<int64_t>(
                    static_cast<int64_t>(chunk->shape[0]), required_source_rows - total);
            chunk_starts.push_back(total);
            chunk_ends.push_back(total + take);
            chunks.push_back(std::move(chunk));
            total += take;
        }
        if (total != required_source_rows) {
            throw std::runtime_error(
                    "embedding chunks provide " + std::to_string(total) +
                    " source rows; need " + std::to_string(required_source_rows));
        }
        logical_new_rows = required_new_rows;
        source_rows = total;
    }

    int64_t rows() const override { return base.n + logical_new_rows; }
    int dimension() const override { return base.d; }

    const float* row(size_t id) const override {
        if (id < static_cast<size_t>(base.n)) return base.row(id);
        const size_t logical_new = id - static_cast<size_t>(base.n);
        if (logical_new >= static_cast<size_t>(logical_new_rows)) {
            throw std::out_of_range("combined source row out of range");
        }
        const auto* perm = reinterpret_cast<const int64_t*>(permutation.payload);
        const int64_t source_id = perm[logical_new];
        if (source_id < 0 || source_id >= source_rows) {
            throw std::runtime_error("permutation row out of range");
        }
        const auto found = std::upper_bound(
                chunk_ends.begin(), chunk_ends.end(), source_id);
        const size_t chunk_id = static_cast<size_t>(found - chunk_ends.begin());
        if (chunk_id >= chunks.size()) {
            throw std::runtime_error("failed to resolve insertion source chunk");
        }
        const size_t local = static_cast<size_t>(source_id - chunk_starts[chunk_id]);
        const auto* values = reinterpret_cast<const float*>(chunks[chunk_id]->payload);
        return values + local * static_cast<size_t>(base.d);
    }
};

void dynamic_pwrite_all(int fd, const void* data, size_t bytes, off_t offset) {
    const char* src = static_cast<const char*>(data);
    size_t written = 0;
    while (written < bytes) {
        const ssize_t rc = ::pwrite(
                fd,
                src + written,
                bytes - written,
                offset + static_cast<off_t>(written));
        if (rc < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("pwrite failed: ") + std::strerror(errno));
        }
        if (rc == 0) throw std::runtime_error("pwrite made no progress");
        written += static_cast<size_t>(rc);
    }
}

void dynamic_sync_fd(int fd, const char* description) {
    if (::fdatasync(fd) != 0) {
        throw std::runtime_error(
                std::string("fdatasync failed for ") + description + ": " +
                std::strerror(errno));
    }
}

std::vector<uint32_t> dynamic_identity_map(size_t rows) {
    if (rows > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::runtime_error("dynamic FP store exceeds UINT32_MAX rows");
    }
    std::vector<uint32_t> out(rows);
    std::iota(out.begin(), out.end(), uint32_t{0});
    return out;
}

class DynamicFpAppendStore {
public:
    DynamicFpAppendStore(
            std::string path,
            const FbinMmap& base,
            int64_t capacity_rows,
            bool rebuild,
            bool durable)
            : path_(std::move(path)),
              dim_(base.d),
              capacity_rows_(capacity_rows),
              base_rows_(base.n),
              current_rows_(base.n),
              row_bytes_(static_cast<size_t>(base.d) * sizeof(float)),
              durable_(durable) {
        if (path_.empty()) throw std::runtime_error("dynamic FP append path is empty");
        if (capacity_rows_ < base_rows_ || capacity_rows_ <= 0 ||
            capacity_rows_ > std::numeric_limits<int32_t>::max()) {
            throw std::runtime_error("invalid dynamic FP append capacity");
        }
        if (fs::path(path_) == fs::path(base.path)) {
            throw std::runtime_error("dynamic FP append store must differ from the base input");
        }
        if (rebuild || !validate_aligned_fbin(
                               path_,
                               static_cast<int32_t>(capacity_rows_),
                               dim_)) {
            build_base_store(base);
        } else {
            std::cout << "Reusing FP append store base image: " << path_ << std::endl;
        }
        open_writer();
    }

    DynamicFpAppendStore(const DynamicFpAppendStore&) = delete;
    DynamicFpAppendStore& operator=(const DynamicFpAppendStore&) = delete;

    ~DynamicFpAppendStore() {
        if (fd_ >= 0) ::close(fd_);
    }

    const std::string& path() const { return path_; }
    int64_t base_rows() const { return base_rows_; }
    int64_t current_rows() const { return current_rows_; }
    int64_t capacity_rows() const { return capacity_rows_; }

    void reset_to_base() {
        current_rows_ = base_rows_;
    }

    std::vector<uint32_t> physical_map() const {
        return dynamic_identity_map(static_cast<size_t>(current_rows_));
    }

    double append_selected(
            const FbinMmap& source,
            const std::vector<int64_t>& ids,
            size_t first,
            size_t last,
            int batch_size,
            int threads) {
        if (source.d != dim_) throw std::runtime_error("FP append source dimension mismatch");
        if (first > last || last > ids.size()) {
            throw std::runtime_error("FP append selected range is out of bounds");
        }
        const size_t expected_first = static_cast<size_t>(current_rows_ - base_rows_);
        if (first != expected_first) {
            throw std::runtime_error(
                    "FP append range is not contiguous with the published prefix");
        }
        const int64_t target_rows = base_rows_ + static_cast<int64_t>(last);
        if (target_rows > capacity_rows_) {
            throw std::runtime_error("FP append would exceed reserved store capacity");
        }

        const auto start_time = std::chrono::steady_clock::now();
        omp_set_num_threads(threads);
        for (size_t start = first; start < last; start += static_cast<size_t>(batch_size)) {
            const size_t take = std::min(last - start, static_cast<size_t>(batch_size));
            std::vector<float> block(take * static_cast<size_t>(dim_));
#pragma omp parallel for schedule(static)
            for (int64_t local = 0; local < static_cast<int64_t>(take); ++local) {
                const int64_t source_id = ids[start + static_cast<size_t>(local)];
                if (source_id < 0 || source_id >= source.n) {
                    continue;
                }
                std::memcpy(
                        block.data() + static_cast<size_t>(local) * dim_,
                        source.row(static_cast<size_t>(source_id)),
                        row_bytes_);
            }
            const int64_t logical_first = base_rows_ + static_cast<int64_t>(start);
            dynamic_pwrite_all(
                    fd_,
                    block.data(),
                    block.size() * sizeof(float),
                    static_cast<off_t>(
                            kFpDataAlignment +
                            static_cast<uint64_t>(logical_first) * row_bytes_));
            std::cout << "  appended FP rows " << (start + take - first) << "/"
                      << (last - first) << std::endl;
        }
        if (durable_) dynamic_sync_fd(fd_, "FP append store");
        current_rows_ = target_rows;
        return seconds_since(start_time);
    }

    double write_compacted(
            const std::string& output_path,
            const std::vector<uint32_t>& physical_row_by_label) const {
        if (output_path.empty()) throw std::runtime_error("compacted FP output path is empty");
        if (fs::path(output_path) == fs::path(path_)) {
            throw std::runtime_error("compacted FP output must differ from append store");
        }
        if (physical_row_by_label.size() != static_cast<size_t>(current_rows_)) {
            throw std::runtime_error("FP compaction map size mismatch");
        }

        std::vector<uint32_t> logical_by_physical(physical_row_by_label.size());
        std::vector<uint8_t> seen(physical_row_by_label.size(), 0);
        for (uint32_t logical = 0;
             logical < static_cast<uint32_t>(physical_row_by_label.size());
             ++logical) {
            const uint32_t physical = physical_row_by_label[logical];
            if (physical >= physical_row_by_label.size() || seen[physical]) {
                throw std::runtime_error("FP compaction map is not a permutation");
            }
            seen[physical] = 1;
            logical_by_physical[physical] = logical;
        }

        const fs::path output(output_path);
        if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
        const std::string tmp_path = output_path + ".tmp." +
                std::to_string(static_cast<long long>(::getpid()));
        const auto start_time = std::chrono::steady_clock::now();
        FbinMmap source(path_);
        std::vector<char> stream_buffer(8 * 1024 * 1024);
        std::ofstream out;
        out.rdbuf()->pubsetbuf(
                stream_buffer.data(),
                static_cast<std::streamsize>(stream_buffer.size()));
        out.open(tmp_path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to create compacted FP store: " + tmp_path);

        const int32_t rows = static_cast<int32_t>(current_rows_);
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
        const std::array<char, kFpDataAlignment - kLegacyFbinHeaderBytes> padding{};
        out.write(padding.data(), static_cast<std::streamsize>(padding.size()));
        for (size_t physical = 0; physical < logical_by_physical.size(); ++physical) {
            const uint32_t logical = logical_by_physical[physical];
            out.write(
                    reinterpret_cast<const char*>(source.row(logical)),
                    static_cast<std::streamsize>(row_bytes_));
            if ((physical + 1) % 100000 == 0 || physical + 1 == logical_by_physical.size()) {
                std::cout << "  compacted FP rows " << (physical + 1) << "/"
                          << logical_by_physical.size() << std::endl;
            }
        }
        out.close();
        if (!out) throw std::runtime_error("failed to write compacted FP store: " + tmp_path);
        if (durable_) {
            const int sync_fd = ::open(tmp_path.c_str(), O_RDWR);
            if (sync_fd < 0) {
                throw std::runtime_error("failed to reopen compacted FP store for sync");
            }
            try {
                dynamic_sync_fd(sync_fd, "compacted FP store");
            } catch (...) {
                ::close(sync_fd);
                throw;
            }
            ::close(sync_fd);
        }
        std::error_code remove_ec;
        fs::remove(output, remove_ec);
        fs::rename(tmp_path, output);
        return seconds_since(start_time);
    }

private:
    std::string path_;
    int32_t dim_ = 0;
    int64_t capacity_rows_ = 0;
    int64_t base_rows_ = 0;
    int64_t current_rows_ = 0;
    size_t row_bytes_ = 0;
    bool durable_ = false;
    int fd_ = -1;

    void open_writer() {
        fd_ = ::open(path_.c_str(), O_RDWR);
        if (fd_ < 0) {
            throw std::runtime_error(
                    std::string("failed to open FP append store: ") + std::strerror(errno));
        }
    }

    void build_base_store(const FbinMmap& base) {
        const fs::path output(path_);
        if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
        const std::string tmp_path = path_ + ".tmp." +
                std::to_string(static_cast<long long>(::getpid()));
        const int fd = ::open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) throw std::runtime_error("failed to create FP append store: " + tmp_path);
        try {
            std::array<char, kFpDataAlignment> header{};
            const int32_t capacity32 = static_cast<int32_t>(capacity_rows_);
            std::memcpy(header.data(), &capacity32, sizeof(capacity32));
            std::memcpy(header.data() + sizeof(capacity32), &dim_, sizeof(dim_));
            dynamic_pwrite_all(fd, header.data(), header.size(), 0);
            const uint64_t file_bytes = kFpDataAlignment +
                    static_cast<uint64_t>(capacity_rows_) * row_bytes_;
            if (::ftruncate(fd, static_cast<off_t>(file_bytes)) != 0) {
                throw std::runtime_error(std::string("ftruncate failed: ") + std::strerror(errno));
            }

            constexpr size_t target_chunk_bytes = 16 * 1024 * 1024;
            const size_t rows_per_chunk = std::max<size_t>(1, target_chunk_bytes / row_bytes_);
            for (size_t first = 0; first < static_cast<size_t>(base_rows_); first += rows_per_chunk) {
                const size_t count = std::min(
                        rows_per_chunk,
                        static_cast<size_t>(base_rows_) - first);
                dynamic_pwrite_all(
                        fd,
                        base.row(first),
                        count * row_bytes_,
                        static_cast<off_t>(
                                kFpDataAlignment +
                                static_cast<uint64_t>(first) * row_bytes_));
                if (first + count == static_cast<size_t>(base_rows_) ||
                    (first + count) % 100000 == 0) {
                    std::cout << "  initialized FP base rows " << (first + count) << "/"
                              << base_rows_ << std::endl;
                }
            }
            if (durable_) dynamic_sync_fd(fd, "FP append base image");
        } catch (...) {
            ::close(fd);
            std::error_code ec;
            fs::remove(tmp_path, ec);
            throw;
        }
        ::close(fd);
        std::error_code remove_ec;
        fs::remove(output, remove_ec);
        fs::rename(tmp_path, output);
        std::cout << "Built FP append store with capacity=" << capacity_rows_
                  << ", base_rows=" << base_rows_ << " -> " << path_ << std::endl;
    }
};

std::string dynamic_stage_gt_path(const DynamicArgs& args, int insert_size) {
    fs::path p(args.gt_dir);
    p /= "qwen1024_1m_plus_" + std::to_string(insert_size) + "_" + args.policy
            + "_gt_k" + std::to_string(args.k) + "_q" + std::to_string(args.max_queries)
            + ".ibin";
    return p.string();
}

void dynamic_pack_sign_bits(
        const float* x,
        size_t n,
        int dim,
        std::vector<uint8_t>& codes,
        int threads) {
    adcbq::validate_sign_packing_dimension(dim);
    const size_t code_size = static_cast<size_t>(dim) / 8;
    codes.resize(n * code_size);
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int64_t row = 0; row < static_cast<int64_t>(n); ++row) {
        const float* src = x + static_cast<size_t>(row) * dim;
        uint8_t* dst = codes.data() + static_cast<size_t>(row) * code_size;
        adcbq::pack_sign_bits_row(src, dim, dst);
    }
}

struct QdrantTqIndexDeleter {
    void operator()(QdrantTqIndex* index) const { qdrant_tq_free(index); }
};

using QdrantTqIndexPtr = std::unique_ptr<QdrantTqIndex, QdrantTqIndexDeleter>;

std::string dynamic_tq_error() {
    const char* message = qdrant_tq_last_error();
    return message && *message ? message : "unknown Qdrant TurboQuant error";
}

float dynamic_tq_read_float(const uint8_t* source) {
    float value = 0.0f;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

// Qdrant's canonical TQ1+ encoder remains in Rust. Traversal uses the same
// exported query planes as the static benchmark so the per-candidate hot path
// stays in C++ and does not cross the FFI boundary.
struct DynamicTqDistanceComputer : faiss::DistanceComputer {
    const QdrantTqIndex* index = nullptr;
    const uint8_t* codes = nullptr;
    size_t code_size = 0;
    size_t code_bytes = 0;
    size_t padded_dim = 0;
    size_t query_plane_len = 0;
    int d = 0;
    QdrantTqQuery* query = nullptr;
    std::vector<int8_t> query_planes;
    float postprocess_scale = 0.0f;
    int64_t bias_correction = 0;
    float ec_correction = 0.0f;
    float query_l2 = 0.0f;

    DynamicTqDistanceComputer(
            const QdrantTqIndex* index_in,
            const uint8_t* codes_in,
            size_t code_size_in,
            int d_in)
            : index(index_in),
              codes(codes_in),
              code_size(code_size_in),
              code_bytes((static_cast<size_t>(d_in) + 7) / 8),
              padded_dim(code_bytes * 8),
              query_plane_len(((code_bytes + 63) / 64) * 64),
              d(d_in),
              query_planes(2 * 8 * query_plane_len) {
        if (!index || !codes || code_size < code_bytes + sizeof(float)) {
            throw std::runtime_error("invalid resident TQ1+ codes");
        }
    }

    ~DynamicTqDistanceComputer() override { qdrant_tq_query_free(query); }

    void set_query(const float* values) override {
        qdrant_tq_query_free(query);
        query = qdrant_tq_query_new(index, values, static_cast<size_t>(d));
        if (!query) throw std::runtime_error(dynamic_tq_error());
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
        const float vector_scale = dynamic_tq_read_float(code + code_bytes);
        return -(dot * vector_scale);
    }

    float operator()(faiss::idx_t i) override {
        return distance_to_code(codes + static_cast<size_t>(i) * code_size);
    }

    void distances_batch_4(
            faiss::idx_t idx0,
            faiss::idx_t idx1,
            faiss::idx_t idx2,
            faiss::idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        dis0 = (*this)(idx0);
        dis1 = (*this)(idx1);
        dis2 = (*this)(idx2);
        dis3 = (*this)(idx3);
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
        const float raw = centroid_sq * static_cast<float>(
                static_cast<int64_t>(padded_dim) - 2LL * hamming);
        return -(raw * dynamic_tq_read_float(left + code_bytes) *
                 dynamic_tq_read_float(right + code_bytes));
    }
};

void dynamic_search_tq(
        const faiss::IndexBinaryHNSW& index,
        const QdrantTqIndex* tq,
        const uint8_t* codes,
        size_t code_size,
        const float* queries,
        int nq,
        int dim,
        int k,
        int ef,
        int threads,
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
        DynamicTqDistanceComputer computer(tq, codes, code_size, dim);
        RH::SingleResultHandler result(block_result);
#pragma omp for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            result.begin(qi);
            computer.set_query(queries + static_cast<size_t>(qi) * dim);
            index.hnsw.search(computer, nullptr, result, visited, &params);
            result.end();
        }
    }
}

// HNSW construction compares query-to-candidate distances with candidate-to-
// candidate distances while diversifying a neighbor list.  Keep both in the
// same normalized inner-product space:
//
//   operator()(i)       = -<normalize(fp_query), sign(code_i) / sqrt(dim)>
//   symmetric_dis(i,j)  = -<sign(code_i), sign(code_j)> / dim
//
// The first expression is the asymmetric ADC-BQ distance requested for online
// insertion.  The second one is its BQ-only symmetric proxy, which avoids
// fetching FP payloads during reciprocal-link pruning.
struct DynamicAdcBqConstructionDistanceComputer : faiss::DistanceComputer {
    PackedAsymBQDistanceComputer adc;
    int dim = 0;
    float inv_sqrt_dim = 0.0f;
    float inv_dim = 0.0f;
    std::vector<float> normalized_query;

    DynamicAdcBqConstructionDistanceComputer(
            const uint8_t* codes,
            size_t code_size,
            size_t code_stride,
            int dim_in)
            : adc(
                      codes,
                      code_size,
                      code_stride,
                      dim_in,
                      AdcMode::Fp32Direct,
                      LutMode::Nibble,
                      false),
              dim(dim_in),
              inv_sqrt_dim(
                      dim_in > 0 ? 1.0f / std::sqrt(static_cast<float>(dim_in)) : 0.0f),
              inv_dim(dim_in > 0 ? 1.0f / static_cast<float>(dim_in) : 0.0f),
              normalized_query(static_cast<size_t>(dim_in)) {
        if (dim_in <= 0) throw std::runtime_error("ADC-BQ construction requires dim > 0");
    }

    void set_query(const float* query) override {
        double squared_norm = 0.0;
        for (int j = 0; j < dim; ++j) {
            squared_norm += static_cast<double>(query[j]) * query[j];
        }
        const float inv_norm = squared_norm > 0.0
                ? 1.0f / static_cast<float>(std::sqrt(squared_norm))
                : 0.0f;
        for (int j = 0; j < dim; ++j) {
            normalized_query[static_cast<size_t>(j)] = query[j] * inv_norm;
        }
        adc.set_query(normalized_query.data());
    }

    float operator()(faiss::idx_t i) override {
        return adc(i) * inv_sqrt_dim;
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
        adc.distances_batch_4(idx0, idx1, idx2, idx3, dis0, dis1, dis2, dis3);
        dis0 *= inv_sqrt_dim;
        dis1 *= inv_sqrt_dim;
        dis2 *= inv_sqrt_dim;
        dis3 *= inv_sqrt_dim;
    }

    float symmetric_dis(faiss::idx_t i, faiss::idx_t j) override {
        const float hamming = adc.symmetric_dis(i, j);
        return (2.0f * hamming - static_cast<float>(dim)) * inv_dim;
    }
};

void dynamic_hnsw_add_adcbq_vertices(
        faiss::IndexBinaryHNSW& index,
        size_t n0,
        size_t n,
        const float* fp_queries) {
    if (n == 0) return;
    if (!fp_queries) throw std::runtime_error("ADC-BQ add requires FP32 queries");
    if (index.ntotal != static_cast<faiss::idx_t>(n0 + n)) {
        throw std::runtime_error("ADC-BQ storage must be appended before graph construction");
    }

    faiss::HNSW& hnsw = index.hnsw;
    const size_t ntotal = n0 + n;
    hnsw.prepare_level_tab(n, false);
    auto& locks = index.locks;
    locks.prepare(ntotal);

    std::vector<int> histogram;
    std::vector<int> order(n);
    for (size_t local = 0; local < n; ++local) {
        const auto pt_id = static_cast<faiss::HNSW::storage_idx_t>(n0 + local);
        const int pt_level = hnsw.levels[static_cast<size_t>(pt_id)] - 1;
        while (pt_level >= static_cast<int>(histogram.size())) histogram.push_back(0);
        ++histogram[static_cast<size_t>(pt_level)];
    }

    std::vector<int> offsets(histogram.size() + 1, 0);
    for (size_t level = 0; level + 1 < histogram.size(); ++level) {
        offsets[level + 1] = offsets[level] + histogram[level];
    }
    for (size_t local = 0; local < n; ++local) {
        const auto pt_id = static_cast<faiss::HNSW::storage_idx_t>(n0 + local);
        const int pt_level = hnsw.levels[static_cast<size_t>(pt_id)] - 1;
        order[static_cast<size_t>(offsets[static_cast<size_t>(pt_level)]++)] = pt_id;
    }

    const uint8_t* codes = storage_codes(index);
    faiss::RandomGenerator rng(789);
    size_t bucket_end = n;
    for (int pt_level = static_cast<int>(histogram.size()) - 1;
         pt_level >= int(!index.init_level0);
         --pt_level) {
        const size_t bucket_begin =
                bucket_end - static_cast<size_t>(histogram[static_cast<size_t>(pt_level)]);
        for (size_t j = bucket_begin; j < bucket_end; ++j) {
            const int remaining = static_cast<int>(bucket_end - j);
            std::swap(order[j], order[j + static_cast<size_t>(rng.rand_int(remaining))]);
        }

#pragma omp parallel
        {
            faiss::VisitedTable visited(ntotal);
            DynamicAdcBqConstructionDistanceComputer distance(
                    codes, index.code_size, index.code_size, index.d);
#pragma omp for schedule(dynamic)
            for (int64_t pos = static_cast<int64_t>(bucket_begin);
                 pos < static_cast<int64_t>(bucket_end);
                 ++pos) {
                const auto pt_id = order[static_cast<size_t>(pos)];
                const size_t local_id = static_cast<size_t>(pt_id) - n0;
                distance.set_query(fp_queries + local_id * static_cast<size_t>(index.d));
                hnsw.add_with_locks(
                        distance,
                        pt_level,
                        pt_id,
                        locks,
                        visited,
                        index.keep_max_size_level0 && pt_level == 0);
            }
        }
        bucket_end = bucket_begin;
    }

    if (!index.retain_locks) locks.clear();
}

void dynamic_hnsw_add_tq_vertices(
        faiss::IndexBinaryHNSW& index,
        const QdrantTqIndex* tq,
        const uint8_t* tq_codes,
        size_t tq_code_size,
        size_t n0,
        size_t n,
        const float* fp_queries) {
    if (n == 0) return;
    if (!fp_queries || !tq || !tq_codes) {
        throw std::runtime_error("TQ1 add requires FP32 queries and resident TQ codes");
    }
    if (index.ntotal != static_cast<faiss::idx_t>(n0 + n)) {
        throw std::runtime_error("TQ1 graph storage must be appended before construction");
    }

    faiss::HNSW& hnsw = index.hnsw;
    const size_t ntotal = n0 + n;
    hnsw.prepare_level_tab(n, false);
    auto& locks = index.locks;
    locks.prepare(ntotal);

    std::vector<int> histogram;
    std::vector<int> order(n);
    for (size_t local = 0; local < n; ++local) {
        const auto pt_id = static_cast<faiss::HNSW::storage_idx_t>(n0 + local);
        const int pt_level = hnsw.levels[static_cast<size_t>(pt_id)] - 1;
        while (pt_level >= static_cast<int>(histogram.size())) histogram.push_back(0);
        ++histogram[static_cast<size_t>(pt_level)];
    }

    std::vector<int> offsets(histogram.size() + 1, 0);
    for (size_t level = 0; level + 1 < histogram.size(); ++level) {
        offsets[level + 1] = offsets[level] + histogram[level];
    }
    for (size_t local = 0; local < n; ++local) {
        const auto pt_id = static_cast<faiss::HNSW::storage_idx_t>(n0 + local);
        const int pt_level = hnsw.levels[static_cast<size_t>(pt_id)] - 1;
        order[static_cast<size_t>(offsets[static_cast<size_t>(pt_level)]++)] = pt_id;
    }

    faiss::RandomGenerator rng(789);
    size_t bucket_end = n;
    for (int pt_level = static_cast<int>(histogram.size()) - 1;
         pt_level >= int(!index.init_level0);
         --pt_level) {
        const size_t bucket_begin =
                bucket_end - static_cast<size_t>(histogram[static_cast<size_t>(pt_level)]);
        for (size_t j = bucket_begin; j < bucket_end; ++j) {
            const int remaining = static_cast<int>(bucket_end - j);
            std::swap(order[j], order[j + static_cast<size_t>(rng.rand_int(remaining))]);
        }

#pragma omp parallel
        {
            faiss::VisitedTable visited(ntotal);
            DynamicTqDistanceComputer distance(tq, tq_codes, tq_code_size, index.d);
#pragma omp for schedule(dynamic)
            for (int64_t pos = static_cast<int64_t>(bucket_begin);
                 pos < static_cast<int64_t>(bucket_end);
                 ++pos) {
                const auto pt_id = order[static_cast<size_t>(pos)];
                const size_t local_id = static_cast<size_t>(pt_id) - n0;
                distance.set_query(fp_queries + local_id * static_cast<size_t>(index.d));
                hnsw.add_with_locks(
                        distance,
                        pt_level,
                        pt_id,
                        locks,
                        visited,
                        index.keep_max_size_level0 && pt_level == 0);
            }
        }
        bucket_end = bucket_begin;
    }

    if (!index.retain_locks) locks.clear();
}

struct DynamicBuildPhaseTiming {
    double input_copy_seconds = 0.0;
    double normalize_seconds = 0.0;
    double bq_encode_seconds = 0.0;
    double graph_add_seconds = 0.0;
};

struct DynamicAdcIndex {
    std::unique_ptr<faiss::IndexHNSWFlat> fp_graph;
    std::unique_ptr<faiss::IndexBinaryHNSW> bq_view;
    QdrantTqIndexPtr tq_index;
    int dim = 0;

    DynamicAdcIndex(int dim_in, int m, int ef_construction)
            : fp_graph(std::make_unique<faiss::IndexHNSWFlat>(
                      dim_in, m, faiss::METRIC_INNER_PRODUCT)),
              dim(dim_in) {
        fp_graph->hnsw.efConstruction = ef_construction;
        fp_graph->verbose = false;
        auto* storage = new faiss::IndexBinaryFlat(dim_in);
        bq_view = std::make_unique<faiss::IndexBinaryHNSW>(storage, m);
        bq_view->own_fields = true;
        bq_view->hnsw.efConstruction = ef_construction;
    }

    DynamicAdcIndex(const DynamicAdcIndex& other) : dim(other.dim) {
        if (other.fp_graph) {
            std::unique_ptr<faiss::Index> fp_clone(faiss::clone_index(other.fp_graph.get()));
            auto* fp_typed = dynamic_cast<faiss::IndexHNSWFlat*>(fp_clone.release());
            if (!fp_typed) throw std::runtime_error("failed to clone FP HNSW base snapshot");
            fp_graph.reset(fp_typed);
        }

        std::unique_ptr<faiss::IndexBinary> bq_clone(
                faiss::clone_binary_index(other.bq_view.get()));
        auto* bq_typed = dynamic_cast<faiss::IndexBinaryHNSW*>(bq_clone.release());
        if (!bq_typed) throw std::runtime_error("failed to clone BQ base snapshot");
        bq_view.reset(bq_typed);
    }

    faiss::IndexBinaryFlat& bq_storage() {
        auto* storage = dynamic_cast<faiss::IndexBinaryFlat*>(bq_view->storage);
        if (!storage) throw std::runtime_error("ADC-BQ storage is not IndexBinaryFlat");
        return *storage;
    }

    void load_tq(const std::string& code_path, const std::string& meta_path) {
        QdrantTqIndex* raw = qdrant_tq_load(code_path.c_str(), meta_path.c_str());
        if (!raw) throw std::runtime_error(dynamic_tq_error());
        tq_index.reset(raw);
        if (tq_rows() > static_cast<size_t>(bq_view->ntotal)) {
            throw std::runtime_error("loaded TQ row count exceeds graph size");
        }
    }

    bool has_tq() const { return static_cast<bool>(tq_index); }

    size_t tq_code_size() const {
        if (!tq_index) throw std::runtime_error("TQ index is unavailable");
        return qdrant_tq_code_size(tq_index.get());
    }

    const uint8_t* tq_codes(size_t* byte_len = nullptr) const {
        if (!tq_index) throw std::runtime_error("TQ index is unavailable");
        size_t bytes = 0;
        const uint8_t* result = qdrant_tq_codes(tq_index.get(), &bytes);
        if (!result) throw std::runtime_error(dynamic_tq_error());
        if (byte_len) *byte_len = bytes;
        return result;
    }

    size_t tq_rows() const {
        size_t bytes = 0;
        tq_codes(&bytes);
        const size_t stride = tq_code_size();
        if (stride == 0 || bytes % stride != 0) {
            throw std::runtime_error("invalid TQ code storage size");
        }
        return bytes / stride;
    }

    void append_tq_block(const std::vector<float>& block, int threads) {
        if (block.empty()) return;
        if (!tq_index) throw std::runtime_error("TQ append requires a loaded base codec");
        if (block.size() % static_cast<size_t>(dim) != 0) {
            throw std::runtime_error("TQ append block shape mismatch");
        }
        const size_t old_rows = tq_rows();
        const size_t rows = block.size() / static_cast<size_t>(dim);
        if (!qdrant_tq_append(
                    tq_index.get(),
                    block.data(),
                    rows,
                    static_cast<size_t>(dim),
                    static_cast<size_t>(threads))) {
            throw std::runtime_error(dynamic_tq_error());
        }
        if (tq_rows() != old_rows + rows) {
            throw std::runtime_error("TQ append row count mismatch");
        }
    }

    bool has_fp_graph() const {
        return static_cast<bool>(fp_graph);
    }

    void seal_fp_graph_for_compressed_updates() {
        if (!fp_graph) return;
        if (fp_graph->ntotal != bq_view->ntotal) {
            throw std::runtime_error("cannot seal divergent FP/BQ indexes");
        }
        std::swap(fp_graph->hnsw, bq_view->hnsw);
        fp_graph.reset();
    }

    void add_normalized_block(
            std::vector<float>& block,
            int threads,
            DynamicBuildPhaseTiming* timing = nullptr) {
        if (block.empty()) return;
        if (!fp_graph) {
            throw std::runtime_error("FP graph add requested after switching to BQ-only updates");
        }
        const size_t rows = block.size() / static_cast<size_t>(dim);
        omp_set_num_threads(threads);
        const auto normalize_start = std::chrono::steady_clock::now();
        normalize_rows_inplace(block.data(), static_cast<int64_t>(rows), dim);
        if (timing) timing->normalize_seconds += seconds_since(normalize_start);
        std::vector<uint8_t> codes;
        const auto encode_start = std::chrono::steady_clock::now();
        dynamic_pack_sign_bits(block.data(), rows, dim, codes, threads);
        if (timing) timing->bq_encode_seconds += seconds_since(encode_start);
        const auto graph_start = std::chrono::steady_clock::now();
        fp_graph->add(static_cast<faiss::idx_t>(rows), block.data());
        bq_storage().add(static_cast<faiss::idx_t>(rows), codes.data());
        if (timing) timing->graph_add_seconds += seconds_since(graph_start);
        bq_view->ntotal = bq_storage().ntotal;
        if (fp_graph->ntotal != bq_view->ntotal) {
            throw std::runtime_error("FP graph and BQ storage ntotal diverged");
        }
    }

    void add_bq_block(const std::vector<float>& block, int threads) {
        if (block.empty()) return;
        const size_t rows = block.size() / static_cast<size_t>(dim);
        std::vector<uint8_t> codes;
        dynamic_pack_sign_bits(block.data(), rows, dim, codes, threads);
        add_bq_codes(codes.data(), rows, threads);
    }

    void add_bq_codes(const uint8_t* codes, size_t rows, int threads) {
        if (rows == 0) return;
        if (fp_graph) {
            throw std::runtime_error("BQ-only add requires a sealed FP base graph");
        }
        omp_set_num_threads(threads);
        bq_view->add(static_cast<faiss::idx_t>(rows), codes);
        if (bq_view->ntotal != bq_storage().ntotal) {
            throw std::runtime_error("BQ graph and storage ntotal diverged");
        }
    }

    void add_adcbq_block(const std::vector<float>& block, int threads) {
        if (block.empty()) return;
        const size_t rows = block.size() / static_cast<size_t>(dim);
        std::vector<uint8_t> codes;
        dynamic_pack_sign_bits(block.data(), rows, dim, codes, threads);
        add_adcbq_codes(block.data(), codes.data(), rows, threads);
    }

    void add_adcbq_codes(
            const float* fp_queries,
            const uint8_t* codes,
            size_t rows,
            int threads) {
        if (rows == 0) return;
        if (fp_graph) {
            throw std::runtime_error("ADC-BQ add requires a sealed FP base graph");
        }
        const size_t n0 = static_cast<size_t>(bq_view->ntotal);
        omp_set_num_threads(threads);
        bq_storage().add(static_cast<faiss::idx_t>(rows), codes);
        bq_view->ntotal = bq_storage().ntotal;
        dynamic_hnsw_add_adcbq_vertices(*bq_view, n0, rows, fp_queries);
        if (bq_view->ntotal != bq_storage().ntotal ||
            bq_view->hnsw.levels.size() != static_cast<size_t>(bq_view->ntotal)) {
            throw std::runtime_error("ADC-BQ graph and storage ntotal diverged");
        }
    }

    void add_tq_codes(
            const float* fp_queries,
            const uint8_t* sign_codes,
            size_t rows,
            int threads) {
        if (rows == 0) return;
        if (fp_graph) {
            throw std::runtime_error("TQ-only add requires a sealed FP base graph");
        }
        const size_t n0 = static_cast<size_t>(bq_view->ntotal);
        if (tq_rows() != n0 + rows) {
            throw std::runtime_error("TQ codes must be appended before graph publication");
        }
        omp_set_num_threads(threads);
        bq_storage().add(static_cast<faiss::idx_t>(rows), sign_codes);
        bq_view->ntotal = bq_storage().ntotal;
        dynamic_hnsw_add_tq_vertices(
                *bq_view,
                tq_index.get(),
                tq_codes(),
                tq_code_size(),
                n0,
                rows,
                fp_queries);
        if (bq_view->ntotal != bq_storage().ntotal ||
            bq_view->hnsw.levels.size() != static_cast<size_t>(bq_view->ntotal)) {
            throw std::runtime_error("TQ graph and storage ntotal diverged");
        }
    }
};

void dynamic_add_contiguous(
        DynamicAdcIndex& index,
        const FbinMmap& source,
        int64_t rows,
        int batch_size,
        int threads,
        const char* label,
        DynamicBuildPhaseTiming* timing = nullptr) {
    for (int64_t start = 0; start < rows; start += batch_size) {
        const int64_t take = std::min<int64_t>(batch_size, rows - start);
        const auto copy_start = std::chrono::steady_clock::now();
        std::vector<float> block(static_cast<size_t>(take) * source.d);
        std::memcpy(block.data(), source.row(static_cast<size_t>(start)), block.size() * sizeof(float));
        if (timing) timing->input_copy_seconds += seconds_since(copy_start);
        index.add_normalized_block(block, threads, timing);
        std::cout << "  " << label << " " << (start + take) << "/" << rows << std::endl;
    }
}

void dynamic_add_selected(
        DynamicAdcIndex& index,
        const DynamicRowSource& source,
        const std::vector<int64_t>& ids,
        size_t first,
        size_t last,
        int batch_size,
        int threads,
        DynamicGraphAddMode graph_add_mode,
        const char* label,
        DynamicBuildPhaseTiming* timing = nullptr) {
    if (last > ids.size() || first > last) {
        throw std::runtime_error("selected row range is out of bounds");
    }
    omp_set_num_threads(threads);
    for (size_t start = first; start < last; start += static_cast<size_t>(batch_size)) {
        const size_t take = std::min(last - start, static_cast<size_t>(batch_size));
        const auto copy_start = std::chrono::steady_clock::now();
        const int dim = source.dimension();
        std::vector<float> block(take * static_cast<size_t>(dim));
#pragma omp parallel for schedule(static)
        for (int64_t local = 0; local < static_cast<int64_t>(take); ++local) {
            const int64_t source_id = ids[start + static_cast<size_t>(local)];
            if (source_id < 0 || source_id >= source.rows()) continue;
            std::memcpy(
                    block.data() + static_cast<size_t>(local) * dim,
                    source.row(static_cast<size_t>(source_id)),
                    static_cast<size_t>(dim) * sizeof(float));
        }
        if (timing) timing->input_copy_seconds += seconds_since(copy_start);
        if (graph_add_mode == DynamicGraphAddMode::BqHamming) {
            index.add_bq_block(block, threads);
        } else if (graph_add_mode == DynamicGraphAddMode::AdcBq) {
            index.add_adcbq_block(block, threads);
        } else {
            index.add_normalized_block(block, threads, timing);
        }
        std::cout << "  " << label << " " << (start + take - first) << "/" << (last - first)
                  << std::endl;
    }
}

std::vector<uint8_t> dynamic_pack_selected_sign_bits(
        const DynamicRowSource& source,
        const std::vector<int64_t>& ids,
        size_t first,
        size_t last,
        int threads) {
    if (first > last || last > ids.size()) {
        throw std::runtime_error("selected BQ encode range is out of bounds");
    }
    const int dim = source.dimension();
    adcbq::validate_sign_packing_dimension(dim);
    const size_t rows = last - first;
    const size_t code_size = static_cast<size_t>(dim) / 8;
    std::vector<uint8_t> codes(rows * code_size);
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int64_t local = 0; local < static_cast<int64_t>(rows); ++local) {
        const int64_t source_id = ids[first + static_cast<size_t>(local)];
        const float* src = source.row(static_cast<size_t>(source_id));
        uint8_t* dst = codes.data() + static_cast<size_t>(local) * code_size;
        adcbq::pack_sign_bits_row(src, dim, dst);
    }
    return codes;
}

std::vector<float> dynamic_copy_selected_fp_queries(
        const DynamicRowSource& source,
        const std::vector<int64_t>& ids,
        size_t first,
        size_t last,
        int threads) {
    if (first > last || last > ids.size()) {
        throw std::runtime_error("selected ADC query range is out of bounds");
    }
    const int dim = source.dimension();
    const size_t rows = last - first;
    std::vector<float> queries(rows * static_cast<size_t>(dim));
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int64_t local = 0; local < static_cast<int64_t>(rows); ++local) {
        const int64_t source_id = ids[first + static_cast<size_t>(local)];
        if (source_id < 0 || source_id >= source.rows()) {
            continue;
        }
        std::memcpy(
                queries.data() + static_cast<size_t>(local) * dim,
                source.row(static_cast<size_t>(source_id)),
                static_cast<size_t>(dim) * sizeof(float));
    }
    return queries;
}

std::unique_ptr<DynamicAdcIndex> dynamic_build_stage(
        const DynamicArgs& args,
        const FbinMmap& base,
        const DynamicRowSource& larger,
        const std::vector<int64_t>& selected,
        int insert_size,
        double& seconds,
        DynamicBuildPhaseTiming& phase_timing) {
    auto index = std::make_unique<DynamicAdcIndex>(base.d, args.m, args.ef_construction);
    const auto start = std::chrono::steady_clock::now();
    dynamic_add_contiguous(
            *index,
            base,
            base.n,
            args.add_batch_size,
            args.build_threads,
            "rebuilt base rows",
            &phase_timing);
    dynamic_add_selected(
            *index,
            larger,
            selected,
            0,
            static_cast<size_t>(insert_size),
            args.add_batch_size,
            args.build_threads,
            DynamicGraphAddMode::Fp,
            "rebuilt insert rows",
            &phase_timing);
    seconds = seconds_since(start);
    return index;
}

struct DynamicGraphSwapGuard {
    DynamicAdcIndex& index;
    bool swapped = false;
    explicit DynamicGraphSwapGuard(DynamicAdcIndex& index_in) : index(index_in) {
        if (index.fp_graph) {
            std::swap(index.fp_graph->hnsw, index.bq_view->hnsw);
            swapped = true;
        }
    }
    ~DynamicGraphSwapGuard() {
        if (swapped) std::swap(index.fp_graph->hnsw, index.bq_view->hnsw);
    }
};

struct DynamicTiming {
    double add_seconds = 0.0;
    double input_copy_seconds = 0.0;
    double normalize_seconds = 0.0;
    double bq_encode_seconds = 0.0;
    double graph_add_seconds = 0.0;
    double fp_append_seconds = 0.0;
    double fp_compaction_seconds = 0.0;
    double publish_lock_wait_seconds = 0.0;
    double publish_critical_seconds = 0.0;
    double publish_critical_max_seconds = 0.0;
    uint64_t publish_batches = 0;
    uint64_t concurrent_search_queries = 0;
    double concurrent_search_wall_seconds = 0.0;
    double concurrent_search_latency_ms = 0.0;
    double concurrent_search_lock_wait_ms = 0.0;
    std::string fp_layout = "source_map";
    std::string graph_add_mode = "fp";
};

void dynamic_accumulate_build_phases(
        DynamicTiming& total,
        const DynamicBuildPhaseTiming& phase) {
    total.input_copy_seconds += phase.input_copy_seconds;
    total.normalize_seconds += phase.normalize_seconds;
    total.bq_encode_seconds += phase.bq_encode_seconds;
    total.graph_add_seconds += phase.graph_add_seconds;
}

struct DynamicPublishedState {
    explicit DynamicPublishedState(
            DynamicAdcIndex& index_in,
            const FbinMmap& fp_store_in,
            std::vector<uint32_t> physical_in)
            : index(index_in),
              fp_store(&fp_store_in),
              physical_row_by_label(std::move(physical_in)) {}

    std::shared_mutex mutex;
    std::atomic<bool> writer_pending{false};
    DynamicAdcIndex& index;
    const FbinMmap* fp_store;
    std::vector<uint32_t> physical_row_by_label;
};

void dynamic_search_one_published(
        const DynamicArgs& args,
        DynamicPublishedState& state,
        const float* query) {
    DynamicGraphSwapGuard graph_guard(state.index);
    const uint8_t* codes = storage_codes(*state.index.bq_view);
    const size_t code_stride = state.index.bq_view->code_size;
    std::vector<float> distances;
    std::vector<faiss::idx_t> candidates;
    search_packed_asym(
            *state.index.bq_view,
            query,
            1,
            state.index.dim,
            args.rerank_n,
            args.concurrent_search_ef,
            1,
            AdcMode::Int16,
            LutMode::Nibble,
            SearchImpl::Inline,
            true,
            true,
            false,
            false,
            16,
            16,
            nullptr,
            codes,
            code_stride,
            nullptr,
            0,
            distances,
            candidates,
            nullptr);
    const std::vector<double> unused_norms;
    FpRerankCacheStats io_stats;
    const std::vector<faiss::idx_t> labels = rerank_cosine(
            *state.fp_store,
            unused_norms,
            query,
            1,
            candidates,
            args.rerank_n,
            args.k,
            RerankDotMode::Fp32,
            FpRerankMetric::Cosine,
            args.fp_source,
            FpRerankNormMode::OnFetch,
            false,
            0,
            0,
            false,
            args.io_queue_depth,
            &state.physical_row_by_label,
            0,
            &io_stats,
            nullptr);
    for (faiss::idx_t label : labels) {
        if (label < 0 || static_cast<size_t>(label) >= state.physical_row_by_label.size()) {
            throw std::runtime_error("concurrent search returned an unpublished label");
        }
    }
}

class DynamicConcurrentSearchWorker {
public:
    DynamicConcurrentSearchWorker(
            bool enabled,
            const DynamicArgs& args,
            DynamicPublishedState& state,
            const std::vector<float>& queries,
            int nq)
            : enabled_(enabled),
              args_(args),
              state_(state),
              queries_(queries),
              nq_(nq) {
        if (enabled_) {
            thread_ = std::thread([this]() { run(); });
            while (!ready_.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
    }

    DynamicConcurrentSearchWorker(const DynamicConcurrentSearchWorker&) = delete;
    DynamicConcurrentSearchWorker& operator=(const DynamicConcurrentSearchWorker&) = delete;

    ~DynamicConcurrentSearchWorker() {
        stop_and_join();
    }

    void finish(DynamicTiming& timing) {
        stop_and_join();
        timing.concurrent_search_queries = queries_completed_;
        timing.concurrent_search_wall_seconds = wall_seconds_;
        timing.concurrent_search_latency_ms = queries_completed_ == 0
                ? 0.0
                : query_seconds_ * 1000.0 / static_cast<double>(queries_completed_);
        timing.concurrent_search_lock_wait_ms = queries_completed_ == 0
                ? 0.0
                : lock_wait_seconds_ * 1000.0 /
                        static_cast<double>(queries_completed_);
        if (error_) std::rethrow_exception(error_);
    }

private:
    bool enabled_ = false;
    const DynamicArgs& args_;
    DynamicPublishedState& state_;
    const std::vector<float>& queries_;
    int nq_ = 0;
    std::atomic<bool> stop_{false};
    std::atomic<bool> ready_{false};
    std::thread thread_;
    std::exception_ptr error_;
    uint64_t queries_completed_ = 0;
    double wall_seconds_ = 0.0;
    double query_seconds_ = 0.0;
    double lock_wait_seconds_ = 0.0;

    void stop_and_join() noexcept {
        stop_.store(true, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    void run() noexcept {
        const auto worker_start = std::chrono::steady_clock::now();
        ready_.store(true, std::memory_order_release);
        try {
            size_t query_index = 0;
            while (!stop_.load(std::memory_order_acquire)) {
                const auto query_start = std::chrono::steady_clock::now();
                const auto lock_start = std::chrono::steady_clock::now();
                while (state_.writer_pending.load(std::memory_order_acquire) &&
                       !stop_.load(std::memory_order_acquire)) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                if (stop_.load(std::memory_order_acquire)) break;
                std::shared_lock<std::shared_mutex> lock(state_.mutex);
                lock_wait_seconds_ += seconds_since(lock_start);
                if (stop_.load(std::memory_order_acquire)) break;
                const float* query = queries_.data() + query_index * state_.index.dim;
                dynamic_search_one_published(args_, state_, query);
                lock.unlock();
                query_seconds_ += seconds_since(query_start);
                ++queries_completed_;
                query_index = (query_index + 1) % static_cast<size_t>(nq_);
            }
        } catch (...) {
            error_ = std::current_exception();
            stop_.store(true, std::memory_order_release);
        }
        wall_seconds_ = seconds_since(worker_start);
    }
};

double dynamic_component_recall(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<int32_t>& gt,
        int nq,
        int gt_dim,
        int k,
        int64_t base_size,
        bool inserted) {
    int64_t hits = 0;
    int64_t total = 0;
    for (int qi = 0; qi < nq; ++qi) {
        for (int gi = 0; gi < k; ++gi) {
            const int32_t truth = gt[static_cast<size_t>(qi) * gt_dim + gi];
            if ((truth >= base_size) != inserted) continue;
            ++total;
            for (int rank = 0; rank < k; ++rank) {
                if (labels[static_cast<size_t>(qi) * k + rank] == truth) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return total == 0 ? std::numeric_limits<double>::quiet_NaN()
                      : static_cast<double>(hits) / static_cast<double>(total);
}

double dynamic_returned_insert_fraction(
        const std::vector<faiss::idx_t>& labels,
        int64_t base_size) {
    int64_t count = 0;
    for (faiss::idx_t id : labels) count += id >= base_size;
    return labels.empty() ? 0.0 : static_cast<double>(count) / labels.size();
}

double dynamic_gt_insert_fraction(
        const std::vector<int32_t>& gt,
        int nq,
        int gt_dim,
        int k,
        int64_t base_size) {
    int64_t count = 0;
    for (int qi = 0; qi < nq; ++qi) {
        for (int gi = 0; gi < k; ++gi) {
            count += gt[static_cast<size_t>(qi) * gt_dim + gi] >= base_size;
        }
    }
    return static_cast<double>(count) / static_cast<double>(nq * k);
}

double dynamic_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
            static_cast<double>(values.size());
}

struct DynamicCalibrationFit {
    float tau = 0.0f;
    double target = 0.0;
    int conformal_order = 0;
    int calibration_queries = 0;
    double empirical_all_top10_coverage = 0.0;
    AdaptiveRerankSelection counts;
};

DynamicCalibrationFit dynamic_fit_target_calibration(
        const DynamicArgs& args,
        const faiss::IndexBinaryHNSW& index,
        const FbinMmap& fp_store,
        const std::vector<uint32_t>& physical_row_by_label,
        const std::vector<float>& calibration_queries,
        int calibration_nq,
        int ef,
        const uint8_t* codes,
        size_t code_stride,
        const DynamicRowSource* exact_fp_source,
        const QdrantTqIndex* tq_index,
        const uint8_t* tq_codes,
        size_t tq_code_size,
        std::optional<float> forced_tau = std::nullopt) {
    if (calibration_nq <= 0) {
        throw std::runtime_error("target calibration requires at least one query");
    }
    const int candidate_k = ef;
    std::vector<float> distances;
    std::vector<faiss::idx_t> candidates;
    const bool tq_mode = tq_index != nullptr;
    if (tq_mode) {
        if (tq_codes == nullptr || tq_code_size == 0) {
            throw std::runtime_error("TQ calibration requires encoded vector storage");
        }
        dynamic_search_tq(
                index,
                tq_index,
                tq_codes,
                tq_code_size,
                calibration_queries.data(),
                calibration_nq,
                index.d,
                candidate_k,
                ef,
                args.search_threads,
                distances,
                candidates);
    } else {
        search_packed_asym(
                index,
                calibration_queries.data(),
                calibration_nq,
                index.d,
                candidate_k,
                ef,
                args.search_threads,
                AdcMode::Int16,
                LutMode::Nibble,
                SearchImpl::Inline,
                true,
                true,
                false,
                false,
                16,
                16,
                nullptr,
                codes,
                code_stride,
                nullptr,
                0,
                distances,
                candidates,
                nullptr);
    }

    PackedAsymBQDistanceComputer scale_computer(
            nullptr,
            static_cast<size_t>((index.d + 7) / 8),
            static_cast<size_t>((index.d + 7) / 8),
            index.d,
            AdcMode::Int16,
            LutMode::Nibble);
    std::vector<double> requirements(static_cast<size_t>(calibration_nq), 0.0);
    std::vector<std::array<int, 10>> exact_positions(static_cast<size_t>(calibration_nq));
    for (int qi = 0; qi < calibration_nq; ++qi) {
        const size_t base = static_cast<size_t>(qi) * candidate_k;
        int valid = 0;
        while (valid < candidate_k && candidates[base + static_cast<size_t>(valid)] >= 0 &&
               std::isfinite(distances[base + static_cast<size_t>(valid)])) {
            ++valid;
        }
        if (valid < args.k || args.k != 10) {
            throw std::runtime_error("calibration candidate pool has fewer than ten vectors");
        }
        const float* query = calibration_queries.data() + static_cast<size_t>(qi) * index.d;
        double query_norm2 = 0.0;
        for (int j = 0; j < index.d; ++j) {
            query_norm2 += static_cast<double>(query[j]) * query[j];
        }
        const double query_norm = std::sqrt(std::max(query_norm2, 1e-30));
        std::vector<std::pair<double, int>> exact;
        exact.reserve(static_cast<size_t>(valid));
        for (int ci = 0; ci < valid; ++ci) {
            const faiss::idx_t label = candidates[base + static_cast<size_t>(ci)];
            if (label < 0 || static_cast<size_t>(label) >= physical_row_by_label.size()) {
                throw std::runtime_error("calibration candidate label is out of range");
            }
            const uint32_t physical = physical_row_by_label[static_cast<size_t>(label)];
            const float* vector = exact_fp_source != nullptr
                    ? exact_fp_source->row(physical)
                    : fp_store.row(physical);
            double dot = 0.0;
            double norm2 = 0.0;
            for (int j = 0; j < index.d; ++j) {
                dot += static_cast<double>(query[j]) * vector[j];
                norm2 += static_cast<double>(vector[j]) * vector[j];
            }
            const double cosine = dot / (query_norm * std::sqrt(std::max(norm2, 1e-30)));
            // Ascending pair order implements higher cosine first, then smaller label.
            exact.emplace_back(-cosine, ci);
        }
        std::sort(
                exact.begin(),
                exact.end(),
                [&](const auto& lhs, const auto& rhs) {
                    if (lhs.first != rhs.first) return lhs.first < rhs.first;
                    return candidates[base + static_cast<size_t>(lhs.second)] <
                            candidates[base + static_cast<size_t>(rhs.second)];
                });
        const float anchor = distances[base + static_cast<size_t>(args.k - 1)];
        double required = 0.0;
        for (int rank = 0; rank < args.k; ++rank) {
            const int position = exact[static_cast<size_t>(rank)].second;
            exact_positions[static_cast<size_t>(qi)][static_cast<size_t>(rank)] = position;
            required = std::max(
                    required,
                    static_cast<double>(
                            distances[base + static_cast<size_t>(position)] - anchor));
        }
        // Official TurboQuant search distances are already reconstructed in
        // normalized inner-product units. ADC-BQ's integer LUT needs its
        // query-specific dequantization scale; TQ therefore uses unit scale
        // while preserving the same conformal margin calibration rule.
        const float scale = tq_mode ? 1.0f : scale_computer.int16_lut_scale(query);
        requirements[static_cast<size_t>(qi)] =
                std::max(0.0, required) / std::max(static_cast<double>(scale), 1e-30);
    }

    const int order = std::min(
            calibration_nq,
            static_cast<int>(std::ceil((calibration_nq + 1) * args.calibration_target)));
    std::vector<double> ordered = requirements;
    std::nth_element(ordered.begin(), ordered.begin() + (order - 1), ordered.end());
    const float tau = forced_tau.has_value()
            ? *forced_tau
            : static_cast<float>(ordered[static_cast<size_t>(order - 1)]);
    std::vector<int> counts(static_cast<size_t>(calibration_nq), 0);
    int covered = 0;
    for (int qi = 0; qi < calibration_nq; ++qi) {
        const size_t base = static_cast<size_t>(qi) * candidate_k;
        const float* query = calibration_queries.data() + static_cast<size_t>(qi) * index.d;
        const float scale = tq_mode ? 1.0f : scale_computer.int16_lut_scale(query);
        const int take = calibrated_threshold_rerank_count(
                candidates.data() + base,
                distances.data() + base,
                candidate_k,
                args.k,
                tau,
                scale);
        counts[static_cast<size_t>(qi)] = take;
        bool all = true;
        for (int rank = 0; rank < args.k; ++rank) {
            all = all && exact_positions[static_cast<size_t>(qi)][static_cast<size_t>(rank)] < take;
        }
        covered += all;
    }
    DynamicCalibrationFit fit;
    fit.tau = tau;
    fit.target = args.calibration_target;
    fit.conformal_order = forced_tau.has_value() ? 0 : order;
    fit.calibration_queries = calibration_nq;
    fit.empirical_all_top10_coverage =
            static_cast<double>(covered) / static_cast<double>(calibration_nq);
    fit.counts = summarize_rerank_counts(std::move(counts));
    return fit;
}

std::vector<faiss::idx_t> dynamic_rerank_chunked_fp(
        const DynamicRowSource& source,
        const std::vector<uint32_t>& physical_row_by_label,
        const float* queries,
        int nq,
        int dim,
        const std::vector<faiss::idx_t>& candidates,
        int candidate_k,
        int k,
        int threads) {
    std::vector<faiss::idx_t> output(static_cast<size_t>(nq) * k, -1);
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int qi = 0; qi < nq; ++qi) {
        const float* query = queries + static_cast<size_t>(qi) * dim;
        std::vector<std::pair<float, faiss::idx_t>> scored;
        scored.reserve(static_cast<size_t>(candidate_k));
        for (int rank = 0; rank < candidate_k; ++rank) {
            const faiss::idx_t label =
                    candidates[static_cast<size_t>(qi) * candidate_k + rank];
            if (label < 0 || static_cast<size_t>(label) >= physical_row_by_label.size()) {
                continue;
            }
            const uint32_t physical = physical_row_by_label[static_cast<size_t>(label)];
            const float* vector = source.row(physical);
            float dot = 0.0f;
            double squared_norm = 0.0;
#pragma omp simd reduction(+ : dot)
            for (int j = 0; j < dim; ++j) dot += query[j] * vector[j];
#pragma omp simd reduction(+ : squared_norm)
            for (int j = 0; j < dim; ++j) {
                squared_norm += static_cast<double>(vector[j]) * vector[j];
            }
            scored.emplace_back(
                    static_cast<float>(
                            static_cast<double>(dot) /
                            std::sqrt(std::max(squared_norm, 1e-30))),
                    label);
        }
        const size_t take = std::min<size_t>(static_cast<size_t>(k), scored.size());
        std::partial_sort(
                scored.begin(),
                scored.begin() + take,
                scored.end(),
                [](const auto& left, const auto& right) {
                    if (left.first != right.first) return left.first > right.first;
                    return left.second < right.second;
                });
        for (size_t rank = 0; rank < take; ++rank) {
            output[static_cast<size_t>(qi) * k + rank] = scored[rank].second;
        }
    }
    return output;
}

void dynamic_search_rows(
        std::ofstream& csv,
        const DynamicArgs& args,
        DynamicAdcIndex& index,
        const FbinMmap& fp_store,
        const std::vector<uint32_t>& physical_row_by_label,
        const std::string& mode,
        int insert_size,
        const std::vector<float>& queries,
        int nq,
        const std::vector<float>& calibration_queries,
        int calibration_nq,
        const std::vector<int32_t>& gt,
        int gt_dim,
        const std::vector<int32_t>& base_gt,
        int base_gt_dim,
        double build_seconds,
        const DynamicTiming& timing,
        const std::unordered_map<int, float>* frozen_tau_by_ef = nullptr,
        std::unordered_map<int, float>* fitted_tau_by_ef = nullptr,
        const DynamicRowSource* chunked_fp_store = nullptr) {
    const int64_t base_size = static_cast<int64_t>(physical_row_by_label.size()) - insert_size;
    const int64_t corpus_size = static_cast<int64_t>(physical_row_by_label.size());
    if (index.bq_view->ntotal != corpus_size) {
        throw std::runtime_error("index size does not match physical row map");
    }
    if (args.online_graph_add == DynamicGraphAddMode::Tq1 &&
        (!index.has_tq() || index.tq_rows() != static_cast<size_t>(corpus_size))) {
        throw std::runtime_error("TQ index size does not match physical row map");
    }

    DynamicGraphSwapGuard graph_guard(index);
    const uint8_t* codes = storage_codes(*index.bq_view);
    const size_t code_stride = index.bq_view->code_size;
    const std::vector<double> unused_norms;

    for (int ef : args.efs) {
        const bool calibrated = !args.calibration_query.empty();
        const int candidate_k = calibrated ? ef : args.rerank_n;
        DynamicCalibrationFit calibration;
        if (calibrated) {
            std::optional<float> forced_tau;
            if (frozen_tau_by_ef != nullptr) {
                const auto found = frozen_tau_by_ef->find(ef);
                if (found == frozen_tau_by_ef->end()) {
                    throw std::runtime_error("missing frozen base calibration tau for efSearch");
                }
                forced_tau = found->second;
            }
            calibration = dynamic_fit_target_calibration(
                    args,
                    *index.bq_view,
                    fp_store,
                    physical_row_by_label,
                    calibration_queries,
                    calibration_nq,
                    ef,
                    codes,
                    code_stride,
                    chunked_fp_store,
                    args.online_graph_add == DynamicGraphAddMode::Tq1
                            ? index.tq_index.get()
                            : nullptr,
                    args.online_graph_add == DynamicGraphAddMode::Tq1
                            ? index.tq_codes()
                            : nullptr,
                    args.online_graph_add == DynamicGraphAddMode::Tq1
                            ? index.tq_code_size()
                            : 0,
                    forced_tau);
            if (fitted_tau_by_ef != nullptr) {
                (*fitted_tau_by_ef)[ef] = calibration.tau;
            }
            std::cout << std::setprecision(9)
                      << (forced_tau.has_value() ? "  frozen-base-calibration " : "  calibrated ")
                      << mode << " insert=" << insert_size
                      << " ef=" << ef << " target=" << calibration.target
                      << " tau=" << calibration.tau
                      << " Rmean=" << calibration.counts.mean_count
                      << " coverage=" << calibration.empirical_all_top10_coverage
                      << std::endl;
        }
        std::vector<double> seconds_values;
        std::vector<double> recall_values;
        std::vector<double> insert_recall_values;
        std::vector<double> current_base_recall_values;
        std::vector<double> base_gt_recall_values;
        std::vector<double> returned_insert_values;
        std::vector<double> direct_rows_values;
        std::vector<double> direct_requests_values;
        std::vector<double> submitted_bytes_values;
        std::vector<double> overlap_query_values;
        std::vector<double> rerank_count_mean_values;
        std::vector<double> rerank_count_min_values;
        std::vector<double> rerank_count_p95_values;
        std::vector<double> rerank_count_p99_values;
        std::vector<double> rerank_count_max_values;

        for (int round = 0; round < args.rounds; ++round) {
            std::vector<float> distances;
            std::vector<faiss::idx_t> candidates;
            std::vector<faiss::idx_t> final_labels;
            FpRerankCacheStats io_stats;
            FpInterQueryPipelineStats pipeline_stats;
            AdaptiveRerankSelection rerank_count_stats;
            const auto start = std::chrono::steady_clock::now();
            if (args.online_graph_add == DynamicGraphAddMode::Tq1) {
                dynamic_search_tq(
                        *index.bq_view,
                        index.tq_index.get(),
                        index.tq_codes(),
                        index.tq_code_size(),
                        queries.data(),
                        nq,
                        index.dim,
                        candidate_k,
                        ef,
                        args.search_threads,
                        distances,
                        candidates);
                std::vector<faiss::idx_t> calibrated_candidates;
                const std::vector<faiss::idx_t>* rerank_candidates = &candidates;
                if (calibrated) {
                    rerank_count_stats = select_calibrated_threshold_rerank_candidates(
                            candidates,
                            distances,
                            queries.data(),
                            nq,
                            index.dim,
                            candidate_k,
                            args.k,
                            calibration.tau,
                            AdcMode::Fp32,
                            LutMode::Nibble);
                    calibrated_candidates = std::move(rerank_count_stats.labels);
                    rerank_candidates = &calibrated_candidates;
                }
                if (chunked_fp_store != nullptr) {
                    final_labels = dynamic_rerank_chunked_fp(
                            *chunked_fp_store,
                            physical_row_by_label,
                            queries.data(),
                            nq,
                            index.dim,
                            *rerank_candidates,
                            candidate_k,
                            args.k,
                            args.search_threads);
                } else {
                    final_labels = rerank_cosine(
                            fp_store,
                            unused_norms,
                            queries.data(),
                            nq,
                            *rerank_candidates,
                            candidate_k,
                            args.k,
                            RerankDotMode::Fp32,
                            FpRerankMetric::Cosine,
                            args.fp_source,
                            FpRerankNormMode::OnFetch,
                            false,
                            0,
                            0,
                            false,
                            args.io_queue_depth,
                            &physical_row_by_label,
                            0,
                            &io_stats,
                            nullptr);
                }
            } else if (args.overlap_queries) {
                final_labels = search_rerank_interquery_pipeline(
                        *index.bq_view,
                        fp_store,
                        unused_norms,
                        queries.data(),
                        nq,
                        index.dim,
                        candidate_k,
                        args.k,
                        ef,
                        args.search_threads,
                        AdcMode::Int16,
                        LutMode::Nibble,
                        true,
                        false,
                        false,
                        16,
                        16,
                        nullptr,
                        codes,
                        code_stride,
                        nullptr,
                        0,
                        RerankDotMode::Fp32,
                        FpRerankMetric::Cosine,
                        FpRerankNormMode::OnFetch,
                        false,
                        0,
                        false,
                        candidate_k,  // fixed_rerank_n (added to the core on 2026-11-19): min(candidate_k,
                                      // candidate_k) keeps the 2026-11-18 behaviour of the Fig. 19 binary
                        calibrated,
                        calibration.tau,
                        args.io_queue_depth,
                        nullptr,
                        &physical_row_by_label,
                        0,
                        distances,
                        candidates,
                        nullptr,
                        &io_stats,
                        &pipeline_stats,
                        nullptr,
                        &rerank_count_stats);
            } else {
                search_packed_asym(
                        *index.bq_view,
                        queries.data(),
                        nq,
                        index.dim,
                        candidate_k,
                        ef,
                        args.search_threads,
                        AdcMode::Int16,
                        LutMode::Nibble,
                        SearchImpl::Inline,
                        true,
                        true,
                        false,
                        false,
                        16,
                        16,
                        nullptr,
                        codes,
                        code_stride,
                        nullptr,
                        0,
                        distances,
                        candidates,
                        nullptr);
                std::vector<faiss::idx_t> calibrated_candidates;
                const std::vector<faiss::idx_t>* rerank_candidates = &candidates;
                if (calibrated) {
                    rerank_count_stats = select_calibrated_threshold_rerank_candidates(
                            candidates,
                            distances,
                            queries.data(),
                            nq,
                            index.dim,
                            candidate_k,
                            args.k,
                            calibration.tau,
                            AdcMode::Int16,
                            LutMode::Nibble);
                    calibrated_candidates = std::move(rerank_count_stats.labels);
                    rerank_candidates = &calibrated_candidates;
                }
                final_labels = rerank_cosine(
                        fp_store,
                        unused_norms,
                        queries.data(),
                        nq,
                        *rerank_candidates,
                        candidate_k,
                        args.k,
                        RerankDotMode::Fp32,
                        FpRerankMetric::Cosine,
                        args.fp_source,
                        FpRerankNormMode::OnFetch,
                        false,
                        0,
                        0,
                        false,
                        args.io_queue_depth,
                        &physical_row_by_label,
                        0,
                        &io_stats,
                        nullptr);
            }
            const double elapsed = seconds_since(start);
            const double recall = recall_at_k(final_labels, gt, nq, gt_dim, args.k);
            const double insert_recall = dynamic_component_recall(
                    final_labels, gt, nq, gt_dim, args.k, base_size, true);
            const double current_base_recall = dynamic_component_recall(
                    final_labels, gt, nq, gt_dim, args.k, base_size, false);
            const double base_gt_recall = recall_at_k(
                    final_labels, base_gt, nq, base_gt_dim, args.k);

            seconds_values.push_back(elapsed);
            recall_values.push_back(recall);
            insert_recall_values.push_back(insert_recall);
            current_base_recall_values.push_back(current_base_recall);
            base_gt_recall_values.push_back(base_gt_recall);
            returned_insert_values.push_back(
                    dynamic_returned_insert_fraction(final_labels, base_size));
            direct_rows_values.push_back(static_cast<double>(io_stats.direct_rows));
            direct_requests_values.push_back(static_cast<double>(io_stats.direct_requests));
            submitted_bytes_values.push_back(static_cast<double>(io_stats.direct_submitted_bytes));
            overlap_query_values.push_back(
                    static_cast<double>(pipeline_stats.queries_overlapped_with_search));
            rerank_count_mean_values.push_back(
                    calibrated ? rerank_count_stats.mean_count : candidate_k);
            rerank_count_min_values.push_back(
                    calibrated ? rerank_count_stats.min_count : candidate_k);
            rerank_count_p95_values.push_back(
                    calibrated ? rerank_count_stats.p95_count : candidate_k);
            rerank_count_p99_values.push_back(
                    calibrated ? rerank_count_stats.p99_count : candidate_k);
            rerank_count_max_values.push_back(
                    calibrated ? rerank_count_stats.max_count : candidate_k);

            std::cout << std::fixed << std::setprecision(6)
                      << "  " << mode << " insert=" << insert_size << " ef=" << ef
                      << " round=" << (round + 1) << "/" << args.rounds
                      << " recall=" << recall
                      << " insert_gt=" << insert_recall
                      << " current_base_gt=" << current_base_recall
                      << " qps=" << (static_cast<double>(nq) / elapsed)
                      << " rerank_mean=" << rerank_count_mean_values.back()
                      << " reads/q=" << (static_cast<double>(io_stats.direct_requests) / nq)
                      << std::endl;
        }

        const double search_seconds = dynamic_mean(seconds_values);
        std::string method = args.online_graph_add == DynamicGraphAddMode::Tq1
                ? (calibrated
                           ? "fp_hnsw_qdrant_tq1plus_fp32_calibrated_adaptive_rerank"
                           : "fp_hnsw_qdrant_tq1plus_fp32_rerank60")
                : "fp_hnsw_int16_adcbq_fp32_rerank";
        if (insert_size > 0 && !index.has_fp_graph() &&
            args.online_graph_add != DynamicGraphAddMode::Tq1) {
            method = timing.graph_add_mode == "adcbq"
                    ? "fp_base_adcbq_insert_int16_adcbq_fp32_rerank"
                    : "fp_base_hamming_bq_insert_int16_adcbq_fp32_rerank";
        } else if (insert_size > 0 &&
                   args.online_graph_add == DynamicGraphAddMode::Tq1 &&
                   mode == "online_add_no_rebuild") {
            method = calibrated
                    ? "fp_base_tq1plus_insert_tq1plus_fp32_calibrated_adaptive_rerank"
                    : "fp_base_tq1plus_insert_tq1plus_fp32_rerank60";
        }
        csv << args.dataset << ',' << mode << ',' << args.policy << ','
            << base_size << ',' << insert_size << ',' << corpus_size << ','
            << index.dim << ',' << args.m << ',' << args.ef_construction << ','
            << method << ','
            << (args.independent_online_checkpoints ? "fixed_base_snapshot" : "cumulative_reuse")
            << ',' << candidate_k << ','
            << (calibrated ? "true" : "false") << ','
            << (calibrated ? calibration.target : 0.0) << ','
            << (calibrated ? calibration.tau : 0.0f) << ','
            << (calibrated ? calibration.conformal_order : 0) << ','
            << (calibrated ? calibration.calibration_queries : 0) << ','
            << (calibrated ? calibration.empirical_all_top10_coverage : 0.0) << ','
            << (calibrated
                        ? (frozen_tau_by_ef != nullptr
                                  ? "frozen_1m_base_tau_no_online_recalibration"
                                  : "exact_fp_top10_within_graph_candidate_pool")
                        : "fixed_rerank") << ','
            << fp_rerank_source_name(args.fp_source) << ','
            << (args.overlap_queries ? "true" : "false") << ','
            << args.io_queue_depth << ',' << ef << ',' << args.k << ',' << nq << ','
            << args.rounds << ',' << args.build_threads << ',' << args.add_threads << ','
            << args.search_threads << ',' << build_seconds << ',' << timing.add_seconds << ','
            << timing.input_copy_seconds << ',' << timing.normalize_seconds << ','
            << timing.bq_encode_seconds << ',' << timing.graph_add_seconds << ','
            << timing.fp_append_seconds << ','
            << timing.fp_compaction_seconds << ',' << timing.fp_layout << ','
            << timing.graph_add_mode << ','
            << timing.publish_batches << ',' << timing.publish_lock_wait_seconds << ','
            << timing.publish_critical_seconds << ','
            << timing.publish_critical_max_seconds << ','
            << timing.concurrent_search_queries << ','
            << timing.concurrent_search_wall_seconds << ','
            << (timing.concurrent_search_wall_seconds > 0.0
                        ? static_cast<double>(timing.concurrent_search_queries) /
                                timing.concurrent_search_wall_seconds
                        : 0.0)
            << ','
            << timing.concurrent_search_latency_ms << ','
            << timing.concurrent_search_lock_wait_ms << ','
            << search_seconds << ',' << (search_seconds * 1000.0 / nq) << ','
            << (static_cast<double>(nq) / search_seconds) << ','
            << dynamic_mean(recall_values) << ','
            << dynamic_mean(insert_recall_values) << ','
            << dynamic_mean(current_base_recall_values) << ','
            << dynamic_mean(base_gt_recall_values) << ','
            << dynamic_mean(returned_insert_values) << ','
            << dynamic_gt_insert_fraction(gt, nq, gt_dim, args.k, base_size) << ','
            << dynamic_mean(direct_rows_values) / nq << ','
            << dynamic_mean(direct_requests_values) / nq << ','
            << dynamic_mean(submitted_bytes_values) / nq << ','
            << dynamic_mean(overlap_query_values) << ','
            << dynamic_mean(rerank_count_mean_values) << ','
            << dynamic_mean(rerank_count_min_values) << ','
            << dynamic_mean(rerank_count_p95_values) << ','
            << dynamic_mean(rerank_count_p99_values) << ','
            << dynamic_mean(rerank_count_max_values) << '\n';
        csv.flush();
        if (!csv) throw std::runtime_error("failed to flush output CSV: " + args.out);
    }
}

std::vector<uint32_t> dynamic_physical_map(
        const std::vector<int64_t>& base_to_larger,
        const std::vector<int64_t>& selected,
        int insert_size,
        int64_t larger_n) {
    std::vector<uint32_t> out;
    out.reserve(base_to_larger.size() + static_cast<size_t>(insert_size));
    auto append = [&](int64_t id) {
        if (id < 0 || id >= larger_n ||
            static_cast<uint64_t>(id) > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error("physical source row is out of range");
        }
        out.push_back(static_cast<uint32_t>(id));
    };
    for (int64_t id : base_to_larger) append(id);
    for (int i = 0; i < insert_size; ++i) append(selected[static_cast<size_t>(i)]);
    return out;
}

void validate_dynamic_ids(
        const std::vector<int64_t>& base_to_larger,
        const std::vector<int64_t>& selected,
        int64_t base_n,
        int64_t larger_n,
        int max_insert) {
    if (base_to_larger.size() != static_cast<size_t>(base_n)) {
        throw std::runtime_error("base-to-larger map size does not match base rows");
    }
    if (selected.size() < static_cast<size_t>(max_insert)) {
        throw std::runtime_error("selected-id stream is shorter than max insert size");
    }
    std::unordered_set<int64_t> seen;
    seen.reserve(static_cast<size_t>(base_n + max_insert) * 2);
    for (int64_t id : base_to_larger) {
        if (id < 0 || id >= larger_n || !seen.insert(id).second) {
            throw std::runtime_error("invalid or duplicate base-to-larger id");
        }
    }
    for (int i = 0; i < max_insert; ++i) {
        const int64_t id = selected[static_cast<size_t>(i)];
        if (id < 0 || id >= larger_n || !seen.insert(id).second) {
            throw std::runtime_error("invalid or duplicate selected insert id");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const DynamicArgs args = parse_dynamic_args(argc, argv);
        FbinMmap base(args.base_train);
        std::unique_ptr<FbinMmap> larger_file;
        std::unique_ptr<DynamicRowSource> larger_source_owner;
        if (args.insert_emb_root.empty()) {
            larger_file = std::make_unique<FbinMmap>(args.larger_train);
            larger_source_owner =
                    std::make_unique<DynamicFbinRowSource>(*larger_file);
        } else {
            larger_source_owner = std::make_unique<DynamicChunkedCombinedSource>(
                    base,
                    args.insert_emb_root,
                    args.insert_permutation,
                    args.insert_sizes.back());
        }
        const DynamicRowSource& larger = *larger_source_owner;
        const DynamicRowSource* chunked_fp_source =
                args.insert_emb_root.empty() ? nullptr : larger_source_owner.get();
        FbinMmap query_file(args.query);
        if (base.d != larger.dimension() || base.d != query_file.d || base.d % 8 != 0) {
            throw std::runtime_error("input dimension mismatch or dimension not divisible by 8");
        }
        const std::vector<int64_t> base_to_larger =
                dynamic_read_i64bin(args.base_to_larger_ids);
        const std::vector<int64_t> selected = dynamic_read_i64bin(args.selected_ids);
        const int max_insert = args.insert_sizes.back();
        validate_dynamic_ids(base_to_larger, selected, base.n, larger.rows(), max_insert);

        const std::vector<float> queries = read_queries(query_file, args.max_queries);
        const int nq = static_cast<int>(queries.size() / static_cast<size_t>(base.d));
        std::vector<float> calibration_queries;
        int calibration_nq = 0;
        if (!args.calibration_query.empty()) {
            FbinMmap calibration_file(args.calibration_query);
            if (calibration_file.d != base.d) {
                throw std::runtime_error("calibration query dimension mismatch");
            }
            calibration_queries = read_queries(calibration_file, 0);
            calibration_nq = static_cast<int>(
                    calibration_queries.size() / static_cast<size_t>(base.d));
        }
        int32_t base_gt_n = 0;
        int32_t base_gt_dim = 0;
        const std::vector<int32_t> base_gt =
                read_ibin(args.base_gt, nq, base_gt_n, base_gt_dim);
        if (base_gt_n != nq || base_gt_dim < args.k) {
            throw std::runtime_error("base GT does not match query count/k");
        }

        std::unique_ptr<DynamicFpAppendStore> fp_append_store;
        std::unique_ptr<FbinMmap> fp_store;
        if (!args.fp_append_store_path.empty()) {
            fp_append_store = std::make_unique<DynamicFpAppendStore>(
                    args.fp_append_store_path,
                    base,
                    base.n + static_cast<int64_t>(max_insert),
                    args.rebuild_fp_append_store,
                    args.fp_durable_writes);
            fp_store = std::make_unique<FbinMmap>(fp_append_store->path());
            if (fp_store->n != base.n + max_insert || fp_store->d != larger.dimension()) {
                throw std::runtime_error("FP append store shape does not match the dynamic run");
            }
        } else if (chunked_fp_source != nullptr) {
            // The row source supplies exact FP vectors. Keep a small valid fbin
            // object for shared interfaces that are inactive in fixed TQ mode.
            fp_store = std::make_unique<FbinMmap>(args.base_train);
        } else {
            if (args.prepare_aligned_store) {
                prepare_aligned_fp_store(args.fp_rerank_path, *larger_file);
            }
            const std::string fp_path = args.fp_rerank_path.empty()
                    ? args.larger_train
                    : args.fp_rerank_path;
            fp_store = std::make_unique<FbinMmap>(fp_path);
            if (fp_store->n != larger.rows() || fp_store->d != larger.dimension()) {
                throw std::runtime_error("FP rerank store shape does not match larger source");
            }
        }
        if (args.fp_source == FpRerankSource::ODirectUring) {
            if (fp_store->data_offset != kFpDataAlignment) {
                throw std::runtime_error("O_DIRECT rerank store payload is not 4 KiB aligned");
            }
            const size_t row_bytes = static_cast<size_t>(fp_store->d) * sizeof(float);
            if (row_bytes % kFpDataAlignment != 0) {
                throw std::runtime_error("O_DIRECT dynamic run requires vector rows to occupy whole 4 KiB pages");
            }
        }

        fs::path output_path(args.out);
        if (!output_path.parent_path().empty()) fs::create_directories(output_path.parent_path());
        std::ofstream csv(args.out);
        if (!csv) throw std::runtime_error("failed to create output CSV: " + args.out);
        csv << "dataset,mode,policy,base_size,insert_size,corpus_size,dim,M,efConstruction,"
            << "method,online_checkpoint_source,candidate_cap,calibrated_target_enabled,"
            << "calibration_target,calibration_tau,calibration_conformal_order,"
            << "calibration_queries,calibration_empirical_all_top10_coverage,"
            << "calibration_reference,fp_rerank_source,interquery_overlap,io_queue_depth,efSearch,k,nq,"
            << "rounds,build_threads,add_threads,search_threads,build_or_rebuild_seconds,add_seconds,"
            << "input_copy_seconds,normalize_seconds,bq_encode_seconds,graph_add_seconds,"
            << "fp_append_seconds,fp_compaction_seconds,"
            << "fp_layout,online_graph_add,publish_batches,publish_lock_wait_seconds,"
            << "publish_critical_seconds,publish_critical_max_seconds,concurrent_search_queries,"
            << "concurrent_search_wall_seconds,concurrent_search_qps,concurrent_search_latency_ms,"
            << "concurrent_search_lock_wait_ms,"
            << "search_seconds_mean,latency_ms_per_query,qps,recall_current_at_k,insert_gt_recall_at_k,"
            << "current_gt_base_recall_at_k,base_gt_recall_at_k,returned_insert_fraction,"
            << "current_gt_insert_fraction,physical_rows_per_query,physical_reads_per_query,"
            << "submitted_bytes_per_query,pipeline_overlapped_queries_mean,"
            << "adaptive_rerank_count_mean,adaptive_rerank_count_min,"
            << "adaptive_rerank_count_p95,adaptive_rerank_count_p99,"
            << "adaptive_rerank_count_max\n";

        std::cout << "base=(" << base.n << ", " << base.d << ") larger=(" << larger.rows()
                  << ", " << larger.dimension() << ") nq=" << nq
                  << " rerank=" << args.rerank_n
                  << " source=" << fp_rerank_source_name(args.fp_source)
                  << " overlap=" << args.overlap_queries
                  << " online_graph_add=" << dynamic_graph_add_mode_name(args.online_graph_add)
                  << " fp_layout=" << (fp_append_store ? "append" : "source_map")
                  << " search_during_add=" << args.search_during_add
                  << " calibration_queries=" << calibration_nq
                  << " calibration_target=" << args.calibration_target
                  << std::endl;

        auto online = std::make_unique<DynamicAdcIndex>(base.d, args.m, args.ef_construction);
        std::cout << "building online ADC-BQ base: FP-HNSW graph + packed BQ storage" << std::endl;
        DynamicBuildPhaseTiming base_phase_timing;
        auto start = std::chrono::steady_clock::now();
        dynamic_add_contiguous(
                *online,
                base,
                base.n,
                args.add_batch_size,
                args.build_threads,
                "base rows",
                &base_phase_timing);
        if (args.online_graph_add == DynamicGraphAddMode::Tq1) {
            online->load_tq(args.tq_code_path, args.tq_meta_path);
        }
        const double base_build_seconds = seconds_since(start);
        if (args.online_graph_add != DynamicGraphAddMode::Fp) {
            std::cout << "sealing FP-built base graph; subsequent nodes use "
                      << dynamic_graph_add_mode_name(args.online_graph_add)
                      << " construction over DRAM BQ storage" << std::endl;
            online->seal_fp_graph_for_compressed_updates();
        }
        const std::vector<uint32_t> base_physical = fp_append_store
                ? fp_append_store->physical_map()
                : dynamic_physical_map(base_to_larger, selected, 0, larger.rows());
        DynamicTiming base_timing;
        dynamic_accumulate_build_phases(base_timing, base_phase_timing);
        base_timing.fp_layout = fp_append_store ? "append" : "source_map";
        base_timing.graph_add_mode = "none";
        std::unordered_map<int, float> base_calibration_tau_by_ef;
        dynamic_search_rows(
                csv,
                args,
                *online,
                *fp_store,
                base_physical,
                "base_only",
                0,
                queries,
                nq,
                calibration_queries,
                calibration_nq,
                base_gt,
                base_gt_dim,
                base_gt,
                base_gt_dim,
                base_build_seconds,
                base_timing,
                nullptr,
                &base_calibration_tau_by_ef,
                chunked_fp_source);

        int previous_insert = 0;
        for (int insert_size : args.insert_sizes) {
            int32_t gt_n = 0;
            int32_t gt_dim = 0;
            const std::string gt_path = dynamic_stage_gt_path(args, insert_size);
            const std::vector<int32_t> stage_gt = read_ibin(gt_path, nq, gt_n, gt_dim);
            if (gt_n != nq || gt_dim < args.k) {
                throw std::runtime_error("stage GT does not match query count/k: " + gt_path);
            }
            std::vector<uint32_t> physical = fp_append_store
                    ? dynamic_identity_map(static_cast<size_t>(base.n) + insert_size)
                    : dynamic_physical_map(base_to_larger, selected, insert_size, larger.rows());
            const FbinMmap* stage_fp_store = fp_store.get();
            std::unique_ptr<FbinMmap> compacted_fp_store;
            DynamicTiming online_timing;
            online_timing.fp_layout = fp_append_store ? "append" : "source_map";
            online_timing.graph_add_mode = dynamic_graph_add_mode_name(args.online_graph_add);

            if (!args.skip_online) {
                std::unique_ptr<DynamicAdcIndex> independent_index;
                DynamicAdcIndex* online_stage = online.get();
                size_t add_first = static_cast<size_t>(previous_insert);
                if (args.independent_online_checkpoints) {
                    std::cout << "cloning fixed 1M base snapshot for independent checkpoint"
                              << std::endl;
                    independent_index = std::make_unique<DynamicAdcIndex>(*online);
                    online_stage = independent_index.get();
                    if (args.online_graph_add == DynamicGraphAddMode::Tq1) {
                        online_stage->load_tq(args.tq_code_path, args.tq_meta_path);
                    }
                    add_first = 0;
                    if (fp_append_store) fp_append_store->reset_to_base();
                }
                std::cout << "\nonline "
                          << dynamic_graph_add_mode_name(args.online_graph_add)
                          << "-HNSW add "
                          << (args.independent_online_checkpoints ? 0 : previous_insert)
                          << "->" << insert_size
                          << std::endl;
                std::vector<uint32_t> initially_published = fp_append_store
                        ? dynamic_identity_map(static_cast<size_t>(base.n) + add_first)
                        : dynamic_physical_map(
                                  base_to_larger,
                                  selected,
                                  static_cast<int>(add_first),
                                  larger.rows());
                DynamicPublishedState published(
                        *online_stage,
                        *fp_store,
                        std::move(initially_published));
                DynamicConcurrentSearchWorker concurrent_search(
                        args.search_during_add,
                        args,
                        published,
                        queries,
                        nq);
                start = std::chrono::steady_clock::now();
                for (size_t batch_first = add_first;
                     batch_first < static_cast<size_t>(insert_size);
                     batch_first += static_cast<size_t>(args.add_batch_size)) {
                    const size_t batch_last = std::min(
                            static_cast<size_t>(insert_size),
                            batch_first + static_cast<size_t>(args.add_batch_size));
                    if (fp_append_store) {
                        online_timing.fp_append_seconds += fp_append_store->append_selected(
                                *larger_file,
                                selected,
                                batch_first,
                                batch_last,
                                args.add_batch_size,
                                args.add_threads);
                    }

                    // Sign packing needs neither normalization nor rotation.
                    // Keep it outside the writer section so readers only wait
                    // for the actual HNSW mutation and atomic batch publish.
                    std::vector<uint8_t> prepared_bq_codes;
                    std::vector<float> prepared_adc_queries;
                    if (args.online_graph_add != DynamicGraphAddMode::Fp) {
                        if (args.online_graph_add == DynamicGraphAddMode::AdcBq ||
                            args.online_graph_add == DynamicGraphAddMode::Tq1) {
                            const auto copy_start = std::chrono::steady_clock::now();
                            prepared_adc_queries = dynamic_copy_selected_fp_queries(
                                    larger,
                                    selected,
                                    batch_first,
                                    batch_last,
                                    args.add_threads);
                            online_timing.input_copy_seconds += seconds_since(copy_start);
                        }
                        const auto encode_start = std::chrono::steady_clock::now();
                        prepared_bq_codes = dynamic_pack_selected_sign_bits(
                                larger,
                                selected,
                                batch_first,
                                batch_last,
                                args.add_threads);
                        online_timing.bq_encode_seconds += seconds_since(encode_start);
                        if (args.online_graph_add == DynamicGraphAddMode::Tq1) {
                            const auto tq_encode_start = std::chrono::steady_clock::now();
                            online_stage->append_tq_block(
                                    prepared_adc_queries, args.add_threads);
                            online_timing.bq_encode_seconds +=
                                    seconds_since(tq_encode_start);
                        }
                    }

                    published.writer_pending.store(true, std::memory_order_release);
                    try {
                        const auto lock_start = std::chrono::steady_clock::now();
                        std::unique_lock<std::shared_mutex> publish_lock(published.mutex);
                        online_timing.publish_lock_wait_seconds += seconds_since(lock_start);
                        const auto critical_start = std::chrono::steady_clock::now();
                        const auto graph_start = std::chrono::steady_clock::now();
                        DynamicBuildPhaseTiming fp_add_phase;
                        if (args.online_graph_add == DynamicGraphAddMode::BqHamming) {
                            online_stage->add_bq_codes(
                                    prepared_bq_codes.data(),
                                    batch_last - batch_first,
                                    args.add_threads);
                        } else if (args.online_graph_add == DynamicGraphAddMode::AdcBq) {
                            online_stage->add_adcbq_codes(
                                    prepared_adc_queries.data(),
                                    prepared_bq_codes.data(),
                                    batch_last - batch_first,
                                    args.add_threads);
                        } else if (args.online_graph_add == DynamicGraphAddMode::Tq1) {
                            online_stage->add_tq_codes(
                                    prepared_adc_queries.data(),
                                    prepared_bq_codes.data(),
                                    batch_last - batch_first,
                                    args.add_threads);
                        } else {
                            dynamic_add_selected(
                                    *online_stage,
                                    larger,
                                    selected,
                                    batch_first,
                                    batch_last,
                                    args.add_batch_size,
                                    args.add_threads,
                                    args.online_graph_add,
                                    "online publish rows",
                                    &fp_add_phase);
                        }
                        if (args.online_graph_add == DynamicGraphAddMode::Fp) {
                            dynamic_accumulate_build_phases(online_timing, fp_add_phase);
                        } else {
                            online_timing.graph_add_seconds += seconds_since(graph_start);
                        }
                        if (fp_append_store) {
                            const size_t expected =
                                    static_cast<size_t>(base.n) + batch_first;
                            if (published.physical_row_by_label.size() != expected) {
                                throw std::runtime_error(
                                        "published FP identity map is not contiguous");
                            }
                            for (size_t logical = expected;
                                 logical < static_cast<size_t>(base.n) + batch_last;
                                 ++logical) {
                                published.physical_row_by_label.push_back(
                                        static_cast<uint32_t>(logical));
                            }
                        } else {
                            for (size_t i = batch_first; i < batch_last; ++i) {
                                published.physical_row_by_label.push_back(
                                        static_cast<uint32_t>(selected[i]));
                            }
                        }
                        ++online_timing.publish_batches;
                        const double critical_seconds = seconds_since(critical_start);
                        online_timing.publish_critical_seconds += critical_seconds;
                        online_timing.publish_critical_max_seconds = std::max(
                                online_timing.publish_critical_max_seconds,
                                critical_seconds);
                    } catch (...) {
                        published.writer_pending.store(false, std::memory_order_release);
                        throw;
                    }
                    published.writer_pending.store(false, std::memory_order_release);
                    if (args.online_graph_add != DynamicGraphAddMode::Fp) {
                        std::cout << "  online publish rows "
                                  << (batch_last - add_first) << "/"
                                  << (static_cast<size_t>(insert_size) - add_first)
                                  << std::endl;
                    }
                }
                online_timing.add_seconds = seconds_since(start);

                if (fp_append_store &&
                    !args.fp_compacted_store_path.empty() &&
                    insert_size == max_insert) {
                    std::cout << "sorting final FP payload by HNSW graph locality" << std::endl;
                    double same_partition_edge_ratio = 0.0;
                    if (online_stage->has_fp_graph()) {
                        published.writer_pending.store(true, std::memory_order_release);
                        std::unique_lock<std::shared_mutex> graph_lock(published.mutex);
                        DynamicGraphSwapGuard graph_guard(*online_stage);
                        physical = build_colocation_physical_rows(
                                *online_stage->bq_view,
                                args.fp_sort_iterations,
                                args.fp_sort_partition_size,
                                static_cast<size_t>(base.d) * sizeof(float),
                                same_partition_edge_ratio);
                        graph_lock.unlock();
                        published.writer_pending.store(false, std::memory_order_release);
                    } else {
                        std::shared_lock<std::shared_mutex> graph_lock(published.mutex);
                        physical = build_colocation_physical_rows(
                                *online_stage->bq_view,
                                args.fp_sort_iterations,
                                args.fp_sort_partition_size,
                                static_cast<size_t>(base.d) * sizeof(float),
                                same_partition_edge_ratio);
                    }
                    const auto compaction_start = std::chrono::steady_clock::now();
                    fp_append_store->write_compacted(
                            args.fp_compacted_store_path,
                            physical);
                    const std::string compact_map_path =
                            args.fp_compacted_store_path + ".coloc.map";
                    write_colocation_map(
                            compact_map_path,
                            static_cast<int32_t>(physical.size()),
                            base.d,
                            args.fp_sort_iterations,
                            args.fp_sort_partition_size,
                            physical);
                    online_timing.fp_compaction_seconds = seconds_since(compaction_start);
                    online_timing.fp_layout = "graph_compacted";
                    compacted_fp_store =
                            std::make_unique<FbinMmap>(args.fp_compacted_store_path);
                    stage_fp_store = compacted_fp_store.get();
                    published.writer_pending.store(true, std::memory_order_release);
                    {
                        std::unique_lock<std::shared_mutex> publish_lock(published.mutex);
                        published.fp_store = stage_fp_store;
                        published.physical_row_by_label = physical;
                    }
                    published.writer_pending.store(false, std::memory_order_release);
                    std::cout << "  FP compaction seconds="
                              << online_timing.fp_compaction_seconds
                              << " same_partition_edge_ratio="
                              << same_partition_edge_ratio
                              << " map=" << compact_map_path << std::endl;
                }
                concurrent_search.finish(online_timing);
                physical = published.physical_row_by_label;
                stage_fp_store = published.fp_store;
                if (args.search_during_add) {
                    const double concurrent_qps =
                            online_timing.concurrent_search_wall_seconds > 0.0
                            ? static_cast<double>(online_timing.concurrent_search_queries) /
                                    online_timing.concurrent_search_wall_seconds
                            : 0.0;
                    std::cout << "  concurrent search queries="
                              << online_timing.concurrent_search_queries
                              << " qps=" << concurrent_qps
                              << " latency_ms="
                              << online_timing.concurrent_search_latency_ms
                              << " lock_wait_ms="
                              << online_timing.concurrent_search_lock_wait_ms
                              << std::endl;
                }
                dynamic_search_rows(
                        csv,
                        args,
                        *online_stage,
                        *stage_fp_store,
                        physical,
                        "online_add_no_rebuild",
                        insert_size,
                        queries,
                        nq,
                        calibration_queries,
                        calibration_nq,
                        stage_gt,
                        gt_dim,
                        base_gt,
                        base_gt_dim,
                        base_build_seconds,
                        online_timing,
                        args.freeze_online_calibration_from_base
                                ? &base_calibration_tau_by_ef
                                : nullptr,
                        nullptr,
                        chunked_fp_source);
            }
            previous_insert = insert_size;

            if (!args.skip_rebuild) {
                std::cout << "\nfull FP-HNSW/BQ rebuild insert=" << insert_size << std::endl;
                double rebuild_seconds = 0.0;
                DynamicBuildPhaseTiming rebuild_phase_timing;
                std::unique_ptr<DynamicAdcIndex> rebuilt = dynamic_build_stage(
                        args,
                        base,
                        larger,
                        selected,
                        insert_size,
                        rebuild_seconds,
                        rebuild_phase_timing);
                if (args.online_graph_add == DynamicGraphAddMode::Tq1) {
                    const auto tq_rebuild_start = std::chrono::steady_clock::now();
                    rebuilt->load_tq(args.tq_code_path, args.tq_meta_path);
                    for (size_t batch_first = 0;
                         batch_first < static_cast<size_t>(insert_size);
                         batch_first += static_cast<size_t>(args.add_batch_size)) {
                        const size_t batch_last = std::min(
                                static_cast<size_t>(insert_size),
                                batch_first + static_cast<size_t>(args.add_batch_size));
                        std::vector<float> tq_block = dynamic_copy_selected_fp_queries(
                                larger,
                                selected,
                                batch_first,
                                batch_last,
                                args.add_threads);
                        rebuilt->append_tq_block(tq_block, args.add_threads);
                    }
                    const double tq_rebuild_seconds = seconds_since(tq_rebuild_start);
                    rebuild_seconds += tq_rebuild_seconds;
                    rebuild_phase_timing.bq_encode_seconds += tq_rebuild_seconds;
                }
                DynamicTiming rebuild_timing;
                dynamic_accumulate_build_phases(rebuild_timing, rebuild_phase_timing);
                rebuild_timing.fp_layout = online_timing.fp_layout;
                rebuild_timing.graph_add_mode = "fp";
                dynamic_search_rows(
                        csv,
                        args,
                        *rebuilt,
                        *stage_fp_store,
                        physical,
                        "full_rebuild_oracle",
                        insert_size,
                        queries,
                        nq,
                        calibration_queries,
                        calibration_nq,
                        stage_gt,
                        gt_dim,
                        base_gt,
                        base_gt_dim,
                        rebuild_seconds,
                        rebuild_timing,
                        nullptr,
                        nullptr,
                        chunked_fp_source);
            }
        }

        std::cout << "saved " << args.out << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }
}
