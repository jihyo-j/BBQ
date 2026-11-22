#define main faiss_hnsw_lowbit_compare_main_disabled
#include "faiss_hnsw_lowbit_compare.cpp"
#undef main

#include <faiss/IndexPreTransform.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <liburing.h>

#include <atomic>
#include <list>
#include <mutex>
#include <unordered_map>

constexpr size_t kCodePageBytes = 4096;

enum class PagedMethod {
    AdcBQ1,
    SymBQ1,
    PQ8M128,
    OPQ8M128,
    PQ8D8,
    OPQ8D8,
    SQ4,
    SQ8,
    RaBitQ1,
    QdrantTQ1,
};

struct PagedArgs {
    std::string dataset;
    std::string query;
    std::string gt;
    std::string graph;
    std::string codec_index;
    std::string meta;
    std::string sidecar;
    std::string fp_sidecar;
    std::string out;
    std::string candidate_trace;
    std::string fp_io_mode = "sync";
    PagedMethod method = PagedMethod::AdcBQ1;
    Metric metric = Metric::IP;
    int k = 10;
    int max_queries = 1000;
    int rounds = 1;
    int threads = 1;
    int rerank_n = 0;
    bool adaptive_rerank = false;
    float adaptive_tau = 0.0f;
    unsigned fp_rerank_qd = 64;
    std::vector<int> efs{32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
    uint64_t bq_budget_bytes = 128000000;
    uint64_t pinned_metadata_bytes = 0;
    bool rebuild_sidecar = false;
    bool resident_codes = false;
    bool candidate_trace_only = false;
    bool query_e2e = false;
    bool fp_overlap_queries = false;
    bool inline_hnsw = false;
    std::string latency_out;
};

const char* paged_method_name(PagedMethod method) {
    switch (method) {
        case PagedMethod::AdcBQ1: return "ADC-BQ1";
        case PagedMethod::SymBQ1: return "Symmetric-BQ1";
        case PagedMethod::PQ8M128: return "Faiss-PQ8-M128";
        case PagedMethod::OPQ8M128: return "Faiss-OPQ-PQ8-M128";
        case PagedMethod::PQ8D8: return "Faiss-PQ8-d/8";
        case PagedMethod::OPQ8D8: return "Faiss-OPQ-PQ8-d/8";
        case PagedMethod::SQ4: return "Faiss-SQ4";
        case PagedMethod::SQ8: return "Faiss-SQ8";
        case PagedMethod::RaBitQ1: return "RaBitQ-1b-c16";
        case PagedMethod::QdrantTQ1: return "Qdrant-TQ1-Plus";
    }
    return "unknown";
}

PagedMethod parse_paged_method(const std::string& value) {
    if (value == "adcbq1" || value == "bq1") return PagedMethod::AdcBQ1;
    if (value == "symbq1" || value == "symbq") return PagedMethod::SymBQ1;
    if (value == "pq8m128" || value == "pq8") return PagedMethod::PQ8M128;
    if (value == "opq8m128" || value == "opq8") return PagedMethod::OPQ8M128;
    if (value == "pq8d8") return PagedMethod::PQ8D8;
    if (value == "opq8d8") return PagedMethod::OPQ8D8;
    if (value == "sq4") return PagedMethod::SQ4;
    if (value == "sq8") return PagedMethod::SQ8;
    if (value == "rabitq1" || value == "rabitq") return PagedMethod::RaBitQ1;
    if (value == "qdranttq1" || value == "tq1") return PagedMethod::QdrantTQ1;
    throw std::runtime_error("invalid --method");
}

[[noreturn]] void paged_usage(const char* argv0) {
    std::cerr
            << "Usage: " << argv0
            << " --dataset NAME --method METHOD --metric ip|l2"
            << " --query Q.fbin --gt G.ibin --graph GRAPH.faiss"
            << " --codec-index CODEC --sidecar CODES.pages --out OUT.csv"
            << " [--meta META] [--bq-budget-bytes 128000000]"
            << " [--pinned-metadata-bytes N] [--efs ...] [--rounds 1]"
            << " [--threads 1]"
            << " [--rerank-n 80 --fp-sidecar FP.aligned4k.fbin]"
            << " [--adaptive-rerank-tau TAU]"
            << " [--fp-io-mode sync|uring|uring-batch --fp-rerank-qd 64]"
            << " [--fp-overlap-queries]"
            << " [--inline-hnsw]"
            << " [--candidate-trace TRACE.csv [--candidate-trace-only]]"
            << " [--resident-codes]"
            << " [--query-e2e [--latency-out QUERY_LATENCY.csv]]"
            << " [--rebuild-sidecar]\n";
    std::exit(2);
}

PagedArgs parse_paged_args(int argc, char** argv) {
    PagedArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
            return argv[++i];
        };
        if (key == "--dataset") args.dataset = need();
        else if (key == "--method") args.method = parse_paged_method(need());
        else if (key == "--metric") args.metric = parse_metric(need());
        else if (key == "--query") args.query = need();
        else if (key == "--gt") args.gt = need();
        else if (key == "--graph") args.graph = need();
        else if (key == "--codec-index") args.codec_index = need();
        else if (key == "--meta") args.meta = need();
        else if (key == "--sidecar") args.sidecar = need();
        else if (key == "--fp-sidecar") args.fp_sidecar = need();
        else if (key == "--out") args.out = need();
        else if (key == "--latency-out") args.latency_out = need();
        else if (key == "--candidate-trace") args.candidate_trace = need();
        else if (key == "--fp-io-mode") args.fp_io_mode = need();
        else if (key == "--k") args.k = std::stoi(need());
        else if (key == "--max-queries") args.max_queries = std::stoi(need());
        else if (key == "--rounds") args.rounds = std::stoi(need());
        else if (key == "--threads") args.threads = std::stoi(need());
        else if (key == "--rerank-n") args.rerank_n = std::stoi(need());
        else if (key == "--adaptive-rerank-tau") {
            args.adaptive_rerank = true;
            args.adaptive_tau = std::stof(need());
        }
        else if (key == "--fp-rerank-qd") {
            args.fp_rerank_qd = static_cast<unsigned>(std::stoul(need()));
        }
        else if (key == "--bq-budget-bytes") {
            args.bq_budget_bytes = std::stoull(need());
        } else if (key == "--pinned-metadata-bytes") {
            args.pinned_metadata_bytes = std::stoull(need());
        } else if (key == "--rebuild-sidecar") {
            args.rebuild_sidecar = true;
        } else if (key == "--resident-codes") {
            args.resident_codes = true;
        } else if (key == "--candidate-trace-only") {
            args.candidate_trace_only = true;
        } else if (key == "--query-e2e") {
            args.query_e2e = true;
        } else if (key == "--fp-overlap-queries") {
            args.fp_overlap_queries = true;
        } else if (key == "--inline-hnsw") {
            args.inline_hnsw = true;
        } else if (key == "--efs") {
            args.efs.clear();
            while (i + 1 < argc && std::strncmp(argv[i + 1], "--", 2) != 0) {
                args.efs.push_back(std::stoi(argv[++i]));
            }
        } else if (key == "--help" || key == "-h") {
            paged_usage(argv[0]);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.dataset.empty() || args.query.empty() || args.gt.empty() ||
        args.graph.empty() || args.codec_index.empty() || args.sidecar.empty() ||
        args.out.empty()) {
        paged_usage(argv[0]);
    }
    if ((args.method == PagedMethod::RaBitQ1 ||
         args.method == PagedMethod::QdrantTQ1) && args.meta.empty()) {
        throw std::runtime_error("method requires --meta");
    }
    if (args.rerank_n > 0 && args.fp_sidecar.empty() && !args.candidate_trace_only) {
        throw std::runtime_error("--rerank-n requires --fp-sidecar");
    }
    if (args.candidate_trace_only && args.candidate_trace.empty()) {
        throw std::runtime_error("--candidate-trace-only requires --candidate-trace");
    }
    if (!args.candidate_trace.empty() && args.rerank_n <= 0) {
        throw std::runtime_error("--candidate-trace requires --rerank-n");
    }
    if (args.query_e2e && args.threads != 1) {
        throw std::runtime_error("--query-e2e requires --threads 1");
    }
    if (args.query_e2e && !args.candidate_trace.empty()) {
        throw std::runtime_error(
                "--query-e2e cannot be combined with --candidate-trace");
    }
    if (args.adaptive_rerank && args.rerank_n <= 0) {
        throw std::runtime_error("--adaptive-rerank-tau requires --rerank-n");
    }
    if (args.adaptive_rerank && args.query_e2e) {
        throw std::runtime_error(
                "--adaptive-rerank-tau currently supports bulk mode only");
    }
    if (args.fp_overlap_queries &&
        (args.threads != 1 || args.query_e2e || args.candidate_trace_only ||
         args.rerank_n <= 0 || !args.resident_codes ||
         args.fp_io_mode != "uring-batch")) {
        throw std::runtime_error(
                "--fp-overlap-queries requires T=1, resident codes, "
                "bulk reranking, and --fp-io-mode uring-batch");
    }
    if (args.inline_hnsw &&
        (!args.resident_codes || args.method != PagedMethod::QdrantTQ1)) {
        throw std::runtime_error(
                "--inline-hnsw currently requires resident Qdrant TQ1+ codes");
    }
    if (args.adaptive_rerank &&
        (!std::isfinite(args.adaptive_tau) || args.adaptive_tau < 0.0f)) {
        throw std::runtime_error("--adaptive-rerank-tau must be finite and nonnegative");
    }
    if (!args.query_e2e && !args.latency_out.empty()) {
        throw std::runtime_error("--latency-out requires --query-e2e");
    }
    if (args.fp_io_mode != "sync" && args.fp_io_mode != "uring" &&
        args.fp_io_mode != "uring-batch") {
        throw std::runtime_error(
                "--fp-io-mode must be sync or uring or uring-batch");
    }
    if (args.k <= 0 || args.max_queries <= 0 || args.rounds <= 0 ||
        args.threads <= 0 || args.rerank_n < 0 ||
        (args.rerank_n > 0 && args.rerank_n < args.k) ||
        args.efs.empty() || args.fp_rerank_qd == 0 ||
        args.bq_budget_bytes <= args.pinned_metadata_bytes) {
        throw std::runtime_error("invalid numeric configuration");
    }
    return args;
}

std::unique_ptr<faiss::IndexBinaryFlat> load_paged_binary_flat(
        const std::string& path) {
    faiss::IndexBinary* raw = faiss::read_index_binary(path.c_str());
    auto* flat = dynamic_cast<faiss::IndexBinaryFlat*>(raw);
    if (!flat) {
        delete raw;
        throw std::runtime_error("not an IndexBinaryFlat: " + path);
    }
    return std::unique_ptr<faiss::IndexBinaryFlat>(flat);
}

struct alignas(64) CodePageHeader {
    char magic[16];
    uint64_t version;
    uint64_t nb;
    uint64_t code_size;
    uint64_t slots_per_page;
    uint64_t data_pages;
    uint64_t page_bytes;
};

CodePageHeader expected_header(uint64_t nb, size_t code_size) {
    if (code_size == 0 || code_size > kCodePageBytes) {
        throw std::runtime_error("invalid code size for 4KB pages");
    }
    CodePageHeader header{};
    std::memcpy(header.magic, "FQCODEPAGE1", 11);
    header.version = 1;
    header.nb = nb;
    header.code_size = code_size;
    header.slots_per_page = kCodePageBytes / code_size;
    header.data_pages = (nb + header.slots_per_page - 1) / header.slots_per_page;
    header.page_bytes = kCodePageBytes;
    return header;
}

bool sidecar_matches(
        const std::string& path,
        const CodePageHeader& expected) {
    if (!fs::exists(path)) return false;
    std::ifstream input(path, std::ios::binary);
    CodePageHeader actual{};
    input.read(reinterpret_cast<char*>(&actual), sizeof(actual));
    if (!input || std::memcmp(actual.magic, expected.magic, sizeof(actual.magic)) != 0 ||
        actual.version != expected.version || actual.nb != expected.nb ||
        actual.code_size != expected.code_size ||
        actual.slots_per_page != expected.slots_per_page ||
        actual.data_pages != expected.data_pages ||
        actual.page_bytes != expected.page_bytes) {
        return false;
    }
    const uint64_t expected_bytes =
            kCodePageBytes + expected.data_pages * kCodePageBytes;
    return fs::file_size(path) == expected_bytes;
}

CodePageHeader ensure_sidecar(
        const PagedArgs& args,
        uint64_t nb,
        size_t code_size,
        const uint8_t* codes) {
    const CodePageHeader header = expected_header(nb, code_size);
    if (!args.rebuild_sidecar && sidecar_matches(args.sidecar, header)) {
        std::cout << "Reusing code sidecar " << args.sidecar << std::endl;
        return header;
    }
    if (!codes) throw std::runtime_error("codes unavailable for sidecar build");
    make_parent(args.sidecar);
    const std::string temporary = args.sidecar + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("open failed: " + temporary);
    std::vector<uint8_t> page(kCodePageBytes, 0);
    std::memcpy(page.data(), &header, sizeof(header));
    output.write(reinterpret_cast<const char*>(page.data()), page.size());
    uint64_t written = 0;
    for (uint64_t page_no = 0; page_no < header.data_pages; ++page_no) {
        std::fill(page.begin(), page.end(), 0);
        const uint64_t count = std::min<uint64_t>(
                header.slots_per_page, nb - written);
        std::memcpy(
                page.data(),
                codes + written * code_size,
                static_cast<size_t>(count) * code_size);
        output.write(reinterpret_cast<const char*>(page.data()), page.size());
        if (!output) throw std::runtime_error("write failed: " + temporary);
        written += count;
        if (page_no + 1 == header.data_pages || (page_no + 1) % 25000 == 0) {
            std::cout << "  sidecar pages " << page_no + 1 << '/'
                      << header.data_pages << std::endl;
        }
    }
    output.close();
    fs::rename(temporary, args.sidecar);
    return header;
}

struct AlignedPage {
    uint8_t* bytes = nullptr;
    AlignedPage() {
        void* pointer = nullptr;
        if (posix_memalign(&pointer, kCodePageBytes, kCodePageBytes) != 0) {
            throw std::bad_alloc();
        }
        bytes = static_cast<uint8_t*>(pointer);
    }
    ~AlignedPage() { std::free(bytes); }
};

struct PageCacheStats {
    uint64_t accesses = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t read_bytes = 0;
};

class CodePageCache {
    struct Entry {
        std::shared_ptr<AlignedPage> page;
        std::list<uint64_t>::iterator lru;
    };

    struct Shard {
        explicit Shard(size_t capacity_in) : capacity(capacity_in) {
            entries.reserve(capacity * 2);
        }
        size_t capacity;
        std::unordered_map<uint64_t, Entry> entries;
        std::list<uint64_t> lru;
        std::mutex mutex;
    };

    int fd_ = -1;
    CodePageHeader header_{};
    size_t capacity_pages_ = 0;
    std::vector<std::unique_ptr<Shard>> shards_;
    std::atomic<uint64_t> accesses_{0};
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};
    std::atomic<uint64_t> read_bytes_{0};

    std::shared_ptr<AlignedPage> fetch(uint64_t page_no) {
        accesses_.fetch_add(1, std::memory_order_relaxed);
        Shard& shard = *shards_[page_no % shards_.size()];
        std::lock_guard<std::mutex> guard(shard.mutex);
        auto found = shard.entries.find(page_no);
        if (found != shard.entries.end()) {
            hits_.fetch_add(1, std::memory_order_relaxed);
            shard.lru.splice(shard.lru.begin(), shard.lru, found->second.lru);
            return found->second.page;
        }
        misses_.fetch_add(1, std::memory_order_relaxed);
        auto page = std::make_shared<AlignedPage>();
        const off_t offset = static_cast<off_t>(
                kCodePageBytes + page_no * kCodePageBytes);
        size_t done = 0;
        while (done < kCodePageBytes) {
            const ssize_t got = ::pread(
                    fd_, page->bytes + done, kCodePageBytes - done, offset + done);
            if (got < 0) {
                throw std::runtime_error(
                        "O_DIRECT code read failed: " + std::string(std::strerror(errno)));
            }
            if (got == 0) throw std::runtime_error("short code page read");
            done += static_cast<size_t>(got);
        }
        read_bytes_.fetch_add(kCodePageBytes, std::memory_order_relaxed);
        if (shard.entries.size() >= shard.capacity) {
            const uint64_t victim = shard.lru.back();
            shard.lru.pop_back();
            shard.entries.erase(victim);
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }
        shard.lru.push_front(page_no);
        shard.entries.emplace(page_no, Entry{page, shard.lru.begin()});
        return page;
    }

public:
    CodePageCache(
            const std::string& path,
            const CodePageHeader& header,
            size_t capacity_pages)
            : header_(header),
              capacity_pages_(std::min<uint64_t>(capacity_pages, header.data_pages)) {
        if (capacity_pages_ == 0) throw std::runtime_error("zero-page code cache");
        fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error(
                    "O_DIRECT open failed: " + path + ": " + std::strerror(errno));
        }
        const size_t shard_count = std::min<size_t>(64, capacity_pages_);
        const size_t base_capacity = capacity_pages_ / shard_count;
        const size_t remainder = capacity_pages_ % shard_count;
        shards_.reserve(shard_count);
        for (size_t i = 0; i < shard_count; ++i) {
            shards_.push_back(std::make_unique<Shard>(
                    base_capacity + (i < remainder ? 1 : 0)));
        }
    }

    ~CodePageCache() {
        if (fd_ >= 0) ::close(fd_);
    }

    template <typename Function>
    float with_code(faiss::idx_t id, Function&& function) {
        if (id < 0 || static_cast<uint64_t>(id) >= header_.nb) {
            return std::numeric_limits<float>::infinity();
        }
        const uint64_t page_no = static_cast<uint64_t>(id) / header_.slots_per_page;
        const uint64_t slot = static_cast<uint64_t>(id) % header_.slots_per_page;
        auto page = fetch(page_no);
        return function(page->bytes + slot * header_.code_size);
    }

    PageCacheStats stats() const {
        return PageCacheStats{
                accesses_.load(std::memory_order_relaxed),
                hits_.load(std::memory_order_relaxed),
                misses_.load(std::memory_order_relaxed),
                evictions_.load(std::memory_order_relaxed),
                read_bytes_.load(std::memory_order_relaxed)};
    }
    size_t capacity_pages() const { return capacity_pages_; }
    size_t resident_pages() const {
        size_t total = 0;
        for (const auto& shard : shards_) total += shard->entries.size();
        return total;
    }
};

struct CodeEvaluator {
    virtual void set_query(const float* query) = 0;
    virtual float distance_to_code(const uint8_t* code) = 0;
    virtual void distances_batch_4(
            const uint8_t* code0,
            const uint8_t* code1,
            const uint8_t* code2,
            const uint8_t* code3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) {
        dis0 = distance_to_code(code0);
        dis1 = distance_to_code(code1);
        dis2 = distance_to_code(code2);
        dis3 = distance_to_code(code3);
    }
    virtual ~CodeEvaluator() = default;
};

struct FaissCodeEvaluator : CodeEvaluator {
    std::unique_ptr<faiss::FlatCodesDistanceComputer> computer;
    const faiss::IndexPreTransform* pretransform;
    bool negate;
    std::unique_ptr<const float[]> transformed;

    FaissCodeEvaluator(
            faiss::IndexFlatCodes* index,
            const faiss::IndexPreTransform* pretransform_in,
            bool negate_in)
            : computer(index->get_FlatCodesDistanceComputer()),
              pretransform(pretransform_in),
              negate(negate_in) {}

    void set_query(const float* query) override {
        const float* values = query;
        if (pretransform) {
            const float* output = pretransform->apply_chain(1, query);
            if (output != query) {
                transformed.reset(output);
                values = output;
            } else {
                transformed.reset();
            }
        }
        computer->set_query(values);
    }

    float distance_to_code(const uint8_t* code) override {
        const float value = computer->distance_to_code(code);
        return negate ? -value : value;
    }
};

template <typename Computer>
struct CustomCodeEvaluator : CodeEvaluator {
    Computer computer;

    template <typename... Arguments>
    explicit CustomCodeEvaluator(Arguments&&... arguments)
            : computer(std::forward<Arguments>(arguments)...) {}

    void set_query(const float* query) override { computer.set_query(query); }
    float distance_to_code(const uint8_t* code) override {
        return computer.distance_to_code(code);
    }
};

// The reference TurboQuant scorer evaluates one database code at a time.  HNSW
// exposes four independent neighbors together, so reuse each loaded query plane
// across four codes.  Besides removing redundant query-plane loads, the four
// independent VNNI dependency chains give the CPU enough instruction-level
// parallelism to hide most of the dot-product latency.
struct QdrantTqBatchedCodeEvaluator : CodeEvaluator {
    QdrantTqDirectDistanceComputer computer;

    QdrantTqBatchedCodeEvaluator(
            const QdrantTqIndex* index,
            size_t code_size,
            int d,
            Metric metric)
            : computer(index, nullptr, code_size, d, metric) {}

    void set_query(const float* query) override { computer.set_query(query); }

    float distance_to_code(const uint8_t* code) override {
        return computer.distance_to_code(code);
    }

    __attribute__((target("avx512f,avx512bw,avx512vnni")))
    static void raw_dot_batch_4_avx512_vnni(
            const int8_t* planes,
            size_t plane_len,
            const uint8_t* code0,
            const uint8_t* code1,
            const uint8_t* code2,
            const uint8_t* code3,
            size_t code_len,
            int64_t output[4]) {
        alignas(16) static constexpr uint8_t codebook_bytes[16] = {
                0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const __m128i codebook_128 = _mm_load_si128(
                reinterpret_cast<const __m128i*>(codebook_bytes));
        const __m512i codebook = _mm512_broadcast_i32x4(codebook_128);
        const __m512i mask = _mm512_set1_epi8(1);

        __m512i low00 = _mm512_setzero_si512();
        __m512i low01 = _mm512_setzero_si512();
        __m512i high00 = _mm512_setzero_si512();
        __m512i high01 = _mm512_setzero_si512();
        __m512i low10 = _mm512_setzero_si512();
        __m512i low11 = _mm512_setzero_si512();
        __m512i high10 = _mm512_setzero_si512();
        __m512i high11 = _mm512_setzero_si512();
        __m512i low20 = _mm512_setzero_si512();
        __m512i low21 = _mm512_setzero_si512();
        __m512i high20 = _mm512_setzero_si512();
        __m512i high21 = _mm512_setzero_si512();
        __m512i low30 = _mm512_setzero_si512();
        __m512i low31 = _mm512_setzero_si512();
        __m512i high30 = _mm512_setzero_si512();
        __m512i high31 = _mm512_setzero_si512();

        const size_t blocks = code_len / 64;
        for (size_t block = 0; block < blocks; ++block) {
            const size_t offset = block * 64;
            __m512i shifted0 = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(code0 + offset));
            __m512i shifted1 = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(code1 + offset));
            __m512i shifted2 = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(code2 + offset));
            __m512i shifted3 = _mm512_loadu_si512(
                    reinterpret_cast<const void*>(code3 + offset));
            for (size_t k = 0; k < 8; ++k) {
                const __m512i query_low = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(
                                planes + k * plane_len + offset));
                const __m512i query_high = _mm512_loadu_si512(
                        reinterpret_cast<const void*>(
                                planes + (8 + k) * plane_len + offset));
                const __m512i values0 = _mm512_shuffle_epi8(
                        codebook, _mm512_and_si512(shifted0, mask));
                const __m512i values1 = _mm512_shuffle_epi8(
                        codebook, _mm512_and_si512(shifted1, mask));
                const __m512i values2 = _mm512_shuffle_epi8(
                        codebook, _mm512_and_si512(shifted2, mask));
                const __m512i values3 = _mm512_shuffle_epi8(
                        codebook, _mm512_and_si512(shifted3, mask));
                if ((k & 1) == 0) {
                    low00 = _mm512_dpbusd_epi32(low00, values0, query_low);
                    high00 = _mm512_dpbusd_epi32(high00, values0, query_high);
                    low10 = _mm512_dpbusd_epi32(low10, values1, query_low);
                    high10 = _mm512_dpbusd_epi32(high10, values1, query_high);
                    low20 = _mm512_dpbusd_epi32(low20, values2, query_low);
                    high20 = _mm512_dpbusd_epi32(high20, values2, query_high);
                    low30 = _mm512_dpbusd_epi32(low30, values3, query_low);
                    high30 = _mm512_dpbusd_epi32(high30, values3, query_high);
                } else {
                    low01 = _mm512_dpbusd_epi32(low01, values0, query_low);
                    high01 = _mm512_dpbusd_epi32(high01, values0, query_high);
                    low11 = _mm512_dpbusd_epi32(low11, values1, query_low);
                    high11 = _mm512_dpbusd_epi32(high11, values1, query_high);
                    low21 = _mm512_dpbusd_epi32(low21, values2, query_low);
                    high21 = _mm512_dpbusd_epi32(high21, values2, query_high);
                    low31 = _mm512_dpbusd_epi32(low31, values3, query_low);
                    high31 = _mm512_dpbusd_epi32(high31, values3, query_high);
                }
                shifted0 = _mm512_srli_epi16(shifted0, 1);
                shifted1 = _mm512_srli_epi16(shifted1, 1);
                shifted2 = _mm512_srli_epi16(shifted2, 1);
                shifted3 = _mm512_srli_epi16(shifted3, 1);
            }
        }

        const auto finish = [](__m512i low0,
                               __m512i low1,
                               __m512i high0,
                               __m512i high1) -> int64_t {
            const int64_t low = static_cast<int64_t>(
                    _mm512_reduce_add_epi32(_mm512_add_epi32(low0, low1)));
            const int64_t high = static_cast<int64_t>(
                    _mm512_reduce_add_epi32(_mm512_add_epi32(high0, high1)));
            return low + 256 * high;
        };
        output[0] = finish(low00, low01, high00, high01);
        output[1] = finish(low10, low11, high10, high11);
        output[2] = finish(low20, low21, high20, high21);
        output[3] = finish(low30, low31, high30, high31);
    }

    float finish_distance(const uint8_t* code, int64_t raw) const {
        const int64_t centered_raw = raw - computer.bias_correction;
        const float reconstructed_dot =
                computer.postprocess_scale * static_cast<float>(centered_raw);
        const float dot = reconstructed_dot + computer.ec_correction;
        const float vector_scale = read_float(code + computer.code_bytes);
        if (computer.metric == Metric::IP) return -(dot * vector_scale);
        const float vector_l2 =
                read_float(code + computer.code_bytes + sizeof(float));
        return computer.query_l2 * computer.query_l2 + vector_l2 * vector_l2 -
                2.0f * dot * vector_scale;
    }

    void distances_batch_4(
            const uint8_t* code0,
            const uint8_t* code1,
            const uint8_t* code2,
            const uint8_t* code3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        // All benchmark dimensions are multiples of 512.  Keep the scalar
        // reference for unusual tails rather than putting copies/predication
        // in the hot path.
        if ((computer.code_bytes & 63) != 0 ||
            !__builtin_cpu_supports("avx512f") ||
            !__builtin_cpu_supports("avx512bw") ||
            !__builtin_cpu_supports("avx512vnni")) {
            CodeEvaluator::distances_batch_4(
                    code0, code1, code2, code3, dis0, dis1, dis2, dis3);
            return;
        }
        int64_t raw[4];
        raw_dot_batch_4_avx512_vnni(
                computer.query_planes.data(),
                computer.query_plane_len,
                code0,
                code1,
                code2,
                code3,
                computer.code_bytes,
                raw);
        dis0 = finish_distance(code0, raw[0]);
        dis1 = finish_distance(code1, raw[1]);
        dis2 = finish_distance(code2, raw[2]);
        dis3 = finish_distance(code3, raw[3]);
    }
};

struct PagedCodeDistanceComputer : faiss::DistanceComputer {
    CodePageCache& cache;
    std::unique_ptr<CodeEvaluator> evaluator;

    PagedCodeDistanceComputer(
            CodePageCache& cache_in,
            std::unique_ptr<CodeEvaluator> evaluator_in)
            : cache(cache_in), evaluator(std::move(evaluator_in)) {}

    void set_query(const float* query) override { evaluator->set_query(query); }

    float operator()(faiss::idx_t id) override {
        return cache.with_code(
                id,
                [&](const uint8_t* code) {
                    return evaluator->distance_to_code(code);
                });
    }

    float symmetric_dis(faiss::idx_t, faiss::idx_t) override {
        return 0.0f;
    }
};

struct ResidentCodeDistanceComputer : faiss::DistanceComputer {
    const uint8_t* codes;
    size_t code_size;
    uint64_t nb;
    std::unique_ptr<CodeEvaluator> evaluator;
    uint64_t distance_evaluations = 0;
    uint64_t upper_distance_evaluations = 0;
    uint64_t base_distance_evaluations = 0;
    uint64_t scalar_distance_calls = 0;
    uint64_t batch4_calls = 0;

    ResidentCodeDistanceComputer(
            const uint8_t* codes_in,
            size_t code_size_in,
            uint64_t nb_in,
            std::unique_ptr<CodeEvaluator> evaluator_in)
            : codes(codes_in),
              code_size(code_size_in),
              nb(nb_in),
              evaluator(std::move(evaluator_in)) {
        if (!codes) throw std::runtime_error("resident codes unavailable");
    }

    void set_query(const float* query) override { evaluator->set_query(query); }

    float operator()(faiss::idx_t id) override {
        ++distance_evaluations;
        ++scalar_distance_calls;
        if (id < 0 || static_cast<uint64_t>(id) >= nb) {
            return std::numeric_limits<float>::infinity();
        }
        return evaluator->distance_to_code(
                codes + static_cast<size_t>(id) * code_size);
    }

    void distances_batch_4(
            faiss::idx_t id0,
            faiss::idx_t id1,
            faiss::idx_t id2,
            faiss::idx_t id3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        distance_evaluations += 4;
        ++batch4_calls;
        evaluator->distances_batch_4(
                codes + static_cast<size_t>(id0) * code_size,
                codes + static_cast<size_t>(id1) * code_size,
                codes + static_cast<size_t>(id2) * code_size,
                codes + static_cast<size_t>(id3) * code_size,
                dis0,
                dis1,
                dis2,
                dis3);
    }

    float distance_idx(faiss::idx_t id) { return (*this)(id); }

    float distance_idx_pruned(faiss::idx_t id, float) {
        return (*this)(id);
    }

    void prefetch_idx(faiss::idx_t id) const {
        if (id < 0 || static_cast<uint64_t>(id) >= nb) return;
        const char* code = reinterpret_cast<const char*>(
                codes + static_cast<size_t>(id) * code_size);
        _mm_prefetch(code, _MM_HINT_T0);
        if (code_size > 64) _mm_prefetch(code + 64, _MM_HINT_T0);
        if (code_size > 128) _mm_prefetch(code + 128, _MM_HINT_T0);
    }

    void distances_batch_8(const faiss::idx_t* ids, float* distances) {
        distances_batch_4(
                ids[0], ids[1], ids[2], ids[3],
                distances[0], distances[1], distances[2], distances[3]);
        distances_batch_4(
                ids[4], ids[5], ids[6], ids[7],
                distances[4], distances[5], distances[6], distances[7]);
    }

    void distances_batch_16(const faiss::idx_t* ids, float* distances) {
        distances_batch_8(ids, distances);
        distances_batch_8(ids + 8, distances + 8);
    }

    void distances_batch_32(const faiss::idx_t* ids, float* distances) {
        distances_batch_16(ids, distances);
        distances_batch_16(ids + 16, distances + 16);
    }

    void distances_batch_n(
            const faiss::idx_t* ids,
            int count,
            float* distances) {
        int offset = 0;
        for (; offset + 4 <= count; offset += 4) {
            distances_batch_4(
                    ids[offset], ids[offset + 1], ids[offset + 2], ids[offset + 3],
                    distances[offset], distances[offset + 1],
                    distances[offset + 2], distances[offset + 3]);
        }
        for (; offset < count; ++offset) distances[offset] = (*this)(ids[offset]);
    }

    float symmetric_dis(faiss::idx_t, faiss::idx_t) override { return 0.0f; }
};

inline bool inline_minimax_push(
        faiss::MinimaxHeap& heap,
        faiss::MinimaxHeap::storage_idx_t id,
        float distance) {
    using HC = faiss::MinimaxHeap::HC;
    if (heap.k == heap.n) {
        if (distance >= heap.dis[0]) return false;
        if (heap.ids[0] != -1) --heap.nvalid;
        faiss::heap_pop<HC>(heap.k--, heap.dis.data(), heap.ids.data());
    }
    faiss::heap_push<HC>(
            ++heap.k, heap.dis.data(), heap.ids.data(), distance, id);
    ++heap.nvalid;
    return true;
}

inline int inline_minimax_count_below(
        const faiss::MinimaxHeap& heap,
        float threshold) {
    const float* distances = heap.dis.data();
    const int count = heap.k;
    int below = 0;
    int offset = 0;
#if defined(__AVX512F__)
    const __m512 values = _mm512_set1_ps(threshold);
    for (; offset + 16 <= count; offset += 16) {
        const __mmask16 mask = _mm512_cmp_ps_mask(
                _mm512_loadu_ps(distances + offset), values, _CMP_LT_OQ);
        below += __builtin_popcount(static_cast<unsigned>(mask));
    }
#elif defined(__AVX2__)
    const __m256 values = _mm256_set1_ps(threshold);
    for (; offset + 8 <= count; offset += 8) {
        const __m256 comparison = _mm256_cmp_ps(
                _mm256_loadu_ps(distances + offset), values, _CMP_LT_OQ);
        below += __builtin_popcount(
                static_cast<unsigned>(_mm256_movemask_ps(comparison)));
    }
#endif
    for (; offset < count; ++offset) below += distances[offset] < threshold;
    return below;
}

void inline_greedy_update_nearest(
        const faiss::HNSW& hnsw,
        ResidentCodeDistanceComputer& computer,
        int level,
        faiss::HNSW::storage_idx_t& nearest,
        float& nearest_distance) {
    for (;;) {
        const auto previous = nearest;
        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        faiss::idx_t ids[16];
        int count = 0;
        auto update = [&](faiss::idx_t id, float distance) {
            if (distance < nearest_distance) {
                nearest = static_cast<faiss::HNSW::storage_idx_t>(id);
                nearest_distance = distance;
            }
        };
        for (size_t offset = begin; offset < end; ++offset) {
            const faiss::idx_t id = hnsw.neighbors[offset];
            if (id < 0) break;
            computer.prefetch_idx(id);
            ids[count++] = id;
            if (count == 16) {
                float distances[16];
                computer.distances_batch_16(ids, distances);
                for (int i = 0; i < 16; ++i) update(ids[i], distances[i]);
                count = 0;
            }
        }
        int offset = 0;
        for (; offset + 8 <= count; offset += 8) {
            float distances[8];
            computer.distances_batch_8(ids + offset, distances);
            for (int i = 0; i < 8; ++i) {
                update(ids[offset + i], distances[i]);
            }
        }
        for (; offset < count; ++offset) {
            update(ids[offset], computer.distance_idx(ids[offset]));
        }
        if (nearest == previous) return;
    }
}

void search_one_inline_hnsw(
        const faiss::IndexBinaryHNSW& index,
        ResidentCodeDistanceComputer& computer,
        faiss::VisitedTable& visited,
        faiss::MinimaxHeap& candidates,
        int k,
        int ef,
        float* result_distances,
        faiss::idx_t* result_labels) {
    using C = faiss::HNSW::C;
    faiss::heap_heapify<C>(k, result_distances, result_labels);
    const faiss::HNSW& hnsw = index.hnsw;
    faiss::HNSW::storage_idx_t nearest = hnsw.entry_point;
    if (nearest < 0) {
        faiss::heap_reorder<C>(k, result_distances, result_labels);
        visited.advance();
        return;
    }

    const uint64_t before_query_evaluations = computer.distance_evaluations;
    float nearest_distance = computer.distance_idx(nearest);
    const uint64_t after_entry_evaluations = computer.distance_evaluations;
    for (int level = hnsw.max_level; level >= 1; --level) {
        inline_greedy_update_nearest(
                hnsw, computer, level, nearest, nearest_distance);
    }
    const uint64_t after_upper_evaluations = computer.distance_evaluations;

    const int search_ef = std::max(ef, k);
    candidates.n = search_ef;
    candidates.clear();
    candidates.n = search_ef;
    inline_minimax_push(candidates, nearest, nearest_distance);
    float threshold = result_distances[0];
    if (nearest_distance < threshold) {
        faiss::heap_replace_top<C>(
                k, result_distances, result_labels, nearest_distance, nearest);
        threshold = result_distances[0];
    }
    visited.set(static_cast<size_t>(nearest));

    while (candidates.size() > 0) {
        float current_distance = 0.0f;
        const int current = candidates.pop_min(&current_distance);
        if (inline_minimax_count_below(candidates, current_distance) >= search_ef) {
            break;
        }

        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(current, 0, &begin, &end);
        faiss::idx_t ids[32];
        int count = 0;
        auto add = [&](faiss::idx_t id, float distance) {
            if (distance < threshold) {
                faiss::heap_replace_top<C>(
                        k, result_distances, result_labels, distance, id);
                threshold = result_distances[0];
            }
            inline_minimax_push(candidates, id, distance);
        };

        for (size_t offset = begin; offset < end; ++offset) {
            const faiss::idx_t id = hnsw.neighbors[offset];
            if (id < 0) break;
            const size_t lookahead = offset + 16;
            if (lookahead < end) {
                const faiss::idx_t future = hnsw.neighbors[lookahead];
                if (future >= 0) {
                    visited.prefetch(static_cast<size_t>(future));
                    computer.prefetch_idx(future);
                }
            }
            if (visited.set(static_cast<size_t>(id))) {
                ids[count++] = id;
                computer.prefetch_idx(id);
                if (count == 32) {
                    float distances[32];
                    computer.distances_batch_32(ids, distances);
                    for (int i = 0; i < 32; ++i) add(ids[i], distances[i]);
                    count = 0;
                }
            }
        }
        int offset = 0;
        for (; offset + 16 <= count; offset += 16) {
            float distances[16];
            computer.distances_batch_16(ids + offset, distances);
            for (int i = 0; i < 16; ++i) add(ids[offset + i], distances[i]);
        }
        for (; offset + 8 <= count; offset += 8) {
            float distances[8];
            computer.distances_batch_8(ids + offset, distances);
            for (int i = 0; i < 8; ++i) add(ids[offset + i], distances[i]);
        }
        for (; offset < count; ++offset) {
            add(ids[offset], computer.distance_idx(ids[offset]));
        }
    }

    faiss::heap_reorder<C>(k, result_distances, result_labels);
    computer.upper_distance_evaluations +=
            after_upper_evaluations - after_entry_evaluations;
    computer.base_distance_evaluations +=
            (after_entry_evaluations - before_query_evaluations) +
            (computer.distance_evaluations - after_upper_evaluations);
    visited.advance();
}

using EvaluatorFactory = std::function<std::unique_ptr<CodeEvaluator>()>;

struct BenchmarkInputs {
    size_t code_size = 0;
    CodePageHeader sidecar_header{};
    const uint8_t* resident_codes = nullptr;
    EvaluatorFactory make_evaluator;
};

struct AdaptivePrefixStats {
    std::vector<faiss::idx_t> labels;
    double mean = 0.0;
    int minimum = 0;
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    int maximum = 0;
};

double count_percentile(const std::vector<int>& sorted, double quantile) {
    if (sorted.empty()) return 0.0;
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = std::min(sorted.size() - 1, lower + 1);
    const double fraction = position - static_cast<double>(lower);
    return static_cast<double>(sorted[lower]) * (1.0 - fraction) +
            static_cast<double>(sorted[upper]) * fraction;
}

AdaptivePrefixStats select_adaptive_prefix(
        const std::vector<faiss::idx_t>& labels,
        const std::vector<float>& distances,
        int nq,
        int candidate_k,
        int k,
        float tau) {
    AdaptivePrefixStats result;
    result.labels = labels;
    std::vector<int> counts(static_cast<size_t>(nq), 0);
    for (int qi = 0; qi < nq; ++qi) {
        const size_t base = static_cast<size_t>(qi) * candidate_k;
        int valid = 0;
        while (valid < candidate_k && labels[base + static_cast<size_t>(valid)] >= 0 &&
               std::isfinite(distances[base + static_cast<size_t>(valid)])) {
            ++valid;
        }
        const int minimum = std::min(k, valid);
        int take = minimum;
        if (take > 0) {
            const float cutoff =
                    distances[base + static_cast<size_t>(minimum - 1)] + tau;
            while (take < valid &&
                   distances[base + static_cast<size_t>(take)] <= cutoff) {
                ++take;
            }
        }
        counts[static_cast<size_t>(qi)] = take;
        for (int rank = take; rank < candidate_k; ++rank) {
            result.labels[base + static_cast<size_t>(rank)] = -1;
        }
    }
    if (!counts.empty()) {
        result.minimum = *std::min_element(counts.begin(), counts.end());
        result.maximum = *std::max_element(counts.begin(), counts.end());
        result.mean = static_cast<double>(
                              std::accumulate(counts.begin(), counts.end(), int64_t{0})) /
                static_cast<double>(counts.size());
        std::sort(counts.begin(), counts.end());
        result.p50 = count_percentile(counts, 0.50);
        result.p95 = count_percentile(counts, 0.95);
        result.p99 = count_percentile(counts, 0.99);
    }
    return result;
}

class FpDirectStore {
    int fd_ = -1;
    uint64_t rows_ = 0;
    int dim_ = 0;
    size_t row_bytes_ = 0;
    size_t stride_bytes_ = 0;
    std::atomic<uint64_t> reads_{0};
    std::atomic<uint64_t> read_bytes_{0};

public:
    FpDirectStore(const std::string& path, uint64_t expected_rows, int expected_dim)
            : rows_(expected_rows), dim_(expected_dim) {
        std::ifstream header(path, std::ios::binary);
        int32_t rows = 0;
        int32_t dim = 0;
        header.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        header.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!header || rows != static_cast<int64_t>(expected_rows) ||
            dim != expected_dim) {
            throw std::runtime_error("aligned FP sidecar header mismatch: " + path);
        }
        row_bytes_ = static_cast<size_t>(dim_) * sizeof(float);
        stride_bytes_ =
                ((row_bytes_ + kCodePageBytes - 1) / kCodePageBytes) * kCodePageBytes;
        const uint64_t expected_bytes = kCodePageBytes + rows_ * stride_bytes_;
        if (fs::file_size(path) != expected_bytes) {
            throw std::runtime_error("aligned FP sidecar size mismatch: " + path);
        }
        fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error(
                    "O_DIRECT FP open failed: " + path + ": " + std::strerror(errno));
        }
    }

    ~FpDirectStore() {
        if (fd_ >= 0) ::close(fd_);
    }

    size_t stride_bytes() const { return stride_bytes_; }
    uint64_t reads() const { return reads_.load(std::memory_order_relaxed); }
    uint64_t read_bytes() const {
        return read_bytes_.load(std::memory_order_relaxed);
    }

    int fd() const { return fd_; }
    off_t offset(faiss::idx_t id) const {
        if (id < 0 || static_cast<uint64_t>(id) >= rows_) {
            throw std::runtime_error("FP rerank id out of range");
        }
        return static_cast<off_t>(
                kCodePageBytes + static_cast<uint64_t>(id) * stride_bytes_);
    }
    void record_reads(uint64_t count) {
        reads_.fetch_add(count, std::memory_order_relaxed);
        read_bytes_.fetch_add(count * stride_bytes_, std::memory_order_relaxed);
    }

    void read(faiss::idx_t id, uint8_t* aligned_buffer) {
        if (id < 0 || static_cast<uint64_t>(id) >= rows_) {
            throw std::runtime_error("FP rerank id out of range");
        }
        const off_t offset = static_cast<off_t>(
                kCodePageBytes + static_cast<uint64_t>(id) * stride_bytes_);
        size_t done = 0;
        while (done < stride_bytes_) {
            const ssize_t got = ::pread(
                    fd_, aligned_buffer + done, stride_bytes_ - done, offset + done);
            if (got < 0) {
                throw std::runtime_error(
                        "O_DIRECT FP read failed: " + std::string(std::strerror(errno)));
            }
            if (got == 0) throw std::runtime_error("short O_DIRECT FP read");
            done += static_cast<size_t>(got);
        }
        reads_.fetch_add(1, std::memory_order_relaxed);
        read_bytes_.fetch_add(stride_bytes_, std::memory_order_relaxed);
    }
};

void rerank_fp_direct(
        FpDirectStore& store,
        const std::vector<float>& queries,
        int nq,
        int dim,
        int candidate_k,
        int k,
        int threads,
        Metric metric,
        const std::vector<faiss::idx_t>& candidate_labels,
        std::vector<float>& final_distances,
        std::vector<faiss::idx_t>& final_labels) {
    final_distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    final_labels.assign(static_cast<size_t>(nq) * k, -1);
    omp_set_num_threads(threads);
#pragma omp parallel
    {
        void* allocation = nullptr;
        if (posix_memalign(&allocation, kCodePageBytes, store.stride_bytes()) != 0) {
            throw std::bad_alloc();
        }
        std::unique_ptr<uint8_t, decltype(&std::free)> buffer(
                static_cast<uint8_t*>(allocation), &std::free);
        std::vector<std::pair<float, faiss::idx_t>> scored;
        scored.reserve(candidate_k);
#pragma omp for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            scored.clear();
            const float* query = queries.data() + static_cast<size_t>(qi) * dim;
            for (int rank = 0; rank < candidate_k; ++rank) {
                const faiss::idx_t id =
                        candidate_labels[static_cast<size_t>(qi) * candidate_k + rank];
                if (id < 0) continue;
                store.read(id, buffer.get());
                const float* vector = reinterpret_cast<const float*>(buffer.get());
                float distance = 0.0f;
                if (metric == Metric::IP) {
                    for (int j = 0; j < dim; ++j) distance -= query[j] * vector[j];
                } else {
                    for (int j = 0; j < dim; ++j) {
                        const float delta = query[j] - vector[j];
                        distance += delta * delta;
                    }
                }
                scored.emplace_back(distance, id);
            }
            const size_t keep = std::min<size_t>(k, scored.size());
            std::partial_sort(
                    scored.begin(),
                    scored.begin() + keep,
                    scored.end(),
                    [](const auto& left, const auto& right) {
                        if (left.first != right.first) return left.first < right.first;
                        return left.second < right.second;
                    });
            for (size_t rank = 0; rank < keep; ++rank) {
                final_distances[static_cast<size_t>(qi) * k + rank] = scored[rank].first;
                final_labels[static_cast<size_t>(qi) * k + rank] = scored[rank].second;
            }
        }
    }
}

void rerank_fp_uring(
        FpDirectStore& store,
        const std::vector<float>& queries,
        int nq,
        int dim,
        int candidate_k,
        int k,
        int threads,
        unsigned queue_depth,
        Metric metric,
        const std::vector<faiss::idx_t>& candidate_labels,
        std::vector<float>& final_distances,
        std::vector<faiss::idx_t>& final_labels) {
    struct Request {
        int query = 0;
        faiss::idx_t id = -1;
    };
    std::vector<Request> requests;
    requests.reserve(static_cast<size_t>(nq) * candidate_k);
    for (int qi = 0; qi < nq; ++qi) {
        for (int rank = 0; rank < candidate_k; ++rank) {
            const faiss::idx_t id =
                    candidate_labels[static_cast<size_t>(qi) * candidate_k + rank];
            if (id >= 0) requests.push_back({qi, id});
        }
    }

    std::vector<std::vector<std::pair<float, faiss::idx_t>>> scored(nq);
    for (auto& values : scored) values.reserve(candidate_k);

    io_uring ring{};
    const int init = io_uring_queue_init(queue_depth, &ring, 0);
    if (init < 0) {
        throw std::runtime_error(
                "io_uring_queue_init failed: " + std::to_string(init));
    }
    void* allocation = nullptr;
    const size_t stride = store.stride_bytes();
    if (posix_memalign(
                &allocation, kCodePageBytes,
                static_cast<size_t>(queue_depth) * stride) != 0) {
        io_uring_queue_exit(&ring);
        throw std::bad_alloc();
    }

    for (size_t begin = 0; begin < requests.size(); begin += queue_depth) {
        const unsigned count = static_cast<unsigned>(
                std::min<size_t>(queue_depth, requests.size() - begin));
        for (unsigned slot = 0; slot < count; ++slot) {
            const Request& request = requests[begin + slot];
            io_uring_sqe* sqe = io_uring_get_sqe(&ring);
            if (!sqe) throw std::runtime_error("io_uring FP SQ exhausted");
            auto* buffer = static_cast<uint8_t*>(allocation) +
                    static_cast<size_t>(slot) * stride;
            io_uring_prep_read(
                    sqe, store.fd(), buffer, static_cast<unsigned>(stride),
                    store.offset(request.id));
            io_uring_sqe_set_data64(sqe, slot);
        }
        const int submitted = io_uring_submit(&ring);
        if (submitted != static_cast<int>(count)) {
            throw std::runtime_error("short io_uring FP submit");
        }
        for (unsigned done = 0; done < count; ++done) {
            io_uring_cqe* cqe = nullptr;
            const int wait = io_uring_wait_cqe(&ring, &cqe);
            if (wait < 0 || !cqe) throw std::runtime_error("io_uring FP wait failed");
            const unsigned slot =
                    static_cast<unsigned>(io_uring_cqe_get_data64(cqe));
            const int actual = cqe->res;
            io_uring_cqe_seen(&ring, cqe);
            if (actual != static_cast<int>(stride)) {
                throw std::runtime_error(
                        "short io_uring FP read: " + std::to_string(actual));
            }
            const Request& request = requests[begin + slot];
            const float* query = queries.data() +
                    static_cast<size_t>(request.query) * dim;
            const float* vector = reinterpret_cast<const float*>(
                    static_cast<uint8_t*>(allocation) +
                    static_cast<size_t>(slot) * stride);
            float distance = 0.0f;
            if (metric == Metric::IP) {
#pragma omp simd reduction(+ : distance)
                for (int j = 0; j < dim; ++j) distance -= query[j] * vector[j];
            } else {
#pragma omp simd reduction(+ : distance)
                for (int j = 0; j < dim; ++j) {
                    const float delta = query[j] - vector[j];
                    distance += delta * delta;
                }
            }
            scored[request.query].emplace_back(distance, request.id);
        }
    }
    store.record_reads(requests.size());
    std::free(allocation);
    io_uring_queue_exit(&ring);

    final_distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    final_labels.assign(static_cast<size_t>(nq) * k, -1);
    omp_set_num_threads(threads);
#pragma omp parallel for schedule(static)
    for (int qi = 0; qi < nq; ++qi) {
        auto& values = scored[qi];
        const size_t keep = std::min<size_t>(k, values.size());
        std::partial_sort(
                values.begin(), values.begin() + keep, values.end(),
                [](const auto& left, const auto& right) {
                    if (left.first != right.first) return left.first < right.first;
                    return left.second < right.second;
                });
        for (size_t rank = 0; rank < keep; ++rank) {
            final_distances[static_cast<size_t>(qi) * k + rank] = values[rank].first;
            final_labels[static_cast<size_t>(qi) * k + rank] = values[rank].second;
        }
    }
}

// Fixed-R reranking knows every candidate ID before exact scoring starts.
// Give each search worker its own ring and submit one query's candidate reads
// together (one submission when queue_depth >= candidate_k).  This is valid
// for every first-stage quantizer: the I/O stage only consumes candidate IDs.
void rerank_fp_uring_batch(
        FpDirectStore& store,
        const std::vector<float>& queries,
        int nq,
        int dim,
        int candidate_k,
        int k,
        int threads,
        unsigned queue_depth,
        Metric metric,
        const std::vector<faiss::idx_t>& candidate_labels,
        std::vector<float>& final_distances,
        std::vector<faiss::idx_t>& final_labels,
        double* io_wait_seconds_out) {
    final_distances.assign(static_cast<size_t>(nq) * k, 0.0f);
    final_labels.assign(static_cast<size_t>(nq) * k, -1);
    const unsigned ring_depth = std::max<unsigned>(
            1, std::min<unsigned>(queue_depth, static_cast<unsigned>(candidate_k)));
    std::atomic<uint64_t> io_wait_nanoseconds{0};
    omp_set_num_threads(threads);
#pragma omp parallel
    {
        io_uring ring{};
        const int init = io_uring_queue_init(ring_depth, &ring, 0);
        if (init < 0) {
            throw std::runtime_error(
                    "io_uring_queue_init(batch) failed: " + std::to_string(init));
        }
        const size_t stride = store.stride_bytes();
        void* allocation = nullptr;
        if (posix_memalign(
                    &allocation,
                    kCodePageBytes,
                    static_cast<size_t>(ring_depth) * stride) != 0) {
            io_uring_queue_exit(&ring);
            throw std::bad_alloc();
        }
        auto* buffers = static_cast<uint8_t*>(allocation);
        std::vector<faiss::idx_t> slot_ids(ring_depth, -1);
        std::vector<std::pair<float, faiss::idx_t>> scored;
        scored.reserve(candidate_k);
        uint64_t local_reads = 0;

#pragma omp for schedule(static)
        for (int qi = 0; qi < nq; ++qi) {
            scored.clear();
            const float* query = queries.data() + static_cast<size_t>(qi) * dim;
            for (int begin = 0; begin < candidate_k;
                 begin += static_cast<int>(ring_depth)) {
                unsigned count = 0;
                const int end = std::min(
                        candidate_k, begin + static_cast<int>(ring_depth));
                for (int rank = begin; rank < end; ++rank) {
                    const faiss::idx_t id = candidate_labels[
                            static_cast<size_t>(qi) * candidate_k + rank];
                    if (id < 0) continue;
                    io_uring_sqe* sqe = io_uring_get_sqe(&ring);
                    if (!sqe) {
                        throw std::runtime_error("io_uring batch SQ exhausted");
                    }
                    slot_ids[count] = id;
                    io_uring_prep_read(
                            sqe,
                            store.fd(),
                            buffers + static_cast<size_t>(count) * stride,
                            static_cast<unsigned>(stride),
                            store.offset(id));
                    io_uring_sqe_set_data64(sqe, count);
                    ++count;
                }
                if (count == 0) continue;
                const auto submit_started = std::chrono::steady_clock::now();
                const int submitted = io_uring_submit(&ring);
                io_wait_nanoseconds.fetch_add(
                        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - submit_started).count()),
                        std::memory_order_relaxed);
                if (submitted != static_cast<int>(count)) {
                    throw std::runtime_error(
                            "short io_uring batch submit: " +
                            std::to_string(submitted) + "/" +
                            std::to_string(count));
                }
                for (unsigned done = 0; done < count; ++done) {
                    io_uring_cqe* cqe = nullptr;
                    const auto wait_started = std::chrono::steady_clock::now();
                    const int wait = io_uring_wait_cqe(&ring, &cqe);
                    io_wait_nanoseconds.fetch_add(
                            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - wait_started).count()),
                            std::memory_order_relaxed);
                    if (wait < 0 || !cqe) {
                        throw std::runtime_error("io_uring batch wait failed");
                    }
                    const unsigned slot = static_cast<unsigned>(
                            io_uring_cqe_get_data64(cqe));
                    const int actual = cqe->res;
                    io_uring_cqe_seen(&ring, cqe);
                    if (actual != static_cast<int>(stride)) {
                        throw std::runtime_error(
                                "short io_uring batch read: " +
                                std::to_string(actual));
                    }
                    const float* vector = reinterpret_cast<const float*>(
                            buffers + static_cast<size_t>(slot) * stride);
                    float distance = 0.0f;
                    if (metric == Metric::IP) {
#pragma omp simd reduction(+ : distance)
                        for (int j = 0; j < dim; ++j) {
                            distance -= query[j] * vector[j];
                        }
                    } else {
#pragma omp simd reduction(+ : distance)
                        for (int j = 0; j < dim; ++j) {
                            const float delta = query[j] - vector[j];
                            distance += delta * delta;
                        }
                    }
                    scored.emplace_back(distance, slot_ids[slot]);
                }
                local_reads += count;
            }

            const size_t keep = std::min<size_t>(k, scored.size());
            std::partial_sort(
                    scored.begin(),
                    scored.begin() + keep,
                    scored.end(),
                    [](const auto& left, const auto& right) {
                        if (left.first != right.first) {
                            return left.first < right.first;
                        }
                        return left.second < right.second;
                    });
            for (size_t rank = 0; rank < keep; ++rank) {
                final_distances[static_cast<size_t>(qi) * k + rank] =
                        scored[rank].first;
                final_labels[static_cast<size_t>(qi) * k + rank] =
                        scored[rank].second;
            }
        }
        store.record_reads(local_reads);
        std::free(allocation);
        io_uring_queue_exit(&ring);
    }
    if (io_wait_seconds_out != nullptr) {
        *io_wait_seconds_out = static_cast<double>(
                io_wait_nanoseconds.load(std::memory_order_relaxed)) / 1e9;
    }
}

// T=1 steady-state pipeline: query q's FP reads remain in flight while the
// compressed HNSW search for q+1 runs.  Only one query owns the ring buffers at
// a time, so completing q before submitting q+1 is sufficient and keeps the
// exact same per-query read batch as the non-overlapped path.
class InterQueryFpReranker {
    FpDirectStore& store_;
    int dim_;
    int candidate_k_;
    int k_;
    Metric metric_;
    unsigned ring_depth_;
    io_uring ring_{};
    bool ring_initialized_ = false;
    uint8_t* buffers_ = nullptr;
    std::vector<faiss::idx_t> slot_ids_;
    unsigned pending_count_ = 0;
    double io_wait_seconds_ = 0.0;

    float exact_distance(const float* query, const float* vector) const {
        float distance = 0.0f;
        if (metric_ == Metric::IP) {
#pragma omp simd reduction(+ : distance)
            for (int j = 0; j < dim_; ++j) distance -= query[j] * vector[j];
        } else {
#pragma omp simd reduction(+ : distance)
            for (int j = 0; j < dim_; ++j) {
                const float delta = query[j] - vector[j];
                distance += delta * delta;
            }
        }
        return distance;
    }

public:
    InterQueryFpReranker(
            FpDirectStore& store,
            int dim,
            int candidate_k,
            int k,
            unsigned queue_depth,
            Metric metric)
            : store_(store),
              dim_(dim),
              candidate_k_(candidate_k),
              k_(k),
              metric_(metric),
              ring_depth_(std::max<unsigned>(
                      1,
                      std::min<unsigned>(
                              queue_depth,
                              static_cast<unsigned>(candidate_k)))),
              slot_ids_(ring_depth_, -1) {
        const int init = io_uring_queue_init(ring_depth_, &ring_, 0);
        if (init < 0) {
            throw std::runtime_error(
                    "io_uring_queue_init(inter-query) failed: " +
                    std::to_string(init));
        }
        ring_initialized_ = true;
        void* allocation = nullptr;
        if (posix_memalign(
                    &allocation,
                    kCodePageBytes,
                    static_cast<size_t>(ring_depth_) * store_.stride_bytes()) !=
            0) {
            io_uring_queue_exit(&ring_);
            ring_initialized_ = false;
            throw std::bad_alloc();
        }
        buffers_ = static_cast<uint8_t*>(allocation);
    }

    InterQueryFpReranker(const InterQueryFpReranker&) = delete;
    InterQueryFpReranker& operator=(const InterQueryFpReranker&) = delete;

    ~InterQueryFpReranker() {
        if (ring_initialized_) io_uring_queue_exit(&ring_);
        std::free(buffers_);
    }

    bool pending() const { return pending_count_ > 0; }
    double io_wait_seconds() const { return io_wait_seconds_; }

    void submit(const faiss::idx_t* candidate_labels) {
        if (pending()) {
            throw std::runtime_error(
                    "inter-query FP submit called with a pending batch");
        }
        const size_t stride = store_.stride_bytes();
        unsigned count = 0;
        for (int rank = 0; rank < candidate_k_; ++rank) {
            const faiss::idx_t id = candidate_labels[rank];
            if (id < 0) continue;
            if (count >= ring_depth_) {
                throw std::runtime_error(
                        "inter-query FP queue depth is smaller than rerank prefix");
            }
            io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
            if (!sqe) {
                throw std::runtime_error(
                        "io_uring inter-query submission queue exhausted");
            }
            slot_ids_[count] = id;
            io_uring_prep_read(
                    sqe,
                    store_.fd(),
                    buffers_ + static_cast<size_t>(count) * stride,
                    static_cast<unsigned>(stride),
                    store_.offset(id));
            io_uring_sqe_set_data64(sqe, count);
            ++count;
        }
        if (count == 0) {
            throw std::runtime_error("empty inter-query FP rerank prefix");
        }
        const auto submit_started = std::chrono::steady_clock::now();
        const int submitted = io_uring_submit(&ring_);
        io_wait_seconds_ += seconds_since(submit_started);
        if (submitted != static_cast<int>(count)) {
            throw std::runtime_error(
                    "short io_uring inter-query submit: " +
                    std::to_string(submitted) + "/" + std::to_string(count));
        }
        pending_count_ = count;
    }

    void finish(
            const float* query,
            float* final_distances,
            faiss::idx_t* final_labels) {
        if (!pending()) {
            throw std::runtime_error(
                    "inter-query FP finish called without a pending batch");
        }
        std::fill(final_distances, final_distances + k_, 0.0f);
        std::fill(final_labels, final_labels + k_, -1);
        const size_t stride = store_.stride_bytes();
        std::vector<std::pair<float, faiss::idx_t>> scored;
        scored.reserve(pending_count_);
        const unsigned count = pending_count_;
        for (unsigned done = 0; done < count; ++done) {
            io_uring_cqe* cqe = nullptr;
            const auto wait_started = std::chrono::steady_clock::now();
            const int wait = io_uring_wait_cqe(&ring_, &cqe);
            io_wait_seconds_ += seconds_since(wait_started);
            if (wait < 0 || !cqe) {
                throw std::runtime_error(
                        "io_uring inter-query wait failed: " +
                        std::to_string(wait));
            }
            const unsigned slot = static_cast<unsigned>(
                    io_uring_cqe_get_data64(cqe));
            const int actual = cqe->res;
            io_uring_cqe_seen(&ring_, cqe);
            if (slot >= count || actual != static_cast<int>(stride)) {
                throw std::runtime_error(
                        "short io_uring inter-query read: " +
                        std::to_string(actual));
            }
            scored.emplace_back(
                    exact_distance(
                            query,
                            reinterpret_cast<const float*>(
                                    buffers_ + static_cast<size_t>(slot) * stride)),
                    slot_ids_[slot]);
        }
        pending_count_ = 0;
        store_.record_reads(count);
        const size_t keep = std::min<size_t>(k_, scored.size());
        std::partial_sort(
                scored.begin(),
                scored.begin() + keep,
                scored.end(),
                [](const auto& left, const auto& right) {
                    if (left.first != right.first) return left.first < right.first;
                    return left.second < right.second;
                });
        for (size_t rank = 0; rank < keep; ++rank) {
            final_distances[rank] = scored[rank].first;
            final_labels[rank] = scored[rank].second;
        }
    }
};

// Persistent T=1 reranker for strict query-at-a-time measurements.  File,
// ring, and aligned buffers live across requests; rerank_one() contains only
// work that a steady-state request actually performs.
class SingleQueryFpReranker {
    FpDirectStore& store_;
    int dim_;
    int candidate_k_;
    int k_;
    Metric metric_;
    bool use_uring_;
    unsigned queue_depth_;
    io_uring ring_{};
    bool ring_initialized_ = false;
    uint8_t* buffers_ = nullptr;
    std::vector<faiss::idx_t> slot_ids_;
    std::vector<std::pair<float, faiss::idx_t>> scored_;

    float exact_distance(const float* query, const float* vector) const {
        float distance = 0.0f;
        if (metric_ == Metric::IP) {
#pragma omp simd reduction(+ : distance)
            for (int j = 0; j < dim_; ++j) distance -= query[j] * vector[j];
        } else {
#pragma omp simd reduction(+ : distance)
            for (int j = 0; j < dim_; ++j) {
                const float delta = query[j] - vector[j];
                distance += delta * delta;
            }
        }
        return distance;
    }

public:
    SingleQueryFpReranker(
            FpDirectStore& store,
            int dim,
            int candidate_k,
            int k,
            unsigned queue_depth,
            Metric metric,
            const std::string& io_mode)
            : store_(store),
              dim_(dim),
              candidate_k_(candidate_k),
              k_(k),
              metric_(metric),
              use_uring_(io_mode == "uring" || io_mode == "uring-batch"),
              queue_depth_(use_uring_ ? queue_depth : 1),
              slot_ids_(queue_depth_, -1) {
        if (queue_depth_ == 0) throw std::runtime_error("zero FP rerank queue depth");
        const size_t allocation_bytes =
                static_cast<size_t>(queue_depth_) * store_.stride_bytes();
        void* allocation = nullptr;
        if (posix_memalign(&allocation, kCodePageBytes, allocation_bytes) != 0) {
            throw std::bad_alloc();
        }
        buffers_ = static_cast<uint8_t*>(allocation);
        if (use_uring_) {
            const int init = io_uring_queue_init(queue_depth_, &ring_, 0);
            if (init < 0) {
                std::free(buffers_);
                buffers_ = nullptr;
                throw std::runtime_error(
                        "io_uring_queue_init(query-e2e) failed: " +
                        std::to_string(init));
            }
            ring_initialized_ = true;
        }
        scored_.reserve(candidate_k_);
    }

    SingleQueryFpReranker(const SingleQueryFpReranker&) = delete;
    SingleQueryFpReranker& operator=(const SingleQueryFpReranker&) = delete;

    ~SingleQueryFpReranker() {
        if (ring_initialized_) io_uring_queue_exit(&ring_);
        std::free(buffers_);
    }

    void rerank_one(
            const float* query,
            const faiss::idx_t* candidate_labels,
            float* final_distances,
            faiss::idx_t* final_labels,
            double* io_wait_seconds_out) {
        std::fill(final_distances, final_distances + k_, 0.0f);
        std::fill(final_labels, final_labels + k_, -1);
        scored_.clear();
        double io_wait_seconds = 0.0;

        if (!use_uring_) {
            for (int rank = 0; rank < candidate_k_; ++rank) {
                const faiss::idx_t id = candidate_labels[rank];
                if (id < 0) continue;
                store_.read(id, buffers_);
                scored_.emplace_back(
                        exact_distance(
                                query,
                                reinterpret_cast<const float*>(buffers_)),
                        id);
            }
        } else {
            const size_t stride = store_.stride_bytes();
            uint64_t local_reads = 0;
            for (int begin = 0; begin < candidate_k_;
                 begin += static_cast<int>(queue_depth_)) {
                unsigned count = 0;
                const int end = std::min(
                        candidate_k_, begin + static_cast<int>(queue_depth_));
                for (int rank = begin; rank < end; ++rank) {
                    const faiss::idx_t id = candidate_labels[rank];
                    if (id < 0) continue;
                    io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
                    if (!sqe) {
                        throw std::runtime_error(
                                "io_uring query-e2e submission queue exhausted");
                    }
                    slot_ids_[count] = id;
                    io_uring_prep_read(
                            sqe,
                            store_.fd(),
                            buffers_ + static_cast<size_t>(count) * stride,
                            static_cast<unsigned>(stride),
                            store_.offset(id));
                    io_uring_sqe_set_data64(sqe, count);
                    ++count;
                }
                if (count == 0) continue;

                const auto submit_started = std::chrono::steady_clock::now();
                const int submitted = io_uring_submit(&ring_);
                io_wait_seconds += seconds_since(submit_started);
                if (submitted != static_cast<int>(count)) {
                    throw std::runtime_error(
                            "short io_uring query-e2e submit: " +
                            std::to_string(submitted) + "/" +
                            std::to_string(count));
                }

                for (unsigned done = 0; done < count; ++done) {
                    io_uring_cqe* cqe = nullptr;
                    const auto wait_started = std::chrono::steady_clock::now();
                    const int wait = io_uring_wait_cqe(&ring_, &cqe);
                    io_wait_seconds += seconds_since(wait_started);
                    if (wait < 0 || !cqe) {
                        throw std::runtime_error(
                                "io_uring query-e2e wait failed: " +
                                std::to_string(wait));
                    }
                    const unsigned slot = static_cast<unsigned>(
                            io_uring_cqe_get_data64(cqe));
                    const int actual = cqe->res;
                    io_uring_cqe_seen(&ring_, cqe);
                    if (slot >= count || actual != static_cast<int>(stride)) {
                        throw std::runtime_error(
                                "short io_uring query-e2e read: " +
                                std::to_string(actual));
                    }
                    scored_.emplace_back(
                            exact_distance(
                                    query,
                                    reinterpret_cast<const float*>(
                                            buffers_ +
                                            static_cast<size_t>(slot) * stride)),
                            slot_ids_[slot]);
                }
                local_reads += count;
            }
            store_.record_reads(local_reads);
        }

        const size_t keep = std::min<size_t>(k_, scored_.size());
        std::partial_sort(
                scored_.begin(),
                scored_.begin() + keep,
                scored_.end(),
                [](const auto& left, const auto& right) {
                    if (left.first != right.first) return left.first < right.first;
                    return left.second < right.second;
                });
        for (size_t rank = 0; rank < keep; ++rank) {
            final_distances[rank] = scored_[rank].first;
            final_labels[rank] = scored_[rank].second;
        }
        if (io_wait_seconds_out != nullptr) {
            *io_wait_seconds_out = io_wait_seconds;
        }
    }
};

double percentile_us(std::vector<double> values, double percentile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position =
            (percentile / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(values.size() - 1, lower + 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

void release_graph_codes(faiss::IndexBinaryHNSW& graph) {
    auto* storage = dynamic_cast<faiss::IndexBinaryFlat*>(graph.storage);
    if (!storage) throw std::runtime_error("graph storage is not binary flat");
    storage->xb = faiss::MaybeOwnedVector<uint8_t>();
    storage->xflags = faiss::MaybeOwnedVector<uint8_t>();
    storage->block_scales = faiss::MaybeOwnedVector<uint8_t>();
}

void run_benchmark(
        const PagedArgs& args,
        faiss::IndexBinaryHNSW& graph,
        const std::vector<float>& queries,
        const std::vector<int32_t>& gt,
        int gt_dim,
        int dim,
        const BenchmarkInputs& inputs) {
    const int nq = static_cast<int>(queries.size() / dim);
    const size_t cache_pages = static_cast<size_t>(
            (args.bq_budget_bytes - args.pinned_metadata_bytes) / kCodePageBytes);
    make_parent(args.out);
    std::ofstream output(args.out);
    const std::string latency_path = args.query_e2e
            ? (args.latency_out.empty()
                       ? args.out + ".query_latency.csv"
                       : args.latency_out)
            : std::string();
    std::unique_ptr<std::ofstream> latency_output;
    if (args.query_e2e) {
        make_parent(latency_path);
        latency_output = std::make_unique<std::ofstream>(latency_path);
        if (!*latency_output) {
            throw std::runtime_error(
                    "failed to open query latency output: " + latency_path);
        }
        *latency_output
                << "dataset,method,efSearch,round,query_id,search_us,"
                << "rerank_us,rerank_io_wait_us,total_us\n";
    }
    std::unique_ptr<std::ofstream> candidate_trace;
    if (!args.candidate_trace.empty()) {
        make_parent(args.candidate_trace);
        candidate_trace = std::make_unique<std::ofstream>(args.candidate_trace);
        if (!*candidate_trace) {
            throw std::runtime_error(
                    "failed to open candidate trace: " + args.candidate_trace);
        }
        *candidate_trace
                << "dataset,method,efSearch,round,query_id,rank,vector_id,"
                << "approx_distance,page_no,byte_offset,candidate_k\n";
    }
    output
            << "dataset,method,metric,k,nq,nb,dim,M,efConstruction,efSearch,round,threads,"
            << "recall_at_10,qps,search_seconds,rerank_seconds,rerank_io_wait_seconds,total_seconds,"
            << "first_stage_distance_evaluations,first_stage_distance_evaluations_per_query,"
            << "first_stage_upper_distance_evaluations_per_query,"
            << "first_stage_base_distance_evaluations_per_query,"
            << "first_stage_scalar_distance_calls,first_stage_batch4_calls,"
            << "first_stage_search_impl,"
            << "rerank_enabled,rerank_n,code_bytes_per_vector,"
            << "adaptive_rerank,adaptive_tau,adaptive_rerank_count_mean,"
            << "adaptive_rerank_count_min,adaptive_rerank_count_p50,"
            << "adaptive_rerank_count_p95,adaptive_rerank_count_p99,"
            << "adaptive_rerank_count_max,"
            << "bq_budget_bytes,pinned_metadata_bytes,cache_capacity_pages,"
            << "data_pages,physical_page_resident_fraction,logical_vector_capacity,"
            << "logical_vector_resident_fraction,page_accesses,page_hits,page_misses,"
            << "page_accesses_per_query,page_reads_per_query,page_evictions,page_hit_ratio,"
            << "page_read_bytes,page_read_bytes_per_query,fp_page_reads,"
            << "fp_page_reads_per_query,fp_read_bytes,fp_read_bytes_per_query,"
            << "total_physical_page_reads_per_query,graph_path,codec_index,"
            << "sidecar_path,fp_sidecar_path,code_storage,fp_io_mode,fp_rerank_qd,"
            << "measurement_mode,query_batch_size,latency_mean_us,"
            << "latency_p50_us,latency_p95_us,latency_p99_us,latency_max_us,"
            << "latency_csv\n";

    for (int ef : args.efs) {
        for (int round = 1; round <= args.rounds; ++round) {
            std::unique_ptr<CodePageCache> cache;
            if (!args.resident_codes) {
                cache = std::make_unique<CodePageCache>(
                        args.sidecar, inputs.sidecar_header, cache_pages);
            }
            std::unique_ptr<FpDirectStore> fp_store;
            if (args.rerank_n > 0 && !args.candidate_trace_only) {
                fp_store = std::make_unique<FpDirectStore>(
                        args.fp_sidecar, graph.ntotal, dim);
            }
            const int first_stage_k = args.rerank_n > 0 ? args.rerank_n : args.k;
            std::vector<float> candidate_distances;
            std::vector<faiss::idx_t> candidate_labels;
            std::vector<float> distances;
            std::vector<faiss::idx_t> labels;
            std::vector<double> query_latencies_us;
            double search_elapsed = 0.0;
            double rerank_elapsed = 0.0;
            double rerank_io_wait_elapsed = 0.0;
            double elapsed = 0.0;
            uint64_t first_stage_distance_evaluations = 0;
            uint64_t first_stage_upper_distance_evaluations = 0;
            uint64_t first_stage_base_distance_evaluations = 0;
            uint64_t first_stage_scalar_distance_calls = 0;
            uint64_t first_stage_batch4_calls = 0;
            AdaptivePrefixStats adaptive_stats;

            if (args.fp_overlap_queries) {
                candidate_distances.assign(
                        static_cast<size_t>(nq) * first_stage_k, 0.0f);
                candidate_labels.assign(
                        static_cast<size_t>(nq) * first_stage_k, -1);
                distances.assign(static_cast<size_t>(nq) * args.k, 0.0f);
                labels.assign(static_cast<size_t>(nq) * args.k, -1);
                if (!fp_store || !inputs.resident_codes) {
                    throw std::runtime_error(
                            "inter-query overlap requires FP and resident codes");
                }

                auto computer = std::make_unique<ResidentCodeDistanceComputer>(
                        inputs.resident_codes,
                        inputs.code_size,
                        graph.ntotal,
                        inputs.make_evaluator());
                InterQueryFpReranker reranker(
                        *fp_store,
                        dim,
                        first_stage_k,
                        args.k,
                        args.fp_rerank_qd,
                        args.metric);
                faiss::SearchParametersHNSW params;
                params.efSearch = ef;
                faiss::VisitedTable visited(graph.ntotal);
                faiss::MinimaxHeap inline_candidates(
                        std::max(ef, first_stage_k));
                using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;
                std::vector<int> adaptive_counts(static_cast<size_t>(nq), 0);
                const auto pipeline_started = std::chrono::steady_clock::now();

                for (int qi = 0; qi < nq; ++qi) {
                    float* query_candidate_distances =
                            candidate_distances.data() +
                            static_cast<size_t>(qi) * first_stage_k;
                    faiss::idx_t* query_candidate_labels =
                            candidate_labels.data() +
                            static_cast<size_t>(qi) * first_stage_k;
                    const float* query =
                            queries.data() + static_cast<size_t>(qi) * dim;

                    const auto search_started = std::chrono::steady_clock::now();
                    computer->set_query(query);
                    if (args.inline_hnsw) {
                        search_one_inline_hnsw(
                                graph,
                                *computer,
                                visited,
                                inline_candidates,
                                first_stage_k,
                                ef,
                                query_candidate_distances,
                                query_candidate_labels);
                    } else {
                        RH block_result(
                                1,
                                query_candidate_distances,
                                query_candidate_labels,
                                first_stage_k);
                        RH::SingleResultHandler result(block_result);
                        result.begin(0);
                        graph.hnsw.search(
                                *computer, nullptr, result, visited, &params);
                        result.end();
                    }
                    search_elapsed += seconds_since(search_started);

                    int valid = 0;
                    while (valid < first_stage_k &&
                           query_candidate_labels[valid] >= 0 &&
                           std::isfinite(query_candidate_distances[valid])) {
                        ++valid;
                    }
                    int take = valid;
                    if (args.adaptive_rerank) {
                        const int minimum = std::min(args.k, valid);
                        take = minimum;
                        if (take > 0) {
                            const float cutoff =
                                    query_candidate_distances[minimum - 1] +
                                    args.adaptive_tau;
                            while (take < valid &&
                                   query_candidate_distances[take] <= cutoff) {
                                ++take;
                            }
                        }
                    }
                    adaptive_counts[static_cast<size_t>(qi)] = take;
                    for (int rank = take; rank < first_stage_k; ++rank) {
                        query_candidate_labels[rank] = -1;
                    }

                    // Finishing q-1 only after q's compressed search is what
                    // creates the same inter-query overlap as ADC-BQ.
                    if (qi > 0) {
                        reranker.finish(
                                queries.data() +
                                        static_cast<size_t>(qi - 1) * dim,
                                distances.data() +
                                        static_cast<size_t>(qi - 1) * args.k,
                                labels.data() +
                                        static_cast<size_t>(qi - 1) * args.k);
                    }
                    reranker.submit(query_candidate_labels);
                }
                reranker.finish(
                        queries.data() + static_cast<size_t>(nq - 1) * dim,
                        distances.data() + static_cast<size_t>(nq - 1) * args.k,
                        labels.data() + static_cast<size_t>(nq - 1) * args.k);
                elapsed = seconds_since(pipeline_started);
                rerank_elapsed = std::max(0.0, elapsed - search_elapsed);
                rerank_io_wait_elapsed = reranker.io_wait_seconds();
                first_stage_distance_evaluations = computer->distance_evaluations;
                first_stage_upper_distance_evaluations =
                        computer->upper_distance_evaluations;
                first_stage_base_distance_evaluations =
                        computer->base_distance_evaluations;
                first_stage_scalar_distance_calls = computer->scalar_distance_calls;
                first_stage_batch4_calls = computer->batch4_calls;

                if (args.adaptive_rerank && !adaptive_counts.empty()) {
                    adaptive_stats.minimum = *std::min_element(
                            adaptive_counts.begin(), adaptive_counts.end());
                    adaptive_stats.maximum = *std::max_element(
                            adaptive_counts.begin(), adaptive_counts.end());
                    adaptive_stats.mean = static_cast<double>(std::accumulate(
                                                  adaptive_counts.begin(),
                                                  adaptive_counts.end(),
                                                  int64_t{0})) /
                            adaptive_counts.size();
                    std::sort(adaptive_counts.begin(), adaptive_counts.end());
                    adaptive_stats.p50 = count_percentile(adaptive_counts, 0.50);
                    adaptive_stats.p95 = count_percentile(adaptive_counts, 0.95);
                    adaptive_stats.p99 = count_percentile(adaptive_counts, 0.99);
                }
            } else if (args.query_e2e) {
                candidate_distances.assign(
                        static_cast<size_t>(nq) * first_stage_k, 0.0f);
                candidate_labels.assign(
                        static_cast<size_t>(nq) * first_stage_k, -1);
                distances.assign(static_cast<size_t>(nq) * args.k, 0.0f);
                labels.assign(static_cast<size_t>(nq) * args.k, -1);
                query_latencies_us.reserve(nq);

                std::unique_ptr<faiss::DistanceComputer> computer;
                if (args.resident_codes) {
                    computer = std::make_unique<ResidentCodeDistanceComputer>(
                            inputs.resident_codes,
                            inputs.code_size,
                            graph.ntotal,
                            inputs.make_evaluator());
                } else {
                    computer = std::make_unique<PagedCodeDistanceComputer>(
                            *cache, inputs.make_evaluator());
                }

                std::unique_ptr<SingleQueryFpReranker> reranker;
                if (fp_store) {
                    reranker = std::make_unique<SingleQueryFpReranker>(
                            *fp_store,
                            dim,
                            first_stage_k,
                            args.k,
                            args.fp_rerank_qd,
                            args.metric,
                            args.fp_io_mode);
                }

                faiss::SearchParametersHNSW params;
                params.efSearch = ef;
                faiss::VisitedTable visited(graph.ntotal);
                using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;

                for (int qi = 0; qi < nq; ++qi) {
                    float* query_candidate_distances =
                            candidate_distances.data() +
                            static_cast<size_t>(qi) * first_stage_k;
                    faiss::idx_t* query_candidate_labels =
                            candidate_labels.data() +
                            static_cast<size_t>(qi) * first_stage_k;
                    float* query_distances =
                            distances.data() + static_cast<size_t>(qi) * args.k;
                    faiss::idx_t* query_labels =
                            labels.data() + static_cast<size_t>(qi) * args.k;
                    const float* query =
                            queries.data() + static_cast<size_t>(qi) * dim;

                    const auto query_started = std::chrono::steady_clock::now();
                    RH block_result(
                            1,
                            query_candidate_distances,
                            query_candidate_labels,
                            first_stage_k);
                    RH::SingleResultHandler result(block_result);
                    result.begin(0);
                    computer->set_query(query);
                    graph.hnsw.search(*computer, nullptr, result, visited, &params);
                    result.end();
                    const auto search_finished = std::chrono::steady_clock::now();

                    double query_io_wait_seconds = 0.0;
                    if (reranker) {
                        reranker->rerank_one(
                                query,
                                query_candidate_labels,
                                query_distances,
                                query_labels,
                                &query_io_wait_seconds);
                    } else {
                        std::copy_n(
                                query_candidate_distances,
                                args.k,
                                query_distances);
                        std::copy_n(
                                query_candidate_labels, args.k, query_labels);
                    }
                    const auto query_finished = std::chrono::steady_clock::now();

                    const double query_search_seconds =
                            std::chrono::duration<double>(
                                    search_finished - query_started)
                                    .count();
                    const double query_rerank_seconds =
                            std::chrono::duration<double>(
                                    query_finished - search_finished)
                                    .count();
                    const double query_total_seconds =
                            std::chrono::duration<double>(
                                    query_finished - query_started)
                                    .count();
                    search_elapsed += query_search_seconds;
                    rerank_elapsed += query_rerank_seconds;
                    rerank_io_wait_elapsed += query_io_wait_seconds;
                    elapsed += query_total_seconds;
                    const double total_us = query_total_seconds * 1e6;
                    query_latencies_us.push_back(total_us);
                    *latency_output
                            << std::setprecision(10) << args.dataset << ','
                            << paged_method_name(args.method) << ',' << ef << ','
                            << round << ',' << qi << ','
                            << query_search_seconds * 1e6 << ','
                            << query_rerank_seconds * 1e6 << ','
                            << query_io_wait_seconds * 1e6 << ',' << total_us
                            << '\n';
                }
                latency_output->flush();
            } else {
                const auto started = std::chrono::steady_clock::now();
                if (args.inline_hnsw) {
                    candidate_distances.assign(
                            static_cast<size_t>(nq) * first_stage_k, 0.0f);
                    candidate_labels.assign(
                            static_cast<size_t>(nq) * first_stage_k, -1);
                    omp_set_num_threads(args.threads);
#pragma omp parallel
                    {
                        faiss::VisitedTable visited(graph.ntotal, false);
                        faiss::MinimaxHeap candidates(
                                std::max(ef, first_stage_k));
                        ResidentCodeDistanceComputer computer(
                                inputs.resident_codes,
                                inputs.code_size,
                                graph.ntotal,
                                inputs.make_evaluator());
#pragma omp for schedule(static)
                        for (int qi = 0; qi < nq; ++qi) {
                            computer.set_query(
                                    queries.data() +
                                    static_cast<size_t>(qi) * dim);
                            search_one_inline_hnsw(
                                    graph,
                                    computer,
                                    visited,
                                    candidates,
                                    first_stage_k,
                                    ef,
                                    candidate_distances.data() +
                                            static_cast<size_t>(qi) *
                                                    first_stage_k,
                                    candidate_labels.data() +
                                            static_cast<size_t>(qi) *
                                                    first_stage_k);
                        }
                    }
                } else if (args.resident_codes) {
                    search_custom<ResidentCodeDistanceComputer>(
                            graph,
                            queries.data(),
                            nq,
                            dim,
                            first_stage_k,
                            ef,
                            args.threads,
                            [&]() {
                                return std::make_unique<ResidentCodeDistanceComputer>(
                                        inputs.resident_codes,
                                        inputs.code_size,
                                        graph.ntotal,
                                        inputs.make_evaluator());
                            },
                            candidate_distances,
                            candidate_labels);
                } else {
                    search_custom<PagedCodeDistanceComputer>(
                            graph,
                            queries.data(),
                            nq,
                            dim,
                            first_stage_k,
                            ef,
                            args.threads,
                            [&]() {
                                return std::make_unique<PagedCodeDistanceComputer>(
                                        *cache, inputs.make_evaluator());
                            },
                            candidate_distances,
                            candidate_labels);
                }
                search_elapsed = seconds_since(started);
                if (candidate_trace) {
                    const uint64_t stride_bytes =
                            ((static_cast<uint64_t>(dim) * sizeof(float) +
                              kCodePageBytes - 1) /
                             kCodePageBytes) *
                            kCodePageBytes;
                    for (int qi = 0; qi < nq; ++qi) {
                        for (int rank = 0; rank < first_stage_k; ++rank) {
                            const faiss::idx_t id = candidate_labels[
                                    static_cast<size_t>(qi) * first_stage_k + rank];
                            if (id < 0) continue;
                            const uint64_t page_no =
                                    1 + static_cast<uint64_t>(id) *
                                            (stride_bytes / kCodePageBytes);
                            const uint64_t byte_offset =
                                    page_no * kCodePageBytes;
                            *candidate_trace
                                    << args.dataset << ','
                                    << paged_method_name(args.method) << ',' << ef
                                    << ',' << round << ',' << qi << ',' << rank
                                    << ',' << id << ','
                                    << std::setprecision(10) << candidate_distances[
                                               static_cast<size_t>(qi) *
                                                       first_stage_k + rank]
                                    << ',' << page_no << ','
                                    << byte_offset << ',' << first_stage_k << '\n';
                        }
                    }
                    candidate_trace->flush();
                }
                if (fp_store) {
                    const std::vector<faiss::idx_t>* rerank_labels =
                            &candidate_labels;
                    if (args.adaptive_rerank) {
                        adaptive_stats = select_adaptive_prefix(
                                candidate_labels,
                                candidate_distances,
                                nq,
                                first_stage_k,
                                args.k,
                                args.adaptive_tau);
                        rerank_labels = &adaptive_stats.labels;
                    }
                    const auto rerank_started = std::chrono::steady_clock::now();
                    if (args.fp_io_mode == "uring-batch") {
                        rerank_fp_uring_batch(
                                *fp_store, queries, nq, dim, first_stage_k, args.k,
                                args.threads, args.fp_rerank_qd, args.metric,
                                *rerank_labels, distances, labels,
                                &rerank_io_wait_elapsed);
                    } else if (args.fp_io_mode == "uring") {
                        rerank_fp_uring(
                                *fp_store, queries, nq, dim, first_stage_k, args.k,
                                args.threads, args.fp_rerank_qd, args.metric,
                                *rerank_labels, distances, labels);
                    } else {
                        rerank_fp_direct(
                                *fp_store, queries, nq, dim, first_stage_k, args.k,
                                args.threads, args.metric, *rerank_labels,
                                distances, labels);
                    }
                    rerank_elapsed = seconds_since(rerank_started);
                } else if (args.candidate_trace_only) {
                    distances.resize(static_cast<size_t>(nq) * args.k);
                    labels.resize(static_cast<size_t>(nq) * args.k);
                    for (int qi = 0; qi < nq; ++qi) {
                        for (int rank = 0; rank < args.k; ++rank) {
                            distances[static_cast<size_t>(qi) * args.k + rank] =
                                    candidate_distances[
                                            static_cast<size_t>(qi) *
                                                    first_stage_k +
                                            rank];
                            labels[static_cast<size_t>(qi) * args.k + rank] =
                                    candidate_labels[
                                            static_cast<size_t>(qi) *
                                                    first_stage_k +
                                            rank];
                        }
                    }
                } else {
                    distances = std::move(candidate_distances);
                    labels = std::move(candidate_labels);
                }
                elapsed = search_elapsed + rerank_elapsed;
            }
            const double recall = recall_at_k(labels, gt, nq, gt_dim, args.k);
            const PageCacheStats stats = cache ? cache->stats() : PageCacheStats{};
            const double hit_ratio = stats.accesses
                    ? static_cast<double>(stats.hits) / stats.accesses
                    : 0.0;
            const double physical_fraction = args.resident_codes
                    ? 1.0
                    : static_cast<double>(
                              std::min<uint64_t>(
                                      cache_pages, inputs.sidecar_header.data_pages)) /
                              inputs.sidecar_header.data_pages;
            const uint64_t logical_capacity = args.resident_codes
                    ? static_cast<uint64_t>(graph.ntotal)
                    : std::min<uint64_t>(
                              graph.ntotal,
                              static_cast<uint64_t>(cache_pages) *
                                      inputs.sidecar_header.slots_per_page);
            const double logical_fraction =
                    static_cast<double>(logical_capacity) / graph.ntotal;
            const uint64_t fp_reads = fp_store ? fp_store->reads() : 0;
            const uint64_t fp_read_bytes = fp_store ? fp_store->read_bytes() : 0;
            const double code_reads_per_query =
                    static_cast<double>(stats.misses) / nq;
            const double fp_reads_per_query = static_cast<double>(fp_reads) / nq;
            output << std::setprecision(10)
                   << args.dataset << ',' << paged_method_name(args.method) << ','
                   << metric_name(args.metric) << ',' << args.k << ',' << nq << ','
                   << graph.ntotal << ',' << dim << ','
                   << graph.hnsw.nb_neighbors(0) / 2 << ','
                   << graph.hnsw.efConstruction << ',' << ef << ',' << round
                   << ',' << args.threads << ',' << recall << ',' << nq / elapsed << ','
                   << search_elapsed << ',' << rerank_elapsed << ','
                   << rerank_io_wait_elapsed << ',' << elapsed << ','
                   << first_stage_distance_evaluations << ','
                   << static_cast<double>(first_stage_distance_evaluations) / nq << ','
                   << static_cast<double>(
                              first_stage_upper_distance_evaluations) /
                              nq
                   << ','
                   << static_cast<double>(
                              first_stage_base_distance_evaluations) /
                              nq
                   << ','
                   << first_stage_scalar_distance_calls << ','
                   << first_stage_batch4_calls << ','
                   << (args.inline_hnsw ? "inline_batched_prefetch" : "faiss")
                   << ','
                   << (fp_store ? "true" : "false") << ','
                   << args.rerank_n << ',' << inputs.code_size << ','
                   << (args.adaptive_rerank ? "true" : "false") << ','
                   << (args.adaptive_rerank ? args.adaptive_tau : 0.0f) << ','
                   << (args.adaptive_rerank ? adaptive_stats.mean : args.rerank_n) << ','
                   << (args.adaptive_rerank ? adaptive_stats.minimum : args.rerank_n) << ','
                   << (args.adaptive_rerank ? adaptive_stats.p50 : args.rerank_n) << ','
                   << (args.adaptive_rerank ? adaptive_stats.p95 : args.rerank_n) << ','
                   << (args.adaptive_rerank ? adaptive_stats.p99 : args.rerank_n) << ','
                   << (args.adaptive_rerank ? adaptive_stats.maximum : args.rerank_n) << ','
                   << args.bq_budget_bytes << ','
                   << args.pinned_metadata_bytes << ',' << cache_pages << ','
                   << inputs.sidecar_header.data_pages << ',' << physical_fraction
                   << ',' << logical_capacity << ',' << logical_fraction << ','
                   << stats.accesses << ',' << stats.hits << ',' << stats.misses << ','
                   << static_cast<double>(stats.accesses) / nq << ','
                   << code_reads_per_query << ','
                   << stats.evictions << ',' << hit_ratio << ',' << stats.read_bytes << ','
                   << static_cast<double>(stats.read_bytes) / nq << ','
                   << fp_reads << ',' << fp_reads_per_query << ',' << fp_read_bytes << ','
                   << static_cast<double>(fp_read_bytes) / nq << ','
                   << code_reads_per_query + fp_reads_per_query
                   << ',' << args.graph << ',' << args.codec_index << ','
                   << args.sidecar << ',' << args.fp_sidecar << ','
                   << (args.resident_codes ? "resident" : "paged") << ','
                   << args.fp_io_mode << ',' << args.fp_rerank_qd << ','
                   << (args.fp_overlap_queries
                               ? "inter_query_overlap"
                               : (args.query_e2e ? "query_e2e" : "bulk_phased"))
                   << ','
                   << ((args.query_e2e || args.fp_overlap_queries) ? 1 : nq)
                   << ',';
            if (args.query_e2e) {
                output << (elapsed * 1e6 / nq) << ','
                       << percentile_us(query_latencies_us, 50.0) << ','
                       << percentile_us(query_latencies_us, 95.0) << ','
                       << percentile_us(query_latencies_us, 99.0) << ','
                       << percentile_us(query_latencies_us, 100.0) << ','
                       << latency_path;
            } else {
                output << ",,,,,";
            }
            output << '\n';
            output.flush();
            std::cout << "  ef=" << ef << " round=" << round
                      << " recall=" << recall << " qps=" << nq / elapsed
                      << " hit=" << hit_ratio << " code_reads=" << stats.misses
                      << " fp_reads=" << fp_reads;
            if (args.adaptive_rerank) {
                std::cout << " Rmean=" << adaptive_stats.mean
                          << " Rp95=" << adaptive_stats.p95
                          << " Rmax=" << adaptive_stats.maximum;
            }
            if (args.query_e2e) {
                std::cout << " latency_p50_us="
                          << percentile_us(query_latencies_us, 50.0)
                          << " latency_p99_us="
                          << percentile_us(query_latencies_us, 99.0);
            }
            std::cout
                      << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    try {
        const PagedArgs args = parse_paged_args(argc, argv);
        FbinMmap query_file(args.query);
        std::vector<float> queries = read_queries(
                query_file, args.max_queries, args.metric == Metric::IP);
        const int nq = static_cast<int>(queries.size() / query_file.d);
        int32_t gt_rows = 0;
        int32_t gt_dim = 0;
        std::vector<int32_t> gt = read_ibin(args.gt, nq, gt_rows, gt_dim);
        if (gt_rows != nq || gt_dim < args.k) {
            throw std::runtime_error("ground-truth shape mismatch");
        }
        auto graph = load_binary_hnsw(args.graph);
        if (graph->ntotal <= 0) throw std::runtime_error("empty graph");
        release_graph_codes(*graph);

        BenchmarkInputs inputs;
        std::unique_ptr<faiss::Index> faiss_codec;
        std::unique_ptr<faiss::IndexBinaryFlat> binary_codec;
        QdrantTqIndexPtr qdrant_codec;
        RaBitQMeta rabitq_meta;

        if (args.method == PagedMethod::PQ8M128 ||
            args.method == PagedMethod::OPQ8M128 ||
            args.method == PagedMethod::PQ8D8 ||
            args.method == PagedMethod::OPQ8D8 ||
            args.method == PagedMethod::SQ4 ||
            args.method == PagedMethod::SQ8) {
            faiss_codec.reset(faiss::read_index(args.codec_index.c_str()));
            faiss::IndexPreTransform* pretransform = nullptr;
            faiss::Index* storage = faiss_codec.get();
            if (args.method == PagedMethod::OPQ8M128 ||
                args.method == PagedMethod::OPQ8D8) {
                pretransform = dynamic_cast<faiss::IndexPreTransform*>(faiss_codec.get());
                if (!pretransform) throw std::runtime_error("OPQ codec type mismatch");
                storage = pretransform->index;
            }
            auto* flat = dynamic_cast<faiss::IndexFlatCodes*>(storage);
            if (!flat || flat->ntotal != graph->ntotal) {
                throw std::runtime_error("Faiss codec shape/type mismatch");
            }
            inputs.code_size = flat->code_size;
            inputs.sidecar_header = args.resident_codes
                    ? expected_header(graph->ntotal, inputs.code_size)
                    : ensure_sidecar(
                              args,
                              graph->ntotal,
                              inputs.code_size,
                              flat->codes.data());
            if (args.resident_codes) {
                inputs.resident_codes = flat->codes.data();
            } else {
                flat->codes = faiss::MaybeOwnedVector<uint8_t>();
            }
            inputs.make_evaluator = [flat, pretransform, &args]() {
                return std::make_unique<FaissCodeEvaluator>(
                        flat, pretransform, args.metric == Metric::IP);
            };
        } else if (args.method == PagedMethod::AdcBQ1 ||
                   args.method == PagedMethod::SymBQ1 ||
                   args.method == PagedMethod::RaBitQ1) {
            binary_codec = load_paged_binary_flat(args.codec_index);
            if (binary_codec->ntotal != graph->ntotal) {
                throw std::runtime_error("binary codec shape mismatch");
            }
            inputs.code_size = binary_codec->code_size;
            inputs.sidecar_header = args.resident_codes
                    ? expected_header(graph->ntotal, inputs.code_size)
                    : ensure_sidecar(
                              args,
                              graph->ntotal,
                              inputs.code_size,
                              binary_codec->xb.data());
            if (args.resident_codes) {
                inputs.resident_codes = binary_codec->xb.data();
            } else {
                binary_codec.reset();
            }
            if (args.method == PagedMethod::AdcBQ1) {
                inputs.make_evaluator = [&args, &query_file, &inputs]() {
                    return std::make_unique<CustomCodeEvaluator<BQDistanceComputer>>(
                            nullptr,
                            inputs.code_size,
                            query_file.d,
                            args.metric);
                };
            } else if (args.method == PagedMethod::SymBQ1) {
                inputs.make_evaluator = [&query_file, &inputs]() {
                    return std::make_unique<
                            CustomCodeEvaluator<SymmetricBQDistanceComputer>>(
                            nullptr, inputs.code_size, query_file.d);
                };
            } else {
                rabitq_meta = load_rabitq_meta(args.meta, args.metric);
                inputs.make_evaluator = [&rabitq_meta, &query_file, &inputs]() {
                    return std::make_unique<CustomCodeEvaluator<RaBitQDistanceComputer>>(
                            nullptr,
                            inputs.code_size,
                            query_file.d,
                            rabitq_meta);
                };
            }
        } else {
            QdrantTqIndex* raw = qdrant_tq_load(
                    args.codec_index.c_str(), args.meta.c_str());
            if (!raw) throw std::runtime_error(qdrant_tq_error());
            qdrant_codec.reset(raw);
            size_t code_bytes = 0;
            const uint8_t* codes = qdrant_tq_codes(qdrant_codec.get(), &code_bytes);
            inputs.code_size = qdrant_tq_code_size(qdrant_codec.get());
            if (!codes || code_bytes != static_cast<size_t>(graph->ntotal) * inputs.code_size) {
                throw std::runtime_error("Qdrant codec shape mismatch");
            }
            inputs.sidecar_header = args.resident_codes
                    ? expected_header(graph->ntotal, inputs.code_size)
                    : ensure_sidecar(args, graph->ntotal, inputs.code_size, codes);
            if (args.resident_codes) inputs.resident_codes = codes;
            inputs.make_evaluator = [&qdrant_codec, &query_file, &args, &inputs]() {
                return std::make_unique<QdrantTqBatchedCodeEvaluator>(
                        qdrant_codec.get(),
                        inputs.code_size,
                        query_file.d,
                        args.metric);
            };
        }

        std::cout << "Dataset=" << args.dataset
                  << " method=" << paged_method_name(args.method)
                  << " code_size=" << inputs.code_size
                  << " data_pages=" << inputs.sidecar_header.data_pages
                  << " cache_pages="
                  << (args.bq_budget_bytes - args.pinned_metadata_bytes) /
                             kCodePageBytes
                  << " code_storage="
                  << (args.resident_codes ? "resident" : "paged")
                  << std::endl;
        run_benchmark(
                args,
                *graph,
                queries,
                gt,
                gt_dim,
                query_file.d,
                inputs);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
