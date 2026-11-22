#define main faiss_hnsw_lowbit_compare_main_disabled
#include "faiss_hnsw_lowbit_compare.cpp"
#undef main

#include <faiss/IndexPreTransform.h>

#include <atomic>
#include <sstream>

enum class ExactMethod {
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

struct ExactArgs {
    std::string dataset;
    std::string train;
    std::string query;
    std::string gt;
    std::string index;
    std::string meta;
    std::string rabitq_centroids;
    std::string summary_out;
    std::string per_query_out;
    ExactMethod method = ExactMethod::AdcBQ1;
    Metric metric = Metric::IP;
    int k = 10;
    int max_queries = 1000;
    int threads = 24;
    int train_size = 100000;
    int add_batch_size = 4096;
    int rotation_seed = 20261020;
    int opq_niter = 25;
    bool rebuild = false;
    bool build_only = false;
};

const char* exact_method_name(ExactMethod method) {
    switch (method) {
        case ExactMethod::AdcBQ1: return "ADC-BQ1";
        case ExactMethod::SymBQ1: return "Symmetric-BQ1";
        case ExactMethod::PQ8M128: return "Faiss-PQ8-M128";
        case ExactMethod::OPQ8M128: return "Faiss-OPQ-PQ8-M128";
        case ExactMethod::PQ8D8: return "Faiss-PQ8-d/8";
        case ExactMethod::OPQ8D8: return "Faiss-OPQ-PQ8-d/8";
        case ExactMethod::SQ4: return "Faiss-SQ4";
        case ExactMethod::SQ8: return "Faiss-SQ8";
        case ExactMethod::RaBitQ1: return "RaBitQ-1b-c16";
        case ExactMethod::QdrantTQ1: return "Qdrant-TQ1-Plus";
    }
    return "unknown";
}

ExactMethod parse_exact_method(const std::string& value) {
    if (value == "adcbq1" || value == "bq1" || value == "bq") {
        return ExactMethod::AdcBQ1;
    }
    if (value == "symbq1" || value == "symbq") {
        return ExactMethod::SymBQ1;
    }
    if (value == "pq8" || value == "pq8m128" || value == "pq") {
        return ExactMethod::PQ8M128;
    }
    if (value == "opq8" || value == "opq8m128" || value == "opq") {
        return ExactMethod::OPQ8M128;
    }
    if (value == "pq8d8") return ExactMethod::PQ8D8;
    if (value == "opq8d8") return ExactMethod::OPQ8D8;
    if (value == "sq4") return ExactMethod::SQ4;
    if (value == "sq8") return ExactMethod::SQ8;
    if (value == "rabitq1" || value == "rabitq") return ExactMethod::RaBitQ1;
    if (value == "qdranttq1" || value == "tq1" || value == "turboquant") {
        return ExactMethod::QdrantTQ1;
    }
    throw std::runtime_error(
            "--method must be adcbq1|symbq1|pq8m128|opq8m128|pq8d8|opq8d8|sq4|sq8|rabitq1|qdranttq1");
}

[[noreturn]] void exact_usage(const char* argv0) {
    std::cerr
            << "Usage: " << argv0
            << " --dataset NAME --method METHOD --metric ip|l2"
            << " --train X.fbin --query Q.fbin --gt G.ibin"
            << " --index INDEX --summary-out SUMMARY.csv --per-query-out QUERY.csv"
            << " [--meta META] [--rabitq-centroids C16.fbin]"
            << " [--max-queries 1000] [--threads 24] [--train-size 100000]"
            << " [--opq-niter 25] [--build-only] [--rebuild]\n";
    std::exit(2);
}

ExactArgs parse_exact_args(int argc, char** argv) {
    ExactArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };
        if (key == "--dataset") args.dataset = need("--dataset");
        else if (key == "--method") args.method = parse_exact_method(need("--method"));
        else if (key == "--metric") args.metric = parse_metric(need("--metric"));
        else if (key == "--train") args.train = need("--train");
        else if (key == "--query") args.query = need("--query");
        else if (key == "--gt") args.gt = need("--gt");
        else if (key == "--index") args.index = need("--index");
        else if (key == "--meta") args.meta = need("--meta");
        else if (key == "--rabitq-centroids") {
            args.rabitq_centroids = need("--rabitq-centroids");
        } else if (key == "--summary-out") {
            args.summary_out = need("--summary-out");
        } else if (key == "--per-query-out") {
            args.per_query_out = need("--per-query-out");
        } else if (key == "--k") {
            args.k = std::stoi(need("--k"));
        } else if (key == "--max-queries") {
            args.max_queries = std::stoi(need("--max-queries"));
        } else if (key == "--threads") {
            args.threads = std::stoi(need("--threads"));
        } else if (key == "--train-size") {
            args.train_size = std::stoi(need("--train-size"));
        } else if (key == "--add-batch-size") {
            args.add_batch_size = std::stoi(need("--add-batch-size"));
        } else if (key == "--rotation-seed") {
            args.rotation_seed = std::stoi(need("--rotation-seed"));
        } else if (key == "--opq-niter") {
            args.opq_niter = std::stoi(need("--opq-niter"));
        } else if (key == "--rebuild") {
            args.rebuild = true;
        } else if (key == "--build-only") {
            args.build_only = true;
        } else if (key == "--help" || key == "-h") {
            exact_usage(argv[0]);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.dataset.empty() || args.train.empty() || args.query.empty() ||
        args.gt.empty() || args.index.empty() || args.summary_out.empty() ||
        args.per_query_out.empty()) {
        exact_usage(argv[0]);
    }
    if (args.k <= 0 || args.max_queries <= 0 || args.threads <= 0 ||
        args.train_size <= 0 || args.add_batch_size <= 0 || args.opq_niter <= 0) {
        throw std::runtime_error("positive numeric arguments required");
    }
    if (args.method == ExactMethod::RaBitQ1 && args.rabitq_centroids.empty()) {
        throw std::runtime_error("RaBitQ requires --rabitq-centroids");
    }
    if ((args.method == ExactMethod::RaBitQ1 ||
         args.method == ExactMethod::QdrantTQ1) && args.meta.empty()) {
        args.meta = args.index + ".meta";
    }
    return args;
}

std::unique_ptr<faiss::Index> load_faiss_index(const std::string& path) {
    return std::unique_ptr<faiss::Index>(faiss::read_index(path.c_str()));
}

std::unique_ptr<faiss::IndexBinaryFlat> load_binary_flat(const std::string& path) {
    faiss::IndexBinary* raw = faiss::read_index_binary(path.c_str());
    auto* flat = dynamic_cast<faiss::IndexBinaryFlat*>(raw);
    if (!flat) {
        delete raw;
        throw std::runtime_error("not an IndexBinaryFlat: " + path);
    }
    return std::unique_ptr<faiss::IndexBinaryFlat>(flat);
}

std::vector<float> exact_training_sample(const FbinMmap& train, const ExactArgs& args) {
    const int64_t rows = std::min<int64_t>(train.n, args.train_size);
    return copy_block(train, 0, rows, args.metric == Metric::IP);
}

std::unique_ptr<faiss::Index> build_or_load_faiss_codec(
        const ExactArgs& args,
        const FbinMmap& train,
        double& train_seconds,
        double& encode_seconds,
        std::string& status) {
    if (fs::exists(args.index) && !args.rebuild) {
        const auto started = std::chrono::steady_clock::now();
        auto index = load_faiss_index(args.index);
        if (index->ntotal != train.n || index->d != train.d) {
            throw std::runtime_error("Faiss codec shape mismatch: " + args.index);
        }
        train_seconds = seconds_since(started);
        encode_seconds = 0.0;
        status = "loaded";
        return index;
    }

    if (train.d % 128 != 0 &&
        (args.method == ExactMethod::PQ8M128 ||
         args.method == ExactMethod::OPQ8M128)) {
        throw std::runtime_error("PQ8-M128 requires dimension divisible by 128");
    }
    if (train.d % 8 != 0 &&
        (args.method == ExactMethod::PQ8D8 ||
         args.method == ExactMethod::OPQ8D8)) {
        throw std::runtime_error("PQ8-d/8 requires dimension divisible by 8");
    }
    const int one_bit_subquantizers = train.d / 8;
    std::unique_ptr<faiss::Index> index;
    if (args.method == ExactMethod::PQ8M128) {
        index = std::make_unique<faiss::IndexPQ>(
                train.d, 128, 8, faiss_metric(args.metric));
    } else if (args.method == ExactMethod::OPQ8M128) {
        auto* transform = new faiss::OPQMatrix(train.d, 128);
        transform->niter = args.opq_niter;
        auto* pq = new faiss::IndexPQ(
                train.d, 128, 8, faiss_metric(args.metric));
        auto* pretransform = new faiss::IndexPreTransform(transform, pq);
        pretransform->own_fields = true;
        index.reset(pretransform);
    } else if (args.method == ExactMethod::PQ8D8) {
        index = std::make_unique<faiss::IndexPQ>(
                train.d,
                one_bit_subquantizers,
                8,
                faiss_metric(args.metric));
    } else if (args.method == ExactMethod::OPQ8D8) {
        auto* transform = new faiss::OPQMatrix(
                train.d, one_bit_subquantizers);
        transform->niter = args.opq_niter;
        auto* pq = new faiss::IndexPQ(
                train.d,
                one_bit_subquantizers,
                8,
                faiss_metric(args.metric));
        auto* pretransform = new faiss::IndexPreTransform(transform, pq);
        pretransform->own_fields = true;
        index.reset(pretransform);
    } else if (args.method == ExactMethod::SQ4) {
        index = std::make_unique<faiss::IndexScalarQuantizer>(
                train.d,
                faiss::ScalarQuantizer::QT_4bit,
                faiss_metric(args.metric));
    } else if (args.method == ExactMethod::SQ8) {
        index = std::make_unique<faiss::IndexScalarQuantizer>(
                train.d,
                faiss::ScalarQuantizer::QT_8bit,
                faiss_metric(args.metric));
    } else {
        throw std::runtime_error("not a Faiss codec method");
    }

    auto sample = exact_training_sample(train, args);
    omp_set_num_threads(args.threads);
    const auto train_started = std::chrono::steady_clock::now();
    index->train(sample.size() / train.d, sample.data());
    train_seconds = seconds_since(train_started);
    std::cout << "  trained " << exact_method_name(args.method)
              << " in " << train_seconds << " s" << std::endl;

    const auto encode_started = std::chrono::steady_clock::now();
    for (int64_t begin = 0; begin < train.n; begin += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(
                train.n, begin + args.add_batch_size);
        auto block = copy_block(
                train, begin, end - begin, args.metric == Metric::IP);
        index->add(end - begin, block.data());
        if (end == train.n || end % 100000 < args.add_batch_size) {
            std::cout << "  encoded " << end << '/' << train.n << std::endl;
        }
    }
    encode_seconds = seconds_since(encode_started);
    make_parent(args.index);
    faiss::write_index(index.get(), args.index.c_str());
    status = "built";
    return index;
}

struct CustomCodec {
    std::unique_ptr<faiss::IndexBinaryFlat> index;
    RaBitQMeta rabitq_meta;
};

CustomCodec build_or_load_custom_codec(
        const ExactArgs& args,
        const FbinMmap& train,
        double& setup_seconds,
        double& encode_seconds,
        std::string& status) {
    CustomCodec result;
    if (fs::exists(args.index) && !args.rebuild) {
        const auto started = std::chrono::steady_clock::now();
        result.index = load_binary_flat(args.index);
        if (result.index->ntotal != train.n) {
            throw std::runtime_error("custom codec ntotal mismatch");
        }
        if (args.method == ExactMethod::RaBitQ1) {
            result.rabitq_meta = load_rabitq_meta(args.meta, args.metric);
        }
        setup_seconds = seconds_since(started);
        encode_seconds = 0.0;
        status = "loaded";
        return result;
    }

    Args legacy;
    legacy.metric = args.metric;
    legacy.rotation_seed = args.rotation_seed;
    size_t code_size = 0;
    std::vector<float> centroids;
    const auto setup_started = std::chrono::steady_clock::now();
    if (args.method == ExactMethod::AdcBQ1) {
        code_size = bq_code_size(train.d, args.metric);
    } else if (args.method == ExactMethod::SymBQ1) {
        code_size = symmetric_bq_code_size(train.d);
    } else if (args.method == ExactMethod::RaBitQ1) {
        constexpr int clusters = 16;
        centroids = read_fbin_matrix(args.rabitq_centroids, clusters, train.d);
        result.rabitq_meta = make_rabitq_meta(legacy, train.d, centroids);
        save_rabitq_meta(result.rabitq_meta, args.meta);
        code_size = rabitq_code_size(result.rabitq_meta.padded_dim);
    } else {
        throw std::runtime_error("not a custom codec method");
    }
    setup_seconds = seconds_since(setup_started);

    result.index = std::make_unique<faiss::IndexBinaryFlat>(code_size * 8);
    omp_set_num_threads(args.threads);
    const auto encode_started = std::chrono::steady_clock::now();
    for (int64_t begin = 0; begin < train.n; begin += args.add_batch_size) {
        const int64_t end = std::min<int64_t>(
                train.n, begin + args.add_batch_size);
        std::vector<uint8_t> codes;
        if (args.method == ExactMethod::AdcBQ1) {
            encode_bq_block(
                    train.row(begin), end - begin, train.d, args.metric, codes);
        } else if (args.method == ExactMethod::SymBQ1) {
            encode_symmetric_bq_block(
                    train.row(begin), end - begin, train.d, codes);
        } else {
            encode_rabitq_block(
                    train.row(begin),
                    end - begin,
                    train.d,
                    centroids,
                    result.rabitq_meta,
                    codes);
        }
        result.index->add(end - begin, codes.data());
        if (end == train.n || end % 100000 < args.add_batch_size) {
            std::cout << "  encoded " << end << '/' << train.n << std::endl;
        }
    }
    encode_seconds = seconds_since(encode_started);
    make_parent(args.index);
    faiss::write_index_binary(result.index.get(), args.index.c_str());
    status = "built";
    return result;
}

bool better_total_order(
        float candidate,
        faiss::idx_t candidate_id,
        float target,
        faiss::idx_t target_id,
        bool larger_is_better) {
    if (std::isnan(candidate)) return false;
    if (std::isnan(target)) return true;
    if (larger_is_better) {
        return candidate > target ||
                (candidate == target && candidate_id < target_id);
    }
    return candidate < target ||
            (candidate == target && candidate_id < target_id);
}

template <typename Factory>
std::vector<int64_t> exact_gt_ranks(
        const ExactArgs& args,
        int64_t nb,
        int dim,
        const std::vector<float>& queries,
        const std::vector<int32_t>& gt,
        int gt_dim,
        bool larger_is_better,
        Factory make_computer,
        double& scan_seconds) {
    const int nq = static_cast<int>(queries.size() / dim);
    std::vector<int64_t> ranks(static_cast<size_t>(nq) * args.k, 1);
    std::atomic<int> finished{0};
    const auto started = std::chrono::steady_clock::now();
    omp_set_num_threads(args.threads);
#pragma omp parallel
    {
        auto computer = make_computer();
        std::vector<float> target(args.k);
#pragma omp for schedule(dynamic, 1)
        for (int qi = 0; qi < nq; ++qi) {
            computer->set_query(queries.data() + static_cast<size_t>(qi) * dim);
            for (int g = 0; g < args.k; ++g) {
                const int32_t id = gt[static_cast<size_t>(qi) * gt_dim + g];
                if (id < 0 || id >= nb) {
                    throw std::runtime_error("ground-truth id outside database");
                }
                target[g] = (*computer)(id);
            }
            int64_t* query_ranks = ranks.data() + static_cast<size_t>(qi) * args.k;
            for (int64_t id = 0; id < nb; ++id) {
                const float distance = (*computer)(id);
                for (int g = 0; g < args.k; ++g) {
                    const int32_t target_id = gt[static_cast<size_t>(qi) * gt_dim + g];
                    query_ranks[g] += better_total_order(
                            distance, id, target[g], target_id, larger_is_better);
                }
            }
            const int done = ++finished;
            if (done == nq || done % 25 == 0) {
#pragma omp critical
                std::cout << "  exact queries " << done << '/' << nq << std::endl;
            }
        }
    }
    scan_seconds = seconds_since(started);
    return ranks;
}

int64_t nearest_rank_quantile(std::vector<int64_t> values, double fraction) {
    if (values.empty()) throw std::runtime_error("empty quantile input");
    std::sort(values.begin(), values.end());
    const size_t index = std::min(
            values.size() - 1,
            static_cast<size_t>(std::ceil(fraction * values.size())) - 1);
    return values[index];
}

double candidate_recall_at(const std::vector<int64_t>& ranks, int64_t budget) {
    const int64_t hits = std::count_if(
            ranks.begin(), ranks.end(),
            [&](int64_t rank) { return rank <= budget; });
    return static_cast<double>(hits) / ranks.size();
}

uintmax_t existing_file_size(const std::string& path) {
    return !path.empty() && fs::exists(path) ? fs::file_size(path) : 0;
}

void write_exact_results(
        const ExactArgs& args,
        int64_t nb,
        int dim,
        int nq,
        const std::vector<int32_t>& gt,
        int gt_dim,
        const std::vector<int64_t>& ranks,
        size_t code_size,
        double setup_seconds,
        double encode_seconds,
        double scan_seconds,
        const std::string& status) {
    std::vector<int64_t> all_ten(static_cast<size_t>(nq), 0);
    for (int qi = 0; qi < nq; ++qi) {
        const int64_t* begin = ranks.data() + static_cast<size_t>(qi) * args.k;
        all_ten[qi] = *std::max_element(begin, begin + args.k);
    }
    const int64_t r095 = nearest_rank_quantile(ranks, 0.95);
    const int64_t r099 = nearest_rank_quantile(ranks, 0.99);
    const int64_t all_p50 = nearest_rank_quantile(all_ten, 0.50);
    const int64_t all_p95 = nearest_rank_quantile(all_ten, 0.95);
    const int64_t all_p99 = nearest_rank_quantile(all_ten, 0.99);

    make_parent(args.per_query_out);
    std::ofstream per_query(args.per_query_out);
    per_query << "dataset,method,query_id,all_top10_budget";
    for (int g = 0; g < args.k; ++g) {
        per_query << ",gt_id_" << g + 1 << ",gt_rank_" << g + 1;
    }
    per_query << '\n';
    for (int qi = 0; qi < nq; ++qi) {
        per_query << args.dataset << ',' << exact_method_name(args.method) << ','
                  << qi << ',' << all_ten[qi];
        for (int g = 0; g < args.k; ++g) {
            per_query << ',' << gt[static_cast<size_t>(qi) * gt_dim + g]
                      << ',' << ranks[static_cast<size_t>(qi) * args.k + g];
        }
        per_query << '\n';
    }

    const uint64_t code_bytes_total = static_cast<uint64_t>(nb) * code_size;
    const uint64_t bq_bytes_per_vector = static_cast<uint64_t>((dim + 7) / 8);
    const uint64_t bq_budget_bytes = static_cast<uint64_t>(nb) * bq_bytes_per_vector;
    const uint64_t artifact_bytes = existing_file_size(args.index) +
            existing_file_size(args.meta);
    const uint64_t noncode_bytes = artifact_bytes > code_bytes_total
            ? artifact_bytes - code_bytes_total
            : 0;
    const uint64_t code_budget_after_metadata = bq_budget_bytes > noncode_bytes
            ? bq_budget_bytes - noncode_bytes
            : 0;
    const uint64_t resident_vectors = std::min<uint64_t>(
            nb, code_budget_after_metadata / std::max<uint64_t>(1, code_size));
    const double resident_fraction = static_cast<double>(resident_vectors) / nb;

    make_parent(args.summary_out);
    std::ofstream summary(args.summary_out);
    summary
            << "dataset,method,metric,k,nq,nb,dim,threads,status,setup_seconds,"
            << "encode_seconds,scan_seconds,exact_scan_qps,code_bytes_per_vector,"
            << "code_bytes_total,artifact_bytes,noncode_artifact_bytes,"
            << "bq_bytes_per_vector,bq_budget_bytes,resident_vectors_at_bq_budget,"
            << "resident_fraction_at_bq_budget,R0.95,R0.99,all_top10_p50,"
            << "all_top10_p95,all_top10_p99,candidate_recall_at_10,"
            << "candidate_recall_at_30,candidate_recall_at_80,"
            << "candidate_recall_at_100,candidate_recall_at_200,"
            << "candidate_recall_at_500,candidate_recall_at_1000,index_path,meta_path\n";
    summary << std::setprecision(10)
            << args.dataset << ',' << exact_method_name(args.method) << ','
            << metric_name(args.metric) << ',' << args.k << ',' << nq << ',' << nb
            << ',' << dim << ',' << args.threads << ',' << status << ','
            << setup_seconds << ',' << encode_seconds << ',' << scan_seconds << ','
            << nq / scan_seconds << ',' << code_size << ',' << code_bytes_total << ','
            << artifact_bytes << ',' << noncode_bytes << ',' << bq_bytes_per_vector
            << ',' << bq_budget_bytes << ',' << resident_vectors << ','
            << resident_fraction << ',' << r095 << ',' << r099 << ',' << all_p50
            << ',' << all_p95 << ',' << all_p99 << ','
            << candidate_recall_at(ranks, 10) << ','
            << candidate_recall_at(ranks, 30) << ','
            << candidate_recall_at(ranks, 80) << ','
            << candidate_recall_at(ranks, 100) << ','
            << candidate_recall_at(ranks, 200) << ','
            << candidate_recall_at(ranks, 500) << ','
            << candidate_recall_at(ranks, 1000) << ','
            << args.index << ',' << args.meta << '\n';

    std::cout << "RESULT method=" << exact_method_name(args.method)
              << " code_bytes=" << code_size
              << " R0.95=" << r095
              << " R0.99=" << r099
              << " all10_p95=" << all_p95
              << " resident@BQ=" << resident_fraction
              << " scan_qps=" << nq / scan_seconds << std::endl;
}

int main(int argc, char** argv) {
    try {
        const ExactArgs args = parse_exact_args(argc, argv);
        FbinMmap train(args.train);
        FbinMmap query_file(args.query);
        if (train.d != query_file.d) {
            throw std::runtime_error("train/query dimension mismatch");
        }
        std::vector<float> queries = read_queries(
                query_file, args.max_queries, args.metric == Metric::IP);
        const int nq = static_cast<int>(queries.size() / train.d);
        int32_t gt_rows = 0;
        int32_t gt_dim = 0;
        std::vector<int32_t> gt = read_ibin(
                args.gt, nq, gt_rows, gt_dim);
        if (gt_rows != nq || gt_dim < args.k) {
            throw std::runtime_error("ground-truth shape mismatch");
        }
        std::cout << "Dataset=" << args.dataset
                  << " method=" << exact_method_name(args.method)
                  << " nb=" << train.n << " nq=" << nq
                  << " d=" << train.d << std::endl;

        double setup_seconds = 0.0;
        double encode_seconds = 0.0;
        double scan_seconds = 0.0;
        std::string status;
        std::vector<int64_t> ranks;
        size_t code_size = 0;

        if (args.method == ExactMethod::PQ8M128 ||
            args.method == ExactMethod::OPQ8M128 ||
            args.method == ExactMethod::PQ8D8 ||
            args.method == ExactMethod::OPQ8D8 ||
            args.method == ExactMethod::SQ4 ||
            args.method == ExactMethod::SQ8) {
            auto index = build_or_load_faiss_codec(
                    args, train, setup_seconds, encode_seconds, status);
            auto* flat = dynamic_cast<faiss::IndexFlatCodes*>(
                    (args.method == ExactMethod::OPQ8M128 ||
                     args.method == ExactMethod::OPQ8D8)
                            ? dynamic_cast<faiss::IndexPreTransform*>(index.get())->index
                            : index.get());
            if (!flat) throw std::runtime_error("codec is not flat codes");
            code_size = flat->code_size;
            if (args.build_only) {
                std::cout << "BUILD_ONLY_COMPLETE method="
                          << exact_method_name(args.method)
                          << " code_bytes=" << code_size
                          << " index=" << args.index << std::endl;
                return 0;
            }
            const bool larger_is_better = args.metric == Metric::IP;
            ranks = exact_gt_ranks(
                    args,
                    train.n,
                    train.d,
                    queries,
                    gt,
                    gt_dim,
                    larger_is_better,
                    [&]() {
                        return std::unique_ptr<faiss::DistanceComputer>(
                                index->get_distance_computer());
                    },
                    scan_seconds);
        } else if (args.method == ExactMethod::AdcBQ1 ||
                   args.method == ExactMethod::SymBQ1 ||
                   args.method == ExactMethod::RaBitQ1) {
            auto codec = build_or_load_custom_codec(
                    args, train, setup_seconds, encode_seconds, status);
            code_size = codec.index->code_size;
            const uint8_t* codes = codec.index->xb.data();
            if (args.method == ExactMethod::AdcBQ1) {
                ranks = exact_gt_ranks(
                        args,
                        train.n,
                        train.d,
                        queries,
                        gt,
                        gt_dim,
                        false,
                        [&]() {
                            return std::unique_ptr<faiss::DistanceComputer>(
                                    new BQDistanceComputer(
                                            codes, code_size, train.d, args.metric));
                        },
                        scan_seconds);
            } else if (args.method == ExactMethod::SymBQ1) {
                ranks = exact_gt_ranks(
                        args,
                        train.n,
                        train.d,
                        queries,
                        gt,
                        gt_dim,
                        false,
                        [&]() {
                            return std::unique_ptr<faiss::DistanceComputer>(
                                    new SymmetricBQDistanceComputer(
                                            codes, code_size, train.d));
                        },
                        scan_seconds);
            } else {
                ranks = exact_gt_ranks(
                        args,
                        train.n,
                        train.d,
                        queries,
                        gt,
                        gt_dim,
                        false,
                        [&]() {
                            return std::unique_ptr<faiss::DistanceComputer>(
                                    new RaBitQDistanceComputer(
                                            codes,
                                            code_size,
                                            train.d,
                                            codec.rabitq_meta));
                        },
                        scan_seconds);
            }
        } else {
            const auto setup_started = std::chrono::steady_clock::now();
            QdrantTqIndexPtr tq;
            if (fs::exists(args.index) && fs::exists(args.meta) && !args.rebuild) {
                QdrantTqIndex* raw = qdrant_tq_load(
                        args.index.c_str(), args.meta.c_str());
                if (!raw) throw std::runtime_error(qdrant_tq_error());
                tq.reset(raw);
                status = "loaded";
                setup_seconds = seconds_since(setup_started);
            } else {
                make_parent(args.index);
                make_parent(args.meta);
                QdrantTqIndex* raw = qdrant_tq_build(
                        train.data,
                        train.n,
                        train.d,
                        args.metric == Metric::IP ? 0 : 1,
                        args.threads,
                        args.index.c_str(),
                        args.meta.c_str());
                if (!raw) throw std::runtime_error(qdrant_tq_error());
                tq.reset(raw);
                status = "built";
                encode_seconds = seconds_since(setup_started);
            }
            size_t code_bytes_total = 0;
            const uint8_t* codes = qdrant_tq_codes(
                    tq.get(), &code_bytes_total);
            code_size = qdrant_tq_code_size(tq.get());
            if (!codes || code_bytes_total != static_cast<size_t>(train.n) * code_size) {
                throw std::runtime_error("invalid Qdrant TurboQuant code buffer");
            }
            if (args.build_only) {
                std::cout << "BUILD_ONLY_COMPLETE method="
                          << exact_method_name(args.method)
                          << " code_bytes=" << code_size
                          << " index=" << args.index << std::endl;
                return 0;
            }
            ranks = exact_gt_ranks(
                    args,
                    train.n,
                    train.d,
                    queries,
                    gt,
                    gt_dim,
                    false,
                    [&]() {
                        return std::unique_ptr<faiss::DistanceComputer>(
                                new QdrantTqDirectDistanceComputer(
                                        tq.get(), codes, code_size, train.d, args.metric));
                    },
                    scan_seconds);
        }

        write_exact_results(
                args,
                train.n,
                train.d,
                nq,
                gt,
                gt_dim,
                ranks,
                code_size,
                setup_seconds,
                encode_seconds,
                scan_seconds,
                status);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
