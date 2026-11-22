#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include <liburing.h>
#include <sys/stat.h>
#include <unistd.h>

#include "inc/Core/Common/QueryResultSet.h"
#include "inc/Core/SPANN/Index.h"
#include "inc/Core/VectorIndex.h"

namespace {

constexpr uint64_t kPageBytes = 4096;
constexpr uint64_t kFpHeaderBytes = 4096;
using Clock = std::chrono::steady_clock;

double elapsed_us(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

template <typename T>
struct Matrix {
  uint32_t rows = 0;
  uint32_t dim = 0;
  std::vector<T> data;
};

template <typename T>
Matrix<T> load_matrix(const std::string& path) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) throw std::runtime_error("cannot open " + path);
  Matrix<T> out;
  if (std::fread(&out.rows, sizeof(out.rows), 1, fp) != 1 ||
      std::fread(&out.dim, sizeof(out.dim), 1, fp) != 1) {
    std::fclose(fp);
    throw std::runtime_error("cannot read matrix header from " + path);
  }
  const uint64_t elements = static_cast<uint64_t>(out.rows) * out.dim;
  out.data.resize(elements);
  if (std::fread(out.data.data(), sizeof(T), elements, fp) != elements) {
    std::fclose(fp);
    throw std::runtime_error("short matrix payload in " + path);
  }
  if (std::fgetc(fp) != EOF) {
    std::fclose(fp);
    throw std::runtime_error("trailing bytes in " + path);
  }
  std::fclose(fp);
  return out;
}

Matrix<uint32_t> load_truth(const std::string& path) {
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (fp == nullptr) throw std::runtime_error("cannot open " + path);
  Matrix<uint32_t> out;
  if (std::fread(&out.rows, sizeof(out.rows), 1, fp) != 1 ||
      std::fread(&out.dim, sizeof(out.dim), 1, fp) != 1) {
    std::fclose(fp);
    throw std::runtime_error("cannot read truth header from " + path);
  }
  const uint64_t elements = static_cast<uint64_t>(out.rows) * out.dim;
  out.data.resize(elements);
  if (std::fread(out.data.data(), sizeof(uint32_t), elements, fp) != elements) {
    std::fclose(fp);
    throw std::runtime_error("short truth payload in " + path);
  }
  const long ids_end = std::ftell(fp);
  if (ids_end < 0 || std::fseek(fp, 0, SEEK_END) != 0) {
    std::fclose(fp);
    throw std::runtime_error("cannot inspect truth file " + path);
  }
  const long file_end = std::ftell(fp);
  std::fclose(fp);
  const uint64_t trailing = static_cast<uint64_t>(file_end - ids_end);
  if (trailing != 0 && trailing != elements * sizeof(float)) {
    throw std::runtime_error("unexpected truth payload size in " + path);
  }
  return out;
}

void normalize(Matrix<float>& matrix) {
  for (uint32_t i = 0; i < matrix.rows; ++i) {
    float* row = matrix.data.data() + static_cast<uint64_t>(i) * matrix.dim;
    double norm2 = 0.0;
    for (uint32_t j = 0; j < matrix.dim; ++j) norm2 += static_cast<double>(row[j]) * row[j];
    const float inv = norm2 > 0.0 ? static_cast<float>(1.0 / std::sqrt(norm2)) : 1.0f;
    for (uint32_t j = 0; j < matrix.dim; ++j) row[j] *= inv;
  }
}

struct Candidate {
  uint32_t id = 0;
  float estimate = 0.0f;
  float lower_bound = 0.0f;
  bool exact_in_memory = false;
};

struct ExactScore {
  float distance = 0.0f;
  uint32_t id = 0;
};

bool score_less(const ExactScore& lhs, const ExactScore& rhs) {
  return lhs.distance < rhs.distance ||
         (lhs.distance == rhs.distance && lhs.id < rhs.id);
}

struct QueryMeasurement {
  double latency_us = 0.0;
  double head_us = 0.0;
  double posting_us = 0.0;
  double rerank_io_us = 0.0;
  double rerank_cpu_us = 0.0;
  double recall = 0.0;
  double posting_requests = 0.0;
  double posting_pages_4k = 0.0;
  double fp_vectors = 0.0;
  double fp_pages_4k = 0.0;
  double total_pages_4k = 0.0;
};

class DirectFpReader {
 public:
  DirectFpReader(const std::string& path, uint32_t dim, uint32_t queue_depth)
      : dim_(dim),
        record_bytes_(((static_cast<uint64_t>(dim) * sizeof(float) + kPageBytes - 1) /
                       kPageBytes) * kPageBytes),
        queue_depth_(queue_depth) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (fd_ < 0) {
      throw std::runtime_error("O_DIRECT open failed for " + path + ": " + std::strerror(errno));
    }
    const int rc = io_uring_queue_init(queue_depth_, &ring_, 0);
    if (rc < 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("io_uring_queue_init failed: " + std::string(std::strerror(-rc)));
    }
    ring_ready_ = true;
    const uint64_t buffer_bytes = record_bytes_ * queue_depth_;
    if (posix_memalign(&buffer_, kPageBytes, buffer_bytes) != 0 || buffer_ == nullptr) {
      throw std::runtime_error("aligned FP buffer allocation failed");
    }
  }

  DirectFpReader(const DirectFpReader&) = delete;
  DirectFpReader& operator=(const DirectFpReader&) = delete;

  ~DirectFpReader() {
    if (buffer_ != nullptr) std::free(buffer_);
    if (ring_ready_) io_uring_queue_exit(&ring_);
    if (fd_ >= 0) ::close(fd_);
  }

  uint64_t pages_per_vector() const { return record_bytes_ / kPageBytes; }

  std::vector<ExactScore> read_and_score(const float* query,
                                         const std::vector<uint32_t>& ids,
                                         double& io_us,
                                         double& cpu_us) {
    if (ids.empty()) return {};
    if (ids.size() > queue_depth_) throw std::runtime_error("FP batch exceeds io_uring queue depth");
    const auto io_start = Clock::now();
    for (uint32_t slot = 0; slot < ids.size(); ++slot) {
      io_uring_sqe* sqe = io_uring_get_sqe(&ring_);
      if (sqe == nullptr) throw std::runtime_error("io_uring SQ exhausted");
      void* dst = static_cast<char*>(buffer_) + static_cast<uint64_t>(slot) * record_bytes_;
      const uint64_t offset = kFpHeaderBytes + static_cast<uint64_t>(ids[slot]) * record_bytes_;
      io_uring_prep_read(sqe, fd_, dst, record_bytes_, offset);
      io_uring_sqe_set_data64(sqe, slot);
    }
    const int submitted = io_uring_submit(&ring_);
    if (submitted != static_cast<int>(ids.size())) throw std::runtime_error("short io_uring submit");
    for (uint32_t done = 0; done < ids.size(); ++done) {
      io_uring_cqe* cqe = nullptr;
      const int rc = io_uring_wait_cqe(&ring_, &cqe);
      if (rc < 0) throw std::runtime_error("io_uring_wait_cqe failed");
      const int result = cqe->res;
      io_uring_cqe_seen(&ring_, cqe);
      if (result != static_cast<int>(record_bytes_)) {
        throw std::runtime_error("short O_DIRECT FP read: " + std::to_string(result));
      }
    }
    const auto io_end = Clock::now();
    io_us += elapsed_us(io_start, io_end);

    const auto cpu_start = Clock::now();
    std::vector<ExactScore> scores;
    scores.reserve(ids.size());
    for (uint32_t slot = 0; slot < ids.size(); ++slot) {
      const float* vector = reinterpret_cast<const float*>(
          static_cast<const char*>(buffer_) + static_cast<uint64_t>(slot) * record_bytes_);
      float distance = 0.0f;
#pragma omp simd reduction(+ : distance)
      for (uint32_t j = 0; j < dim_; ++j) {
        const float delta = query[j] - vector[j];
        distance += delta * delta;
      }
      scores.push_back({distance, ids[slot]});
    }
    const auto cpu_end = Clock::now();
    cpu_us += elapsed_us(cpu_start, cpu_end);
    return scores;
  }

 private:
  uint32_t dim_ = 0;
  uint64_t record_bytes_ = 0;
  uint32_t queue_depth_ = 0;
  int fd_ = -1;
  io_uring ring_{};
  bool ring_ready_ = false;
  void* buffer_ = nullptr;
};

std::vector<uint32_t> adaptive_refine(const float* query,
                                      std::vector<Candidate> candidates,
                                      uint32_t k,
                                      DirectFpReader& reader,
                                      double& io_us,
                                      double& cpu_us,
                                      uint32_t& fp_vectors) {
  if (candidates.size() < k) throw std::runtime_error("too few SPANN candidates");
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
    return lhs.estimate < rhs.estimate ||
           (lhs.estimate == rhs.estimate && lhs.id < rhs.id);
  });

  std::vector<uint32_t> seed_ids;
  seed_ids.reserve(k);
  std::vector<ExactScore> exact;
  exact.reserve(candidates.size());
  for (uint32_t i = 0; i < k; ++i) {
    if (candidates[i].exact_in_memory) {
      exact.push_back({candidates[i].estimate, candidates[i].id});
    } else {
      seed_ids.push_back(candidates[i].id);
    }
  }
  auto seed_scores = reader.read_and_score(query, seed_ids, io_us, cpu_us);
  exact.insert(exact.end(), seed_scores.begin(), seed_scores.end());
  fp_vectors = static_cast<uint32_t>(seed_ids.size());
  std::sort(exact.begin(), exact.end(), score_less);
  ExactScore kth = exact[k - 1];

  std::vector<uint32_t> additional_ids;
  additional_ids.reserve(candidates.size() - k);
  for (size_t i = k; i < candidates.size(); ++i) {
    const Candidate& candidate = candidates[i];
    if (candidate.exact_in_memory) {
      const ExactScore score{candidate.estimate, candidate.id};
      if (score_less(score, kth)) exact.push_back(score);
    } else if (candidate.lower_bound < kth.distance ||
        (candidate.lower_bound == kth.distance && candidate.id < kth.id)) {
      additional_ids.push_back(candidate.id);
    }
  }
  auto additional = reader.read_and_score(query, additional_ids, io_us, cpu_us);
  fp_vectors += static_cast<uint32_t>(additional_ids.size());
  exact.insert(exact.end(), additional.begin(), additional.end());
  std::partial_sort(exact.begin(), exact.begin() + k, exact.end(), score_less);
  std::vector<uint32_t> output(k);
  for (uint32_t i = 0; i < k; ++i) output[i] = exact[i].id;
  return output;
}

double recall_at_k(const std::vector<uint32_t>& ids, const uint32_t* truth, uint32_t k) {
  std::unordered_set<uint32_t> expected;
  expected.reserve(k * 2);
  for (uint32_t i = 0; i < k; ++i) expected.insert(truth[i]);
  uint32_t hits = 0;
  for (uint32_t id : ids) hits += expected.count(id) != 0;
  return static_cast<double>(hits) / k;
}

template <typename Getter>
void emit_distribution(const std::vector<QueryMeasurement>& values,
                       const char* metric,
                       Getter getter) {
  std::vector<double> sorted;
  sorted.reserve(values.size());
  double sum = 0.0;
  for (const auto& value : values) {
    const double item = getter(value);
    sorted.push_back(item);
    sum += item;
  }
  std::sort(sorted.begin(), sorted.end());
  const auto pick = [&](double q) {
    return sorted[static_cast<size_t>(q * static_cast<double>(sorted.size() - 1))];
  };
  std::cout << std::setprecision(12)
            << "SPANN_RABITQ_QUERY_STATS metric=" << metric
            << " mean=" << sum / sorted.size()
            << " min=" << sorted.front()
            << " max=" << sorted.back()
            << " p95=" << pick(0.95)
            << " p99=" << pick(0.99) << '\n';
}

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <index_dir> <query.fbin> <truth.ibin> <aligned_fp.bin>"
               " <threads> <candidate_n> <posting_page_limit> <k>\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) {
    usage(argv[0]);
    return 2;
  }
  try {
    const std::string index_dir = argv[1];
    const std::string query_path = argv[2];
    const std::string truth_path = argv[3];
    const std::string fp_path = argv[4];
    const uint32_t threads = std::stoul(argv[5]);
    const uint32_t candidate_n = std::stoul(argv[6]);
    const uint32_t posting_page_limit = std::stoul(argv[7]);
    const uint32_t k = std::stoul(argv[8]);
    if (threads == 0 || k == 0 || candidate_n < k) throw std::runtime_error("invalid arguments");

    Matrix<float> queries = load_matrix<float>(query_path);
    Matrix<uint32_t> truth = load_truth(truth_path);
    if (queries.rows != truth.rows || truth.dim < k) {
      throw std::runtime_error("query/truth shape mismatch");
    }
    normalize(queries);

    std::shared_ptr<SPTAG::VectorIndex> generic_index;
    if (SPTAG::VectorIndex::LoadIndex(index_dir, generic_index) != SPTAG::ErrorCode::Success ||
        generic_index == nullptr) {
      throw std::runtime_error("SPANN index load failed");
    }
    auto* index = dynamic_cast<SPTAG::SPANN::Index<float>*>(generic_index.get());
    if (index == nullptr || index->m_pQuantizer == nullptr ||
        !index->m_pQuantizer->isRaBitQ()) {
      throw std::runtime_error("expected a Float SPANN index with RaBitQ quantizer");
    }
    if (index->m_pQuantizer->ReconstructDim() != static_cast<SPTAG::DimensionType>(queries.dim)) {
      throw std::runtime_error("query dimension does not match RaBitQ quantizer");
    }
    index->SetParameter("SearchInternalResultNum", std::to_string(candidate_n).c_str(), "BuildSSDIndex");
    index->SetParameter("SearchPostingPageLimit", std::to_string(posting_page_limit).c_str(), "BuildSSDIndex");
    index->SetParameter("NumberOfThreads", std::to_string(threads).c_str(), "BuildSSDIndex");

    std::vector<SPTAG::QueryResult> results(
        queries.rows, SPTAG::QueryResult(nullptr, candidate_n, false));
    std::vector<SPTAG::SPANN::SearchStats> engine_stats(queries.rows);
    std::vector<QueryMeasurement> measurements(queries.rows);
    std::atomic<uint32_t> next_query{0};
    std::atomic<bool> failed{false};
    std::string failure_message;
    std::mutex failure_mutex;
    const auto wall_start = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (uint32_t worker_id = 0; worker_id < threads; ++worker_id) {
      workers.emplace_back([&, worker_id]() {
        try {
          DirectFpReader reader(fp_path, queries.dim, candidate_n);
          while (!failed.load(std::memory_order_relaxed)) {
            const uint32_t i = next_query.fetch_add(1);
            if (i >= queries.rows) break;
            const float* query = queries.data.data() + static_cast<uint64_t>(i) * queries.dim;
            auto& query_result =
                reinterpret_cast<SPTAG::COMMON::QueryResultSet<float>&>(results[i]);
            const auto total_start = Clock::now();
            query_result.SetTarget(query, index->m_pQuantizer);
            query_result.Reset();
            const auto head_start = total_start;
            if (index->GetMemoryIndex()->SearchIndex(results[i]) != SPTAG::ErrorCode::Success) {
              throw std::runtime_error("SPANN head search failed");
            }
            const auto head_end = Clock::now();
            if (index->SearchDiskIndex(results[i], &engine_stats[i]) != SPTAG::ErrorCode::Success) {
              throw std::runtime_error("SPANN posting search failed");
            }
            const auto posting_end = Clock::now();

            std::vector<Candidate> candidates;
            candidates.reserve(candidate_n);
            std::unordered_set<uint32_t> seen;
            for (uint32_t j = 0; j < candidate_n; ++j) {
              const auto* item = results[i].GetResult(j);
              if (item == nullptr || item->VID < 0) continue;
              const uint32_t id = static_cast<uint32_t>(item->VID);
              if (!seen.insert(id).second) continue;
              const bool exact_in_memory =
                  item->LowerBound <= -SPTAG::MaxDist * 0.5f;
              candidates.push_back({id, item->Dist, item->LowerBound, exact_in_memory});
            }
            if (candidates.size() < k) throw std::runtime_error("SPANN returned fewer than k candidates");

            double rerank_io_us = 0.0;
            double rerank_cpu_us = 0.0;
            uint32_t fp_vectors = 0;
            const auto final_ids = adaptive_refine(
                query, std::move(candidates), k, reader,
                rerank_io_us, rerank_cpu_us, fp_vectors);
            const auto total_end = Clock::now();

            auto& out = measurements[i];
            out.latency_us = elapsed_us(total_start, total_end);
            out.head_us = elapsed_us(head_start, head_end);
            out.posting_us = elapsed_us(head_end, posting_end);
            out.rerank_io_us = rerank_io_us;
            out.rerank_cpu_us = rerank_cpu_us;
            out.posting_requests = engine_stats[i].m_diskIOCount;
            out.posting_pages_4k = engine_stats[i].m_diskAccessCount;
            out.fp_vectors = fp_vectors;
            out.fp_pages_4k = fp_vectors * reader.pages_per_vector();
            out.total_pages_4k = out.posting_pages_4k + out.fp_pages_4k;
            out.recall = recall_at_k(
                final_ids, truth.data.data() + static_cast<uint64_t>(i) * truth.dim, k);
            results[i].CleanQuantizedTarget();
          }
        } catch (const std::exception& error) {
          failed.store(true);
          std::lock_guard<std::mutex> lock(failure_mutex);
          if (failure_message.empty()) failure_message = error.what();
        }
      });
    }
    for (auto& worker : workers) worker.join();
    const auto wall_end = Clock::now();
    if (failed.load()) throw std::runtime_error(failure_message);

    const double wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    const double recall = std::accumulate(
        measurements.begin(), measurements.end(), 0.0,
        [](double sum, const QueryMeasurement& value) { return sum + value.recall; }) /
        measurements.size();
    std::cout << std::setprecision(12)
              << "SPANN_RABITQ_RESULT threads=" << threads
              << " candidate_n=" << candidate_n
              << " posting_page_limit=" << posting_page_limit
              << " k=" << k
              << " nq=" << queries.rows
              << " dim=" << queries.dim
              << " qps=" << static_cast<double>(queries.rows) / wall_seconds
              << " recall_at_k=" << recall
              << " wall_seconds=" << wall_seconds << '\n';

#define EMIT(field) emit_distribution(measurements, #field, [](const QueryMeasurement& m) { return m.field; })
    EMIT(latency_us);
    EMIT(head_us);
    EMIT(posting_us);
    EMIT(rerank_io_us);
    EMIT(rerank_cpu_us);
    EMIT(posting_requests);
    EMIT(posting_pages_4k);
    EMIT(fp_vectors);
    EMIT(fp_pages_4k);
    EMIT(total_pages_4k);
#undef EMIT
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
