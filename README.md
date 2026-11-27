# BBQ: Blessing ANN Search with Binary Quantization on Transformer Embeddings

Artifact for the PVLDB Experiment, Analysis & Benchmark paper.

BBQ keeps two things in DRAM: sign-only 1-bit codes and an HNSW graph built with FP32 distances. It traverses the
graph with asymmetric distance computation, comparing the FP32 query against the 1-bit codes through a compact
SIMD lookup table. After traversal it reranks a calibrated, per-query set of candidates, reading their FP32 vectors
from SSD in a single `io_uring` batch.

```
bbq/   the system: ADC-BQ HNSW search, calibrated SSD rerank, online insertion, colocation  → bbq/README.md
exp/   every experiment in Secs 1–6 and Appendix B: results, runners, figures, a claim checker → exp/README.md
```

## Quick start

```bash
make verify          # check 312 paper claims (Secs 1-6, Appendix B) against the stored results; g++ only
make plots           # re-draw Figs 5, 7-21 from the stored results (gnuplot >= 5.4)

# BBQ itself: upstream FAISS, no fork (exp/baselines/patches/faiss/README.md has the exact commit and cmake line)
source exp/env.sh    # set FAISS_ROOT / FAISS_BUILD / BLAS_LIB
make test            # AVX-512 sign-packing unit test
make smoke           # ~1 min: build BBQ, search, calibrate tau, serve with R(q), then the same from disk via O_DIRECT/io_uring
```

`make smoke` prints, for example:

```
== 1. ADC-BQ traversal, no rerank vs fixed R=40
ef=128 round=1/1 recall@10=0.557
ef=128 round=1/1 recall@10=0.939
== 2. calibration pool (1000 queries) -> tau
tau = 1.0663265221384595
== 3. calibrated query-adaptive rerank
recall@10=0.9965 ... adaptive_rerank_mean=80.46 min=36 p95=120 p99=127.01 max=128
== 4. same calibrated rerank, FP32 vectors read from disk with O_DIRECT (Fig. 16 modes)
sequential           Recall@10=0.9965  QPS=   111.6  4KiB pages/query=80.5
async-batch          Recall@10=0.9965  QPS=  1183.0  4KiB pages/query=80.5
async-batch+overlap  Recall@10=0.9965  QPS=  1442.1  4KiB pages/query=80.5
calibrated: Recall@10 = 0.9965, mean R(q) = 80.5 (min 36, max 128)
SMOKE OK
```

## Paper → experiment

See [`exp/MATRIX.md`](exp/MATRIX.md). Each `exp/figNN_*/README.md` names:
- the runner scripts,
- the result files,
- the claims checked against them,
- any caveats.

The places where the manuscript and the stored data disagree are listed in
[`exp/README.md` → "Paper ↔ data"](exp/README.md#paper--data). That list includes the two claims `verify` reports
as failing. Datasets are described in [`exp/DATA.md`](exp/DATA.md); they are not part of the repository.

## Requirements

- **Verify:** a C++17 compiler.
- **Plots:** gnuplot ≥ 5.4 with the Liberation Sans font, and python3 with numpy.
- **BBQ:** Linux with liburing, g++ ≥ 11, OpenMP, OpenBLAS, upstream FAISS `46def469` (AVX2 build), and an AVX-512 CPU.
- **Full reproduction:** the paper's testbed and datasets (see `exp/README.md`).
