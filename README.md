# BBQ: Blessing ANN Search with Binary Quantization on Transformer Embeddings

Artifact for the PVLDB Experiment, Analysis & Benchmark paper.

BBQ keeps two things in DRAM: sign-only 1-bit codes and an HNSW graph built with FP32 distances. It traverses the
graph with asymmetric distance computation, comparing the FP32 query against the 1-bit codes through a compact
SIMD lookup table. After traversal it reranks a calibrated, per-query set of candidates, reading their FP32 vectors
from SSD in a single `io_uring` batch.

```
bbq/   the system: ADC-BQ HNSW search, calibrated SSD rerank, online insertion, colocation  → bbq/README.md
exp/   every experiment in Sec. 4–6: results, runners, figures, and a claim checker           → exp/README.md
```

## Quick start

```bash
make verify          # check all 96 numbers quoted in the paper against the stored results (no deps but g++)
make plots           # re-draw Figs 5, 8–20 from the stored results (gnuplot >= 5.4)

source exp/env.sh    # set FAISS_ROOT / FAISS_BUILD / BLAS_LIB
make test            # AVX-512 sign-packing unit test
make smoke           # ~1 min: build BBQ, search, calibrate tau, serve with R(q) on synthetic data
```

`make smoke` prints, for example:

```
== 1. ADC-BQ traversal, no rerank vs fixed R=40
ef=128 round=1/1 recall@10=0.557
ef=128 round=1/1 recall@10=0.939
== 2. calibration pool (1000 queries) -> tau
tau = 1.0663265221384595
== 3. calibrated query-adaptive rerank
calibrated: Recall@10 = 0.9965, mean R(q) = 80.5 (min 36, max 128)
SMOKE OK
```

## Paper → experiment

See [`exp/MATRIX.md`](exp/MATRIX.md). Each `exp/figNN_*/README.md` names:
- the runner scripts,
- the result files,
- the claims checked against them,
- any caveats.

## Requirements

- **Verify:** a C++17 compiler.
- **Plots:** gnuplot ≥ 5.4 with the Liberation Sans font, and python3 with numpy.
- **BBQ:** Linux with liburing, g++ ≥ 11, OpenMP, OpenBLAS, FAISS 1.14.x (AVX2 build), and an AVX-512 CPU.
- **Full reproduction:** the paper's testbed and datasets (see `exp/README.md`).
