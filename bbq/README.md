# BBQ core

Sign-only 1-bit codes and an FP32-built HNSW graph in DRAM, asymmetric distance computation (ADC) against the
full-precision query, and a calibrated, query-adaptive FP32 rerank read from SSD in one `io_uring` batch after traversal.

```
bbq/
├── build.sh                      build: search | dynamic | test
├── src/
│   ├── bbq_hnsw.cpp              everything on the query path (Sec. 5): build, ADC search, calibrated SSD rerank,
│   │                             overlap, memory-constrained paged graph + colocation
│   ├── adcbq_sign_packing.hpp    sign-bit packing (scalar + AVX-512)
│   └── bbq_dynamic_insert.cpp    online insertion (Sec. 6.8); textually includes bbq_hnsw.cpp
├── calibration/calibrate.py      Theorem 1: fit the shared threshold tau from a calibration pool
├── tests/test_adcbq_sign_packing.cpp
└── docs/IMPLEMENTATION.md        code map and the overlap schedules in detail
```

## Build

```bash
export FAISS_ROOT=/path/to/faiss            # FAISS 1.14.x, built with AVX2 (libfaiss_avx2)
export FAISS_BUILD=$FAISS_ROOT/build/faiss  # directory holding libfaiss_avx2.so
export BLAS_LIB=/path/to/openblas/lib       # optional
./bbq/build.sh            # -> build/bbq_hnsw
./bbq/build.sh test       # -> build/test_adcbq_sign_packing
./bbq/build.sh dynamic    # -> build/bbq_dynamic_insert (also builds the TurboQuant FFI via cargo)
```

Needs g++ ≥ 11, OpenMP, liburing, OpenBLAS, and an AVX-512 CPU (`-march=native`). `PAGE_BYTES=8192|16384` changes
the graph page size of the memory-constrained mode (Fig. 20). `make smoke` at the repo root runs the whole path on
synthetic data in about a minute.

## Where each design element lives (`src/bbq_hnsw.cpp`)

| Paper | Code |
|---|---|
| Sign codes, PGVector bit order (dim *i* → byte *i/8*, bit 7−*i*%8) | `pack_pgvector_sign_bits`, `adcbq_sign_packing.hpp` |
| §5.1 query LUT: 16 × int16 per 4 dims, per-query scale = 32767 / max block L1 | `PackedAsymBQDistanceComputer::set_query_fp32_*` |
| §5.1 SIMD ADC, 16 neighbours per AVX-512 batch | `distance_to_code_*`, `search_one_packed_inline` |
| §5.2 FP32-built graph, 1-bit storage | `build_fp_graph_bq_storage_index` (`--graph-build fp`); `--graph-build binary` for Fig. 8 |
| §5.3 calibrated selection C_τ(q) | `select_calibrated_threshold_rerank_candidates` (`--calibrated-threshold-rerank-tau`) |
| §5.3 calibration pool for fitting τ | `--candidate-only --no-rerank --candidates-out pool.csv` |
| §5.4 post-traversal batched O_DIRECT reads | `--fp-rerank-source odirect-uring --fp-rerank-qd N` |
| §5.4 overlap with the next query's traversal | `--fp-rerank-overlap-queries` |
| §6.9 paged graph under a DRAM budget, colocation | `--paged-index-path --index-page-cache-mib`, `build_colocation_physical_rows` (`--colocation-*`) |

The paper configuration is `--graph-build fp --adc-mode int16 --lut-mode nibble --search-impl inline --m 24
--ef-construction 200`. FP32 vectors for the SSD path are expected 4 KiB-aligned (`*.fp32.header4096.fbin`).

The source also carries flags for experiments outside this paper (`--adaptive-cfr*`, `--xflag-*`,
`--early-abandon`); they are off by default and do not affect the paper configuration.

## Calibration (Theorem 1)

```bash
# 1. dump the candidate pool L(q) of m calibration queries with the frozen search configuration
build/bbq_hnsw ... --query cal.fbin --efs 256 --rerank-n 256 --candidate-only --no-rerank --candidates-out pool.csv
# 2. tau_hat = T_(l),  l = ceil((m+1)(1-rho))
python3 bbq/calibration/calibrate.py --pool pool.csv --queries cal.fbin --base base.fbin --ef 256 --target 0.99
# 3. serve
build/bbq_hnsw ... --efs 256 --rerank-n 256 --calibrated-threshold-rerank-tau <tau> \
    --fp-rerank-source odirect-uring --fp-rerank-path base.fp32.header4096.fbin
```

`calibrate.py` is a standalone port of `fit_pool_top10_conformal` (`exp/scripts/run_adcbq_vs_fp_hnsw_fullvec_ram.py`),
which produced the paper's thresholds. On the MS MARCO Qwen-1024 calibration pool (ef = 256, m = 1000, seed 11) it
returns τ = 1.63409 and mean R = 53.248, against τ = 1.63412 and mean R = 53.248 in the paper run. The pool CSV
stores ADC distances to 6 significant digits, which accounts for the gap in τ.
Calibration needs m ≥ 1/ρ − 1 queries (99 at target 0.99); otherwise it reports failure.
