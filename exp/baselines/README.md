# Baselines

Every system the paper compares against is listed below with the exact upstream commit, our changes, and how the runner
in `exp/scripts` finds it. `patches/<system>/BASE` pins the commit; `patches/<system>/*.patch` is the full diff of the
tree that produced the results (tracked files; untracked sources are copied under `untracked/` where they exist).

| Baseline (paper name) | Upstream, commit | Our changes | Build used | Runner (in `exp/scripts`) |
|---|---|---|---|---|
| Faiss PQ / OPQ, TurboQuant-1bit (search side), Sym-BQ | facebookresearch/faiss `46def469` | none needed (`patches/faiss/README.md`) | `faiss_avx2`, `-DFAISS_OPT_LEVEL=avx2`, Release | `run_faiss_hnsw_bq_budget_paged_codes.py` → `faiss_hnsw_bq_budget_paged_codes.cpp` |
| TurboQuant-1bit codec | Qdrant `222b2e4` (v1.20.0) `lib/quantization` | `patches/qdrant/*` (export query/storage planes) + Rust FFI `qdrant_tq_ffi/` | `cargo build --release` | `run_faiss_exact_matched_budget_compare.py` |
| HNSW-RaBitQ-1bit (adaptive rerank), RaBitQ-7bit | RaBitQ-Library `v0.5.1` (`44f8a607`) | `patches/rabitq_v051/rabitq_v051_adaptive_odirect.cumulative.patch` | `pip install .` | `run_nq_quora_c16_rabitq.py`, `run_diskveclab_full_matrix_rabitq{1_adaptive,7}.py` |
| RaBitQ headers for the FAISS drivers | RaBitQ-Library `7c2d0d71` | `patches/rabitq-library/include.patch` | header-only | `-I<RaBitQ-Library>/include` |
| DiskANN-PQ, DiskANN-RaBitQ-1bit | alibaba/DiskVecLab `dc8befa` (`src/DiskAnn`) | `patches/diskveclab/DiskVecLab.patch` (+ `untracked/`) | `src/DiskAnn/build-release`, Release | `run_diskveclab_full_matrix_diskann.py`, `run_onebit_disk_baselines_t1.py` |
| SPANN (PQ128 + FP@80) | microsoft/SPTAG `cafdd0a` | `patches/sptag/SPTAG.patch`; driver `spann_odirect_rerank_benchmark.cpp` | `build/`, Release, `GPU=OFF` | `run_nq_quora_spann_pipeann.py` |
| PipeANN-RaBitQ-1bit | thustorage/PipeANN `ad9d7be` | `patches/pipeann/PipeANN.patch` | `build-uring-current`, Release, `USE_TCMALLOC=ON`, io_uring | `run_nq_quora_spann_pipeann.py` |
| RaBitQ-SSD 9bit | VectorDB-NTU/RaBitQ-SSD `7ab80aa` | `patches/rabitq-ssd/RaBitQ-SSD.patch` | `build/`, Release | `run_diskveclab_full_matrix_mrq_rabitq_ssd.py` |
| MRQ | RESQ `79ef6e1` | `patches/resq/RESQ.patch` | `_resq_build`, Release (built by the runner) | `run_diskveclab_full_matrix_mrq_rabitq_ssd.py` |
| ES-BBQ | Elasticsearch **9.5.5** (`linux-x86_64` tarball, sha512 `36bf5d68…709ced`) | none | official distribution | `run_diskveclab_full_matrix_elasticsearch.py` → `run_nq_quora_elasticsearch_bbq_rerank80.py` |

Repository URLs for each base are in the `BASE` files.

## Elasticsearch configuration

- **Node:**
  - single node, `-Ediscovery.type=single-node`
  - security off (`xpack.security.enabled=false`)
  - `ES_JAVA_OPTS="-Xms16g -Xmx16g"`
- **Index:**
  - one shard, `refresh_interval=-1` while loading, then force-merged to 1 segment
  - `dense_vector` with `index_options: {type: bbq_hnsw, m: 24, ef_construction: 200}`, dot product on unit vectors
- **Search:**
  - one sequential client
  - `k=10`, with `num_candidates` swept
  - `rescore_vector.oversample = 8.0` (80 full-precision vectors reranked per query)

## Applying a patch

```bash
git clone <repo from BASE> && cd <repo> && git checkout <commit from BASE>
git apply /path/to/BBQ/exp/baselines/patches/<system>/<System>.patch
cp -r /path/to/BBQ/exp/baselines/patches/<system>/untracked/. .     # DiskVecLab only
```

`patches/diskveclab/untracked/src/BBQ` is a BBQ adapter for DiskVecLab that was developed alongside. It is **not** used
by any paper figure; the paper's BBQ numbers come from `bbq/`.
