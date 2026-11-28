# RaBitQ-Library v0.5.1

One pinned tree serves every use of RaBitQ-Library ([VectorDB-NTU/RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library))
in the artifact: tag `v0.5.1` (`44f8a607`) plus `rabitq_v051_adaptive_odirect.cumulative.patch`.

| Use | How |
|---|---|
| **RaBitQ baselines.** Every HNSW-RaBitQ-1bit (adaptive lower-bound FP32 rerank from SSD) and RaBitQ-7bit number in the paper (Figs 5, 9/20/21, 10, 16–17, 18; `supplementary/ram_bbq_rbq7`) | `pip install .` → Python module `_rabitqlib` |
| **Headers for the FAISS drivers** in `exp/scripts` (`faiss_hnsw_lowbit_compare.cpp`, `faiss_hnsw_bq_budget_paged_codes.cpp`), which run TurboQuant-1bit, PQ, OPQ and Sym-BQ | `-I<tree>/include`; `exp/tools/setup_workspace.sh` links it as `exp/include` |

```bash
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git && cd RaBitQ-Library
git checkout v0.5.1                                    # 44f8a607741e9e54f6dda3b38c325071e808aa45
git apply /path/to/BBQ/exp/baselines/patches/rabitq-library/rabitq_v051_adaptive_odirect.cumulative.patch
pip install .                                          # Python binding _rabitqlib with the entries below
```

## The patch

`rabitq_v051_adaptive_odirect.cumulative.patch` is the final source state (2026-11-17) of the binding used by the
paper runs, as a single diff against v0.5.1. It touches `include/rabitqlib/index/hnsw/hnsw.hpp` and
`python_bindings/hnsw_bindings.cpp`, and adds:

- **`search_1bit_adaptive_fp_odirect`:** the 1-bit estimator / lower-bound policy of RaBitQ. Only the accepted
  candidates are read from the `.fbin`, with synchronous aligned `O_DIRECT` `pread`.
- **Both `.fbin` layouts:** the legacy 8-byte header and the 4 KiB-aligned layout, inferred from the file size
  (`fbin_data_offset`).
- **Counters:** per-query latency (`query_latency_ns`) and FP32 read counts.
- **Online add,** for the insertion experiment (Fig 18).

`notes/` keeps the original per-step change notes:

- The `*.ctx.patch` files there are context-only and do not apply with `patch`; they document how the cumulative
  diff was built up.
- `03_io_wait_timing` (the I/O-wait split for Fig 5) was a separate build (2026-11-11) and is **not** part of the
  cumulative patch.

## The FAISS drivers and the earlier header tree

- **What the drivers were built with.** The FAISS drivers were originally compiled against the `include/` of the
  development fork (branch `rabitq/adaptive-ef`, forked from `7c2d0d71`, 226 commits before v0.5.1). That patch is
  kept in `history/7c2d0d71-headers/` for provenance.
- **Why switching to v0.5.1 does not change any paper number.**
  - Only two driver modes call RaBitQ-Library code: `--method rotbq1` (its rotator) and `--method rabitq1`.
  - No stored result comes from either mode.
  - The modes behind the paper's numbers (Qdrant TurboQuant-1bit, FAISS PQ and OPQ, Sym-BQ) execute no
    RaBitQ-Library code; the headers are only compiled in.
- **What was checked.** Both drivers compile without errors against v0.5.1 plus the patch (full `-c` build, g++,
  upstream FAISS `46def469`).

`rabitq_paged_adaptive_benchmark.cpp` (paged RaBitQ HNSW) is the exception. It needs paging methods
(`prepare_1bit_paged_index`, ...) that are in neither tree. It was compiled against the working tree
`experiment/rabitq_v051_adaptive_odirect_build` (2026-11-06), which the repository does not reproduce. Only the
earlier ada-002 colocation runs, which the current Fig 19 replaces, used it.
