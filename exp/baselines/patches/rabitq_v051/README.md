# RaBitQ-Library v0.5.1 — benchmark binding used for the RaBitQ baselines

Used for: HNSW-RaBitQ-1bit with adaptive (lower-bound) FP32 rerank from SSD, and RaBitQ-7bit
(Figs 5, 10, 11, 15, 17, 18, 19).

```bash
git clone https://github.com/VectorDB-NTU/RaBitQ-Library.git && cd RaBitQ-Library
git checkout v0.5.1                                    # 44f8a607741e9e54f6dda3b38c325071e808aa45
git apply /path/to/BBQ/exp/baselines/patches/rabitq_v051/rabitq_v051_adaptive_odirect.cumulative.patch
pip install .                                          # Python binding _rabitqlib with the entries below
```

`rabitq_v051_adaptive_odirect.cumulative.patch` is the final source state (2026-11-17) of the binding used by
the paper runs, as a single diff against v0.5.1 (`include/rabitqlib/index/hnsw/hnsw.hpp`,
`python_bindings/hnsw_bindings.cpp`). It adds:

- `search_1bit_adaptive_fp_odirect`: the 1-bit estimator / lower-bound policy of RaBitQ, with only the accepted
  candidates read from the `.fbin` with synchronous aligned `O_DIRECT` `pread`;
- legacy (8-byte header) and 4 KiB-aligned `.fbin` layouts, inferred from the file size (`fbin_data_offset`);
- per-query latency (`query_latency_ns`) and FP read counters;
- online add for the insertion experiment (Fig. 19).

`notes/` keeps the original per-step change notes. The `*.ctx.patch` files there are context-only and do not
apply with `patch`; they document how the cumulative diff was built up. `03_io_wait_timing` (I/O-wait split for
Fig. 5) was a separate build (2026-11-11) and is **not** part of the cumulative patch.
