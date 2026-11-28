# RaBitQ v0.5.1 adaptive O_DIRECT binding with aligned fbin support

This binding is based on official RaBitQ-Library v0.5.1 commit
`44f8a607741e9e54f6dda3b38c325071e808aa45` and the benchmark-only adaptive
FP O_DIRECT callback described in
`../rabitq_v051_adaptive_odirect_build/README.md`.

The O_DIRECT fbin reader accepts only these two exact file layouts:

- legacy fbin: 8-byte `(n, d)` header followed by the FP32 payload;
- aligned fbin: the same header padded to 4096 bytes followed by the FP32
  payload.

The layout is inferred from the exact file size. The search statistics expose
`fbin_data_offset`, which must equal `4096` in the aligned-header benchmark.
The source-level change is recorded in `aligned4k_header.patch`.

The tail-latency build also exposes `query_latency_ns`, measured with
`std::chrono::steady_clock` inside the per-query HNSW loop.  It covers 1-bit
traversal, synchronous O_DIRECT FP reads, and exact scoring while excluding
Python result conversion.  The incremental source change is recorded in
`per_query_tail_stats.patch`; the NQ runner is
`../run_nq_rabitq_tail_stats.py`.
