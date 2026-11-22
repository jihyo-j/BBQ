# RaBitQ v0.5.1 adaptive O_DIRECT benchmark binding

This directory contains the Python binding built from official RaBitQ-Library
v0.5.1, commit `44f8a607741e9e54f6dda3b38c325071e808aa45`, plus a benchmark-only
HNSW search entry point named `search_1bit_adaptive_fp_odirect`.

The added entry point preserves the 1-bit estimator/lower-bound policy used by
the in-memory adaptive-FP prototype. Only candidates accepted by that policy
are fetched from the source `.fbin` file with aligned synchronous `O_DIRECT`
`pread`. A fetched vector is normalized on demand before exact cosine scoring.
The returned statistics include per-query FP-read counts and total direct-I/O
bytes and requests.

A smoke test against the in-memory adaptive-FP path on DBpedia BGE-768 produced
identical IDs and FP-read counts (maximum distance difference below `9e-8`).
ADC-BQ does not use this reader; its fixed rerank@80 path uses the existing
`O_DIRECT` + `io_uring` implementation in
`faiss_fpgraph_packed_asym_bq_hnsw`.
