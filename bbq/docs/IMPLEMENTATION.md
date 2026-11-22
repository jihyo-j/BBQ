# Optimized ADC+BQ HNSW

This directory is the presentation entry point for the optimized asymmetric
BQ HNSW experiments.

The implementation still depends on the local FAISS build. The ADC+BQ logic is
kept in the experiment source files so we do not duplicate a large C++ file.

## Variants

### ADC+BQ binary graph

Use this when the HNSW graph is also built from packed BQ codes.

- Source: `../faiss_packed_asym_bq_hnsw.cpp`
- Canonical dual-mode source: `../faiss_fpgraph_packed_asym_bq_hnsw.cpp`
- Build mode: `--graph-build binary`
- Stored payload: packed sign bits only
- Search score: `distance = -(sign(x) . q)`
- Rerank: optional FP32 cosine rerank from the original fbin data

### ADC+BQ FP graph

Use this when the graph topology is built with normalized FP vectors, but the
stored search payload is still packed BQ.

- Source: `../faiss_fpgraph_packed_asym_bq_hnsw.cpp`
- Build mode: `--graph-build fp`
- Build graph: normalized FP vectors with `IndexHNSWFlat`
- Stored payload: packed sign bits in `IndexBinaryFlat`
- Saved index: `IndexBinaryHNSW`; FP vectors are not stored in the index
- Search score: `distance = -(sign(x) . q)`
- Rerank: optional FP32 cosine rerank from the original fbin data

## Core Logic Map

- PGVector-compatible bit packing: `pack_pgvector_sign_bits`
- Query LUT construction: `PackedAsymBQDistanceComputer::set_query_fp32_*`
- Exact ADC distance: `PackedAsymBQDistanceComputer::distance_to_code_*`
- Inline HNSW search loop: `search_one_packed_inline`
- FP graph + BQ storage build: `build_fp_graph_bq_storage_index`
- Binary graph build: `build_binary_graph_index`
- FP rerank: `rerank_cosine`

Bit order is PGVector-compatible:

```text
dim i -> byte[i / 8], bit position (7 - i % 8)
```

So the first dimension in a byte uses the MSB mask `0x80`.

## Optimizations Included

- Packed BQ storage, `dim / 8` bytes per vector
- Native FAISS packed-code traversal without aligned or block32 duplicates
- FP32 byte LUT and nibble LUT modes
- Nibble LUT mode: one byte is split into two 4-bit lookups
- Inline HNSW search path with batched distance evaluation
- Vector visited table path used by the optimized inline implementation
- Prefetch lookahead for candidate codes
- FP32 rerank dot path for rerank@N
- FP graph build mode while keeping BQ-only index storage

## Non-speculative SSD/CPU overlap

`--fp-rerank-overlap-queries` pipelines exact FP reranking across queries.  It
does **not** prefetch FP rows for tentative HNSW candidates.  A worker instead:

1. finishes ADC-BQ HNSW search for query A and obtains its final rerank set;
2. submits exactly those FP row reads to its persistent `io_uring`;
3. runs the complete HNSW search for query B while A's reads are in flight;
4. drains any remaining CQEs, scores A, and then submits B.

This preserves the baseline candidate set and SSD request count.  The final
query assigned to each worker is the only query that cannot hide its I/O behind
a following search.  Use it with the direct-I/O rerank path:

```text
--fp-rerank-source odirect-uring \
--fp-rerank-overlap-queries \
--fp-rerank-qd 80
```

The current implementation requires inline ADC-BQ search, RAM-resident HNSW/BQ,
and a fixed rerank count.  Used alone, it does not use an FP row/page cache; a
row cache is supported through the hybrid mode below.  CSV output includes the
number of overlapped queries, overlapped I/O requests, completions already
available after the next HNSW search, residual wait time, and scoring time.

## Same-query cache-hit/I/O overlap

`--fp-rerank-overlap-cache-hits` is the single-query alternative for an FP row
cache.  After ADC-BQ search has produced the final rerank set, a worker:

1. partitions the final candidates into FP row-cache hits and misses;
2. submits all miss reads as one O_DIRECT `io_uring` batch;
3. computes exact distances for cache hits while the miss reads are in flight;
4. drains the remaining completions, caches and scores the missed rows, and
   returns the exact top-k over the unchanged candidate set.

Use it with:

```text
--fp-rerank-source odirect-uring \
--fp-rerank-cache \
--fp-rerank-cache-rows 32768 \
--fp-rerank-overlap-cache-hits \
--fp-rerank-qd 80
```

This mode does not speculate and therefore preserves recall and candidate
count.  The CSV includes overlapped query/hit counts, completions ready after
hit scoring, residual wait time, and hit-scoring time.

## Hybrid cache-hit and inter-query overlap

Enable both overlap flags to combine the two schedules:

```text
--fp-rerank-source odirect-uring \
--fp-rerank-cache \
--fp-rerank-cache-rows 32768 \
--fp-rerank-overlap-cache-hits \
--fp-rerank-overlap-queries \
--fp-rerank-qd 80
```

For query A, the worker partitions the final rerank set, submits only FP cache
misses, and scores cache hits while those reads are active.  It then runs query
B's complete ADC-BQ HNSW traversal before draining A's remaining completions,
caching and scoring the fetched rows, and selecting A's exact top-k.  Thus the
same miss I/O can overlap first with A's cache-hit scoring and then with B's
graph traversal.  The final query assigned to each worker still requires a
pipeline drain.  This hybrid path keeps the graph and BQ codes in RAM and does
not issue speculative FP reads.  The row cache uses a fixed FP slab and an
atomic ID-to-slot publication table: hits are lock-free, a concurrent publish
is treated as a miss rather than waited on, and cached rows are never moved or
evicted during the run.

The reproducible DBpedia Qwen-1024 A/B runner is:

```bash
python3 experiment/benchmark_adcbq_intraquery_cache_overlap.py
```

The T=24 comparison of inter-query, cache-only intra-query, and hybrid overlap
is reproducible with:

```bash
python3 experiment/benchmark_adcbq_hybrid_overlap.py
```

The default presentation setting is exact FP32 ADC:

```text
--adc-mode fp32 --lut-mode nibble --search-impl inline --rerank-dot fp32
```

This keeps the asymmetric BQ score exact. Approximate `int8` LUT mode exists in
the source, but should not be used when the claim requires a float query and
exact ADC score.

## Build

```bash
./experiment/adcbq_optimized/build.sh
```

This produces:

```text
experiment/adcbq_optimized/faiss_adcbq_hnsw
```

Environment overrides:

```bash
FAISS_ROOT=/home/jihyo/workspace/faiss \
FAISS_BUILD=/home/jihyo/workspace/faiss/build_hnsw_conda/faiss \
CONDA_LIB=/home/jihyo/anaconda3/envs/hnsw/lib \
./experiment/adcbq_optimized/build.sh
```

## OpenAI1536 Examples

Binary graph:

```bash
./experiment/adcbq_optimized/run_openai1536_binary_graph.sh
```

FP graph with BQ storage:

```bash
./experiment/adcbq_optimized/run_openai1536_fp_graph.sh
```

Both scripts accept extra arguments and pass them through to the binary. For
example, to force rebuild:

```bash
./experiment/adcbq_optimized/run_openai1536_fp_graph.sh --rebuild
```
