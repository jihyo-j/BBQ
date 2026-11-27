# Matched-recall cause analysis (MS MARCO Qwen-1024)

This ablation explains the throughput gap at the operating point used in the
in-memory HNSW comparison.  It uses the same held-out 5,980-query split,
FP-built M=24/efConstruction=200 topology, CPU 20, and one search thread.

| Method/mode | efSearch | Recall@10 | Median QPS | Time/query | Distance evaluations/query | FP32 reads/query |
|---|---:|---:|---:|---:|---:|---:|
| BBQ, calibrated SSD refinement | 512 | 0.992575 | 646.953 | 1.546 ms | 9,532.71 | 50.4172 |
| BBQ, traversal only | 512 | 0.803997 before refinement | 765.121 | 1.308 ms | 9,532.71 | 0 |
| In-memory FP32 HNSW | 384 | 0.992642 | 335.796 | 2.978 ms | 7,256.38 | 0 |

The difference between the full and traversal-only BBQ runs is 0.238 ms/query,
or 15.4% of full BBQ latency.  This is the exposed critical-path cost of SSD
submission/completion, exact scoring, and pipeline bookkeeping; it is not the
sum of device service times because refinement is overlapped with the next
query's traversal.  All measured overlapped reads were ready by the end of the
subsequent traversal (`pipeline_ready_ratio=1.0`), and explicit wait time was
negligible.  Exact FP32 scoring itself took 0.0162 ms/query.

BBQ evaluates 1.314x as many neighbors as FP32 HNSW at matched recall.  Even
so, the nominal scored-vector payload is 1.22 MB/query for BBQ
(9,532.71 x 128 B) versus 29.72 MB/query for FP32 HNSW
(7,256.38 x 4,096 B), or 24.36x less.  These payload products exclude common
graph-edge traffic and cache effects, so they should be described as nominal
payload rather than measured DRAM traffic.

The full BBQ rerun used five rounds.  Search counters and the FP32 HNSW point
used one round because the counters are deterministic for this fixed graph and
query set.  The measured 1.93x throughput advantage agrees with the paper's
clean uninstrumented comparison (about 1.90x).

Source files:

- `msmarco_full_ef512.csv`
- `msmarco_traversal_only_ef512.csv`
- `msmarco_traversal_stats_ef512.search_stats.csv`
- `msmarco_fp_hnsw_ef384_stats.csv`

For the DBpedia point at Recall@10 about 0.9875, the existing run refines
51.39 FP32 candidates/query.  Thus the two datasets refine 50.42--51.39
candidates/query at the operating points quoted in the paper.
