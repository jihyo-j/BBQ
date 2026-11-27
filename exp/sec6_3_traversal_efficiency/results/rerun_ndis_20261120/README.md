# DBpedia matched-recall HNSW work counters

This rerun measures the two operating points that both reach Recall@10 =
0.994333 on the held-out DBpedia Qwen-1024 test split (100 calibration queries
and 300 test queries). Both indexes use the same M=24, efConstruction=200 graph
topology and run single-threaded on CPU 20 for 20 rounds.

| Method | efSearch | Mean distance evaluations/query | Mean FP32 refinements/query | Median QPS |
|---|---:|---:|---:|---:|
| Calibrated ADC-BQ HNSW + SSD refinement | 1024 | 22,976.9 | 54.9567 | 269.297 |
| In-memory FP32 HNSW | 256 | 6,834.33 | 0 | 360.717 |

ADC-BQ performs 3.362x as many traversal distance evaluations at this matched
recall point. Its count is 124.943 upper-level evaluations plus 22,851.9
base-level evaluations on average. The refinement distribution is p95=125,
p99=171.16, and max=264 candidates/query.

The ADC-BQ run used query-level counters, and the FP-HNSW run reset and sampled
FAISS `hnsw_stats` around each search round. The QPS values are useful as a
sanity check, but the uninstrumented clean rerun remains the source for the
paper's primary throughput numbers.
