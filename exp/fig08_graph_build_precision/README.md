# Fig 8 + Table 4: FP32-built vs 1-bit-built HNSW graph (MS MARCO, Quora; Qwen-1024)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F8.`  (10 claims) |
| Re-plot | `./exp/fig08_graph_build_precision/plot/regen.sh` |
| Paper figure | `plot/index_quality.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_msmarco_quora_qwen1024_graph_construction_precision.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/graph_construction_precision/fresh_build_timing/msmarco.binary_graph.build.csv`
- `results/graph_construction_precision/fresh_build_timing/msmarco.binary_graph.build.wall_seconds.txt`
- `results/graph_construction_precision/fresh_build_timing/msmarco.fp_graph.build.csv`
- `results/graph_construction_precision/fresh_build_timing/msmarco.fp_graph.build.wall_seconds.txt`
- `results/graph_construction_precision/fresh_build_timing/quora.binary_graph.build.csv`
- `results/graph_construction_precision/fresh_build_timing/quora.binary_graph.build.wall_seconds.txt`
- `results/graph_construction_precision/fresh_build_timing/quora.fp_graph.build.csv`
- `results/graph_construction_precision/fresh_build_timing/quora.fp_graph.build.wall_seconds.txt`
- `results/graph_construction_precision/msmarco.binary_graph.raw.csv`
- `results/graph_construction_precision/msmarco.binary_graph.stats.csv`
- `results/graph_construction_precision/msmarco.fp_graph.raw.csv`
- `results/graph_construction_precision/msmarco.fp_graph.stats.csv`
- `results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_build_times.csv`
- `results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision.csv`
- `results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv`
- `results/graph_construction_precision/quora.binary_graph.raw.csv`
- `results/graph_construction_precision/quora.binary_graph.stats.csv`
- `results/graph_construction_precision/quora.fp_graph.raw.csv`
- `results/graph_construction_precision/quora.fp_graph.stats.csv`
- `results/graph_construction_precision/summary.md`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F8.1 | FP32-built graph Recall@10, MS MARCO ef=128 | 0.8015 |
| F8.2 | 1-bit-built graph Recall@10, MS MARCO ef=128 | 0.8010 |
| F8.3 | FP32-built graph QPS, MS MARCO ef=128 | 2982 |
| F8.4 | 1-bit-built graph QPS, MS MARCO ef=128 | 2715 |
| F8.5 | FP32-built graph QPS, Quora ef=128 | 3744 |
| F8.6 | 1-bit-built graph QPS, Quora ef=128 | 3415 |
| T4.1 | MS MARCO distance evals ef=256, FP32 graph | 5221 |
| T4.2 | MS MARCO distance evals ef=256, 1-bit graph | 6136 |
| T4.3 | MS MARCO build time, FP32 graph (s) | 141.9 |
| T4.4 | MS MARCO build time, 1-bit graph (s) | 32.7 |

## Notes

- `index_quality.dat` was recovered from the vector coordinates of the original matplotlib PDF; the measured numbers are in `results/graph_construction_precision/*_paired.csv` and `*_build_times.csv`.
- The run used `experiment/faiss_fpgraph_packed_asym_bq_hnsw` built on 2026-09-29 from an earlier snapshot of `bbq/src/bbq_hnsw.cpp` (`--graph-build fp|binary`).
