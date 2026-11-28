# Fig 8 + Table 4: FP32-built vs 1-bit-built HNSW graph (MS MARCO, Quora; Qwen-1024)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only T4.`  (24 claims here) |
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
| F8.1 | FP32-built Recall@10, MS MARCO ef=128 | 0.8015 |
| F8.2 | 1-bit-built Recall@10, MS MARCO ef=128 | 0.801 |
| F8.3 | FP32-built QPS, MS MARCO ef=128 | 2982 |
| F8.4 | 1-bit-built QPS, MS MARCO ef=128 | 2715 |
| F8.5 | FP32-built Recall@10, Quora ef=128 | 0.8277 |
| F8.6 | 1-bit-built Recall@10, Quora ef=128 | 0.8275 |
| F8.7 | FP32-built QPS, Quora ef=128 | 3744 |
| F8.8 | 1-bit-built QPS, Quora ef=128 | 3415 |
| T4.1 | msmarco distance evals ef=256, FP32 | 5221 |
| T4.2 | msmarco distance evals ef=256, 1-bit | 6136 |
| T4.3 | msmarco distance evals ef=1024, FP32 | 17433 |
| T4.4 | msmarco distance evals ef=1024, 1-bit | 20379 |
| T4.5 | quora distance evals ef=256, FP32 | 3716 |
| T4.6 | quora distance evals ef=256, 1-bit | 4491 |
| T4.7 | quora distance evals ef=1024, FP32 | 12153 |
| T4.8 | quora distance evals ef=1024, 1-bit | 14634 |
| T4.9 | MS MARCO build time, FP32 graph (s) | 141.9 |
| T4.10 | MS MARCO build time, 1-bit graph (s) | 32.7 |
| T4.11 | Quora build time, FP32 graph (s) | 55.7 |
| T4.12 | Quora build time, 1-bit graph (s) | 10.5 |
| D8.1 | 1-bit build speedup, MS MARCO (= `T4.9/T4.10`) | 4.3 |
| D8.2 | 1-bit build speedup, Quora (= `T4.11/T4.12`) | 5.3 |
| D8.3 | extra distance evals of 1-bit graph, MS MARCO (%) (= `(T4.2/T4.1-1)*100`) | 17 |
| D8.4 | extra distance evals of 1-bit graph, Quora (%) (= `(T4.6/T4.5-1)*100`) | 20 |

## Notes

- `plot/make_index_quality_dat.py` builds `index_quality.dat` from `results/graph_construction_precision/*_paired.csv` (`run_msmarco_quora_qwen1024_graph_construction_precision.py`). The printed figure's coordinates, recovered from its PDF, are kept as `plot/index_quality.recovered.dat`. Their Recall@10 values equal the 20 measured efSearch points of all four curves to five decimals. Their QPS differs by up to about 20 % in either direction: single-thread QPS of a separate measurement, not a different experiment.
