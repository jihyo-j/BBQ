# Sec 5.1: query-side bit-width sweep with 1-bit database codes (NQ / Quora Qwen-1024, ef=256, T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F7.`  (2 claims here) |
| Re-plot | `./exp/sec5_1_query_bits/plot/regen.sh` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_nq_quora_query_bitwidth_sweep_t1.py`
- `exp/scripts/run_nq_quora_query_bitwidth_int_path_t1.py`
- `exp/scripts/faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/fp32_saturation_t1.csv`
- `results/manifest.json`
- `results/query_improvement_transitions_t1.csv`
- `results/summary_t1.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F7.1 | 8-bit query Recall@10, NQ ef=256 | 0.993 |
| F7.2 | 8-bit vs FP32 query gap (pp) (= `(R7.1-F7.1)*100`) | 0.2 |

## Notes

- Query bit-width sweep at efSearch 256: 8-bit queries reach Recall@10 0.9934 on NQ against 0.9956 with FP32 queries (claim F7.1-F7.2).
