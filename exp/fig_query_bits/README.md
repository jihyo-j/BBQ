# Sec 3.1: query-side bit-width sweep with 1-bit database codes (NQ / Quora Qwen-1024, ef=256, T=1)

| | |
|---|---|
| Verify | — |
| Re-plot | `./exp/fig_query_bits/plot/regen.sh` |
| Paper figure | `plot/query_bits.paper.pdf` |

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

## Notes

- Query side only: database codes stay 1-bit sign codes; the query is quantised to b bits (int path: `scripts/faiss_fpgraph_packed_asym_bq_hnsw_intquery.cpp`) or kept in FP32. `query_improvement_transitions_t1.csv` counts the queries whose recall changes between consecutive bit-widths.
