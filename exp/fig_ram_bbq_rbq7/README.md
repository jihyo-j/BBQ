# All in DRAM: BBQ (calibrated, FP32 rerank from RAM) vs RaBitQ-7bit (NQ Qwen-1024, T=1)

| | |
|---|---|
| Verify | — |
| Re-plot | `./exp/fig_ram_bbq_rbq7/plot/regen.sh` |
| Paper figure | `plot/ram_bbq_rbq7.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_nq_qwen1024_bbq_vs_int16query_nolut_t1.py`
- `exp/scripts/run_nq_qwen1024_rabitq_b7_ram_testsplit_t1.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/manifest.json`
- `results/paired_ef_comparison_t1.csv`
- `results/rabitq_nq_qwen06_1024_search_results.csv`
- `results/summary_t1.csv`

## Notes

- Everything in DRAM, one search thread. BBQ = calibrated rerank with FP32 vectors in process RAM (2,452 held-out test queries, QPS = median of 3); RaBitQ-7bit c16 is all 3,452 NQ queries (QPS = mean of 3), as stated in `make_ram_bbq_rbq7_dat.py`. The query sets differ: compare curves, not individual points.
