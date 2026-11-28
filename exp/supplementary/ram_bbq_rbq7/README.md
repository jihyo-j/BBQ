# Supplementary (not in the paper): all in DRAM, BBQ vs RaBitQ-7bit (NQ Qwen-1024, T=1)

| | |
|---|---|
| Verify | — |
| Re-plot | `./exp/supplementary/ram_bbq_rbq7/plot/regen.sh` |

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
