# Fig 20 / Sec 6.9: colocation under constrained DRAM (MS MARCO OpenAI-1536; 24 threads)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F20.`  (4 claims) |
| Re-plot | `./exp/fig20_colocation/plot/regen.sh` |
| Paper figure | `plot/colocation.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/rerun_msmarco_adcbq_calibrated_4k_p075_ef256.py`
- `exp/scripts/run_msmarco_page4k_ef256_qps_rerun.py`
- `exp/scripts/run_msmarco_adcbq_calibrated_rabitq_memory_page_sweep.py`
- `exp/scripts/run_msmarco_adcbq_rabitq_memory_page_sweep.py`
- `exp/scripts/run_msmarco_adcbq_graph_io_calibration.py`
- `exp/scripts/plot_msmarco_adcbq_calibrated_4k8k16k_ef256.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv`
- `results/page16k_qps_rerun_summary.csv`
- `results/page4k_memory_sweep_ef256_allratios_rerun_adjusted.csv`
- `results/page8k_qps_rerun_summary.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F20.1 | 16KB baseline QPS at 50% DRAM | 112.2 |
| F20.2 | 16KB colocated QPS at 50% DRAM | 170.6 |
| F20.3 | 16KB baseline graph page hit ratio at 50% DRAM | 0.576 |
| F20.4 | 16KB colocated graph page hit ratio at 50% DRAM | 0.751 |

## Notes

- `colocation.dat` was recovered from the original matplotlib PDF; the measured QPS and graph page hit ratios are in `results/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv` (4K from `page4k_*`, 8K/16K from `page{8k,16k}_qps_rerun_summary.csv`).
- Colocation itself is `build_colocation_physical_rows()` in `bbq/src/bbq_hnsw.cpp` (LDG partitioning + page-aware reorder, `--colocation-iterations 20 --colocation-partition-size 64`), ported from the pgvector implementation of Shim et al. [28].
