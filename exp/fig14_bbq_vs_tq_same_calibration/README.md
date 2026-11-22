# Fig 14: BBQ vs TurboQuant-1bit under the same calibration (NQ, MS MARCO; Qwen-1024; T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F14.`  (5 claims) |
| Re-plot | `./exp/fig14_bbq_vs_tq_same_calibration/plot/regen.sh` |
| Paper figure | `plot/recall_qps_bbq_tq.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_turboquant_calibrated_target099_main4_t1.py`
- `exp/scripts/run_turboquant_calibrated_target099_nq_optimized_t1.py`
- `exp/scripts/run_msmarco_qwen1024_calibrated_adcbq_turboquant_optimized_t1.py`
- `exp/scripts/combine_nq_msmarco_calibrated_adcbq_turboquant.py`
- `exp/scripts/plot_nq_qwen1024_calibrated_adcbq_vs_tq_t1.py`
- `exp/scripts/plot_nq_fixed_k_and_msmarco_calibrated_tq_adcbq.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/summary_t1.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F14.1 | TurboQuant+calibration QPS, MS MARCO @0.9853 | 2060 |
| F14.2 | BBQ QPS, MS MARCO @0.9853 | 1809 |
| F14.3 | BBQ QPS, NQ @0.9976 | 651 |
| F14.4 | TurboQuant FP reads/query, NQ | 42.1 |
| F14.5 | BBQ FP reads/query, NQ | 61.3 |

## Notes

- `combine_nq_msmarco_calibrated_adcbq_turboquant.py` merges the four per-method runs into `results/summary_t1.csv`.
