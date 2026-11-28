# Fig 13 / Sec 6.3: BBQ vs TurboQuant-1bit under the same calibration (NQ, MS MARCO; Qwen-1024; T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F13.`  (7 claims here) |
| Re-plot | `./exp/fig13_bbq_vs_tq_same_calibration/plot/regen.sh` |

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
| F13.1 | TurboQuant QPS, MS MARCO @0.9853 | 2060 |
| F13.2 | BBQ QPS, MS MARCO @0.9853 | 1809 |
| F13.3 | BBQ Recall@10, NQ | 0.9976 |
| F13.4 | TurboQuant Recall@10, NQ | 0.9973 |
| F13.5 | TurboQuant / BBQ QPS, MS MARCO (= `F13.1/F13.2`) | 1.14 |
| F13.6 | BBQ / TurboQuant QPS, NQ (= `R13.1/R13.2`) | 2.01 |
| F13.7 | TurboQuant / BBQ distance evals, NQ (= `R13.4/R13.3`) | 1.77 |

## Notes

- NQ '2.01x' compares BBQ at ef=512 (651 QPS, Recall@10 0.9976) with TurboQuant at ef=1024 (324 QPS, 0.9973); at the same ef=512 TurboQuant reaches 654 QPS at 0.9963. The reads and the 1.77x distance evaluations compare the same two points.
