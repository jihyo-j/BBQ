# Fig 17 (MRL dimension, NQ Qwen3-8B) + Fig 18 (scale 10M/50M/100M, MS MARCO Qwen-1024)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F17.`  (18 claims) |
| Re-plot | `./exp/fig17_18_dimension_scale/plot/regen.sh` |
| Paper figure | `plot/mrl_dimension.paper.pdf`, `plot/scalability_100m.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_four_methods_series_t1.py`
- `exp/scripts/run_nq_qwen_multidim_calibrated_adcbq_tq_t1.py`
- `exp/scripts/prepare_nq_qwen8_mrl_inputs.py`
- `exp/scripts/prepare_msmarco_qwen1024_100m_scalability_inputs.py`
- `exp/scripts/run_qdrant_native_nq_qwen8_mrl_fixed_r_t1.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/alt_qdrant_native_tq1/fixed_r_summary_t1.csv`
- `results/alt_qdrant_native_tq1/selected_r_by_dimension.csv`
- `results/msmarco100m/msmarco_qwen1024_v2_100m/summary_t1.csv`
- `results/msmarco100m/msmarco_qwen1024_v2_10m/summary_t1.csv`
- `results/msmarco100m/msmarco_qwen1024_v2_50m/summary_t1.csv`
- `results/nq_mrl/nq_qwen8_mrl1024/summary_t1.csv`
- `results/nq_mrl/nq_qwen8_mrl256/summary_t1.csv`
- `results/nq_mrl/nq_qwen8_mrl2560/summary_t1.csv`
- `results/nq_mrl/nq_qwen8_mrl4096/summary_t1.csv`
- `results/nq_mrl/nq_qwen8_mrl512/summary_t1.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F17.1 | BBQ QPS @>=0.995, 512D | 1178 |
| F17.2 | BBQ QPS @>=0.995, 4096D | 972 |
| F17.3 | BBQ FP reads/query @>=0.995, 512D | 97.5 |
| F17.8 | BBQ FP reads/query @>=0.995, 4096D | 23.4 |
| F17.4 | BBQ Recall@10 at ef=128, 512D | 0.9868 |
| F17.5 | BBQ Recall@10 at ef=128, 4096D | 0.9952 |
| F17.6 | BBQ max Recall@10, 4096D | 0.9988 |
| F17.7 | TurboQuant max Recall@10, 4096D | 0.9990 |
| F18.1 | BBQ QPS @>=0.99, 10M | 563 |
| F18.2 | BBQ QPS @>=0.99, 50M | 541 |
| F18.3 | BBQ QPS @>=0.99, 100M | 539 |
| F18.4 | BBQ FP reads/query @>=0.99, 10M | 51.7 |
| F18.5 | BBQ FP reads/query @>=0.99, 100M | 71.5 |
| F18.6 | BBQ max Recall@10, 100M | 0.9944 |
| F18.7 | RaBitQ-7bit max Recall@10, 100M | 0.9849 |
| D18.1 | BBQ QPS drop 10M->100M (%) (= `(1-F18.3/F18.1)*100`) | 4.3 |
| D18.2 | BBQ / TurboQuant QPS @>=0.99, 100M (= `F18.3/R18.1`) | 1.98 |
| D18.3 | BBQ / RaBitQ-1bit QPS @>=0.99, 100M (= `F18.3/R18.2`) | 9.59 |

## Notes

- Fig. 18: TurboQuant's best QPS at Recall@10 >= 0.99 is 237 / 457 / 273 at 10M / 50M / 100M (3 rounds each), so the 1.98x at 100M depends on that single non-monotone point.
- The TurboQuant arm is the DiskVecLab TQ1 (`method_key = turboquant_tq1`). An alternative Qdrant-native TQ1 fixed-R run is kept in `results/alt_qdrant_native_tq1/` but is not plotted.
- Fig. 18 footprint: RaBitQ-7bit `index_bytes` = 112,834,728,305 B = 105.1 GiB (100M `summary_t1.csv`). The 31.3 GiB BBQ footprint is not recorded in any result file.
- `mrl256` is measured but not plotted.
