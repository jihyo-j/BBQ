# Fig 16 (MRL dimension, NQ Qwen3-8B) + Fig 17 (scale 10M/50M/100M, MS MARCO Qwen-1024), Sec 6.5

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F17.`  (20 claims here) |
| Re-plot | `./exp/fig16_17_dimension_scale/plot/regen.sh` |

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
| F16.1 | BBQ QPS @>=0.995, 512D | 1178 |
| F16.2 | BBQ QPS @>=0.995, 4096D | 972 |
| F16.3 | BBQ FP32 reads, 512D | 97.5 |
| F16.4 | BBQ FP32 reads, 4096D | 23.4 |
| F16.5 | BBQ Recall@10 at ef=128, 512D | 0.9868 |
| F16.6 | BBQ Recall@10 at ef=128, 4096D | 0.9952 |
| F16.7 | BBQ max Recall@10, 4096D | 0.9988 |
| F16.8 | TurboQuant max Recall@10, 4096D | 0.999 |
| F17.1 | BBQ QPS @>=0.99, 10M | 563 |
| F17.2 | BBQ QPS @>=0.99, 50M | 541 |
| F17.3 | BBQ QPS @>=0.99, 100M | 539 |
| F17.4 | QPS decline 10M -> 100M (%) (= `(1-F17.3/F17.1)*100`) | 4.3 |
| F17.5 | BBQ / TurboQuant @0.99, 100M (= `F17.3/R17.1`) | 1.98 |
| F17.6 | BBQ / RaBitQ-1bit @0.99, 100M (= `F17.3/R17.2`) | 9.59 |
| F17.7 | BBQ max Recall@10, 100M | 0.9944 |
| F17.8 | TurboQuant max Recall@10, 100M | 0.9931 |
| F17.9 | RaBitQ-1bit max Recall@10, 100M | 0.9923 |
| F17.10 | RaBitQ-7bit max Recall@10, 100M | 0.9849 |
| F17.11 | BBQ FP32 reads @0.99, 10M | 51.7 |
| F17.12 | BBQ FP32 reads @0.99, 100M | 71.5 |

## Notes

- Fig. 16 uses the DiskVecLab TurboQuant arm (`turboquant_tq1`); a Qdrant-native TQ1 fixed-R run is kept in `results/alt_qdrant_native_tq1/`.
- Fig. 17: TurboQuant's best QPS at Recall@10 >= 0.99 is 237 / 457 / 273 at 10M / 50M / 100M, so the 1.98x at 100M rests on one non-monotone point. The 105.1 GiB RaBitQ-7bit footprint is `index_bytes` in the 100M summary; BBQ's 31.3 GiB is not recorded.
