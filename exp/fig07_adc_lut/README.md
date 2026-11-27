# Fig 7: (a) ADC vs sym-BQ candidate preservation (MS MARCO BGE); (b) LUT vs scalar ADC throughput (Quora BGE-M3, no calibration, T=1)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F7.`  (5 claims here) |
| Re-plot | `./exp/fig07_adc_lut/plot/regen.sh` |
| Paper figure | `plot/lut_ablation.paper.pdf`, `plot/preserve.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_nq_quora_qwen1024_adcbq_lut_no_calibration_t1.py`
- `exp/scripts/run_nq_quora_qwen1024_adcbq_lut_ablation_t1.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/manifest.json`
- `results/paired_ef_comparison_t1.csv`
- `results/quora_bge_m3/manifest.json`
- `results/quora_bge_m3/paired_ef_comparison_t1.csv`
- `results/quora_bge_m3/summary_t1.csv`
- `results/summary_t1.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F7.3 | scalar ADC QPS, Quora ef=256 | 71.1 |
| F7.4 | LUT ADC QPS, Quora ef=256 | 1906.3 |
| F7.5 | LUT speedup, Quora ef=256 | 26.8 |
| F7.6 | LUT speedup, low end (ef 80-1024) | 20.5 |
| F7.7 | LUT speedup, high end | 29.4 |

## Notes

- (a) `preserve.dat` was recovered from the original figure; its numbers (CandidateRecall@30 0.847 -> 0.947 on MS MARCO BGE-M3, 0.977 -> 0.994 on Quora) are checked against the Sec. 3 harness outputs (claims S3.1-S3.2) in `../sec2_3_analysis`.
- (b) is rebuilt from `results/quora_bge_m3/paired_ef_comparison_t1.csv`: same graph and efSearch, scalar FP32 ADC (`bbq/build.sh` with `ADCBQ_FORCE_FP32_DIRECT_SCALAR=1`) vs the int16 nibble LUT, no calibration, no rerank. The NQ / Quora Qwen-1024 runs of the same ablation are kept in `results/`.
