# Fig 5 + Sec 4 end-to-end: query preprocessing vs exact-FP I/O wait (Qwen-1024)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F5.`  (8 claims here) |
| Re-plot | `./exp/fig05_preprocessing_io/plot/regen.sh` |
| Paper figure | `plot/preprocessing_io.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_query_preprocess_overhead_qwen1024.py`
- `exp/scripts/run_nq_qwen1024_query_preprocess_t1.py`
- `exp/scripts/run_turboquant_query_preprocess_qwen1024_t1.py`
- `exp/scripts/run_qwen1024_fixed_rerank_io_timing_t1.py`
- `exp/scripts/run_qwen1024_rabitq_adaptive_io_timing_t1.py`
- `exp/scripts/run_qwen1024_diskann_pq128_opq128_adaptive_t1.py`
- `exp/scripts/run_qwen1024_all_quantizers_resident_rerank_pareto.py`
- `exp/scripts/plot_qwen1024_diskann_adaptive_compare_t1.py`
- `exp/scripts/plot_qwen1024_native_resident_uring_batch_pareto_t1.py`
- `exp/scripts/query_preprocess_overhead_qwen1024.cpp`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/qwen1024_five_method_recall099_timing_t1.csv`
- `results/qwen1024_fixed_rerank_io_timing_t1.summary.csv`
- `results/qwen1024_native_resident_target_summary_t1.csv`
- `results/qwen1024_rabitq_adaptive_io_timing_t1.summary.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F5.1 | ADC-BQ query LUT build, MS MARCO (us) | 4.7 |
| F5.2 | TurboQuant query preprocessing, MS MARCO (us) | 9.6 |
| F5.3 | ADC-BQ exact-FP I/O wait, MS MARCO (us) | 517.6 |
| F5.4 | TurboQuant exact-FP I/O wait, MS MARCO (us) | 280 |
| S4.1 | ADC-BQ QPS @0.99, MS MARCO | 349 |
| S4.2 | TurboQuant QPS @0.99, MS MARCO | 362 |
| S4.3 | ADC-BQ QPS @0.99, NQ | 984 |
| S4.4 | TurboQuant QPS @0.99, NQ | 1063 |

## Notes

- The ADC-BQ bar uses the *fixed* R=120 (MS MARCO) / R=80 (NQ) operating point at Recall@10 >= 0.99, as in Sec. 4.
- `plot_qwen1024_diskann_adaptive_compare_t1.py` merges the four timing campaigns into the five-method CSV used by the figure.
