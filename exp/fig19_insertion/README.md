# Fig 19 / Sec 6.8: online insertion 1M -> 100M (MS MARCO Qwen-1024; 32 threads)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F19.`  (7 claims) |
| Re-plot | `./exp/fig19_insertion/plot/regen.sh` |
| Paper figure | `plot/insertion_scale.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_msmarco_qwen1024_101m_frozen_adc_and_rabitq.py`
- `exp/scripts/run_msmarco_qwen1024_101m_rabitq_b1_adaptive_c16.py`
- `exp/scripts/run_msmarco_qwen1024_101m_tq1_r60_dynamic.py`
- `exp/scripts/make_dynamic_insert_101m_adcbq_frozen_vs_rabitq_c16.py`
- `exp/scripts/prepare_msmarco_qwen1024_101m_dynamic_inputs.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/frozen_adcbq_vs_rabitq_c16_comparison.csv`
- `results/rabitq_b1_adaptive_c16_random_independent_t32.csv`
- `results/tq1plus_r60_random_independent_t32.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F19.1 | BBQ cumulative insertion time, 10M (s) | 521.7 |
| F19.2 | BBQ cumulative insertion time, 100M (s) | 5762.2 |
| F19.3 | RaBitQ-1bit cumulative insertion time, 10M (s) | 1073.5 |
| F19.4 | RaBitQ-1bit cumulative insertion time, 100M (s) | 8568.5 |
| F19.5 | BBQ Recall@10 after 100M inserts | 0.9853 |
| F19.6 | RaBitQ-1bit Recall@10 after 100M inserts | 0.9842 |
| F19.7 | TurboQuant Recall@10 after 100M inserts | 0.9766 |

## Notes

- Online insertion: `bbq/src/bbq_dynamic_insert.cpp` (`./bbq/build.sh dynamic`). The 1M base uses a frozen tau; inserted vectors get sign codes only, no retraining.
- Input prep (`prepare_msmarco_qwen1024_101m_dynamic_inputs.py`) materialises a 414 GB aligned FP32 file.
