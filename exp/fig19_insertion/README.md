# Fig 19 / Sec 6.8: online insertion 1M -> 100M (MS MARCO Qwen-1024; 32 threads)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F19.`  (11 claims) |
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
| I.3 | insertion speedup vs RaBitQ-7bit, 100M (low end of 1.8-2.2x) (= `R19.2/F19.2`) | 1.8 |
| I.4 | insertion speedup vs RaBitQ-7bit, 10M (high end of 1.8-2.2x) (= `R19.1/R19.3`) | 2.2 |
| D19.1 | RaBitQ-1bit / BBQ insertion time, 10M (= `F19.3/F19.1`) | 2.06 |
| D19.2 | RaBitQ-1bit / BBQ insertion time, 100M (= `F19.4/F19.2`) | 1.49 |

## Notes

- Inserts were measured at 10M, 30M, 50M and 100M (the paper text says 25M increments).
- TurboQuant inserts faster than BBQ at 10M, 30M and 100M (388 / 1,342 / 5,046 s vs 522 / 1,637 / 5,762 s); RaBitQ-1bit has higher recall than BBQ at 10M and 50M.
- The intro's "1.8-2.2x vs RaBitQ-7bit" is 1.82-2.13x in the data (claims I.3, I.4); Sec. 6.8's 2.06x / 1.49x is against RaBitQ-1bit.
- Online insertion: `bbq/src/bbq_dynamic_insert.cpp` (`./bbq/build.sh dynamic`). The 1M base uses a frozen tau; inserted vectors get sign codes only, no retraining.
- Input prep (`prepare_msmarco_qwen1024_101m_dynamic_inputs.py`) materialises a 414 GB aligned FP32 file.
