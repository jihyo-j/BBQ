# Fig 18 / Sec 6.5: incremental insertion 1M + 10M..100M (MS MARCO Qwen-1024; 32 threads)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F18.`  (12 claims here) |
| Re-plot | `./exp/fig18_insertion/plot/regen.sh` |
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
| F18.1 | BBQ insertion time, 10M (s) | 521.7 |
| F18.2 | BBQ insertion time, 100M (s) | 5762.2 |
| F18.3 | RaBitQ-1bit insertion time, 10M (s) | 1073.5 |
| F18.4 | RaBitQ-1bit insertion time, 100M (s) | 8568.5 |
| F18.5 | insertion speedup vs RaBitQ-1bit, 10M (= `F18.3/F18.1`) | 2.06 |
| F18.6 | insertion speedup vs RaBitQ-1bit, 100M (= `F18.4/F18.2`) | 1.49 |
| F18.7 | BBQ Recall@10 after 100M | 0.9853 |
| F18.8 | RaBitQ-1bit Recall@10 after 100M | 0.9842 |
| F18.9 | TurboQuant Recall@10 after 100M | 0.9766 |
| F18.10 | TurboQuant Recall@10 at 10M | 0.9883 |
| F18.11 | BBQ min Recall@10 over checkpoints | 0.9849 |
| F18.12 | RaBitQ-1bit Recall@10 at 50M (as plotted: 0.9852) | 0.9852 |

## Notes

- Inserts measured at 10M, 30M, 50M and 100M, each from the same 1M index. TurboQuant inserts faster than BBQ at 10M, 30M and 100M.
- **Not reproduced.** Panel (a) plots RaBitQ-1bit at 50M as 0.9852; the data (`rabitq_b1_adaptive_c16_random_independent_t32.csv`) gives 0.9878 and no file contains 0.9852 (claim F18.12). The shipped `insertion_scale.dat` is the printed one; `regen.sh` gives the data.
