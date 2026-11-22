# Fig 12 / Sec 6.4: ablation sym-BQ -> ADC-BQ -> fixed R=75 -> calibrated R (MS MARCO Qwen-1024)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F12.`  (11 claims) |
| Re-plot | `./exp/fig12_ablation/plot/regen.sh` |
| Paper figure | `plot/perquery.paper.pdf`, `plot/recall_ablation.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_msmarco_qwen1024_recall_ablation_t1.py`
- `exp/scripts/run_msmarco_qwen1024_equal_pool_qd_ablation_t1.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/equal_pool_qd_ablation/equal_pool_qd_fixed75_vs_calibrated.csv`
- `results/equal_pool_qd_ablation/manifest.json`
- `results/equal_pool_qd_ablation/raw/calibrated.pool100.qd100.ef100.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool1024.qd1024.ef1024.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool120.qd120.ef120.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool140.qd140.ef140.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool160.qd160.ef160.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool180.qd180.ef180.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool200.qd200.ef200.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool220.qd220.ef220.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool240.qd240.ef240.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool256.qd256.ef256.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool320.qd320.ef320.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool512.qd512.ef512.csv`
- `results/equal_pool_qd_ablation/raw/calibrated.pool80.qd80.ef80.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool100.qd100.ef100.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool1024.qd1024.ef1024.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool120.qd120.ef120.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool140.qd140.ef140.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool160.qd160.ef160.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool180.qd180.ef180.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool200.qd200.ef200.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool220.qd220.ef220.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool240.qd240.ef240.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool256.qd256.ef256.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool320.qd320.ef320.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool512.qd512.ef512.csv`
- `results/equal_pool_qd_ablation/raw/fixed_r75.pool80.qd80.ef80.csv`
- `results/msmarco_qwen1024_recall_ablation_t1.csv`
- `results/per_query_recall_groups_target099.exact_cosine_aligned.csv`
- `results/per_query_recall_groups_target099.ip_aligned.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F12.1 | sym-BQ Recall@10, ef=256 | 0.7231 |
| F12.2 | ADC-BQ Recall@10, ef=256 | 0.8030 |
| F12.3 | ADC-BQ + fixed R=75 Recall@10, ef=256 | 0.9893 |
| F12.4 | BBQ calibrated Recall@10, ef=256 | 0.9897 |
| F12.5 | BBQ calibrated FP reads/query, ef=256 | 54.0 |
| F12.6 | fixed R=75 QPS (equal pool/QD) | 1109 |
| F12.7 | calibrated QPS (equal pool/QD) | 1191 |
| F12.8 | share of queries with Recall@10=1.0 (%) | 91.79 |
| F12.9 | FP reads/query, Recall@10=1.0 group | 51.5 |
| F12.10 | FP reads/query, Recall@10=0.9 group | 75.4 |
| F12.11 | FP reads/query, Recall@10<=0.8 group | 119.8 |

## Notes

- **GT variant.** Panel (b) (91.8 % / 51.5 / 75.4 / 119.8) matches the `ip_aligned` per-query groups; the `exact_cosine_aligned` variant gives 94.2 % / 51.7 / 84.8 / 120.2. Both files are kept.
- **Paper typo.** The text says 91.79 % of queries; the data gives 5490/5980 = 91.806 %, i.e. 91.81 % (claim F12.8 fails on purpose).
- Fixed R=75 vs calibrated QPS (1,109 vs 1,191) is the equal-pool / equal-queue-depth run in `results/equal_pool_qd_ablation/`.
