# Fig 11 / Sec 6.3: ablation sym-BQ -> ADC-BQ -> fixed R=75 -> calibrated R(q), per-query reads (MS MARCO Qwen-1024)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F11.`  (19 claims here) |
| Re-plot | `./exp/fig11_ablation/plot/regen.sh` |
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
- `results/per_query_recall_groups_target099.legacy_gt.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| I.8 | calibration reads fewer than fixed R=75, low end (%) | 26 |
| I.9 | calibration reads fewer than fixed R=75, high end (%) | 40.5 |
| F11.1 | sym-BQ Recall@10, ef=256 | 0.7231 |
| F11.2 | ADC-BQ Recall@10, ef=256 | 0.803 |
| F11.3 | fixed R=75 Recall@10, ef=256 | 0.9893 |
| F11.4 | calibrated Recall@10, ef=256 | 0.9897 |
| F11.5 | calibrated FP32 reads, ef=256 | 54 |
| F11.6 | fixed R=75 QPS, ef=256 | 1109 |
| F11.7 | calibrated QPS, ef=256 | 1191 |
| F11.8 | calibrated reads fewer than R=75 (%) (= `(75-F11.5)/75*100`) | 28 |
| F11.9 | calibrated QPS gain (%) (= `(F11.7/F11.6-1)*100`) | 7.4 |
| F11.10 | QPS gain over the sweep, low end (%) | 2.7 |
| F11.11 | QPS gain over the sweep, high end (%) | 19.9 |
| F11.12 | share of queries with Recall@10=1.0 (%) | 91.79 |
| F11.13 | FP32 reads, Recall@10=1.0 group | 51.5 |
| F11.14 | FP32 reads, Recall@10=0.9 group | 75.4 |
| F11.15 | FP32 reads, Recall@10<=0.8 group | 119.8 |
| F11.16 | latency, Recall@10=1.0 group (ms) | 0.921 |
| F11.17 | latency, Recall@10<=0.8 group (ms) | 1.192 |

## Notes

- **Panel (b)** is built by `plot/make_recall_groups.py` from the default (legacy-GT) output of `run_msmarco_qwen1024_calibration_analysis.py`, `../fig12_calibration/results/legacy_gt/per_query_target099.csv`. It groups the test queries exactly as the runner does. Every printed number reproduces: 5,489 queries (91.79 %), 51.5 / 75.4 / 119.8 FP32 reads and 0.921 / 1.035 / 1.192 ms. The `ip_aligned` and `exact_cosine_aligned` groups, which differ slightly (91.81 %, 0.904 ms, ...), are kept for reference. Fig 11 therefore uses the legacy GT throughout, like Fig 12.
- `plot/perquery.gp` (per-query R* vs z10 clouds, DBpedia z10 vs relevant documents) is an earlier Sec 3 figure, not part of Fig 11; its clouds are raster images reused from that figure.
- Fixed R=75 vs calibrated (1,109 vs 1,191 QPS; 26.0-40.5 % fewer reads and 2.7-19.9 % more QPS over the sweep) is the equal-pool / equal-queue-depth run in `results/equal_pool_qd_ablation/`.
