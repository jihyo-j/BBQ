# Fig 12 / Sec 6.3: calibration target sweep and L_visit / L_select loss split (MS MARCO Qwen-1024, ef=256)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F12.`  (11 claims here) |
| Re-plot | `./exp/fig12_calibration/plot/regen.sh` |
| Paper figure | `plot/calibration.paper.pdf`, `plot/calibration_analysis.paper.pdf` |

## Re-run the experiment

Scripts live in [`exp/scripts/`](../scripts) (one flat directory, because the runners import each other). Set the paths in [`exp/env.sh`](../env.sh) first; see the top-level README.

- `exp/scripts/run_msmarco_qwen1024_calibration_analysis.py`
- `exp/scripts/run_msmarco_same_pool_calibration_ssd.py`

## Stored results

Result files behind the figure (copied verbatim from the original run):

- `results/exact_cosine_aligned/calibration_size_sensitivity.csv`
- `results/exact_cosine_aligned/error_decomposition_target099.csv`
- `results/exact_cosine_aligned/fitted_thresholds.json`
- `results/exact_cosine_aligned/fixed_r_sweep.csv`
- `results/exact_cosine_aligned/manifest.json`
- `results/exact_cosine_aligned/per_query_recall_groups_target099.csv`
- `results/exact_cosine_aligned/per_query_target099.csv`
- `results/exact_cosine_aligned/target_sweep.csv`
- `results/ip_aligned/error_decomposition_target099.csv`
- `results/ip_aligned/per_query_recall_groups_target099.csv`
- `results/ip_aligned/target_sweep.csv`
- `results/legacy_gt/error_decomposition_target099.csv`
- `results/legacy_gt/fixed_r_sweep.csv`
- `results/legacy_gt/loss_split_target099.csv`
- `results/legacy_gt/manifest.json`
- `results/legacy_gt/per_query_target099.csv`
- `results/legacy_gt/target_sweep.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F12.1 | mean FP32 reads at target 0.90 | 28.3 |
| F12.2 | mean FP32 reads at target 0.995 | 58.6 |
| F12.3 | pool top-10 preserved at target 0.99 | 0.9928 |
| F12.4 | extra reads 0.99 -> 0.995 (= `F12.2-R12.1`) | 4.6 |
| F12.5 | Recall@10 gain 0.99 -> 0.995 (= `R12.3-R12.2`) | 0.0002 |
| F12.6 | L_visit at target 0.99 | 0.0096 |
| F12.7 | L_select at target 0.99 | 0.0007 |
| F12.8 | total 1 - Recall@10 | 0.0103 |
| F12.9 | queries with traversal loss | 454 |
| F12.10 | queries with selection loss | 40 |
| F12.11 | traversal share of the loss (%) | 0.93 |

## Notes

- Both panels use the legacy GT of the MS MARCO run, as does Fig. 11(a): (a) the target sweep (`results/legacy_gt/target_sweep.csv`); (b) L_visit = 0.0096 (454 queries), L_select = 0.0007 (40 queries), sum 1 - Recall@10 = 0.0103, traversal share 93 % (`results/legacy_gt/loss_split_target099.csv`, from `per_query_target099.csv` by `plot/make_loss_split.py`).
- `plot/calibration.gp` + `make_calibration_dat.py` draw the paper figure. Under legacy GT, L_visit includes 147 queries whose in-pool GT is not reproduced by exact rerank (legacy GT is inexact at rank 10 for ~2.6 % of MS MARCO Qwen queries). The exact-cosine and ip variants are kept under `results/exact_cosine_aligned/` and `results/ip_aligned/` (exact cosine: 307 traversal-limited queries).
