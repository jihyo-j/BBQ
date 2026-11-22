# Fig 13 / Sec 6.5: calibration target sweep and L_visit / L_select (MS MARCO Qwen-1024, ef=256)

| | |
|---|---|
| Verify | `make build/verify && ./build/verify --root exp --only F13.`  (5 claims) |
| Re-plot | `./exp/fig13_calibration_analysis/plot/regen.sh` |
| Paper figure | `plot/calibration_analysis.paper.pdf` |

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
- `results/exact_cosine_aligned/target_sweep.csv`
- `results/ip_aligned/error_decomposition_target099.csv`
- `results/ip_aligned/per_query_recall_groups_target099.csv`
- `results/ip_aligned/target_sweep.csv`
- `results/legacy_gt/error_decomposition_target099.csv`
- `results/legacy_gt/fixed_r_sweep.csv`
- `results/legacy_gt/manifest.json`
- `results/legacy_gt/target_sweep.csv`

## Claims checked by `verify`

| id | claim | paper |
|---|---|---|
| F13.1 | mean FP reads at target 0.90 | 28.3 |
| F13.2 | mean FP reads at target 0.995 | 58.6 |
| F13.3 | pool top-10 preserved at target 0.90 | 0.9117 |
| F13.4 | pool top-10 preserved at target 0.995 | 0.9953 |
| F13.5 | queries whose pool misses part of global top-10 (of 5980) | 307 |

## Notes

- **GT variant.** The *plotted* final-recall curve (0.9897 at target 0.99) is the legacy-GT sweep in `results/legacy_gt/`; the 307-of-5,980 traversal-limited count quoted in Sec. 6.5 comes from `results/exact_cosine_aligned/` (legacy GT reports 454; `ip_aligned` reports 307 plus 146 metric/reference mismatches). FP reads and pool coverage agree across all three. Panel (a) (L_visit / L_select) is still a placeholder in the paper.
