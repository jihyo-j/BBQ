#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
cp $R/msmarco_qwen1024_recall_ablation_t1.csv .
python3 make_recall_ablation_dat.py $R/equal_pool_qd_ablation/equal_pool_qd_fixed75_vs_calibrated.csv
python3 make_recall_groups.py   # panel (b) <- ../../fig12_calibration/results/legacy_gt/per_query_target099.csv
gnuplot recall_ablation.gp
# perquery: scatter clouds are raster images reused from the original figure
gnuplot perquery.gp
