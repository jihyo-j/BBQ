#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_comparision_hnsw_dat.py $R/clean_rerun_20261110_cpu20/recall_qps_t1_from_recall096_rerun.csv   # figure = 11-10 rerun; text numbers = paper_run_20261109
gnuplot comparision_hnsw.gp
