#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 update_recall-qps_full16_t24.py $R/recall_qps_all_latest_t24.csv > /dev/null
python3 ../../scripts/summarize_full16.py $R/recall_qps_all_latest_t24.csv $R/full16_summary_t24.csv --t1 ../../fig09_20_overall_t1/results/recall_qps_all_latest_t1.csv
gnuplot recall-qps_main_t24.gp
