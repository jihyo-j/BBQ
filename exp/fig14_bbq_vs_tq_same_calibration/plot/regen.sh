#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_recall_qps_bbq_tq_dat.py $R/summary_t1.csv > recall_qps_bbq_tq.dat
gnuplot recall_qps_bbq_tq.gp
