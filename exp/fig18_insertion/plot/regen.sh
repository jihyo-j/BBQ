#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_insertion_scale_dat.py $R/frozen_adcbq_vs_rabitq_c16_comparison.csv 256 \
  $R/rabitq_b1_adaptive_c16_random_independent_t32.csv $R/tq1plus_r60_random_independent_t32.csv
gnuplot insertion_scale.gp
