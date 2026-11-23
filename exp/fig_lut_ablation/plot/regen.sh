#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_lut_ablation_dat.py $R/summary_t1.csv > lut_ablation.dat
gnuplot lut_ablation.gp
