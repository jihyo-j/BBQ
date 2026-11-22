#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_target_sweep_dat.py $R/legacy_gt/target_sweep.csv   # the paper figure uses the legacy-GT sweep (see README)
gnuplot calibration_analysis.gp
