#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_target_sweep_dat.py $R/legacy_gt/target_sweep.csv   # the paper figure uses the legacy-GT sweep (see README)
gnuplot calibration_analysis.gp
# fig:cali (Sec 6.5): (a) target sweep, (b) L_visit/L_select at target 0.99 -- legacy GT (matches Fig 11)
python3 make_calibration_dat.py $R/legacy_gt > calibration.dat
gnuplot calibration.gp
python3 make_loss_split.py $R/legacy_gt > $R/legacy_gt/loss_split_target099.csv
