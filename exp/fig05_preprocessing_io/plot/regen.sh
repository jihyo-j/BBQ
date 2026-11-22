#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_preprocessing_io_dat.py $R/qwen1024_five_method_recall099_timing_t1.csv
gnuplot preprocessing_io.gp
