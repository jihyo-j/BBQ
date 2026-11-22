#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_page_io_dat.py $R/page_io/normalized_page_io.csv $R/page_io/operating_points.csv > page_io.dat
gnuplot page_io.gp
