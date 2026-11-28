#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Carlito", python3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# colocation.dat <- ../results/te3small_ef256_t24/summary.csv
#   (scripts/run_msmarco_te3small1536_adcbq_colocation.py: MS MARCO text-embedding-3-small, ef=256, 24 threads)
python3 make_colocation_dat.py
gnuplot colocation.gp
