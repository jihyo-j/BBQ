#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_query_bits_dat.py $R/summary_t1.csv $R/query_improvement_transitions_t1.csv > query_bits.dat
gnuplot query_bits.gp
