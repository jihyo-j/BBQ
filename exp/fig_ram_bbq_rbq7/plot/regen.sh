#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_ram_bbq_rbq7_dat.py $R/summary_t1.csv $R/rabitq_nq_qwen06_1024_search_results.csv > ram_bbq_rbq7.dat
gnuplot ram_bbq_rbq7.gp
