#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_recall_k_dat.py $R/nq_qwen1024_recall_k30_50_100_t1.csv > recall_k.dat
gnuplot recall_k.gp
