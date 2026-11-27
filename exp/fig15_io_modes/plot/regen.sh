#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
python3 make_io_ablation_dat.py $R/qwen1024_summary_t1.csv $R/openai1536_summary_t1.csv > io_ablation.dat
gnuplot io_ablation.gp
