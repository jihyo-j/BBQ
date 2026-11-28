#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, python3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# index_quality.dat <- ../results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv
#   (scripts/run_msmarco_quora_qwen1024_graph_construction_precision.py)
python3 make_index_quality_dat.py
gnuplot index_quality.gp
