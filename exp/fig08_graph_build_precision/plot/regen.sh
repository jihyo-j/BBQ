#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
# index_quality.dat was recovered from the original matplotlib PDF (extract_index_quality.py);
# the numbers behind it are in $R/graph_construction_precision/*_paired.csv.
gnuplot index_quality.gp
