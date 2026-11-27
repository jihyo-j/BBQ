#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
# make_plot.py re-derives candidate_generation.csv from the raw HNSW / IVF runs (needs make workspace + the runs);
# the stored CSVs in ../results are its output.
echo "stored results: $R/first_measured_point_at_recall099.csv"
