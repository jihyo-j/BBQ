#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
# (a) preserve.dat <- ../../sec2_3_analysis/harness/work/msmarco_bgem3_perq.npz (Sec 2-3 harness, MS MARCO BGE-M3)
#     re-measure it with ../remeasure_preserve.sh; (b) preserve_lut.dat <- $R/quora_bge_m3/paired_ef_comparison_t1.csv
python3 make_preserve_dat.py
python3 make_preserve_lut_dat.py $R/quora_bge_m3/paired_ef_comparison_t1.csv > preserve_lut.dat
gnuplot preserve.gp
