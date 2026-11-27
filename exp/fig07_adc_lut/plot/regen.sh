#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
# (a) preserve.dat was recovered from the original figure (MS MARCO BGE CandidateRecall@R, Sec. 3 harness); (b) is rebuilt.
python3 make_preserve_lut_dat.py $R/quora_bge_m3/paired_ef_comparison_t1.csv > preserve_lut.dat
gnuplot preserve.gp
