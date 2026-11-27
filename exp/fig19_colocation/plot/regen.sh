#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
# colocation.dat was recovered from the original matplotlib PDF (extract_colocation.py);
# the numbers behind it are in $R/msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv.
gnuplot colocation.gp
