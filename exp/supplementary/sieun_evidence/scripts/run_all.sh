#!/bin/sh
# Regenerate every transcript in out/.  Scripts 01-04 are pure stdlib;
# 05 needs numpy.
set -e
cd "$(dirname "$0")"
mkdir -p ../out
for s in 01_verify_sec3 02_verify_sec6 03_eta_spearman 04_rotation_ablation; do
    echo "--- $s"
    python3 $s.py > ../out/$s.txt 2>&1
done
echo "--- 05_variance_decomposition (needs numpy)"
python3 05_variance_decomposition.py > ../out/05_variance_decomposition.txt 2>&1 \
    || echo "    skipped: numpy unavailable"
echo "done -> ../out/"
