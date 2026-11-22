#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, font "Liberation Sans", python3 + numpy).
# The data builders below also rewrite the panel/legend gnuplot files from older templates; the final
# hand-styled versions are kept, so only the .dat files are rebuilt.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
keep=(recall-qps_main_panels.gp recall-qps_md_panels.gp recall-qps_main.legend recall-qps_md.legend
      recall-qps_main_legend.gp recall-qps_md_legend.gp)
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cp "${keep[@]}" "$tmp"/
python3 update_recall-qps_main_tq_pq_opq.py $R/recall_qps_12methods_16datasets_t1.csv > /dev/null
python3 add_recall-qps_main_pipeann.py $R/spann_pipeann_all_search_results.csv > /dev/null
python3 make_recall-qps_md.py $R/spann_pipeann_all_search_results.csv > /dev/null
for f in "${keep[@]}"; do cp "$tmp/$f" .; done
gnuplot recall-qps_main.gp
gnuplot recall-qps_md.gp
