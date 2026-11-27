#!/usr/bin/env bash
# Regenerate this figure from ../results (gnuplot >= 5.4, python3 + numpy).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
R=../results
# update_recall-qps_full16.py reads every panel range from ONE panels file, so the Fig 20 (a-h) and Fig 9 (i-p)
# panel files are joined for the run and split again afterwards. Markers/curves follow the builder docstring
# (Pareto frontier, monotone trend); the measured values themselves are in $R/recall_qps_all_latest_t1.csv.
tmp=$(mktemp -d); trap "rm -rf $tmp" EXIT
python3 - "$tmp" <<PY
import re, sys
m = open("recall-qps_main_panels.gp").read(); d = open("recall-qps_md_panels.gp").read()
open(sys.argv[1] + "/md_head.gp", "w").write(re.split(r"(?m)^(?=# \\([a-p]\\)\\n)", d)[0])
parts = [p for p in re.split(r"(?m)^(?=# \\([a-p]\\)\\n)", d) if re.match(r"# \\([i-p]\\)\\n", p)]
open("recall-qps_main_panels.gp", "w").write(m.rstrip("\\n") + "\\n\\n" + "".join(parts))
PY
# The paper figure was drawn from recall_qps_as_printed_t1.csv; recall_qps_all_latest_t1.csv is the later table in
# which 4 curves were re-measured with 20 points (see README). Pass that file instead to draw the rerun.
python3 update_recall-qps_full16.py ${CSV:-$R/recall_qps_as_printed_t1.csv} > /dev/null
python3 - "$tmp" <<PY
import re, sys
c = open("recall-qps_main_panels.gp").read()
parts = re.split(r"(?m)^(?=# \\([a-p]\\)\\n)", c)
open("recall-qps_main_panels.gp", "w").write("".join(p for p in parts if not re.match(r"# \\([i-p]\\)\\n", p)).rstrip("\\n") + "\\n")
open("recall-qps_md_panels.gp", "w").write(open(sys.argv[1] + "/md_head.gp").read() + "".join(p for p in parts if re.match(r"# \\([i-p]\\)\\n", p)))
PY
cp recall-qps_main.dat recall-qps_md.dat
python3 ../../scripts/summarize_full16.py $R/recall_qps_as_printed_t1.csv $R/full16_summary_t1.csv
python3 ../../scripts/summarize_full16.py $R/recall_qps_all_latest_t1.csv $R/full16_summary_rerun_t1.csv
gnuplot recall-qps_main.gp
gnuplot recall-qps_md.gp
