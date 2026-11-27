#!/bin/sh
# Two passes: the first renders the legend strips with estimated label widths,
# fixwidths.py measures what was actually set, the second lays them out on the
# real widths.  Then the fit checks.
set -e
cd /home/sieun/projects/BBQETA/gp
python3 ../src/export_cq.py >/dev/null
python3 cut_fig2.py
for g in fig1 fig3 fig4p fig2leg; do gnuplot $g.gp; done
python3 fixwidths.py f1_legend.pdf f2_legend.pdf f3_legend.pdf f3d_legend.pdf f4_legend.pdf 2>/dev/null >/dev/null || true
for g in fig1 fig3 fig4p fig2leg; do gnuplot $g.gp; done
python3 checkfit.py f1a.pdf f1b.pdf f1_legend.pdf f2a.pdf f2b.pdf f2_legend.pdf \
  f3a.pdf f3b.pdf f3c.pdf f3d.pdf f3_legend.pdf f3d_legend.pdf \
  f4a.pdf f4b.pdf f4_legend.pdf
python3 legcheck.py f1_legend.pdf f2_legend.pdf f3_legend.pdf f3d_legend.pdf f4_legend.pdf | grep -E "tight|canvas" || true
