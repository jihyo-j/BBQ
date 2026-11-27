# Panel (a) of the calibration figure: the mean-recall guarantee, checked.
#
# One fixed search setting (MS MARCO Qwen-1024, ef_search = 256).  At each
# target the whole calibration is repeated 400 times -- a fresh split and a
# fresh ground-truth draw each time -- and the mean over repetitions is
# compared with the target, because that is the quantity Theorem 1 bounds.
# Whiskers are the 5th to 95th percentile over repetitions, which is what a
# single deployment draw would land in.  Repetitions that certify nothing are
# carried at recall 1, the value the exhaustive fallback guarantees, and their
# share is written beside the two targets where it is not negligible.
#
# Canvas, frame fractions and type match panel (b), cropped out of
# recall_ablation.pdf (120.6 x 106.4 pt, frame l .254 r .949 b .182 t .905,
# title and axis labels 6.5 pt, ticks 6 pt, annotations 5.5 pt).
set terminal pdfcairo enhanced font "Liberation Sans,6.5" fontscale 0.75 \
    linewidth 1 size 1.6750in,1.4778in
set output "calsweep.pdf"

ACH = "#E8336D"   # achieved, the same accent (b) gives the calibrated arm
TGT = "#8C8C8C"   # the target itself
INK = "#262626"

set border 3 lw 1.2
set tics in nomirror scale 1.0,0.5 font ",6" tc rgb INK
unset x2tics; unset y2tics; unset grid
unset key

set lmargin at screen 0.254
set rmargin at screen 0.949
set bmargin at screen 0.182
set tmargin at screen 0.905

set title "MS MARCO Qwen 1,024" font ",6.5" offset 0,-0.7 tc rgb INK
set xrange [0.45:5.55]
set xtics ("0.9" 1, "0.95" 2, "0.975" 3, "0.99" 4, "0.995" 5) \
          font ",6" offset 0,0.35 scale 0
unset mxtics
set xlabel "Target recall" font ",6.5" offset 0,0.75 tc rgb INK
set yrange [-2.05:2.05]
set ytics ("{/Symbol -}2" -2, "0" 0, "+2" 2) font ",6"
set mytics 2
set ylabel "Margin (pp)" font ",6.5" offset 0.35,0 tc rgb INK

# the guarantee binds at zero: on or above it the target is met
set arrow 1 from 0.45,0 to 5.55,0 nohead lw 0.9 dt 2 lc rgb TGT back
set label 1 "target" at 5.45, -0.42 right font ",5.5" tc rgb TGT front
# the share of repetitions that certified nothing, where it is not negligible
set label 2 "47\%" at 4.0, -1.35 center font ",5.5" tc rgb INK front
set label 3 "96\%" at 5.0, -1.35 center font ",5.5" tc rgb INK front
set label 4 "escalate" at 4.5, -1.78 center font ",5.5" tc rgb INK front

plot "calsweep.dat" u 1:3:4:5 w yerrorbars lw 1.1 pt 7 ps 0.45 lc rgb ACH notitle, \
     "calsweep.dat" u 1:3 w l lw 1.5 lc rgb ACH notitle
unset output
