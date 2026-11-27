# Calibration analysis, panel (a): every calibration as a dot.
#
# One filled dot per calibration replicate that certified a finite threshold,
# stacked into bins of 0.0005 recall -- a dot histogram, so the shape is the
# distribution and each dot is still one replicate.  Target 0.99, ef=256,
# m=1000, 400 replicates, MS MARCO Qwen-1024.  The 187 replicates that
# certified nothing are a failure, not a recall of 1, and are reported in the
# caption rather than placed on the axis.
#
# Two vertical lines fix the reading.  The target is what the rule promises in
# expectation.  The ceiling, 1 - pi = 0.99073, is what any threshold can reach
# at all: 0.93% of the exact top-10 never entered the search pool.  Target 0.99
# sits 0.073 pp under it, which is why the distribution is pressed against the
# ceiling and left-skewed, and why target 0.995 -- above the ceiling -- can
# only escalate.
#
#   gnuplot fcal2.gp   ->  fcal_a_strip.pdf

ACH = "#E8336D"; TGT = "#262626"; CEIL = "#8C8C8C"; INK = "#262626"

PW = 1.6384
ML = 0.330; MR = 0.075; MB = 0.330; MT = 0.060
BW = PW - ML - MR
BH = BW / 1.45
PH = BH + MB + MT
eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH)
set output "fcal_a_strip.pdf"

set border 1 lw 1.0
set tics in nomirror scale 0.8,0.4 font ",6" tc rgb INK
unset ytics; unset x2tics; unset y2tics; unset grid; unset key
set lmargin at screen ML/PW
set rmargin at screen (PW-MR)/PW
set bmargin at screen MB/PH
set tmargin at screen (PH-MT)/PH

set xrange [0.9742:0.99165]
set xtics ("0.975" 0.975,"0.98" 0.98,"0.985" 0.985,"0.99" 0.99) font ",6" offset 0,0.4
unset mxtics
set xlabel "Achieved Recall\\@10" font ",7" offset 0,0.6 tc rgb INK
set yrange [0:36]

set arrow 1 from 0.99,0 to 0.99,29 nohead lw 1.1 dt 2 lc rgb TGT front
set arrow 2 from 0.99073,0 to 0.99073,32.5 nohead lw 1.1 lc rgb CEIL front
set label 1 "ceiling 0.9907" at 0.99055,33 right font ",6" tc rgb CEIL front
set label 2 "target 0.99"    at 0.98980,27.5 right font ",6" tc rgb TGT front

plot "motiv/calrep99.dat" u 1:2 w p pt 7 ps 0.30 lc rgb ACH notitle
unset output
