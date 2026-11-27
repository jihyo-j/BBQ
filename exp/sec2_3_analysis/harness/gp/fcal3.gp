# Calibration analysis, panel (a): what a query gets, against what was asked.
#
# x is the calibration target, y is the per-query Recall@10 the rule actually
# delivered, and the area of each dot is the share of queries at that value.
# Recall@10 is a count out of ten, so a query lands on one of eleven heights
# and a plain scatter would overplot; the area keeps every query visible.
# Shares are pooled over the replicates that certified a finite threshold
# (target 0.99 and 0.995 certify in only 53% and 4% of calibrations, and a
# replicate that certifies nothing is a failure, not a recall of 1).
#
# The blue dash is the mean the target is compared against.  It is a number no
# query experiences: at every target the queries sit at 1.0 or 0.9.
#
#   gnuplot fcal3.gp   ->  fcal_a_bub.pdf

ACH = "#E8336D"; MEAN = "#0A0AF0"; TGT = "#8C8C8C"; INK = "#262626"

PW = 1.6384
ML = 0.380; MR = 0.075; MB = 0.330; MT = 0.060
BW = PW - ML - MR
BH = BW / 1.15
PH = BH + MB + MT
eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH)
set output "fcal_a_bub.pdf"

set border 3 lw 1.0
set tics in nomirror scale 0.8,0.4 font ",6" tc rgb INK
unset x2tics; unset y2tics; unset grid; unset key
set lmargin at screen ML/PW
set rmargin at screen (PW-MR)/PW
set bmargin at screen MB/PH
set tmargin at screen (PH-MT)/PH

set xrange [0.45:5.55]
set xtics ("0.9" 1,"0.95" 2,"0.975" 3,"0.99" 4,"0.995" 5) font ",6" \
    offset 0,0.4 scale 0
unset mxtics
set xlabel "Calibration target" font ",7" offset 0,0.6 tc rgb INK
set yrange [0.13:1.21]
set ytics ("0.2" 0.2,"0.4" 0.4,"0.6" 0.6,"0.8" 0.8,"1.0" 1.0) font ",6"
unset mytics
set ylabel "Per-query Recall\\@10" font ",7" offset 1.9,0 tc rgb INK

plot "motiv/calbub.dat"     u 1:3:(0.20+2.1*sqrt($4)) w p pt 7 ps variable \
         lc rgb ACH notitle, \
     "motiv/calbub_ref.dat" u 1:3 w p pt 1 ps 0.9 lw 1.6 lc rgb MEAN notitle
unset output
