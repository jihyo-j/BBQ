# Calibration analysis, panel (a): what a query gets, against what was asked.
#
# Per-query Recall@10 is a count out of ten, so the outcome is an ordered
# category, and the share of queries in each is stacked to 100%.  An ordered
# variable gets an ordered ramp, not four unrelated hues, so the four slabs run
# from the paper's pink down to grey.  Only the first bar is labelled: the
# slabs keep their order and their colour across the figure.
#
# Shares are pooled over the calibration replicates that certified a finite
# threshold.  A replicate that certifies nothing is a failure, not a recall of
# 1 -- honouring that guarantee costs a full scan, 468k vectors per query at
# target 0.99 -- so the failure rate is printed above the two targets where it
# is not zero, in red.
#
# The mean the calibration delivered is panel (b)'s quantity and is not
# repeated here; what this panel adds is that no query experiences it -- at
# every target the queries sit at ten or nine out of ten.
#
#   gnuplot fcal.gp   ->  fcal_a.pdf

TEN = "#E8336D"; NINE = "#F07FA0"; EIGHT = "#F9C3D1"; LOW = "#D9D9D9"
INK = "#262626"; FAIL = "#B01030"

PW = 1.6384
ML = 0.400; MR = 0.075; MB = 0.330; MT = 0.060
BW = PW - ML - MR
BH = BW / 1.23
PH = BH + MB + MT
eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH)
set output "fcal_a.pdf"

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
set yrange [0:112]
set ytics ("0" 0,"50" 50,"100" 100) font ",6"
unset mytics
set ylabel "Queries (%)" font ",7" offset 1.6,0 tc rgb INK
set boxwidth 0.70
set style fill solid 1.0 noborder

#  i target ten nine eight seven_or_less fail mean reads
plot "motiv/calqdist.dat" u 1:(100) w boxes lc rgb LOW notitle, \
     ""  u 1:(100*($3+$4+$5)) w boxes lc rgb EIGHT notitle, \
     ""  u 1:(100*($3+$4))    w boxes lc rgb NINE  notitle, \
     ""  u 1:(100*$3)         w boxes lc rgb TEN   notitle, \
     ""  u 1:($7 > 0 ? 106 : NaN):(sprintf("%.0f%%", 100*$7)) w labels \
         center font ",6" tc rgb FAIL notitle, \
     ""  u ($1 == 1 ? 1 : NaN):(100*$3/2):("10") w labels center \
         font ",5.5" tc rgb "white" notitle, \
     ""  u ($1 == 1 ? 1 : NaN):(100*($3+$4/2)):("9") w labels center \
         font ",5.5" tc rgb "white" notitle, \
     ""  u ($1 == 1 ? 1 : NaN):(100*($3+$4+$5/2)):("8") w labels center \
         font ",5.5" tc rgb INK notitle, \
     ""  u ($1 == 1 ? 1 : NaN):(100*(1-$6/2)):("≤7") w labels center \
         font ",5" tc rgb INK notitle
unset output
