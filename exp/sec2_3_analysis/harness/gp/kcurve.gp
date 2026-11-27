# R*(k) past the k=10 convention, one panel per corpus.
#
# The x axis is broken the way Figure 3(b) breaks its y axis.  The left strip
# runs k = 1..10 linearly, which is the range the convention lives in and where
# the codes are close enough that a log axis piles them on top of each other;
# the right strip runs 10..100 logarithmically, which is where they separate.
# One shared log y axis, so heights are comparable across the break.
#
#   gnuplot kcurve.gp   ->  fkc_<corpus>.pdf

PW = 1.7227
AR = 1.00                        # a taller box: the codes sit within 2x of
                                 # each other below k=10 and need the room
load "house.gp"

set style line 41 lc rgb "#909090" lw LW pt 2  ps 0.72*PSC   # Sym-BQ
set style line 42 lc rgb "#E8336D" lw LW pt 4  ps 0.72*PSC   # ADC-BQ
set style line 43 lc rgb "#0A0AF0" lw LW pt 8  ps 0.80*PSC   # RaBitQ
set style line 44 lc rgb "#1A9850" lw LW pt 12 ps 0.86*PSC   # PQ
set style line 45 lc rgb "#262626" lw LW pt 6  ps 0.72*PSC   # OPQ

X0 = ML/PW; X1 = (PW-MR)/PW; WB = X1-X0
SPLIT = 0.56                 # share of the box given to k = 1..10
GAP   = 0.055

if (!exists("CORP")) CORP = "msmarco quora"
do for [c in CORP] {
  f = sprintf("../out/kcurve_%s_qw06.dat", c)
  set output sprintf("fkc_%s.pdf", c)
  set multiplot

  set logscale y; set yrange [1.8:4500]      # the measured range, no slack
  set ytics ("2" 2,"10" 10,"10^{2}" 100,"10^{3}" 1000) font FT; unset mytics

  #   left strip: the convention's own range, linear in k
  set lmargin at screen X0
  set rmargin at screen X0 + (SPLIT-GAP/2)*WB
  unset logscale x
  set xrange [0.4:10.6]
  set xtics ("1" 1,"4" 4,"7" 7,"10" 10) font FT; set mxtics 3
  set xlabel ""
  set ylabel "p95 {/:Italic R}^{*}" font FA offset 1.4,0 tc rgb INK
  set border 31 lw BLW
  set label 20 "{/*1.15 \342\211\210}" at screen X0+SPLIT*WB, screen 0.247 \
      center font FT tc rgb INK front
  #  one x label for both strips, so it centres on the box and not on a strip
  set label 22 "Answers kept ({/:Italic k})" at screen (X0+X1)/2, screen 0.072 \
      center font FA tc rgb INK front
  plot for [i=0:4] f u 1:(column(3+2*i)) w l ls (41+i) notitle, \
       for [i=0:4] f u 1:(column(3+2*i)) every 1 w p ls (41+i) lw MLW notitle
  unset label 20; unset label 22

  #   right strip: past the convention, logarithmic
  set lmargin at screen X0 + (SPLIT+GAP/2)*WB
  set rmargin at screen X1
  set logscale x
  set xrange [10:112]
  set xtics ("20" 20,"50" 50,"100" 100) font FT; unset mxtics
  set ylabel ""
  set ytics ("" 2,"" 10,"" 100,"" 1000) font FT
  plot for [i=0:4] f u 1:(column(3+2*i)) w l ls (41+i) notitle, \
       for [i=0:4] f u 1:(column(3+2*i)) every 6 w p ls (41+i) lw MLW notitle

  unset multiplot
  unset output
}

# the x label spans the break, so it is its own strip under both
set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 \
    linewidth 1 size 5.2634in,0.190in
set output "fkc_legend.pdf"
unset border; unset tics; unset xlabel; unset ylabel
unset label 20; unset label 21; unset label 22
unset logscale x; unset logscale y
set lmargin at screen 0; set rmargin at screen 1
set bmargin at screen 0; set tmargin at screen 1
set xrange [0:1]; set yrange [0:1]
set key at graph 0.5,1.05 center top horizontal maxrows 1 reverse Left \
    samplen 2.2 spacing 1.1 width 0 font ",6.5" tc rgb INK
plot NaN w lp ls 41 lw MLW t "Sym-BQ", NaN w lp ls 42 lw MLW t "ADC-BQ", \
     NaN w lp ls 43 lw MLW t "RaBitQ", NaN w lp ls 44 lw MLW t "PQ", \
     NaN w lp ls 45 lw MLW t "OPQ"
unset output
