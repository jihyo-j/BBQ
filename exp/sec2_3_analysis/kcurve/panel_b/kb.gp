# Figure 3(b), redrawn: the candidate budget against how many neighbors are
# asked for.  Same six configurations and same styles as panel (a), the same
# two-strip broken axis the panel already used, but k now runs to 100 and the
# lower strip is logarithmic, because at k=100 the Transformer budgets reach
# the hundreds and a linear strip cannot hold both ends.
#
# p_kcorpus6.dat: k, then mean R* under ADC-BQ for Quora/Qwen, NQ/Qwen,
# MS MARCO/Qwen, MS MARCO/BGE, GloVe, GIST (src/kcurve.py, 1500 queries;
# 1000 on GIST, which is all the collection has).
load "house_paper.gp"
set style line 22 lc rgb "#9933E6" lw LW pt 10 ps 0.80*PSC   # NQ/Qwen
set style line 27 lc rgb "#909090" lw LW pt 2  ps 0.72*PSC   # GloVe
set style line 28 lc rgb INK       lw LW pt 1  ps 0.72*PSC   # GIST
F = "p_kcorpus6.dat"
set output "f3b_new.pdf"
Y0 = MB/PH; Y1 = (PH-MT)/PH; HB = Y1-Y0
set multiplot
set logscale x; set xrange [0.85:120]

#   upper strip: the two non-learned controls, which never leave the corpus
set bmargin at screen Y0+0.80*HB
set tmargin at screen Y1
unset xtics; set xlabel ""
set logscale y; set yrange [150:2.2e6]
set ytics ("10^{3}" 1000,"10^{6}" 1000000) font FT nomirror scale 0.6,0
unset mytics; set ylabel ""
set label 30 "{/*1.15 \342\211\210}" at graph -0.012, graph -0.24 center font FT tc rgb INK front
set label 31 "{/*1.15 \342\211\210}" at graph  1.012, graph -0.24 center font FT tc rgb INK front
plot F u 1:6 w l ls 27 notitle, F u 1:7 w l ls 28 notitle, \
     F u 1:6 every 30::8 w p ls 27 lw MLW notitle, \
     F u 1:7 every 30::20 w p ls 28 lw MLW notitle

#   lower strip: the four Transformer configurations, carrying both labels
unset label 30; unset label 31
set bmargin at screen Y0
set tmargin at screen Y0+0.70*HB
set xtics ("1" 1,"10" 10,"100" 100) font FT nomirror scale 0.6,0 offset 0,0.4
set mxtics 10
set xlabel "{/:Italic k}" font FA offset 0,1.15 tc rgb INK
set logscale y; set yrange [0.9:1400]
set ytics ("1" 1,"10" 10,"10^{2}" 100) font FT nomirror scale 0.6,0; unset mytics
set ylabel "mean {/:Italic R}^{*}" font FA offset 1.6,1.6 tc rgb INK
plot for [i=2:5] F u 1:i w l ls (i==2?21:i==3?22:i==4?24:25) notitle, \
     for [i=2:5] F u 1:i every 26::(4*(i-2)+3) w p ls (i==2?21:i==3?22:i==4?24:25) lw MLW notitle
unset multiplot
unset output
