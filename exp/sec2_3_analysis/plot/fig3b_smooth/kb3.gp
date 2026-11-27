# Figure 3(b) on a per-answer axis: candidates needed for each neighbor kept,
# R*(k)/k.  Dividing out k removes the part of the growth that is arithmetic
# -- holding k answers needs at least k candidates -- and leaves the slack the
# code has to be given, which is what varies between representations.
#
# The axis break is kept, but it now separates a band the Transformer
# configurations share (one decade, 1.2 to 8.6) from the two non-learned
# controls.  GIST falls across the panel because its R* saturates at the
# corpus size, so dividing by k drives the ratio down; that is a ceiling, not
# an improvement, and the caption says so.
LW  = 1.2                        # jihyo copy: same curve width as the other f3 panels (fig3_src/fig3.gp)
load "house_paper.gp"
set style line 22 lc rgb "#9933E6" lw LW pt 10 ps 0.80*PSC
set style line 27 lc rgb "#909090" lw LW pt 2  ps 0.72*PSC
set style line 28 lc rgb INK       lw LW pt 1  ps 0.72*PSC
F = "p_kcorpus6_smooth.dat"                # smoothed copy (smooth_kcorpus6.py); raw: p_kcorpus6.dat
set output "f3b_perk.pdf"
Y0 = MB/PH; Y1 = (PH-MT)/PH; HB = Y1-Y0
set multiplot
set logscale x; set xrange [0.85:120]

#   upper strip: the two non-learned controls
set bmargin at screen Y0+0.72*HB
set tmargin at screen Y1
unset xtics; set xlabel ""; set ylabel ""
set logscale y; set yrange [120:1.6e6]
set ytics ("10^{3}" 1000,"10^{6}" 1000000) font FT nomirror scale 0.6,0
unset mytics
set label 30 "{/*1.15 \342\211\210}" at graph -0.012, graph -0.34 center font FT tc rgb INK front
set label 31 "{/*1.15 \342\211\210}" at graph  1.012, graph -0.34 center font FT tc rgb INK front
plot F u 1:($6/$1) w l ls 27 notitle, F u 1:($7/$1) w l ls 28 notitle, \
     F u 1:($6/$1) every 30::8  w p ls 27 lw MLW notitle, \
     F u 1:($7/$1) every 30::20 w p ls 28 lw MLW notitle

#   lower strip: the four Transformer configurations, one decade to themselves
unset label 30; unset label 31
set bmargin at screen Y0
set tmargin at screen Y0+0.62*HB
set xtics ("1" 1,"10" 10,"100" 100) font FT nomirror scale 0.6,0 offset 0,0.4
set mxtics 10
set xlabel "{/:Italic k}" font FA offset 0,1.15 tc rgb INK
set logscale y; set yrange [1.05:11]
set ytics ("1" 1,"2" 2,"5" 5,"10" 10) font FT nomirror scale 0.6,0; unset mytics
set ylabel "{/:Italic R}^{*}/{/:Italic k}" font FA offset 1.0,1.4 tc rgb INK
plot for [i=2:5] F u 1:(column(i)/$1) w l ls (i==2?21:i==3?22:i==4?24:25) notitle, \
     for [i=2:5] F u 1:(column(i)/$1) every 26::(4*(i-2)+3) \
         w p ls (i==2?21:i==3?22:i==4?24:25) lw MLW notitle
unset multiplot
unset output
