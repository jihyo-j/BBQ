# Figure 3(b), with the low-k end given room.  The y axis keeps the break the
# panel already used (the two non-learned controls sit three decades above the
# Transformer configurations); the x axis is broken as well, so k = 1..10 --
# the range the convention lives in -- is linear and takes more than half the
# width, and 10..100 is logarithmic beyond it.
load "house_paper.gp"
set style line 22 lc rgb "#9933E6" lw LW pt 10 ps 0.80*PSC
set style line 27 lc rgb "#909090" lw LW pt 2  ps 0.72*PSC
set style line 28 lc rgb INK       lw LW pt 1  ps 0.72*PSC
F = "p_kcorpus6.dat"
set output "f3b_split.pdf"

X0 = ML/PW; X1 = (PW-MR)/PW; WB = X1-X0
Y0 = MB/PH; Y1 = (PH-MT)/PH; HB = Y1-Y0
XS = 0.58; XG = 0.055              # x split: 58% to k = 1..10
YS = 0.80; YG = 0.10               # y split, as before
LX1 = X0 + (XS-XG/2)*WB
RX0 = X0 + (XS+XG/2)*WB
UY0 = Y0 + YS*HB
LY1 = Y0 + (YS-YG)*HB
set multiplot

# ---- upper band: GloVe and GIST ------------------------------------------
set logscale y; set yrange [150:2.2e6]
set bmargin at screen UY0; set tmargin at screen Y1
unset xtics; set xlabel ""; set ylabel ""
set lmargin at screen X0; set rmargin at screen LX1
unset logscale x; set xrange [0.4:10.6]
set ytics ("10^{3}" 1000,"10^{6}" 1000000) font FT nomirror scale 0.6,0
unset mytics
plot F u 1:6 w l ls 27 notitle, F u 1:7 w l ls 28 notitle, \
     F u 1:6 every 4::1 w p ls 27 lw MLW notitle, \
     F u 1:7 every 4::3 w p ls 28 lw MLW notitle
set lmargin at screen RX0; set rmargin at screen X1
set logscale x; set xrange [10:112]
unset ytics
plot F u 1:6 w l ls 27 notitle, F u 1:7 w l ls 28 notitle, \
     F u 1:6 every 24::13 w p ls 27 lw MLW notitle, \
     F u 1:7 every 24::25 w p ls 28 lw MLW notitle

# ---- lower band: the four Transformer configurations ---------------------
set logscale y; set yrange [0.9:1400]
set bmargin at screen Y0; set tmargin at screen LY1
set lmargin at screen X0; set rmargin at screen LX1
unset logscale x; set xrange [0.4:10.6]
set xtics ("1" 1,"4" 4,"7" 7,"10" 10) font FT nomirror scale 0.6,0 offset 0,0.4
set mxtics 3
set ytics ("1" 1,"10" 10,"10^{2}" 100) font FT nomirror scale 0.6,0; unset mytics
set ylabel "mean {/:Italic R}^{*}" font FA offset 1.6,1.9 tc rgb INK
set label 40 "{/*1.15 \342\211\210}" at screen X0-0.012, screen (LY1+UY0)/2 \
    center font FT tc rgb INK front
set label 41 "{/:Italic k}" at screen (X0+X1)/2, screen 0.055 center font FA tc rgb INK front
set label 42 "{/*1.15 \342\211\210}" at screen (LX1+RX0)/2, screen Y0-0.012 \
    center font FT tc rgb INK front
plot for [i=2:5] F u 1:i w l ls (i==2?21:i==3?22:i==4?24:25) notitle, \
     for [i=2:5] F u 1:i every 3::(i-2) w p ls (i==2?21:i==3?22:i==4?24:25) lw MLW notitle
unset label 40; unset label 41; unset label 42
set lmargin at screen RX0; set rmargin at screen X1
set logscale x; set xrange [10:112]
set xtics ("20" 20,"50" 50,"100" 100) font FT nomirror scale 0.6,0 offset 0,0.4
unset mxtics; unset ytics; set ylabel ""
plot for [i=2:5] F u 1:i w l ls (i==2?21:i==3?22:i==4?24:25) notitle, \
     for [i=2:5] F u 1:i every 22::(13+2*(i-2)) w p ls (i==2?21:i==3?22:i==4?24:25) lw MLW notitle
unset multiplot
unset output
