# Section 3 figure.  Three claims, one panel each, no overlap.
# Style follows recall_qps.gp (the Section 6 main figure).
set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 linewidth 1 size 7in,2.00in
set output "mech.pdf"
LW = 2.6; MLW = 1.3; INK="#262626"
C_R="#E8336D"; C_B="#0A0AF0"; C_G="#1A9850"; C_K="#505050"
set border 31 lw 2
set tics in mirror scale 1.3,0.7 font ",8"
unset grid; unset x2tics; unset y2tics
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0
W=0.250; H=0.560; B=0.320
set multiplot

# ---- (a) the two quantities move independently ------------------------------
set size W,H; set origin 0.075,B
set title "(a) The two move independently" font ",9" offset 0,-0.5
set logscale x; set xrange[0.6:600]
set xtics ("1{/*0.8 \327}" 1,"10{/*0.8 \327}" 10,"100{/*0.8 \327}" 100); unset mxtics
set xlabel "factor change from the base code" font ",9" offset 0,0.6
set yrange[-0.05:1.80]; unset ytics; unset ylabel
set style arrow 1 head filled size screen 0.009,16,50 lc rgb C_K  lw 3.0
set style arrow 2 head filled size screen 0.009,16,50 lc rgb C_R  lw 3.0
set label 21 "rank-preserving"   at 0.70,1.68 left font ",7.5" tc rgb INK front
set label 22 "warp of the score" at 0.70,1.55 left font ",7.5" tc rgb INK front
set label 23 "noise in the"      at 0.70,0.88 left font ",7.5" tc rgb INK front
set label 24 "exact top-1000"    at 0.70,0.75 left font ",7.5" tc rgb INK front
set label 25 "134{/*0.8 \327}"   at 170,1.25 left font ",8" tc rgb C_K front
set label 26 "1.00{/*0.8 \327}"  at 1.22,1.05 left font ",8" tc rgb C_R front
set label 27 "1.01{/*0.8 \327}"  at 1.22,0.45 left font ",8" tc rgb C_K front
set label 28 "55{/*0.8 \327}"    at 72,0.25 left font ",8" tc rgb C_R front
set key at graph 0.99,0.52 right center reverse Left samplen 1.2 spacing 1.15 font ",7.5" tc rgb INK
plot "m_dumb.dat" u (1):($1*0.80+0.45):($2-1):(0) w vectors as 1 t "{/:Italic c_Q}", \
     ""           u (1):($1*0.80+0.25):($3-1):(0) w vectors as 2 t "p95 {/:Italic R}^{*}"

# ---- (b) the count predicts the budget, over three orders of magnitude ------
set size W,H; set origin 0.400,B
set title "(b) Counted vs measured" font ",9" offset 0,-0.5
unset label 21; unset label 22; unset label 23; unset label 24
unset label 25; unset label 26; unset label 27; unset label 28
set logscale x; set xrange[15:4.0e4]
set xtics ("10^{1}" 10,"10^{2}" 100,"10^{3}" 1000,"10^{4}" 10000); set mxtics 10
set xlabel "measured p95 {/:Italic R}^{*}" font ",9" offset 0,0.6
unset logscale y; set yrange[0.5:2.1]; set ytics 0.5 format "%.1f"; set mytics 2
set ylabel "counted / measured" font ",9" offset 0.2,0
set arrow 31 from 15,1 to 4.0e4,1 nohead lc rgb "#9A9A9A" lw 1.8 front
set key at graph 0.97,0.97 right top reverse Left samplen 1.2 spacing 1.15 font ",7.5" tc rgb INK
plot "m_ratio.dat" u 1:($3==0?$2:1/0) w p lc rgb C_G lw MLW pt 8  ps 0.95 t "12 embeddings", \
     ""            u 1:($3==1?$2:1/0) w p lc rgb C_B lw MLW pt 4  ps 0.85 t "MRL ladder"

# ---- (c) and it tracks query to query, where c_Q does not -------------------
set size W,H; set origin 0.725,B
set title "(c) Query to query" font ",9" offset 0,-0.5
unset arrow 31
set logscale x; set xrange[8:260]
set xtics ("10" 10,"100" 100); set mxtics 10
set xlabel "that query's {/:Italic R}^{*}" font ",9" offset 0,0.6
set logscale y; set yrange[8:260]; set ytics ("10" 10,"100" 100); set mytics 10
set ylabel "counted" font ",9" offset 0.6,0
unset key
set label 35 "{/Symbol r} = +0.63" at 11,185 left font ",8" tc rgb C_G front
set label 36 "({/:Italic c_Q} alone: {/Symbol r} = {-}0.03)" at 11,140 left font ",7.5" tc rgb "#808080" front
plot x w l lc rgb "#B0B0B0" lw 1.6 dt 2 notitle, \
     "m_perq.dat" u 1:2 w p lc rgb C_G lw 1.0 pt 6 ps 0.45 notitle

unset multiplot
