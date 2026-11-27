# Section 4: what c_Q does and does not tell you about the budget.
# Style follows recall_qps.gp (the Section 6 main figure).
set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 linewidth 1 size 3.4in,1.98in
set output "cq.pdf"
LW = 2.6; MLW = 1.3; INK="#262626"
# panel (a): quantities, achromatic.  panel (b): the same colours and
# markers these configurations carry in Figures 2 and 3.
Q_CQ="#9A9A9A"; Q_RS="#262626"
C_MSQ="#E8336D"; C_BGE="#1A9850"; INK2="#262626"
set border 31 lw 2
set tics in mirror scale 1.3,0.7 font ",8"
unset grid; unset x2tics; unset y2tics
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0
W=0.360; H=0.470; B=0.355
set multiplot

# ---- (a) the two quantities move independently ------------------------------
set size W,H; set origin 0.150,B
set title "(a) Two interventions" font ",9" offset 0,-0.4
set logscale x; set xrange[0.6:600]
set xtics ("1×" 1,"10×" 10,"100×" 100); unset mxtics
set xlabel "factor change" font ",9" offset 0,0.5
set yrange[-0.05:1.80]; unset ytics; unset ylabel
set style arrow 1 head filled size screen 0.012,16,50 lc rgb Q_CQ lw 3.0
set style arrow 2 head filled size screen 0.012,16,50 lc rgb Q_RS lw 3.0
set label 21 "rank-preserving warp" at 0.70,1.68 left font ",7" tc rgb INK front

set label 23 "noise in the top-1000" at 0.70,0.82 left font ",7" tc rgb INK front

set label 25 "{/:Italic c_Q} 134×"  at 2.4,1.44 left font ",7.5" tc rgb Q_CQ front
set label 26 "{/:Italic R}^{*} 1.00×" at 1.35,1.05 left font ",7.5" tc rgb Q_RS front
set label 27 "{/:Italic c_Q} 1.01×" at 1.35,0.45 left font ",7.5" tc rgb Q_CQ front
set label 28 "{/:Italic R}^{*} 55×"   at 72,0.25 left font ",7.5" tc rgb Q_RS front
unset key
plot "m_dumb.dat" u (1):($1*0.80+0.45):($2-1):(0) w vectors as 1 t "{/:Italic c_Q}", \
     ""           u (1):($1*0.80+0.25):($3-1):(0) w vectors as 2 t "p95 {/:Italic R}^{*}"

# ---- (b) the ordering inverts ----------------------------------------------
set size W,H; set origin 0.605,B
set title "(b) Same-size codes" font ",9" offset 0,-0.4
unset label 21; unset label 22; unset label 23; unset label 24
unset label 25; unset label 26; unset label 27; unset label 28
unset logscale x; set xrange[0.18:0.47]
set xtics 0.1 format "%.1f"; set mxtics 2
set xlabel "score distortion {/:Italic c_Q}" font ",9" offset 0,0.5
set logscale y; set yrange[17:95]
set ytics ("20" 20,"30" 30,"50" 50); unset mytics
set ylabel "p95 {/:Italic R}^{*}" font ",9" offset 0.55,0
unset key
set label 31 "MS MARCO/Qwen" at 0.192,80 left font ",7" tc rgb C_MSQ front
set label 32 "Quora/BGE-M3"  at 0.192,66 left font ",7" tc rgb C_BGE front
plot "m_inv_ms.dat" u 1:2:3:4 w yerrorbars lc rgb C_MSQ lw MLW pt 6 ps 0.85 notitle, \
     "m_inv_qu.dat" u 1:2:3:4 w yerrorbars lc rgb C_BGE lw MLW pt 8 ps 0.90 notitle

unset multiplot
