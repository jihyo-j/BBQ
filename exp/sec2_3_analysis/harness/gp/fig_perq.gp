# Section 3, the per-query candidate requirement.
# Panel box proportions, border weight, tic style, fonts, markers and the
# bottom legend all follow recall_qps.gp (the Section 6 main figure).
set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 linewidth 1 size 3.4in,1.98in
set output "perq.pdf"
LW = 2.6; MLW = 1.3; INK="#262626"
# same colour and marker per series as Figure 2's legend
set style line 21 lc rgb "#0A0AF0" lw LW pt 4  ps 0.80   # Quora/Qwen      square
set style line 23 lc rgb "#FF8000" lw LW pt 12 ps 0.95   # DBpedia/Qwen    diamond
set style line 24 lc rgb "#E8336D" lw LW pt 6  ps 0.80   # MS MARCO/Qwen   circle
set style line 25 lc rgb "#1A9850" lw LW pt 8  ps 0.90   # MS MARCO/BGE    triangle
set border 31 lw 2
set tics in mirror scale 1.3,0.7 font ",8"
unset grid; unset x2tics; unset y2tics
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0
# wide-and-short box, as in the main figure
W=0.360; H=0.445; B=0.375
set multiplot

# ---- (a) how many queries a budget of R serves completely ------------------
set size W,H; set origin 0.150,B
set title "(a) Coverage vs {/:Italic R}" font ",9" offset 0,-0.4
set logscale x; set xrange[9:700]
set xtics ("10" 10,"100" 100); set mxtics 10
set xlabel "candidate budget {/:Italic R}" font ",9" offset 0,0.5
set yrange[0:1.04]; set ytics ("0" 0,"0.5" 0.5,"1" 1); set mytics 2
set ylabel "queries fully served" font ",9" offset 1.05,0
unset key
set arrow 1 from 9,0.95 to 700,0.95 nohead lc rgb "#9A9A9A" lw 1.4 dt 3 front
set label 1 "0.95" at 10.5,0.885 left font ",6.5" tc rgb "#707070" front
plot "p_cdf.dat" u 1:2 w lp ls 21 lw LW pi 16 pt 4  notitle, \
     ""          u 1:3 w lp ls 24 lw LW pi 16 pt 6  notitle, \
     ""          u 1:4 w lp ls 23 lw LW pi 16 pt 12 notitle, \
     ""          u 1:5 w lp ls 25 lw LW pi 16 pt 8  notitle

# ---- (b) what one fixed budget costs against a per-query oracle ------------
unset arrow 1; unset label 1
set size W,H; set origin 0.615,B
set title "(b) Fixed vs oracle" font ",9" offset 0,-0.4
unset logscale x; set xrange[0.80:0.998]
set xtics ("0.80" 0.80,"0.90" 0.90,"0.99" 0.99); set mxtics 2
set xlabel "coverage target" font ",9" offset 0,0.5
set yrange[1:6.3]; set ytics 1; set mytics 2
set ylabel "fixed / oracle" font ",9" offset 1.30,0
unset key
plot "p_overpay.dat" u 1:2 w lp ls 21 lw LW pi 7 pt 4  notitle, \
     ""              u 1:3 w lp ls 24 lw LW pi 7 pt 6  notitle, \
     ""              u 1:4 w lp ls 23 lw LW pi 7 pt 12 notitle, \
     ""              u 1:5 w lp ls 25 lw LW pi 7 pt 8  notitle

# ---- shared legend, line + marker, as in the main figure -------------------
unset border; unset tics; unset xlabel; unset ylabel; unset title
unset logscale x; unset logscale y
set size 1,1; set origin 0,0; set xrange[0:1]; set yrange[0:1]
set key at screen 0.5,0.175 center top horizontal maxrows 2 reverse Left \
    samplen 3 spacing 1.25 width 1 font ",8.5" tc rgb INK
plot NaN w lp ls 21 lw MLW t "Quora/Qwen",    NaN w lp ls 23 lw MLW t "DBpedia/Qwen", \
     NaN w lp ls 24 lw MLW t "MS MARCO/Qwen", NaN w lp ls 25 lw MLW t "MS MARCO/BGE"

unset multiplot
