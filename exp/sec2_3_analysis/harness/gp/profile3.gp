# BBQ Section 4, Figure 1 -- style copied from recall_qps.gp
# pdfcairo renders font size x fontscale x 4/3 -> fontscale 0.75 makes sizes real points
set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 linewidth 1 size 7in,2.18in
set output "profile3.pdf"

# ---------- style (same palette as recall_qps.gp) ----------
LW  = 2.6
MLW = 1.3
C_QUORA = "#E8336D"   # BQ-family pink
C_DBP   = "#0A0AF0"   # blue
C_MSM   = "#FF8000"   # orange
C_NQ    = "#9933E6"   # purple
C_FEV   = "#1A9850"   # green
C_GLV   = "#808080"   # grey
set style line 1 lc rgb C_QUORA lw LW pt 6  ps 0.75   # Quora    open circle
set style line 2 lc rgb C_DBP   lw LW pt 4  ps 0.75   # DBpedia  open square
set style line 3 lc rgb C_MSM   lw LW pt 12 ps 0.95   # MS MARCO open diamond
set style line 4 lc rgb C_NQ    lw LW pt 10 ps 0.85   # NQ       open down-triangle
set style line 5 lc rgb C_FEV   lw LW pt 8  ps 0.85   # FEVER    open up-triangle
set style line 6 lc rgb C_GLV   lw LW pt 2  ps 0.70   # GloVe    x
set style line 9 lc rgb "black" lw 2.0 dt 2           # fit

set border lw 2
set xtics in mirror scale 1.3,0.7 font ",8"
set ytics in mirror scale 1.3,0.7 font ",8"
unset grid
unset key
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0

W = 0.245; H = 0.490; B = 0.415
L1 = 0.075; L2 = 0.405; L3 = 0.735

set multiplot

# ---------- legend (bottom), two-pass so the sample shows line + marker ----------
unset border; unset tics; unset xlabel; unset ylabel
set size 1,1; set origin 0,0
set xrange [0:1]; set yrange [0:1]
set key vertical maxrows 2 reverse Left samplen 3 spacing 1.25 font ",8.5"
KEYY = 0.235
set key at screen 0.165,KEYY left top textcolor rgb "black"
plot NaN w l ls 1 t "Quora", NaN w l ls 2 t "DBpedia"
set key at screen 0.165,KEYY left top textcolor rgb "#FFFFFFFF"
plot NaN w p ls 1 lw MLW t "Quora", NaN w p ls 2 lw MLW t "DBpedia"
set key at screen 0.325,KEYY left top textcolor rgb "black"
plot NaN w l ls 3 t "MS MARCO", NaN w l ls 4 t "NQ"
set key at screen 0.325,KEYY left top textcolor rgb "#FFFFFFFF"
plot NaN w p ls 3 lw MLW t "MS MARCO", NaN w p ls 4 lw MLW t "NQ"
set key at screen 0.495,KEYY left top textcolor rgb "black"
plot NaN w l ls 5 t "FEVER", NaN w l ls 6 t "GloVe"
set key at screen 0.495,KEYY left top textcolor rgb "#FFFFFFFF"
plot NaN w p ls 5 lw MLW t "FEVER", NaN w p ls 6 lw MLW t "GloVe"
set key at screen 0.670,KEYY left top textcolor rgb "black"
plot NaN w p ls 1 lw MLW t "(c) Qwen3-0.6B", NaN w p ls 2 lw MLW t "(c) other encoders"
unset key
set border lw 2
set xtics in mirror scale 1.3,0.7; set ytics in mirror scale 1.3,0.7

# ---------- (a) exact ranking ----------
set size W,H; set origin L1,B
set title "(a) Exact ranking" font ",9.5" offset 0,-0.6
set logscale x; set xrange [0.9:115]; set mxtics 10
set xtics ("1" 1, "10" 10, "100" 100)
set xlabel "rank {/:Italic r}" font ",9.5" offset 0,0.5
set ylabel "{/:Italic z}({/:Italic r})" font ",9.5" offset 0.8,0
set yrange [3.5:11]; set ytics 2; set mytics 2
plot for [i=2:7] "zrank.dat"     u 1:i w l  ls (i-1) notitle, \
     for [i=2:7] "zrank_pts.dat" u 1:i w p  ls (i-1) lw MLW notitle

# ---------- (b) 1-bit cost ----------
set size W,H; set origin L2,B
set title "(b) 1-bit cost" font ",9.5" offset 0,-0.6
set logscale x; set xrange [0.9:11]; set xtics ("1" 1, "10" 10); set mxtics 10
set xlabel "rank {/:Italic r}" font ",9.5" offset 0,0.5
set ylabel "p95 {/:Italic R}^{*}({/:Italic r})/{/:Italic r}" font ",9.5" offset 0.8,0
unset logscale y; set yrange [0:10.5]; set ytics 2; set mytics 2
plot for [i=2:6] "cost.dat" u 1:i w lp ls (i-1) lw LW pointsize \
        (i==2?0.75:i==3?0.75:i==4?0.95:i==5?0.85:0.85) notitle

# ---------- (c) price of one bit ----------
set size W,H; set origin L3,B
set title "(c) Price of one bit" font ",9.5" offset 0,-0.6
unset logscale x; set xrange [0.45:0.95]; set xtics 0.1; set mxtics 2
set xlabel "β" font ",9.5" offset 0,0.5
set logscale y; set yrange [2:420]
set ytics ("3" 3, "10" 10, "30" 30, "100" 100, "300" 300); set mytics 10
set ylabel "extra candidates" font ",9.5" offset 0.8,0
set label 1 "FEVER"   at 0.505,150 font ",7.5" tc rgb "black" left
set label 2 "ada-002" at 0.565,12.5 font ",7.5" tc rgb "black" left
set label 3 "Quora"   at 0.845,4.6  font ",7.5" tc rgb "black" right
plot "price_qwen.dat"  u 1:2 w p ls 1 lw MLW notitle, \
     "price_other.dat" u 1:2 w p ls 2 lw MLW notitle
unset label 1; unset label 2; unset label 3

unset multiplot
