# calibration_analysis — per-query outcomes of calibrated BBQ, MS MARCO Qwen 1,024
# (efSearch 256, calibration target 0.99, 5,980 test queries); values from tab:calibration_analysis
# one panel: bars = FP reads/query (left y), line = latency (right y), share of queries written inside each bar
load "../../common/plot/common.gp"
NROWS = 1
LP = 24.0; RP = 26.0
PW = COLW - LP - RP; PHc = 82.0
XLABc = XLAB + 1.0
PBc = LEGTOP(NROWS) + XLABc
W = COLW; H = PBc + PHc + TOPM + 3.0; OUT = "calibration_analysis"
eval SETTERM

$D << EOD
0 "1.0" 5489  91.79  51.5   0.921
1 "0.9" 414   6.92   75.4   1.035
2 "{/Symbol \243}0.8" 77    1.29   119.8  1.192
EOD

set style line 1 lc rgb "#3B6FD4" lw MLW         # bars: medium blue
set style line 2 lc rgb C_BLACK lw LW pt 7 ps PS*1.1

set multiplot
load "calibration_analysis_legend.gp"; eval DRAWSCREEN

eval PANEL(LP, PBc, PW, PHc)
set boxwidth 0.42
set style fill solid 1.0 border lc rgb "#1F3F8A"
set xrange [-0.6:2.6]
set xtics scale 0 offset 0,-0.1
set yrange [0:200]; set ytics 0,50,200 nomirror; set mytics 2
set y2range [0:1.5]; set y2tics 0,0.3,1.5 nomirror; set my2tics 3
set y2tics scale TICMAJ,TICMIN
eval YLABEL("FP reads / query"); set ylabel offset 0.6,0
set y2label "Latency (ms)" font ",".sprintf("%g",FL) offset -0.8,0
eval XLABEL("Per-query Recall\\\\@10")
plot $D u 1:5:xtic(2) w boxes ls 1 notitle, \
     $D u 1:5:(sprintf("%.1f",$5)) w labels offset 0,0.5 font ",".sprintf("%g",FA) notitle, \
     $D u 1:(28):(sprintf("%.1f%%\nof queries",$4)) w labels tc rgb "white" font ",5" notitle, \
     $D u 1:6 axes x1y2 w lp ls 2 lw LW notitle, \
     $D u 1:6 axes x1y2 w p ls 2 lw MLW notitle, \
     $D u 1:6:(sprintf("%.3f ms",$6)) axes x1y2 w labels offset 0,0.75 font ",".sprintf("%g",FA) notitle
unset multiplot
