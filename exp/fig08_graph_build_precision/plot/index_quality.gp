# index_quality — graph built from FP32 vs from 1-bit codes (Sec. 5)
# data: index_quality.dat (make_index_quality_dat.py, from ../results/graph_construction_precision/*_paired.csv)
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt, plot-box aspect 1.509, major ticks on x/y axes only,
# legend on top above the titles, page cropped to the content
load "../../common/plot/common.gp"
FONT = "Carlito"; FT = 6.5; FL = 7.0
set border 15 lw BLW; set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
LM = 27.0; GM = 21.0; RM = 7.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5                            # x tick labels + x title, pulled up to the axis
LEGB = 0.0                             # legend is inside each panel now
W = COLW; H = PBOT + PHX + 1.5 + TTL + LEGB; OUT = "index_quality"
eval SETTERM

set style line 1 lc rgb C_BLUE lw LW pt 6 ps PS    # FP32
set style line 2 lc rgb C_PINK lw LW pt 4 ps PS    # 1-bit codes

set multiplot
eval DRAWSCREEN
# in-panel key, bottom left (both curves start top-left and fall to the bottom right)
LFS = 6.0; LX0 = 0.04; LX1 = 0.15; LTX = 0.18; LY1 = 0.20; LY2 = 0.09
set arrow 81 from graph LX0,LY1 to graph LX1,LY1 nohead ls 1 front
set label 82 "" at graph (LX0+LX1)/2,LY1 point ls 1 lw MLW front
set label 83 "FP32" at graph LTX,LY1 left font sprintf(",%g",LFS) front
set arrow 84 from graph LX0,LY2 to graph LX1,LY2 nohead ls 2 front
set label 85 "" at graph (LX0+LX1)/2,LY2 point ls 2 lw MLW front
set label 86 "1-bit codes" at graph LTX,LY2 left font sprintf(",%g",LFS) front
set border 15 lw BLW; set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)

D = "index_quality.dat"
N = "fp32 onebit"
PLOTPANEL(t) = sprintf('plot for [i=1:2] D index word(N,i)."_%s_curve" u 1:2 w l ls i notitle, for [i=1:2] D index word(N,i)."_%s_pts" u 1:2 w p ls i lw MLW notitle', t, t)
set logscale y
unset mxtics; unset mytics
XL = 'set xlabel "Recall\\@10" font ",7" offset 0,1.0'
TT(s) = sprintf('set title "%s" font ",%g" offset 0,-0.75', s, FL)

# (a) MS MARCO
eval PANEL(LM, PBOT, PW, PHX)
set xrange [0.78706286:0.80598915]; set yrange [263.77732:10660.321]
set xtics ("0.790" 0.790, "0.795" 0.795, "0.800" 0.800, "0.805" 0.805) offset 0,0.45
set format y "%g"
set ytics ("500" 500, "1,000" 1000, "2,000" 2000, "5,000" 5000, "10,000" 10000) offset 0.3,0
set ylabel "QPS" font ",7" offset 1.6,0
eval XL
eval TT("MS MARCO")
eval PLOTPANEL("a")

# (b) Quora
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
set xrange [0.82432745:0.82900252]; set yrange [349.35151:13658.818]
set xtics ("0.825" 0.825, "0.826" 0.826, "0.827" 0.827, "0.828" 0.828, "0.829" 0.829) offset 0,0.45
set ytics ("500" 500, "1,000" 1000, "2,000" 2000, "5,000" 5000, "10,000" 10000) offset 0.3,0
unset ylabel
eval XL
eval TT("Quora")
eval PLOTPANEL("b")
unset multiplot
