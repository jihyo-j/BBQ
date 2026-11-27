# preserve — CandidateRecall@R vs candidate budget R: sym-BQ vs ADC (FP query)
# data: preserve.dat (recovered from the original figure)
load "../../common/plot/common.gp"
NROWS = 0                       # legend sits inside panel (a)
LP = L2 + 3.0                   # a little more room: y title moved away from the tick labels
GP = G2 + 9; PWP = (COLW - LP - GP - R2) / 2.0
PHP = PWP / 1.509               # plot-box aspect of f2a (1.231in x 0.816in)
PBOT = PB(NROWS) - 5.0          # x title pulled up towards the axis
W = COLW; H = PBOT + PHP + TOPM + TTL; OUT = "preserve"
# tight crop (no white border): trim measured margins [pt], 1.5pt pad on every side
CL = 4.25; CR = 4.25; CB = 1.5; CT = 3.25
W = W - CL - CR; H = H - CB - CT
PANEL(x0,y0,w,h) = sprintf("set origin %g,%g; set size %g,%g", (x0-CL)/W, (y0-CB)/H, w/W, h/H)
FONT = "Carlito"                # same typeface as f1/f2 (this figure only)
eval SETTERM

set style line 1 lc rgb C_PINK lw LW          pt 6 ps PS     # sym-BQ (solid)
set style line 2 lc rgb C_BLUE lw LW          pt 4 ps PS     # ADC, FP query (blue solid)

set multiplot

D = "preserve.dat"
set logscale x
set format x "10^{%L}"
set xtics 1,10 nomirror
unset mxtics; unset mytics     # major ticks only, on the bottom/left axes only
set xlabel "candidates {/:Italic R}" font sprintf(",%g",FL) offset 0,0.5
PLOTPANEL(t) = sprintf('plot D index "sym_%s" u 1:2 w l ls 1 notitle, D index "adc_%s" u 1:2 w l ls 2 notitle, D index "sym_%s" u 1:2 w p ls 1 lw MLW notitle, D index "adc_%s" u 1:2 w p ls 2 lw MLW notitle', t, t, t, t)

# (a) MS MARCO
eval PANEL(LP, PBOT, PWP, PHP)
set xrange [8.1228:787.99]; set yrange [0.6:1.015]   # headroom so markers at 1.0 stay inside the frame
set ytics 0.6,0.1,1.0 format "%.1f" nomirror; unset mytics
eval YLABEL("CandidateRecall\\\\@{/:Italic R}"); set ylabel offset 0.3,0
eval TITLE("MS MARCO")
LFS = 6.0; LX0 = 0.56; LX1 = 0.69; LTX = 0.72; LY1 = 0.21; LY2 = 0.09
set arrow 81 from graph LX0,LY1 to graph LX1,LY1 nohead ls 2 front     # ADC-BQ on top (as LUT in panel b)
set label 82 "" at graph (LX0+LX1)/2,LY1 point ls 2 lw MLW front
set label 83 "ADC-BQ" at graph LTX,LY1 left font sprintf(",%g",LFS) front
set arrow 84 from graph LX0,LY2 to graph LX1,LY2 nohead ls 1 front
set label 85 "" at graph (LX0+LX1)/2,LY2 point ls 1 lw MLW front
set label 86 "Sym-BQ" at graph LTX,LY2 left font sprintf(",%g",LFS) front
eval PLOTPANEL("a")

# (b) Quora: ADC-BQ scalar (no LUT) vs LUT, Recall@10-QPS (efSearch 80..1024; recall identical)
# data: preserve_lut.dat <- experiment/quora_bge1024_adcbq_scalar_vs_lut_no_calibration_t1/paired_ef_comparison_t1.csv
DL = "preserve_lut.dat"
unset arrow; unset label
eval PANEL(LP+PWP+GP, PBOT, PWP, PHP)
set style line 3 lc rgb C_PINK lw LW pt 8 ps PS*1.3      # ADC-BQ, scalar (no LUT)
set style line 4 lc rgb C_BLUE lw LW pt 12 ps PS*1.3     # ADC-BQ, LUT (open diamond)
unset logscale x
set xrange [0.844:0.848]; set yrange [10:10000]
set xtics 0.844,0.001,0.847 format "%.3f" nomirror
set xlabel "Recall\\@10" font sprintf(",%g",FL) offset 0,0.5
set logscale y; set ytics 10,10 format "10^{%L}" nomirror; unset mytics
eval YLABEL("QPS"); set ylabel offset 1.6,0
eval TITLE("Quora")
LX0 = 0.05; LX1 = 0.18; LTX = 0.21; LY1 = 0.21; LY2 = 0.09   # legend: bottom left
set arrow 91 from graph LX0,LY1 to graph LX1,LY1 nohead ls 4 front
set label 92 "" at graph (LX0+LX1)/2,LY1 point ls 4 lw MLW front
set label 93 "LUT" at graph LTX,LY1 left font sprintf(",%g",LFS) front
set arrow 94 from graph LX0,LY2 to graph LX1,LY2 nohead ls 3 front
set label 95 "" at graph (LX0+LX1)/2,LY2 point ls 3 lw MLW front
set label 96 "No LUT" at graph LTX,LY2 left font sprintf(",%g",LFS) front
plot DL index "lut" u 3:2 w l ls 4 notitle, DL index "nolut" u 3:2 w l ls 3 notitle, \
     DL index "lut" u 3:2 w p ls 4 lw MLW notitle, DL index "nolut" u 3:2 w p ls 3 lw MLW notitle
unset multiplot
