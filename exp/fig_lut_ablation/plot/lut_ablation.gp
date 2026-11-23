# lut_ablation — QPS vs recall@10, ADC-BQ scorer: int16 nibble LUT vs scalar direct (no LUT)
# no calibration, no FP reranking, t=1, ef 80/128/256/512/1024, NQ / Quora Qwen 1,024
# data: lut_ablation.dat (make_lut_ablation_dat.py <nq_quora_qwen1024_adcbq_scalar_vs_lut_no_calibration_t1/summary_t1.csv>)
# both modes return identical results per ef (same recall); only QPS differs. Curves: PCHIP through the markers.
load "../../common/plot/common.gp"
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 24.0; GM = 9.0; RM = 8.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5
W = COLW; H = PBOT + PHX + 1.0 + TTL; OUT = "lut_ablation"
eval SETTERM

set style line 1 lc rgb C_PINK  lw LW pt 4 ps PS     # BBQ, int16 nibble LUT
set style line 2 lc rgb C_BLACK lw LW pt 6 ps PS     # BBQ, scalar direct (no LUT)

set multiplot
ISMP = 7.5; IGST = 1.5; IFS = 5.0; IROW = 5.7
IENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, x, y, x+ISMP, y, ls, n+1, x+ISMP/2, y, ls, n+2, txt, x+ISMP+IGST, y, IFS)
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
unset mxtics; unset mytics
XL(t) = sprintf('set xlabel "%s" font ",7" offset 0,1.0', t)
TT(t) = sprintf('set title "%s" font ",7" offset 0,-0.75', t)

YMIN = 9.0; YMAX = 11000.0
YS(v) = PBOT + PHX*log(v/YMIN)/log(YMAX/YMIN)      # data QPS -> screen pt
# legend at the bottom-left of each panel, under the scalar curve (5pt)
LEGIN(x0) = sprintf('eval IENT(100, %g, YS(13.5)+IROW, 1, "BBQ (int16 nibble LUT)"); eval IENT(103, %g, YS(13.5), 2, "BBQ (scalar, no LUT)")', x0, x0)
D = "lut_ablation.dat"
set logscale y
set yrange [YMIN:YMAX]
set ytics ("10" 10, "100" 100, "1k" 1000, "10k" 10000)
PLOTPANEL(t) = sprintf('plot D index "lut_%s_curve" u 2:3 w l ls 1 notitle, D index "scalar_%s_curve" u 2:3 w l ls 2 notitle, D index "lut_%s_pts" u 2:3 w p ls 1 lw MLW notitle, D index "scalar_%s_pts" u 2:3 w p ls 2 lw MLW notitle', t, t, t, t)

# NQ / Qwen 1,024
eval PANEL(LM, PBOT, PW, PHX)
set xrange [0.7937:0.7978]
set xtics ("0.794" 0.794, "0.795" 0.795, "0.796" 0.796, "0.797" 0.797) offset 0,0.45
set ylabel "QPS" font ",7" offset 1.4,0
eval XL("Recall\\\\@10")
eval TT("NQ")
eval LEGIN(LM+3.0)
eval PLOTPANEL("nq")

# Quora / Qwen 1,024 (y shared)
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
set xrange [0.8271:0.8289]
set xtics ("0.8275" 0.8275, "0.828" 0.828, "0.8285" 0.8285) offset 0,0.45
set ytics ("" 10, "" 100, "" 1000, "" 10000)
unset ylabel
eval XL("Recall\\\\@10")
eval TT("Quora")
eval LEGIN(LM+PW+GM+3.0)
eval PLOTPANEL("qu")
unset multiplot
