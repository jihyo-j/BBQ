# recall_k — QPS vs Recall@k for k = 30 / 50 / 100 on NQ Qwen 1,024 (t=1)
# data: recall_k.dat (make_recall_k_dat.py <nq_msmarco_qwen1024_recall_k30_50_100_t1.csv>)
#   BBQ: per-ef pool-top-k split-conformal rerank (target 0.99); TurboQuant 1bit: fixed FP rerank 160/320/512;
#   RaBitQ 1bit: c16 + native adaptive FP rerank (sync pread). Curves: PCHIP through the markers.
load "../../common/plot/common.gp"
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt; labelled ticks only, on the x / y axes;
# legend on top; page cropped to the content
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM3 = 23.0; G3 = 9.0; RM3 = 7.0
PW3 = (COLW - LM3 - RM3 - 2*G3) / 3.0
PBOT = 17.5; LEGB = 9.5
W = COLW; H = PBOT + PH2 + 1.5 + LEGB; OUT = "recall_k"
eval SETTERM

set style line 1 lc rgb C_PINK   lw LW pt 4  ps PS             # BBQ
set style line 4 lc rgb C_PURPLE lw LW pt 12 ps PS*1.25        # TurboQuant 1bit
set style line 3 lc rgb C_BLUE   lw LW pt 6  ps PS             # RaBitQ 1bit

set multiplot
# legend: one row at the very top; entry widths measured from Carlito 6.5pt
array LT[3] = ["BBQ", "TurboQuant 1bit", "RaBitQ 1bit + adaptive rerank"]
array LWD[3] = [11.44, 44.03, 77.86]
array LSS[3] = [1, 4, 3]
SMP = 14.0; GST = 3.0; GCOL = 10.0; LY = H - 4.75
# centred on the panel block (LM3 .. LM3+3*PW3+2*G3), not on the page
LX = LM3 + ((3*PW3 + 2*G3) - (3*(SMP+GST) + LWD[1] + LWD[2] + LWD[3] + 2*GCOL)) / 2.0
do for [i=1:3] {
    eval sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front', 100+i, LX, LY, LX+SMP, LY, LSS[i])
    eval sprintf('set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front', 110+i, LX+SMP/2, LY, LSS[i])
    eval sprintf('set label %d "%s" at screen sx(%g),sy(%g) left font ",6.5" front', 120+i, LT[i], LX+SMP+GST, LY)
    LX = LX + SMP + GST + LWD[i] + GCOL
}
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
unset mxtics; unset mytics
XL(s) = sprintf('set xlabel "%s" font ",%g" offset 0,1.0', s, FL)

D = "recall_k.dat"
N = "bbq tq rabitq"; LS = "1 4 3"
set logscale y
set yrange [12:2000]
PLOTPANEL(k) = sprintf('plot for [i=1:3] D index word(N,i)."_k%d_curve" u 2:3 w l ls int(word(LS,i)) notitle, for [i=1:3] D index word(N,i)."_k%d_pts" u 2:3 w p ls int(word(LS,i)) lw MLW notitle', k, k)
RECALL = 'set nonlinear x via -log10(1-x) inverse 1-10**(-x)'

# k = 30
eval PANEL(LM3, PBOT, PW3, PH2)
set xrange [0.965:0.999]; eval RECALL
set xtics ("0.97" 0.97, "0.99" 0.99, "0.998" 0.998) offset 0,0.45
set ytics ("20" 20, "50" 50, "100" 100, "200" 200, "500" 500, "1k" 1000, "2k" 2000) offset 0.3,0
set ylabel "QPS" font ",7" offset 0.6,0
eval XL("Recall\\\\@30")
eval PLOTPANEL(30)

# k = 50 (y shared)
eval PANEL(LM3+PW3+G3, PBOT, PW3, PH2)
unset nonlinear x; set xrange [0.978:0.9978]; eval RECALL
set xtics ("0.98" 0.98, "0.99" 0.99, "0.995" 0.995) offset 0,0.45
set ytics ("" 20, "" 50, "" 100, "" 200, "" 500, "" 1000, "" 2000)
unset ylabel
eval XL("Recall\\\\@50")
eval PLOTPANEL(50)

# k = 100
eval PANEL(LM3+2*(PW3+G3), PBOT, PW3, PH2)
unset nonlinear x; set xrange [0.98:0.9965]; eval RECALL
set xtics ("0.98" 0.98, "0.99" 0.99, "0.995" 0.995) offset 0,0.45
eval XL("Recall\\\\@100")
eval PLOTPANEL(100)
unset multiplot
