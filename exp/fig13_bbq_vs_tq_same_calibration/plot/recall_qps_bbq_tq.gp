# recall_qps_bbq_tq — QPS vs recall@10, BBQ vs TurboQuant 1bit (calibrated adaptive rerank, t=1)
# data: recall_qps_bbq_tq.dat (make_recall_qps_bbq_tq_dat.py <summary_t1.csv>); PCHIP curves through markers at ef 80/128/256/512/1024
load "../../common/plot/common.gp"
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt, plot-box aspect 1.509, labelled ticks only on the
# x / y axes, legend on top (widths measured from Carlito 6.5pt), page cropped to the content
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 24.0; GM = 9.0; RM = 8.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5
LEGB = 0.0   # legend now inside the panels
W = COLW; H = PBOT + PHX + 1.0 + TTL + LEGB; OUT = "recall_qps_bbq_tq"
eval SETTERM

set style line 1 lc rgb C_PINK   lw LW pt 4  ps PS             # BBQ
set style line 4 lc rgb C_PURPLE lw LW pt 12 ps PS*1.25        # TurboQuant 1bit

set multiplot
# in-panel legend entry (pt, screen): 9pt sample + marker, 2pt gap, 6pt text
ISMP = 9.0; IGST = 2.0; IFS = 6.0; IROW = 6.8
IENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, x, y, x+ISMP, y, ls, n+1, x+ISMP/2, y, ls, n+2, txt, x+ISMP+IGST, y, IFS)
ITXT(n, x, y, txt) = sprintf('set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, txt, x+ISMP+IGST, y, IFS)
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
unset mxtics; unset mytics
XL(t) = sprintf('set xlabel "%s" font ",7" offset 0,1.0', t)
TT(t) = sprintf('set title "%s" font ",7" offset 0,-0.75', t)

LEGIN(x0) = sprintf('eval IENT(100, %g, PBOT+3.0+1.5*IROW, 1, "BBQ"); eval IENT(103, %g, PBOT+3.0+0.5*IROW, 4, "TurboQuant 1bit + calibration")', x0, x0)
D = "recall_qps_bbq_tq.dat"
set logscale y
set yrange [180:3500]   # 180 (was 280): room for the one-line legend under the last markers
set ytics ("300" 300, "500" 500, "1k" 1000, "2k" 2000, "3k" 3000)
unset mxtics; unset mytics
PLOTPANEL(t) = sprintf('plot D index "bbq_%s_curve" u 2:3 w l ls 1 notitle, D index "tq_%s_curve" u 2:3 w l ls 4 notitle, D index "bbq_%s_pts" u 2:3 w p ls 1 lw MLW notitle, D index "tq_%s_pts" u 2:3 w p ls 4 lw MLW notitle', t, t, t, t)

# NQ / Qwen 1,024
eval PANEL(LM, PBOT, PW, PHX)
set xrange [0.975:0.9992]
set nonlinear x via -log10(1-x) inverse 1-10**(-x)
set xtics ("0.98" 0.98, "0.99" 0.99, "0.995" 0.995, "0.998" 0.998) offset 0,0.45
unset mxtics; unset mytics
set ylabel "QPS" font ",7" offset 1.4,0
eval XL("Recall\\\\@10")
eval TT("NQ")
eval LEGIN(LM+3.0)
eval PLOTPANEL("nq")

# MS MARCO / Qwen 1,024 (y shared)
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
unset nonlinear x
set xrange [0.975:0.9962]
set nonlinear x via -log10(1-x) inverse 1-10**(-x)
set xtics ("0.98" 0.98, "0.99" 0.99, "0.995" 0.995) offset 0,0.45
set ytics ("" 300, "" 500, "" 1000, "" 2000, "" 3000)
unset ylabel
eval XL("Recall\\\\@10")
eval TT("MS MARCO")
eval LEGIN(LM+PW+GM+3.0)
eval PLOTPANEL("ms")
unset multiplot
