# io_ablation — BBQ rerank I/O modes: Sequential vs Async Batch vs Async Batch + Overlap (t=1)
# data: io_ablation.dat (make_io_ablation_dat.py <- Qwen: experiment/bbq_calibrated_io_modes_t1/summary_t1.csv,
#       OpenAI: experiment/bbq_calibrated_io_modes_t1_openai_priority_20261121/summary_t1.csv)
load "../../common/plot/common.gp"
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt, plot-box aspect 1.509, labelled ticks only on the
# x / y axes, legend on top (widths measured from Carlito 6.5pt), page cropped to the content
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 24.0; GM = 9.0; RM = 8.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5
LEGB = 0.0   # legend now inside the panels (bottom left)
W = COLW; H = PBOT + PHX + 1.0 + TTL + LEGB; OUT = "io_ablation"
eval SETTERM

set style line 1 lc rgb C_GRAY  lw LW pt 6  ps PS        # BBQ-Sequential
set style line 2 lc rgb C_BLUE  lw LW pt 12 ps PS*1.25   # BBQ-Async Batch
set style line 3 lc rgb C_PINK  lw LW pt 4  ps PS        # BBQ-Async Batch + Overlap

set multiplot
# in-panel legend entry (pt, screen): 9pt sample + marker, 2pt gap, 6pt text
ISMP = 9.0; IGST = 2.0; IFS = 6.0; IROW = 6.8
IENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, x, y, x+ISMP, y, ls, n+1, x+ISMP/2, y, ls, n+2, txt, x+ISMP+IGST, y, IFS)
LEGIN(x0) = sprintf('eval IENT(100, %g, PBOT+3.0+2.5*IROW, 3, "BBQ-Async Batch + Overlap"); eval IENT(103, %g, PBOT+3.0+1.5*IROW, 2, "BBQ-Async Batch"); eval IENT(106, %g, PBOT+3.0+0.5*IROW, 1, "BBQ-Sequential")', x0, x0, x0)
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
unset mxtics; unset mytics
XL(t) = sprintf('set xlabel "%s" font ",7" offset 0,1.0', t)
TT(t) = sprintf('set title "%s" font ",7" offset 0,-0.75', t)

D = "io_ablation.dat"
N = "seq async overlap"
PLOTPANEL(t) = sprintf('plot for [i=1:3] D index word(N,i)."_%s_curve" u 2:3 w l ls i notitle, for [i=1:3] D index word(N,i)."_%s_pts" u 2:3 w p ls i lw MLW notitle', t, t)
set logscale y
set yrange [30:4500]   # 30 (was 100): room for the in-panel legend under the Sequential curve
unset mxtics; unset mytics
YT  = 'set ytics ("100" 100, "200" 200, "500" 500, "1k" 1000, "2k" 2000)'
YT0 = 'set ytics ("" 100, "" 200, "" 500, "" 1000, "" 2000)'
eval RECALLAXIS

# MS MARCO Qwen 1,024
eval PANEL(LM, PBOT, PW, PHX)
set xrange [0.97:0.996]
set xtics ("0.97" 0.97, "0.98" 0.98, "0.99" 0.99, "0.995" 0.995) offset 0,0.45
eval YT
set ylabel "QPS" font ",7" offset 0.6,0
eval XL("Recall\\\\@10")
eval TT("Qwen 1,024")
eval LEGIN(LM+3.0)
eval PLOTPANEL("a")

# MS MARCO OpenAI 1,536 (y shared)
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
unset nonlinear x; set xrange [0.986:0.9975]; eval RECALLAXIS
set xtics ("0.99" 0.99, "0.995" 0.995, "0.997" 0.997) offset 0,0.45
eval YT0
unset ylabel
eval XL("Recall\\\\@10")
eval TT("OpenAI 1,536")
eval LEGIN(LM+PW+GM+3.0)
eval PLOTPANEL("b")
unset multiplot
