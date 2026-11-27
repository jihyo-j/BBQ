# comparision_hnsw — BBQ (RAM codes + SSD FP32) vs in-RAM FP32 HNSW (Sec. 6)
# data: comparision_hnsw.dat (make_comparision_hnsw_dat.py <- recall_qps_t1_from_recall096_rerun.csv)
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt, plot-box aspect 1.509, major ticks on x/y axes only,
# legend on top above the titles, page cropped to the content
load "../../common/plot/common.gp"
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 26.0; GM = 20.0; RM = 8.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5
LEGB = 0.0   # legend now inside the panels
W = COLW; H = PBOT + PHX + 1.5 + TTL + LEGB; OUT = "comparision_hnsw"
eval SETTERM

set style line 1 lc rgb C_PINK  lw LW pt 4 ps PS   # BBQ (RAM codes + SSD FP32)
set style line 2 lc rgb C_BLACK lw LW pt 6 ps PS   # FP32 HNSW (all in RAM)

set multiplot
# in-panel legend entry (pt, screen): 9pt sample + marker, 2pt gap, 6pt text
ISMP = 9.0; IGST = 2.0; IFS = 6.0; IROW = 6.8
IENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, x, y, x+ISMP, y, ls, n+1, x+ISMP/2, y, ls, n+2, txt, x+ISMP+IGST, y, IFS)
ITXT(n, x, y, txt) = sprintf('set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, txt, x+ISMP+IGST, y, IFS)
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)

D = "comparision_hnsw.dat"
N = "adcbq hnsw"
PLOTPANEL(t) = sprintf('plot for [i=1:2] D index word(N,i)."_%s_curve" u 1:2 w l ls i notitle, for [i=1:2] D index word(N,i)."_%s_pts" u 1:2 w p ls i lw MLW notitle', t, t)
set logscale y
unset mxtics; unset mytics
YLO = 22.0; YHI = 4500.0   # YLO 22 (was 90): room for the in-panel legend
YT = 'set ytics ("100" 100, "200" 200, "500" 500, "1,000" 1000, "2,000" 2000) offset 0.3,0'
eval RECALLAXIS
XL = 'set xlabel "Recall\\@10" font ",7" offset 0,1.0'
LEGIN(x0) = sprintf('eval IENT(100, %g, PBOT+3.0+2.5*IROW, 1, "BBQ (1-bit codes in DRAM,"); eval ITXT(110, %g, PBOT+3.0+1.5*IROW, "FP32 on SSD)"); eval IENT(103, %g, PBOT+3.0+0.5*IROW, 2, "FP32 HNSW (all in DRAM)")', x0, x0, x0)
TT(s) = sprintf('set title "%s" font ",%g" offset 0,-0.75', s, FL)

# (a) MS MARCO
eval PANEL(LM, PBOT, PW, PHX)
set xrange [0.965:0.997]; set yrange [YLO:YHI]
set xtics ("0.97" 0.97, "0.98" 0.98, "0.99" 0.99, "0.995" 0.995) offset 0,0.45
set format y "%g"
eval YT
set ylabel "QPS" font ",7" offset 1.6,0
eval XL
eval TT("MS MARCO")
eval LEGIN(LM+3.0)
eval PLOTPANEL("a")

# (b) DBpedia
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
set xrange [0.965:0.997]; set yrange [YLO:YHI]
set xtics ("0.97" 0.97, "0.98" 0.98, "0.99" 0.99, "0.995" 0.995) offset 0,0.45
eval YT
unset ylabel
eval XL
eval TT("DBpedia")
eval LEGIN(LM+PW+GM+3.0)
eval PLOTPANEL("b")
unset multiplot
