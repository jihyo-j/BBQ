# page_io — SSD pages read per query vs recall@10 (Sec. 6)
# data: page_io.dat (make_page_io_dat.py <- experiment/page_io_text_embedding_3_small/figure/normalized_page_io.csv,
#       checked against operating_points.csv in the same folder); t=1, MS MARCO
load "../../common/plot/common.gp"
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt, plot-box aspect 1.509, labelled ticks only on the
# x / y axes, legend on top (widths measured from Carlito 6.5pt), page cropped to the content
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 27.0; GM = 9.0; RM = 8.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5
LEGB = 0.0   # legend inside the OpenAI panel (top left)
W = COLW; H = PBOT + PHX + 1.0 + TTL + LEGB; OUT = "page_io"
eval SETTERM

set style line 1 lc rgb C_PINK  lw LW pt 4  ps PS                 # BBQ
set style line 2 lc rgb C_GREEN lw LW pt 10 ps PS*1.1             # DiskANN
set style line 3 lc rgb C_BLUE  lw LW pt 6  ps PS               # DiskANN RaBitQ+adaptive rerank (solid)
set style line 4 lc rgb C_PURPLE lw LW pt 12 ps PS*1.25         # TurboQuant1 (fixed R: 120 / 60 FP reads)

set multiplot
# in-panel legend entry (pt, screen): 7.5pt sample + marker, 1.5pt gap, 5pt text
ISMP = 7.5; IGST = 1.5; IFS = 5.0; IROW = 5.7
IENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, x, y, x+ISMP, y, ls, n+1, x+ISMP/2, y, ls, n+2, txt, x+ISMP+IGST, y, IFS)
ITXT(n, x, y, txt) = sprintf('set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, txt, x+ISMP+IGST, y, IFS)
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
unset mxtics; unset mytics
XL(t) = sprintf('set xlabel "%s" font ",7" offset 0,1.0', t)
TT(t) = sprintf('set title "%s" font ",7" offset 0,-0.75', t)

D = "page_io.dat"
N = "bbq diskann rabitq tq"
set logscale y
set yrange [40:10000]   # 10k: headroom for the in-panel legends (top left)
set ytics ("50" 50, "100" 100, "200" 200, "500" 500, "1k" 1000, "2k" 2000, "5k" 5000)
unset mxtics; unset mytics
PLOTPANEL(t) = sprintf('plot for [i=1:4] D index word(N,i)."_%s_curve" u 1:2 w l ls i notitle, for [i=1:4] D index word(N,i)."_%s_pts" u 1:2 w p ls i lw MLW notitle', t, t)

# (a) MS MARCO / OpenAI1536
eval PANEL(LM, PBOT, PW, PHX)
# recall axis x' = -log10(1-x): spreads the high-recall end (BBQ vs TurboQuant1 final recall)
set xrange [0.981:0.9975]
set nonlinear x via -log10(1-x) inverse 1-10**(-x)
set xtics ("0.985" 0.985, "0.99" 0.99, "0.995" 0.995, "0.997" 0.997) offset 0,0.45
unset mxtics; unset mytics
set ylabel "pages read / query" font ",7" offset 0.8,0
eval XL("recall\\\\@10")
eval TT("MS MARCO OpenAI 1,536")
# legend top-left, rows in the curves' top-to-bottom order; widest text 36.57pt at 5pt clears the DiskANN curves
LX0 = LM+3.0; LYT = PBOT+PHX-3.0-0.5*IROW
eval IENT(100, LX0, LYT, 3, "DiskANN RaBitQ +"); eval ITXT(103, LX0, LYT-IROW, "adaptive rerank")
eval IENT(104, LX0, LYT-2*IROW, 2, "DiskANN PQ 1bit"); eval IENT(107, LX0, LYT-3*IROW, 4, "TurboQuant 1bit")
eval IENT(110, LX0, LYT-4*IROW, 1, "BBQ")
eval PLOTPANEL("a")
unset arrow; do for [t=100:112] { unset label t }

# (b) MS MARCO / Qwen1024 (y axis shared with (a))
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
set xrange [0.965:0.9955]
set xtics ("0.97" 0.97, "0.98" 0.98, "0.99" 0.99, "0.995" 0.995) offset 0,0.45
set ytics ("" 50, "" 100, "" 200, "" 500, "" 1000, "" 2000, "" 5000)
unset ylabel
eval XL("recall\\\\@10")
eval TT("MS MARCO Qwen 1,024")
# same legend, top left of the Qwen panel
LX0 = LM+PW+GM+3.0; LYT = PBOT+PHX-3.0-0.5*IROW
eval IENT(100, LX0, LYT, 3, "DiskANN RaBitQ +"); eval ITXT(103, LX0, LYT-IROW, "adaptive rerank")
eval IENT(104, LX0, LYT-2*IROW, 2, "DiskANN PQ 1bit"); eval IENT(107, LX0, LYT-3*IROW, 4, "TurboQuant 1bit")
eval IENT(110, LX0, LYT-4*IROW, 1, "BBQ")
eval PLOTPANEL("b")
unset multiplot
