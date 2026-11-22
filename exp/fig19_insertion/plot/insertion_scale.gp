# insertion_scale — Recall@10 and insert/rebuild time vs. number of inserted vectors
# (MS MARCO, Qwen3-0.6B, 1M base + 10M..100M random inserts, efSearch 256, 32 threads)
# data: insertion_scale.dat (make_insertion_scale_dat.py <- frozen_adcbq_vs_rabitq_c16_comparison.csv)
load "../../common/plot/common.gp"
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt, plot-box aspect 1.509, labelled ticks only
# (x / y axes), legend on top, page cropped
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 25.0; GM = 27.0; RM = 4.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5; LEGB = 0.0   # legend now inside both panels
W = COLW; H = PBOT + PHX + 3.0 + LEGB; OUT = "insertion_scale"   # no title row
eval SETTERM

set style line 1 lc rgb C_PINK lw LW         pt 4 ps PS   # BBQ, online insert (frozen 1M tau)
set style line 2 lc rgb C_PINK lw LW dt (4,2) pt 5 ps PS  # BBQ, full rebuild (recalibrated)
set style line 3 lc rgb C_BLUE lw LW         pt 6 ps PS   # RaBitQ, online insert
set style line 4 lc rgb C_BLUE lw LW dt (4,2) pt 7 ps PS  # RaBitQ, full rebuild
set style line 5 lc rgb C_PURPLE lw LW pt 12 ps PS*1.25   # TurboQuant 1bit (TQ1+, R=60), online insert

CAPY = 25.0
CAP(tag,s,x) = sprintf('set label %d "%s" at screen sx(%g),sy(%g) center font ",%g" front', tag, s, x, PBOT-CAPY, FL)

set multiplot
# in-panel legend entry (pt, screen): 9pt sample + marker, 2pt gap, 6pt text
ISMP = 7.5; IGST = 1.5; IFS = 5.0; IROW = 5.7
# key-style entry: text right-aligned, sample at the right end (xr = right edge)
RENT(n, xr, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) right font ",%g" front', n, xr-ISMP, y, xr, y, ls, n+1, xr-ISMP/2, y, ls, n+2, txt, xr-ISMP-IGST, y, IFS)
IENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, x, y, x+ISMP, y, ls, n+1, x+ISMP/2, y, ls, n+2, txt, x+ISMP+IGST, y, IFS)
ITXT(n, x, y, txt) = sprintf('set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, txt, x+ISMP+IGST, y, IFS)
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)

D = "insertion_scale.dat"
N = "bbq_online bbq_rebuild rabitq_online rabitq_rebuild"
set xrange [-5:105]
set xtics ("0" 0, "25" 25, "50" 50, "75" 75, "100" 100) offset 0,0.45; unset mxtics

# (a) Recall@10
eval PANEL(LM, PBOT, PW, PHX)
set yrange [0.9745:0.9985]   # top 0.9985 (was 0.996): headroom for the in-panel legend
set ytics ("0.975" 0.975, "0.98" 0.98, "0.985" 0.985, "0.99" 0.99, "0.995" 0.995); unset mytics
set ylabel "Recall\\@10" font ",7" offset 1.0,0
set xlabel "Inserted vectors (M)" font ",7" offset 0,1.0
# legend top right, above where the BBQ / RaBitQ lines level off
# sample before the text; block right-aligned to the panel (widest text 59.89pt at 5pt)
LX0 = LM+PW-3.0-(ISMP+IGST+59.89); LY0 = PBOT+PHX-3.0-0.5*IROW
eval IENT(100, LX0, LY0, 3, "RaBitQ 1bit + adaptive rerank"); eval IENT(103, LX0, LY0-IROW, 1, "BBQ"); eval IENT(106, LX0, LY0-2*IROW, 5, "TurboQuant 1bit")
plot for [i=1:3:2] D index word(N,i) u 1:2 w l ls i notitle, \
     for [i=1:3:2] D index word(N,i) u 1:2 w p ls i lw MLW notitle, \
     D index "tq_online" u 1:2 w l ls 5 notitle, D index "tq_online" u 1:2 w p ls 5 lw MLW notitle
unset label 91; unset arrow 100; unset arrow 103; unset arrow 106; do for [i=100:108] { unset label i }

# (b) cumulative online-insert time (online series only)
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
set yrange [0:3.2]
set ytics ("0" 0, "1" 1, "2" 2, "3" 3); unset mytics
set ylabel "Time (h)" font ",7" offset 0.4,0
# legend top-left, rows in the order the curves stack at 100M (RaBitQ, BBQ, TurboQuant)
LX0 = LM+PW+GM+3.0; LY0 = PBOT+PHX-3.0-0.5*IROW
eval IENT(100, LX0, LY0, 3, "RaBitQ 1bit + adaptive rerank"); eval IENT(103, LX0, LY0-IROW, 1, "BBQ"); eval IENT(106, LX0, LY0-2*IROW, 5, "TurboQuant 1bit")
set xlabel "Inserted vectors (M)" font ",7" offset 0,1.0
plot for [i=1:3:2] D index word(N,i) u 1:3 w l ls i notitle, \
     for [i=1:3:2] D index word(N,i) u 1:3 w p ls i lw MLW notitle, \
     D index "tq_online" u 1:3 w l ls 5 notitle, D index "tq_online" u 1:3 w p ls 5 lw MLW notitle
unset label 92
unset multiplot
