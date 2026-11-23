# ram_bbq_rbq7 — QPS vs recall@10, NQ Qwen 1,024, everything in RAM, 1 thread:
#   BBQ (1-bit codes + calibrated FP32 rerank, FP32 vectors in RAM) vs RaBitQ 7-bit c16 (RAM, no rerank)
# data: ram_bbq_rbq7.dat (make_ram_bbq_rbq7_dat.py); PCHIP curves through the measured markers
load "../../common/plot/common.gp"
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 24.0; RM = 8.0
PW = 104.0
PHX = PW / 1.509
PBOT = 17.5
W = LM + PW + RM; H = PBOT + PHX + 1.0 + TTL; OUT = "ram_bbq_rbq7"
eval SETTERM

set style line 1 lc rgb C_PINK lw LW pt 4 ps PS          # BBQ
set style line 4 lc rgb C_BLUE lw LW pt 8 ps PS*1.1      # RaBitQ 7bit

set multiplot
ISMP = 9.0; IGST = 2.0; IFS = 6.0; IROW = 6.8
IENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, x, y, x+ISMP, y, ls, n+1, x+ISMP/2, y, ls, n+2, txt, x+ISMP+IGST, y, IFS)
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
unset mxtics; unset mytics
D = "ram_bbq_rbq7.dat"

eval PANEL(LM, PBOT, PW, PHX)
eval IENT(100, LM+3.0, PBOT+3.0+1.5*IROW, 1, "BBQ (FP32 in RAM)")
eval IENT(103, LM+3.0, PBOT+3.0+0.5*IROW, 4, "RaBitQ 7bit (RAM)")
set logscale y
set yrange [150:12000]
set ytics ("200" 200, "500" 500, "1k" 1000, "2k" 2000, "5k" 5000, "10k" 10000) offset 0.3,0
set xrange [0.945:0.9992]
set nonlinear x via -log10(1-x) inverse 1-10**(-x)
set xtics ("0.95" 0.95, "0.98" 0.98, "0.99" 0.99, "0.995" 0.995, "0.998" 0.998) offset 0,0.45
unset mxtics; unset mytics
set ylabel "QPS" font ",7" offset 1.4,0
set xlabel "Recall\\@10" font ",7" offset 0,1.0
set title "NQ (all in RAM)" font ",7" offset 0,-0.75
plot D index "bbq_curve" u 2:3 w l ls 1 notitle, D index "rbq7_curve" u 2:3 w l ls 4 notitle, \
     D index "bbq_pts" u 2:3 w p ls 1 lw MLW notitle, D index "rbq7_pts" u 2:3 w p ls 4 lw MLW notitle
unset multiplot
