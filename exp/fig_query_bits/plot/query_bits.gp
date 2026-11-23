# query_bits — query-side bit-width sweep (efSearch 256, t=1), NQ / Quora Qwen 1,024
# top: mean recall@10 vs query precision (2/4/8/16 bit, FP32); bottom: % of queries whose recall@10 improves
# at each step (2→4, 4→8, 8→16, 16→FP32)
# data: query_bits.dat (make_query_bits_dat.py <summary_t1.csv> <query_improvement_transitions_t1.csv>)
load "../../common/plot/common.gp"
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 24.0; GM = 9.0; RM = 6.0
PW = (COLW - LM - GM - RM) / 2.0
PHT = 52.0; PHB = 44.0; GV = 19.0
PBOT = 17.5; LEGB = 11.5
W = COLW; H = PBOT + PHB + GV + PHT + TTL + LEGB; OUT = "query_bits"
eval SETTERM

set style line 1 lc rgb C_PINK lw LW pt 4 ps PS          # BBQ
set style line 3 lc rgb C_BLUE lw LW pt 6 ps PS          # RaBitQ 1bit + adaptive rerank

set multiplot
# legend: one row at the top, centred on the panel block (Carlito 6.5pt widths)
array LT[2] = ["BBQ", "RaBitQ 1bit + adaptive rerank"]
array LWD[2] = [11.44, 77.86]
array LSS[2] = [1, 3]
SMP = 14.0; GST = 3.0; GCOL = 12.0; LY = H - 4.75
LX = LM + ((2*PW + GM) - (2*(SMP+GST) + LWD[1] + LWD[2] + GCOL)) / 2.0
do for [i=1:2] {
    eval sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front', 100+i, LX, LY, LX+SMP, LY, LSS[i])
    eval sprintf('set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front', 110+i, LX+SMP/2, LY, LSS[i])
    eval sprintf('set label %d "%s" at screen sx(%g),sy(%g) left font ",6.5" front', 120+i, LT[i], LX+SMP+GST, LY)
    LX = LX + SMP + GST + LWD[i] + GCOL
}
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
unset mxtics; unset mytics
TT(t) = sprintf('set title "%s" font ",7" offset 0,-0.75', t)
D = "query_bits.dat"
YT = PBOT + PHB + GV

# ---- top row: mean recall@10 vs query bits
set xrange [0.6:5.4]
set xtics ("2" 1, "4" 2, "8" 3, "16" 4, "FP32" 5) offset 0,0.45
set yrange [0.82:1.005]
set ytics ("0.85" 0.85, "0.9" 0.9, "0.95" 0.95, "1" 1.0) offset 0.3,0
set xlabel "Query bits" font ",7" offset 0,1.0
RECPLOT(t) = sprintf('plot D index "bbq_%s_recall" u 1:3 w lp ls 1 lw LW notitle, D index "rabitq_%s_recall" u 1:3 w lp ls 3 lw LW notitle', t, t)

eval PANEL(LM, YT, PW, PHT)
set ylabel "Recall\\@10" font ",7" offset 1.2,0
eval TT("NQ")
eval RECPLOT("nq")

eval PANEL(LM+PW+GM, YT, PW, PHT)
set ytics ("" 0.85, "" 0.9, "" 0.95, "" 1.0)
unset ylabel
eval TT("Quora")
eval RECPLOT("qu")

# ---- bottom row: % of queries improved per step (grouped bars, measured values labelled)
unset title
set xrange [0.5:4.5]
set xtics ("2→4" 1, "4→8" 2, "8→16" 3, "16→FP32" 4) offset 0,0.45
set yrange [0:90]
set ytics ("0" 0, "25" 25, "50" 50, "75" 75) offset 0.3,0
set xlabel "Query bits step" font ",7" offset 0,1.0
BW = 0.34
BARPLOT(t) = sprintf('plot D index "bbq_%s_improved" u ($1-0.18):2:($1-0.18-BW/2):($1-0.18+BW/2):(0):2 w boxxyerror fs solid 1.0 noborder lc rgb C_PINK notitle, D index "rabitq_%s_improved" u ($1+0.18):2:($1+0.18-BW/2):($1+0.18+BW/2):(0):2 w boxxyerror fs solid 1.0 noborder lc rgb C_BLUE notitle, D index "bbq_%s_improved" u ($1-0.18):($2+2.5):($2 == 0 ? "0" : $2 < 9.95 ? sprintf("%%.1f", $2) : sprintf("%%.0f", $2)) w labels font ",5" offset 0,0.25 notitle, D index "rabitq_%s_improved" u ($1+0.18):($2+2.5):($2 == 0 ? "0" : $2 < 9.95 ? sprintf("%%.1f", $2) : sprintf("%%.0f", $2)) w labels font ",5" offset 0,0.25 notitle', t, t, t, t)

eval PANEL(LM, PBOT, PW, PHB)
set ylabel "Improved (%)" font ",7" offset 1.2,0
eval BARPLOT("nq")

eval PANEL(LM+PW+GM, PBOT, PW, PHB)
set ytics ("" 0, "" 25, "" 50, "" 75)
unset ylabel
eval BARPLOT("qu")
unset multiplot
