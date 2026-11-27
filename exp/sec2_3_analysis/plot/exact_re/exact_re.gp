# exact_re — candidate recall vs. reranking budget R (left) and the
# minimum budget needed to reach a target recall (right).  Sec. 3.
# data: exact_re.dat (recovered from the original figure)
load "../../../common/plot/common.gp"
NROWS = 3                              # legend rows (2 rows would be 265pt wide)
BOT = 37.0                             # room under the panels: rotated bar labels (> XLAB)
PBOT = LEGTOP(NROWS) + BOT
GX = 30.0                              # wider gap than G2: the bar panel has its own (2-line) y label
PWX = (COLW - L2 - GX - R2) / 2.0
W = COLW; H = PBOT + PH2 + TOPM + TTL; OUT = "exact_re"
eval SETTERM

set style line 1 lc rgb C_PINK   lw LW pt 6  ps PS        # SymBQ
set style line 2 lc rgb C_PINK   lw LW pt 4  ps PS        # ADC-BQ
set style line 3 lc rgb C_GREEN  lw LW pt 10 ps PS*1.1    # PQ
set style line 4 lc rgb C_GREEN  lw LW pt 12 ps PS*1.25   # OPQ
set style line 5 lc rgb C_PURPLE lw LW pt 12 ps PS*1.25   # TurboQuant 1bit
set style line 6 lc rgb C_BLUE   lw LW pt 8  ps PS*1.1    # RaBitQ 1bit
set style line 7 lc rgb C_ORANGE lw LW pt 6  ps PS        # SQ 4bit
set style line 8 lc rgb C_ORANGE lw LW pt 4  ps PS        # SQ 8bit
BAR_LT = "#D2C3EE"; BAR_DK = "#8566C4"                   # lavender / muted purple
set style line 11 lc rgb BAR_LT lw MLW                    # recall >= 0.95
set style line 12 lc rgb BAR_DK lw MLW                    # recall >= 0.99

set multiplot
load "exact_re_legend.gp"; eval DRAWSCREEN

D = "exact_re.dat"
N = "symbq adcbq pq opq tq rabitq sq4 sq8"

# ---- (left) candidate recall@10 vs candidate budget R ----
eval PANEL(L2, PBOT, PWX, PH2)
set xrange [10:80]; set yrange [0.7:1.00802]
set xtics 10,20,70
set ytics 0.7,0.1,1.0 format "%.2f"
set mxtics 1; set mytics 1
eval TITLE("MS MARCO Qwen 1,024")
eval XLABEL("candidate budget {/:Italic R}")
eval YLABEL("candidate recall\\\\@10")
plot for [i=1:8] D index word(N,i)."_curve" u 1:2 w l ls i notitle, \
     for [i=1:8] D index word(N,i)."_pts" u 1:2 w p ls i lw MLW notitle

# ---- (right) minimum reranking budget at target recall ----
eval PANEL(L2+PWX+GX, PBOT, PWX, PH2)
BW = 0.38
set xrange [-0.768:7.768]; set yrange [0:112]
set xtics ("SymBQ" 0, "ADC-BQ" 1, "PQ" 2, "OPQ" 3, "TurboQuant 1bit" 4, "RaBitQ 1bit" 5, "SQ 4bit" 6, "SQ 8bit" 7) rotate by 38 right offset 0.6,0.2 nomirror
set ytics 0,20,100 format "%g"
set mytics 1
set style fill solid 1.0 noborder
eval TITLE("MS MARCO Qwen 1,024")
unset xlabel
eval YLABEL("minimum reranking\\nbudget {/:Italic R}")
set ylabel offset -0.4,0               # 2-line label: shift left of the tick labels
plot D index "bars" u ($1-BW/2):3:(BW) w boxes ls 11 notitle, \
     D index "bars" u ($1+BW/2):4:(BW) w boxes ls 12 notitle
unset multiplot
