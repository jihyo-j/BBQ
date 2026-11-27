# profile — (a) exact-ranking margin z(r), (b) 1-bit cost p95 R*(r)/r, per dataset
# data: profile.dat (recovered from the original figure)
load "../../../common/plot/common.gp"
NROWS = 2
W = COLW; H = PB(NROWS) + PH2 + TOPM + TTL; OUT = "profile"
eval SETTERM
# panel (b) has its own y label -> wider gap than G2
GP = G2 + 9; PWP = (COLW - L2 - GP - R2) / 2.0

# dataset colours (original mapping, snapped to the palette)
set style line 1 lc rgb C_PINK   lw LW pt 6  ps PS        # Quora
set style line 2 lc rgb C_BLUE   lw LW pt 4  ps PS        # DBpedia
set style line 3 lc rgb C_ORANGE lw LW pt 12 ps PS*1.25   # MS MARCO
set style line 4 lc rgb C_PURPLE lw LW pt 10 ps PS*1.1    # NQ
set style line 5 lc rgb C_GREEN  lw LW pt 8  ps PS*1.1    # FEVER
set style line 6 lc rgb C_GRAY   lw LW pt 2  ps PS        # GloVe

set multiplot
load "profile_legend.gp"; eval DRAWSCREEN

D = "profile.dat"
N = "quora dbpedia msmarco nq fever glove"
set logscale x
set xtics ("1" 1, "10" 10, "100" 100)
set mxtics 10
set xrange [1:100]
eval XLABEL("rank {/:Italic r}")
PLOTPANEL(t, n) = sprintf('plot for [i=1:%d] D index word(N,i)."_%s" u 1:2 w l ls i notitle, for [i=1:%d] D index word(N,i)."_%s" u 1:2 w p ls i lw MLW notitle', n, t, n, t)

# (a) exact ranking
eval PANEL(L2, PB(NROWS), PWP, PH2)
set yrange [3.4:11.0]; set ytics 4,2,10
eval YLABEL("{/:Italic z}({/:Italic r})")
eval TITLE("(a) Exact ranking")
eval PLOTPANEL("a", 6)

# (b) 1-bit cost (no GloVe)
eval PANEL(L2+PWP+GP, PB(NROWS), PWP, PH2)
set yrange [0:11.5]; set ytics 0,2,10
eval YLABEL("p95 {/:Italic R}^*({/:Italic r})/{/:Italic r}")
eval TITLE("(b) 1-bit cost")
eval PLOTPANEL("b", 5)
unset multiplot
