# colocation — colocated vs baseline page layout for 4/8/16 KB pages (Sec. 6)
# data: colocation.dat (recovered from the original figure)
load "../../common/plot/common.gp"
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt, plot-box aspect 1.509, labelled ticks only
# (x / y axes), keys stay inside the panels, page cropped
FONT = "Carlito"; FT = 6.5; FL = 7.0
LM = 22.0; GM = 27.0; RM = 4.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.509
PBOT = 17.5
W = COLW; H = PBOT + PHX + 2.0; OUT = "colocation"
eval SETTERM
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)

DT = "(9,4)"                          # baseline = dashed (2.5pt on / 1.1pt off)
set style line 1 lc rgb C_BLUE   lw LW pt 6  ps PS            # 4 KB, colocated
set style line 2 lc rgb C_BLUE   lw LW pt 6  ps PS dt @DT     # 4 KB, baseline
set style line 3 lc rgb C_PURPLE lw LW pt 4  ps PS            # 8 KB, colocated
set style line 4 lc rgb C_PURPLE lw LW pt 4  ps PS dt @DT     # 8 KB, baseline
set style line 5 lc rgb C_PINK   lw LW pt 12 ps PS*1.25       # 16 KB, colocated
set style line 6 lc rgb C_PINK   lw LW pt 12 ps PS*1.25 dt @DT # 16 KB, baseline
set style line 7 lc rgb C_BLACK  lw LW                       # key: colocated (solid)
set style line 8 lc rgb C_BLACK  lw LW dt @DT                # key: baseline (dashed)

# in-panel key entry: short line sample (+ marker) and text, in graph coordinates
KX = 0.05; KS = 0.13; KG = 0.03
KEYL(tag,ls,y,s) = sprintf('set arrow %d from graph KX,%g to graph KX+KS,%g nohead ls %d front; set label %d "%s" at graph KX+KS+KG,%g left font ",%g" front', tag, y, y, ls, tag+1, s, y, 6.0)
KEYLX(tag,ls,x,y,s) = sprintf('set arrow %d from graph %g,%g to graph %g+KS,%g nohead ls %d front; set label %d "%s" at graph %g+KS+KG,%g left font ",%g" front', tag, x, y, x, y, ls, tag+1, s, x, y, 6.0)
KEYP(tag,ls,y) = sprintf('set label %d "" at graph KX+KS/2,%g point ls %d lw MLW front', tag, y, ls)

# sub-captions are drawn as screen labels so they can be kept inside the canvas
# and apart from each other (they are wider than the panels at FL)
CAPY = 17.5                            # caption centre below the panel bottom [pt]
CAP(tag,s,x) = sprintf('set label %d "%s" at screen sx(%g),sy(%g) center font ",%g" front', tag, s, x, PBOT-CAPY, FL)

set multiplot

D = "colocation.dat"
N = "k4co k4base k8co k8base k16co k16base"
PLOTPANEL(t) = sprintf('plot for [i=1:6] D index word(N,i)."_%s_curve" u 1:2 w l ls i notitle, for [i=1:6] D index word(N,i)."_%s_pts" u 1:2 w p ls i lw MLW dt 1 notitle', t, t)
XT = 'set xtics ("10" 0.10, "25" 0.25, "50" 0.50, "75" 0.75) offset 0,0.45; unset mxtics; unset mytics'

# (a) QPS
eval PANEL(LM, PBOT, PW, PHX)
set xrange [0.050011019:0.79997851]; set yrange [43.499968:285]   # 285 (was 255.9): headroom for the two-column key
eval XT
set ytics 50,50,250 offset 0.3,0
set ylabel "QPS" font ",7" offset 1.2,0
set xlabel "Memory ratio (%)" font ",7" offset 0,1.0
# key (a): page size = colour / marker
eval KEYL(11, 1, 0.92, "4 KB");  eval KEYP(13, 1, 0.92)
eval KEYL(14, 3, 0.82, "8 KB");  eval KEYP(16, 3, 0.82)
eval KEYL(17, 5, 0.72, "16 KB"); eval KEYP(19, 5, 0.72)
# key (a), second column: line style = layout (colocated solid / baseline dashed)
eval KEYLX(21, 7, 0.42, 0.92, "colocated")
eval KEYLX(23, 8, 0.42, 0.82, "baseline")
eval PLOTPANEL("a")
unset arrow; do for [t=11:24] { unset label t }

# (b) graph page hit ratio
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
set xrange [0.050001717:0.80000856]; set yrange [10.869277:94.865916]
eval XT
set ytics 20,20,80 offset 0.3,0
set ylabel "Graph page hit ratio (%)" font ",7" offset 1.2,0
set xlabel "Memory ratio (%)" font ",7" offset 0,1.0
eval PLOTPANEL("b")
unset multiplot
