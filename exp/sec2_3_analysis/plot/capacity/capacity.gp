# capacity — index size vs number of vectors, d = 1024 / 4096 (Sec. 2)
# data: capacity.dat (recovered from the original figure)
load "../../../common/plot/common.gp"
NROWS = 1
W = COLW; H = PB(NROWS) + PH2 + TOPM + TTL; OUT = "capacity"
eval SETTERM

set style line 1 lc rgb C_BLACK lw LW pt 6  ps PS      # FP
set style line 2 lc rgb C_BLUE  lw LW pt 4  ps PS      # 4-bit code
set style line 3 lc rgb C_PINK  lw LW pt 12 ps PS*1.2  # 1-bit code
C_SHADE  = "#FBF0EF"     # "Exceeds DRAM budget" region
C_SHADET = "#8A3B38"     # its label
BUDGET = 1007.2564

set multiplot
load "capacity_legend.gp"; eval DRAWSCREEN

D = "capacity.dat"
set logscale xy
set format x "10^{%L}"; set format y "10^{%L}"
set xtics 10; set ytics 10
set mxtics 10; set mytics 10
set xrange [1e7:1e10]; set yrange [0.5:29992.5]
set clip two
set style textbox opaque fc rgb "white" noborder margins 0.5,0.5

# vlab: value labels at the foot of each drop line
PANELPLOT(t, l1, l2, l3) = sprintf('\
set object 1 rect from 1e7,BUDGET to 1e10,29992.5 fc rgb C_SHADE fs solid 1.0 noborder behind; \
set label 1 "Exceeds DRAM budget" at 1.25e7,2730 left font ",%g" tc rgb C_SHADET front; \
set label 2 "1007 GiB\nDRAM budget" at graph 0.975, first 120 right font ",%g" boxed front; \
plot D index "drop_%s" every ::0::0 u 1:2:(0):(0.5-$2) w vectors nohead lc rgb C_BLACK lw MLW dt (1.5,1.5) notitle, \
     D index "drop_%s" every ::1::1 u 1:2:(0):(0.5-$2) w vectors nohead lc rgb C_BLUE  lw MLW dt (1.5,1.5) notitle, \
     D index "drop_%s" every ::2::2 u 1:2:(0):(0.5-$2) w vectors nohead lc rgb C_PINK  lw MLW dt (1.5,1.5) notitle, \
     BUDGET w l lc rgb C_BLACK lw MLW*1.6 dt (6,3) notitle, \
     D index "fp_%s" u 1:2 w l ls 1 notitle, D index "b4_%s" u 1:2 w l ls 2 notitle, D index "b1_%s" u 1:2 w l ls 3 notitle, \
     D index "drop_%s" every ::0::0 u 1:2 w p ls 1 lw MLW notitle, D index "drop_%s" every ::1::1 u 1:2 w p ls 2 lw MLW notitle, D index "drop_%s" every ::2::2 u 1:2 w p ls 3 lw MLW notitle, \
     D index "drop_%s" every ::0::0 u 1:(0.62):("%s") w labels left rotate by 90 offset 0.9,0 font ",%g" tc rgb C_BLACK notitle, \
     D index "drop_%s" every ::1::1 u 1:(0.62):("%s") w labels left rotate by 90 offset 0.9,0 font ",%g" tc rgb C_BLUE notitle, \
     D index "drop_%s" every ::2::2 u 1:(0.62):("%s") w labels left rotate by 90 offset 0.9,0 font ",%g" tc rgb C_PINK notitle', \
  FA, FA, t,t,t,t,t,t,t,t,t, t,l1,FA, t,l2,FA, t,l3,FA)

# (a) d = 1024
eval PANEL(L2, PB(NROWS), PW2, PH2)
eval TITLE("{/:Italic d} = 1024")
eval YLABEL("Index size (GiB)")
eval XLABEL("Number of vectors")
eval PANELPLOT("a", "251M", "1.5B", "3.2B")

# (b) d = 4096
eval PANEL(L2+PW2+G2, PB(NROWS), PW2, PH2)
eval TITLE("{/:Italic d} = 4096")
unset ylabel
set format y ""
eval PANELPLOT("b", "65M", "479M", "1.5B")
unset multiplot
