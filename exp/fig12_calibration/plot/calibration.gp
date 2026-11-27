# calibration — Sec 6.5 fig:cali, MS MARCO Qwen-1024, efSearch 256, legacy GT (same as Fig 11)
#   left : calibration-target sweep: pool top-10 preserved, Recall@10, FP32 reads/query (numbers)
#   right: Recall@10 loss at target 0.99 split into L_visit / L_select
# data: calibration.dat (make_calibration_dat.py ../results/legacy_gt)
load "../../common/plot/common.gp"
FONT = "Carlito"; FT = 6.5; FL = 7.0; FA = 6.0
LM = 27.0; GM = 33.0; RM = 4.0
PW = (COLW - LM - GM - RM) / 2.0
PHX = PW / 1.2
PBOT = 22.0
W = COLW; H = PBOT + PHX + 4.0; OUT = "calibration"
eval SETTERM
D = "calibration.dat"

set style line 1 lc rgb C_PINK  lw LW pt 12 ps PS*1.3      # pool top-10 preserved
set style line 2 lc rgb C_BLACK lw LW pt 7  ps PS*1.1      # Recall@10
set style line 3 lc rgb C_GRAY  lw 1.0 dt (3,3)            # y = target / 1-recall reference

set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)
unset mxtics; unset mytics
XL(t) = sprintf('set xlabel "%s" font ",%g" offset 0,0.6', t, FL)
YL(t) = sprintf('set ylabel "%s" font ",%g" offset 0.4,0', t, FL)

set multiplot

# ---------------- right: loss decomposition ----------------
eval PANEL(LM+PW+GM, PBOT, PW, PHX)
set xrange [-0.6:1.6]
set yrange [0:0.0125]
set ytics 0,0.002,0.012 format "%.3f"
set xtics ("{/:Italic L}_{visit}" 0, "{/:Italic L}_{select}" 1) offset 0,0.3
set boxwidth 0.5
set style fill solid 1.0 border lc rgb "#1F1F1F"
eval YL("Recall\\\\@10 loss"); set ylabel offset -0.6,0
eval XL("Loss split at target 0.99")
stats D index "decomp" u 3 nooutput
TOT = STATS_sum
set arrow 1 from -0.6,TOT to 1.6,TOT nohead ls 3 back
set label 1 sprintf("Total: 1 {/Symbol -} Recall\\@10 = %.4f", TOT) at 1.55,TOT+0.0006 right font ",".sprintf("%g",FA)
plot D index "decomp" u 1:3:($1==0 ? 0x0A0AF0 : 0xE8336D) w boxes lc rgb variable notitle, \
     D index "decomp" u 1:($1==0 ? $3/2 : NaN):(sprintf("%d\nqueries", $4)) w labels tc rgb "white" font ",5.5" notitle, \
     D index "decomp" u 1:($1==1 ? $3 : NaN):(sprintf("%d queries", $4)) w labels offset 0,0.8 font ",".sprintf("%g",FA) notitle
unset arrow; unset label

# ---------------- left: calibration-target sweep ----------------
eval PANEL(LM, PBOT, PW, PHX)
set xrange [0.87:0.9977]
set yrange [0.86:0.9968]
set nonlinear x via -log10(1-x) inverse 1-10**(-x)
set nonlinear y via -log10(1-y) inverse 1-10**(-y)
set xtics ("0.9" 0.9, "0.95" 0.95, "0.975" 0.975, "0.99" 0.99, "0.995" 0.995) offset 0,0.3
set ytics ("0.9" 0.9, "0.95" 0.95, "0.975" 0.975, "0.99" 0.99, "0.995" 0.995) format "%g"
eval YL("Achieved")
eval XL("Calibration target")
set samples 200
# in-panel legend (upper left), graph coords
LX = 0.04; LY = 0.95; LR = 0.08
set arrow 11 from graph LX,LY to graph LX+0.09,LY nohead ls 1 front
set label 12 "" at graph LX+0.045,LY point ls 1 lw MLW front
set label 13 "Pool top-10 preserved" at graph LX+0.12,LY left font ",".sprintf("%g",FA) front
set arrow 14 from graph LX,LY-LR to graph LX+0.09,LY-LR nohead ls 2 front
set label 15 "" at graph LX+0.045,LY-LR point ls 2 front
set label 16 "Recall\\@10" at graph LX+0.12,LY-LR left font ",".sprintf("%g",FA) front
set arrow 17 from graph LX,LY-2*LR to graph LX+0.09,LY-2*LR nohead ls 3 front
set label 18 "{/:Italic y} = target" at graph LX+0.12,LY-2*LR left font ",".sprintf("%g",FA) front
set label 19 "{/:Italic n}: FP32 reads / query" at graph 0.97,0.06 right font ",".sprintf("%g",FA) tc rgb C_PINK front
plot x w l ls 3 notitle, \
     D index "sweep" u 1:2 w l ls 1 notitle, \
     D index "sweep" u 1:2 w p ls 1 lw MLW notitle, \
     D index "sweep" u 1:3 w lp ls 2 notitle, \
     D index "sweep" u 1:($1==0.99 ? NaN : $2):(sprintf("%.0f", $4)) w labels offset 0.5,-0.6 left font ",".sprintf("%g",FA) tc rgb C_PINK notitle, \
     D index "sweep" u 1:($1==0.99 ? $2 : NaN):(sprintf("%.0f", $4)) w labels offset -0.5,0.5 right font ",".sprintf("%g",FA) tc rgb C_PINK notitle
unset multiplot
