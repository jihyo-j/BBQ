# recall_ablation — (left) Recall@10 vs QPS ablation, (right) per-query outcomes of calibrated BBQ
# column width, square panels. MS MARCO Qwen 1,024.  left data: recall_ablation.dat; right: recall_groups.gp (make_recall_groups.py) ; Fixed/Calibrated from equal_pool_qd_fixed75_vs_calibrated.csv
#   (efSearch 256, calibration target 0.99, 5,980 test queries)
# previous version (QPS + target-vs-achieved): recall_ablation_v7.gp; target-vs-achieved moved to calibration.gp
load "../../common/plot/common.gp"
# style matched to f2: Carlito, ticks 6.5pt / labels 7pt; labelled ticks only, no mirrored ticks;
# legend on top, one group centred over each panel; page cropped to the content
FONT = "Carlito"; FT = 6.5; FL = 7.0
TICMAJ = 0.6                             # shorter ticks (common.gp default 1.0)
LM = 24.0; GM = 30.0; RM = 21.0
PW = (COLW - LM - GM - RM) / 2.0
PH = PW                                  # square panels
PBOT = 18.0
LEGB = 2.5                               # legend now inside the left panel
W = COLW; H = PBOT + PH + 2.0 + LEGB; OUT = "recall_ablation"
eval SETTERM

set style line 1 lc rgb C_ORANGE lw LW pt 6  ps PS        # Sym-BQ
set style line 2 lc rgb C_GREEN  lw LW pt 4  ps PS        # ADC-BQ
set style line 3 lc rgb C_BLUE   lw LW pt 8  ps PS*1.1    # BBQ + Fixed R@75
set style line 4 lc rgb C_PINK   lw LW pt 12 ps PS*1.25   # BBQ + Calibrated R
set style line 5 lc rgb C_BLUE lw MLW                     # bars: FP reads / query
set style line 6 lc rgb C_BLACK  lw LW pt 7  ps PS*1.1    # latency

set multiplot
# legend: the four left-panel methods in one row across the top (widths measured from Carlito 6.5pt);
# FP reads / Latency are keyed inside the right panel
# in-panel legend entry (pt, screen): 9pt sample + marker, 2pt gap, 6pt text
ISMP = 9.0; IGST = 2.0; IFS = 6.0; IROW = 6.8
IENT(n, x, y, ls, txt) = sprintf('set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead ls %d front; set label %d "" at screen sx(%g),sy(%g) point ls %d lw MLW front; set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, x, y, x+ISMP, y, ls, n+1, x+ISMP/2, y, ls, n+2, txt, x+ISMP+IGST, y, IFS)
ITXT(n, x, y, txt) = sprintf('set label %d "%s" at screen sx(%g),sy(%g) left font ",%g" front', n, txt, x+ISMP+IGST, y, IFS)
eval DRAWSCREEN
set tics in nomirror scale TICMAJ,0 font ",".sprintf("%g",FT)

D = "recall_ablation.dat"
# (left) recall axis: x' = -log10(1-x) with the empty 0.83-0.97 range cut out ("//" on the x axes)
XA = 0.68; XB1 = 0.83; XB2 = 0.97; XZ = 0.996
XLW = 0.20; XGW = 0.05; XRW = 1.0 - XLW - XGW
ul(x) = -log10(1-x)
KXL = XLW/(ul(XB1)-ul(XA)); KXG = XGW/(ul(XB2)-ul(XB1)); KXR = XRW/(ul(XZ)-ul(XB2))
fx(x) = x <= XB1 ? (ul(x)-ul(XA))*KXL : (x <= XB2 ? XLW + (ul(x)-ul(XB1))*KXG : XLW + XGW + (ul(x)-ul(XB2))*KXR)
iu(u) = 1-10**(-u)
gx(p) = p <= XLW ? iu(ul(XA) + p/KXL) : (p <= XLW+XGW ? iu(ul(XB1) + (p-XLW)/KXG) : iu(ul(XB2) + (p-XLW-XGW)/KXR))
set xrange [XA:XZ]
set nonlinear x via fx(x) inverse gx(x)
set xtics ("0.7" 0.7, "0.8" 0.8, "0.98" 0.98, "0.99" 0.99) offset 0,0.45; unset mxtics

XBRK = LM + PW*(XLW + XGW/2)
OLD_BRKX(n,x,y) = sprintf('set object %d rect from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) fs solid 1.0 noborder fc rgb "white" front; set arrow from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead lw BLW front; set arrow from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead lw BLW front', \
  n, x-1.2, y-0.9, x+1.2, y+0.9, x-2.6, y-2.0, x-0.2, y+2.0, x+0.2, y-2.0, x+2.6, y+2.0)

# wavy cut: white band between two vertical sine waves, across the whole panel height (+ overhang)
WA = 0.9; WL = 7.0; WD = 3.2; WOV = 2.5; WN = 40
wx(x0,y) = x0 + WA*sin(2*pi*(y-(PBOT-WOV))/WL)

eval PANEL(LM, PBOT, PW, PH)
set logscale y
set yrange [150:6000]   # 150 (was 250): room for the in-panel legend
set ytics ("500" 500, "1,000" 1000, "2,000" 2000, "5,000" 5000) nomirror offset 0.3,0; unset mytics
set ylabel "QPS" font ",7" offset 1.2,0
set xlabel "Recall\\@10" font ",7" offset 0,1.0
# axis cut: white gap through the whole panel; each edge is a small vertical wave (rotated "≈")
# where it crosses the bottom/top axis, joined by a straight vertical line in between
BH = 6.0; BA = 0.8; BN = 12; BD = 2.6         # wave height, amplitude, segments, gap width [pt]
bx(x0,t) = x0 + BA*sin(2*pi*t)
set object 71 rect from screen sx(XBRK-BD/2),sy(PBOT-BH/2) to screen sx(XBRK+BD/2),sy(PBOT+PH+BH/2) fs solid 1.0 noborder fc rgb "white" front
do for [s=-1:1:2] {
  eval sprintf("set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead lw BLW*0.8 front", \
    190+s, XBRK+s*BD/2, PBOT+BH/2, XBRK+s*BD/2, PBOT+PH-BH/2)
  do for [j=0:1] { YC = PBOT + j*PH
    do for [k=0:BN-1] { ta = 1.0*k/BN; tb = 1.0*(k+1)/BN
      eval sprintf("set arrow %d from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead lw BLW*0.8 front", \
        200+j*50+(s+1)*12+k, bx(XBRK+s*BD/2,ta), YC-BH/2+ta*BH, bx(XBRK+s*BD/2,tb), YC-BH/2+tb*BH) } } }
# legend: right of the axis cut, bottom; long entries at the bottom where the curves are highest
LX0 = XBRK + BD/2 + 3.0
eval IENT(100, LX0, PBOT+3.0+3.5*IROW, 1, "Sym-BQ"); eval IENT(103, LX0, PBOT+3.0+2.5*IROW, 2, "ADC-BQ")
eval IENT(106, LX0, PBOT+3.0+1.5*IROW, 3, "BBQ + Fixed R\\\\@75"); eval IENT(109, LX0, PBOT+3.0+0.5*IROW, 4, "BBQ + Calibrated R")
plot for [i=1:4] D index "m".i u 2:3 w l ls i notitle, \
     for [i=1:4] D index "m".i u 2:3 w p ls i lw MLW notitle
unset object 71; unset object 72; unset arrow; do for [i=100:111] { unset label i }

# (right) per-query outcomes: bars = FP reads / query (left y), line = latency in ms (right y),
#         number inside each bar = share of test queries (state in the caption)
load "recall_groups.gp"    # $C, written by make_recall_groups.py
eval PANEL(LM+PW+GM, PBOT, PW, PH)
unset nonlinear x; unset logscale y
set boxwidth 0.62
set style fill solid 1.0 border lc rgb C_BLUE
set xrange [-0.6:2.6]
unset xtics; unset mxtics; set xtics scale 0 nomirror offset 0,0.45
set yrange [0:200]; set ytics 0,50,200 nomirror offset 0.3,0; unset mytics
set y2range [0:1.5]; set y2tics 0,0.5,1.5 nomirror offset -0.3,0 scale TICMAJ; unset my2tics
set ylabel "FP reads / query" font ",7" offset 1.0,0
set y2label "Latency (ms)" font ",7" offset -2.3,0
set xlabel "Per-query Recall\\@10" font ",7" offset 0,1.0
# in-panel key, top left (above the bars and the 1.0 / 0.9 latency labels)
set object 81 rect from graph 0.04, graph 0.905 to graph 0.12, graph 0.945 fc ls 5 fs solid 1.0 border lc ls 5 lw MLW front
set label 82 "FP reads / query" at graph 0.15, graph 0.925 left font ",6.5" front
set arrow 83 from graph 0.03, graph 0.845 to graph 0.13, graph 0.845 nohead ls 6 front
set label 84 "" at graph 0.08, graph 0.845 point ls 6 lw MLW front
set label 85 "Latency" at graph 0.15, graph 0.845 left font ",6.5" front
plot $C u 1:5:xtic(2) w boxes ls 5 notitle, \
     $C u 1:5:(sprintf("%.1f",$5)) w labels offset 0,0.5 font ",".sprintf("%g",FA) notitle, \
     $C u 1:(14):(sprintf("%.1f%%",$4)) w labels tc rgb "white" font ",4.8" notitle, \
     $C u 1:6 axes x1y2 w l ls 6 notitle, \
     $C u 1:6 axes x1y2 w p ls 6 lw MLW notitle, \
     $C u 1:6:(sprintf("%.3f",$6)) axes x1y2 w labels offset 0,0.75 font ",".sprintf("%g",FA) notitle
unset multiplot
