# preprocessing_io — per-query time: query preprocessing (solid) + exact-FP I/O wait (hatched)
# data: preprocessing_io.dat  (make_preprocessing_io_dat.py <- qwen1024_five_method_recall099_timing_t1.csv)
# LINEAR y axis with one break (piecewise-linear "nonlinear" axis): bars stay continuous,
# only the y axes get a "//" break mark.  0..YB1 ms -> HLO pt,  YB1..YB2 -> GAPB pt,  YB2..YTOP -> rest
load "../../common/plot/common.gp"
HATCH = 5
# hand-drawn 45-degree hatch (gnuplot's pattern fill has a fixed, coarse spacing)
HSP = 3.0          # horizontal spacing between hatch strokes [pt]
HLW = 0.7          # stroke width (0.35pt)
HN = 0
HATCHPT(x0,y0,x1,y1) = sprintf('do for [hk=0:int(((%g)-(%g)+(%g)-(%g))/HSP)+1] { hc = (%g)-((%g)-(%g)) + hk*HSP; ht0 = ((%g)-hc > 0 ? (%g)-hc : 0); ht1 = ((%g)-(%g) < (%g)-hc ? (%g)-(%g) : (%g)-hc); if (ht1 > ht0) { HN = HN + 1; eval sprintf("set arrow %%d from screen sx(%%g),sy(%%g) to screen sx(%%g),sy(%%g) nohead lc rgb MBLUE lw HLW back", 1000+HN, hc+ht0, (%g)+ht0, hc+ht1, (%g)+ht1) } }', \
   x1, x0, y1, y0,  x0, y1, y0,  x0, x0,  y1, y0, x1, y1, y0, x1,  y0, y0)
NROWS = 1
XLABB = 25.5
# legend row now sits at the very top, above the panel titles (10.5pt band)
LEGBAND = 0.0   # legend now inside both panels (top-left)
FONT = "Carlito"          # same typeface as f1/f2 (this figure only)
PHX = PW2 / 1.509          # plot-box aspect of f2a/f2b
W = COLW; H = 0.5 + XLABB + PHX + TOPM + TTL + LEGBAND; OUT = "preprocessing_io"
eval SETTERM
PBB = 0.5 + XLABB

YB1 = 0.6;  YB2 = 1.0;  YTOP = 70.0   # 70 (was 64): headroom for the in-panel legend
HLO = 36.0*PHX/77.0; GAPB = 3.0; HUP = PHX - HLO - GAPB
K1 = HLO/YB1; KG = GAPB/(YB2-YB1); K2 = HUP/(YTOP-YB2)
fy(y) = y <= YB1 ? y*K1 : (y <= YB2 ? HLO + (y-YB1)*KG : HLO + GAPB + (y-YB2)*K2)
gy(p) = p <= HLO ? p/K1 : (p <= HLO+GAPB ? YB1 + (p-HLO)/KG : YB2 + (p-HLO-GAPB)/K2)

MBLUE = "#505050"    # monochrome (was C_BLUE #0A0AF0): solid prep, hatch and outlines in dark grey
array MC[5] = [MBLUE, MBLUE, MBLUE, MBLUE, MBLUE]   # one blue for every method; text stays dark grey
set style line 9  lc rgb MBLUE lw LW
set style line 10 lc rgb MBLUE lw LW
set style textbox opaque noborder margins 0.15,0.1

set multiplot
eval DRAWSCREEN

D = "preprocessing_io.dat"
BW = 0.62                 # bar width (prep-time labels may overhang a little; they sit on a white box)
PFS = 5.5                 # prep-time label font
set xrange [0.5:5.5]
mc(i) = int("0x".MC[int(i)][2:7])
set yrange [0:YTOP]
set nonlinear y via fy(y) inverse gy(y)
set ytics ("0" 0, "0.2" 0.2, "0.4" 0.4, "20" 20, "40" 40, "60" 60) nomirror   # ticks on the left y axis only
set mytics 1
set xtics ("ADC-BQ" 1, "TurboQuant" 2, "RaBitQ" 3, "DiskANN PQ" 4, "DiskANN OPQ" 5) rotate by 30 right offset 0.6,0.1 scale 0 nomirror

# break mark on a y axis at screen x: white patch over the axis line + two slanted strokes
YBRK = PBB + HLO + GAPB/2
BRK(n,x) = sprintf('set object %d rect from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) fs solid 1.0 noborder fc rgb "white" front; set arrow from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead lw BLW front; set arrow from screen sx(%g),sy(%g) to screen sx(%g),sy(%g) nohead lw BLW front', \
  n, x-0.9, YBRK-1.2, x+0.9, YBRK+1.2, x-2.0, YBRK-2.6, x+2.0, YBRK-0.2, x-2.0, YBRK+0.2, x+2.0, YBRK+2.6)

# prep-time label: horizontal, just above the solid (preprocessing) part of each bar
# solid prep 0..p, hatched I/O p..t, outline 0..t, total on top, prep time as boxed text at the bar foot
BARS(t) = sprintf('plot \
  D index "bars_%s" u 1:($2/2):($1-BW/2):($1+BW/2):(0):2:(mc($1)) w boxxyerror fs solid 1.0 noborder lc rgb variable notitle, \
  D index "bars_%s" u 1:($3/2):($1-BW/2):($1+BW/2):(0):3 w boxxyerror fs empty lw MLW*1.6 lc rgb "#000000" notitle, \
  D index "bars_%s" u 1:(gy(fy($2)+4.0)):6:(mc($1)) w labels boxed center font FONT.":Bold,%g" tc rgb "#000000" notitle, \
  D index "bars_%s" u 1:3:4:(mc($1)) w labels center offset 0,0.55 font FONT.":Bold,%g" tc rgb "#000000" notitle', t, t, t, PFS, t, FA)

do for [P in "a b"] {
  X0 = (P eq "a") ? L2 : L2+PW2+G2
  eval PANEL(X0, PBB, PW2, PHX)
  if (P eq "a") { set format y "%g"; eval YLABEL("Time per query (ms)"); unset xlabel; eval TITLE("MS MARCO") } \
  else          { set ytics ("" 0, "" 0.2, "" 0.4, "" 20, "" 40, "" 60) nomirror; unset ylabel; unset xlabel; eval TITLE("NQ") }
  set xlabel offset 0,-0.4
  eval BRK(301, X0); eval BRK(302, X0+PW2)
  LX = X0 + ((P eq "a") ? 2.0 : 5.0); LT = PBB + PHX - 3.0; load "preprocessing_io_inleg.gp"   # legend top-left in both panels (MS MARCO: pulled left to clear the 35.9 ms bar)
  # hatch over the I/O part (prep .. total) of each bar, in screen pt
  do for [i=1:5] {
    vp = real(system(sprintf("awk '/^# bars_%s/{f=1;next} /^# /{f=0} f && $1==%d {print $2}' %s", P, i, D)))
    vt = real(system(sprintf("awk '/^# bars_%s/{f=1;next} /^# /{f=0} f && $1==%d {print $3}' %s", P, i, D)))
    bx0 = X0 + (i - BW/2 - 0.5)/5.0*PW2; bx1 = X0 + (i + BW/2 - 0.5)/5.0*PW2
    eval HATCHPT(bx0, PBB + fy(vp), bx1, PBB + fy(vt))
  }
  eval BARS(P)
  unset object 301; unset object 302; unset object 120; unset object 121; unset label 130; unset label 131; unset arrow; HN = 0
}
unset multiplot
