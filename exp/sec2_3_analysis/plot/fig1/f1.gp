# f1 — Figure 1: (a) downloads | (b) dataset share by year, grouped bars, legend inside (b).
# Built from f1a.gp / f1b.gp (jihyo copies of sieun's fig1.gp).
#   gnuplot f1.gp                 ->  f1.pdf  (both panels on one canvas)
#   gnuplot -e 'PART="a"' f1.gp   ->  f1a.pdf (panel (a) alone, same size as in f1.pdf)
#   gnuplot -e 'PART="b"' f1.gp   ->  f1b.pdf (panel (b) alone, same size as in f1.pdf)
NAVY = "#0A0AF0"; UNRS = "#5A5A5A";   # dark grey: Modern (swapped with Unresolved, which is white)
 CONV = "#C9C9C9"; HRULE= "#B4B4B4"; INK = "#262626"
XLO = 2021.45; XHI = 2025.55
MODP = 4; OTHP = 2

W   = 3.3437                     # \columnwidth [in]
PWA = W/2; PWB = W/2            # equal canvases: f1a and f1b scale identically side by side
ML = 0.540; MR = 0.095; MB = 0.235; MT = 0.060
BH = 0.8158                      # plot-box height, as before (1.6384in panel, aspect 1.23)
PH = BH + MB + MT
H  = PH
YLXPT = 20.0                     # y-title centre, pt from the panel's left edge
XA = 0.0; XB = PWA               # panel origins [in]
if (!exists("PART")) PART = "ab"
if (PART eq "a") { W = PWA }
if (PART eq "b") { W = PWB; XB = 0.0 }
OUTF = (PART eq "ab") ? "f1" : "f1".PART

if (exists("PNG")) { eval sprintf('set terminal pngcairo enhanced font "Carlito,8" fontscale 3 linewidth 4 pointscale 4 size %d,%d', int(W*288), int(H*288)) } else { eval sprintf('set terminal pdfcairo enhanced font "Carlito,8" fontscale 0.75 linewidth 1 size %.4fin,%.4fin', W, H) }
set output (exists("PNG") ? OUTF."_preview.png" : OUTF.".pdf")

PANEL(x0,pw) = sprintf('set lmargin at screen %g; set rmargin at screen %g; set bmargin at screen %g; set tmargin at screen %g', (x0+ML)/W, (x0+pw-MR)/W, MB/H, (PH-MT)/H)
YTITLE(x0,s) = sprintf('set label 90 "%s" at screen %g, screen %g center rotate by 90 font ",7" tc rgb INK front', s, (x0+YLXPT/72.0)/W, (MB+BH/2)/H)

set multiplot
unset x2tics; unset y2tics; unset grid; unset key; unset ylabel

if (PART ne "b") {
# ---------------------------------------------------------------- (a) -------
eval PANEL(XA, PWA)
set border 3 lw 1.0
set tics in nomirror scale 0.6,0.3 font ",7" tc rgb INK
set xrange [XLO:XHI]; unset mxtics
set xtics ("2022" 2022,"2023" 2023,"2024" 2024,"2025" 2025) font ",6.5" offset 0,0.4
eval YTITLE(XA, "Annual downloads")
set yrange [0:3.3]
set ytics ("0" 0,"1B" 1,"2B" 2,"3B" 3) font ",6.5"
unset mytics
plot "motiv/st_activity.dat" u 1:($2/1e9) w lp lc rgb NAVY lw 1.9 pt 7 ps 0.45 notitle, \
     "motiv/st_activity.dat" u 1:($1 == 2022 ? $2/1e9 : NaN):("0.21B") \
         w labels center offset 0,0.75 font ",6" tc rgb INK notitle, \
     "motiv/st_activity.dat" u 1:($1 == 2025 ? $2/1e9 : NaN):("2.75B") \
         w labels center offset 0,0.75 font ",6" tc rgb INK notitle
unset label 90

}
if (PART ne "a") {
# ---------------------------------------------------------------- (b) -------
# per year, four bars side by side: Modern | Conventional | Other learned | Unresolved
# shares from the cumulative columns of vldb_pct.dat: m=$2, o=$3-$2, c=$4-$3, u=$5-$4
eval PANEL(XB, PWB)
set xrange [2021.55:2025.45]; unset mxtics   # tighter than (a): more room for the bars
set border 3 lw 1.0
set tics in nomirror scale 0.6,0.3 font ",7" tc rgb INK
set yrange [0:112]
set ytics 0,25,100 font ",6.5"
unset mytics
set xtics ("2022" 2022,"2023" 2023,"2024" 2024,"2025" 2025)
set xtics font ",6.5" offset 0,0.4 scale 0
eval YTITLE(XB, "Dataset share (%)")
BWID = 0.21; DX = 0.21    # bars touch within a year; 0.16 gap between years
# legend: one column of four, top right of the panel (graph coords)
LFS = 5.5
LSW = 0.06; LSH = 0.07          # swatch size (graph fractions)
array LTXT[4] = ["Modern", "Conventional", "Other learned", "Unresolved"]
array LCX[4]  = [0.52, 0.52, 0.52, 0.52]
array LCY[4]  = [0.95, 0.85, 0.75, 0.65]
set object 701 rect from graph LCX[1],LCY[1]-LSH/2 to graph LCX[1]+LSW,LCY[1]+LSH/2 fc rgb UNRS fs solid 1.0 border lc rgb INK lw 0.5 front
set object 702 rect from graph LCX[2],LCY[2]-LSH/2 to graph LCX[2]+LSW,LCY[2]+LSH/2 fc rgb CONV fs solid 1.0 border lc rgb INK lw 0.5 front
set object 703 rect from graph LCX[3],LCY[3]-LSH/2 to graph LCX[3]+LSW,LCY[3]+LSH/2 fc rgb INK fs pattern OTHP border lc rgb INK lw 0.5 front
set object 704 rect from graph LCX[4],LCY[4]-LSH/2 to graph LCX[4]+LSW,LCY[4]+LSH/2 fc rgb "white" fs solid 1.0 border lc rgb INK lw 0.5 front
do for [i=1:4] { eval sprintf('set label %d "%s" at graph %g, graph %g left font ",%g" tc rgb INK front', 710+i, LTXT[i], LCX[i]+LSW+0.03, LCY[i], LFS) }
D = "motiv/vldb_pct.dat"
plot D u ($1-1.5*DX):2:(BWID)       w boxes fs solid 1.0 border lc rgb INK lw 0.5 lc rgb UNRS notitle, \
     D u ($1-0.5*DX):($4-$3):(BWID) w boxes fs solid 1.0 border lc rgb INK lw 0.5 lc rgb CONV notitle, \
     D u ($1+0.5*DX):($3-$2):(BWID) w boxes fs pattern OTHP border lc rgb INK lw 0.5 lc rgb INK notitle, \
     D u ($1+1.5*DX):($5-$4):(BWID) w boxes fs solid 1.0 border lc rgb INK lw 0.5 lc rgb "white" notitle
unset label 90
}
unset multiplot
unset output
