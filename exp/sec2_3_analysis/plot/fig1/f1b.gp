# Figure 1 (Section 1), one PDF per panel.
#
# This figure deliberately does NOT follow house.gp: the original
# motiv/embuse_side.gp design is kept as it was -- thin left/bottom border,
# unmirrored short ticks, its own type sizes, the grey ramp with a hatched
# "unresolved" slab and the navy download line.  The only changes are
# structural: the two panels are separate PDFs so they can be LaTeX
# subfigures, the in-figure titles are gone (they are subcaptions now), and
# the key is its own strip shared by the figure.
#
# Panel (b) covers 2022-2025: 193 evaluation groups from 40 papers.  The
# caption has to carry those two numbers, not the six-year 361/71.
#
#   gnuplot f1b.gp    ->  f1b.pdf   (jihyo copy of fig1.gp panel (b), monochrome)

NAVY = "#0A0AF0"    # the main figure's blue
MOD  = "#E8336D"    # the paper's pink, as in Figures 3 and 5
OTHR = "#9933E6"    # purple, same palette
CONV = "#C9C9C9"
HATCH= "#A8A8A8"
HRULE= "#B4B4B4"
INK  = "#262626"
XLO = 2021.45
XHI = 2025.55

# 0.49\columnwidth printed; the box keeps the original 1.149 x 0.934in shape,
# narrowed only by the right margin a standalone panel needs for its last
# tick label, which the shared canvas used to absorb in the gutter.
PW = 1.6384
ML = 0.540; MR = 0.095; MB = 0.380; MT = 0.060
BW = PW - ML - MR
BH = BW / 1.23
PH = BH + MB + MT
if (exists("PNG")) { eval sprintf('set terminal pngcairo enhanced font "Carlito,8" fontscale 3 linewidth 4 pointscale 4 size %d,%d', int(PW*288), int(PH*288)) } else { eval sprintf('set terminal pdfcairo enhanced font "Carlito,8" fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH) }

set border 3 lw 1.0
set tics in nomirror scale 0.8,0.4 font ",7" tc rgb INK
unset x2tics; unset y2tics; unset grid; unset key
set lmargin at screen ML/PW
set rmargin at screen (PW-MR)/PW
set bmargin at screen MB/PH
set tmargin at screen (PH-MT)/PH
set xrange [XLO:XHI]
unset mxtics

# ---------------------------------------------------------------- (a) -------
# ---------------------------------------------------------------- (b) -------
# jihyo copy: monochrome -- Modern and Other learned are black hatch patterns
# (each drawn over a white base so the grey slab underneath does not show through)
MODP = 4; OTHP = 2; PINK_INK = "#000000"
set output "f1b.pdf"
# y-axis title at the same distance from the axis as in f1a (centre 24.38pt from the
# left edge, measured from f1a's own ylabel placement)
YLXPT = 20.0   # y-title centre, pt from the left edge (same in f1a)
unset ylabel
set label 90 "Dataset share (%)" at screen (YLXPT/72.0)/PW, screen (MB+BH/2)/PH center rotate by 90 font ",7" tc rgb INK front
set yrange [0:100]
set ytics 0,25,100 font ",6.5"
unset mytics
set boxwidth 0.80
set xtics ()
do for [i=1:6] {
    yr = system(sprintf("awk '!/^#/ && NF {c++; if (c==%d) print $1}' motiv/vldb_corpora.dat", i))
    ng = system(sprintf("awk '!/^#/ && NF {c++; if (c==%d) print $3}' motiv/vldb_corpora.dat", i))
    if (yr+0 >= XLO && yr+0 <= XHI) {
        set xtics add (sprintf("%s\n{/*0.85 n=%s}", yr, ng) yr+0)
    }
}
set xtics font ",6.5" offset 0,0.4 scale 0
set style textbox opaque fc rgb "white" noborder margins 0.6,0.1
# axes, tick labels and the y label are drawn once (in the last pass): drawing
# them in every pass stacked the anti-aliased text and made it look bold
save set "axes_f1b.tmp"
set multiplot
unset border; set format x ""; set format y ""; set tics scale 0; unset ylabel; unset label 90
set xtics ("" 2022); set ytics ("" 0)
set style fill pattern MODP border lc rgb HRULE   # Unresolved: black diagonals (pattern swapped with Modern)
plot "motiv/vldb_pct.dat" u 1:5 w boxes lc rgb INK notitle
set style fill solid 1.0 noborder
plot "motiv/vldb_pct.dat" u 1:4 w boxes lc rgb CONV notitle, \
     ""                   u 1:3 w boxes lc rgb "white" notitle
set style fill pattern OTHP border lc rgb HRULE
plot "motiv/vldb_pct.dat" u 1:3 w boxes lc rgb INK notitle
set style fill solid 1.0 noborder
plot "motiv/vldb_pct.dat" u 1:2 w boxes lc rgb "white" notitle
set style fill pattern MODP border lc rgb HRULE
load "axes_f1b.tmp"; set style fill empty border lc rgb HRULE   # Modern: no pattern
plot "motiv/vldb_pct.dat" u 1:2 w boxes lc rgb HRULE notitle, \
     ""                   u 1:($2 > 0 ? $2/2 : NaN):7 \
         w labels center boxed font ",6" tc rgb INK notitle
unset multiplot
unset output
system("rm -f axes_f1b.tmp")

# ------------------------------------------------------------ legend --------
LOUT = "f1_legend.pdf"
LW_PT = 3.3437*72; LH_PT = 13
LFS = 6.5; LN = 4
array LTXT[4] = ["Modern", "Conventional", "Other learned", "Unresolved"]
array LSTY[4] = [0, 0, 0, 0]
array LDRAW[4]
LDRAW[1] = sprintf('set object 601 rect from LX0,LYLO to LX1,LYHI fc rgb "white" fs empty border lc rgb "%s" front', HRULE)
LDRAW[2] = sprintf('set object 602 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs solid 1.0 noborder front', CONV)
LDRAW[3] = sprintf('set object 603 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs pattern %d border lc rgb "%s" front', INK, OTHP, HRULE)
LDRAW[4] = sprintf('set object 604 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs pattern %d border lc rgb "%s" front', INK, MODP, HRULE)
LSAMP = 11; LGAP = 4; LSEP = 10
load "legend.gp"
