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
#   gnuplot f1a.gp    ->  f1a.pdf   (jihyo copy of fig1.gp panel (a); value labels centred on their points)

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
if (exists("PNG")) { eval sprintf('set terminal pngcairo enhanced font "Carlito,8" fontscale 3 linewidth 4 size %d,%d', int(PW*288), int(PH*288)) } else { eval sprintf('set terminal pdfcairo enhanced font "Carlito,8" fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH) }

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
set output "f1a.pdf"
set xtics ("2022" 2022,"2023" 2023,"2024" 2024,"2025" 2025) font ",6.5" offset 0,0.4
# the counts as published, not as a multiple of 2022
YLXPT = 20.0   # y-title centre, pt from the left edge (same in f1b)
unset ylabel
set label 90 "Annual downloads" at screen (YLXPT/72.0)/PW, screen (MB+BH/2)/PH center rotate by 90 font ",7" tc rgb INK front
set yrange [0:3.3]
set ytics ("0" 0,"1B" 1,"2B" 2,"3B" 3) font ",6.5"
unset mytics
plot "motiv/st_activity.dat" u 1:($2/1e9) w lp lc rgb NAVY lw 1.9 pt 7 ps 0.45 notitle, \
     "motiv/st_activity.dat" u 1:($1 == 2022 ? $2/1e9 : NaN):("0.21B") \
         w labels center offset 0,0.75 font ",6" tc rgb INK notitle, \
     "motiv/st_activity.dat" u 1:($1 == 2025 ? $2/1e9 : NaN):("2.75B") \
         w labels center offset 0,0.75 font ",6" tc rgb INK notitle
unset output
