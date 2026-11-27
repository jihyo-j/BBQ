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
#   gnuplot fig1.gp    ->  f1a.pdf f1b.pdf f1_legend.pdf

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
ML = 0.470; MR = 0.095; MB = 0.380; MT = 0.060
BW = PW - ML - MR
BH = BW / 1.23
PH = BH + MB + MT
eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH)

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
set ylabel "Annual downloads" font ",7" offset 1.4,0 tc rgb INK
set yrange [0:3.3]
set ytics ("0" 0,"1B" 1,"2B" 2,"3B" 3) font ",6.5"
unset mytics
plot "motiv/st_activity.dat" u 1:($2/1e9) w lp lc rgb NAVY lw 1.9 pt 7 ps 0.45 notitle, \
     "motiv/st_activity.dat" u 1:($1 == 2022 ? $2/1e9 : NaN):("0.21B") \
         w labels left offset char -0.1,0.9 font ",6" tc rgb INK notitle, \
     "motiv/st_activity.dat" u 1:($1 == 2025 ? $2/1e9 : NaN):("2.75B") \
         w labels right offset char -0.5,0.6 font ",6" tc rgb INK notitle
unset output

# ---------------------------------------------------------------- (b) -------
set output "f1b.pdf"
set ylabel "Dataset share (%)" font ",7" offset 1.4,0 tc rgb INK
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
# two passes on one page: the hatched slab first, the solid ones over it
set multiplot
set style fill pattern 5 border lc rgb HRULE
plot "motiv/vldb_pct.dat" u 1:5 w boxes lc rgb HATCH notitle
set style fill solid 1.0 noborder
plot "motiv/vldb_pct.dat" u 1:4 w boxes lc rgb CONV notitle, \
     ""                   u 1:3 w boxes lc rgb OTHR notitle, \
     ""                   u 1:2 w boxes lc rgb MOD  notitle, \
     ""                   u 1:($2 >= 15 ? $2/2 : NaN):7 \
         w labels center font ",6" tc rgb "white" notitle, \
     ""                   u 1:($2 > 0 && $2 < 15 ? $2 + 5.6 : NaN):7 \
         w labels center font ",6" tc rgb "white" notitle
#    a share too thin to hold its own label is written just above the slab,
#    which puts it on the "other learned" band -- so it is white like the rest
unset multiplot
unset output

# ------------------------------------------------------------ legend --------
# One row of four, centred on the figure, laid out by hand so every label
# keeps the same distance from its own swatch and from the next one.
LOUT = "f1_legend.pdf"
LW_PT = 3.3437*72; LH_PT = 13
LFS = 6.5; LN = 4
array LTXT[4] = ["Modern", "Conventional", "Other learned", "Unresolved"]
array LSTY[4] = [0, 0, 0, 0]
array LDRAW[4]
LDRAW[1] = sprintf('set object 601 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs solid 1.0 noborder front', MOD)
LDRAW[2] = sprintf('set object 602 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs solid 1.0 noborder front', CONV)
LDRAW[3] = sprintf('set object 603 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs solid 1.0 noborder front', OTHR)
LDRAW[4] = sprintf('set object 604 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs pattern 5 border lc rgb "%s" front', HATCH, HRULE)
LSAMP = 11; LGAP = 4; LSEP = 10
load "legend.gp"
