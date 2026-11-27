# Panel (b) of the intro motivation figure, drawn standalone so panel (a) can
# be decided separately and dropped into a multiplot later.  Terminal, fonts
# and tick style follow embuse.gp.
#
# What this version changes over the embuse.gp copy:
#   * the group count n sits under each year, so the 2024 spike and the 2025
#     dip are read against their denominators (6/20 and 23/116), not as trend;
#   * 2026 carries a partial marker -- it is not a finished year;
#   * the slabs run bottom to top in legend order (modern, other learned,
#     conventional, unresolved).  They used to run modern, conventional, other
#     learned, which disagreed with the legend and left the "how learned"
#     ordering non-monotone, so no single-hue ramp could carry it;
#   * tick labels come out of vldb_corpora.dat and the bars out of the
#     vldb_pct.dat it generates, so the two cannot drift apart.
#
# The palette is overridable so variants can be rendered without editing this
# file:  gnuplot -e 'MOD="#1c5cab"; OTHR="#3987e5"; OUT="v1.pdf"' vldb_mix.gp
# Provenance and sampling method belong in the LaTeX caption, not here.

# One accent on neutral greys, the arrangement CHANGES_v16 describes: only the
# modern share is the point, so only it gets a hue and the rest recede.  The
# accent is the paper's own #E8336D darkened one step, which is what lifts the
# white in-slab type from 4.1:1 to 4.8:1 against it.  Greys are the paper's
# neutral steps, not the blue-greys the first draft used.
if (!exists("MOD"))   MOD   = "#DA2460"  # Transformer / modern API
if (!exists("OTHR"))  OTHR  = "#9A9A9A"  # other learned / neural
if (!exists("CONV"))  CONV  = "#C9C9C9"  # conventional / raw
if (!exists("HATCH")) HATCH = "#A8A8A8"  # unresolved provenance: hatch on white
if (!exists("HRULE")) HRULE = "#B4B4B4"  # its box outline
if (!exists("OUT"))   OUT   = "vldb_mix.pdf"

INK = "#262626"
PARTIAL = "2026"    # year still in progress at the data cut, marked with *

set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 \
    linewidth 1 size 3.33in,1.80in
set output OUT

set border 3 lw 1.0
set tics in nomirror scale 0.8,0.4 font ",7" tc rgb INK
unset x2tics; unset y2tics; unset grid
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0

set multiplot

set lmargin at screen 0.150
set rmargin at screen 0.980
set bmargin at screen 0.339
set tmargin at screen 0.944
set ylabel "Evaluation groups (%)" font ",7.5" offset 2.6,0 tc rgb INK
set xrange [2020.45:2026.55]
set yrange [0:100]
set ytics 0,25,100 font ",7"
unset mytics
unset key
set boxwidth 0.64

# year over group count, both lifted out of the counts file
set xtics ()
do for [i=1:6] {
    yr = system(sprintf("awk '!/^#/ && NF {c++; if (c==%d) print $1}' vldb_corpora.dat", i))
    ng = system(sprintf("awk '!/^#/ && NF {c++; if (c==%d) print $3}' vldb_corpora.dat", i))
    set xtics add (sprintf("%s%s\n{/*0.86 n=%s}", yr, (yr eq PARTIAL ? "*" : ""), ng) yr+0)
}
set xtics font ",7" offset 0,0.35 scale 0
unset mxtics

# drawn top down, each slab painting over the one beneath it.  The modern share
# is labelled white inside its own slab where the slab is tall enough for the
# type, and in ink just above it otherwise (2023 only).
set style fill pattern 5 border lc rgb HRULE
plot "vldb_pct.dat" u 1:5 w boxes lc rgb HATCH notitle
set style fill solid 1.0 noborder
plot "vldb_pct.dat" u 1:4 w boxes lc rgb CONV notitle, \
     "vldb_pct.dat" u 1:3 w boxes lc rgb OTHR notitle, \
     "vldb_pct.dat" u 1:2 w boxes lc rgb MOD  notitle, \
     "vldb_pct.dat" u 1:($2 >= 15 ? $2/2 : NaN):7 \
         w labels center font ",6.5" tc rgb "white" notitle, \
     "vldb_pct.dat" u 1:($2 > 0 && $2 < 15 ? $2 + 5.6 : NaN):7 \
         w labels center font ",6.5" tc rgb INK notitle

# =====================================================================
# legend, two rows beneath the panel, in slab order
# =====================================================================
unset border; unset tics; unset xlabel; unset ylabel; unset title
set lmargin at screen 0; set rmargin at screen 1
set bmargin at screen 0; set tmargin at screen 1
set xrange [0:1]; set yrange [0:1]
set style fill solid 1.0 noborder
set key horizontal maxcols 2 center at screen 0.545,0.083 samplen 1.1 \
    spacing 1.05 width -1 font ",6.5" tc rgb INK
plot NaN w boxes fs solid 1.0 noborder lc rgb MOD  t "Transformer / modern API", \
     NaN w boxes fs solid 1.0 noborder lc rgb OTHR t "Other learned / neural", \
     NaN w boxes fs solid 1.0 noborder lc rgb CONV t "Conventional / raw", \
     NaN w boxes fs pattern 5 border lc rgb HRULE lc rgb HATCH \
         t "Unresolved provenance"

unset multiplot
unset output
