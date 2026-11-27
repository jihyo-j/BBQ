# Stacked version of the intro motivation figure -- NOT the one the paper
# ships; see embuse_side.gp.  Also off-span: this one draws all six years of
# vldb_corpora.dat, while the paper's figure and its caption and body text now
# cover 2022-2025 only.  Kept for reference; re-check the caption counts
# (193 groups / 40 papers for 2022-2025, 361 / 71 for all six) before reusing.
#
# Intro motivation figure: how much embedding activity there is, against what
# ANN evaluations actually run on.  Terminal, fonts and tick style follow
# recall_qps.gp; the border is lighter than that template because two panels
# are stacked in one column.  Provenance, normalisation and the meaning of the
# excluded 2026 row belong in the LaTeX caption, not here.
#
# Palette is overridable so variants render without editing this file:
#   gnuplot -e 'NAVY="#6A62F2"; MOD="#DB2E5C"; OUT="alt.pdf"' embuse.gp
# The previous panel (a), a per-model API request ranking, is kept as
# embuse_apiusage.gp.old.

# (a) and (b)'s accent are the two ends of the figure's own swatch.  The pink
# is #FF6082 darkened one step: at #FF6082 itself the white in-slab type sits
# at 2.90:1, below the 4.5:1 a 6.5pt label needs, and #DB2E5C clears it at
# 4.61:1 for the same colour.
if (!exists("NAVY")) NAVY = "#6A62F2"  # (a) the single series
if (!exists("MOD"))  MOD  = "#DB2E5C"  # (b) Transformer / modern API
if (!exists("OTHR")) OTHR = "#9A9A9A"  # (b) other learned / neural
if (!exists("CONV")) CONV = "#C9C9C9"  # (b) conventional / raw
if (!exists("HATCH"))HATCH= "#A8A8A8"  # (b) unresolved provenance: hatch on white
if (!exists("HRULE"))HRULE= "#B4B4B4"  # (b) its box outline
if (!exists("OUT"))  OUT  = "embuse.pdf"

INK = "#262626"
# (b) 2026 is a part-year at the data cut.  That is said in the caption rather
# than marked on the axis.

set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 \
    linewidth 1 size 3.33in,3.00in
set output OUT

set border 3 lw 1.0
set tics in nomirror scale 0.8,0.4 font ",7" tc rgb INK
unset x2tics; unset y2tics; unset grid
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0

set multiplot

# =====================================================================
# (a)  Download activity index -- one series, straight segments between
#      the observed years, no interpolation, first and last points labelled
# =====================================================================
set lmargin at screen 0.200
set rmargin at screen 0.975
set tmargin at screen 0.940
set bmargin at screen 0.662
set title "(a) Sentence Transformers activity" font ",8" offset 0,-0.6 tc rgb INK
unset xlabel
set ylabel "Relative downloads" font ",7.5" offset 1.0,0 tc rgb INK
# ALIGNX=1 widens (a) to (b)'s year span so the same year sits at the same x in
# both panels.  It adds no data points -- 2021 and 2026 stay absent, not zero --
# it only stops the reader from aligning 2023 in (a) with 2024 in (b).  Off by
# default, since the figure spec fixes (a)'s ticks at 2022-2025.
if (!exists("ALIGNX")) ALIGNX = 0
set xrange [ALIGNX ? 2020.45 : 2021.78 : ALIGNX ? 2026.55 : 2025.62]
set xtics ("2022" 2022, "2023" 2023, "2024" 2024, "2025" 2025) \
          font ",7" offset 0,0.35
unset mxtics
set yrange [0:15]
set ytics 0,5,15 font ",7"
unset mytics
unset key
plot "st_activity.dat" u 1:3 w lp lc rgb NAVY lw 2.2 pt 7 ps 0.55 notitle, \
     "st_activity.dat" u 1:($1 == 2022 ? $3 : NaN):("1×") \
         w labels center offset char 0,0.9 font ",6.5" tc rgb INK notitle, \
     "st_activity.dat" u 1:($1 == 2025 ? $3 : NaN):("13.3×") \
         w labels left offset char 0.7,0 font ",6.5" tc rgb INK notitle

# =====================================================================
# (b)  VLDB evaluation mix -- 100 per cent stacked bars by year.
#      Slabs run bottom to top in legend order (modern, other learned,
#      conventional, unresolved); n under each year keeps the 2024 spike and
#      the 2025 dip readable against their denominators.  2026 is partial.
#      (a) covers 2022-2025 and (b) 2021-2026: each panel shows the period it
#      actually has, and the years (a) lacks are left out rather than zeroed.
# =====================================================================
set lmargin at screen 0.200
set rmargin at screen 0.975
set tmargin at screen 0.505
set bmargin at screen 0.218
set title "(b) VLDB evaluation mix" font ",8" offset 0,-0.6 tc rgb INK
set xlabel ""
set ylabel "Dataset share (%)" font ",7.5" offset 1.0,0 tc rgb INK
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
    set xtics add (sprintf("%s\n{/*0.86 n=%s}", yr, ng) yr+0)
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
# legend for (b) only, two rows beneath it
# =====================================================================
unset border; unset tics; unset xlabel; unset ylabel; unset title
set lmargin at screen 0; set rmargin at screen 1
set bmargin at screen 0; set tmargin at screen 1
set xrange [0:1]; set yrange [0:1]
set style fill solid 1.0 noborder
# reverse puts the colour swatch before the name and Left butts the name up
# against it -- without Left, gnuplot right-justifies the text and the swatch
# drifts away from the name it belongs to.  maxcols 2 over four entries makes
# the 2x2 block.  Names are short here and defined in full in the caption.
set key horizontal maxcols 2 reverse Left center at screen 0.588,0.068 \
    samplen 1.4 spacing 1.1 font ",6.5" tc rgb INK
plot NaN w boxes fs solid 1.0 noborder lc rgb MOD  t "Modern", \
     NaN w boxes fs solid 1.0 noborder lc rgb CONV t "Conventional", \
     NaN w boxes fs solid 1.0 noborder lc rgb OTHR t "Other learned", \
     NaN w boxes fs pattern 5 border lc rgb HRULE lc rgb HATCH \
         t "Unresolved"

unset multiplot
unset output
