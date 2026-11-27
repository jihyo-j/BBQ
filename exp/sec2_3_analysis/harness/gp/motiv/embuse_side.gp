# Side-by-side variant of the intro motivation figure, both panels on the same
# 2022-2025 span so the two x axes line up and the panels are the same width.
# Fonts and tick style follow recall_qps.gp.
#
# Restricting (b) to 2022-2025 is a data decision, not only a layout one: it
# drops 2021 (n=18) and 2026 (n=150), so the bars show 193 groups from 40
# papers rather than the full 361 from 71.  The caption numbers have to match
# whichever span ships.  embuse_side.gp.6yr keeps all six years.

if (!exists("NAVY")) NAVY = "#6A62F2"
if (!exists("MOD"))  MOD  = "#DB2E5C"
if (!exists("OTHR")) OTHR = "#9A9A9A"
if (!exists("CONV")) CONV = "#C9C9C9"
if (!exists("HATCH"))HATCH= "#A8A8A8"
if (!exists("HRULE"))HRULE= "#B4B4B4"
if (!exists("OUT"))  OUT  = "embuse_side.pdf"

INK = "#262626"
XLO = 2021.45
XHI = 2025.55

set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 \
    linewidth 1 size 3.33in,1.78in
set output OUT

set border 3 lw 1.0
set tics in nomirror scale 0.8,0.4 font ",7" tc rgb INK
unset x2tics; unset y2tics; unset grid
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0
set xrange [XLO:XHI]
set xtics ("2022" 2022, "2023" 2023, "2024" 2024, "2025" 2025) \
          font ",6.5" offset 0,0.4
unset mxtics

set multiplot

# ---------------------------------------------------------------- (a)
set lmargin at screen 0.145
set rmargin at screen 0.490
set tmargin at screen 0.845
set bmargin at screen 0.320
set title "(a) ST downloads" font ",7.5" offset 0,-0.5 tc rgb INK
set ylabel "Relative downloads" font ",7" offset 1.4,0 tc rgb INK
set yrange [0:15]
set ytics 0,5,15 font ",6.5"
unset mytics
unset key
plot "st_activity.dat" u 1:3 w lp lc rgb NAVY lw 1.9 pt 7 ps 0.45 notitle, \
     "st_activity.dat" u 1:($1 == 2022 ? $3 : NaN):("1×") \
         w labels left offset char -0.1,0.9 font ",6" tc rgb INK notitle, \
     "st_activity.dat" u 1:($1 == 2025 ? $3 : NaN):("13.3×") \
         w labels right offset char -0.6,0.5 font ",6" tc rgb INK notitle

# ---------------------------------------------------------------- (b)
# Four bars in this width leave a 17.5pt slab, so the share figures fit inside
# it again and the years can stay four digits.
set lmargin at screen 0.6444
set rmargin at screen 0.990
set title "(b) VLDB evaluation mix" font ",7" offset 0,-0.5 tc rgb INK
set ylabel "Dataset share (%)" font ",7" offset 1.4,0 tc rgb INK
set yrange [0:100]
set ytics 0,25,100 font ",6.5"
unset mytics
unset key
set boxwidth 0.80

set xtics ()
do for [i=1:6] {
    yr = system(sprintf("awk '!/^#/ && NF {c++; if (c==%d) print $1}' vldb_corpora.dat", i))
    ng = system(sprintf("awk '!/^#/ && NF {c++; if (c==%d) print $3}' vldb_corpora.dat", i))
    if (yr+0 >= XLO && yr+0 <= XHI) {
        set xtics add (sprintf("%s\n{/*0.85 n=%s}", yr, ng) yr+0)
    }
}
set xtics font ",6.5" offset 0,0.4 scale 0

set style fill pattern 5 border lc rgb HRULE
plot "vldb_pct.dat" u 1:5 w boxes lc rgb HATCH notitle
set style fill solid 1.0 noborder
plot "vldb_pct.dat" u 1:4 w boxes lc rgb CONV notitle, \
     "vldb_pct.dat" u 1:3 w boxes lc rgb OTHR notitle, \
     "vldb_pct.dat" u 1:2 w boxes lc rgb MOD  notitle, \
     "vldb_pct.dat" u 1:($2 >= 15 ? $2/2 : NaN):7 \
         w labels center font ",6" tc rgb "white" notitle, \
     "vldb_pct.dat" u 1:($2 > 0 && $2 < 15 ? $2 + 5.6 : NaN):7 \
         w labels center font ",6" tc rgb INK notitle

# ---------------------------------------------------------------- legend
unset border; unset tics; unset xlabel; unset ylabel; unset title
set lmargin at screen 0; set rmargin at screen 1
set bmargin at screen 0; set tmargin at screen 1
set xrange [0:1]; set yrange [0:1]
set style fill solid 1.0 noborder
set key horizontal maxcols 2 reverse Left center at screen 0.500,0.072 \
    samplen 1.3 spacing 1.1 font ",6.5" tc rgb INK
plot NaN w boxes fs solid 1.0 noborder lc rgb MOD  t "Modern", \
     NaN w boxes fs solid 1.0 noborder lc rgb CONV t "Conventional", \
     NaN w boxes fs solid 1.0 noborder lc rgb OTHR t "Other learned", \
     NaN w boxes fs pattern 5 border lc rgb HRULE lc rgb HATCH t "Unresolved"

unset multiplot
unset output
