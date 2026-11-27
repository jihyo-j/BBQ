# Figure 2 (Section 2.1), the two motivation panels, one PDF each.
#
# (a) is redrawn from the arithmetic rather than clipped out of the old
# capacity figure: at d = 4096 a vector costs 4d bytes in FP32, d/2 at 4 bits
# and d/8 at 1 bit, and an HNSW graph with M = 24 adds 208 B to each, so the
# three lines and the three crossings of the 1007 GiB budget (65M, 479M, 1.5B)
# are exact, not fitted.  Series are labelled on the lines; the panel needs no
# key, which also keeps the legend from looking as if it covered panel (b).
#
#   gnuplot fig2.gp    ->  f2a.pdf f2b.pdf

PW = 1.6384                      # 0.49\columnwidth, two across in a figure
load "house.gp"

GIB   = 1073741824.0
BUD   = 1007*GIB                 # the evaluation server's DRAM
GRAPH = 208.0                    # HNSW M = 24, bytes per vector
D     = 4096
sz(n,b) = n*(b+GRAPH)/GIB        # index size in GiB

C_FP = "#262626"; C_4B = "#0A0AF0"; C_1B = "#E8336D"

# ---------------------------------------------------------------- (a) -------
set output "f2a.pdf"
set logscale x; set xrange [8e6:1.3e10]
set xtics ("10^{7}" 1e7,"10^{8}" 1e8,"10^{9}" 1e9,"10^{10}" 1e10) font FT
set mxtics 10
set xlabel "number of vectors" font FA offset 0,0.35 tc rgb INK
set logscale y; set yrange [1:3e5]
set ytics ("10^{0}" 1,"10^{2}" 100,"10^{4}" 10000) font FT; set mytics 10
set ylabel "index (GiB)" font FA offset 0.6,0 tc rgb INK
set style textbox 2 opaque fc rgb "white" noborder

# everything above the budget line has to be served from somewhere else
set object 1 rect from graph 0, first 1007 to graph 1, graph 1 \
    fc rgb "#E8336D" fs solid 0.07 noborder behind
set arrow 1 from 8e6,1007 to 1.3e10,1007 nohead lw 1.6 dt 2 lc rgb INK front
set label 1 "1007 GiB" at 1.22e10, 420 right font ",7.5" tc rgb INK front boxed bs 2

# where each representation runs out of DRAM
set arrow 2 from 6.517e7,45 to 6.517e7,1200 nohead lw 1.4 dt 3 lc rgb C_FP
set arrow 3 from 4.792e8,45 to 4.792e8,1200 nohead lw 1.4 dt 3 lc rgb C_4B
set arrow 4 from 1.502e9,45 to 1.502e9,1200 nohead lw 1.4 dt 3 lc rgb C_1B
set label 2 "65M"  at 5.9e7,1250 left rotate by 90 font ",7.5" tc rgb C_FP front boxed bs 2
set label 3 "479M" at 4.35e8,1250 left rotate by 90 font ",7.5" tc rgb C_4B front boxed bs 2
set label 4 "1.5B" at 1.36e9,1250 left rotate by 90 font ",7.5" tc rgb C_1B front boxed bs 2

plot sz(x,4*D) w l lw LW lc rgb C_FP notitle, \
     sz(x,D/2) w l lw LW lc rgb C_4B notitle, \
     sz(x,D/8) w l lw LW lc rgb C_1B notitle
unset output

# ---------------------------------------------------------------- (b) -------
# One checkpoint, one corpus, MRL-truncated and renormalised: the only thing
# moving is d.  The flat reference is what a sign code would cost on an
# isotropic vector of the same width, so the gap is the embedding's doing.
set output "f2b.pdf"
load "house.gp"
GREY = "#6E6E6E"
unset logscale y; unset object 1
do for [i=1:7] { unset label i }
do for [i=1:4] { unset arrow i }
set logscale x 2; set xrange [26:6000]
set xtics ("32" 32,"128" 128,"512" 512,"4096" 4096) font FT; set mxtics 2
set xlabel "dimension {/:Italic d}" font FA offset 0,0.35 tc rgb INK
set yrange [0:1.05]; set ytics 0,0.2,0.8 font FT; set mytics 2
set ylabel "score error {/:Italic c_{Q}}" font FA offset 1.4,0 tc rgb INK
set label 1 "isotropic sign code" at 300,0.955 center font ",7.5" tc rgb GREY front
set label 2 "measured" at 2900,0.215 right font ",7.5" tc rgb C_1B front
set arrow 1 from 4096,0.8166 to 4096,0.2998 heads size 1.0,20 lw 0.9 lc rgb INK front
set label 3 "2.7×" at 3500,0.56 right font ",7.5" tc rgb INK front
plot "motiv/mrl_scoreerr.dat" u 1:3 w l  lw 2.0 dt 2 lc rgb GREY notitle, \
     ""                       u 1:2 w lp lw LW pt 7 ps 0.5 lc rgb C_1B notitle
unset output

# ------------------------------------------------------------ legend --------
# Centred under both panels: the three colours are the paper's own and (b)
# uses the same pink for the 1-bit code, so a shared strip is not misleading.
set terminal pdfcairo enhanced font "Liberation Sans,8" fontscale 0.75 \
    linewidth 1 size 3.3437in,0.190in
set output "f2_legend.pdf"
unset border; unset tics; unset xlabel; unset ylabel; unset object 1
do for [i=1:7] { unset label i }
do for [i=1:4] { unset arrow i }
unset logscale x; unset logscale y
set lmargin at screen 0; set rmargin at screen 1
set bmargin at screen 0; set tmargin at screen 1
set xrange [0:1]; set yrange [0:1]
set key at graph 0.5,1.05 center top horizontal maxrows 1 reverse Left \
    samplen 2.2 spacing 1.1 width 0 font ",7.5" tc rgb INK
plot NaN w l lw LW lc rgb C_FP t "FP32", NaN w l lw LW lc rgb C_4B t "4-bit", \
     NaN w l lw LW lc rgb C_1B t "1-bit"
unset output
