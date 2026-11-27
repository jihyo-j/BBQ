# Panel (b) of the motivation figure: what widening the embedding does to the
# sign code's score error.  Same corpus and checkpoint throughout -- MS MARCO
# under Qwen3-Embedding-8B, truncated with MRL and renormalised -- so the only
# thing moving is d.  The flat reference is what a sign code would cost on an
# isotropic vector of the same width: dimension by itself buys nothing, and
# everything the curve gains is the embedding's doing.  Section 3.2 takes that
# apart into kappa and beta; here it is only the motivation.
#
# Sized and fonted to sit beside the d=4096 capacity panel at half a column.
# Canvas, plot-box fractions and type sizes are matched to panel (a)
# (capacity4096_crop.pdf, cropped tight to its ink at 126.7 x 111.5 pt, frame at
# l .174 r .952 b .242 t .929, ticks 6 pt and axis labels 6.5 pt) so that at
# equal printed widths the type matches.  The left margin is the one departure:
# (a)'s y labels are 10^k and narrow, (b)'s are 0.2-0.8 and are not, so (b) needs
# 0.235 to keep the rotated label clear of them.
# equal printed widths the two panels carry the same plot box and the same type.
set terminal pdfcairo enhanced font "Liberation Sans,6.5" fontscale 0.75 \
    linewidth 1 size 1.7597in,1.5481in
set output "mrl_scoreerr.pdf"

BQ   = "#E8336D"    # the 1-bit code, the same accent panel (a) gives it
GREY = "#9A9A9A"
INK  = "#262626"

set border 3 lw 1.2
set tics in nomirror scale 1.0,0.5 font ",6" tc rgb INK
unset x2tics; unset y2tics; unset grid
unset key
set title "MS MARCO / Qwen3-8B" font ",6.5" offset 0,-0.75 tc rgb INK

set logscale x 2
set xrange [26:6000]
set xtics ("32" 32, "128" 128, "512" 512, "4096" 4096) font ",6" offset 0,0.35
set mxtics 2
# baseline matched to panel (a)'s "Number of vectors" (y 98.6..105.9 of 111.5)
set xlabel "Embedding dimension {/:Italic d}" font ",6.5" offset 0,0.10 tc rgb INK
set yrange [0:0.92]
set ytics 0,0.2,0.8 font ",6"
set mytics 2
# the rotated label is placed by hand: gnuplot anchors it to the left margin,
# and (b)'s 0.2-0.8 tick labels are wide enough that any offset that clears
# them pushes the label off the canvas.
unset ylabel
set label 99 "Score error {/:Italic c_Q}" at screen 0.052, graph 0.5 \
    center rotate by 90 font ",6.5" tc rgb INK

set lmargin at screen 0.235
set rmargin at screen 0.952
set bmargin at screen 0.242
set tmargin at screen 0.929

# the isotropic reference, and how far the measured code runs below it
set label 1 "isotropic sign code" at 300, 0.845 center font ",5.5" tc rgb GREY front
set label 2 "measured" at 2900, 0.225 right font ",5.5" tc rgb BQ front
set arrow 1 from 4096, 0.8166 to 4096, 0.2998 heads size 1.0,20 \
    lw 0.9 lc rgb INK front
set label 3 "2.7×" at 3600, 0.56 right font ",5.5" tc rgb INK front

plot "mrl_scoreerr.dat" u 1:3 w l lw 2.0 dt 2 lc rgb GREY notitle, \
     "mrl_scoreerr.dat" u 1:2 w lp lw 2.4 pt 7 ps 0.5 lc rgb BQ notitle
unset output
