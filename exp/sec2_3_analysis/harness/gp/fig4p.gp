# Figure 4 (Section 3.3), the per-query candidate requirement, one PDF per
# panel.  Same furniture as Figure 3; the panel letters are LaTeX subcaptions
# and the legend is a strip spanning both panels.
#
#   gnuplot fig4p.gp   ->  f4a.pdf f4b.pdf f4_legend.pdf

PW = 1.6384                      # 0.49\columnwidth, two across in a figure
load "house.gp"

# ---------------------------------------------------------------- (a) -------
# How many queries a fixed budget R serves completely.  The dotted rule is the
# 95% coverage the fixed budget in Section 6 is set for.
set output "f4a.pdf"
set logscale x; set xrange [9:760]
set xtics ("10" 10,"100" 100); set mxtics 10
set xlabel "budget {/:Italic R}" font FA offset 0,0.35 tc rgb INK
set yrange [0:1.045]; set ytics ("0" 0,"0.5" 0.5,"1" 1); set mytics 2
set ylabel "coverage" font FA offset 0.9,0 tc rgb INK
set arrow 1 from 9,0.95 to 760,0.95 nohead lc rgb "#9A9A9A" lw 1.4 dt 3 front
set label 1 "0.95" at 700,0.880 right font FN tc rgb "#707070" front
# curve on the integers the step function actually lives on; markers come
# from a sparse file so they sit evenly on the log axis
plot "p_cdf.dat"     u 1:2 w l ls 21 lw LW notitle, \
     ""              u 1:3 w l ls 24 lw LW notitle, \
     ""              u 1:4 w l ls 25 lw LW notitle, \
     "p_cdf_pts.dat" u 1:2 w p ls 21 lw MLW notitle, \
     ""              u 1:3 w p ls 24 lw MLW notitle, \
     ""              u 1:4 w p ls 25 lw MLW notitle
unset output

# ---------------------------------------------------------------- (b) -------
# What one fixed budget costs against an oracle that knew each query's own
# requirement, at matched coverage.  Both axes count candidates, not I/O.
set output "f4b.pdf"
load "house.gp"
set xrange [0.80:0.998]
set xtics ("0.80" 0.80,"0.90" 0.90,"0.99" 0.99); set mxtics 2
set xlabel "coverage target" font FA offset 0,0.35 tc rgb INK
set yrange [1:6.4]; set ytics 1 font FT; set mytics 2
set ylabel "fixed / oracle" font FA offset 0.9,0 tc rgb INK
# parameterised by the integer budget: each setting carries its own coverage
plot "p_overpay.dat"     u 2:3 w l ls 21 lw LW notitle, \
     ""                  u 4:5 w l ls 24 lw LW notitle, \
     ""                  u 6:7 w l ls 25 lw LW notitle, \
     "p_overpay_pts.dat" u 2:3 w p ls 21 lw MLW notitle, \
     ""                  u 4:5 w p ls 24 lw MLW notitle, \
     ""                  u 6:7 w p ls 25 lw MLW notitle
unset output

# ------------------------------------------------------------ legend --------
# One row across the full column, centred between the two panels.
LOUT = "f4_legend.pdf"
LW_PT = 3.3437*72; LH_PT = 14
LFS = 6.5; LN = 3
array LTXT[3] = ["Quora/Qwen", "MS MARCO/Qwen", "MS MARCO/BGE"]
array LSTY[3] = [21, 24, 25]
array LDRAW[3]
do for [i=1:3] {
    LDRAW[i] = sprintf('set arrow %d from LX0,LYM to LX1,LYM nohead ls %d', \
                       600+i, LSTY[i])
}
LSAMP = 13; LGAP = 5; LSEP = 12
load "legend.gp"
