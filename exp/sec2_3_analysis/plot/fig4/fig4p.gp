# Figure 4 (Section 3.3), the per-query candidate requirement, one PDF per
# panel.  Same furniture as Figure 3; the panel letters are LaTeX subcaptions
# and the legend is a strip spanning both panels.
#
#   gnuplot fig4p.gp   ->  f4a.pdf f4b.pdf f4_legend.pdf

PW = 1.6384                      # 0.49\columnwidth, two across in a figure
AR = 1.509                       # jihyo copy: plot-box aspect of f2a/f2b
load "house.gp"

# ---------------------------------------------------------------- (a) -------
# How many queries a fixed budget R serves completely.  The dotted rule is the
# 95% coverage the fixed budget in Section 6 is set for.
set output (exists("PNG") ? "f4a_p.png" : "f4a.pdf")
set logscale x; set xrange [9:760]
set xtics ("10" 10,"100" 100) offset 0,0.4; unset mxtics
set xlabel "budget {/:Italic R}" font FA offset 0,1.15 tc rgb INK
set yrange [0:1.045]; set ytics ("0" 0,"0.5" 0.5,"1" 1); unset mytics
set ylabel "coverage" font FA offset 1.7,0 tc rgb INK
# jihyo copy: legend inside the panel
array E_LS[3]; array E_TX[3]; array E_W[3]
E_LS[1]=21; E_TX[1]="Quora/Qwen";    E_W[1]=34.78
E_LS[2]=24; E_TX[2]="MS MARCO/Qwen"; E_W[2]=48.81
E_LS[3]=25; E_TX[3]="MS MARCO/BGE";  E_W[3]=43.95
LNC=1; LNR=3; IFS=6.0; ISMP=9.0; IGAP=1.8; ICOL=5.0; IROW=7.0
IX0 = BW*72 - 4.0 - (ISMP + IGAP + 48.81*IFS/6.5); IY0 = 4.0   # bottom-right, clear of the x tick marks
load "inleg.gp"
# curve on the integers the step function actually lives on; markers come
# from a sparse file so they sit evenly on the log axis
plot "p_cdf.dat"     u 1:2 w l ls 21 lw LW notitle, \
     ""              u 1:3 w l ls 24 lw LW notitle, \
     ""              u 1:4 w l ls 25 lw LW notitle, \
     "p_cdf_pts.dat" u 1:2 w p ls 21 lw MLW notitle, \
     ""              u 1:3 w p ls 24 lw MLW notitle, \
     ""              u 1:4 w p ls 25 lw MLW notitle
unset output
unset arrow; do for [i=801:860] { unset label i }

# ---------------------------------------------------------------- (b) -------
# What one fixed budget costs against an oracle that knew each query's own
# requirement, at matched coverage.  Both axes count candidates, not I/O.
set output (exists("PNG") ? "f4b_p.png" : "f4b.pdf")
do for [i=41:43] { unset arrow i; unset label i }
do for [i=51:53] { unset label i }
load "house.gp"
set xrange [0.79:0.999]   # a little room so the curves and markers clear the frame
set xtics ("0.80" 0.80,"0.90" 0.90,"0.99" 0.99) offset 0,0.4; unset mxtics
set xlabel "coverage target" font FA offset 0,1.15 tc rgb INK
set yrange [1:6.8]; set ytics 1,1,6 font FT; unset mytics
set ylabel "fixed / oracle" font FA offset 1.7,0 tc rgb INK
# jihyo copy: legend inside the panel
array E_LS[3]; array E_TX[3]; array E_W[3]
E_LS[1]=21; E_TX[1]="Quora/Qwen";    E_W[1]=34.78
E_LS[2]=24; E_TX[2]="MS MARCO/Qwen"; E_W[2]=48.81
E_LS[3]=25; E_TX[3]="MS MARCO/BGE";  E_W[3]=43.95
LNC=1; LNR=3; IFS=6.0; ISMP=9.0; IGAP=1.8; ICOL=5.0; IROW=7.0
IX0 = 4.0; IY0 = BH*72 - 3.0 - LNR*IROW   # top-left
load "inleg.gp"
# parameterised by the integer budget: each setting carries its own coverage
plot "p_overpay_smooth.dat" u 2:3 w l ls 21 lw LW notitle, \
     ""                  u 4:5 w l ls 24 lw LW notitle, \
     ""                  u 6:7 w l ls 25 lw LW notitle, \
     "p_overpay_pts.dat" u 2:3 w p ls 21 lw MLW notitle, \
     ""                  u 4:5 w p ls 24 lw MLW notitle, \
     ""                  u 6:7 w p ls 25 lw MLW notitle
unset output
unset arrow; do for [i=801:860] { unset label i }

# ------------------------------------------------------------ legend --------
# One row across the full column, centred between the two panels.
LOUT = "f4_legend.pdf"
LW_PT = 3*(13+5) + 34.78+48.81+43.95 + 2*12 + 1.5; LH_PT = 8.5   # cropped to the row itself
LFS = 6.5; LN = 3
array LTXT[3] = ["Quora/Qwen", "MS MARCO/Qwen", "MS MARCO/BGE"]
array LSTY[3] = [21, 24, 25]
array LDRAW[3]
do for [i=1:3] {
    LDRAW[i] = sprintf('set arrow %d from LX0,LYM to LX1,LYM nohead ls %d', \
                       600+i, LSTY[i])
}
LSAMP = 13; LGAP = 5; LSEP = 12; LKW = 0.47   # Carlito average glyph width
load "legend.gp"
