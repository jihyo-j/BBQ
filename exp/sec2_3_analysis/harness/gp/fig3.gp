# Figure 3 (Section 3), one PDF per panel.
#
# The panel letter and its caption are no longer drawn here: they are LaTeX
# subcaptions, so the panels carry axes and data only.  The legend is its own
# strip, sized to span panels (a)-(c), which are the three it describes --
# panel (d) plots quantizers, not configurations, and keys itself.
#
#   gnuplot fig3.gp    ->  f3a.pdf f3b.pdf f3c.pdf f3d.pdf f3_legend.pdf

PW  = 1.7227                     # 0.245\textwidth, four across in a figure*
AR  = 1.12                       # taller box: a figure* has the room
load "house.gp"

# Panels (c) and (d) plot quantizers, not datasets.  House rule for this
# figure: an OPEN marker is a dataset (ls 21-28) and a FILLED one a quantizer,
# so the two families stay apart even where the colours repeat.
# Shape is the only thing that departs from the dataset styles above: the same
# colours, the same lw, and the same ps as the open marker of the same shape in
# ls 21-28, so the two families are one family seen filled and unfilled.
set style line 31 lc rgb "#909090" lw LW pt 3  ps 0.72*PSC   # Sym-BQ   star
set style line 32 lc rgb "#E8336D" lw LW pt 5  ps 0.72*PSC   # ADC-BQ   sq
set style line 33 lc rgb "#9933E6" lw LW pt 11 ps 0.80*PSC   # TQ-1bit  v-tri
set style line 34 lc rgb "#0A0AF0" lw LW pt 9  ps 0.80*PSC   # RaBitQ   ^-tri
set style line 35 lc rgb "#1A9850" lw LW pt 13 ps 0.86*PSC   # PQ       diamond
set style line 36 lc rgb "#262626" lw LW pt 7  ps 0.72*PSC   # OPQ      circle

# ---------------------------------------------------------------- (a) -------
# Exact score profile: how far the top ranks stand above a query's own corpus
# scores.  Steep here means a code's error has to be large to reorder.
set output "f3a.pdf"
set logscale x; set xrange [0.9:135]
set xtics ("1" 1,"10" 10,"100" 100); set mxtics 10
set xlabel "Exact rank ({/:Italic r})" font FA offset 0,0.35 tc rgb INK
set yrange [1.8:11.4]; set ytics 2; set mytics 2
set ylabel "{/:Italic z}({/:Italic r})" font FA offset 1.6,0 tc rgb INK
KEEP = "2 3 5 6 8 9"            # Quora, NQ, MS MARCO, MSM/BGE, GloVe, GIST
plot for [c in KEEP] "p_z6.dat"     u 1:(column(c+0)) w l ls (c+19) notitle, \
     for [c in KEEP] "p_z6_pts.dat" u 1:(column(c+0)) w p ls (c+19) lw MLW notitle
unset output

# ---------------------------------------------------------------- (b) -------
# One checkpoint truncated with MRL, every quantity relative to its value at
# d = 32, ground truth recomputed at each width.  Widening improves both terms
# of eta, so the budget falls.
set output "f3b.pdf"
load "house.gp"
set logscale x; set xrange [26:5200]
set xtics ("32" 32,"256" 256,"4096" 4096); set mxtics 2
set xlabel "Dimension ({/:Italic d})" font FA offset 0,0.35 tc rgb INK
set logscale y; set yrange [0.00007:16]
set ytics ("10" 10,"1" 1,"10^{-2}" 0.01,"10^{-4}" 0.0001) font FT
unset mytics
set ylabel "rel. to {/:Italic d}=32" font FA offset 1.25,0 tc rgb INK
set key at graph 0.03,0.40 left top vertical maxrows 3 reverse Left \
    samplen 1.6 spacing 1.05 width 0 font ",5.5" tc rgb INK
plot "p_dimrel.dat" u 1:5 w lp lc rgb "#1A9850" lw LW pt 10 ps 0.80*PSC t "β", \
     ""             u 1:2 w lp lc rgb "#707070" lw LW dt 2 pt 2 ps 0.72*PSC t "{/:Italic f}", \
     ""             u 1:3 w lp lc rgb "#E8336D" lw LW pt 4  ps 0.72*PSC t "{/:Italic c_Q}", \
     ""             u 1:6 w lp lc rgb "#9933E6" lw LW pt 12 ps 0.86*PSC t "η", \
     ""             u 1:4 w lp lc rgb "#0A0AF0" lw LW pt 8  ps 0.80*PSC t "{/:Italic R}^{*}"
unset output

# ---------------------------------------------------------------- (c) -------
# The budget past the k=10 convention, for the same six codes panel (d) uses.
# The x axis is broken: k = 1..10 linearly, which is the range the convention
# lives in and where the codes lie on top of each other, then 10..100
# logarithmically, where they separate.  One shared log y axis.
#
# The statistic is the MEAN budget, not the p95 the tables report: a p95 over
# queries is fixed by its top 5% and jumps from one k to the next, which is
# unreadable once six codes share one axis, and the mean is also what an
# average read count costs.  Columns of kcurve_msmarco_qw06.dat
# (src/kcurve.py, 1500 queries): 1 k, then (median, p95, mean) per code in the
# order Sym-BQ ADC-BQ TQ-1bit RaBitQ PQ OPQ, so the means are 4+3i.
KF = "../out/kcurve_msmarco_qw06.dat"
set output "f3c.pdf"
load "house.gp"
unset key
unset arrow 1; unset object 1
X0 = ML/PW; X1 = (PW-MR)/PW; WB = X1-X0
SPLIT = 0.56; GAP = 0.055
set multiplot
set logscale y; set yrange [1.15:1900]
set ytics ("10" 10,"10^{2}" 100,"10^{3}" 1000) font FT; unset mytics
#   left strip: the convention's own range, linear in k
set lmargin at screen X0
set rmargin at screen X0 + (SPLIT-GAP/2)*WB
unset logscale x; set xrange [0.4:10.6]
set xtics ("1" 1,"4" 4,"7" 7,"10" 10) font FT; set mxtics 3
set xlabel ""
set ylabel "mean {/:Italic R}^{*}" font FA offset 1.4,0 tc rgb INK
set label 20 "{/*1.15 \342\211\210}" at screen X0+SPLIT*WB, screen 0.247 \
    center font FT tc rgb INK front
set label 22 "{/:Italic k}" at screen (X0+X1)/2, screen 0.072 \
    center font FA tc rgb INK front
#  markers are staggered across the codes -- every third k, starting one k
#  later for each -- because six curves inside 1.5x of each other otherwise
#  stack their markers on one another and read as a single band
plot for [i=0:5] KF u 1:(column(4+3*i)) w l ls (31+i) notitle, \
     for [i=0:5] KF u 1:(column(4+3*i)) every 3::(i%3) w p ls (31+i) lw MLW notitle
unset label 20; unset label 22
#   right strip: past the convention, logarithmic
set lmargin at screen X0 + (SPLIT+GAP/2)*WB
set rmargin at screen X1
set logscale x; set xrange [10:112]
set xtics ("20" 20,"50" 50,"100" 100) font FT; unset mxtics
set ylabel ""
set ytics ("" 10,"" 100,"" 1000) font FT
plot for [i=0:5] KF u 1:(column(4+3*i)) w l ls (31+i) notitle, \
     for [i=0:5] KF u 1:(column(4+3*i)) every 7::(9+i) w p ls (31+i) lw MLW notitle
unset multiplot
unset output

# ---------------------------------------------------------------- (d) -------
# The same MRL ladder as (c), read as the distortion each quantizer actually
# delivers: one Qwen3-8B checkpoint truncated and renormalised, ground truth
# recomputed at every width, every code sized to that width.  It separates the
# two things (c) folds together -- widening helps, and it helps every code,
# but the codes keep their order and their spacing all the way up.
#
# Columns of p_dimperq.dat (src/export_dimperq.py, from
#   out/dimperq_msmarco_qw8.json):
#   1 d  2 f  3 c_Q(isotropic)  4 Sym-BQ  5 ADC-BQ  6 TQ-1bit  7 ADC+c
#   8 ADC+cs  9 RaBitQ  10 PQ  11 OPQ  12 SQ-4bit  13 SQ-8bit
set output "f3d.pdf"
load "house.gp"
do for [i=101:105] { unset label i }     # (c)'s inline series labels
unset key
set logscale x 2; set xrange [26:5200]
set xtics ("32" 32,"256" 256,"4096" 4096) font FT; set mxtics 2
set xlabel "Dimension ({/:Italic d})" font FA offset 0,0.35 tc rgb INK
set logscale y; set yrange [0.115:1.45]
set ytics ("0.2" 0.2,"0.5" 0.5,"1" 1) font FT; unset mytics
set ylabel "Score error {/:Italic c_{Q}}" font FA offset 1.4,0 tc rgb INK
plot for [c in "4 5 6 9 10 11"] "p_dimperq.dat" u 1:(column(c+0)) \
         w l  ls (c==4?31:c==5?32:c==6?33:c==9?34:c==10?35:36) lw LW notitle, \
     for [c in "4 5 6 9 10 11"] "p_dimperq.dat" u 1:(column(c+0)) \
         w p  ls (c==4?31:c==5?32:c==6?33:c==9?34:c==10?35:36) lw MLW notitle
unset output

# ------------------------------------------------------------ legend --------
# One key per panel, because the three families they name are different kinds
# of thing and the figure should not invite reading a dataset marker as a
# quantizer.  The rule the markers follow: OPEN = dataset, FILLED = quantizer.
# Panels (c) and (d) draw the same six codes, so they share one key, and a
# shared key says what it covers.

LW_D = 3.4930                    # a key over the pair it describes
LW_Q = 3.4930                    # a key centred under two
LH   = 0.260                    # two rows, no title: the strips sit above
                                # the panels now, where a title is redundant

#   (a): the datasets
eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', LW_D, LH)
set output "f3_legend.pdf"
unset border; unset tics; unset xlabel; unset ylabel
unset logscale x; unset logscale y; unset arrow 1; unset object 1
set lmargin at screen 0; set rmargin at screen 1
set bmargin at screen 0; set tmargin at screen 1
set xrange [0:1]; set yrange [0:1]
set key at graph 0.5,0.98 center top horizontal maxcols 3 reverse Left \
    samplen 1.8 spacing 1.10 width 2.6 font ",6" tc rgb INK
plot NaN w lp ls 21 lw MLW t "Quora/Qwen",   NaN w lp ls 22 lw MLW t "NQ/Qwen", \
     NaN w lp ls 24 lw MLW t "MS MARCO/Qwen", NaN w lp ls 25 lw MLW t "MS MARCO/BGE", \
     NaN w lp ls 27 lw MLW t "GloVe",         NaN w lp ls 28 lw MLW t "GIST"
unset output

#   (c) and (d): the same six codes, so one key, centred under the pair
eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', LW_Q, LH)
set output "f3q_legend.pdf"
set lmargin at screen 0; set rmargin at screen 1
set bmargin at screen 0; set tmargin at screen 1
set key at graph 0.5,0.98 center top horizontal maxcols 3 reverse Left \
    samplen 1.8 spacing 1.10 width 2.6 font ",6" tc rgb INK
plot NaN w lp ls 31 lw MLW t "Sym-BQ",  NaN w lp ls 34 lw MLW t "RaBitQ", \
     NaN w lp ls 32 lw MLW t "ADC-BQ",  NaN w lp ls 35 lw MLW t "PQ", \
     NaN w lp ls 33 lw MLW t "TQ-1bit", NaN w lp ls 36 lw MLW t "OPQ"
unset output
