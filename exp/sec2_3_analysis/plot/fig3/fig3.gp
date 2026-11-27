# Figure 3 (Section 3), one PDF per panel.
#
# The panel letter and its caption are no longer drawn here: they are LaTeX
# subcaptions, so the panels carry axes and data only.  The legend is its own
# strip, sized to span panels (a)-(c), which are the three it describes --
# panel (d) plots quantizers, not configurations, and keys itself.
#
#   gnuplot fig3.gp    ->  f3a.pdf f3b.pdf f3c.pdf f3d.pdf f3_legend.pdf

PW  = 1.7227                     # 0.245\textwidth, four across in a figure*
AR  = 1.509                      # jihyo copy: plot-box aspect of f2a/f2b
LW  = 1.2                        # jihyo copy: thinner curves (0.6pt; house default 1.8 = 0.9pt)
load "house.gp"

# panel (d) plots quantizers, not configurations, so it gets its own styles
set style line 31 lc rgb "#909090" lw LW pt 15 ps 0.70*PSC   # Sym-BQ   filled pentagon (a star looked like GloVe's cross)
set style line 32 lc rgb "#E8336D" lw LW pt 5  ps 0.62*PSC   # ADC-BQ   filled square
set style line 33 lc rgb "#9933E6" lw LW pt 11 ps 0.70*PSC   # TQ-1bit  filled v-tri
set style line 34 lc rgb "#0A0AF0" lw LW pt 9  ps 0.70*PSC   # RaBitQ   filled ^-tri
set style line 35 lc rgb "#1A9850" lw LW pt 13 ps 0.72*PSC   # PQ       filled diamond
set style line 36 lc rgb "#262626" lw LW pt 7  ps 0.62*PSC   # OPQ      filled circle

# panel (c) quantities (open markers, own styles so the legend can reuse them)
set style line 41 lc rgb "#1A9850" lw LW pt 10 ps 0.80*PSC          # beta
set style line 42 lc rgb "#707070" lw LW pt 2  ps 0.72*PSC dt 2     # f
set style line 43 lc rgb "#E8336D" lw LW pt 4  ps 0.72*PSC          # c_Q
set style line 44 lc rgb "#9933E6" lw LW pt 12 ps 0.86*PSC          # eta
set style line 45 lc rgb "#0A0AF0" lw LW pt 8  ps 0.80*PSC          # R*

# ---------------------------------------------------------------- (a) -------
# Exact score profile: how far the top ranks stand above a query's own corpus
# scores.  Steep here means a code's error has to be large to reorder.
set output "f3a.pdf"
set logscale x; set xrange [0.9:135]
set xtics ("1" 1,"10" 10,"100" 100); unset mxtics
set xlabel "Exact rank ({/:Italic r})" font FA offset 0,1.3 tc rgb INK
set yrange [1.8:11.4]; set ytics 2; unset mytics
set ylabel "{/:Italic z}({/:Italic r})" font FA offset 1.6,0 tc rgb INK
KEEP = "2 3 5 6 8 9"            # Quora, NQ, MS MARCO, MSM/BGE, GloVe, GIST
plot for [c in KEEP] "p_z6.dat"     u 1:(column(c+0)) w l ls (c+19) notitle, \
     for [c in KEEP] "p_z6_pts.dat" u 1:(column(c+0)) w p ls (c+19) lw MLW notitle
unset output

# ---------------------------------------------------------------- (b) -------
# The budget those profiles buy.  Two strips on a broken axis: the Transformer
# configurations need tens of candidates, the two non-learned controls
# thousands, and plotting both on one scale flattens the part that matters.
set output "f3b.pdf"
Y0 = MB/PH; Y1 = (PH-MT)/PH; HB = Y1-Y0
set multiplot

#   upper strip: GloVe and GIST, no x axis of its own
set bmargin at screen Y0+0.82*HB
set tmargin at screen Y1
unset logscale x; set xrange [0.3:10.7]; unset mxtics
unset xtics
set xlabel ""
set logscale y; set yrange [400:9e6]
set ytics ("10^{3}" 1000,"10^{6}" 1000000) font FT nomirror scale 0.6,0; unset mytics
set ylabel ""
set label 30 "{/*1.15 \342\211\210}" at graph -0.012, graph -0.20 center font FT tc rgb INK front
set label 31 "{/*1.15 \342\211\210}" at graph  1.012, graph -0.20 center font FT tc rgb INK front
plot for [i=8:9] "p_rk6.dat" u 1:i w lp ls (i+19) lw MLW notitle

#   lower strip: the Transformer configurations, carrying both axis labels
unset label 30; unset label 31
set bmargin at screen Y0
set tmargin at screen Y0+0.72*HB
set xtics ("1" 1,"4" 4,"7" 7,"10" 10) nomirror scale 0.6,0 offset 0,0.4; unset mxtics
set xlabel "{/:Italic k}" font FA offset 0,1.3 tc rgb INK
unset logscale y; set yrange [0:64]
set ytics 20 font FT; unset mytics
set ylabel "{/:Italic R}^{*}" font FA offset 1.6,1.0 tc rgb INK
plot for [c in "2 3 5 6"] "p_rk6.dat" u 1:(column(c+0)) w lp ls (c+19) lw MLW notitle
unset multiplot
unset output

# ---------------------------------------------------------------- (c) -------
# One checkpoint truncated with MRL, every quantity relative to its value at
# d = 32, ground truth recomputed at each width.  Widening improves both terms
# of eta, so the budget falls.
set output "f3c.pdf"
load "house.gp"
set logscale x; set xrange [26:5200]
set xtics ("32" 32,"256" 256,"4096" 4096); unset mxtics
set xlabel "Dimension ({/:Italic d})" font FA offset 0,1.3 tc rgb INK
set logscale y; set yrange [0.00002:16]
set ytics ("10" 10,"1" 1,"10^{-2}" 0.01,"10^{-4}" 0.0001) font FT
unset mytics
set ylabel "rel. to {/:Italic d}=32" font FA offset 2.1,0 tc rgb INK
# jihyo copy: legend inside the panel (bottom-left)
array E_LS[5]; array E_TX[5]; array E_W[5]
E_LS[1]=41; E_TX[1]="β";                  E_W[1]=3.45
E_LS[2]=42; E_TX[2]="{/:Italic f}";       E_W[2]=2.5
E_LS[3]=43; E_TX[3]="{/:Italic c_Q}";     E_W[3]=7.5
E_LS[4]=44; E_TX[4]="η";                  E_W[4]=3.49
E_LS[5]=45; E_TX[5]="{/:Italic R}^{*}";   E_W[5]=7.0
LNC=3; LNR=2; IX0=4.0; IY0=2.5; IFS=6.0; ISMP=9.0; IGAP=1.8; ICOL=5.0; IROW=7.0
IFLIP=1   # bottom row: beta, c_Q, R*; top row: f, eta
load "inleg.gp"
IFLIP=0
plot "p_dimrel.dat" u 1:5 w lp ls 41 notitle, \
     ""             u 1:2 w lp ls 42 notitle, \
     ""             u 1:3 w lp ls 43 notitle, \
     ""             u 1:6 w lp ls 44 notitle, \
     ""             u 1:4 w lp ls 45 notitle
unset output
unset arrow; do for [i=801:860] { unset label i }

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
set logscale x 2; set xrange [26:5200]
set xtics ("32" 32,"256" 256,"4096" 4096) font FT; unset mxtics
set xlabel "Dimension ({/:Italic d})" font FA offset 0,1.3 tc rgb INK
set logscale y; set yrange [0.075:1.45]
set ytics ("0.2" 0.2,"0.5" 0.5,"1" 1) font FT; unset mytics
set ylabel "Score error {/:Italic c_{Q}}" font FA offset 1.4,0 tc rgb INK
# jihyo copy: legend inside the panel (bottom-left)
array E_LS[6]; array E_TX[6]; array E_W[6]
E_LS[1]=31; E_TX[1]="Sym-BQ";  E_W[1]=21.02
E_LS[2]=32; E_TX[2]="ADC-BQ";  E_W[2]=21.12
E_LS[3]=33; E_TX[3]="TQ-1bit"; E_W[3]=19.91
E_LS[4]=34; E_TX[4]="RaBitQ";  E_W[4]=18.22
E_LS[5]=35; E_TX[5]="PQ";      E_W[5]=7.73
E_LS[6]=36; E_TX[6]="OPQ";     E_W[6]=12.04
LNC=2; LNR=3; IX0=4.0; IY0=2.5; IFS=6.0; ISMP=9.0; IGAP=1.8; ICOL=5.0; IROW=7.0
load "inleg.gp"
plot "p_dimperq.dat" u 1:4  w lp ls 31 lw LW notitle, \
     ""                 u 1:5  w lp ls 32 lw LW notitle, \
     ""                 u 1:6  w lp ls 33 lw LW notitle, \
     ""                 u 1:9  w lp ls 34 lw LW notitle, \
     ""                 u 1:10 w lp ls 35 lw LW notitle, \
     ""                 u 1:11 w lp ls 36 lw LW notitle
unset output
unset arrow; do for [i=801:860] { unset label i }

# ------------------------------------------------------------ legends -------
# Hand-laid strips cropped to their content: entry = 12pt line sample + marker, 3pt gap, text
# (text widths measured from Carlito 6.5pt); columns 8pt apart, rows 8.5pt apart.
LSMP = 12.0; LGAP = 3.0; LCOL = 8.0; LROW = 8.5; LPAD = 1.0
# LEGEND(file, ncol, nrow): entries in E_LS[], E_TX[], E_W[], filled column-major
LEGEND(file, ncol, nrow) = sprintf('LF="%s"; LNC=%d; LNR=%d; load "legend_strip.gp"', file, ncol, nrow)

array E_LS[6]; array E_TX[6]; array E_W[6]
E_LS[1]=21; E_TX[1]="Quora/Qwen";    E_W[1]=34.78
E_LS[2]=22; E_TX[2]="NQ/Qwen";       E_W[2]=26.75
E_LS[3]=24; E_TX[3]="MS MARCO/Qwen"; E_W[3]=48.81
E_LS[4]=25; E_TX[4]="MS MARCO/BGE";  E_W[4]=43.95
E_LS[5]=27; E_TX[5]="GloVe";         E_W[5]=15.94
E_LS[6]=28; E_TX[6]="GIST";          E_W[6]=11.89
eval LEGEND("f3_legend.pdf", 3, 2)

E_LS[1]=31; E_TX[1]="Sym-BQ";  E_W[1]=21.02
E_LS[2]=32; E_TX[2]="ADC-BQ";  E_W[2]=21.12
E_LS[3]=33; E_TX[3]="TQ-1bit"; E_W[3]=19.91
E_LS[4]=34; E_TX[4]="RaBitQ";  E_W[4]=18.22
E_LS[5]=35; E_TX[5]="PQ";      E_W[5]=7.73
E_LS[6]=36; E_TX[6]="OPQ";     E_W[6]=12.04
eval LEGEND("f3d_legend.pdf", 3, 2)

array E_LS[5]; array E_TX[5]; array E_W[5]
E_LS[1]=41; E_TX[1]="β";                  E_W[1]=3.45
E_LS[2]=42; E_TX[2]="{/:Italic f}";       E_W[2]=2.5
E_LS[3]=43; E_TX[3]="{/:Italic c_Q}";     E_W[3]=7.5
E_LS[4]=44; E_TX[4]="η";                  E_W[4]=3.49
E_LS[5]=45; E_TX[5]="{/:Italic R}^{*}";   E_W[5]=7.0
eval LEGEND("f3c_legend.pdf", 5, 1)
