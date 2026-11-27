# Score error against MRL dimension, one curve per quantizer.
#
# One Qwen3-Embedding-8B checkpoint on MS MARCO, truncated to the first d
# coordinates and renormalised, so the only thing that moves is d.  Both axes
# are logarithmic, which is the point: a quantizer that gets the same
# proportional benefit from width as another draws a PARALLEL line, and the
# dashed grey reference is what a sign code would cost if the direction carried
# nothing beyond its coordinate magnitudes (c_Q = sqrt(1-f^2)/f).
#
#   gnuplot fig_dimperq.gp   ->  f_dimperq.pdf

PW = 3.3437                      # \columnwidth
AR = 1.30
FA = ",9"
FT = ",8"
load "house.gp"

# house.gp sets the margins unconditionally, so the wider right margin that
# carries the curve labels has to be imposed afterwards -- and the canvas
# recomputed from it, or the plot box stops being BW x BW/AR.
MR = 0.620
BW = PW - ML - MR
BH = BW / AR
PH = BH + MB + MT
eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH)
set lmargin at screen ML/PW
set rmargin at screen (PW-MR)/PW
set bmargin at screen MB/PH
set tmargin at screen (PH-MT)/PH

set output "f_dimperq.pdf"
DMAX = real(word(system("tail -1 p_dimperq.dat"), 1))   # last d in the table
set logscale x 2; set xrange [30:DMAX*1.10]
set xtics ("32" 32,"128" 128,"512" 512,"4096" 4096) font FT; set mxtics 2
set xlabel "MRL dimension {/:Italic d}" font FA offset 0,0.35 tc rgb INK
set logscale y 10; set yrange [0.042:1.8]
set ytics ("1" 1,"0.5" 0.5,"0.2" 0.2,"0.1" 0.1) font FT; set mytics 10
set ylabel "score error {/:Italic c_{Q}} = ν/κ" \
    font FA offset 1.4,0 tc rgb INK

INKG = "#6E6E6E"
C_SYM = "#262626"; C_ADC = "#E8336D"; C_CS = "#0A0AF0"; C_TQ = "#9933E6"
C_RAB = "#1A9850"; C_PQ = "#D97706";  C_OPQ = "#00919B"; C_SQ = "#A8A8A8"

# columns of p_dimperq.dat: 1 d, 2 f, 3 iso, then
# 4 Sym 5 ADC 6 TQ 7 ADC+c 8 ADC+cs 9 RaBitQ 10 PQ 11 OPQ 12 SQ4 13 SQ8
load "p_dimperq_lab.gp"

plot "p_dimperq.dat" u 1:3  w l  lw 2.0 dt 2 lc rgb INKG notitle, \
     "" u 1:12 w lp lw LW dt 4 lc rgb C_SQ  pt 2  ps 0.72*PSC notitle, \
     "" u 1:4  w lp lw LW       lc rgb C_SYM pt 1  ps 0.80*PSC notitle, \
     "" u 1:5  w lp lw LW       lc rgb C_ADC pt 7  ps 0.52*PSC notitle, \
     "" u 1:6  w lp lw LW dt 3  lc rgb C_TQ  pt 6  ps 0.60*PSC notitle, \
     "" u 1:8  w lp lw LW       lc rgb C_CS  pt 4  ps 0.66*PSC notitle, \
     "" u 1:9  w lp lw LW       lc rgb C_RAB pt 12 ps 0.80*PSC notitle, \
     "" u 1:10 w lp lw LW       lc rgb C_PQ  pt 10 ps 0.74*PSC notitle, \
     "" u 1:11 w lp lw LW       lc rgb C_OPQ pt 8  ps 0.74*PSC notitle
unset output
