# house.gp -- shared panel style for the BBQ figures.
#
# One panel per PDF.  The caller sets PW, the width the panel will be printed
# at in inches, and loads this file; everything that has to look the same from
# figure to figure -- border weight, tick style, type sizes, margins and the
# plot-box aspect -- is fixed here in absolute inches, so two panels printed at
# different widths still carry the same furniture and the same type size.
#
#   acmart sigconf: \textwidth = 506.295pt = 7.0319in, \columnwidth = 3.3437in
#   figure*, four across at 0.245\textwidth : PW = 1.7227
#   figure,  two across at 0.49\columnwidth : PW = 1.6384
#
# pdfcairo renders a font at size x fontscale x 4/3, so fontscale 0.75 makes
# the numbers below real points on the printed page.

# Callers may set any of these before loading, to depart from the default.
if (!exists("AR"))  AR  = 1.30   # plot box width / height
if (!exists("LW"))  LW  = 1.8    # curve width (0.9pt, as the untouched figures)
if (!exists("MLW")) MLW = 0.8    # marker outline width (0.4pt)
if (!exists("BLW")) BLW = 1.2    # border width (0.6pt, likewise)
if (!exists("PSC")) PSC = 0.80   # marker size scale
ML = 0.410           # margins, inches: left clears a rotated label + tick
                     # text -- measured, not guessed: the widest case (10^-4
                     # plus a rotated label at 6.5pt) leaves ~2pt to spare
MR = 0.090           # right clears the last x tick label
MB = 0.400           # bottom clears tick text + one axis label,
                     # or a row of rotated category labels
MT = 0.080           # top clears a tick label sitting on the top border
BW = PW - ML - MR
BH = BW / AR
PH = BH + MB + MT

eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH)

INK = "#262626"
if (!exists("FA")) FA = ",6.5"  # axis label, as in the figures we did not touch
if (!exists("FT")) FT = ",6"    # tick label, likewise
FN  = ",6"           # in-panel annotation

set border 31 lw BLW
set tics in mirror scale 1.3,0.7 font FT tc rgb INK
unset grid; unset x2tics; unset y2tics; unset key

set lmargin at screen ML/PW
set rmargin at screen (PW-MR)/PW
set bmargin at screen MB/PH
set tmargin at screen (PH-MT)/PH

# Series.  The palette is the main figure's and nothing else: #0A0AF0,
# #1A9850, #E8336D, #9933E6 plus grey and ink.  Colour carries the corpus and
# the open marker carries the encoder, which is how the main figure separates
# twelve methods with four hues.
set style line 21 lc rgb "#0A0AF0" lw LW pt 4  ps 0.72*PSC  # Quora/Qwen    square
set style line 22 lc rgb "#9933E6" lw LW pt 10 ps 0.80*PSC  # NQ/Qwen       v-tri
set style line 23 lc rgb "#1A9850" lw LW pt 12 ps 0.86*PSC  # DBpedia/Qwen  diamond
set style line 24 lc rgb "#E8336D" lw LW pt 6  ps 0.72*PSC  # MS MARCO/Qwen circle
set style line 25 lc rgb "#E8336D" lw LW pt 8  ps 0.80*PSC  # MS MARCO/BGE  ^-tri
set style line 26 lc rgb "#E8336D" lw LW pt 8  ps 0.80*PSC  # MS MARCO/E5   ^-tri
set style line 27 lc rgb "#909090" lw LW pt 2  ps 0.72*PSC  # GloVe         cross
set style line 28 lc rgb INK       lw LW pt 1  ps 0.72*PSC  # GIST          plus
