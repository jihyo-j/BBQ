# house_paper.gp -- panel furniture copied from the figures already in the
# paper, so a new panel drops into Figure 3 or 4 without looking foreign.
#
# Measured off figures/bbq/f3a.pdf in the Overleaf snapshot:
#   canvas 124 x 82 pt, plot box 94.5 x 62.2 pt (aspect 1.518),
#   margins left 24.0, right 5.5, top 3.0, bottom 16.8 pt.
# A figure-width panel is 0.245\textwidth; a column panel (Figure 4) is
# 117 x 81 pt with a 90.5 pt box.

if (!exists("PW")) PW = 124.0/72.0          # 0.245\textwidth
if (!exists("PH")) PH = 82.0/72.0
if (!exists("ML")) ML = 24.0/72.0
if (!exists("MR")) MR = 5.5/72.0
if (!exists("MT")) MT = 3.0/72.0
if (!exists("MB")) MB = 16.8/72.0
if (!exists("LW"))  LW  = 1.8
if (!exists("MLW")) MLW = 0.8
if (!exists("BLW")) BLW = 1.2
if (!exists("PSC")) PSC = 0.80

eval sprintf('set terminal pdfcairo enhanced font "Liberation Sans,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', PW, PH)

INK = "#262626"
# sampled off f3a.pdf: tick labels render at 6.5pt and axis labels at 7.0pt
if (!exists("FA")) FA = ",7"
if (!exists("FT")) FT = ",6.5"

set border 31 lw BLW
set tics in mirror scale 1.3,0.7 font FT tc rgb INK
unset grid; unset x2tics; unset y2tics; unset key

set lmargin at screen ML/PW
set rmargin at screen (PW-MR)/PW
set bmargin at screen MB/PH
set tmargin at screen (PH-MT)/PH

# the paper's own series colours (sampled from figures/bbq/f3_legend.pdf):
# #0A0AF0 #9933E6 #E8336D #909090 #262626, with MS MARCO/Qwen and /BGE
# sharing the red and separated by marker.
set style line 21 lc rgb "#0A0AF0" lw LW pt 4 ps 0.72*PSC   # Quora/Qwen     sq
set style line 24 lc rgb "#E8336D" lw LW pt 6 ps 0.72*PSC   # MS MARCO/Qwen  circ
set style line 25 lc rgb "#E8336D" lw LW pt 8 ps 0.80*PSC   # MS MARCO/BGE   ^tri
