# =====================================================================
#  common.gp — shared style for every BBQ paper figure (gnuplot 6)
#
#  All sizes are REAL PRINT POINTS: figures are drawn at the exact width
#  they are included at (\columnwidth = 241.15pt, \textwidth = 506.30pt,
#  acmart sigconf), so a 6pt font here is a 6pt font in the paper.
#
#  Usage in a figure script:
#      load "common.gp"
#      W = COLW; H = PB(1) + PH2 + TOPM; OUT = "adc_ivf"; eval SETTERM
#  Preview (PNG, 4 px per pt):  gnuplot -e 'PREVIEW=1' fig.gp
# =====================================================================

COLW = 241.15          # \columnwidth  [pt]
TEXTW = 506.30         # \textwidth    [pt]

# ---- fonts [pt] ------------------------------------------------------
FONT = "Liberation Sans"   # metric-compatible with Helvetica/Arial
FT  = 6.0                  # tick labels
FL  = 6.5                  # axis labels, (a)/(b) sub-captions, panel titles
FK  = 6.5                  # legend text
FA  = 5.5                  # in-plot annotations

# ---- strokes [gnuplot lw units; 1 = 0.5pt] ---------------------------
LW  = 1.8                  # data curves         (0.9pt)
MLW = 1.0                  # marker outlines     (0.5pt)
BLW = 1.2                  # frame               (0.6pt)
PS  = 0.55                 # default marker size
HATCH = 5                  # fill pattern for hatched bars (cairo: 5 = ///)

# ---- tick geometry -----------------------------------------------------
TICMAJ = 0.6               # major tick scale (was 1.0)
TICMIN = 0.3               # minor tick scale (was 0.5)

# ---- palette (method families) ---------------------------------------
C_PINK   = "#E8336D"   # BBQ / BBQ / SymBQ / 1-bit codes
C_GREEN  = "#1A9850"   # PQ / DiskANN-PQ / SPANN / Faiss
C_TEAL   = "#00998A"   # OPQ (when separate from PQ)
C_PURPLE = "#9933E6"   # TurboQuant / MRQ
C_BLUE   = "#0A0AF0"   # RaBitQ family / FP32 graph
C_ORANGE = "#FF8000"   # SQ
C_BLACK  = "#000000"   # FP32 / exact references
C_GRAY   = "#808080"   # GloVe / misc
C_RED    = "#C0392B"   # (spare)

# ---- terminal ----------------------------------------------------------
# pdfcairo renders font size x fontscale x 4/3  -> fontscale 0.75 = true pt
# pdfcairo lw 1 = 0.5pt.   Call  `eval SETTERM`  after W, H, OUT are set.
PSCALE = 0.72  # png pointscale per px/pt (calibrated against pdf)
SETTERM = 'if (exists("PREVIEW")) { K = 4.0; set terminal pngcairo enhanced font FONT.",".sprintf("%g",FT) fontscale 0.75*K linewidth 0.5*K pointscale PSCALE*K size int(W*K),int(H*K); set output OUT.".png" } else { set terminal pdfcairo enhanced font FONT.",".sprintf("%g",FT) fontscale 0.75 linewidth 1 size W/72.0,H/72.0; set output OUT.".pdf" }'

# screen-coordinate helpers: positions given in pt from the bottom-left corner
sx(p) = p / W
sy(p) = p / H

# ---- axes --------------------------------------------------------------
set border 15 lw BLW
set tics in mirror scale TICMAJ,TICMIN font ",".sprintf("%g",FT)
set xtics in mirror; set ytics in mirror
unset x2tics; unset y2tics
set mxtics default; set mytics default
set grid noxtics noytics
unset key
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0
# gnuplot resets the label font when the text is re-set -> always use these:
#   eval XLABEL("(a)  Recall\\@10 (...)")   eval YLABEL("QPS")   eval TITLE("MS MARCO / Qwen")
XLABEL(s) = sprintf('set xlabel "%s" font ",%g" offset 0,-0.15', s, FL)
YLABEL(s) = sprintf('set ylabel "%s" font ",%g" offset 1.0,0', s, FL)
TITLE(s)  = sprintf('set title "%s" font ",%g" offset 0,-0.7', s, FL)
set style fill solid 1.0 border
set pointintervalbox 0
set clip two               # draw segments whose both ends lie outside the range

# panel placement in pt:  PANEL(x0, y0, w, h)
PANEL(x0,y0,w,h) = sprintf("set origin %g,%g; set size %g,%g", x0/W, y0/H, w/W, h/H)

# ---- standard layouts [pt] -------------------------------------------
# Two panels side by side in one column (most figures):
L2  = 22.0                 # left edge -> first panel (room for y tick labels + ylabel)
G2  = 16.0                 # gap between panels (room for 2nd panel's y tick labels)
R2  = 6.0                  # right margin
PW2 = (COLW - L2 - G2 - R2) / 2.0
PH2 = 77.0                 # panel height
TOPM = 4.0                 # top margin (add TTL when panels have titles)
TTL  = 9.0                 # extra top room for a panel title
XLAB = 24.0                # room under a panel for x tick labels + x label
ROW  = 8.5                 # legend row pitch (must match tools/legend.py)
LEGPAD = 2.0               # canvas bottom -> legend bottom
LEGTOP(n) = LEGPAD + ROW*n # top of an n-row legend
PB(n) = LEGTOP(n) + XLAB   # panel bottom above an n-row legend

# nonlinear "recall" axis used by matplotlib originals: x' = -sqrt(1-x)
RECALLAXIS = 'set nonlinear x via -sqrt(1-x) inverse 1-x*x'

# Empty full-canvas plot used to render screen-space decorations (legend)
DRAWSCREEN = 'set origin 0,0; set size 1,1; unset nonlinear x; unset nonlinear y; unset border; unset tics; unset xlabel; unset ylabel; unset title; set xrange [0:1]; set yrange [0:1]; unset logscale; plot NaN notitle; unset arrow; unset label; unset object; set border 15 lw BLW; set tics in mirror scale TICMAJ,TICMIN; set xtics in mirror; set ytics in mirror; unset x2tics; unset y2tics'
