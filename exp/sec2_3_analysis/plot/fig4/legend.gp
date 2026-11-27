# legend.gp -- lay a legend strip out by hand.
#
# gnuplot's own key sizes every column to the longest label and then leaves a
# fixed ~1.6pt before the next sample, so a long label ends up touching its
# neighbour, and no combination of font, samplen or width fixes it: widening
# the column wraps the row instead.  This lays the entries out in canvas
# points, so the sample length, the sample-to-text gap and the gap between
# entries are what they say they are, and the row is centred on its ink.
#
# The caller sets, before loading:
#   LOUT          output file
#   LW_PT, LH_PT  canvas size in points
#   LN            number of entries
#   LTXT[LN]      labels
#   LDRAW[LN]     one command per entry drawing its sample.  It is eval'd with
#                 LX0, LX1 (sample ends), LYM (middle) and LYLO, LYHI (a box
#                 half as tall as the strip) already set, so e.g.
#                   'set arrow 601 from LX0,LYM to LX1,LYM nohead ls 21'
#                   'set object 601 rect from LX0,LYLO to LX1,LYHI fc rgb "#..."'
#   LSTY[LN]      line style for the entry's marker, or 0 for no marker
#   LFS           label font size (pt)
# and optionally LSAMP (sample length, 13), LGAP (sample->text, 4.5),
# LSEP (entry->entry, 13), LCX (row centre as a fraction of the canvas, 0.5),
# LKW (width of an average glyph as a fraction of the font size, 0.52),
# LMLW (marker outline width, 1.0).

if (!exists("LSAMP")) LSAMP = 13.0
if (!exists("LGAP"))  LGAP  = 4.5
if (!exists("LSEP"))  LSEP  = 13.0
if (!exists("LCX"))   LCX   = 0.5
if (!exists("LKW"))   LKW   = 0.52
if (!exists("LMLW"))  LMLW  = 1.0

eval sprintf('set terminal pdfcairo enhanced font "Carlito,8" \
  fontscale 0.75 linewidth 1 size %.4fin,%.4fin', LW_PT/72.0, LH_PT/72.0)
set output LOUT
unset border; unset tics; unset xlabel; unset ylabel; unset key; unset grid
unset logscale x; unset logscale y
unset nonlinear x; unset nonlinear y
unset arrow; unset object; unset label
set lmargin at screen 0; set rmargin at screen 1
set bmargin at screen 0; set tmargin at screen 1
set xrange [0:LW_PT]; set yrange [0:LH_PT]

# entry widths.  A previous pass may have measured the real label widths;
# use them when they are there, and the average-glyph estimate otherwise.
LWF = sprintf("w_%s.gp", LOUT[1:strlen(LOUT)-4])
if (system(sprintf("test -f %s && echo 1 || echo 0", LWF)) eq "1") { load LWF }
array LTW[LN]
do for [i=1:LN] {
    LTW[i] = (exists("LWID") && |LWID| == LN) ? LWID[i] : strlen(LTXT[i])*LKW*LFS
}
LTOT = (LN-1)*LSEP
do for [i=1:LN] { LTOT = LTOT + LSAMP + LGAP + LTW[i] }
LCUR = LCX*LW_PT - LTOT/2.0
LYM  = LH_PT/2.0
LYLO = LYM - LH_PT*0.22
LYHI = LYM + LH_PT*0.22

array LMID[LN]
do for [i=1:LN] {
    LX0 = LCUR; LX1 = LCUR + LSAMP
    eval LDRAW[i]
    eval sprintf('set label %d "%s" at %f,%f left font ",%g" tc rgb "#262626" front', \
                 500+i, LTXT[i], LX1+LGAP, LYM, LFS)
    LMID[i] = (LX0 + LX1)/2.0
    LCUR = LX1 + LGAP + LTW[i] + LSEP
}

set print $LPTS
do for [i=1:LN] { print sprintf("%f %f", LMID[i], LYM) }
set print
LCMD = ""
do for [i=1:LN] {
    if (LSTY[i] > 0) {
        LCMD = LCMD . sprintf('%s $LPTS u 1:2 every ::%d::%d w p ls %d lw %g notitle', \
                              (strlen(LCMD) == 0 ? "" : ", "), i-1, i-1, LSTY[i], LMLW)
    }
}
if (strlen(LCMD) == 0) { LCMD = "NaN notitle" }
eval "plot " . LCMD
unset output
do for [i=1:LN] { eval sprintf("unset label %d", 500+i) }
unset arrow; unset object
