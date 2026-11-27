# inleg.gp -- legend drawn inside the current panel, in pt from the plot box's lower-left corner.
# inputs: arrays E_LS / E_TX / E_W (text widths at 6.5pt, column-major), LNC, LNR,
#         IX0, IY0 (pt: left edge, bottom edge of the legend), IFS (font pt),
#         ISMP (sample), IGAP (sample->text), ICOL (column gap), IROW (row pitch)
BWPT = BW*72.0; BHPT = BH*72.0
array ICW[LNC]
do for [c=1:LNC] { ICW[c] = 0; do for [r=1:LNR] { i = (c-1)*LNR + r; if (i <= |E_W|) { if (E_W[i]*IFS/6.5 > ICW[c]) { ICW[c] = E_W[i]*IFS/6.5 } } } }
X = IX0
do for [c=1:LNC] {
  do for [r=1:LNR] {
    i = (c-1)*LNR + r
    if (i <= |E_W|) {
      Y = (exists("IFLIP") && IFLIP) ? IY0 + (r - 0.5)*IROW : IY0 + (LNR - r + 0.5)*IROW
      eval sprintf('set arrow %d from graph %f,%f to graph %f,%f nohead ls %d front', 800+i, X/BWPT, Y/BHPT, (X+ISMP)/BWPT, Y/BHPT, E_LS[i])
      eval sprintf('set label %d "" at graph %f,%f point ls %d lw MLW front', 820+i, (X+ISMP/2)/BWPT, Y/BHPT, E_LS[i])
      eval sprintf('set label %d "%s" at graph %f,%f left font ",%g" tc rgb "#262626" front', 840+i, E_TX[i], (X+ISMP+IGAP)/BWPT, Y/BHPT, IFS)
    }
  }
  X = X + ISMP + IGAP + ICW[c] + ICOL
}
