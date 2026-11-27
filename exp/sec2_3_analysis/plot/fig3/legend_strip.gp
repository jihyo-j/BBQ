# legend_strip.gp -- one cropped legend strip (called via LEGEND() in fig3.gp)
# inputs: LF (output file), LNC columns, LNR rows, arrays E_LS / E_TX / E_W (column-major)
array CW[LNC]
do for [c=1:LNC] { CW[c] = 0; do for [r=1:LNR] { i = (c-1)*LNR + r; if (i <= |E_W|) { if (E_W[i] > CW[c]) { CW[c] = E_W[i] } } } }
LWT = 2*LPAD + (LNC-1)*LCOL
do for [c=1:LNC] { LWT = LWT + LSMP + LGAP + CW[c] }
LHT = LNR*LROW + 1.0
if (exists("PNG")) { eval sprintf('set terminal pngcairo enhanced font "Carlito,8" fontscale 3 linewidth 4 pointscale 4 size %d,%d background rgb "#EEEEEE"', int(LWT*4), int(LHT*4)) } else { eval sprintf('set terminal pdfcairo enhanced font "Carlito,8" fontscale 0.75 linewidth 1 size %.4fin,%.4fin', LWT/72.0, LHT/72.0) }
set output (exists("PNG") ? LF[1:strlen(LF)-4].".png" : LF)
unset border; unset tics; unset xlabel; unset ylabel; unset title; unset key
unset logscale x; unset logscale y; unset arrow; unset label; unset object
set lmargin at screen 0; set rmargin at screen 1; set bmargin at screen 0; set tmargin at screen 1
set xrange [0:LWT]; set yrange [0:LHT]
set print $LPTS
X = LPAD
do for [c=1:LNC] {
  do for [r=1:LNR] {
    i = (c-1)*LNR + r
    if (i <= |E_W|) {
      Y = LHT - 0.5 - (r-0.5)*LROW
      eval sprintf('set arrow %d from %f,%f to %f,%f nohead ls %d', 600+i, X, Y, X+LSMP, Y, E_LS[i])
      eval sprintf('set label %d "%s" at %f,%f left font ",6.5" tc rgb "#262626"', 700+i, E_TX[i], X+LSMP+LGAP, Y)
      print sprintf("%f %f %d", X+LSMP/2, Y, E_LS[i])
    }
  }
  X = X + LSMP + LGAP + CW[c] + LCOL
}
set print
plot for [i=1:|E_W|] $LPTS u 1:2 every ::i-1::i-1 w p ls E_LS[i] lw MLW notitle
unset output
unset arrow; unset label
