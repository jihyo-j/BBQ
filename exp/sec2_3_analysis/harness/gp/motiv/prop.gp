# Panel (a) of the calibration figure: the check on Proposition 5.
#
# The proposition says the rule escalates exactly when the sampled answer has
# no finite entry threshold more often than the target allows --
# P[tau = inf] = P[Bin(m, pi) > m - l] -- so the rate goes to 0 when pi < rho,
# to 1 when pi > rho, and sits at a half at pi = rho.  The figure shows one
# cell per regime, because the trichotomy is the claim; Table 11 carries the
# measured-against-predicted check on all ten cells.
#
# Lines are the prediction, not a fit, evaluated at the measured m: m - l steps
# by one every 1/rho queries, so the prediction is a sawtooth in m and reading
# it off the same abscissas compares like with like.
#
# Canvas, frame fractions and type match panel (b) cropped out of
# recall_ablation.pdf (120.6 x 106.4 pt, frame l .254 r .949 b .182 t .905,
# ticks 6 pt, axis labels 6.5 pt, annotations 5.5 pt).
set terminal pdfcairo enhanced font "Liberation Sans,6.5" fontscale 0.75 \
    linewidth 1 size 1.6750in,1.4778in
set output "prop.pdf"

HI  = "#E8336D"   # pi > rho
AT  = "#8C8C8C"   # pi = rho
LO  = "#0A0AF0"   # pi < rho
INK = "#262626"

set border 3 lw 1.2
set tics in nomirror scale 1.0,0.5 font ",6" tc rgb INK
unset x2tics; unset y2tics; unset grid
unset key

set lmargin at screen 0.254
set rmargin at screen 0.949
set bmargin at screen 0.182
set tmargin at screen 0.905

set title "Escalation, {/Symbol r} = 0.01" font ",6.5" offset 0,-0.7 tc rgb INK
set logscale x
set xrange [85:3600]
set xtics ("100" 100, "300" 300, "1000" 1000, "3000" 3000) font ",6" offset 0,0.35
set mxtics 10
set xlabel "Calibration queries {/:Italic m}" font ",6.5" offset 0,0.75 tc rgb INK
set yrange [-0.05:1.08]
set ytics ("0" 0, "0.5" 0.5, "1" 1) font ",6"
set mytics 2
set ylabel "Escalation rate" font ",6.5" offset 2.4,0 tc rgb INK

# the value the proposition pins the boundary case to
set arrow 1 from 85,0.5 to 3600,0.5 nohead lw 0.8 dt 3 lc rgb "#C8C8C8" back

# one label per regime, sitting on the curve it belongs to
set label 1 "{/Symbol p} = .0147 > {/Symbol r}" at 105, 0.985 left \
    font ",5.5" tc rgb HI front
set label 2 "{/Symbol p} = .0100 = {/Symbol r}" at 3400, 0.385 right \
    font ",5.5" tc rgb AT front
set label 3 "{/Symbol p} = .0042 < {/Symbol r}" at 3400, 0.175 right \
    font ",5.5" tc rgb LO front

# index 4 = FEVER/BGE-M3 (above), 2 = MS MARCO/BGE-M3 (at), 0 = Quora (below)
plot "prop_pred.dat" i 4 u 1:2 w l lw 1.3 lc rgb HI notitle, \
     "prop_pred.dat" i 2 u 1:2 w l lw 1.3 lc rgb AT notitle, \
     "prop_pred.dat" i 0 u 1:2 w l lw 1.3 lc rgb LO notitle, \
     "prop_pts.dat"  i 4 u 1:2 w p pt 7 ps 0.42 lc rgb HI notitle, \
     "prop_pts.dat"  i 2 u 1:2 w p pt 7 ps 0.42 lc rgb AT notitle, \
     "prop_pts.dat"  i 0 u 1:2 w p pt 7 ps 0.42 lc rgb LO notitle
unset output
