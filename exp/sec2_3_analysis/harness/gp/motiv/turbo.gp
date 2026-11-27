# BBQ against TurboQuant-1bit under one calibration policy.
#
# Same FP32 graph, same width (ef_search = 256), same threshold rule, same
# global ground truth: only the scorer differs.  Sweeping the calibration
# target from 0.90 to 0.995 traces each arm's achieved global Recall@10
# against the full-precision reads it spends, so the two are read at matched
# recall rather than at matched budget.
#
# Open markers are targets where the rule certified nothing in more than half
# the calibration draws; their recall is conditional on the minority that did,
# so they are plotted but not relied on.
#
# Two panels in one column, each matched to the calibration figure's geometry
# (frame l .254 r .949, ticks 6 pt, axis labels 6.5 pt).
set terminal pdfcairo enhanced font "Liberation Sans,6.5" fontscale 0.75 \
    linewidth 1 size 3.3472in,1.5819in
set output "turbo.pdf"

BBQ = "#E8336D"
TQ  = "#0A0AF0"
INK = "#262626"

set border 3 lw 1.2
set tics in nomirror scale 1.0,0.5 font ",6" tc rgb INK
unset x2tics; unset y2tics; unset grid
unset key
set bmargin 0; set tmargin 0; set lmargin 0; set rmargin 0

PT = 0.868        # top of both frames, in screen units
PB = 0.228        # bottom

set multiplot

# ---------------------------------------------------------------- (a)
set lmargin at screen 0.127
set rmargin at screen 0.474
set tmargin at screen PT
set bmargin at screen PB
set title "MS MARCO / Qwen-1024" font ",6.5" offset 0,-0.7 tc rgb INK
set xlabel "FP32 reads per query" font ",6.5" offset 0,0.75 tc rgb INK
set xrange [8:64]
set xtics 20,20,60 font ",6" offset 0,0.35
set mxtics 2
set yrange [0.893:0.997]
set ytics ("0.90" 0.90, "0.95" 0.95, "0.99" 0.99) font ",6"
set mytics 2
unset ylabel
set label 99 "Achieved Recall@10" at screen 0.030, screen 0.548 \
    center rotate by 90 font ",6.5" tc rgb INK
set label 1 "BBQ"         at 47, 0.9640 left font ",5.5" tc rgb BBQ front
set label 2 "TurboQuant"  at 24, 0.9880 left font ",5.5" tc rgb TQ  front
plot "turbo.dat" i 0 u 1:2 w l lw 1.6 lc rgb BBQ notitle, \
     "turbo.dat" i 1 u 1:2 w l lw 1.6 dt 2 lc rgb TQ notitle, \
     "turbo.dat" i 0 u 1:($4 < 0.5 ? $2 : 1/0) w p pt 7 ps 0.42 lc rgb BBQ notitle, \
     "turbo.dat" i 0 u 1:($4 >= 0.5 ? $2 : 1/0) w p pt 6 ps 0.42 lc rgb BBQ notitle, \
     "turbo.dat" i 1 u 1:($4 < 0.5 ? $2 : 1/0) w p pt 5 ps 0.38 lc rgb TQ notitle, \
     "turbo.dat" i 1 u 1:($4 >= 0.5 ? $2 : 1/0) w p pt 4 ps 0.38 lc rgb TQ notitle

# ---------------------------------------------------------------- (b)
unset label 1; unset label 2; unset label 99
set lmargin at screen 0.620
set rmargin at screen 0.967
set title "DBpedia / BGE-large" font ",6.5" offset 0,-0.7 tc rgb INK
set xrange [12:84]
set xtics 20,20,80 font ",6" offset 0,0.35
set yrange [0.893:0.997]
set ytics ("0.90" 0.90, "0.95" 0.95, "0.99" 0.99) font ",6"
set label 3 "BBQ"        at 63, 0.9660 left font ",5.5" tc rgb BBQ front
set label 4 "TurboQuant" at 24, 0.9830 left font ",5.5" tc rgb TQ  front
plot "turbo.dat" i 2 u 1:2 w l lw 1.6 lc rgb BBQ notitle, \
     "turbo.dat" i 3 u 1:2 w l lw 1.6 dt 2 lc rgb TQ notitle, \
     "turbo.dat" i 2 u 1:($4 < 0.5 ? $2 : 1/0) w p pt 7 ps 0.42 lc rgb BBQ notitle, \
     "turbo.dat" i 2 u 1:($4 >= 0.5 ? $2 : 1/0) w p pt 6 ps 0.42 lc rgb BBQ notitle, \
     "turbo.dat" i 3 u 1:($4 < 0.5 ? $2 : 1/0) w p pt 5 ps 0.38 lc rgb TQ notitle, \
     "turbo.dat" i 3 u 1:($4 >= 0.5 ? $2 : 1/0) w p pt 4 ps 0.38 lc rgb TQ notitle

unset multiplot
unset output
