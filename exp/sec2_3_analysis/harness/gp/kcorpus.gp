# The candidate budget against how many neighbors are asked for.
# Columns of p_kcorpus.dat: k, then mean R* for
#   Quora/Qwen {tq1, rabitq1, adc_bq}, MS MARCO/Qwen {...}, MS MARCO/BGE {...}
# A solid line is TurboQuant-1bit and a dashed one RaBitQ-1bit; the pair
# sits inside its corpus's band, which is the point of the panel.
load "house_paper.gp"
F = "p_kcorpus.dat"
set output "f3e.pdf"
set logscale x; set xrange [0.9:115]
set xtics ("1" 1,"10" 10,"100" 100) font FT; set mxtics 10
set xlabel "{/:Italic k}" font FA offset 0,1.3 tc rgb INK
set logscale y; set yrange [0.95:1300]
set ytics ("1" 1,"10" 10,"10^{2}" 100,"10^{3}" 1000) font FT; unset mytics
set ylabel "mean {/:Italic R}^{*}" font FA offset 1.9,0 tc rgb INK
plot F u 1:2  w l ls 21 notitle,            F u 1:3  w l ls 21 dt 2 notitle, \
     F u 1:5  w l ls 24 notitle,            F u 1:6  w l ls 24 dt 2 notitle, \
     F u 1:8  w l ls 25 notitle,            F u 1:9  w l ls 25 dt 2 notitle, \
     F u 1:2  every 18::3 w p ls 21 lw MLW notitle, \
     F u 1:5  every 18::9 w p ls 24 lw MLW notitle, \
     F u 1:8  every 18::15 w p ls 25 lw MLW notitle
unset output
