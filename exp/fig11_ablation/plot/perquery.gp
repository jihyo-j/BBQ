# perquery — (a) per-query R* vs z10 (8,400 queries) with dataset centroids,
#            (b) DBpedia z10 vs #relevant documents, binned median + IQR
# data: perquery.dat + raster point clouds perquery_cloud_{a,b}.png
#       (the original scatter clouds were rasterized; they are reused as-is)
load "../../common/plot/common.gp"
load "perquery_cloud.gp"
NROWS = 2
W = COLW; H = PB(NROWS) + PH2 + TOPM + TTL; OUT = "perquery"
eval SETTERM
# panel (b) has its own y label -> wider gap than G2
GP = G2 + 9; PWP = (COLW - L2 - GP - R2) / 2.0

# dataset colours (original mapping, snapped to the palette); centroids are
# white-filled, so each has a fill style (10+i) drawn underneath
set style line 1 lc rgb C_PINK   lw MLW pt 6  ps PS*1.6    # Quora
set style line 2 lc rgb C_BLUE   lw MLW pt 4  ps PS*1.6    # DBpedia
set style line 3 lc rgb C_ORANGE lw MLW pt 12 ps PS*1.9    # MS MARCO
set style line 4 lc rgb C_PURPLE lw MLW pt 10 ps PS*1.7    # NQ
set style line 5 lc rgb C_GREEN  lw MLW pt 8  ps PS*1.7    # FEVER
do for [i=1:5] { set style line 10+i lc rgb "white" pt (i==1?7:i==2?5:i==3?13:i==4?11:9) ps (i==1?PS*1.6:i==2?PS*1.6:i==3?PS*1.9:PS*1.7) }
set style line 20 lc rgb C_BLACK lw LW pt 6 ps PS*1.1      # binned median
set style line 21 lc rgb "#73808080" lw 2.6                 # IQR bar (alpha 0.55)

set multiplot
load "perquery_legend.gp"; eval DRAWSCREEN

D = "perquery.dat"
N = "quora dbpedia msmarco nq fever"

# raster cloud underneath a panel: CLOUD(t) draws it in panel-fraction coords
CLOUD(t) = sprintf('unset border; unset tics; unset xlabel; unset ylabel; unset title; unset logscale; set xrange [0:1]; set yrange [0:1]; plot CLOUD_%s binary filetype=png dx=CFW_%s/CW_%s dy=CFH_%s/CH_%s origin=(CX_%s,CY_%s) with rgbalpha notitle; set border 15 lw BLW; set tics in mirror scale TICMAJ,TICMIN; set xtics in mirror; set ytics in mirror; unset x2tics; unset y2tics', t,t,t,t,t,t,t)

# (a) pooled scatter
eval PANEL(L2, PB(NROWS), PWP, PH2)
eval CLOUD("a")
set xrange [3:10.5]; set xtics 4,2,10
set logscale y; set yrange [7:320]; set ytics 1,10; set format y "10^{%L}"; set mytics 10
eval XLABEL("{/:Italic z}_{10}")
eval YLABEL("{/:Italic R}^*")
eval TITLE("(a) 8,400 queries, pooled {/:Italic ρ} = −0.61")
plot for [i=1:5] D index "cent_".word(N,i) u 1:2 w p ls 10+i notitle, \
     for [i=1:5] D index "cent_".word(N,i) u 1:2 w p ls i notitle

# (b) DBpedia
eval PANEL(L2+PWP+GP, PB(NROWS), PWP, PH2)
eval CLOUD("b")
set format y "%g"
set logscale x; set xrange [0.6962:2007.2]; set xtics 1,10; set format x "10^{%L}"; set mxtics 10
set yrange [4.5437:9.8352]; set ytics 5,1,9; set mytics default
eval XLABEL("relevant documents")
eval YLABEL("{/:Italic z}_{10}"); set ylabel offset -0.3,0
eval TITLE("(b) DBpedia, {/:Italic ρ} = +0.38")
plot D index "iqr_b" u 1:2:(0):($3-$2) w vectors nohead ls 21 notitle, \
     D index "median_b" u 1:2 w l ls 20 notitle, \
     D index "median_b" u 1:2 w p pt 7 lc rgb "white" ps PS*1.1 notitle, \
     D index "median_b" u 1:2 w p ls 20 lw MLW notitle
unset multiplot
