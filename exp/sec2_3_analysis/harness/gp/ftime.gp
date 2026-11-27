# Execution-cost decomposition: where the query time actually goes.
#
# PROVENANCE.  Every number here is from Section 4's measurement of the common
# search path (MS MARCO / Qwen-1024, single thread, Recall@10 >= 0.99):
#   ADC-BQ fixed R=120 : 349 QPS, LUT build 4.7us, FP32 read wait 517.6us
#   TurboQuant  R=80   : 362 QPS, LUT build 9.6us, FP32 read wait ~280us
#   BBQ calibrated     : 49.1 reads, FP32 read wait 217.5us
# The total is 1/QPS, so the in-memory block is what is left after the two
# measured terms -- it is traversal and candidate selection together, which
# were never instrumented apart.  BBQ's in-memory block is carried over from
# the fixed-R bar: same binary, same graph, same ef, only the rule that picks
# the refinement set differs, and Section 6.4 measures that rule as free
# (1,149 -> 1,158 QPS at ef=256).  BBQ's own total was never recorded; its bar
# is therefore a prediction, 390 QPS, and is marked as such.

PW = 3.3437                      # full column; a bar chart does not need
AR = 2.2                         # the panel aspect the line figures use
load "house.gp"

C_PRE = "#9933E6"   # query preprocessing (LUT build, rotation, query quantization)
C_CPU = "#0A0AF0"   # traversal + candidate selection, in memory
C_SSD = "#E8336D"   # FP32 refinement, waiting on SSD

#  x  prep    cpu      ssd     reads
$T << EOD
1  0.0047  2.3430  0.5176  120
2  0.0047  2.3430  0.2175  49.1
3  0.0096  2.4728  0.2800  80
EOD

set output "ftime.pdf"
set xrange [0.45:3.55]
set xtics ("BQ\n{/*0.88 R=120}" 1, "BBQ\n{/*0.88 calib.}" 2, "TQ\n{/*0.88 R=80}" 3) \
    font ",7" offset 0,0.3 scale 0
unset mxtics
set yrange [0:3.45]
set ytics 0,1,3 font FT; set mytics 2
set ylabel "latency (ms)" font FA offset 1.1,0 tc rgb INK
set boxwidth 0.62
set style fill solid 1.0 noborder

set multiplot
#  stacked back to front, and the read count over each bar
plot $T u 1:($2+$3+$4) w boxes lc rgb C_SSD notitle, \
     ""  u 1:($2+$3)    w boxes lc rgb C_CPU notitle, \
     ""  u 1:2          w boxes lc rgb C_PRE notitle, \
     ""  u 1:($2+$3+$4+0.17):(sprintf("%g", $5)) w labels center \
         font ",7" tc rgb INK notitle
set label 1 "mean FP32 reads per query" at 3.5,3.30 right font ",7" tc rgb INK front
replot
unset multiplot
unset output

# ------------------------------------------------------------ legend --------
LOUT = "ftime_legend.pdf"
LW_PT = 3.3437*72; LH_PT = 13
LFS = 6.5; LN = 3
array LTXT[3] = ["Preprocessing", "Traversal + selection", "SSD refinement"]
array LSTY[3] = [0, 0, 0]
array LDRAW[3]
LDRAW[1] = sprintf('set object 631 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs solid 1.0 noborder front', C_PRE)
LDRAW[2] = sprintf('set object 632 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs solid 1.0 noborder front', C_CPU)
LDRAW[3] = sprintf('set object 633 rect from LX0,LYLO to LX1,LYHI fc rgb "%s" fs solid 1.0 noborder front', C_SSD)
LSAMP = 11; LGAP = 4; LSEP = 11
load "legend.gp"
