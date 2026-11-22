# recall-qps_md — Recall@10 vs QPS, MS MARCO + DBpedia x 4 embedding models, with PipeANN-RaBitQ1bit (\textwidth)
# (subset of recall-qps_main; generated inputs: make_recall-qps_md.py)
# data: recall-qps_main.dat (recovered from the original figure by extract_recall-qps_main.py)
# per-panel ranges / ticks / captions / plot commands: recall-qps_main_panels.gp (generated)
load "../../common/plot/common.gp"
# style matched to f2 (Carlito, ticks 6.5pt / labels 7pt); legend (2 rows) on top; page cropped
FONT = "Carlito"; FT = 6.5; FL = 7.0
XLABN = 17.5                                # x tick labels + x title, pulled up to the axis
LEGB  = 26.5                                # three legend rows on top, one column per colour group
XTKN  = 10.5                                # top row: x tick labels only (x title shared with the bottom row)
R4  = 11.0                                  # right margin: last column's centred caption overhangs the panel
G4  = (TEXTW - L2 - R4 - 4*PW2) / 3.0       # gap between panel columns
RP  = PH2 + XTKN + TTL + 2.0                # row pitch (panel + x tick labels + title)
PXL(c) = L2 + c*(PW2 + G4)
PYB(r) = XLABN + (1 - r)*RP
W = TEXTW; H = XLABN + RP + PH2 + 1.0 + TTL + LEGB; OUT = "recall-qps_msmarco_dbpedia"
eval SETTERM
set tics font ",".sprintf("%g",FT)
XLABEL(s) = sprintf('set xlabel "%s" font ",%g" offset 0,1.0', s, FL)
TITLE(s)  = sprintf('set title "%s" font ",%g" offset 0,-0.75', s, FL)

# methods (legend order) -- family colours / markers as in the original
# readability: BBQ thicker and drawn last (on top). One colour per group, methods inside a group told
# apart by marker only (all lines solid); groups use clearly different hues. Baselines a little thinner.
LWB = LW*0.8                                # baseline curves
B_RAM = C_BLUE;  B_SSD = "#00A0D8"          # RaBitQ in memory  /  RaBitQ on SSD
G_SSD = C_GREEN; G_FAISS = "#8C510A"        # SPANN, DiskANN-PQ /  Faiss PQ, OPQ
DT2 = "(4,2.5)"; DT3 = "(5,2,1.2,2)"
set style line 1  lc rgb C_PINK   lw LW*1.45 pt 4  ps PS*1.05        # BBQ          square
set style line 2  lc rgb C_PINK   lw LWB pt 6  ps PS         # ES-BBQ       circle
set style line 3  lc rgb C_PURPLE lw LWB pt 6  ps PS                 # TurboQuant1  circle
set style line 5  lc rgb B_RAM    lw LWB pt 8  ps PS*1.1             # RaBitQ-7bit        up-triangle
set style line 7  lc rgb B_RAM    lw LWB pt 6  ps PS         # HNSW-RaBitQ1bit    circle
set style line 6  lc rgb B_SSD    lw LWB pt 8  ps PS*1.1             # RaBitQ-SSD         right-triangle (character)
set style line 8  lc rgb B_SSD    lw LWB pt 10 ps PS*1.1     # DiskANN-RaBitQ1bit down-triangle
set style line 4  lc rgb B_SSD    lw LWB pt 12 ps PS*1.25    # PipeANN-RaBitQ1bit diamond
set style line 9  lc rgb G_SSD    lw LWB pt 4  ps PS                 # SPANN       square
set style line 10 lc rgb G_SSD    lw LWB pt 10 ps PS*1.1     # DiskANN-PQ  down-triangle
set style line 11 lc rgb G_FAISS  lw LWB pt 14 ps PS                 # Faiss PQ    pentagon
set style line 12 lc rgb G_FAISS  lw LWB pt 14 ps PS         # Faiss OPQ   hexagon (character)
CH_RTRI = "▷"; CH_HEX = "⬡"; CHFS = 4.5    # character markers and their font size [pt]

set multiplot
load "recall-qps_md_legend.gp"; eval DRAWSCREEN

D = "recall-qps_md.dat"
set logscale y
set format y "10^{%L}"
set ytics 1,10
unset mytics; unset mxtics                  # labelled ticks only
set xtics nomirror; set ytics nomirror     # no ticks on the top / right border
eval RECALLAXIS
set xtics offset 0,0.45
load "recall-qps_md_panels.gp"
unset multiplot
