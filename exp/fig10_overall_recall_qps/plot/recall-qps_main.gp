# recall-qps_main — Recall@10 vs QPS, 4 datasets x 4 embedding models (Sec. 6, \textwidth)
# data: recall-qps_main.dat (recovered from the original figure by extract_recall-qps_main.py)
# per-panel ranges / ticks / captions / plot commands: recall-qps_main_panels.gp (generated)
load "../../common/plot/common.gp"
# style matched to f2 (Carlito, ticks 6.5pt / labels 7pt); legend (2 rows) on top; page cropped
FONT = "Carlito"; FT = 6.5; FL = 7.0
XLABN = 17.5                                # x tick labels + x title, pulled up to the axis
LEGB  = 18.0                                # two legend rows on top
R4  = 11.0                                  # right margin: last column's centred caption overhangs the panel
G4  = (TEXTW - L2 - R4 - 4*PW2) / 3.0       # gap between panel columns
RP  = PH2 + XLABN + TTL + 2.0               # row pitch (panel + x labels + title)
PXL(c) = L2 + c*(PW2 + G4)
PYB(r) = XLABN + (3 - r)*RP
W = TEXTW; H = XLABN + 3*RP + PH2 + 1.0 + TTL + LEGB; OUT = "recall-qps_main"
eval SETTERM
set tics font ",".sprintf("%g",FT)
XLABEL(s) = sprintf('set xlabel "%s" font ",%g" offset 0,1.0', s, FL)
TITLE(s)  = sprintf('set title "%s" font ",%g" offset 0,-0.75', s, FL)

# methods (legend order) -- family colours / markers as in the original
set style line 1  lc rgb C_PINK   lw LW pt 4  ps PS        # BBQ          square
set style line 2  lc rgb C_PINK   lw LW pt 6  ps PS        # ES-BBQ          circle
set style line 3  lc rgb C_PURPLE lw LW pt 6  ps PS        # TurboQuant1 (dataset-tuned R) circle
set style line 4  lc rgb C_BLUE   lw LW pt 12 ps PS*1.25   # PipeANN-RaBitQ1bit diamond (add_recall-qps_main_pipeann.py)
set style line 5  lc rgb C_BLUE   lw LW pt 8  ps PS*1.1    # RaBitQ-7bit     up-triangle
set style line 6  lc rgb C_BLUE   lw LW pt 8  ps PS*1.1    # RaBitQ-SSD      right-triangle (character, see CH_RTRI)
set style line 7  lc rgb C_BLUE   lw LW pt 6  ps PS        # HNSW-RaBitQ1    circle
set style line 8  lc rgb C_BLUE   lw LW pt 10 ps PS*1.1    # DiskANN-RaBitQ  down-triangle
set style line 9  lc rgb C_GREEN  lw LW pt 4  ps PS        # SPANN           square
set style line 10 lc rgb C_GREEN  lw LW pt 10 ps PS*1.1    # DiskANN-PQ      down-triangle
set style line 11 lc rgb C_GREEN  lw LW pt 14 ps PS        # Faiss PQ        pentagon
set style line 12 lc rgb C_GREEN  lw LW pt 14 ps PS        # Faiss OPQ       hexagon (character, see CH_HEX)
CH_RTRI = "▷"; CH_HEX = "⬡"; CHFS = 4.5    # character markers and their font size [pt]

set multiplot
load "recall-qps_main_legend.gp"; eval DRAWSCREEN

D = "recall-qps_main.dat"
set logscale y
set format y "10^{%L}"
set ytics 1,10
set mytics 10
eval RECALLAXIS
set xtics offset 0,0.45
load "recall-qps_main_panels.gp"
unset multiplot
