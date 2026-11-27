# f2 — Figure 2 (jihyo redraw in the f1 style: Carlito, left/bottom axes, same panel size as f1a/f1b)
#   (a) index size vs number of vectors at d = 4096 (exact arithmetic, as in sieun's fig2.gp):
#       FP32 4d B, 4-bit d/2 B, 1-bit d/8 B per vector + 208 B HNSW graph (M = 24); DRAM budget 1007 GiB
#   (b) BQ score error c_Q vs embedding dimension, MS MARCO / Qwen3-Embedding-8B (motiv/mrl_scoreerr.dat)
#   gnuplot -e 'PART="a"' f2.gp -> f2a.pdf     gnuplot -e 'PART="b"' f2.gp -> f2b.pdf
#   gnuplot -e 'PART="l"' f2.gp -> f2_legend.pdf
if (!exists("PART")) PART = "a"
INK = "#262626"; C_FP = "#262626"; C_4B = "#0A0AF0"; C_1B = "#E8336D"; GREY = "#8C8C8C"; SHADE = "#E6E6E6"

W  = 3.3437/2                    # same canvas width as f1a / f1b
ML = 0.540; MR = 0.095; MB = 0.300; MT = 0.170
BH = 0.8158
H  = BH + MB + MT
YLXPT = 20.0
if (PART ne "l") { ML = ML - 14.0/72.0; YLXPT = YLXPT - 14.0 }   # f2a and f2b: same margins -> same plot-box size
if (PART eq "l") { W = 3.3437; H = 13.0/72.0 }

OUTF = (PART eq "l") ? "f2_legend" : "f2".PART
if (exists("PNG")) { eval sprintf('set terminal pngcairo enhanced font "Carlito,8" fontscale 3 linewidth 4 pointscale 4 size %d,%d', int(W*288), int(H*288)) } else { eval sprintf('set terminal pdfcairo enhanced font "Carlito,8" fontscale 0.75 linewidth 1 size %.4fin,%.4fin', W, H) }
set output (exists("PNG") ? OUTF."_preview.png" : OUTF.".pdf")

unset key; unset grid; unset x2tics; unset y2tics; unset ylabel
set lmargin at screen ML/W; set rmargin at screen (W-MR)/W
set bmargin at screen MB/H; set tmargin at screen (H-MT)/H
set border 15 lw 1.0      # full frame
set tics in nomirror scale 0.6,0.3 font ",6.5" tc rgb INK
set label 90 "" at 0,0                                    # placeholder, replaced per panel
YTITLE(s) = sprintf('set label 90 "%s" at screen %g, screen %g center rotate by 90 font ",7" tc rgb INK front', s, (YLXPT/72.0)/W, (MB+BH/2)/H)
XTITLE(s) = sprintf('set xlabel "%s" font ",7" offset 0,0.9 tc rgb INK', s)
TITLE(s)  = sprintf('set title "%s" font ",7" offset 0,-0.6 tc rgb INK', s)

if (PART eq "a") {
    GIB = 1073741824.0; GRAPH = 208.0; D = 4096; BUD = 1007.0
    sz(n,b) = n*(b+GRAPH)/GIB
    cross(b) = BUD*GIB/(b+GRAPH)
    set logscale x; set xrange [6.5e6:1.4e10]
    set xtics ("10^{7}" 1e7,"10^{8}" 1e8,"10^{9}" 1e9,"10^{10}" 1e10) offset 0,0.4 scale 0
    unset mxtics
    set logscale y; set yrange [0.4:30000]
    set ytics ("10^{0}" 1,"10^{1}" 10,"10^{2}" 100,"10^{3}" 1000,"10^{4}" 10000) scale 0
    unset mytics
    eval XTITLE("Number of vectors")
    eval YTITLE("Index size (GiB)")
    eval TITLE("{/:Italic d} = 4096")
    # above the DRAM budget: grey band; budget itself: dashed line (no in-plot text)
    set object 1 rect from graph 0, first BUD to graph 1, graph 1 fc rgb SHADE fs solid 1.0 noborder behind
    set arrow 1 from graph 0, first BUD to graph 1, first BUD nohead lw 1.2 dt 2 lc rgb INK front
    # where each representation runs out of DRAM: dotted drop line + marker
    set arrow 2 from cross(4*D),0.4 to cross(4*D),BUD nohead lw 0.8 dt 3 lc rgb C_FP
    set arrow 3 from cross(D/2),0.4 to cross(D/2),BUD nohead lw 0.8 dt 3 lc rgb C_4B
    set arrow 4 from cross(D/8),0.4 to cross(D/8),BUD nohead lw 0.8 dt 3 lc rgb C_1B
    # legend inside the panel, bottom right, on a white patch (the dotted drop lines pass under it)
    LFS = 6.0; LX0 = 0.52; LX1 = 0.60; LTX = 0.63
    array LTXT[3] = ["FP", "4-bit code", "1-bit code"]
    array LCOL[3] = [C_FP, C_4B, C_1B]
    array LY[3]   = [0.30, 0.19, 0.08]
    set object 2 rect from graph LX0-0.02, graph 0.02 to graph 0.985, graph 0.365 fc rgb "white" fs solid 1.0 noborder front
    do for [i=1:3] {
        eval sprintf('set arrow %d from graph %g, graph %g to graph %g, graph %g nohead lw 1.9 lc rgb "%s" front', 20+i, LX0, LY[i], LX1, LY[i], LCOL[i])
        eval sprintf('set label %d "%s" at graph %g, graph %g left font ",%g" tc rgb INK front', 20+i, LTXT[i], LTX, LY[i], LFS)
    }
    set samples 200
    plot sz(x,4*D) w l lw 1.9 lc rgb C_FP notitle, \
         sz(x,D/2) w l lw 1.9 lc rgb C_4B notitle, \
         sz(x,D/8) w l lw 1.9 lc rgb C_1B notitle, \
         "+" u (cross(4*D)):(BUD) every ::0::0 w p pt 6  ps 0.55 lw 1.0 lc rgb C_FP notitle, \
         "+" u (cross(D/2)):(BUD) every ::0::0 w p pt 4  ps 0.55 lw 1.0 lc rgb C_4B notitle, \
         "+" u (cross(D/8)):(BUD) every ::0::0 w p pt 12 ps 0.65 lw 1.0 lc rgb C_1B notitle
}

if (PART eq "b") {
    set logscale x 2; set xrange [26:5200]
    set xtics ("32" 32,"128" 128,"512" 512,"4096" 4096) offset 0,0.4
    unset mxtics
    set yrange [0:1.0]; set ytics ("0" 0, "0.2" 0.2, "0.4" 0.4, "0.6" 0.6, "0.8" 0.8, "1" 1.0)
    unset mytics
    eval XTITLE("Dimension {/:Italic d}")
    eval YTITLE("Score error {/:Italic c_Q}")
    eval TITLE("MS MARCO")
    # the gap to the isotropic reference is shaded
    GAPC = "#D3DAFC"; C_SE = C_4B   # blue line, light-blue gap
    # in-panel key, bottom left (empty: both curves are high at small d)
    LFS = 6.0
    set arrow 21 from graph 0.09, graph 0.28 to graph 0.19, graph 0.28 nohead lw 1.9 lc rgb C_SE front
    set label 21 "" at graph 0.14, graph 0.28 point pt 7 ps 0.28 lc rgb C_SE front
    set label 22 "Score error" at graph 0.22, graph 0.28 left font sprintf(",%g",LFS) tc rgb INK front
    set arrow 23 from graph 0.09, graph 0.17 to graph 0.19, graph 0.17 nohead lw 1.9 lc rgb GREY front
    set label 24 "Isotropic sign code" at graph 0.22, graph 0.17 left font sprintf(",%g",LFS) tc rgb INK front
    plot "motiv/mrl_scoreerr.dat" u 1:2:($3 > $2 ? $3 : $2) w filledcurves fc rgb GAPC fs solid 1.0 noborder notitle, \
         ""                       u 1:3 w l  lw 1.9 lc rgb GREY notitle, \
         ""                       u 1:2 w lp lw 1.9 pt 7 ps 0.28 lc rgb C_SE notitle
}

if (PART eq "l") {
    unset border; unset tics; unset xlabel; unset title; unset label 90
    set lmargin at screen 0; set rmargin at screen 1; set bmargin at screen 0; set tmargin at screen 1
    LW_PT = W*72; LH_PT = 13.0
    set xrange [0:LW_PT]; set yrange [0:LH_PT]
    LFS = 6.5; LKW = 0.47; LSAMP = 14; LGAP = 4; LSEP = 12
    array LTXT[3] = ["FP", "4-bit code", "1-bit code"]
    array LCOL[3] = [C_FP, C_4B, C_1B]
    LTOT = 2*LSEP
    do for [i=1:3] { LTOT = LTOT + LSAMP + LGAP + strlen(LTXT[i])*LKW*LFS }
    LCUR = LW_PT/2.0 - LTOT/2.0; LYM = LH_PT/2.0
    do for [i=1:3] {
        eval sprintf('set arrow %d from %f,%f to %f,%f nohead lw 1.9 lc rgb "%s" front', 620+i, LCUR, LYM, LCUR+LSAMP, LYM, LCOL[i])
        eval sprintf('set label %d "%s" at %f,%f left font ",%g" tc rgb INK front', 630+i, LTXT[i], LCUR+LSAMP+LGAP, LYM, LFS)
        LCUR = LCUR + LSAMP + LGAP + strlen(LTXT[i])*LKW*LFS + LSEP
    }
    plot NaN notitle
}
unset output
