# Figure 2's key, redrawn at a readable size.
#
# The panels themselves are the supplied artwork, cut up by cut_fig2.py; only
# the key is redrawn, because it was set at 4.8pt inside the capacity panel and
# a crop cannot enlarge type.  The colours are the ones sampled out of that
# file (black, #0A0AF0, #E8336D) and the labels are its own, so the strip says
# exactly what the original key said.

LOUT = "f2_legend.pdf"
LW_PT = 3.3437*72; LH_PT = 13
LFS = 6.5; LN = 3
array LTXT[3] = ["FP", "4-bit code", "1-bit code"]
array LSTY[3] = [0, 0, 0]
array LDRAW[3]
LDRAW[1] = 'set arrow 621 from LX0,LYM to LX1,LYM nohead lw 2.0 lc rgb "#000000"'
LDRAW[2] = 'set arrow 622 from LX0,LYM to LX1,LYM nohead lw 2.0 lc rgb "#0A0AF0"'
LDRAW[3] = 'set arrow 623 from LX0,LYM to LX1,LYM nohead lw 2.0 lc rgb "#E8336D"'
LSAMP = 14; LGAP = 5; LSEP = 16
load "legend.gp"
