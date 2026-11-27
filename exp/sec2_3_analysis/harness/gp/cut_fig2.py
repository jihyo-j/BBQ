#!/usr/bin/env python3
"""Figure 2 is the original artwork, cut up -- not redrawn.

The supplied capacity panel carries its own key along the bottom edge.  That
band is cut off here and the panel padded back to its original height, so the
two panels still line up at equal printed width; fig2leg.gp then redraws the
key at a readable size in the same colours.  Nothing inside either plot is
touched.
"""
import pymupdf

OLD = "/home/sieun/Blessing BQ/figures-v2/old"
GP  = "/home/sieun/projects/BBQETA/gp"
CUT = 105.2          # below the x-axis label, above the key
BAND = 104.8         # top of the key band, 0.4pt of shared descent space
COLW = 3.3437 * 72   # \columnwidth, for the centred strip

src = pymupdf.open(f"{OLD}/capacity4096_crop.pdf")
W, H = src[0].rect.width, src[0].rect.height

# panel (a): the plot, padded back to the original height
out = pymupdf.open()
pg = out.new_page(width=W, height=H)
pg.show_pdf_page(pymupdf.Rect(0, 0, W, CUT), src, 0,
                 clip=pymupdf.Rect(0, 0, W, CUT))
out.save(f"{GP}/f2a.pdf")

# Panel (b) is the same artwork, re-rendered from its own motiv/mrl_scoreerr.gp
# rather than copied: the supplied PDF was made before Liberation Sans existed
# on this machine, so it carries the DejaVu fallback while every other panel
# now gets the face the scripts actually ask for.  Nothing else about it moves.
import subprocess
subprocess.run(["gnuplot", "mrl_scoreerr.gp"], cwd=f"{GP}/motiv", check=True)
pymupdf.open(f"{GP}/motiv/mrl_scoreerr.pdf").save(f"{GP}/f2b.pdf")

print(f"f2a {W:.1f}x{H:.1f}pt (key band below y={CUT} cut off)   "
      f"f2b from mrl_scoreerr.pdf")
