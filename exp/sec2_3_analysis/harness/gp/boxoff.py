#!/usr/bin/env python3
"""How far right of its own canvas centre does each panel's plot box sit?

A panel reserves 0.41in on the left for the rotated y label and the widest
tick label, and 0.09in on the right for the last x tick label, so the box --
the thing the eye reads as "the panel" -- is not centred on the PDF canvas.
A subcaption centred on the canvas therefore sits left of the box.  This
prints the offset each subcaption has to be shifted by, in the points it will
be printed at, which is what install_figs.py hard-codes.

    python3 boxoff.py
"""
import pymupdf, numpy as np, sys

# panel -> width it is printed at, in points (acmart sigconf)
TW, CW = 506.295, 241.14
PRINT = {"f1a": 0.49*CW, "f1b": 0.49*CW, "f2a": 0.49*CW, "f2b": 0.49*CW,
         "f3a": 0.245*TW, "f3b": 0.245*TW, "f3c": 0.245*TW, "f3d": 0.245*TW,
         "f4a": 0.49*CW, "f4b": 0.49*CW}

def box(f):
    """Plot-box x0,x1 in canvas points, from the rendered pixels rather than
    the drawing list: f1 keeps the original design's `border 3` (so it has no
    right border to find) and f2a is a crop of the supplied artwork with ink
    outside its own MediaBox.  What every panel does have is the bottom axis
    line, and it is the longest horizontal run of dark pixels on the canvas."""
    p = pymupdf.open(f + ".pdf")[0]
    W, Z = p.rect.width, 8
    pm = p.get_pixmap(matrix=pymupdf.Matrix(Z, Z), colorspace="gray")
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width)
    best = (0, 0, 0)
    for row in (a < 100):
        x = 0
        while x < len(row):
            if row[x]:
                x0 = x
                while x < len(row) and row[x]:
                    x += 1
                if x - x0 > best[0]:
                    best = (x - x0, x0, x)
            x += 1
    if best[0] < 0.3*len(row):
        raise SystemExit(f + ": no axis line found")
    return best[1]/Z, best[2]/Z, W

print(" panel   box (canvas pt)   canvas    scale   shift to apply")
for f, pw in PRINT.items():
    x0, x1, W = box(f)
    off = ((x0 + x1)/2 - W/2) * pw/W
    print(f"  {f}   {x0:6.1f}-{x1:6.1f} {W:9.1f} {pw/W:8.4f}   {off:+6.1f}pt"
          f"   (margin={2*off:.0f}pt)")
