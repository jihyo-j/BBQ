#!/usr/bin/env python3
"""Did the subcaptions end up under their plot boxes?

Measures, in the built page, each panel's bottom axis line against the centre
of the subcaption below it.  Run after install_figs.py + preview.sh; a residual
of a point or two is invisible, more than that reads as a shifted row.

    python3 align.py preview/fig3_inpaper.pdf
"""
import pymupdf, numpy as np, sys, re

doc = pymupdf.open(sys.argv[1])
for pno, p in enumerate(doc):
    caps = []
    for b in p.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            t = "".join(s["text"] for s in l["spans"]).strip()
            if re.match(r"^\([a-d]\) ", t):
                caps.append((t, l["bbox"]))
    if not caps:
        continue
    caps.sort(key=lambda c: c[1][0])
    Z = 6
    pm = p.get_pixmap(matrix=pymupdf.Matrix(Z, Z), colorspace="gray")
    a = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width)
    dark = a < 110
    ytop = min(c[1][1] for c in caps)
    # only the band directly above the captions: a page can carry a second
    # float whose rules would otherwise be mistaken for one of these panels
    runs = []
    for y in range(max(0, int((ytop - 140)*Z)), int((ytop - 4)*Z)):
        row = dark[y]
        x = 0
        while x < len(row):
            if row[x]:
                x0 = x
                while x < len(row) and row[x]:
                    x += 1
                if 55*Z < x - x0 < 230*Z:
                    runs.append((x0/Z, x/Z))
            x += 1
    print("%s page %d" % (sys.argv[1], pno))
    print("  caption                        box x0-x1      box cen  cap cen  resid")
    for t, bb in caps:
        cc = (bb[0] + bb[2])/2
        near = [r for r in runs if r[0] < cc + 70 and r[1] > cc - 70]
        if not near:
            print("  %-30s  -- no axis line --" % t[:30]); continue
        x0 = min(r[0] for r in near); x1 = max(r[1] for r in near)
        bc = (x0 + x1)/2
        print("  %-30s %6.1f-%6.1f %9.1f %8.1f %+7.1fpt"
              % (t[:30], x0, x1, bc, cc, bc - cc))
