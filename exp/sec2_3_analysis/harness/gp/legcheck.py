#!/usr/bin/env python3
"""Measure a legend strip: for each entry, the key sample's x-range, its text's
x-range, the gap between them, and the gap to the next entry.  Catches the one
thing a canvas-bounds check cannot see -- a label sitting on its neighbour."""
import sys, pymupdf

for f in sys.argv[1:]:
    p = pymupdf.open(f)[0]
    texts = sorted((s["bbox"][0], s["bbox"][2], s["text"])
                   for b in p.get_text("dict")["blocks"]
                   for l in b.get("lines", []) for s in l["spans"])
    # key samples are the only drawings in a legend-only canvas
    marks = sorted((d["rect"].x0, d["rect"].x1) for d in p.get_drawings())
    merged = []
    for x0, x1 in marks:                      # a sample may be several paths
        if merged and x0 - merged[-1][1] < 2.0:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    print(f"{f}  canvas {p.rect.width:.0f}x{p.rect.height:.0f} pt")
    for i, (tx0, tx1, t) in enumerate(texts):
        sam = [m for m in merged if m[1] <= tx0 + 0.5]
        gap = tx0 - sam[-1][1] if sam else float("nan")
        nxt = texts[i+1][0] - tx1 if i + 1 < len(texts) else float("nan")
        nsam = [m for m in merged if m[0] > tx1]
        nxtsam = nsam[0][0] - tx1 if nsam else float("nan")
        flag = "  <-- tight" if (gap < 1.5 or nxtsam < 3.0) else ""
        print(f"   {t:16s} sample->text {gap:5.1f}pt   text->next sample "
              f"{nxtsam:5.1f}pt{flag}")
