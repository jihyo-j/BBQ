#!/usr/bin/env python3
"""Search (font, samplen, width) for a legend strip that stays one row, keeps
every label clear of the next sample, and fits the canvas; then centre it."""
import re, subprocess, sys, pymupdf, pathlib

def geom(pdf):
    p = pymupdf.open(pdf)[0]
    texts = sorted((s["bbox"][0], s["bbox"][2], s["text"])
                   for b in p.get_text("dict")["blocks"]
                   for l in b.get("lines", []) for s in l["spans"])
    marks = sorted((d["rect"].x0, d["rect"].x1) for d in p.get_drawings())
    merged = []
    for x0, x1 in marks:
        if merged and x0 - merged[-1][1] < 2.0: merged[-1][1] = max(merged[-1][1], x1)
        else: merged.append([x0, x1])
    gaps = []
    for i, (tx0, tx1, t) in enumerate(texts):
        nsam = [m for m in merged if m[0] > tx1 - 0.3]
        if i + 1 < len(texts):
            gaps.append(nsam[0][0] - tx1 if nsam else -99.0)
    lo = min([m[0] for m in merged] + [t[0] for t in texts])
    hi = max([m[1] for m in merged] + [t[1] for t in texts])
    return len(texts), gaps, lo, hi, p.rect.width

def run(gp): subprocess.run(["gnuplot", gp], check=True)

def tune(gp, pdf, nentries, key_re, line_tmpl, grid):
    src = pathlib.Path(gp); orig = src.read_text()
    best = None
    for font, samplen, width in grid:
        src.write_text(re.sub(key_re, line_tmpl.format(s=samplen, w=width, f=font), orig))
        run(gp); n, gaps, lo, hi, W = geom(pdf)
        ok = n == nentries and (not gaps or min(gaps) >= 3.0) and hi <= W and lo >= 0
        print(f"  font {font} samplen {samplen} width {width}: {n} entries, "
              f"min gap {min(gaps) if gaps else 0:.1f}, span {lo:.0f}..{hi:.0f}/{W:.0f} "
              f"{'OK' if ok else ''}")
        if ok and best is None: best = (font, samplen, width)
    if best is None:
        src.write_text(orig); print("  no combination worked"); return None
    font, samplen, width = best
    src.write_text(re.sub(key_re, line_tmpl.format(s=samplen, w=width, f=font), orig))
    run(gp); n, gaps, lo, hi, W = geom(pdf)
    print(f"  chosen font {font} samplen {samplen} width {width}; "
          f"ink centre {(lo+hi)/2:.1f} want {W/2:.1f}")
    return best, (lo + hi) / 2 - W / 2
