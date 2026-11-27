#!/usr/bin/env python3
"""Flag text that runs outside a panel's canvas, or collides with the y tick
column.  Cheap guard against the clipping that is invisible at paper size."""
import sys, pymupdf

for f in sys.argv[1:]:
    d = pymupdf.open(f); p = d[0]; W, H = p.rect.width, p.rect.height
    bad = []
    for b in p.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                x0, y0, x1, y1 = s["bbox"]
                if x0 < -1.2 or y0 < -1.2 or x1 > W + 1.2 or y1 > H + 1.2:  # tolerate font-box padding
                    bad.append((s["text"], [round(v, 1) for v in s["bbox"]]))
    # the rotated y label must stay left of the y tick column
    spans = [(s["text"], s["bbox"], s["size"])
             for b in p.get_text("dict")["blocks"] for l in b.get("lines", [])
             for s in l["spans"]]
    ylab = [s for s in spans if s[1][3] - s[1][1] > 20 and s[1][0] < W * 0.25]
    ticks = [s for s in spans if s not in ylab and s[1][2] < W * 0.35
             and s[1][3] - s[1][1] < 15 and s[2] >= 7.5]
    for lb in ylab:
        for tk in ticks:
            ox = min(lb[1][2], tk[1][2]) - max(lb[1][0], tk[1][0])
            oy = min(lb[1][3], tk[1][3]) - max(lb[1][1], tk[1][1])
            if ox > 0.3 and oy > 0.3:
                bad.append((f"y label hits tick {tk[0]!r} "
                            f"(overlap {ox:.1f}x{oy:.1f}pt)", []))
    print(("FAIL " if bad else "ok   ") + f"{f}  ({W:.0f}x{H:.0f} pt)")
    for t, bb in bad:
        print(f"       {'out of canvas: ' + repr(t) if bb else t} {bb if bb else ''}")
