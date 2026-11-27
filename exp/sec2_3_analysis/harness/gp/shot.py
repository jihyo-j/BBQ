import sys, pymupdf
# shot.py <pdf> <out.png> [x0 y0 x1 y1 as page fractions] -- 300dpi, no rescale
d = pymupdf.open(sys.argv[1]); p = d[0]; r = p.rect
if len(sys.argv) > 3:
    x0,y0,x1,y1 = [float(v) for v in sys.argv[3:7]]
    clip = pymupdf.Rect(r.x0+r.width*x0, r.y0+r.height*y0, r.x0+r.width*x1, r.y0+r.height*y1)
else:
    clip = None
pix = p.get_pixmap(dpi=300, clip=clip)
pix.save(sys.argv[2]); print(pix.width, pix.height)
