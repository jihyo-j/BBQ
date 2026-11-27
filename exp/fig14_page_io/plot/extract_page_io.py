from cairo_extract import *
o, texts, frames = load('../bbq_src/figures/bbq/page_io.pdf')
blocks = []
order = ['bbq', 'diskann', 'rabitq']
fy = None
for tag, box in zip('ab', frames):
    xt, yt = ticks(o, box)
    px, py = pairs(texts, box, xt, yt)
    kx, rx, fx = fit(px, ['lin', 'sqrt', 'log1m'])
    if py: ky, ry, fy = fit(py, ['log'])
    else: ky, ry = 'shared', 0
    print(tag, box, kx, rx, px, ky, ry, 'xrange', fx(box[0]), fx(box[2]), 'yrange', fy(box[1]), fy(box[3]))
    cur, mk = series(o, box)
    for nm, (col, dash, pts) in zip(order, cur):
        blocks.append((f'{nm}_{tag}_curve', [(fx(x), fy(y)) for x, y in pts]))
        print('  curve', col, dash, len(pts))
    for nm, col in zip(order, dict.fromkeys(c for c, _ in mk)):
        blocks.append((f'{nm}_{tag}_pts', sorted((fx(x), fy(y)) for c, (x, y) in mk if c == col)))
        print('  pts', col, len([1 for c,_ in mk if c == col]))
F.write_blocks('page_io.dat', blocks)
