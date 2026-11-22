# local helper for line-plot extraction (forked worker; not shared)
import sys, math; sys.path.insert(0, '../tools'); import figlib as F

def segdist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx-ax, by-ay
    L = dx*dx+dy*dy
    t = 0 if L == 0 else max(0, min(1, ((px-ax)*dx+(py-ay)*dy)/L))
    return math.hypot(px-(ax+t*dx), py-(ay+t*dy))

def polydist(p, pts):
    return min(segdist(p, pts[i], pts[i+1]) for i in range(len(pts)-1))

def panel_blocks(ax, names, tag):
    """names: list in series plotting order. Returns blocks [(name_tag_curve, pts), (name_tag_pts, pts)]"""
    ser = ax.series(minpts=3)
    assert len(ser) == len(names), (len(ser), [str(s) for s in ser])
    mk = ax.markers()
    assign = {i: [] for i in range(len(ser))}
    for m in mk:
        cands = [i for i, s in enumerate(ser) if s.color == m.color] or list(range(len(ser)))
        i = min(cands, key=lambda i: polydist(m.px, ser[i].raw))
        assign[i].append((m.x, m.y, m.shape))
    blocks = []
    for i, (n, s) in enumerate(zip(names, ser)):
        blocks.append((f'{n}_{tag}_curve', s.pts))
        blocks.append((f'{n}_{tag}_pts', sorted((x, y) for x, y, _ in assign[i])))
        print(tag, n, s.color, s.dash, 'markers', len(assign[i]), set(sh for *_, sh in assign[i]))
    return blocks

def report(ax, tag):
    for a in 'xy':
        c = ax.cal[a]
        print(tag, a, c['kind'], '%.8g %.8g' % (c['lo'], c['hi']))
