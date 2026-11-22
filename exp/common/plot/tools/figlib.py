#!/usr/bin/env python3
"""Helpers to recover plotted data from matplotlib/cairo PDF figures.

    import figlib as F
    fig = F.Fig('x.pdf')
    for ax in fig.axes:            # one per panel (frame rectangle)
        print(ax)                  # box, x/y axis calibration, labels
        for s in ax.series():      # polylines inside the panel
            ...  s.color, s.lw, s.dash, s.pts (data coords)
        for m in ax.markers():     # scatter / line markers
            ...  m.name, m.color, m.fill, m.x, m.y (data coords)
        ax.bars()                  # filled rectangles -> (x0,x1,y0,y1 data), color, hatch
        ax.texts()                 # annotations inside the panel (data coords)

Axis calibration uses tick marks on the frame + the tick label next to
each; scale kind is chosen among lin / log10 / sqrtrecall (x' = -sqrt(1-x))
by least squares.  Override with fig.axes[i].calibrate('x', kind='log').
"""
import math, re, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pdfvec

# ---------------- text width (Helvetica AFM) ----------------
_W = {}
for _l in open(os.path.join(os.path.dirname(__file__), 'Helvetica.afm'), encoding='latin1'):
    _m = re.match(r'C (-?\d+) ; WX (\d+) ; N (\S+)', _l)
    if _m: _W[_m.group(3)] = int(_m.group(2))
_N = {' ': 'space', '.': 'period', ',': 'comma', '-': 'hyphen', '+': 'plus', '(': 'parenleft', ')': 'parenright',
      '/': 'slash', ':': 'colon', '@': 'at', '%': 'percent', '=': 'equal', '0': 'zero', '1': 'one', '2': 'two',
      '3': 'three', '4': 'four', '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', '−': 'minus'}
def textw(s, size):
    return sum(_W.get(_N.get(c, c), 556) for c in s) * size / 1000.0

def num(s):
    """parse tick label -> float (handles 1,000  1k  2.5M  −3  10^2 already merged as 1e2)"""
    s = s.strip().replace('−', '-').replace(',', '').replace(' ', '')
    mult = 1.0
    if s and s[-1] in 'kKMGB':
        mult = {'k': 1e3, 'K': 1e3, 'M': 1e6, 'G': 1e9, 'B': 1e9}[s[-1]]; s = s[:-1]
    s = s.rstrip('%')
    try:
        return float(s) * mult
    except ValueError:
        return None

TRANSFORMS = {
    'lin': (lambda v: v, lambda t: t),
    'log': (lambda v: math.log10(v), lambda t: 10 ** t),
    'sqrtrecall': (lambda v: -math.sqrt(max(1 - v, 0)), lambda t: 1 - t * t),
}

def bbox(p):
    xs = []; ys = []
    for op in p['ops']:
        if op[0] == 're':
            xs += [op[1], op[1] + op[3]]; ys += [op[2], op[2] + op[4]]
        elif op[0] != 'h':
            xs += op[1::2]; ys += op[2::2]
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None

def verts(p):
    pts = []
    for op in p['ops']:
        if op[0] in ('m', 'l'):
            pts.append((op[1], op[2]))
        elif op[0] == 'c':
            pts.append((op[5], op[6]))
        elif op[0] == 'h' and pts:
            pass
    return pts

def merge_texts(texts):
    """merge mathtext fragments (e.g. '1','0' + superscript '2') into tokens"""
    ts = sorted(texts, key=lambda t: (round(t['y']), t['x']))
    out = []
    used = [False] * len(ts)
    for i, t in enumerate(ts):
        if used[i]:
            continue
        grp = [t]; used[i] = True
        for j, u in enumerate(ts):
            if used[j] or abs(u['angle'] - t['angle']) > 1:
                continue
            last = grp[-1]
            lw = textw(last['s'], last['size'])
            if t['angle'] == 0 and abs(u['y'] - t['y']) < t['size'] * 0.7 and -0.5 < u['x'] - (last['x'] + lw) < t['size'] * 0.35:
                grp.append(u); used[j] = True
        base = [g for g in grp if abs(g['size'] - t['size']) < 0.3 and abs(g['y'] - t['y']) < 0.5]
        sup = [g for g in grp if g not in base and g['y'] > t['y'] + 0.5]
        s = ''.join(g['s'] for g in base)
        val = None
        if sup and s.replace('×', '') in ('10',):
            e = num(''.join(g['s'] for g in sup))
            if e is not None:
                val = 10 ** e
                s = '10^{' + ''.join(g['s'] for g in sup) + '}'
        elif sup:
            s = s + '^{' + ''.join(g['s'] for g in sup) + '}'
        x1 = max(g['x'] + textw(g['s'], g['size']) for g in grp)
        out.append({'s': s, 'x': t['x'], 'x1': x1, 'y': t['y'], 'size': t['size'], 'angle': t['angle'],
                    'fill': t['fill'], 'val': val if val is not None else num(s)})
    return out

class Axis:
    def __init__(self, fig, box):
        self.fig = fig
        self.box = box  # x0,y0,x1,y1 (pt)
        self.cal = {}
        self.ticks = {'x': [], 'y': []}
        self.labels = {}
        self._find_ticks()
        for a in ('x', 'y'):
            try:
                self.calibrate(a)
            except Exception as e:
                self.cal[a] = None

    def __repr__(self):
        b = self.box
        s = f'Axis box=({b[0]:.1f},{b[1]:.1f})-({b[2]:.1f},{b[3]:.1f}) {b[2]-b[0]:.1f}x{b[3]-b[1]:.1f}pt'
        for a in ('x', 'y'):
            c = self.cal.get(a)
            if c:
                s += f'\n  {a}: {c["kind"]} range [{c["lo"]:.6g}, {c["hi"]:.6g}] ticks {[(t[1]) for t in c["pairs"]]} rms={c["rms"]:.3g}'
            else:
                s += f'\n  {a}: UNCALIBRATED'
        for k, v in self.labels.items():
            s += f'\n  {k}: {v!r}'
        return s

    def _find_ticks(self):
        x0, y0, x1, y1 = self.box
        tol = 1.0
        for p in self.fig.o['paths']:
            if p['paint'] not in ('S', 'B', 'b') or len(p['ops']) != 2:
                continue
            b = bbox(p)
            if b is None:
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            if w < 0.05 and 0.5 < h < 8 and x0 - tol <= b[0] <= x1 + tol:
                if abs(b[1] - y0) < tol or abs(b[3] - y0) < tol:
                    self.ticks['x'].append((b[0], h))
            if h < 0.05 and 0.5 < w < 8 and y0 - tol <= b[1] <= y1 + tol:
                if abs(b[0] - x0) < tol or abs(b[2] - x0) < tol:
                    self.ticks['y'].append((b[1], w))
        for a in ('x', 'y'):
            t = sorted(set((round(p, 2), round(l, 2)) for p, l in self.ticks[a]))
            self.ticks[a] = t

    def tick_labels(self, a):
        x0, y0, x1, y1 = self.box
        res = []
        for t in self.fig.mtexts:
            if t['val'] is None or t['angle'] != 0:
                continue
            if a == 'x' and y0 - 3 * t['size'] < t['y'] < y0 and x0 - 15 < (t['x'] + t['x1']) / 2 < x1 + 15:
                res.append(((t['x'] + t['x1']) / 2, t['val'], t['s']))
            if a == 'y' and x0 - 40 < t['x1'] <= x0 + 1 and y0 - 6 < t['y'] < y1 + 6:
                res.append((t['y'] + t['size'] * 0.33, t['val'], t['s']))
        return res

    def calibrate(self, a, kind=None, pairs=None):
        if pairs is None:
            labs = self.tick_labels(a)
            major = [p for p, l in self.ticks[a]]
            if self.ticks[a]:
                L = max(l for p, l in self.ticks[a])
                major = sorted(set(p for p, l in self.ticks[a] if l > 0.75 * L))
            pairs = []
            for pos, val, s in labs:
                if major:
                    m = min(major, key=lambda q: abs(q - pos))
                    if abs(m - pos) < 6:
                        pairs.append((m, val))
                        continue
                pairs.append((pos, val))
            pairs = sorted(set(pairs))
        best = None
        for k in ([kind] if kind else ['lin', 'log', 'sqrtrecall']):
            f, inv = TRANSFORMS[k]
            try:
                tv = [(p, f(v)) for p, v in pairs]
            except (ValueError, ZeroDivisionError):
                continue
            if len(tv) < 2:
                continue
            n = len(tv); mx = sum(p for p, _ in tv) / n; my = sum(t for _, t in tv) / n
            sxx = sum((p - mx) ** 2 for p, _ in tv)
            if sxx == 0:
                continue
            slope = sum((p - mx) * (t - my) for p, t in tv) / sxx
            icpt = my - slope * mx
            if slope == 0:
                continue
            rms = math.sqrt(sum(((t - icpt) / slope - p) ** 2 for p, t in tv) / n)   # in pt
            pow10 = all(v > 0 and abs(math.log10(v) - round(math.log10(v))) < 1e-9 for _, v in pairs)
            tie = best is not None and abs(rms - best['rms']) <= 0.05
            if best is None or rms < best['rms'] - 0.05 or (tie and k == 'log' and pow10 and len(pairs) <= 3):
                best = {'kind': k, 'slope': slope, 'icpt': icpt, 'rms': rms, 'pairs': pairs}
        if best is None:
            raise ValueError('cannot calibrate ' + a)
        f, inv = TRANSFORMS[best['kind']]
        lo_p, hi_p = (self.box[0], self.box[2]) if a == 'x' else (self.box[1], self.box[3])
        best['lo'] = inv(best['icpt'] + best['slope'] * lo_p)
        best['hi'] = inv(best['icpt'] + best['slope'] * hi_p)
        self.cal[a] = best
        return best

    def to_data(self, a, p):
        c = self.cal[a]
        return TRANSFORMS[c['kind']][1](c['icpt'] + c['slope'] * p)

    def xy(self, x, y):
        return self.to_data('x', x), self.to_data('y', y)

    def inside(self, x, y, pad=0.5):
        b = self.box
        return b[0] - pad <= x <= b[2] + pad and b[1] - pad <= y <= b[3] + pad

    def series(self, minpts=2):
        """stroked polylines whose clip box is this panel (data curves)"""
        out = []
        for p in self.fig.o['paths']:
            if p['paint'] not in ('S', 's', 'B'):
                continue
            c = p['clip']
            if not c or abs(c[0] - self.box[0]) > 1 or abs(c[2] - self.box[2]) > 1 or abs(c[1] - self.box[1]) > 1:
                # cairo/gnuplot figures have no clip: accept if all vertices inside
                pts = verts(p)
                if c or not pts or not all(self.inside(*q, pad=2) for q in pts):
                    continue
            pts = verts(p)
            if len(pts) < minpts:
                continue
            if self._is_axis_furniture(pts):
                continue
            out.append(Series(self, p, pts))
        return out

    def _is_axis_furniture(self, pts):
        x0, y0, x1, y1 = self.box
        if len(pts) == 2:
            (ax_, ay), (bx, by) = pts
            on_v = abs(ax_ - bx) < 0.05 and (min(abs(ax_ - x0), abs(ax_ - x1)) < 0.5 or abs(ay - by) < 8 and (min(abs(ay - y0), abs(ay - y1), abs(by - y0), abs(by - y1)) < 0.5))
            on_h = abs(ay - by) < 0.05 and (min(abs(ay - y0), abs(ay - y1)) < 0.5 or abs(ax_ - bx) < 8 and (min(abs(ax_ - x0), abs(ax_ - x1), abs(bx - x0), abs(bx - x1)) < 0.5))
            return on_v or on_h
        return False

    def markers(self):
        out = []
        for m in self.fig.o['marks']:
            if self.inside(m['x'], m['y'], pad=3):
                out.append(Marker(self, m))
        return out

    def bars(self):
        out = []
        for p in self.fig.o['paths']:
            if p['paint'] not in ('f', 'f*', 'B', 'B*'):
                continue
            b = bbox(p)
            if not b or not self.inside(b[0], b[1], 1) or not self.inside(b[2], b[3], 1):
                continue
            if b[2] - b[0] > (self.box[2] - self.box[0]) * 0.95 and b[3] - b[1] > (self.box[3] - self.box[1]) * 0.95:
                continue  # axes background
            if len(p['ops']) > 6 or (len(p['ops']) < 4 and p['ops'][0][0] != 're'):
                continue
            out.append({'x0': self.to_data('x', b[0]) if self.cal.get('x') else b[0],
                        'x1': self.to_data('x', b[2]) if self.cal.get('x') else b[2],
                        'y0': self.to_data('y', b[1]) if self.cal.get('y') else b[1],
                        'y1': self.to_data('y', b[3]) if self.cal.get('y') else b[3],
                        'pt': b, 'fill': hexc(p['fill']), 'stroke': hexc(p['stroke']), 'hatch': p['hatch'],
                        'falpha': p['falpha'], 'paint': p['paint']})
        return out

    def texts(self):
        """text inside (or just above) the panel, excluding tick labels"""
        b = self.box
        return [t for t in self.fig.mtexts if b[0] < t['x'] < b[2] and b[1] < t['y'] < b[3]]

class Series:
    def __init__(self, ax, p, pts):
        self.raw = pts
        self.color = hexc(p['stroke'])
        self.lw = p['lw']
        self.dash = p['dash']
        self.alpha = p['salpha']
        self.pts = [ax.xy(x, y) for x, y in pts] if ax.cal.get('x') and ax.cal.get('y') else pts
    def __repr__(self):
        return f'Series {self.color} lw={self.lw:.2f} dash={self.dash} n={len(self.pts)} first={self.pts[0]} last={self.pts[-1]}'

def shape_sig(shape):
    """rough marker shape name from its path"""
    if not shape:
        return '?'
    p = shape[0]
    ops = [o[0] for o in p['ops']]
    if 'c' in ops:
        return 'circle'
    pts = verts(p)
    n = len(set((round(x, 1), round(y, 1)) for x, y in pts))
    b = bbox(p)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    if n == 4:
        if any(abs(x - cx) < 0.1 for x, y in pts):
            return 'diamond'
        return 'square'
    if n == 3:
        ys = sorted(y for x, y in pts); xs = sorted(x for x, y in pts)
        if abs(ys[0] - ys[1]) < 0.1:
            return 'triangle_up'
        if abs(ys[1] - ys[2]) < 0.1:
            return 'triangle_down'
        if abs(xs[0] - xs[1]) < 0.1:
            return 'triangle_right'
        return 'triangle_left'
    if n == 5:
        return 'pentagon'
    if n == 6:
        return 'hexagon'
    if n == 10:
        return 'star'
    if n == 2 or len(shape) == 2:
        return 'x' if len(shape) == 2 else 'line'
    return f'poly{n}'

class Marker:
    def __init__(self, ax, m):
        self.name = m['name']
        self.color = hexc(m['stroke'])
        self.fill = hexc(m['fill'])
        self.falpha = m['falpha']
        self.salpha = m['salpha']
        self.shape = shape_sig(m['shape'])
        self.paint = m['shape'][0]['paint'] if m['shape'] else '?'
        self.size = 0
        if m['shape']:
            b = bbox(m['shape'][0])
            self.size = max(b[2] - b[0], b[3] - b[1]) if b else 0
        self.px = (m['x'], m['y'])
        self.x, self.y = ax.xy(m['x'], m['y']) if ax.cal.get('x') and ax.cal.get('y') else (m['x'], m['y'])
    def __repr__(self):
        return f'Marker {self.shape} {self.color} fill={self.fill}/{self.falpha} paint={self.paint} size={self.size:.1f} ({self.x:.6g},{self.y:.6g})'

def hexc(c):
    return '#%02X%02X%02X' % tuple(int(round(max(0, min(1, v)) * 255)) for v in c[:3])

class Fig:
    def __init__(self, fn):
        self.fn = fn
        self.o = pdfvec.extract(fn)
        self.W, self.H = self.o['page']
        self.mtexts = merge_texts(self.o['texts'])
        self.axes = [Axis(self, b) for b in self._frames()]

    def _frames(self):
        cand = []
        # matplotlib: axes background rect ('f', white, inside page)
        for p in self.o['paths']:
            b = bbox(p)
            if not b:
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            if w > self.W * 0.95 and h > self.H * 0.95:
                continue
            if p['paint'] in ('f', 'B') and w > 30 and h > 30 and len(p['ops']) <= 6 and p['fill'] == [1.0, 1.0, 1.0]:
                cand.append(b)
        if not cand:
            # cairo: closed stroked rectangles drawn as frame
            for p in self.o['paths']:
                b = bbox(p)
                if not b:
                    continue
                w, h = b[2] - b[0], b[3] - b[1]
                if p['paint'] in ('S', 's') and w > 30 and h > 30 and len(p['ops']) in (4, 5, 6) and p['stroke'][:3] in ([0, 0, 0], [0.0, 0.0, 0.0]) or (p['paint'] == 'S' and len(p['ops']) == 1 and p['ops'][0][0] == 're' and w > 30 and h > 30):
                    cand.append(b)
        # dedupe
        out = []
        for b in cand:
            if not any(abs(b[0] - c[0]) < 1 and abs(b[1] - c[1]) < 1 and abs(b[2] - c[2]) < 1 and abs(b[3] - c[3]) < 1 for c in out):
                out.append(b)
        out.sort(key=lambda b: (-round(b[3] / 10), b[0]))
        return out

    def outside_texts(self):
        """texts not inside any panel (titles, xlabels, legend labels...)"""
        res = []
        for t in self.mtexts:
            if not any(a.inside(t['x'], t['y'], 0) for a in self.axes):
                res.append(t)
        return res

    def legend_items(self):
        """paths/markers outside all panels (legend samples)"""
        ps = [p for p in self.o['paths'] if (lambda b: b and not any(a.inside((b[0]+b[2])/2, (b[1]+b[3])/2, 0) for a in self.axes))(bbox(p))]
        return ps

if __name__ == '__main__':
    f = Fig(sys.argv[1])
    print(f'{f.fn}: page {f.W:.1f}x{f.H:.1f}pt, {len(f.axes)} panels')
    for i, a in enumerate(f.axes):
        print(f'[{i}]', a)
        for s in a.series():
            print('   ', s)
        from collections import Counter
        print('    markers:', Counter((m.shape, m.color, m.fill, round(m.falpha, 2)) for m in a.markers()))
        bs = a.bars()
        if bs:
            print('    bars:', len(bs))
    print('outside texts:')
    for t in f.outside_texts():
        print(f"   {t['s']!r} x={t['x']:.1f}-{t['x1']:.1f} y={t['y']:.1f} size={t['size']} ang={t['angle']}")

def write_blocks(fname, blocks, fmt='%.8g'):
    """blocks: list of (name, rows) ; rows = list of tuples.  Each block is
    preceded by '# name' so gnuplot can select it with  index "name"."""
    with open(fname, 'w') as f:
        f.write('# data recovered from the original matplotlib PDF (vector coordinates)\n')
        for name, rows in blocks:
            f.write(f'\n\n# {name}\n')
            for r in rows:
                f.write(' '.join((fmt % v) if isinstance(v, (int, float)) else str(v) for v in r) + '\n')
