# shared helper for page_io / dimension (cairo-made originals)
import sys, math; sys.path.insert(0, '../tools'); import pdfvec, figlib as F
def load(src):
    o = pdfvec.extract(src)
    texts = F.merge_texts(o['texts'])
    frames = []
    for p in o['paths']:
        if p['paint'] == 'S' and len(p['ops']) == 1 and p['ops'][0][0] == 're':
            b = F.bbox(p)
            if b[2]-b[0] > 30 and not any(abs(b[0]-f[0]) < 1 and abs(b[1]-f[1]) < 1 for f in frames):
                frames.append(b)
    frames.sort(key=lambda b: b[0])
    return o, texts, frames
def ticks(o, box):
    """short dark tick marks below/left of the frame"""
    xt, yt = [], []
    for p in o['paths']:
        if p['dash'] or len(p['ops']) != 2: continue
        b = F.bbox(p)
        if abs(b[2]-b[0]) < 0.05 and b[3]-b[1] < 2 and abs(b[3]-box[1]) < 0.3 and box[0]-0.5 <= b[0] <= box[2]+0.5: xt.append(b[0])
        if abs(b[3]-b[1]) < 0.05 and b[2]-b[0] < 2 and abs(b[2]-box[0]) < 0.3 and box[1]-0.5 <= b[1] <= box[3]+0.5: yt.append(b[1])
    return sorted(set(round(v,3) for v in xt)), sorted(set(round(v,3) for v in yt))
def pairs(texts, box, xt, yt):
    px, py = [], []
    for t in texts:
        if t['val'] is None or t['angle']: continue
        c = (t['x']+t['x1'])/2
        if box[1]-14 < t['y'] < box[1] and box[0]-10 < c < box[2]+10 and xt:
            m = min(xt, key=lambda q: abs(q-c))
            if abs(m-c) < 4: px.append((m, t['val']))
        if t['x1'] <= box[0]+1 and box[0]-30 < t['x1'] and yt:
            cy = t['y'] + t['size']*0.33
            m = min(yt, key=lambda q: abs(q-cy))
            if abs(m-cy) < 4: py.append((m, t['val']))
    return sorted(set(px)), sorted(set(py))
TR = {'lin': (lambda v: v, lambda t: t), 'log': (math.log10, lambda t: 10**t),
      'sqrt': (lambda v: -math.sqrt(1-v), lambda t: 1-t*t), 'log1m': (lambda v: -math.log10(1-v), lambda t: 1-10**(-t))}
def fit(pp, kinds):
    best = None
    for k in kinds:
        f, inv = TR[k]
        tv = [(p, f(v)) for p, v in pp]
        n = len(tv); mx = sum(a for a,_ in tv)/n; my = sum(b for _,b in tv)/n
        sl = sum((a-mx)*(b-my) for a,b in tv)/sum((a-mx)**2 for a,_ in tv); ic = my-sl*mx
        rms = math.sqrt(sum(((b-ic)/sl-a)**2 for a,b in tv)/n)
        if best is None or rms < best[1]: best = (k, rms, sl, ic)
    k, rms, sl, ic = best
    return k, rms, (lambda p: TR[k][1](ic+sl*p))
def series(o, box):
    """coloured curves (n>3) and filled marker blobs inside box"""
    cur, mk = [], []
    for p in o['paths']:
        b = F.bbox(p)
        if not b or not (box[0]-3 <= b[0] and b[2] <= box[2]+3 and box[1]-3 <= b[1] and b[3] <= box[3]+3): continue
        col = F.hexc(p['stroke'])
        if max(p['stroke']) - min(p['stroke']) < 0.2: continue   # gray / black furniture
        if p['paint'] == 'S' and len(p['ops']) > 3:
            cur.append((col, bool(p['dash']), F.verts(p)))
        elif p['paint'] in ('B', 'f') and b[2]-b[0] < 5:
            mk.append((col, ((b[0]+b[2])/2, (b[1]+b[3])/2)))
    return cur, mk
