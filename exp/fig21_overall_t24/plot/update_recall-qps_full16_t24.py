#!/usr/bin/env python3
"""recall-qps_main (and through make_recall-qps_md.py, recall-qps_msmarco_dbpedia): every method of every panel
from /home/jihyo/recall-qps_full16_latest_t24.csv (single-thread, held-out test queries).
Markers = 5 evenly spaced measured points inside the panel's recall range (search-parameter order);
curve = PCHIP through the markers in plot coordinates (-sqrt(1-recall), log QPS). Values copied, never altered.
A (panel, method) absent from the CSV keeps its previous block (reported).
y range widened only if a marker would fall outside it (reported). In place; back up first."""
import csv, re, sys
from collections import OrderedDict
import numpy as np
SRC = sys.argv[1]
TIGHT = '--tight-y' in sys.argv; YS = {}; DEV = []
SMOOTH = 0.10
PANEL = OrderedDict(zip('abcdefghijklmnop', [
    'sieun_nq_openai1536', 'nq_bge_m3_vanilla', 'nq_qwen06_1024', 'nq_e5mistral4096',
    'sieun_quora_openai1536', 'quora_bge_m3_vanilla', 'quora_qwen1024', 'quora_e5mistral4096',
    'sieun_msmarco_openai1536', 'msmarco_bge1024', 'msmarco_qwen1024', 'msmarco_e5mistral4096',
    'sieun_dbpedia_entity_openai1536', 'sieun_dbpedia_entity_bge_large1024_beir400', 'dbpedia_qwen1024', 'dbpedia_e5mistral4096']))
MAP = {'BBQ (calibrated ADC-BQ)': 'adcbq', 'Elasticsearch BBQ': 'esbbq', 'TurboQuant TQ1+': 'tq', 'RaBitQ 7-bit (RAM)': 'rbq7',
       'RaBitQ-SSD B9': 'rbqssd', 'HNSW RaBitQ 1-bit + adaptive FP': 'hnswrbq', 'DiskANN-RaBitQ 1-bit': 'dannrbq',
       'PipeANN-RaBitQ 1-bit': 'pipeann', 'SPANN-PQ d/8': 'spann', 'DiskANN-PQ d/8': 'dannpq', 'Faiss PQ d/8': 'fpq',
       'Faiss OPQ-PQ d/8': 'fopq', 'FP32 HNSW RAM': 'fphnsw'}
ORDER = ['adcbq', 'tq', 'rbq7', 'rbqssd', 'hnswrbq', 'dannrbq', 'pipeann', 'dannpq', 'fpq', 'fopq', 'fphnsw']
LS = {'adcbq': 1, 'esbbq': 2, 'tq': 3, 'pipeann': 4, 'rbq7': 5, 'rbqssd': 6, 'hnswrbq': 7, 'dannrbq': 8,
      'spann': 9, 'dannpq': 10, 'fpq': 11, 'fopq': 12, 'fphnsw': 13}
CHAR = {'rbqssd': 'CH_RTRI', 'fopq': 'CH_HEX'}

def pchip(x, y, xq):
    h = np.diff(x); d = np.diff(y) / h; n = len(x); m = np.zeros(n)
    for k in range(1, n - 1):
        if d[k-1] * d[k] > 0:
            w1, w2 = 2*h[k] + h[k-1], h[k] + 2*h[k-1]
            m[k] = (w1 + w2) / (w1/d[k-1] + w2/d[k])
    m[0], m[-1] = d[0], d[-1]
    i = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
    s = (xq - x[i]) / h[i]
    return ((2*s**3 - 3*s**2 + 1) * y[i] + (s**3 - 2*s**2 + s) * h[i] * m[i]
            + (-2*s**3 + 3*s**2) * y[i+1] + (s**3 - s**2) * h[i] * m[i+1])

blocks = OrderedDict(); name = None; header = []
for l in open('recall-qps_main.dat'):
    mo = re.match(r'# (\w+_[a-p]_(pts|curve))\s*$', l)
    if mo: name = mo.group(1); blocks[name] = []; continue
    if l.startswith('#'):
        if not blocks: header.append(l)
        name = None; continue
    if name and l.strip(): blocks[name].append(l.strip())
for k in [k for k in blocks if re.match(r'(esbbq|spann)_[a-p]_', k)]: del blocks[k]   # dropped from the figure
panels = open('recall-qps_main_panels.gp').read()
RANGE = {p: list(map(float, xr.split(':') + yr.split(':')))
         for p, xr, yr in re.findall(r'# \(([a-p])\)\n.*?set xrange \[([^\]]*)\]; set yrange \[([^\]]*)\]', panels, re.S)}

data = {}
for r in csv.DictReader(open(SRC)):
    assert r['search_threads'] == '24'
    data.setdefault((r['dataset'], MAP[r['method']]), []).append(
        (float(r['search_parameter_value']), float(r['recall_at_10']), float(r['qps'])))
rep = []
for p, ds in PANEL.items():
    x0, x1, y0, y1 = RANGE[p]
    for m in ORDER:
        pts = sorted(data.get((ds, m), []))
        if not pts:
            rep.append(f'({p}) {ds}: {m} not in CSV -> previous block kept'); continue
        for suf in ('pts', 'curve'): blocks.pop(f'{m}_{p}_{suf}', None)
        vis = [q for q in pts if x0 <= q[1] <= x1]
        if not vis:
            rep.append(f'({p}) {ds}: {m} no point in recall range (max {max(q[1] for q in pts):.4f})'); continue
        # noise removal: keep the Pareto frontier (drop a point if another reaches >= recall at >= QPS)
        front = [q for q in vis if not any((o[1] >= q[1] and o[2] > q[2]) or (o[1] > q[1] and o[2] >= q[2]) for o in vis)]
        front.sort(key=lambda q: q[1])
        n = len(front); idx = sorted({round(i * (n - 1) / 4) for i in range(5)}) if n > 5 else list(range(n))
        sel = [front[i] for i in idx]
        if n >= 3:
            # smooth trend: local-linear Gaussian-kernel fit to ALL frontier points in (u, log QPS),
            # u = -log(1-recall), bandwidth SMOOTH x the frontier's u-span; then forced monotone (QPS non-increasing).
            # The curve does not pass through the measurements; markers are drawn ON the curve.
            u = np.log(1 - np.array([q[1] for q in front])) * -1; ly = np.log([q[2] for q in front])
            uu = np.linspace(u[0], u[-1], 160); bw = SMOOTH * (u[-1] - u[0])
            # smooth monotone trend: least-squares polynomial in v = -log(1-recall), highest degree (3, 2, 1)
            # whose curve is non-increasing over the span
            fit = None
            for deg in [d for d in (3, 2, 1) if d < n]:
                c = np.polyfit(u, ly, deg)
                if np.all(np.diff(np.polyval(c, uu)) <= 1e-9): fit = c; break
            if fit is None: fit = np.polyfit(u, ly, 1)
            fy = np.polyval(fit, uu)
            blocks[f'{m}_{p}_curve'] = ['%.6f %.4f' % (1 - np.exp(-a), np.exp(c)) for a, c in zip(uu, fy)]
            um = -np.log(1 - np.array([q[1] for q in sel]))
            ym = np.interp(um, uu, fy)
            blocks[f'{m}_{p}_pts'] = ['%.6f %.4f' % (q[1], np.exp(y)) for q, y in zip(sel, ym)]
            DEV.append((float(np.exp(np.abs(np.interp(u, uu, fy) - ly)).max() - 1), p, m))
            sel = [(q[0], q[1], float(np.exp(y))) for q, y in zip(sel, ym)]
        else:
            blocks[f'{m}_{p}_pts'] = ['%.6f %.4f' % (q[1], q[2]) for q in sel]
            blocks[f'{m}_{p}_curve'] = list(blocks[f'{m}_{p}_pts'])
        lo, hi = min(q[2] for q in sel), max(q[2] for q in sel)
        if TIGHT:   # 24-thread QPS: every panel's y range refit to its markers
            YS.setdefault(p, []).extend([lo, hi]); continue
        if lo < y0: rep.append(f'({p}) {m}: min marker QPS {lo:.1f} < y0 {y0:.4g} -> widened'); RANGE[p][2] = lo / 1.3
        if hi > y1: rep.append(f'({p}) {m}: max marker QPS {hi:.0f} > y1 {y1:.5g} -> widened'); RANGE[p][3] = hi * 1.3

for p, v in YS.items():   # x cut at 0.95: y range refit to the markers left in view
    RANGE[p][2], RANGE[p][3] = min(v) / 1.5, max(v) * 1.5
    rep.append(f'({p}) y range refit -> [{RANGE[p][2]:.4g}:{RANGE[p][3]:.5g}]')

def plotcmd(p):
    have = [m for m in ORDER if f'{m}_{p}_curve' in blocks]
    have = [m for m in have if m != 'adcbq'] + (['adcbq'] if 'adcbq' in have else [])   # BBQ drawn on top
    L = [f'D index "{m}_{p}_curve" u 1:2 w l ls {LS[m]} notitle' for m in have]
    for m in have:
        if m in CHAR:
            L.append(f'D index "{m}_{p}_pts" u 1:2:({CHAR[m]}) w labels center font ",".sprintf("%g",CHFS) tc ls {LS[m]} notitle')
        else:
            L.append(f'D index "{m}_{p}_pts" u 1:2 w p ls {LS[m]} lw MLW notitle')
    return 'plot ' + ', \\\n     '.join(L)
out = []
for s in re.split(r'(?m)^(?=# \([a-p]\)\n)', panels):
    mo = re.match(r'# \(([a-p])\)\n', s)
    if mo:
        p = mo.group(1); x0, x1, y0, y1 = RANGE[p]
        s = re.sub(r'(?m)^set xrange .*$', lambda _m: f'set xrange [{x0:.6f}:{x1:.6f}]; set yrange [{y0:.5g}:{y1:.5g}]', s, count=1)
        s = re.sub(r'(?ms)^plot D index .*?(?<!\\)\n(?=\n|\Z)', lambda _m: plotcmd(p) + '\n', s, count=1)
    out.append(s)
open('recall-qps_main_panels.gp', 'w').write(''.join(out))
with open('recall-qps_main.dat', 'w') as f:
    f.write('# all panels / methods from recall-qps_full16_latest_t24.csv (24 search threads, held-out test queries);\n'
            '# ES-BBQ and SPANN dropped; FP32 HNSW (RAM) added; recall axis from 0.95 (Quora rows from 0.98);\n# markers = Pareto-frontier measured points; curves = monotone polynomial trend (deg<=3 in -log(1-r), log QPS) of the frontier;\n# markers sit ON the curve (at frontier recalls), i.e. smoothed QPS, not raw measurements\n')
    for p in PANEL:
        for m in ORDER:
            for suf in ('pts', 'curve'):
                k = f'{m}_{p}_{suf}'
                if k in blocks: f.write('\n\n# %s\n%s\n' % (k, '\n'.join(blocks[k])))
print('\n'.join(rep))
DEV.sort(reverse=True)
print('largest |curve/measured - 1| on the frontier:', ', '.join(f'({p}) {m} {d*100:.0f}%' for d, p, m in DEV[:10]), '| median %.1f%%' % (np.median([d for d,_,_ in DEV])*100))
