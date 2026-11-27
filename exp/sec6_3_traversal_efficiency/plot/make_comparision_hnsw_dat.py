# Smooth trade-off curves + a subset of measured points for comparision_hnsw.
# Markers: an evenly spaced subset of the measured efSearch sweep (first and last always kept).
# Curve: monotone cubic (Fritsch-Carlson PCHIP) through those markers, parametric in
# t = log(efSearch) for u = -sqrt(1-recall) (the plot's x transform) and log(QPS),
# so it passes through every marker and never turns back in recall or QPS.
import csv, sys
import numpy as np
src = sys.argv[1]
rows = list(csv.DictReader(open(src)))
DS = {'a': 'msmarco_qwen1024', 'b': 'dbpedia_qwen1024'}
M = {'adcbq': 'Calibrated ADC-BQ HNSW + SSD FP32 rerank', 'hnsw': 'Full FP32 HNSW (in-memory)'}
NPTS = {'adcbq': 5, 'hnsw': 6}            # markers per curve
# explicit marker sets (efSearch) overriding the even spacing
PICK = {(m, t): [64, 128, 256, 512, 1024] for m in ('adcbq', 'hnsw') for t in ('a', 'b')}   # same efSearch for both methods
U = lambda r: -np.sqrt(1 - r); R = lambda u: 1 - u * u

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

out = ['# data: recall_qps_t1_from_recall096_rerun.csv (T=1, qps_median)',
       '# curves: PCHIP through the markers (param. log efSearch); pts: measured subset', '']
for m, mname in M.items():
    for t, ds in DS.items():
        pts = sorted((int(r['efSearch']), float(r['recall_at_10']), float(r['qps_median'])) for r in rows
                     if r['dataset'] == ds and r['method'] == mname)
        idx = sorted(set(np.round(np.linspace(0, len(pts) - 1, NPTS[m])).astype(int)))
        if (m, t) in PICK: idx = [i for i, p in enumerate(pts) if p[0] in PICK[(m, t)]]
        sel = [pts[i] for i in idx]
        ef = np.log([p[0] for p in sel]); xs = np.array([p[1] for p in sel]); ys = np.array([p[2] for p in sel])
        assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) < 0), (m, t, sel)
        tt = np.linspace(ef[0], ef[-1], 120)
        cx = R(pchip(ef, U(xs), tt)); cy = np.exp(pchip(ef, np.log(ys), tt))
        out += ['', f'# {m}_{t}_curve'] + [f'{a:.6f} {b:.2f}' for a, b in zip(cx, cy)] + ['']
        out += ['', f'# {m}_{t}_pts'] + [f'{x:.6f} {y:g} {e}' for e, x, y in sel] + ['']
        print(m, t, [p[0] for p in sel], file=sys.stderr)
# same-efSearch pairs (BBQ point -> FP32 point), one segment per ef
SEL = {}
for m, mname in M.items():
    for t, ds in DS.items():
        SEL[(m, t)] = {int(r['efSearch']): (float(r['recall_at_10']), float(r['qps_median'])) for r in rows
                       if r['dataset'] == ds and r['method'] == mname and int(r['efSearch']) in PICK[(m, t)]}
for t in DS:
    out += ['', f'# pair_{t}']
    for e in sorted(SEL[('adcbq', t)]):
        (x1, y1), (x2, y2) = SEL[('adcbq', t)][e], SEL[('hnsw', t)][e]
        out += [f'{x1:.6f} {y1:g} {e}', f'{x2:.6f} {y2:g} {e}', '']
    out += ['']
open('comparision_hnsw.dat', 'w').write('\n'.join(out) + '\n')
