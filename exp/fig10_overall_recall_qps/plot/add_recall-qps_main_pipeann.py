#!/usr/bin/env python3
"""Add PipeANN-RaBitQ1bit (native PipeSearch, RaBitQ 1-bit c=16, T=1) to all 16 panels of recall-qps_main.
argv[1] = spann_pipeann/all_search_results.csv.  Idempotent: removes earlier pipeann blocks/lines first.
Markers = 5 evenly spaced measured points; curve = PCHIP through those markers
(as log QPS vs -sqrt(1-recall), i.e. in the plot's own coordinates) so the line is smooth."""
import csv, re, sys
import numpy as np
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
SRC = sys.argv[1]
PANEL = dict(zip('abcdefghijklmnop', [
    'nq_bge_m3_vanilla', 'nq_qwen06_1024', 'nq_nemotron4096', 'nq_e5mistral4096',
    'quora_bge_m3_vanilla', 'quora_qwen1024', 'quora_nemotron4096', 'quora_e5mistral4096',
    'msmarco_openai1536', 'msmarco_bge1024', 'msmarco_qwen1024', 'msmarco_e5mistral4096',
    'dbpedia_openai1536', 'dbpedia_bge1024', 'dbpedia_qwen1024', 'dbpedia_e5mistral4096']))
rows = [r for r in csv.DictReader(open(SRC)) if r['family'] == 'pipeann_native_rabitq_1bit_c16']
dat = open('recall-qps_main.dat').read()
dat = dat.split('\n\n\n# ---- PipeANN-RaBitQ1bit')[0].rstrip('\n') + '\n'
dat += f'\n\n# ---- PipeANN-RaBitQ1bit (native PipeSearch, RaBitQ 1-bit c=16, T=1) from {SRC.split("experiment/")[-1]}\n'
for p, ds in PANEL.items():
    pts = sorted((int(r['Lsearch']), float(r['recall_at_10']), float(r['qps'])) for r in rows if r['dataset'] == ds)
    n = len(pts); idx = sorted({round(i * (n - 1) / 4) for i in range(5)})
    sel = [pts[i] for i in idx]
    # interpolate in plot coordinates: x' = -sqrt(1-r) (RECALLAXIS), y' = log QPS, as y'(x')
    xs = np.array([s[1] for s in sel]); ys = np.array([s[2] for s in sel])
    u = -np.sqrt(1 - xs); assert np.all(np.diff(u) > 0), (p, sel)
    uu = np.linspace(u[0], u[-1], 160)
    cx = 1 - uu**2; cy = np.exp(pchip(u, np.log(ys), uu))
    dat += f'\n\n# pipeann_{p}_curve\n' + ''.join('%.6f %.4f\n' % (x, y) for x, y in zip(cx, cy))
    dat += f'\n\n# pipeann_{p}_pts\n' + ''.join('%.6f %.4f\n' % pts[i][1:] for i in idx)
    print(p, ds, n, 'recall %.4f-%.4f  qps %.0f-%.0f' % (pts[0][1], pts[-1][1], pts[-1][2], pts[0][2]))
open('recall-qps_main.dat', 'w').write(dat)
pan = open('recall-qps_main_panels.gp').read()
pan = re.sub(r'(?m)^     D index "pipeann_[a-p]_(curve|pts)".*\n', '', pan)
for p in PANEL:
    pan = pan.replace(f'     D index "dannrbq_{p}_curve" u 1:2 w l ls 8 notitle, \\\n',
                      f'     D index "dannrbq_{p}_curve" u 1:2 w l ls 8 notitle, \\\n     D index "pipeann_{p}_curve" u 1:2 w l ls 4 notitle, \\\n')
    pan = pan.replace(f'     D index "dannrbq_{p}_pts" u 1:2 w p ls 8 lw MLW notitle, \\\n',
                      f'     D index "dannrbq_{p}_pts" u 1:2 w p ls 8 lw MLW notitle, \\\n     D index "pipeann_{p}_pts" u 1:2 w p ls 4 lw MLW notitle, \\\n')
    assert pan.count(f'"pipeann_{p}_') == 2, p
open('recall-qps_main_panels.gp', 'w').write(pan)
