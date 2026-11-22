#!/usr/bin/env python3
"""page_io.dat <- experiment/page_io_text_embedding_3_small/figure/normalized_page_io.csv (argv[1])
   checked against operating_points.csv (argv[2]): every operating point must be a row of the sweep.
curves: monotone PCHIP through the marker points only (as io_ablation / recall_k), parameterised by
        log parameter, in log(1-recall) / log(pages) -- passes through every marker;
pts   : measured points at parameter in MARK (5 per method)."""
import csv, sys, numpy as np
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
rows = list(csv.DictReader(open(sys.argv[1])))
ops = list(csv.DictReader(open(sys.argv[2])))
for o in ops:
    hit = [r for r in rows if r['dataset'] == o['dataset'] and r['method_key'] == o['method_key']
           and r['parameter_value'] == o['parameter_value']]
    assert len(hit) == 1 and abs(float(hit[0]['recall_at_10']) - float(o['measured_recall_at_10'])) < 1e-6 \
        and abs(float(hit[0]['pages_per_query']) - float(o['pages_per_query'])) < 1e-6, o
MARK = {64, 80, 128, 256, 512, 1024}
print('# page_io: recall@10 / pages_per_query, from', sys.argv[1].split('experiment/')[-1])
print('# curves: every measured point (straight segments); pts: parameter in', sorted(MARK))
for ds, tag in [('sieun_msmarco_openai1536', 'a'), ('msmarco_qwen1024', 'b')]:
    for mk, m in [('adcbq', 'bbq'), ('diskann_classic', 'diskann'), ('diskann_rabitq', 'rabitq'), ('turboquant', 'tq')]:
        sel = sorted((int(r['parameter_value']), float(r['recall_at_10']), float(r['pages_per_query']))
                     for r in rows if r['dataset'] == ds and r['method_key'] == mk)
        print(f'\n\n# {m}_{tag}_curve')
        lo0 = min(p for p, _, _ in sel)
        mk_ = [(p, x, y) for p, x, y in sel if p in MARK and not (p == 80 and lo0 == 64)]
        ef = np.log([p for p, _, _ in mk_]); xs = np.array([x for _, x, _ in mk_]); ys = np.array([y for _, _, y in mk_])
        tt = np.linspace(ef[0], ef[-1], 240)
        cx = 1 - np.exp(pchip(ef, np.log(1 - xs), tt)); cy = np.exp(pchip(ef, np.log(ys), tt))
        for x, y in zip(cx, cy): print(f'{x:.6f} {y:.4f}')
        lo = min(p for p, _, _ in sel)
        print(f'\n\n# {m}_{tag}_pts')
        for p, x, y in sel:
            if p in MARK and not (p == 80 and lo == 64): print(f'{x:.6f} {y:.4f}')
