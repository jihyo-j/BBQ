# recall_qps_bbq_tq data: curves = PCHIP through the measured marker points
# (parametric in log efSearch; recall in -log(1-r), QPS in log), pts = measured values at those ef.
import csv, sys
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
MARK = [80, 128, 256, 512, 1024]

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

print("# recall_qps_bbq_tq: efSearch / recall@10 / QPS (t=1) from", sys.argv[1].split('experiment/')[-1])
print("# curves: PCHIP through the marker points (ef", MARK, "); pts: measured")
for ds, tag in [("nq_qwen06_1024", "nq"), ("msmarco_qwen1024", "ms")]:
    for mk, m in [("adc_bq_calibrated", "bbq"), ("turboquant_tq1_calibrated", "tq")]:
        sel = sorted((int(r['efSearch']), float(r['recall_at_10']), float(r['qps'])) for r in rows
                     if r['dataset'] == ds and r['method_key'] == mk and int(r['efSearch']) in MARK)
        ef = np.log([p[0] for p in sel]); xs = np.array([p[1] for p in sel]); ys = np.array([p[2] for p in sel])
        assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) < 0), sel
        tt = np.linspace(ef[0], ef[-1], 120)
        cx = 1 - np.exp(pchip(ef, np.log(1 - xs), tt)); cy = np.exp(pchip(ef, np.log(ys), tt))
        print(f"\n\n# {m}_{tag}_curve")
        for e, a, b in zip(np.exp(tt), cx, cy): print(f"{e:.1f} {a:.6f} {b:.2f}")
        print(f"\n\n# {m}_{tag}_pts")
        for e, a, b in sel: print(e, a, b)
