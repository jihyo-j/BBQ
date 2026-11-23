# recall_k data (NQ Qwen 1,024, t=1): per k in {30,50,100} and method,
# curve = PCHIP through the marker points (parametric in log efSearch; recall in log(1-r), QPS in log),
# pts = measured values at the marker efSearch (same ef set for all methods within a k).
import csv, sys
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1])))
MARK = {30: [160, 224, 320, 448, 616, 1024], 50: [320, 416, 520, 640, 792, 1024], 100: [512, 608, 712, 816, 936, 1024]}
METH = [("adc_bq_calibrated", "bbq"), ("turboquant_tq1_fp", "tq"), ("rabitq_1bit_adaptive_fp", "rabitq"), ("fp32_hnsw_ram", "fp")]

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

print("# recall_k: efSearch / recall@k / QPS / FP reads per query, NQ Qwen 1,024, t=1, from", sys.argv[1].split('experiment/')[-1])
for k, ks in MARK.items():
    for mk, m in METH:
        sel = sorted((int(r['efSearch']), float(r['recall_at_k']), float(r['qps']), float(r['fp_reads_per_query']))
                     for r in rows if int(r['k']) == k and r['method_key'] == mk and int(r['efSearch']) in ks)
        assert [p[0] for p in sel] == ks, (k, m, sel)
        ef = np.log([p[0] for p in sel]); xs = np.array([p[1] for p in sel]); ys = np.array([p[2] for p in sel])
        tt = np.linspace(ef[0], ef[-1], 120)
        cx = 1 - np.exp(pchip(ef, np.log(1 - xs), tt)); cy = np.exp(pchip(ef, np.log(ys), tt))
        print(f"\n\n# {m}_k{k}_curve")
        for e, a, b in zip(np.exp(tt), cx, cy): print(f"{e:.1f} {a:.6f} {b:.3f}")
        print(f"\n\n# {m}_k{k}_pts")
        for p in sel: print(*p)
