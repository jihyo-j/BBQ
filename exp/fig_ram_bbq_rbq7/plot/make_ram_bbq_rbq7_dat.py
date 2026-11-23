# ram_bbq_rbq7 data: NQ Qwen-1024 (Qwen3-Embedding-0.6B), everything in RAM, 1 search thread.
#   argv[1] = nq_qwen1024_bbq_vs_int16query_nolut_t1/summary_t1.csv  (BBQ: method_key bbq_int16_nibble_lut,
#             calibrated rerank with FP32 vectors in process RAM, 2,452 held-out test queries, QPS = median of 3)
#   argv[2] = nq_quora_search_t1_odirect/rabitq/nq_qwen06_1024/search_results.csv  (RaBitQ 7-bit c16 RAM,
#             family rabitq_b7_c16_ram, all 3,452 queries, QPS = mean of 3 rounds)
# curves = PCHIP through the measured points (parametric in log efSearch; log(1-recall), log QPS); pts = measured.
import csv, sys
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

bbq = sorted((int(r['efSearch']), float(r['recall_at_10']), float(r['qps_median']))
             for r in csv.DictReader(open(sys.argv[1])) if r['method_key'] == 'bbq_int16_nibble_lut')
rbq = sorted((int(r['efSearch']), float(r['recall_at_10']), float(r['qps']))
             for r in csv.DictReader(open(sys.argv[2])) if r['family'] == 'rabitq_b7_c16_ram')
assert len(bbq) == 5 and len(rbq) == 5
print("# ram_bbq_rbq7: efSearch / recall@10 / QPS from", *(a.split('experiment/')[-1] for a in sys.argv[1:3]))
for tag, pts in (("bbq", bbq), ("rbq7", rbq)):
    ef = np.log([p[0] for p in pts]); xs = np.array([p[1] for p in pts]); ys = np.array([p[2] for p in pts])
    assert np.all(np.diff(xs) > 0) and np.all(np.diff(ys) < 0)
    tt = np.linspace(ef[0], ef[-1], 120)
    cx = 1 - np.exp(pchip(ef, np.log(1 - xs), tt)); cy = np.exp(pchip(ef, np.log(ys), tt))
    print(f"\n\n# {tag}_curve")
    for e, a, b in zip(np.exp(tt), cx, cy): print(f"{e:.1f} {a:.6f} {b:.2f}")
    print(f"\n\n# {tag}_pts")
    for e, a, b in pts: print(e, a, b)
