# io_ablation.dat <- argv[1] (Qwen: experiment/bbq_calibrated_io_modes_t1/summary_t1.csv)
#                  + argv[2] (OpenAI: experiment/bbq_calibrated_io_modes_t1_openai_priority_20261121/summary_t1.csv)
# (t=1, qps_median)
# curves: PCHIP through the marker points only (param. log efSearch); pts: measured (MARK)
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
SRC = {'msmarco_qwen1024': sys.argv[1], 'sieun_msmarco_openai1536': sys.argv[2] if len(sys.argv) > 2 else sys.argv[1]}
ROWS = {k: list(csv.DictReader(open(v))) for k, v in SRC.items()}
MARK = {64, 96, 128, 192, 256, 512, 1024}
print("# io_ablation: efSearch / recall@10 / QPS(median, t=1); Qwen from", SRC['msmarco_qwen1024'].split('experiment/')[-1], "; OpenAI from", SRC['sieun_msmarco_openai1536'].split('experiment/')[-1])
print("# curves: PCHIP through the marker points; pts: measured, ef in", sorted(MARK))
for ds, tag in [("msmarco_qwen1024", "a"), ("sieun_msmarco_openai1536", "b")]:
    for mk, m in [("sequential", "seq"), ("async_batch", "async"), ("async_batch_overlap", "overlap")]:
        sel = sorted((int(r['efSearch']), float(r['recall_at_10']), float(r['qps_median'])) for r in ROWS[ds]
                     if r['dataset'] == ds and r['method_key'] == mk and int(r['efSearch']) in MARK)
        ef = np.log([p[0] for p in sel]); xs = np.array([p[1] for p in sel]); ys = np.array([p[2] for p in sel])
        tt = np.linspace(ef[0], ef[-1], 240)
        cx = 1 - np.exp(pchip(ef, np.log(1 - xs), tt)); cy = np.exp(pchip(ef, np.log(ys), tt))
        print(f"\n\n# {m}_{tag}_curve")
        for e, a, b in zip(np.exp(tt), cx, cy): print(f"{e:.1f} {a:.6f} {b:.2f}")
        print(f"\n\n# {m}_{tag}_pts")
        for e, a, b in sel: print(e, a, b)
