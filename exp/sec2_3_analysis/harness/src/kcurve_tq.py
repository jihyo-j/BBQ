"""Add TQ-1bit to an existing kcurve table.

The Figure 3 quantizer panels have to carry the same six codes so that one
legend serves both; kcurve.py measured five.  This measures the sixth on an
existing corpus and rewrites the table with it inserted in the order the
legend uses (Sym-BQ, ADC-BQ, TQ-1bit, RaBitQ, PQ, OPQ).

  BBQETA_THREADS=24 python3 src/kcurve_tq.py msmarco_qw06
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, CONFIGS, OUT
import quantizers as QZ
from kcurve import KMAX, NQ

key = sys.argv[1] if len(sys.argv) > 1 else "msmarco_qw06"
X, Q, _ = load(key, NQ)
N = len(X)
ex = np.zeros((NQ, KMAX), dtype=np.int64)
for qi in range(NQ):
    t = X @ Q[qi]
    part = np.argpartition(-t, KMAX)[:KMAX]
    ex[qi] = part[np.argsort(-t[part], kind="stable")]

Qt, Xh, _, _ = QZ.build("tq1", X, Q, log=lambda *a: None)
R = np.zeros((NQ, KMAX), dtype=np.int64)
rs = np.random.default_rng(0).choice(N, 50_000, replace=False)
cq = []
for qi in range(NQ):
    a = Xh @ Qt[qi]
    order = np.argsort(-a, kind="stable")
    pos = np.empty(N, dtype=np.int64); pos[order] = np.arange(1, N + 1)
    R[qi] = np.maximum.accumulate(pos[ex[qi]])
    ts = X[rs] @ Q[qi]; as_ = a[rs]
    ts = (ts - ts.mean()) / ts.std(); as_ = (as_ - as_.mean()) / as_.std()
    k = float(np.corrcoef(ts, as_)[0, 1])
    cq.append(np.sqrt(max(1 - k * k, 0)) / k)

src = f"{OUT}/kcurve_{key}.dat"
old = [l for l in open(src)]
hdr = [l for l in old if l.startswith("#")]
d = np.loadtxt(src)
# old column order: k, then (med,p95) for sym_bq adc_bq rabitq1 pq opq
cols = {"sym_bq": 1, "adc_bq": 3, "rabitq1": 5, "pq": 7, "opq": 9}
NEW = ["sym_bq", "adc_bq", "tq1", "rabitq1", "pq", "opq"]
out = [d[:, 0]]
for a in NEW:
    if a == "tq1":
        out += [np.median(R, 0), np.percentile(R, 95, axis=0)]
    else:
        out += [d[:, cols[a]], d[:, cols[a] + 1]]
M = np.column_stack(out)
head = (hdr[0] + hdr[1].rstrip() + f"  tq1={np.median(cq):.3f}\n" +
        "# k  " + "  ".join(f"{a}_med  {a}_p95" for a in NEW) + "\n")
open(src, "w").write(head + "\n".join(" ".join(f"{v:.0f}" for v in r) for r in M) + "\n")
print(f"rewrote {src} with tq1 (c_Q median {np.median(cq):.3f})")
for k in (1, 5, 10, 50, 100):
    print(f"   k={k:3d}  tq1 p95 R* = {np.percentile(R[:, k-1], 95):.0f}")
