"""R*(k) past k=10, for a ladder of quantizers on the same corpus.

Section 3 stops at k=10 because that is the convention, and Figure 3(b) is
drawn there.  Two things are invisible at that cut.  How fast the budget grows
once k leaves the convention, and whether the gap between a coarse code and a
fine one is itself a function of k -- if the curves converge as k falls, then a
workload that only needs a short list can afford a larger score distortion, and
the quantizer comparison the paper reports is a statement about k=10 rather
than about the codes.

For every query: the exact FP32 top-100, each document's rank under each
quantizer's score (score desc, id asc, the paper's total order), and
R*(k) = max rank over the first k.  Reported as the median and p95 over
queries, which is what Figure 3(b) plots.

Writes one gnuplot table per corpus, columns
  k  then median and p95 per arm in ARMS order.

  BBQETA_THREADS=24 python3 src/kcurve.py [corpus ...]  ->  out/kcurve_*.dat
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, CONFIGS, OUT
import quantizers as QZ

ARMS = os.environ.get("KCURVE_ARMS",
                      "sym_bq,adc_bq,tq1,rabitq1,pq,opq").split(",")
CORPORA = ["msmarco_qw06", "quora_qw06", "nq_qw06", "dbpedia_qw06",
           "glove"]   # glove is the conventional control: low beta
KMAX = 100
# p95 over 300 queries is set by its top 15 points and jumps from k to k,
# which a ratio against another arm doubles.  Override for a clean curve.
NQ = int(os.environ.get("KCURVE_NQ", "300"))


def run(key):
    X, Q, _ = load(key, NQ)
    nq, N = len(Q), len(X)
    ex = np.zeros((nq, KMAX), dtype=np.int64)
    for qi in range(nq):
        t = X @ Q[qi]
        part = np.argpartition(-t, KMAX)[:KMAX]
        ex[qi] = part[np.argsort(-t[part], kind="stable")]
    res, cq = {}, {}
    for arm in ARMS:
        Qt, Xh, _, _ = QZ.build(arm, X, Q, log=lambda *a: None)
        R = np.zeros((nq, KMAX), dtype=np.int64)
        kap = np.zeros(nq); nu = np.zeros(nq)
        rs = np.random.default_rng(0).choice(N, 50_000, replace=False)
        for qi in range(nq):
            a = Xh @ Qt[qi]
            order = np.argsort(-a, kind="stable")
            pos = np.empty(N, dtype=np.int64)
            pos[order] = np.arange(1, N + 1)
            R[qi] = np.maximum.accumulate(pos[ex[qi]])
            ts = X[rs] @ Q[qi]; as_ = a[rs]
            ts = (ts - ts.mean()) / ts.std(); as_ = (as_ - as_.mean()) / as_.std()
            kap[qi] = float(np.corrcoef(ts, as_)[0, 1])
            nu[qi] = float(np.sqrt(max(1 - kap[qi] ** 2, 0)))
        res[arm] = R
        cq[arm] = float(np.median(nu / kap))
        del Xh, Qt
    del X, Q
    return res, cq


if __name__ == "__main__":
    want = sys.argv[1:] or CORPORA
    os.makedirs(OUT, exist_ok=True)
    for key in want:
        res, cq = run(key)
        head = ("# " + CONFIGS[key]["corpus"] + " / " + CONFIGS[key]["encoder"] +
                f", {NQ} queries\n# c_Q (median): " +
                "  ".join(f"{a}={cq[a]:.3f}" for a in ARMS) +
                "\n# k  " + "  ".join(f"{a}_med  {a}_p95  {a}_mean" for a in ARMS) + "\n")
        lines = []
        for k in range(1, KMAX + 1):
            row = [str(k)]
            for a in ARMS:
                col = res[a][:, k - 1]
                row += [f"{np.median(col):.0f}", f"{np.percentile(col, 95):.0f}",
                        f"{col.mean():.1f}"]
            lines.append(" ".join(row))
        open(f"{OUT}/kcurve_{key}.dat", "w").write(head + "\n".join(lines) + "\n")
        # the per-query budgets themselves, so a different statistic can be
        # read off later without paying for the sweep again
        np.savez_compressed(f"{OUT}/kcurve_{key}_perq.npz",
                            arms=np.array(ARMS),
                            cq=np.array([cq[a] for a in ARMS]),
                            **{a: res[a] for a in ARMS})
        print(f"wrote {OUT}/kcurve_{key}.dat and _perq.npz")
        for a in ARMS:
            p95 = [np.percentile(res[a][:, k - 1], 95) for k in (1, 5, 10, 30, 100)]
            print(f"   {a:8s} c_Q={cq[a]:.2f}  p95 R* at k=1,5,10,30,100: " +
                  " ".join(f"{v:7.0f}" for v in p95))
