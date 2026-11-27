"""What the candidate budget is actually buying: the exact top-k, or the answer.

Section 3 reports R* against the encoder's own exact FP32 top-10.  k=10 is a
convention; the relevance judgments say how many documents a query really has
an answer in.  This measures both targets on the same queries under the same
ADC-BQ ranking:

  R*(k)     depth that holds the exact FP32 top-k, k = 1..10
  R_judged  depth that holds every document the judgments call relevant

Only MS MARCO and Quora can be measured this way -- the other corpora are 1M
subsamples whose rows do not map back to their qrels (see raw_material.py).

  BBQETA_THREADS=24 python3 src/ktarget.py   ->  out/ktarget.txt
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, CONFIGS, OUT, K
import quantizers as QZ
from raw_material import judged_msmarco, judged_quora

NQ = 500
LOG = []
def out(s=""):
    print(s); LOG.append(s)


def run(key, rel):
    X, Q, _ = load(key, NQ)
    Qt, Xh, _, _ = QZ.build("adc_bq", X, Q)
    nq = len(Q)
    Rk = np.zeros((nq, K), dtype=np.int64)
    Rj = np.full(nq, -1, dtype=np.int64)
    nj = np.zeros(nq, dtype=np.int64)
    def rank_of(a, d):
        """Rank of document d under the code's score, score desc then id asc --
        counted rather than sorted, which is what makes 500 queries tractable."""
        v = a[d]
        return int(np.count_nonzero(a > v) +
                   np.count_nonzero((a[:d] == v)) + 1)

    for qi in range(nq):
        t = X @ Q[qi]
        a = Xh @ Qt[qi]
        part = np.argpartition(-t, K)[:K]
        ex = part[np.argsort(-t[part], kind="stable")]  # exact FP32 top-10
        Rk[qi] = np.maximum.accumulate([rank_of(a, int(d)) for d in ex])
        j = [int(d) for d in rel[qi]] if qi < len(rel) else []
        nj[qi] = len(j)
        if j:
            Rj[qi] = max(rank_of(a, d) for d in j)
    del X, Q, Xh
    return Rk, Rj, nj


if __name__ == "__main__":
    for key, rel in (("msmarco_qw06", judged_msmarco()),
                     ("quora_qw06", judged_quora())):
        Rk, Rj, nj = run(key, rel)
        c = CONFIGS[key]["corpus"]
        out("=" * 70)
        out(f"{c}  (Qwen3-0.6B 1024d, {len(Rk)} queries)")
        out("=" * 70)
        out(f"  judged answers per query: median {np.median(nj):.0f}  "
            f"mean {nj.mean():.2f}  exactly one {np.mean(nj == 1):.1%}")
        out()
        out(f"  {'target':28s}{'median':>8s}{'p95':>8s}{'p99':>8s}")
        for k in (1, 2, 3, 5, 10):
            out(f"  {'exact FP32 top-' + str(k):28s}"
                f"{np.median(Rk[:, k-1]):8.0f}{np.percentile(Rk[:, k-1], 95):8.0f}"
                f"{np.percentile(Rk[:, k-1], 99):8.0f}")
        ok = Rj > 0
        out(f"  {'judged answers':28s}{np.median(Rj[ok]):8.0f}"
            f"{np.percentile(Rj[ok], 95):8.0f}{np.percentile(Rj[ok], 99):8.0f}")
        out()
        out(f"  ratio, exact top-10 over judged:  median "
            f"{np.median(Rk[:, K-1]) / np.median(Rj[ok]):.1f}x   "
            f"p95 {np.percentile(Rk[:, K-1], 95) / np.percentile(Rj[ok], 95):.1f}x")
        import scipy.stats as st
        out(f"  per-query Spearman, R*(10) vs judged depth: "
            f"{st.spearmanr(Rk[ok, K-1], Rj[ok]).statistic:+.3f}")
        out(f"  queries the two targets disagree on by >5x: "
            f"{np.mean(np.maximum(Rk[ok, K-1], Rj[ok]) > 5*np.minimum(Rk[ok, K-1], Rj[ok])):.1%}")
        np.savez(f"{OUT}/ktarget_{key}.npz", Rk=Rk, Rj=Rj, nj=nj)
        out(f"  R*(10) / R*(1):  median "
            f"{np.median(Rk[:, K-1]) / max(np.median(Rk[:, 0]), 1):.1f}x   "
            f"p95 {np.percentile(Rk[:, K-1], 95) / max(np.percentile(Rk[:, 0], 95), 1):.1f}x")
        out()
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/ktarget.txt", "w").write("\n".join(LOG) + "\n")
    print(f"wrote {OUT}/ktarget.txt")
