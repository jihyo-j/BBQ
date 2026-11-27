"""At a fixed byte budget, buy dimensions or buy bits?

The paper's headline is that a wider representation needs a shorter shortlist
under a sign code.  Widening also buys bits -- 32 dimensions at one bit is
4 B/vector and 4096 is 512 B -- so the ladder of Section 3.2 cannot separate
"more dimensions" from "more bits".  This holds the bytes fixed instead.

One checkpoint (Qwen3-8B on MS MARCO), one ground truth: the exact FP32 top-10
of the FULL 4096-dimensional vectors, so every arm is asked to recover the
same answers.  MRL truncation takes a prefix and renormalizes, as elsewhere in
the paper.  Code sizes: a sign code is d/8 B, SQ-b is bd/8 B, PQ is M B.

  512 B/vector : 4096D 1-bit | 1024D 4-bit | 512D 8-bit | 4096D PQ(M=512)
  128 B/vector : 1024D 1-bit |  256D 4-bit | 128D 8-bit | 4096D PQ(M=128)

  BBQETA_THREADS=24 python3 src/bitsvsdims.py   ->  out/bitsvsdims.txt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, OUT, K
import quantizers as QZ
import faiss

NQ = int(os.environ.get("BVD_NQ", "500"))
PLAN = [(512, [("BQ", 4096, "adc_bq"), ("SQ-4bit", 1024, "sq4"),
               ("SQ-8bit", 512, "sq8"), ("PQ", 4096, "pq512")]),
        (128, [("BQ", 1024, "adc_bq"), ("SQ-4bit", 256, "sq4"),
               ("SQ-8bit", 128, "sq8"), ("PQ", 4096, "pq128")])]
LOG = []
def out(s=""):
    print(s, flush=True); LOG.append(s)


def truncate(X, d):
    """MRL prefix, renormalized -- the operation the rest of the paper uses."""
    Y = np.ascontiguousarray(X[:, :d])
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def pq_codes(X, Q, M):
    pq = faiss.ProductQuantizer(X.shape[1], M, 8)
    pq.cp.niter = 10
    pq.train(QZ._train_sample(X))
    Xh = np.empty_like(X)
    CH = 100_000
    for lo in range(0, len(X), CH):
        c = pq.compute_codes(np.ascontiguousarray(X[lo:lo+CH]))
        Xh[lo:lo+CH] = pq.decode(c)
    return Q.copy(), Xh


if __name__ == "__main__":
    X, Q, _ = load("msmarco_qw8", NQ)
    N, D = X.shape
    out(f"Qwen3-8B on MS MARCO, N={N}, d={D}, {len(Q)} queries")
    out("Ground truth: exact FP32 top-10 of the full 4096D vectors.\n")

    GT = np.zeros((len(Q), K), dtype=np.int64)
    for qi in range(len(Q)):
        t = X @ Q[qi]
        part = np.argpartition(-t, K)[:K]
        GT[qi] = part[np.argsort(-t[part], kind="stable")]

    for budget, arms in PLAN:
        out("=" * 64)
        out(f"{budget} B/vector")
        out("=" * 64)
        out(f"  {'arm':10s}{'d':>6s}{'B/vec':>7s}{'mean R*':>10s}{'p95 R*':>9s}"
            f"{'median':>8s}{'CandRec@100':>13s}")
        for name, d, arm in arms:
            Xd = truncate(X, d); Qd = truncate(Q, d)
            if arm.startswith("pq"):
                M = int(arm[2:])
                Qt, Xh = pq_codes(Xd, Qd, M); nb = float(M)
            else:
                Qt, Xh, nb, _ = QZ.build(arm, Xd, Qd, log=lambda *a: None)
            R = np.zeros(len(Q), dtype=np.int64)
            rec = np.zeros(len(Q))
            for qi in range(len(Q)):
                a = Xh @ Qt[qi]
                order = np.argsort(-a, kind="stable")
                pos = np.empty(N, dtype=np.int64); pos[order] = np.arange(1, N+1)
                r = pos[GT[qi]]
                R[qi] = r.max()
                rec[qi] = (r <= 100).mean()
            out(f"  {name:10s}{d:6d}{nb:7.0f}{R.mean():10.1f}{np.percentile(R,95):9.0f}"
                f"{np.median(R):8.0f}{rec.mean():13.3f}")
            del Xh, Qt, Xd, Qd
        out()
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/bitsvsdims.txt", "w").write("\n".join(LOG) + "\n")
    print(f"wrote {OUT}/bitsvsdims.txt")
