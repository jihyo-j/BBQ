"""Quantizers, each expressed as (transformed query Qt, reconstruction Xh) so that
the estimated score of document i is  a_i = Qt_q . Xh_i  (+ a per-query constant,
which changes neither the ranking nor the correlation with the exact score).

This is exactly what the deployed scorers compute:
  * PQ / OPQ LUT sum      == q . decode(code)
  * SQ int dot            == q . decode(code)
  * ADC-BQ                == q . sign(x)
  * RaBitQ-1bit           == q.c + beta_i * (Pq) . sign(P(x-c))
so reconstruct-then-GEMM is algebraically identical and far cheaper to sweep.
"""
import numpy as np
from common import SEED

CH = 100_000


def _sign(A, out=None):
    return np.copysign(np.float32(1.0), A, out=out)


def _rot(d, seed):
    rng = np.random.default_rng(seed)
    P, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return np.ascontiguousarray(P.astype(np.float32))


def _pq_m(d):
    """Subquantizer count giving ~d/8 bytes/vector (one byte each), d divisible by M."""
    target = max(1, d // 8)
    for m in range(target, 0, -1):
        if d % m == 0:
            return m
    return 1


def _train_sample(X, n=100_000, seed=SEED):
    if X.shape[0] <= n:
        return np.ascontiguousarray(X)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], n, replace=False))
    return np.ascontiguousarray(X[idx])


ARMS = ["sym_bq", "adc_bq", "adc_bq_c", "adc_bq_cb", "pq", "opq", "sq4", "sq8",
        "rabitq1", "tq1"]

NICE = {"sym_bq": "Sym-BQ", "adc_bq": "ADC-BQ", "adc_bq_c": "ADC-BQ +centre",
        "adc_bq_cb": "ADC-BQ +centre+scale", "pq": "PQ", "opq": "OPQ",
        "sq4": "SQ-4bit", "sq8": "SQ-8bit", "rabitq1": "RaBitQ-1bit",
        "tq1": "TurboQuant-1bit"}


def build(arm, X, Q, log=print):
    """Return (Qt, Xh, bytes_per_vector, note)."""
    import faiss
    faiss.omp_set_num_threads(int(__import__("os").environ.get("BBQETA_THREADS", 24)))
    N, d = X.shape

    if arm == "sym_bq":
        Xh = np.empty_like(X)
        for lo in range(0, N, CH):
            _sign(X[lo:lo + CH], out=Xh[lo:lo + CH])
        return _sign(Q), Xh, d / 8.0, "sign(q).sign(x)"

    if arm == "adc_bq":
        Xh = np.empty_like(X)
        for lo in range(0, N, CH):
            _sign(X[lo:lo + CH], out=Xh[lo:lo + CH])
        return Q.copy(), Xh, d / 8.0, "q.sign(x)"

    if arm in ("adc_bq_c", "adc_bq_cb"):
        # RaBitQ minus the rotation.  Isolates how much of RaBitQ's edge is a
        # free shared centroid (no per-vector bytes) and how much needs the
        # per-vector scalar / the rotation.
        c = np.zeros(d, dtype=np.float64)
        for lo in range(0, N, CH):
            c += X[lo:lo + CH].sum(0, dtype=np.float64)
        c = (c / N).astype(np.float32)
        Xh = np.empty_like(X)
        for lo in range(0, N, CH):
            hi = min(lo + CH, N)
            Z = X[lo:hi] - c
            B = _sign(Z)
            if arm == "adc_bq_cb":
                beta = np.einsum("ij,ij->i", Z, Z) / np.maximum(np.abs(Z).sum(1), 1e-20)
                B *= beta[:, None]
            Xh[lo:hi] = B
        nb = d / 8.0 + (4.0 if arm == "adc_bq_cb" else 0.0)
        return Q.copy(), Xh, nb, ("shared centroid" if arm == "adc_bq_c"
                                  else "shared centroid + per-vector scale")

    if arm in ("pq", "opq"):
        M = _pq_m(d)
        Xt, Qt = X, Q
        note = f"M={M} x 8bit"
        if arm == "opq":
            opq = faiss.OPQMatrix(d, M)
            opq.niter = 8
            opq.niter_pq = 6
            opq.verbose = False
            opq.train(_train_sample(X))
            Xt = np.empty_like(X)
            for lo in range(0, N, CH):
                Xt[lo:lo + CH] = opq.apply(np.ascontiguousarray(X[lo:lo + CH]))
            Qt = opq.apply(np.ascontiguousarray(Q))
            note += " + learned rotation"
        pq = faiss.ProductQuantizer(d, M, 8)
        pq.cp.niter = 10
        pq.cp.seed = SEED % (2 ** 31)
        pq.train(_train_sample(Xt))
        Xh = np.empty_like(X)
        for lo in range(0, N, CH):
            c = pq.compute_codes(np.ascontiguousarray(Xt[lo:lo + CH]))
            Xh[lo:lo + CH] = pq.decode(c)
        if arm == "opq":
            del Xt
        return np.ascontiguousarray(Qt), Xh, float(M), note

    if arm in ("sq4", "sq8"):
        qt = faiss.ScalarQuantizer.QT_4bit if arm == "sq4" else faiss.ScalarQuantizer.QT_8bit
        sq = faiss.ScalarQuantizer(d, qt)
        sq.train(_train_sample(X))
        Xh = np.empty_like(X)
        for lo in range(0, N, CH):
            c = sq.compute_codes(np.ascontiguousarray(X[lo:lo + CH]))
            Xh[lo:lo + CH] = sq.decode(c)
        return Q.copy(), Xh, d * (0.5 if arm == "sq4" else 1.0), "per-dim range"

    if arm == "rabitq1":
        # validated against faiss IndexRaBitQ(nb=1) in projects/BBQFLOW (step 9/12):
        # the per-vector scalar is beta = ||z||^2 / ||z||_1, NOT ||z||_1 / d.
        c = np.zeros(d, dtype=np.float64)
        for lo in range(0, N, CH):
            c += X[lo:lo + CH].sum(0, dtype=np.float64)
        c = (c / N).astype(np.float32)
        P = _rot(d, SEED)
        Xh = np.empty_like(X)
        beta = np.empty(N, dtype=np.float32)
        for lo in range(0, N, CH):
            hi = min(lo + CH, N)
            Z = (X[lo:hi] - c) @ P
            B = _sign(Z)
            s2 = np.einsum("ij,ij->i", Z, Z)
            l1 = np.abs(Z).sum(1)
            beta[lo:hi] = s2 / np.maximum(l1, 1e-20)
            Xh[lo:hi] = B * beta[lo:hi, None]
        return np.ascontiguousarray(Q @ P), Xh, d / 8.0 + 4.0, "centroid + rotation + beta"

    if arm == "tq1":
        # TurboQuant 1-bit: random rotation, then the MSE-optimal 1-bit codebook.
        # The lab's reference codebook (jihyo/turboquant_hnsw, d=960/1536, bits=1)
        # has centroids +-0.025758 = sqrt(2/(pi*d)) at d=960, which this reproduces.
        # On unit-norm data the global scale is constant, so the RANKING of this arm
        # equals rotated ADC-BQ; only the estimate's scale differs.
        P = _rot(d, SEED + 1)
        c0 = np.float32(np.sqrt(2.0 / (np.pi * d)))
        Xh = np.empty_like(X)
        for lo in range(0, N, CH):
            hi = min(lo + CH, N)
            Xh[lo:hi] = _sign(X[lo:hi] @ P) * c0
        return np.ascontiguousarray(Q @ P), Xh, d / 8.0, "random rotation + MSE 1-bit codebook"

    raise ValueError(arm)
