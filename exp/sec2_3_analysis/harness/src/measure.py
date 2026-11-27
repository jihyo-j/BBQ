"""Per-configuration measurement of the Section-4 quantities, for every quantizer.

For each query q (cosine, unit-normalised vectors):
  t = exact score over the whole corpus, standardised -> z(r) by exact rank r
  beta  = |OLS slope of z(r) on ln r|, r = 1..100          (corpus/encoder only)
  kappa = Pearson(t, a) over the whole corpus              (per quantizer)
  nu    = sqrt(1 - kappa^2)   (residual sd of the standardised fit, eq. 1)
  eta   = nu / (kappa * beta)                              (eq. 2)
  r_j   = rank of the exact rank-j document under a, ties broken by id (App. A)
  R*    = max_{j<=10} r_j ;  CandidateRecall@R = (1/10) #{j : r_j <= R}

kappa is obtained exactly from Gram matrices:
  sum_i t_i a_i = q^T (X^T Xh) qt,   sum_i a_i^2 = qt^T (Xh^T Xh) qt,  etc.
which avoids materialising the N x nq exact score matrix once per quantizer.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import (CONFIGS, ORDER, load, label, OUT, WORK, NQ_MAX, K, RANK_FIT,
                    NTHREADS)
import quantizers as QZ

GRID = np.array([10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 800,
                 1000, 2000, 5000, 10000, 20000, 50000, 100000], dtype=np.int64)
CH = 100_000          # row chunk for Gram accumulation
QCH_BYTES = 1.5e9     # cap on the (nq_chunk x N) score block


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def colmean(A, ch=CH):
    s = np.zeros(A.shape[1], dtype=np.float64)
    for lo in range(0, A.shape[0], ch):
        s += A[lo:lo + ch].sum(0, dtype=np.float64)
    return s / A.shape[0]


def grams(A, B, ch=CH):
    """float64 accumulation of A^T B and B.sum(0) over row chunks."""
    G = np.zeros((A.shape[1], B.shape[1]), dtype=np.float64)
    s = np.zeros(B.shape[1], dtype=np.float64)
    for lo in range(0, A.shape[0], ch):
        hi = min(lo + ch, A.shape[0])
        G += (A[lo:hi].T @ B[lo:hi]).astype(np.float64)
        s += B[lo:hi].sum(0, dtype=np.float64)
    return G, s


def grams_c(A, mA, B, mB, ch=CH):
    """Centred cross-product (A-mA)^T (B-mB), accumulated in float64.

    Centring before the product rather than subtracting N*mA (x) mB afterwards:
    for a degenerate code (GIST, where sign(x) is the same vector for every
    document) the raw product and its correction agree to 10 digits, so the
    variance came out as pure rounding noise and kappa as a meaningless 0.995.
    """
    G = np.zeros((A.shape[1], B.shape[1]), dtype=np.float64)
    for lo in range(0, A.shape[0], ch):
        hi = min(lo + ch, A.shape[0])
        G += ((A[lo:hi] - mA).T @ (B[lo:hi] - mB)).astype(np.float64)
    return G


def qform(A, G, B):
    """Row-wise quadratic form a_q^T G b_q, via BLAS rather than einsum."""
    return np.einsum("ij,ij->i", A @ G, B)


def exact_side(X, Q, nq, N, d):
    """Exact top-RANK_FIT ids/scores (score desc, id asc) and the corpus moments."""
    mx = colmean(X)
    Gtt = grams_c(X, mx.astype(np.float32), X, mx.astype(np.float32))
    qd = Q.astype(np.float64)
    mean_t = qd @ mx
    var_t = np.maximum(qform(qd, Gtt, qd) / N, 1e-30)
    sd_t = np.sqrt(var_t)

    qch = max(1, int(QCH_BYTES / (N * 4)))
    gt_id = np.empty((nq, RANK_FIT), dtype=np.int64)
    gt_sc = np.empty((nq, RANK_FIT), dtype=np.float64)
    M = RANK_FIT * 3
    for lo in range(0, nq, qch):
        hi = min(lo + qch, nq)
        T = Q[lo:hi] @ X.T
        for i in range(hi - lo):
            t = T[i]
            cand = np.argpartition(t, -M)[-M:]
            order = np.lexsort((cand, -t[cand]))[:RANK_FIT]
            sel = cand[order]
            gt_id[lo + i] = sel
            gt_sc[lo + i] = t[sel]
        del T
    z = (gt_sc - mean_t[:, None]) / sd_t[:, None]            # (nq, RANK_FIT)

    lr = np.log(np.arange(1, RANK_FIT + 1, dtype=np.float64))
    lrc = lr - lr.mean()
    beta = -(z - z.mean(1, keepdims=True)) @ lrc / (lrc @ lrc)   # per query
    # robustness variant: ranks 10..100 only
    sl = slice(9, RANK_FIT)
    lr2 = lr[sl] - lr[sl].mean()
    beta_10_100 = -(z[:, sl] - z[:, sl].mean(1, keepdims=True)) @ lr2 / (lr2 @ lr2)
    return dict(Gtt=Gtt, mx=mx, mean_t=mean_t, sd_t=sd_t, gt_id=gt_id, gt_sc=gt_sc,
                z=z, beta=beta, beta_10_100=beta_10_100, z10=z[:, K - 1].copy())


def arm_ranks(Qt, Xh, gt_id, nq, N):
    """Rank (1-based, ties broken by ascending id) of each exact top-K document."""
    qch = max(1, int(QCH_BYTES / (N * 4)))
    ids = np.arange(N, dtype=np.int64)
    R = np.empty((nq, K), dtype=np.int64)
    for lo in range(0, nq, qch):
        hi = min(lo + qch, nq)
        A = Qt[lo:hi] @ Xh.T
        for i in range(hi - lo):
            a = A[i]
            g = gt_id[lo + i, :K]
            av = a[g]
            for j in range(K):
                v = av[j]
                gt = np.count_nonzero(a > v)
                m = (a == v)
                eq = np.count_nonzero(m)
                tie = np.count_nonzero(m & (ids < g[j])) if eq > 1 else 0
                R[lo + i, j] = 1 + gt + tie
        del A
    return R


def run(key, nq_max=NQ_MAX, arms=None, verify=True):
    t0 = time.time()
    X, Q, cfg = load(key, nq_max)
    N, d = X.shape
    nq = Q.shape[0]
    log(f"{key}: N={N} d={d} nq={nq}  ({label(key)})")

    ex = exact_side(X, Q, nq, N, d)
    beta = ex["beta"]
    log(f"  beta mean={beta.mean():.4f}  beta(10-100)={ex['beta_10_100'].mean():.4f}"
        f"  z10 mean={ex['z10'].mean():.3f}")

    res = dict(key=key, corpus=cfg["corpus"], encoder=cfg["encoder"], group=cfg["group"],
               N=int(N), d=int(d), nq=int(nq), threads=NTHREADS,
               beta=float(beta.mean()), beta_sd=float(beta.std()),
               beta_10_100=float(ex["beta_10_100"].mean()),
               z10=float(ex["z10"].mean()),
               z10_p10=float(np.quantile(ex["z10"], .10)),
               z10_p90=float(np.quantile(ex["z10"], .90)),
               arms={})
    per_q = dict(beta=beta, z10=ex["z10"], z_curve=ex["z"].mean(0))

    for arm in (arms or QZ.ARMS):
        ta = time.time()
        Qt, Xh, nbytes, note = QZ.build(arm, X, Q)
        mx32 = ex["mx"].astype(np.float32)
        mh = colmean(Xh)
        mh32 = mh.astype(np.float32)
        Gta = grams_c(X, mx32, Xh, mh32)
        Gaa = grams_c(Xh, mh32, Xh, mh32)
        qd, qtd = Q.astype(np.float64), Qt.astype(np.float64)
        var_a = qform(qtd, Gaa, qtd) / N
        cov = qform(qd, Gta, qtd) / N
        # a degenerate code (every document the same) has no variance at all:
        # kappa is undefined, not zero.  Flag relative to the mean square.
        ms_a = (qtd @ mh) ** 2 + var_a
        ok = var_a > 1e-12 * np.maximum(ms_a, 1e-30)
        kap = np.full(nq, np.nan)
        kap[ok] = cov[ok] / (ex["sd_t"][ok] * np.sqrt(var_a[ok]))
        kap = np.clip(kap, -1.0, 1.0)
        nu = np.sqrt(np.maximum(1.0 - kap ** 2, 0.0))
        eta = nu / np.maximum(kap, 1e-9) / beta

        if verify and arm in ("adc_bq", "pq"):
            i = 0
            tt = (X @ Q[i]).astype(np.float64)
            aa = (Xh @ Qt[i]).astype(np.float64)
            kd = np.corrcoef(tt, aa)[0, 1]
            log(f"    [verify] {arm} q0 kappa gram={kap[i]:.6f} direct={kd:.6f} "
                f"diff={abs(kap[i]-kd):.2e}")
            del tt, aa

        Rk = arm_ranks(Qt, Xh, ex["gt_id"], nq, N)
        del Qt, Xh
        Rstar = Rk.max(1)
        cr = {int(g): float((Rk <= g).mean()) for g in GRID}
        # p95 of R*(r): smallest candidate set retaining the exact rank-r document
        p95_r = [float(np.quantile(Rk[:, j], .95)) for j in range(K)]

        res["arms"][arm] = dict(
            name=QZ.NICE[arm], bytes_per_vec=float(nbytes), note=note,
            kappa=float(np.nanmean(kap)), nu=float(np.nanmean(nu)),
            eta=float(np.nanmean(eta)), eta_med=float(np.nanmedian(eta)),
            Rstar_p95=float(np.quantile(Rstar, .95)),
            Rstar_med=float(np.median(Rstar)), Rstar_mean=float(Rstar.mean()),
            candrec=cr, p95_Rstar_by_rank=p95_r,
            spearman_z10_Rstar=float(_spearman(ex["z10"], Rstar)),
            secs=round(time.time() - ta, 1))
        per_q[f"kappa_{arm}"] = kap
        per_q[f"eta_{arm}"] = eta
        per_q[f"Rstar_{arm}"] = Rstar
        per_q[f"ranks_{arm}"] = Rk
        a = res["arms"][arm]
        log(f"  {QZ.NICE[arm]:16s} {nbytes:7.1f}B  kappa={a['kappa']:.4f} "
            f"eta={a['eta']:.3f}  p95R*={a['Rstar_p95']:.0f}  "
            f"CR@30={cr[30]:.4f} CR@80={cr[80]:.4f}  [{a['secs']}s]")

    res["secs"] = round(time.time() - t0, 1)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/{key}.json", "w"), indent=1)
    np.savez_compressed(f"{WORK}/{key}_perq.npz", **per_q)
    log(f"{key} done in {res['secs']}s -> {OUT}/{key}.json")
    return res


def _spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else np.nan


if __name__ == "__main__":
    keys = sys.argv[1:] or ORDER
    for k in keys:
        run(k)
