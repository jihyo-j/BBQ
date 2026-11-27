"""Theorem 1 run exactly as written, on the pools of pool.py.

The theorem calibrates on *global* exact FP32 top-k answers, draws one answer
per calibration query uniformly, and takes the l-th order statistic of the
entry levels.  What it guarantees is a marginal average: the expectation is
over the calibration sample, the answer draw and the test query jointly.  This
module runs that procedure over many (split, seed) replicates and reports the
three things the guarantee is about -- achieved global mean Recall@k, the
refinement budget it spends, and how often the rule certifies nothing and falls
back to exhaustive search.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import WORK, OUT, CONFIGS, K

M_DEFAULT = 24


def load_pool(key, ef, M=M_DEFAULT):
    z = np.load(f"{WORK}/pool_{key}_M{M}_ef{ef}.npz")
    return z["pool_s"], z["gold_lv"], z["ndis"]


def _csize(pool_s, tau):
    """|C_tau(q)| for every q: pool rows are score-descending."""
    sk = pool_s[:, K - 1:K]
    return (pool_s >= sk - tau).sum(1)


def replicate(pool_s, gold_lv, rho, m, rng, N):
    """One (calibration split, answer draw) replicate of Theorem 1."""
    nq, k = gold_lv.shape
    perm = rng.permutation(nq)
    cal, tst = perm[:m], perm[m:]
    J = rng.integers(0, k, size=m)
    Z = gold_lv[cal, J]                      # Z_i = T_{J_i}(Q_i), +inf allowed
    ell = int(np.ceil((m + 1) * (1.0 - rho)))
    tau = np.inf if ell > m else float(np.sort(Z)[ell - 1])

    gl = gold_lv[tst]
    if not np.isfinite(tau):                 # exhaustive fallback
        return dict(tau=np.inf, recall=1.0, R_mean=float(N), R_p95=float(N),
                    R_p99=float(N), fallback=1.0)
    R = _csize(pool_s[tst], tau).astype(np.float64)
    return dict(tau=tau,
                recall=float((gl <= tau).mean()),
                R_mean=float(R.mean()),
                R_p95=float(np.quantile(R, .95)),
                R_p99=float(np.quantile(R, .99)),
                fallback=0.0)


def campaign(key, ef, rho=0.01, m=1000, B=400, seed=20261118):
    """Theorem 1 over B (split, answer-draw) replicates.

    Reported separately because they are different regimes: the replicates
    where the rule certifies a finite threshold, and the replicates where the
    (1-rho) quantile of the entry level is +inf and the rule escalates to the
    exhaustive fallback.  The guarantee is on the mean over both.
    """
    pool_s, gold_lv, ndis = load_pool(key, ef)
    N = int(CONFIGS[key].get("N", 1_000_000))
    rng = np.random.default_rng(seed)
    nq = len(gold_lv)
    m = min(m, nq // 2)
    r = [replicate(pool_s, gold_lv, rho, m, rng, N) for _ in range(B)]
    g = lambda f: np.array([x[f] for x in r], dtype=np.float64)
    rec, fin = g("recall"), np.isfinite(g("tau"))
    ceiling = float(np.isfinite(gold_lv).mean())
    ell = int(np.ceil((m + 1) * (1.0 - rho)))
    # P[tau = inf] = P[ #{Z_i = inf} > m - ell ], #{Z=inf} ~ Binom(m, L_visit)
    from scipy.stats import binom
    pred = float(binom.sf(m - ell, m, 1 - ceiling))
    o = dict(key=key, ef=int(ef), rho=rho, m=int(m), B=int(B), ell=ell,
             nq_test=int(nq - m), nq=int(nq), ceiling=ceiling,
             L_visit=1 - ceiling,
             recall_mean=float(rec.mean()), recall_sd=float(rec.std(ddof=1)),
             recall_se=float(rec.std(ddof=1) / np.sqrt(len(rec))),
             recall_p05=float(np.quantile(rec, .05)),
             recall_min=float(rec.min()),
             shortfall=float((rec < 1 - rho).mean()),
             fallback=float(1.0 - fin.mean()), fallback_pred=pred,
             ndis=float(ndis.mean()))
    if fin.any():
        o.update(recall_fin=float(rec[fin].mean()),
                 R_fin=float(g("R_mean")[fin].mean()),
                 Rp95_fin=float(g("R_p95")[fin].mean()),
                 Rp99_fin=float(g("R_p99")[fin].mean()),
                 tau_fin=float(np.median(g("tau")[fin])))
    else:
        o.update(recall_fin=np.nan, R_fin=np.nan, Rp95_fin=np.nan,
                 Rp99_fin=np.nan, tau_fin=np.nan)
    return o


# --------------------------------------------- fixed-budget reference curves
def fixed_curve(key, ef, Rs=(10, 20, 30, 40, 50, 60, 80, 100, 150, 200, 300,
                             500, 800, 1024)):
    """Recall@k after reranking the pool's top-R, and the loss decomposition.

    recall(R) = mean_j 1[ |C_{T_j}| <= R ] : an answer is reranked iff the
    score prefix that reaches it is no longer than R.  L_visit is the share of
    answers the traversal never returned, L_select(R) the share it returned and
    the budget did not reach.
    """
    pool_s, gold_lv, ndis = load_pool(key, ef)
    sk = pool_s[:, K - 1:K]
    ceiling = float(np.isfinite(gold_lv).mean())
    miss = ~np.isfinite(gold_lv)
    thr = sk - gold_lv                            # ADC score of each answer
    thr[miss] = -np.inf
    pos = np.empty(gold_lv.shape, dtype=np.float64)
    for j in range(gold_lv.shape[1]):             # pool rank of answer j
        pos[:, j] = (pool_s >= thr[:, j:j + 1]).sum(1)
    pos[miss] = np.inf
    out = {}
    for R in Rs:
        if R > pool_s.shape[1]:
            continue
        rec = float((pos <= R).mean())
        out[int(R)] = dict(recall=rec, L_visit=1 - ceiling,
                           L_select=ceiling - rec)
    return dict(key=key, ef=int(ef), ceiling=ceiling,
                ndis=float(ndis.mean()), curve=out)


if __name__ == "__main__":
    for key in sys.argv[1:]:
        for ef in (128, 256, 1024):
            try:
                print(json.dumps(campaign(key, ef)))
            except FileNotFoundError:
                pass


# ------------------------------------------- the pool-relative rule, for contrast
def pool_relative(key, ef, rho=0.01, m=1000, B=400, seed=20261118, M=M_DEFAULT):
    """Calibrate against the *pool's own* FP32 top-k instead of the global one.

    This is the quantity Figure~7(b) reports.  It is a valid conformal rule for
    a different target -- the selection loss inside L(q) -- and it cannot see
    the visit loss, so the global recall it delivers is lower than its target
    by roughly that amount.
    """
    z = np.load(f"{WORK}/poolid_{key}_M{M}_ef{ef}.npz")
    ids, fp, GT = z["ids"], z["fp"], z["gt"]
    ps, gold_lv, _ = load_pool(key, ef)
    nq = len(ids)
    sk = ps[:, K - 1:K]

    # pool-relative answers: top-K of the pool under exact FP32 score
    ordr = np.argsort(-fp, axis=1, kind="stable")[:, :K]
    # their ADC levels: T = max(0, s_k - s(x)); ADC score of pool entry j is ps[:, j]
    # ids/fp/ps share the pool's ADC-descending order, so column j is the same doc
    lv_pool = np.maximum(0.0, sk - np.take_along_axis(ps, ordr, 1))

    rng = np.random.default_rng(seed)
    gl_glob = gold_lv
    out = []
    for _ in range(B):
        perm = rng.permutation(nq)
        cal, tst = perm[:m], perm[m:]
        J = rng.integers(0, K, size=m)
        Z = lv_pool[cal, J]
        ell = int(np.ceil((m + 1) * (1.0 - rho)))
        tau = np.inf if ell > m else float(np.sort(Z)[ell - 1])
        if not np.isfinite(tau):
            out.append((1.0, 1.0, float(ids.shape[1])))
            continue
        out.append((float((lv_pool[tst] <= tau).mean()),       # pool recall
                    float((gl_glob[tst] <= tau).mean()),       # global recall
                    float(_csize(ps[tst], tau).mean())))
    a = np.array(out)
    return dict(key=key, ef=int(ef), rho=rho, m=int(m), B=int(B),
                pool_recall=float(a[:, 0].mean()),
                global_recall=float(a[:, 1].mean()),
                global_recall_sd=float(a[:, 1].std(ddof=1)),
                R_mean=float(a[:, 2].mean()),
                L_visit=float(1 - np.isfinite(gold_lv).mean()))
