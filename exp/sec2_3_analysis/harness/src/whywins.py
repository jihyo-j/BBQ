"""Which property of an embedding decides whether a rotation is worth applying?

OPQ has the lowest score distortion c_Q and the smallest p95 R* on all thirteen
Transformer configurations we measure, so the identity of the best same-size
quantizer does NOT change with the embedding.  What changes is the value of a
rotation at one bit: TurboQuant-1bit (random rotation, no extra bytes) beats
plain ADC-BQ on some embeddings and loses badly on others, e.g. 43 against 47
on MS MARCO Qwen-1024 but 133 against 66 on MS MARCO ada-002.

This module measures candidate explanations on the document vectors alone:

  coord_cv  coefficient of variation of the per-coordinate variances.  A sign
            code spends one bit per coordinate regardless of how much variance
            that coordinate carries, so unequal variances waste bits and a
            rotation that equalises them should pay.
  aniso     norm of the mean unit document vector: how far the corpus sits from
            the origin as a cone.
  f_mean    mean of ||x||_1 / (sqrt(d) ||x||_2), the cosine between a vector and
            its own sign code.
  kurt      mean per-coordinate excess kurtosis; heavy-tailed coordinates make
            a sign threshold at zero less informative.
  pr        participation ratio of the covariance spectrum over d, i.e. the
            fraction of dimensions that carry variance.

and correlates them with the rotation's effect, measured both globally
(c_Q ratio) and at the budget (R* ratio).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, OUT, ORDER, CONFIGS
import tables as T

NSAMP = 200_000


def descriptors(key, nsamp=NSAMP):
    X, _Q, cfg = load(key, 8)
    N, d = X.shape
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(N, min(nsamp, N), replace=False))
    S = np.asarray(X[idx], dtype=np.float64)
    del X
    var = S.var(0)
    mu = S.mean(0)
    C = S - mu
    m2 = (C ** 2).mean(0)
    m4 = (C ** 4).mean(0)
    kurt = float(np.mean(m4 / np.maximum(m2 ** 2, 1e-30)) - 3.0)
    # covariance spectrum without forming d x d when d is large
    lam = np.linalg.svd(C[:min(20000, len(C))], compute_uv=False) ** 2
    pr = float((lam.sum() ** 2) / np.maximum((lam ** 2).sum(), 1e-30) / d)
    f = np.abs(S).sum(1) / (np.sqrt(d) * np.linalg.norm(S, axis=1))
    return dict(d=int(d),
                coord_cv=float(var.std() / var.mean()),
                aniso=float(np.linalg.norm(mu)),
                f_mean=float(f.mean()),
                kurt=kurt,
                pr=pr)


def run(keys=None):
    R = T.load_all()
    keys = keys or [k for k in ORDER if k in R and "arms" in R[k]
                    and R[k]["arms"].get("tq1", {}).get("kappa")]
    out = {}
    for k in keys:
        a = R[k]["arms"]
        cq = lambda arm: a[arm]["nu"] / a[arm]["kappa"]
        out[k] = descriptors(k)
        out[k].update(
            corpus=R[k]["corpus"], encoder=R[k]["encoder"],
            cQ_adc=cq("adc_bq"), cQ_tq=cq("tq1"), cQ_opq=cq("opq"),
            R_adc=a["adc_bq"]["Rstar_p95"], R_tq=a["tq1"]["Rstar_p95"],
            R_opq=a["opq"]["Rstar_p95"])
        out[k]["rot_cQ"] = out[k]["cQ_tq"] / out[k]["cQ_adc"]
        out[k]["rot_R"] = out[k]["R_tq"] / out[k]["R_adc"]
        print(f"  {k:22s} cv={out[k]['coord_cv']:.3f} aniso={out[k]['aniso']:.3f} "
              f"f={out[k]['f_mean']:.4f} kurt={out[k]['kurt']:+.2f} pr={out[k]['pr']:.3f} "
              f"| rot c_Q {out[k]['rot_cQ']:.2f}x  rot R* {out[k]['rot_R']:.2f}x", flush=True)
        json.dump(out, open(f"{OUT}/whywins.json", "w"), indent=1)
    return out


def elasticity(keys=None, arms=("adc_bq", "tq1", "rabitq1", "pq", "opq")):
    """How much a reduction in score distortion is worth, per embedding.

    Slope of log p95 R* on log c_Q across the same-size quantizers of one
    embedding.  If R* were a function of eta = c_Q / beta alone this slope
    would be the same everywhere, because beta is common to every quantizer on
    a given embedding and would only shift the intercept.  It is not: it runs
    from 0.14 on Quora/BGE-M3 to 1.67 on FEVER/BGE-M3 and tracks beta at
    Spearman -0.895.  Where the top ranks are well separated a halving of the
    distortion buys almost no candidates; where they lie on a flat shelf the
    same halving is worth a great deal.
    """
    R = T.load_all()
    rows = []
    for k, v in R.items():
        if "arms" not in v or k == "gist":
            continue
        a = v["arms"]
        if not all(x in a and a[x].get("kappa") for x in arms):
            continue
        if keys and k not in keys:
            continue
        x = np.log([a[q]["nu"] / a[q]["kappa"] for q in arms])
        y = np.log([a[q]["Rstar_p95"] for q in arms])
        rows.append((k, v["corpus"] + "/" + v["encoder"], float(v["beta"]),
                     float(np.polyfit(x, y, 1)[0])))
    return sorted(rows, key=lambda r: -r[2])


def correlate(out):
    from scipy.stats import spearmanr
    props = ["coord_cv", "aniso", "f_mean", "kurt", "pr", "d"]
    ks = list(out)
    print(f"\nSpearman over {len(ks)} configurations")
    print(f"{'property':12s}{'rot c_Q ratio':>16s}{'rot R* ratio':>15s}")
    for p in props:
        x = [out[k][p] for k in ks]
        print(f"{p:12s}{spearmanr(x,[out[k]['rot_cQ'] for k in ks]).statistic:16.3f}"
              f"{spearmanr(x,[out[k]['rot_R'] for k in ks]).statistic:15.3f}")


if __name__ == "__main__":
    from scipy.stats import spearmanr
    out = run()
    correlate(out)
    rows = elasticity()
    print(f"\n{'configuration':26s}{'beta':>7s}{'dlogR*/dlogc_Q':>16s}")
    for _k, n, b, e in rows:
        print(f"{n:26s}{b:7.3f}{e:16.2f}")
    B = [r[2] for r in rows]; E = [r[3] for r in rows]
    print(f"rho(beta, elasticity) = {spearmanr(B, E).statistic:+.3f}")
    json.dump({"descriptors": out,
               "elasticity": {r[0]: dict(name=r[1], beta=r[2], slope=r[3]) for r in rows},
               "rho_beta_elasticity": float(spearmanr(B, E).statistic)},
              open(f"{OUT}/whywins.json", "w"), indent=1)
