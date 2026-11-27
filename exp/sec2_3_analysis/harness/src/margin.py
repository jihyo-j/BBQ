"""Is there a cliff after rank 10, and how much score error does the boundary
actually absorb?

Two questions, both per query and both standardized by that query's own corpus
scores so they are comparable across corpora:

  knee    z(r) for r = 1..60.  The top-10 is a cut we impose; the embedding
          does not know about it, so a step at 10 would have to come from the
          corpus, not from k.
  budget  n(delta) = |{x : z(x) >= z(10) - delta}|, the number of documents a
          perturbation of delta standard deviations could bring above the tenth
          answer.  This is what "tolerating more error" means in candidates,
          and c_Q is the size of the perturbation the code actually applies.

  BBQETA_THREADS=24 python3 src/margin.py   ->  out/margin.txt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, CONFIGS, WORK, OUT, K

ARMS = ["quora_qw06", "msmarco_qw06", "nq_qw06", "dbpedia_qw06"]
NQ = 500
SAMPLE = 200_000
DELTAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
RMAX = 60

LOG = []
def out(s=""):
    print(s); LOG.append(s)


def run(key):
    X, Q, _ = load(key, NQ)
    z = np.load(f"{WORK}/{key}_perq.npz")
    nq = len(Q)
    rs = np.random.default_rng(0).choice(len(X), SAMPLE, replace=False)
    Xs = X[rs]
    Zc = np.zeros((nq, RMAX))          # z at ranks 1..RMAX
    N = np.zeros((nq, len(DELTAS)))    # documents within delta of z(10)
    for qi in range(nq):
        s = Xs @ Q[qi]; mu, sd = s.mean(), s.std()
        t = X @ Q[qi]
        part = np.argpartition(-t, RMAX)[:RMAX]
        top = t[part][np.argsort(-t[part], kind="stable")]
        Zc[qi] = (top - mu) / sd
        z10 = top[K - 1]
        for j, d in enumerate(DELTAS):
            N[qi, j] = np.count_nonzero(t >= z10 - d * sd)
    cq = z["eta_adc_bq"][:nq] * z["beta"][:nq]      # c_Q = nu/kappa = eta*beta
    res = dict(Z=Zc, N=N, cq=cq, rstar=z["Rstar_adc_bq"][:nq], N_corpus=len(X))
    del X, Q
    return res


if __name__ == "__main__":
    R = {k: run(k) for k in ARMS}

    out("=" * 78)
    out("1.  IS THERE A CLIFF AT RANK 10?   median z(r), and the drop per rank")
    out("=" * 78)
    out(f"{'corpus':9s}" + "".join(f"{'z('+str(r)+')':>8s}"
        for r in (1, 5, 9, 10, 11, 15, 20, 30, 60)))
    for k in ARMS:
        m = np.median(R[k]["Z"], 0)
        out(f"{CONFIGS[k]['corpus']:9s}" +
            "".join(f"{m[r-1]:8.2f}" for r in (1, 5, 9, 10, 11, 15, 20, 30, 60)))
    out()
    out("  drop in z from one rank to the next, median over queries:")
    out(f"{'corpus':9s}" + "".join(f"{str(r)+'->'+str(r+1):>9s}"
        for r in (5, 8, 9, 10, 11, 15, 20)))
    for k in ARMS:
        d = np.median(-np.diff(R[k]["Z"], axis=1), 0)
        out(f"{CONFIGS[k]['corpus']:9s}" +
            "".join(f"{d[r-1]:9.3f}" for r in (5, 8, 9, 10, 11, 15, 20)))

    out()
    out("=" * 78)
    out("2.  HOW MUCH ERROR DOES THE RANK-10 BOUNDARY ABSORB?")
    out("    n(delta) = documents within delta sd of the tenth answer (median)")
    out("=" * 78)
    out(f"{'corpus':9s}{'corpus N':>10s}" +
        "".join(f"{'d='+str(d):>9s}" for d in DELTAS) +
        f"{'c_Q':>7s}{'R* med':>8s}")
    for k in ARMS:
        r = R[k]
        out(f"{CONFIGS[k]['corpus']:9s}{r['N_corpus']:10d}" +
            "".join(f"{v:9.0f}" for v in np.median(r['N'], 0)) +
            f"{np.median(r['cq']):7.2f}{np.median(r['rstar']):8.0f}")
    out()
    out("  per-query Spearman against log R*, over the four corpora pooled:")
    import scipy.stats as st
    allN = np.vstack([R[k]["N"] for k in ARMS])
    allR = np.concatenate([R[k]["rstar"] for k in ARMS])
    allC = np.concatenate([R[k]["cq"] for k in ARMS])
    for j, d in enumerate(DELTAS):
        out(f"    n(delta={d}) : {st.spearmanr(allN[:, j], allR).statistic:+.3f}")
    out(f"    c_Q         : {st.spearmanr(allC, allR).statistic:+.3f}")
    # the matched prediction: documents within one c_Q of the boundary
    out()
    out("  n(delta = that query's own c_Q) against its R*:")
    for k in ARMS:
        r = R[k]
        pred = np.array([np.interp(c, DELTAS, r["N"][i]) for i, c in enumerate(r["cq"])])
        rho = st.spearmanr(pred, r["rstar"]).statistic
        out(f"    {CONFIGS[k]['corpus']:9s} median predicted {np.median(pred):6.0f}"
            f"   actual R* {np.median(r['rstar']):5.0f}   rho {rho:+.3f}")

    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/margin.txt", "w").write("\n".join(LOG) + "\n")
    print(f"\nwrote {OUT}/margin.txt")
