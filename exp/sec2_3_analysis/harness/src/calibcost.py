"""What calibration costs, end to end, for one (index, procedure, scorer).

The rule of Section 5.4 needs three things per calibration query: the global
exact FP32 top-k (the theorem is stated over global ground truth, not the
pool's own top-k), one pass of the deployed search, and the entry threshold of
one uniformly sampled answer.  Only the first is expensive -- it is m
brute-force scans of the corpus -- so that is what sets the bill.

Reported against the FP32 graph build of the same corpus, which is the other
one-time cost a deployment already pays.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BBQETA_THREADS", "24")
import numpy as np
from common import WORK, CONFIGS, K, load
import pool


def time_gt(X, Q):
    t = time.perf_counter()
    pool.gt_topk(X, Q)
    return time.perf_counter() - t


def time_search(key, g, Sc, Q, ef, nproc=16):
    import multiprocessing as mp
    pool._G.update(g)
    pool._G.update(Sc=Sc, Q=Q, GT=np.zeros((len(Q), K), dtype=np.int64))
    t = time.perf_counter()
    with mp.Pool(nproc) as p:
        list(p.imap_unordered(pool._idwork,
                              [(i, ef) for i in range(len(Q))], chunksize=16))
    return time.perf_counter() - t


def run(key, m=1000, ef=512, nproc=16):
    X, Q, cfg = load(key, nq=m)
    N, d = X.shape
    out = dict(key=key, corpus=cfg["corpus"], encoder=cfg["encoder"],
               N=int(N), d=int(d), m=int(len(Q)), ef=int(ef))

    out["gt_s"] = time_gt(X, Q)

    g = pool.build_graph(key, X)          # cached; the build is timed separately
    Sc = np.sign(X).astype(np.int8)
    out["sign_s"] = None
    t = time.perf_counter(); np.sign(X).astype(np.int8); out["sign_s"] = time.perf_counter() - t
    del X
    out["search_s"] = time_search(key, g, Sc, Q, ef, nproc)

    # the rule itself: one order statistic over m sampled thresholds
    rng = np.random.default_rng(0)
    z = np.load(f"{WORK}/pool_{key}_M24_ef{ef}.npz")["gold_lv"]
    t = time.perf_counter()
    for _ in range(100):
        J = rng.integers(0, K, size=m)
        T = z[:m][np.arange(m), J]
        ell = int(np.ceil((m + 1) * 0.99))
        _ = np.sort(T)[ell - 1] if ell <= m else np.inf
    out["rule_s"] = (time.perf_counter() - t) / 100
    out["total_s"] = out["gt_s"] + out["search_s"] + out["rule_s"]
    return out


if __name__ == "__main__":
    res = []
    for key in sys.argv[1:]:
        r = run(key)
        res.append(r)
        print(f"{r['corpus']}/{r['encoder']:16s} N={r['N']} d={r['d']} m={r['m']}  "
              f"GT {r['gt_s']:7.1f}s  search {r['search_s']:5.1f}s  "
              f"rule {r['rule_s']*1e3:5.1f}ms  total {r['total_s']:7.1f}s", flush=True)
    json.dump(res, open("/home/sieun/projects/BBQETA/out/calibcost.json", "w"), indent=1)
