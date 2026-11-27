"""Smallest k whose EXACT top-k holds *every* judged-relevant document of a query."""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, OUT
import measure as MS
from ktune import msmarco_qrels, quora_qrels, SETS

R100 = 100


def run(keys):
    out = {}
    for key in keys:
        rel = SETS[key]()
        X, Q, c = load(key, 1000)
        N, d = X.shape; nq = Q.shape[0]
        ex = MS.exact_side(X, Q, nq, N, d)
        gt = ex["gt_id"]                       # nq x 100, exact order
        last, nrel = [], []
        for i in range(nq):
            r = rel[i]
            if not r:
                continue
            pos = {int(doc): j + 1 for j, doc in enumerate(gt[i].tolist())}
            ranks = [pos.get(doc) for doc in r]
            nrel.append(len(r))
            last.append(max(ranks) if all(p is not None for p in ranks) else None)
        cov = {k: float(np.mean([l is not None and l <= k for l in last]))
               for k in (1, 3, 5, 10, 20, 50, 100)}
        got = np.array([l for l in last if l is not None])
        out[key] = dict(corpus=c["corpus"], encoder=c["encoder"], nq=len(last),
                        nrel=float(np.mean(nrel)),
                        frac_all_within_100=float(np.mean([l is not None for l in last])),
                        p50=float(np.median(got)), p90=float(np.quantile(got, .9)),
                        p95=float(np.quantile(got, .95)), coverage=cov)
        o = out[key]
        print(f"\n{o['corpus']}/{o['encoder']}  ({o['nrel']:.2f} judged per query)")
        print(f"   queries whose judged set is fully inside the exact top-100: {o['frac_all_within_100']:.3f}")
        print(f"   rank of the LAST judged doc (those that are): p50 {o['p50']:.0f}  "
              f"p90 {o['p90']:.0f}  p95 {o['p95']:.0f}")
        print("   fraction of queries fully covered at k = " +
              "  ".join(f"{k}:{v:.3f}" for k, v in cov.items()))
        del X, Q
    json.dump(out, open(f"{OUT}/kfull.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1:] or ["msmarco_qw06", "quora_qw06", "quora_bgem3"])
