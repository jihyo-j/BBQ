"""Per-query Recall@10 under the calibrated rule, as a distribution.

Two things change against the earlier sweep.

A replicate whose threshold is +inf certifies nothing.  The earlier panel kept
those in the average at recall 1, the value the exhaustive fallback guarantees;
here they are reported as what they are -- a failure to certify -- and the
remaining statistics are taken over the replicates that did certify.  The two
conventions disagree only where escalation is common, and that is exactly
where the difference matters: the guaranteed value costs a full scan.

And the panel now shows the distribution over queries rather than the mean
alone.  Recall@10 is a count out of ten, so it has eleven atoms; the share of
queries at each of them is pooled over the certifying replicates, which keeps
the average marginal over the calibration draw the way the theorem is.

  out/calqdist.json   every number below
  gp/motiv/calqdist.dat   the figure's data
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import calib
from common import K

KEY, EF, N = "msmarco_qw06", 256, 1_000_000
TARGETS = [0.90, 0.95, 0.975, 0.99, 0.995]
GP = "/home/sieun/projects/BBQETA/gp/motiv"


def run(m=1000, B=400, seed=20261119):
    pool_s, gold_lv, _ = calib.load_pool(KEY, EF)
    nq = len(gold_lv)
    pi = float(1 - np.isfinite(gold_lv).mean())
    rows = []
    for t in TARGETS:
        rho, rng = 1 - t, np.random.default_rng(seed)
        hist = np.zeros(K + 1)            # counts of per-query recall 0/10..10/10
        means, Rs, nfin = [], [], 0
        for _ in range(B):
            perm = rng.permutation(nq)
            cal, tst = perm[:m], perm[m:]
            J = rng.integers(0, K, size=m)
            Z = gold_lv[cal, J]
            ell = int(np.ceil((m + 1) * (1.0 - rho)))
            tau = np.inf if ell > m else float(np.sort(Z)[ell - 1])
            if not np.isfinite(tau):
                continue                   # certifies nothing: a failure
            nfin += 1
            hit = (gold_lv[tst] <= tau).sum(1)          # answers kept, per query
            hist += np.bincount(hit, minlength=K + 1)
            means.append(hit.mean() / K)
            Rs.append(calib._csize(pool_s[tst], tau).mean())
        share = hist / hist.sum()
        rows.append(dict(target=t, B=B, certified=nfin, fail=1 - nfin / B,
                         mean=float(np.mean(means)),
                         margin_pp=float(100 * (np.mean(means) - t)),
                         R_mean=float(np.mean(Rs)),
                         share=[float(v) for v in share],
                         perfect=float(share[K]), nine=float(share[K - 1]),
                         eight_or_less=float(share[:K - 1].sum())))
        r = rows[-1]
        print(f"target {t:.3f}  certified {nfin}/{B} (fail {100*r['fail']:5.1f}%)"
              f"  mean {r['mean']:.5f} ({r['margin_pp']:+.3f} pp)"
              f"  perfect {100*r['perfect']:5.2f}%  9/10 {100*r['nine']:4.2f}%"
              f"  <=8 {100*r['eight_or_less']:4.2f}%  reads {r['R_mean']:.1f}")
    json.dump(dict(key=KEY, ef=EF, pi=pi, m=m, B=B, rows=rows),
              open("out/calqdist.json", "w"), indent=1)

    with open(f"{GP}/calqdist.dat", "w") as fh:
        fh.write(f"# {KEY} ef={EF}, m={m}, {B} replications, pi={pi:.5f}\n")
        fh.write("# per-query Recall@10 shares, pooled over certifying replicates;\n"
                 "# replicates with tau=inf are counted as failures, not as recall 1.\n")
        fh.write("# i target perfect nine eight_or_less fail mean margin_pp reads\n")
        for i, r in enumerate(rows, 1):
            fh.write(f"{i} {r['target']:.4f} {r['perfect']:.5f} {r['nine']:.5f} "
                     f"{r['eight_or_less']:.5f} {r['fail']:.4f} {r['mean']:.5f} "
                     f"{r['margin_pp']:.3f} {r['R_mean']:.1f}\n")
    print(f"wrote {GP}/calqdist.dat and out/calqdist.json")


if __name__ == "__main__":
    run()
