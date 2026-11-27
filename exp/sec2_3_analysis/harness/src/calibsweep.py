"""Target sweep for the mean-recall guarantee, on one fixed search setting.

Theorem 1 promises a marginal mean: the expectation runs over the calibration
split, the sampled ground-truth position and the test query jointly.  So the
check is to repeat the whole calibration many times at each target and ask
whether the *average over repetitions* clears the target -- not whether any one
repetition does.  Replications that certify nothing are kept in the average at
their guaranteed value, recall 1 under the exhaustive fallback, and their share
is reported separately because that is what the deployment would have to pay.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import calib
from common import K

KEY, EF, N = "msmarco_qw06", 256, 1_000_000
TARGETS = [0.90, 0.95, 0.975, 0.99, 0.995]


def run(m=1000, B=400, seed=20261119):
    ps, gl, _ = calib.load_pool(KEY, EF)
    pi = float(1 - np.isfinite(gl).mean())
    rows = []
    for t in TARGETS:
        rho = 1 - t
        rng = np.random.default_rng(seed)
        rep = [calib.replicate(ps, gl, rho, m, rng, N) for _ in range(B)]
        rec = np.array([r["recall"] for r in rep])
        fin = np.array([np.isfinite(r["tau"]) for r in rep])
        R = np.array([r["R_mean"] for r in rep])
        rows.append(dict(
            target=t, rho=rho, pi=pi, m=m, B=B,
            mean=float(rec.mean()), sd=float(rec.std(ddof=1)),
            p05=float(np.quantile(rec, .05)), p95=float(np.quantile(rec, .95)),
            escal=float(1 - fin.mean()),
            recall_fin=float(rec[fin].mean()) if fin.any() else float("nan"),
            R_fin=float(R[fin].mean()) if fin.any() else float("nan"),
            ER=float((fin.mean()) * (R[fin].mean() if fin.any() else 0.0)
                     + (1 - fin.mean()) * N)))
        r = rows[-1]
        print(f"target {t:.3f}  achieved {r['mean']:.5f} "
              f"[{r['p05']:.4f},{r['p95']:.4f}]  escal {r['escal']:.3f}  "
              f"R_fin {r['R_fin']:7.1f}  E[R] {r['ER']:9.0f}", flush=True)
    json.dump(dict(key=KEY, ef=EF, pi=pi, rows=rows),
              open("out/calibsweep.json", "w"), indent=1)

    with open("gp/motiv/calsweep.dat", "w") as fh:
        fh.write(f"# {KEY} ef={EF}, pi={pi:.5f}, m={m}, {B} replications\n")
        fh.write("# i target achieved p05 p95 escal R_fin\n")
        for i, r in enumerate(rows, 1):
            fh.write(f"{i} {r['target']:.4f} {r['mean']:.5f} {r['p05']:.5f} "
                     f"{r['p95']:.5f} {r['escal']:.4f} {r['R_fin']:.1f}\n")
    print("wrote gp/motiv/calsweep.dat")


if __name__ == "__main__":
    run()
