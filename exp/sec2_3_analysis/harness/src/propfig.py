"""Data for the panel that checks Proposition 5.

The proposition says the rule escalates exactly when the sampled calibration
answer has no finite entry threshold more often than the target miss rate:
P[tau = inf] = P[Binomial(m, pi) > m - l], which tends to 0 when pi < rho and
to 1 when pi > rho.  This sweeps m at five (configuration, width) cells chosen
to straddle pi = rho, and reports the measured rate beside that prediction.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import binom
import calib

RHO = 0.01
CELLS = [("quora_qw06",  1024, "Quora/Qwen"),
         ("msmarco_qw06", 512, "MS MARCO/Qwen"),
         ("msmarco_bgem3",1024,"MS MARCO/BGE-M3"),
         ("nq_qw06",      128, "NQ/Qwen"),
         ("fever_bgem3", 1024, "FEVER/BGE-M3")]
MS = [100, 200, 400, 800, 1600, 3000]


def main(B=400):
    rows = []
    for key, ef, nice in CELLS:
        ps, gl, _ = calib.load_pool(key, ef)
        pi = float(1 - np.isfinite(gl).mean())
        nq = len(gl)
        pts = []
        for m in MS:
            if m > nq // 2:
                continue
            c = calib.campaign(key, ef, rho=RHO, m=m, B=B)
            pts.append((c["m"], c["fallback"], c["fallback_pred"]))
        rows.append(dict(key=key, ef=ef, nice=nice, pi=pi, nq=nq, pts=pts))
        print(f"{nice:18s} ef={ef:5d} pi={pi:.5f} "
              + "  ".join(f"m{m}:{f:.3f}/{p:.3f}" for m, f, p in pts), flush=True)
    json.dump(rows, open("out/propfig.json", "w"), indent=1)

    # curves: measured points, and the binomial prediction on a fine m grid
    with open("gp/motiv/prop_pts.dat", "w") as fh:
        fh.write("# measured escalation rate, rho=0.01, 400 replicates\n")
        fh.write("# m  rate  (one block per cell, in CELLS order)\n")
        for r in rows:
            fh.write(f"# {r['nice']} ef={r['ef']} pi={r['pi']:.5f}\n")
            for m, f, p in r["pts"]:
                fh.write(f"{m} {f:.4f}\n")
            fh.write("\n\n")
    grid = np.unique(np.round(np.logspace(np.log10(90), np.log10(3300), 60)).astype(int))
    with open("gp/motiv/prop_pred.dat", "w") as fh:
        fh.write("# binomial prediction P[Bin(m,pi) > m-l], l=ceil((m+1)(1-rho))\n")
        for r in rows:
            fh.write(f"# {r['nice']} pi={r['pi']:.5f}\n")
            for m in grid:
                ell = int(np.ceil((m + 1) * (1 - RHO)))
                fh.write(f"{m} {float(binom.sf(m - ell, m, r['pi'])):.4f}\n")
            fh.write("\n\n")
    print("wrote gp/motiv/prop_pts.dat and prop_pred.dat")


if __name__ == "__main__":
    main()
