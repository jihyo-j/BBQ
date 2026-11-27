"""Write the Figure 4 data files from the per-query R* arrays.

  p_cdf.dat / p_cdf_pts.dat
      R, then the fraction of queries whose whole exact top-10 is inside the
      top R, for each setting; the _pts file carries the marker positions.
  p_overpay.dat / p_overpay_pts.dat
      for each setting a (coverage, ratio) pair, where the ratio is
      R_c / mean_i min(R*_i, R_c) at the coverage R_c achieves.

Both quantities are step functions of an integer budget, so sampling them on a
grid of non-integer R (or of coverage targets) draws the steps rather than the
curve.  Both are therefore evaluated **at the integers themselves** and joined:
the CDF at R = 9, 10, 11, ..., and the overpayment as a curve parameterised by
R rather than by a coverage grid.  Markers are a separate, sparse file so that
they can sit evenly on the log axis without thinning the curve.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import WORK

GP = "/home/sieun/projects/BBQETA/gp"

# column order fixes the line styles in fig4p.gp: ls 21, 24, 25
PANEL = [("quora_qw06",    "quora_qw06"),
         ("msmarco_qw06",  "msmarco_qw06"),
         ("msmarco_bgem3", "msmarco_bgem3")]

R = {k: np.load(f"{WORK}/{k}_perq.npz")["Rstar_adc_bq"].astype(float)
     for k, _ in PANEL}
RMAX = 1200
GRID = np.arange(9, RMAX + 1)                      # the integers the step sits on
MARK_R = [10, 16, 25, 40, 63, 100, 160, 250, 400, 630]
MARK_C = [0.80, 0.85, 0.89, 0.92, 0.945, 0.965, 0.978, 0.987, 0.993]


def cov(k, r):
    return (R[k] <= r).mean()


def ratio(k, r):
    return r / np.minimum(R[k], r).mean()


def write(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(" ".join(f"{v:.5f}" if isinstance(v, float) else str(v)
                             for v in row) + "\n")
    return path


def cdf():
    head = "# R  " + "  ".join(n for _, n in PANEL)
    write(f"{GP}/p_cdf.dat", head,
          [[int(r)] + [cov(k, r) for k, _ in PANEL] for r in GRID])
    write(f"{GP}/p_cdf_pts.dat", head,
          [[int(r)] + [cov(k, r) for k, _ in PANEL] for r in MARK_R])


def overpay():
    # one (coverage, ratio) pair per setting: the settings reach a given
    # coverage at different budgets, so each needs its own abscissa
    head = "# R  " + "  ".join(f"{n}_cov {n}_ratio" for _, n in PANEL)
    rows = []
    for r in GRID:
        row = [int(r)]
        for k, _ in PANEL:
            row += [cov(k, r), ratio(k, r)]
        rows.append(row)
    write(f"{GP}/p_overpay.dat", head, rows)
    # markers at round coverage targets, one budget per setting
    mrows = []
    for c in MARK_C:
        row = [c]
        for k, _ in PANEL:
            rc = float(np.quantile(R[k], c, method="higher"))
            row += [cov(k, rc), ratio(k, rc)]
        mrows.append(row)
    write(f"{GP}/p_overpay_pts.dat", "# target  " + head[6:], mrows)


if __name__ == "__main__":
    for k, _ in PANEL:
        x = R[k]
        print(f"{k:16s} n={len(x):5d}  median {np.median(x):6.1f}  "
              f"p95 {np.quantile(x, .95):6.1f}  p99 {np.quantile(x, .99):6.1f}  "
              f"ratio@0.95 {ratio(k, np.quantile(x, .95, method='higher')):.2f}")
    cdf(); overpay()
    print(f"wrote p_cdf{{,_pts}}.dat and p_overpay{{,_pts}}.dat "
          f"({len(GRID)} integer budgets)")
