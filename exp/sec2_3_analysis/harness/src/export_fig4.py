"""Write the gnuplot data files for Figure 1.

Panels (a) and (d) show the same eight settings, so they are generated from
one list.  FEVER and ada-002 are out of scope for the paper; the eighth slot
that FEVER used to occupy is now MS MARCO E5-Mistral, which is also the
embedding Section 4.3 needs in the figure because it is the one place where a
rotation clearly pays.

  p_z6.dat   rank r, then z(r) for each setting over ranks 1..100
  p_rk6.dat  k,      then p95 R*(k) for each setting over k = 1..10
  p_dimrel.dat  the MRL ladder for panel (c), each quantity relative to d=32
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import WORK, OUT

GP = "/home/sieun/projects/BBQETA/gp"

# order must match the line styles ls 21..28 and the shared legend in fig4.gp
PANEL = [("quora_qw06",   "Quora"),
         ("nq_qw06",      "NQ"),
         ("dbpedia_qw06", "DBpedia"),
         ("msmarco_qw06", "MS MARCO"),
         ("msmarco_bgem3", "MSM/BGE"),
         ("msmarco_e5m",  "MSM/E5"),
         ("glove",        "GloVe"),
         ("gist",         "GIST")]


def zcurves(path=f"{GP}/p_z6.dat"):
    cols = [np.load(f"{WORK}/{k}_perq.npz")["z_curve"] for k, _ in PANEL]
    with open(path, "w") as f:
        f.write("# r  " + "  ".join(n for _, n in PANEL) + "\n")
        for r in range(len(cols[0])):
            f.write(f"{r+1} " + " ".join(f"{c[r]:.4f}" for c in cols) + "\n")
    return path


def zcurve_points(path=f"{GP}/p_z6_pts.dat", ranks=(1, 2, 3, 5, 8, 13, 21, 34, 55, 89)):
    """Marker positions for panel (a): a log-spaced subset of the same curves."""
    cols = [np.load(f"{WORK}/{k}_perq.npz")["z_curve"] for k, _ in PANEL]
    with open(path, "w") as f:
        f.write("# r  " + "  ".join(n for _, n in PANEL) + "\n")
        for r in ranks:
            f.write(f"{r} " + " ".join(f"{c[r-1]:.4f}" for c in cols) + "\n")
    return path


def budget_by_k(path=f"{GP}/p_rk6.dat", arm="adc_bq"):
    cols = []
    for k, _ in PANEL:
        d = json.load(open(f"{OUT}/{k}.json"))
        cols.append(d["arms"][arm]["p95_Rstar_by_rank"])
    with open(path, "w") as f:
        f.write("# k  " + "  ".join(n for _, n in PANEL) + "\n")
        for i in range(len(cols[0])):
            f.write(f"{i+1} " + " ".join(f"{c[i]:.0f}" for c in cols) + "\n")
    return path


# panel (b): the ladder, every 1024-d configuration plus the two drawn apart
LADDER = ["sym_bq", "adc_bq", "tq1", "adc_bq_cb", "rabitq1", "pq", "opq"]
SETS_1024 = ["msmarco_qw06", "nq_qw06", "quora_qw06", "dbpedia_qw06",
             "msmarco_bgem3", "nq_bgem3", "quora_bgem3", "dbpedia_bgel"]
APART = ["msmarco_e5m", "glove"]


def ladder(path=f"{GP}/p_cq.dat"):
    """c_Q of each arm divided by ADC-BQ's on the same embedding."""
    def col(key):
        a = json.load(open(f"{OUT}/{key}.json"))["arms"]
        base = a["adc_bq"]["nu"] / a["adc_bq"]["kappa"]
        return [(a[q]["nu"] / a[q]["kappa"]) / base for q in LADDER]
    cols = {k: col(k) for k in SETS_1024 + APART}
    gm = [float(np.exp(np.mean(np.log([cols[k][i] for k in SETS_1024]))))
          for i in range(len(LADDER))]
    names = SETS_1024 + ["geomean"] + APART
    with open(path, "w") as f:
        f.write("# idx " + " ".join(names) + "\n")
        for i in range(len(LADDER)):
            row = [cols[k][i] for k in SETS_1024] + [gm[i]] + [cols[k][i] for k in APART]
            f.write(f"{i} " + " ".join(f"{v:.4f}" for v in row) + "\n")
    return path


def dimrel(path=f"{GP}/p_dimrel.dat", key="msmarco_qw8", arm="adc_bq"):
    d = json.load(open(f"{OUT}/dimsweep_{key}.json"))
    def vals(r):
        a = r["arms"][arm]; c = a["c_Q"]; b = r["beta"]
        return r["d"], r["f_sign_cos"], c, a["Rstar_p95"], b, c / b
    base = vals(d["rows"][0])
    with open(path, "w") as f:
        f.write("# d  f  c_Q  Rstar  beta  eta   (each relative to its value at d=32)\n")
        for r in d["rows"]:
            v = vals(r)
            f.write(f"{v[0]:6d} " + " ".join(f"{v[i]/base[i]:.6f}" for i in range(1, 6)) + "\n")
    return path


if __name__ == "__main__":
    for p in (zcurves(), zcurve_points(), budget_by_k(), ladder(), dimrel()):
        print("wrote", p)
