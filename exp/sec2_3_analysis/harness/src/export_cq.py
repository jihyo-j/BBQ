"""Write panel (d) of Figure 3 as the score distortion itself.

The panel used to be normalised to ADC-BQ, which hides the level: a reader
could not tell whether a 0.55 on the OPQ tick meant a small error or a large
one.  This writes c_Q = nu/kappa as measured, per (configuration, quantizer).
The quantizer order is the panel's x axis.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import OUT

GP = "/home/sieun/projects/BBQETA/gp"
ARMS = [("sym_bq", "Sym"), ("adc_bq", "ADC"), ("tq1", "TQ"),
        ("adc_bq_cb", "+cs"), ("rabitq1", "RaB"), ("pq", "PQ"), ("opq", "OPQ")]
D1024 = ["msmarco_qw06", "nq_qw06", "quora_qw06", "dbpedia_qw06",
         "msmarco_bgem3", "nq_bgem3", "quora_bgem3", "dbpedia_bgel"]
EXTRA = ["msmarco_e5m", "glove"]

def cq(key):
    a = json.load(open(f"{OUT}/{key}.json"))["arms"]
    return [a[arm]["nu"] / a[arm]["kappa"] for arm, _ in ARMS]

cols = {k: cq(k) for k in D1024 + EXTRA}
geo = [float(np.exp(np.mean(np.log([cols[k][i] for k in D1024]))))
       for i in range(len(ARMS))]

with open(f"{GP}/p_cq_abs.dat", "w") as f:
    f.write("# idx " + " ".join(D1024) + " geomean " + " ".join(EXTRA) + "\n")
    for i, (_, name) in enumerate(ARMS):
        row = [cols[k][i] for k in D1024] + [geo[i]] + [cols[k][i] for k in EXTRA]
        f.write(f"{i} " + " ".join(f"{v:.4f}" for v in row) + f"   # {name}\n")

flat = [v for k in cols for v in cols[k]] + geo
print(f"p_cq_abs.dat written; c_Q range {min(flat):.3f} .. {max(flat):.3f}")
for i, (_, name) in enumerate(ARMS):
    print(f"  {name:4s} geomean {geo[i]:.3f}   E5 {cols['msmarco_e5m'][i]:.3f}"
          f"   GloVe {cols['glove'][i]:.3f}")
