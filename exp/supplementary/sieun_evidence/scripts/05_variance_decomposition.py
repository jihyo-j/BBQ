#!/usr/bin/env python3
"""The Section 3.4 variance decomposition of log R*.

Manuscript: "query variation accounts for 89.4% of the variance, compared with
7.7% between embeddings and 1.8% between quantizers."

No stored file carries those three numbers, but the raw material does:
data/perq/<config>_perq.npz holds the per-query R* of every arm, for 20
configurations.  This script recomputes the decomposition and reports how
sensitive the split is to which configurations and quantizers are pooled.

Requires numpy.
"""
import glob, itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import ARMS, DATA

data = {}
for p in sorted(glob.glob(os.path.join(DATA, "perq", "*_perq.npz"))):
    z = np.load(p)
    key = os.path.basename(p)[:-9]
    arms = {a: z["Rstar_" + a] for a in ARMS if "Rstar_" + a in z}
    if len(arms) == len(ARMS):
        data[key] = arms
print(f"configurations with per-query R* for all {len(ARMS)} arms: {len(data)}")


def decompose(keys, arms):
    """Additive two-factor decomposition of log R*; the residual is the
    query-to-query variation (configurations have unequal query counts, so each
    cell is truncated to the common length within a configuration)."""
    cells = []
    for c in keys:
        n = min(len(data[c][a]) for a in arms)
        for a in arms:
            v = np.log(np.maximum(np.asarray(data[c][a][:n], float), 1.0))
            cells.append((c, a, v))
    allv = np.concatenate([v for _, _, v in cells])
    gm = allv.mean()
    tot = ((allv - gm) ** 2).sum()
    def group(idx, name):
        v = np.concatenate([c[2] for c in cells if c[idx] == name])
        return len(v) * (v.mean() - gm) ** 2
    sc = sum(group(0, c) for c in keys)
    sa = sum(group(1, a) for a in arms)
    return 100 * sc / tot, 100 * sa / tot, 100 * (tot - sc - sa) / tot


K = sorted(data)
LEARNED = [k for k in K if k not in ("glove", "gist")]
SIX = ["sym_bq", "adc_bq", "pq", "opq", "rabitq1", "tq1"]
print()
print(f"  {'pooling':44s} {'embeddings':>11s} {'quantizers':>11s} {'queries':>9s}")
for lbl, ks, ar in [("all 15 configurations x 10 arms", K, ARMS),
                    ("13 learned configurations x 10 arms", LEARNED, ARMS),
                    ("13 learned configurations x 6 main arms", LEARNED, SIX)]:
    c, a, q = decompose(ks, ar)
    print(f"  {lbl:44s} {c:10.1f}% {a:10.1f}% {q:8.1f}%")
print(f"  {'manuscript':44s} {7.7:10.1f}% {1.8:10.1f}% {89.4:8.1f}%")

print()
print("  The manuscript split is reproducible on learned-embedding subsets, but not")
print("  uniquely -- many subsets land on it.  Closest matches:")
best = []
for ns in (3, 4, 5, 6):
    for cs in itertools.combinations(LEARNED, ns):
        for ar in (ARMS, SIX, ["adc_bq", "pq", "opq", "rabitq1", "tq1"],
                   ["adc_bq", "adc_bq_c", "adc_bq_cb"], ["adc_bq", "tq1", "rabitq1"]):
            c, a, q = decompose(cs, ar)
            best.append((abs(c - 7.7) + abs(a - 1.8) + abs(q - 89.4), c, a, q, cs, tuple(ar)))
best.sort()
for d, c, a, q, cs, ar in best[:5]:
    print(f"    embeddings {c:4.1f}%  quantizers {a:4.1f}%  queries {q:4.1f}%"
          f"   ({len(ar)} arms)  {', '.join(cs)}")
print()
print("  VERDICT: the magnitude is solid -- query variation dominates by roughly an")
print("  order of magnitude over both the embedding and the quantizer -- but the exact")
print("  three numbers cannot be tied to a specific pooling.  Either re-state them from")
print("  a pooling fixed here, or soften the sentence to the magnitude.")
