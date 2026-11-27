"""Where in the ranking the distortion falls (Section 4.1).

Measured on the BGE / Qwen / E5 configurations the paper evaluates;
ada-002 and Nemotron are out of scope.

c_Q and kappa are corpus-wide averages, so they cannot see that a sign code's
per-document error scales with sqrt(1 - rho^2) in that document's own cosine
rho and is therefore smallest exactly at the top ranks, where the ranking is
decided.  Measuring the same residual a/kappa - t at the exact top-100 gives
sigma_top, which is the instrument c_Q should have been.

Over the 25 (configuration, quantizer) cells measured here, sigma_top orders
p95 R* with Spearman +0.92 against c_Q's +0.60; c_Q places ADC-BQ fourth or
fifth of five in every configuration while sigma_top recovers its true rank in
four of five.  That is why ADC-BQ beats RaBitQ-1bit and PQ on ada-002 despite
a higher c_Q: 63 and 66 candidates against 106/149 and 88/103.

`top_residual.py` is the two-arm version of this (plain ADC-BQ against
centring) and is what the centring paradox in Section 4.1 is measured with.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import OUT
import top_residual as TR
import tables as T

ARMS = ("adc_bq", "tq1", "rabitq1", "pq", "opq")
KEYS = ["msmarco_qw06", "nq_qw06", "quora_qw06", "dbpedia_qw06",
        "msmarco_bgem3", "nq_bgem3", "quora_bgem3",
        "dbpedia_bgel", "dbpedia_bgeb", "msmarco_e5m"]


def run(keys=KEYS, arms=ARMS):
    R = T.load_all()
    out = {}
    for k in keys:
        out[k] = TR.run(k, arms=arms)
        for a, v in out[k].items():
            v["Rstar_p95"] = R[k]["arms"][a]["Rstar_p95"]
        json.dump(out, open(f"{OUT}/top_residual_arms.json", "w"), indent=1)
    return out


def compare(res):
    """Within each configuration, does c_Q or sigma_top order p95 R* better?

    Pooled Spearman after centring each configuration's logs, so the comparison
    is about the quantizer ordering and not about the spread between settings.
    """
    from scipy.stats import spearmanr
    CQ, ST, RS, rows = [], [], [], []
    for k, v in res.items():
        arms = list(v)
        cq = np.log([v[a]["c_Q"] for a in arms])
        st = np.log([v[a]["sigma_top"] for a in arms])
        rs = np.log([v[a]["Rstar_p95"] for a in arms])
        i = arms.index("adc_bq")
        rows.append((k, spearmanr(cq, rs).statistic, spearmanr(st, rs).statistic,
                     int(np.argsort(np.argsort(cq))[i]) + 1,
                     int(np.argsort(np.argsort(st))[i]) + 1,
                     int(np.argsort(np.argsort(rs))[i]) + 1))
        CQ += list(cq - cq.mean()); ST += list(st - st.mean()); RS += list(rs - rs.mean())
    return rows, float(spearmanr(CQ, RS).statistic), float(spearmanr(ST, RS).statistic)


if __name__ == "__main__":
    res = run()
    rows, rho_cq, rho_st = compare(res)
    print(f"\n{'config':16s}{'rho(c_Q,R*)':>13s}{'rho(sig_top,R*)':>17s}"
          f"{'ADC rank by c_Q':>17s}{'by sig_top':>12s}{'true':>6s}")
    for k, a, b, i, j, t in rows:
        print(f"{k:16s}{a:13.3f}{b:17.3f}{i:>15d}/5{j:>10d}/5{t:>5d}/5")
    print(f"\npooled within-config: rho(c_Q, R*) {rho_cq:+.3f}   "
          f"rho(sigma_top, R*) {rho_st:+.3f}")
