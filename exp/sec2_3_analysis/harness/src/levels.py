"""The three levels at which the candidate budget varies (Section 4, Table 1).

Decomposes the variance of log R* over the crossed design
(setting x quantizer x query), which is the measurement the whole section is
organised around.  Queries are nested in a setting, so the query term is the
within-cell residual.

Measured, over seven Transformer settings and the five same-size quantizers a
system would deploy: query 85.7%, embedding 10.3%, quantizer 2.1%, interaction
1.9%.  Restricted to the four Qwen-1024 settings of Section 6: 96.7/1.7/1.4/0.2.
Adding Sym-BQ, the one quantizer that also binarizes the query, takes the
quantizer share to 8.5% -- i.e. three quarters of that level is the single
decision to keep the query in fp32, which costs nothing.

`corpus_vs_encoder` separates the two axes the "embedding" level confounds,
using the balanced 3 corpora x 2 encoders grid.  The corpus is the larger term
(9.1% against 3.4%) but only as far as the encoder allows: within BGE-M3 it
explains 18.5% and within Qwen3-0.6B 2.3%.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import WORK, OUT

DEPLOYABLE = ["adc_bq", "pq", "opq", "rabitq1", "tq1"]
TAB = ["msmarco_qw06", "nq_qw06", "quora_qw06", "dbpedia_qw06",
       "msmarco_bgem3", "msmarco_ada2", "msmarco_e5m"]
QWEN = TAB[:4]


def _logR(key, arm):
    z = np.load(f"{WORK}/{key}_perq.npz")
    return np.log(np.maximum(z[f"Rstar_{arm}"], 1).astype(np.float64))


def decompose(keys, arms):
    """Two-way crossed decomposition with queries nested in the setting.

    Returns shares summing to 1: setting, arm, interaction, query.
    """
    cell = {(s, a): _logR(s, a) for s in keys for a in arms}
    n = min(len(v) for v in cell.values())
    M = np.array([[cell[(s, a)][:n] for a in arms] for s in keys])   # S x A x Q
    gm = M.mean()
    S = M.mean((1, 2)) - gm
    A = M.mean((0, 2)) - gm
    SA = M.mean(2) - gm - S[:, None] - A[None, :]
    E = M - M.mean(2, keepdims=True)
    tot = ((M - gm) ** 2).sum()
    nA, nS = len(arms), len(keys)
    return dict(setting=(S ** 2).sum() * nA * n / tot,
                arm=(A ** 2).sum() * nS * n / tot,
                interaction=(SA ** 2).sum() * n / tot,
                query=(E ** 2).sum() / tot,
                n_obs=nS * nA * n)


def corpus_vs_encoder(corpora=("msmarco", "nq", "quora"),
                      encoders=("qw06", "bgem3"), arm="adc_bq"):
    A = []
    for c in corpora:
        A.append([_logR(f"{c}_{e}", arm) for e in encoders])
    n = min(len(v) for r in A for v in r)
    A = np.array([[v[:n] for v in r] for r in A])                   # C x E x Q
    gm = A.mean()
    cc = A.mean((1, 2)) - gm
    ee = A.mean((0, 2)) - gm
    ce = A.mean(2) - gm - cc[:, None] - ee[None, :]
    res = A - A.mean(2, keepdims=True)
    tot = ((A - gm) ** 2).sum()
    return dict(corpus=(cc ** 2).sum() * len(encoders) * n / tot,
                encoder=(ee ** 2).sum() * len(corpora) * n / tot,
                interaction=(ce ** 2).sum() * n / tot,
                query=(res ** 2).sum() / tot)


def corpus_share_within(encoder, corpora=("msmarco", "nq", "quora"), arm="adc_bq"):
    A = [_logR(f"{c}_{encoder}", arm) for c in corpora]
    n = min(map(len, A))
    A = np.array([a[:n] for a in A])
    gm = A.mean()
    b = A.mean(1) - gm
    return float((b ** 2).sum() * n / ((A - gm) ** 2).sum())


if __name__ == "__main__":
    out = {}
    for tag, keys, arms in (("7 settings", TAB, DEPLOYABLE),
                            ("Qwen-1024 only", QWEN, DEPLOYABLE),
                            ("7 settings + Sym-BQ", TAB, ["sym_bq"] + DEPLOYABLE)):
        d = decompose(keys, arms)
        out[tag] = d
        print(f"{tag:22s} query {d['query']:6.1%} | embedding {d['setting']:6.1%} "
              f"| quantizer {d['arm']:6.1%} | inter {d['interaction']:5.1%} "
              f"({d['n_obs']:,} obs)")
    out["corpus_vs_encoder"] = corpus_vs_encoder()
    d = out["corpus_vs_encoder"]
    print(f"\n{'3 corpora x 2 encoders':22s} corpus {d['corpus']:6.1%} | "
          f"encoder {d['encoder']:6.1%} | inter {d['interaction']:5.1%} | "
          f"query {d['query']:6.1%}")
    for e in ("qw06", "bgem3"):
        out[f"corpus_share_{e}"] = corpus_share_within(e)
        print(f"  corpus share within {e:6s}: {out[f'corpus_share_{e}']:6.1%}")
    json.dump(out, open(f"{OUT}/levels.json", "w"), indent=1)
