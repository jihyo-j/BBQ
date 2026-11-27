"""Raw material for the Section 3.2 contrast: z(r) by corpus, and how many
documents a query really has an answer in against the fixed top-10 it is judged
on.

Two axes, measured rather than asserted:

  profile   the exact cosine of the rank-r document, standardized by that
            query's own corpus scores, on four corpora at one encoder
            (Qwen3-0.6B, 1024d) so only the corpus differs, with the budget
            R* it carries into.
  judged    how many documents the relevance judgments actually mark relevant
            for a query, and where those documents land in the exact ranking.

Provenance of the judgments.  MS MARCO carries them inline in
msmarco-1m_queries.jsonl ("rel"), already keyed by corpus row: verified by
locating every judged document under the vectors (median rank 1, 99.5% inside
the top-1000).  Quora needs the id list from BeIR/quora's corpus parquet, which
this writes to $WORK/quora_corpus_ids.json; row order was verified against
quora_corpus_text.json and the mapping spot-checked against query text.  NQ and
DBpedia are 1M subsamples whose rows do not map back, so for those two only the
count per query is available, from qrels, and not where they rank.

  BBQETA_THREADS=24 python3 src/raw_material.py   ->  out/raw_material.txt
"""
import os, sys, csv, json, gzip
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, CONFIGS, WORK, OUT, BEIR, K
import instruction as INS

MS = "/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/"
QRELS = "/home/smrc/samsung-nvme/sieun/beir_emb"
ARMS = ["msmarco_qw06", "nq_qw06", "quora_qw06", "dbpedia_qw06"]
RANKS = (1, 2, 3, 5, 10, 100)
NQ = 2000
SAMPLE = 200_000          # corpus rows used for the mean/sd a query standardizes by


def out(s=""):
    print(s)
    LOG.append(s)


# ----------------------------------------------------------------- judgments
def judged_msmarco():
    """qid -> {row: grade}, inline and already keyed by corpus row."""
    return [{int(k): v for k, v in json.loads(l)["rel"].items()}
            for l in open(MS + "msmarco-1m_queries.jsonl")]


def judged_quora():
    ids = json.load(open(f"{WORK}/quora_corpus_ids.json"))
    row = {c: i for i, c in enumerate(ids)}
    rel = {}
    with open(f"{WORK}/quora_qrels_test.tsv") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for q, c, s in r:
            if float(s) > 0:
                rel.setdefault(q, {})[row[c]] = float(s)
    order = sorted(rel)                       # the query order the vectors use
    return [rel[q] for q in order]


def judged_counts(path):
    """Counts only -- for the corpora whose rows we cannot map back."""
    n = {}
    with open(path) as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for q, _c, s in r:
            if float(s) > 0:
                n[q] = n.get(q, 0) + 1
    return np.array(sorted(n.values()))


# ------------------------------------------------------------------ profiles
def profile(key, rel=None):
    X, Q, _ = load(key, NQ)
    z = np.load(f"{WORK}/{key}_perq.npz")
    nq = len(Q)
    P = np.zeros((nq, len(RANKS)))
    jrank = []                                 # rank of each judged document
    jin10 = []                                 # judged documents inside the top-10
    rs = np.random.default_rng(0).choice(len(X), SAMPLE, replace=False)
    Xs = X[rs]
    mu = np.empty(nq); sd = np.empty(nq)
    for qi in range(nq):
        s = Xs @ Q[qi]
        mu[qi], sd[qi] = s.mean(), s.std()
        t = X @ Q[qi]
        idx = np.argpartition(-t, 120)[:120]
        idx = idx[np.argsort(-t[idx], kind="stable")]
        P[qi] = [t[idx[r - 1]] for r in RANKS]
        if rel is not None and qi < len(rel):
            order = np.argsort(-t, kind="stable")
            pos = {int(v): r + 1 for r, v in enumerate(order[:2000])}
            rk = [pos.get(d, 10 ** 9) for d in rel[qi]]
            jrank += rk
            jin10.append(sum(r <= K for r in rk))
    Z = (P - mu[:, None]) / sd[:, None]
    res = dict(N=len(X), nq=nq, P=P, Z=Z, sd=sd, beta=z["beta"][:nq],
               rstar=z["Rstar_adc_bq"][:nq],
               jrank=np.array(jrank), jin10=np.array(jin10))
    del X, Q
    return res


if __name__ == "__main__":
    LOG = []
    REL = {"msmarco_qw06": judged_msmarco(), "quora_qw06": judged_quora()}
    R = {k: profile(k, REL.get(k)) for k in ARMS}

    out("=" * 74)
    out("1.  z(r) BY CORPUS -- one encoder (Qwen3-0.6B, 1024d), corpus varies")
    out("=" * 74)
    out(f"{'corpus':9s}{'N':>9s}{'sd':>8s}" +
        "".join(f"{'cos@'+str(r):>8s}" for r in RANKS) + f"{'beta':>7s}")
    for k in ARMS:
        r = R[k]
        out(f"{CONFIGS[k]['corpus']:9s}{r['N']:9d}{np.median(r['sd']):8.4f}" +
            "".join(f"{v:8.3f}" for v in np.median(r['P'], 0)) +
            f"{np.median(r['beta']):7.3f}")
    out()
    out(f"{'corpus':9s}" + "".join(f"{'z('+str(r)+')':>8s}" for r in RANKS) +
        f"{'R* med':>8s}{'R* p95':>8s}")
    for k in ARMS:
        r = R[k]
        out(f"{CONFIGS[k]['corpus']:9s}" +
            "".join(f"{v:8.2f}" for v in np.median(r['Z'], 0)) +
            f"{np.median(r['rstar']):8.0f}{np.percentile(r['rstar'], 95):8.0f}")

    out()
    out("=" * 74)
    out("2.  JUDGED ANSWERS PER QUERY vs THE FIXED TOP-10")
    out("=" * 74)
    out(f"{'corpus':9s}{'queries':>8s}{'median':>8s}{'mean':>7s}{'max':>6s}"
        f"{'=1':>7s}{'>10':>7s}   in exact top-10")
    rows = [("MS MARCO", np.array([len(r) for r in REL["msmarco_qw06"]]), "msmarco_qw06"),
            ("Quora",    np.array([len(r) for r in REL["quora_qw06"]]),   "quora_qw06"),
            ("NQ",       judged_counts(f"{QRELS}/nq/qrels_test.tsv"), None),
            ("DBpedia",  judged_counts(f"{QRELS}/dbpedia-entity/qrels_test.tsv"), None)]
    for name, c, key in rows:
        tail = "--  (rows do not map back)"
        if key:
            j = R[key]["jin10"]
            n = np.array([len(r) for r in REL[key]])[:len(j)]
            tail = (f"{j.mean():.2f} of {n.mean():.2f} judged, "
                    f"{(j == n).mean():.0%} of queries complete")
        out(f"{name:9s}{len(c):8d}{np.median(c):8.0f}{c.mean():7.2f}{c.max():6d}"
            f"{(c == 1).mean():7.1%}{(c > K).mean():7.1%}   {tail}")
    for key, name in (("msmarco_qw06", "MS MARCO"), ("quora_qw06", "Quora")):
        jr = R[key]["jrank"]
        out(f"   {name}: judged document rank -- median {np.median(jr):.0f}, "
            f"p95 {np.percentile(jr, 95):.0f}, inside top-10 {(jr <= K).mean():.1%}")

    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/raw_material.txt", "w").write("\n".join(LOG) + "\n")
    np.savez(f"{OUT}/raw_material.npz",
             **{f"{k}_{f}": R[k][f] for k in ARMS for f in ("P", "Z", "sd", "beta", "rstar")})
    print(f"\nwrote {OUT}/raw_material.txt and raw_material.npz")
