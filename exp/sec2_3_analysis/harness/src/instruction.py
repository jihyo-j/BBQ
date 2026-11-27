"""The query-side instruction as part of the embedding term (Section 4.2).

Ablates the instruction the model card prescribes, against identical document
vectors and identical queries, and prices it against the alternatives Section
4.1 costs out.  Four conditions plus one linear control:

  as prescribed      the recipe the rest of the paper uses
  none               the raw query
  mismatched         same template, different task  (MISMATCH in encoders.py)
  null               same template, no task         (NULL)
  u_X removed        corpus mean direction projected out of the raw query

The linear control is the point of the exercise: if the instruction were only
removing the query's component along the corpus mean direction, doing that
arithmetically at the index would reproduce it.  It does not.  On MS MARCO it
recovers beta to 0.736 of the 0.523 -> 0.825 gap (70%) but p95 R* only to 60 of
the 74 -> 47 gap (half), because projecting also reweights every document by
its own projection on u_X.

Measured (MS MARCO / Qwen3-0.6B, 2000 queries):
  prescribed 0.825 / 47   none 0.523 / 74   mismatched 0.325 / 130
  null 0.668 / 62         u_X removed 0.736 / 60
and on Quora, where the instruction names a task the corpus is not (duplicate
questions, not passages that answer a query), 0.889 / 35 against 0.740 / 30:
half the beta and no budget.
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, fbin, OUT, BEIR, K, RANK_FIT
import encoders as EN

MS = "/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/"
_LR = np.log(np.arange(1, RANK_FIT + 1.0))
_LRC = _LR - _LR.mean()


def beta_and_rstar(X, B, Q):
    """beta over exact ranks 1..RANK_FIT and p95 R* under ADC-BQ, per query.

    R* uses the Appendix-A total order: ties broken by ascending identifier.
    """
    n = Q.shape[0]
    R = np.zeros(n)
    be = np.zeros(n)
    for qi in range(n):
        t = X @ Q[qi]
        a = B @ Q[qi]
        gt = np.argpartition(-t, RANK_FIT + 10)[:RANK_FIT + 10]
        gt = gt[np.argsort(-t[gt], kind="stable")]
        z = (t[gt[:RANK_FIT]] - t.mean()) / t.std()
        be[qi] = -((z - z.mean()) @ _LRC) / (_LRC @ _LRC)
        R[qi] = max(int((a > a[g]).sum()) + int((a[:g] == a[g]).sum())
                    for g in gt[:K]) + 1
    return float(be.mean()), float(np.percentile(R, 95)), float(np.median(R))


def signs(X):
    B = np.sign(X).astype(np.float32)
    B[B == 0] = 1.0
    return B


def msmarco_texts(n):
    return [json.loads(l)["text"] for l in open(MS + "msmarco-1m_queries.jsonl")][:n]


def quora_texts(n):
    """Query order used by the released Quora vectors: the test-qrels query ids
    sorted as strings.  Verified on 5/5 spot checks and by the encoder gate."""
    qj = json.load(open(f"{BEIR}/quora_queries.json"))
    qmap = dict(zip(qj["id"], qj["text"]))
    seen = []
    with open(f"{BEIR}/quora_qrels_test.tsv") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for a, _c, _s in r:
            if a not in seen:
                seen.append(a)
    return [qmap[i] for i in sorted(set(seen))][:n]


def run(key, texts, nq=2000, model="qwen06", linear_control=True):
    X, Q, _ = load(key, nq)
    ok, ca, cb = EN.gate(texts, Q, model)
    print(f"  gate: with={ca:.5f} without={cb:.5f}", flush=True)
    if not ok:
        print("  GATE FAILED - query order or pooling does not correspond; stopping")
        return None
    B = signs(X)
    res = {}
    res["as prescribed"] = beta_and_rstar(X, B, Q)
    conds = [("none", None),
             ("mismatched", EN.MISMATCH),
             ("null", EN.NULL)]
    enc = {}
    for nm, pre in conds:
        enc[nm] = EN.encode(texts[:nq], model, pre, log=nm)
        res[nm] = beta_and_rstar(X, B, enc[nm])
    if linear_control:
        uX = X[:200_000].mean(0)
        uX /= np.linalg.norm(uX)
        Qr = enc["none"] - np.outer(enc["none"] @ uX, uX)
        Qr /= np.linalg.norm(Qr, axis=1, keepdims=True)
        res["u_X removed"] = beta_and_rstar(X, B, Qr)
        # how much of the query's energy the shared direction takes, which is
        # the quantity the instruction actually moves (29.9% -> 8.4%)
        res["_uX_share"] = dict(prescribed=float(np.mean((Q @ uX) ** 2)),
                                none=float(np.mean((enc["none"] @ uX) ** 2)))
    return res


if __name__ == "__main__":
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    out = {}
    for key, texts in (("msmarco_qw06", msmarco_texts(nq)),
                       ("quora_qw06", quora_texts(nq))):
        print(f"\n=== {key} ===", flush=True)
        out[key] = run(key, texts, nq)
        json.dump(out, open(f"{OUT}/instruction.json", "w"), indent=1)
    print(f"\n{'setting':16s}{'condition':16s}{'beta':>8s}{'p95 R*':>9s}{'med':>7s}")
    for k, v in out.items():
        if not v:
            continue
        for c, val in v.items():
            if c.startswith("_"):
                continue
            print(f"{k:16s}{c:16s}{val[0]:8.3f}{val[1]:9.0f}{val[2]:7.0f}")
