"""The two documents-level examples that go with out/raw_material.txt.

Prints, for one Quora query and one MS MARCO query, the exact FP32 top-10 with
the cosine, the standardized height z(r), and a mark on the documents the
relevance judgments actually call relevant.  That mark is the point: the top-10
is a fixed window, the judged answer set is not.

Quora quotes its documents (its vectors are the full BEIR corpus in native row
order); MS MARCO cannot (its rows are a 1M subsample that does not map back to
the released passages), so its documents are shown by score and judgment only.

Queries are chosen by a rule, not by hand: each is the query whose standardized
profile over ranks 1..100 is closest to its own corpus's median.

  BBQETA_THREADS=24 python3 src/raw_examples.py  ->  out/raw_examples.txt
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, WORK, OUT, K
import instruction as INS
from raw_material import judged_msmarco, judged_quora, MS, RANKS, NQ, SAMPLE

LOG = []
def out(s=""):
    print(s); LOG.append(s)


def dump(key, qtext, rel, ctext=None, topn=12):
    X, Q, _ = load(key, NQ)
    z = np.load(f"{WORK}/{key}_perq.npz")
    nq = len(Q)
    rs = np.random.default_rng(0).choice(len(X), SAMPLE, replace=False)
    Xs = X[rs]
    P = np.zeros((nq, len(RANKS))); mu = np.empty(nq); sd = np.empty(nq)
    for qi in range(nq):
        s = Xs @ Q[qi]; mu[qi], sd[qi] = s.mean(), s.std()
        t = X @ Q[qi]
        idx = np.argpartition(-t, 120)[:120]
        idx = idx[np.argsort(-t[idx], kind="stable")]
        P[qi] = [t[idx[r - 1]] for r in RANKS]
    Z = (P - mu[:, None]) / sd[:, None]
    qi = int(np.argsort(np.abs(Z - np.median(Z, 0)).sum(1))[0])

    t = X @ Q[qi]
    ex = np.argsort(-t, kind="stable")
    judged = rel[qi] if qi < len(rel) else {}
    out(f'\n{key}   q#{qi}   "{qtext[qi]}"')
    out(f'   judged relevant: {len(judged)} document(s);  '
        f'beta={z["beta"][qi]:.3f}  R*={int(z["Rstar_adc_bq"][qi])}')
    out(f'   {"r":>3s}{"cos":>8s}{"z(r)":>7s}  judged  document')
    for r in range(topn):
        i = int(ex[r])
        mark = " yes " if i in judged else "  .  "
        doc = f'"{ctext[i][:60]}"' if ctext else "(text not recoverable)"
        out(f'   {r+1:3d}{t[i]:8.3f}{(t[i]-mu[qi])/sd[qi]:7.2f}  {mark}  {doc}')
    miss = [d for d in judged if d not in set(ex[:topn].tolist())]
    for d in miss:
        r = int(np.where(ex == d)[0][0])
        doc = f'"{ctext[d][:60]}"' if ctext else "(text not recoverable)"
        out(f'   {r+1:3d}{t[d]:8.3f}{(t[d]-mu[qi])/sd[qi]:7.2f}   yes   {doc}'
            f'   <- judged, outside the window above')
    del X, Q


if __name__ == "__main__":
    dump("quora_qw06", INS.quora_texts(NQ), judged_quora(),
         json.load(open(f"{WORK}/quora_corpus_text.json")))
    dump("msmarco_qw06",
         [json.loads(l)["text"] for l in open(MS + "msmarco-1m_queries.jsonl")],
         judged_msmarco())
    os.makedirs(OUT, exist_ok=True)
    open(f"{OUT}/raw_examples.txt", "w").write("\n".join(LOG) + "\n")
    print(f"\nwrote {OUT}/raw_examples.txt")
