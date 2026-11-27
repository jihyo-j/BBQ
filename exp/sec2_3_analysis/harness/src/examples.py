"""The raw-data examples that open each subsection of Section 4.

Only Quora can be shown as text: it is the one corpus whose vectors are the
full BEIR corpus in its native row order (522,931 rows, verified by inspecting
nearest neighbours).  MS MARCO, NQ and DBpedia are 1M subsamples whose row
order cannot be mapped back to the released text, so the cross-encoder example
is shown as scores against the query text only, which the pack does provide.

  quantizer   one Quora query whose exact top-10 the 1-bit code reorders
  embedding   one MS MARCO query under two encoders over IDENTICAL documents
              (msmarco_qw06 and msmarco_e5m share a document set and order:
              exact top-1 agreement 0.610 between them, 0.000 against
              msmarco_bgem3 and msmarco_ada2, which are different subsamples)
  query       two Quora queries, same corpus and same quantizer, R* 10 vs 119
  corpus      the most typical query of Quora and of MS MARCO under one encoder
"""
import os, sys, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, WORK, BEIR, K
import quantizers as QZ
import instruction as INS

MS = "/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/"


def _quora_text():
    return json.load(open(f"{BEIR}/quora_corpus_text.json"))


def quantizer_example(qi=197, nq=2000, topn=14):
    """How a 1-bit code reorders a top-10 whose exact scores are close."""
    ctext = _quora_text()
    qtext = INS.quora_texts(nq)
    X, Q, _ = load("quora_qw06", nq)
    Qt, Xh, _, _ = QZ.build("adc_bq", X, Q)
    t = X @ Q[qi]; a = Xh @ Qt[qi]
    ex = np.argsort(-t, kind="stable")
    bq = np.argsort(-a, kind="stable")
    arank = {int(v): r for r, v in enumerate(bq[:5000])}
    print(f'query: "{qtext[qi]}"')
    print(f'{"exact":>6s}{"BQ":>5s}{"cos":>8s}  document')
    for r in range(topn):
        i = int(ex[r])
        print(f'{r+1:6d}{arank.get(i,-1)+1:5d}{t[i]:8.4f}  "{ctext[i][:66]}"')
    print("  documents the code promotes into the top 10 from outside it:")
    top10 = set(ex[:K].tolist())
    for r in range(K):
        i = int(bq[r])
        if i not in top10:
            er = int(np.where(ex == i)[0][0]) + 1
            print(f'{er:6d}{r+1:5d}{t[i]:8.4f}  "{ctext[i][:66]}"')


def embedding_example(nq=400, ranks=(1, 2, 3, 5, 10, 100), top=4):
    """The same query and the same documents under two encoders."""
    qtext = [json.loads(l)["text"] for l in open(MS + "msmarco-1m_queries.jsonl")]
    prof, rstar = {}, {}
    for key in ("msmarco_qw06", "msmarco_e5m"):
        X, Q, _ = load(key, nq)
        P = np.zeros((nq, len(ranks)))
        for qi in range(nq):
            t = X @ Q[qi]
            idx = np.argpartition(-t, 120)[:120]
            idx = idx[np.argsort(-t[idx], kind="stable")]
            P[qi] = [t[idx[r - 1]] for r in ranks]
        prof[key] = P
        rstar[key] = np.load(f"{WORK}/{key}_perq.npz")["Rstar_adc_bq"]
        del X, Q
    d = np.abs(rstar["msmarco_qw06"][:nq].astype(int) - rstar["msmarco_e5m"][:nq].astype(int))
    for qi in np.argsort(-d)[:top]:
        print(f'\nq#{qi}  "{qtext[qi][:66]}"')
        for key, nm in (("msmarco_qw06", "Qwen3-0.6B 1024d"),
                        ("msmarco_e5m", "E5-Mistral 4096d")):
            print(f'   {nm:18s}' + " ".join(f"{v:7.3f}" for v in prof[key][qi])
                  + f"   R*={int(rstar[key][qi]):5d}")


def query_example(nq=2000, n=3, ranks=(1, 2, 3, 5, 10, 100)):
    """Same corpus, same quantizer: which queries are expensive and why."""
    ctext = _quora_text()
    qtext = INS.quora_texts(nq)
    X, Q, _ = load("quora_qw06", nq)
    z = np.load(f"{WORK}/quora_qw06_perq.npz")
    rstar, beta = z["Rstar_adc_bq"], z["beta"]
    pool = list(range(300))
    for tag, sel in (("EASIEST", sorted(pool, key=lambda i: (rstar[i], -beta[i]))[:n]),
                     ("HARDEST", sorted(pool, key=lambda i: -rstar[i])[:n])):
        for qi in sel:
            t = X @ Q[qi]
            ex = np.argsort(-t, kind="stable")
            c = [float(t[int(ex[r - 1])]) for r in ranks]
            print(f'\n[{tag} q#{qi}] R*={int(rstar[qi]):4d} beta={beta[qi]:.3f}  '
                  f'"{qtext[qi][:60]}"')
            print("   cos at ranks " + "/".join(map(str, ranks)) + ": "
                  + " ".join(f"{v:.3f}" for v in c))
            for r in range(2):
                print(f'      #{r+1} "{ctext[int(ex[r])][:64]}"')


def corpus_example(nq=2000, ranks=(1, 2, 3, 5, 10, 100), topn=10, show=2):
    """What beta looks like in documents: the most typical query of each corpus.

    The two queries that open Section 3.2 are chosen by a rule, not by hand --
    each is the query whose standardized profile z(1..100) is closest to its own
    corpus's median -- so the example is representative of the corpus rather
    than of a search for a good anecdote.  Both corpora are read under one
    encoder (Qwen3-0.6B, 1024d) so that only the corpus differs.  Quora prints
    its documents; MS MARCO prints scores only, for the reason in the module
    docstring.
    """
    ctext = _quora_text()
    qt = {"quora_qw06": INS.quora_texts(nq),
          "msmarco_qw06": [json.loads(l)["text"]
                           for l in open(MS + "msmarco-1m_queries.jsonl")]}
    for key in ("quora_qw06", "msmarco_qw06"):
        X, Q, _ = load(key, nq)
        z = np.load(f"{WORK}/{key}_perq.npz")
        P = np.zeros((len(Q), len(ranks)))
        for qi in range(len(Q)):
            t = X @ Q[qi]
            idx = np.argpartition(-t, 120)[:120]
            idx = idx[np.argsort(-t[idx], kind="stable")]
            P[qi] = [t[idx[r - 1]] for r in ranks]
        # standardize by each query's own corpus scores, on a fixed sample
        rs = np.random.default_rng(0).choice(len(X), 200000, replace=False)
        Xs = X[rs]
        mu = np.array([float((Xs @ q).mean()) for q in Q])
        sd = np.array([float((Xs @ q).std()) for q in Q])
        Z = (P - mu[:, None]) / sd[:, None]
        pick = np.argsort(np.abs(Z - np.median(Z, 0)).sum(1))[:show]
        print(f"\n=== {key}  N={len(X)}  n={len(Q)} ===")
        print("  median cos at ranks " + "/".join(map(str, ranks)) + ": "
              + " ".join(f"{v:.3f}" for v in np.median(P, 0)))
        print(f"  p95 R* (ADC-BQ) {np.percentile(z['Rstar_adc_bq'], 95):.0f}"
              f"   median beta {np.median(z['beta']):.3f}")
        for qi in pick:
            print(f'\n  q#{qi} R*={int(z["Rstar_adc_bq"][qi])} '
                  f'beta={z["beta"][qi]:.3f}  "{qt[key][qi]}"')
            print("    cos " + " ".join(f"{v:.3f}" for v in P[qi]))
            if key.startswith("quora"):
                t = X @ Q[qi]
                ex = np.argsort(-t, kind="stable")
                for r in range(topn):
                    i = int(ex[r])
                    print(f'    {r+1:2d} {t[i]:.3f}  "{ctext[i][:70]}"')
        del X, Q


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "quantizer"):
        print("=" * 72 + "\nQUANTIZER LEVEL\n" + "=" * 72); quantizer_example()
    if which in ("all", "embedding"):
        print("\n" + "=" * 72 + "\nEMBEDDING LEVEL\n" + "=" * 72); embedding_example()
    if which in ("all", "corpus"):
        print("\n" + "=" * 72 + "\nCORPUS LEVEL\n" + "=" * 72); corpus_example()
    if which in ("all", "query"):
        print("\n" + "=" * 72 + "\nQUERY LEVEL\n" + "=" * 72); query_example()
