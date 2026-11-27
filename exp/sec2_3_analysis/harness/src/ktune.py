"""Is top-10 the right target?  Value (qrels recall) against cost (BQ candidates).

For each corpus we have relevance judgments for, measure
  value(k)  = fraction of the query's judged-relevant documents that the EXACT
              top-k contains          (the ceiling any ANN system can reach at k)
  cost(k)   = p95 R*(k), the ADC-BQ candidate budget that holds the exact top-k
and report the marginal candidates bought per point of value.
"""
import os, sys, json, csv, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import WORK, OUT, load, K

BD = "/tmp/claude-1007/-home-sieun/f2a0a973-d46a-4f9e-8675-19793fb4c84e/scratchpad/beir/"
MS = "/home/smrc/samsung-nvme/sieun/msmarco-qwen1024/"


def msmarco_qrels():
    rows = [json.loads(l) for l in open(MS + "msmarco-1m_queries.jsonl")]
    return [{int(x) for x in r["rel"]} for r in rows]


def quora_qrels():
    import pyarrow.parquet as pq
    cid = pq.read_table(BD + "quora_corpus.parquet")["_id"].to_pylist()
    row = {c: i for i, c in enumerate(cid)}
    q = {}
    with open(BD + "quora_qrels_test.tsv") as f:
        r = csv.reader(f, delimiter="\t"); next(r)
        for a, c, s in r:
            if float(s) > 0:
                q.setdefault(a, set()).add(row[c])
    return [q[i] for i in sorted(q)]


SETS = {"msmarco_qw06": msmarco_qrels, "msmarco_bgem3": msmarco_qrels,
        "msmarco_ada2": msmarco_qrels, "quora_qw06": quora_qrels,
        "quora_bgem3": quora_qrels}


def run(keys=None):
    out = {}
    for key in (keys or SETS):
        rel = SETS[key]()
        X, Q, c = load(key, 1000)
        N, d = X.shape; nq = Q.shape[0]
        # exact top-10 ids, same total order as the rest of the paper
        import measure as MS_
        ex = MS_.exact_side(X, Q, nq, N, d)
        gt = ex["gt_id"][:, :K]
        rk = np.load(f"{WORK}/{key}_perq.npz")["ranks_adc_bq"][:nq]
        val, cost = [], []
        for k in range(1, K + 1):
            v = [len(set(gt[i, :k].tolist()) & rel[i]) / max(len(rel[i]), 1)
                 for i in range(nq) if rel[i]]
            val.append(float(np.mean(v)))
            cost.append(float(np.quantile(rk[:, :k].max(1), .95)))
        out[key] = dict(corpus=c["corpus"], encoder=c["encoder"],
                        nrel=float(np.mean([len(r) for r in rel[:nq] if r])),
                        value=val, cost=cost)
        print(f"\n{c['corpus']}/{c['encoder']}   {out[key]['nrel']:.2f} judged-relevant per query")
        print(f"   {'k':>3s} {'value':>7s} {'cost':>6s} {'d value':>8s} {'d cost':>7s} {'cost per +0.01':>15s}")
        for k in range(K):
            dv = val[k] - (val[k-1] if k else 0.0)
            dc = cost[k] - (cost[k-1] if k else 0.0)
            pc = f"{dc/(dv*100):.1f}" if dv > 1e-6 else "--"
            print(f"   {k+1:3d} {val[k]:7.3f} {cost[k]:6.0f} {dv:+8.3f} {dc:+7.0f} {pc:>15s}")
        del X, Q
    json.dump(out, open(f"{OUT}/ktune.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1:] or None)
