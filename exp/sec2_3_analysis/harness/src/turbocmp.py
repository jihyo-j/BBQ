"""BBQ against TurboQuant-1bit under the *same* calibration policy.

Same FP32 graph, same search width, same threshold rule, same global ground
truth; only the scorer differs.  Sweeping the calibration target traces each
arm's achieved global Recall@10 against the full-precision reads it spends, so
the two can be read at matched recall rather than at matched budget.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import WORK, K
import calib

TARGETS = [0.90, 0.95, 0.975, 0.99, 0.995]
CELLS = [("msmarco_qw06", 256, 1_000_000, "MS MARCO / Qwen-1024"),
         ("dbpedia_bgel", 256, 1_000_000, "DBpedia / BGE-large")]


def load(key, ef, arm):
    f = f"{WORK}/{'pool' if arm=='bbq' else 'pooltq'}_{key}_M24_ef{ef}.npz"
    z = np.load(f)
    return z["pool_s"], z["gold_lv"], z["ndis"]


def sweep(key, ef, N, arm, m=1000, B=400, seed=20261119):
    ps, gl, nd = load(key, ef, arm)
    m = min(m, len(gl) // 2)
    out = []
    for t in TARGETS:
        rng = np.random.default_rng(seed)
        rep = [calib.replicate(ps, gl, 1 - t, m, rng, N) for _ in range(B)]
        rec = np.array([r["recall"] for r in rep])
        fin = np.array([np.isfinite(r["tau"]) for r in rep])
        R = np.array([r["R_mean"] for r in rep])
        out.append(dict(target=t, recall=float(rec.mean()),
                        reads=float(R[fin].mean()) if fin.any() else float("nan"),
                        recall_fin=float(rec[fin].mean()) if fin.any() else float("nan"),
                        escal=float(1 - fin.mean()), m=int(m)))
    return dict(key=key, ef=ef, arm=arm, pi=float(1 - np.isfinite(gl).mean()),
                ndis=float(nd.mean()), rows=out)


def main():
    res = []
    for key, ef, N, nice in CELLS:
        for arm in ("bbq", "tq"):
            r = sweep(key, ef, N, arm)
            r["nice"] = nice
            res.append(r)
            print(f"{nice:22s} {arm:4s} pi={r['pi']:.4f} ndis={r['ndis']:.0f}  " +
                  "  ".join(f"{x['target']:.3f}:{x['recall_fin']:.4f}/{x['reads']:.0f}"
                            f"({x['escal']:.2f})" for x in r["rows"]), flush=True)
    json.dump(res, open("out/turbocmp.json", "w"), indent=2)

    with open("gp/motiv/turbo.dat", "w") as fh:
        fh.write("# BBQ vs TurboQuant-1bit under the same calibration rule.\n")
        fh.write("# Blocks: msmarco bbq, msmarco tq, dbpedia bbq, dbpedia tq.\n")
        fh.write("# reads  recall_on_certified_runs  target  escalation\n")
        for r in res:
            fh.write(f"# {r['nice']} {r['arm']} pi={r['pi']:.4f}\n")
            for x in r["rows"]:
                fh.write(f"{x['reads']:.2f} {x['recall_fin']:.5f} "
                         f"{x['target']:.3f} {x['escal']:.3f}\n")
            fh.write("\n\n")
    print("wrote gp/motiv/turbo.dat")


if __name__ == "__main__":
    main()
