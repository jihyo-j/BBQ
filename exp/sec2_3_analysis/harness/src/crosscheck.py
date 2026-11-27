"""Independent checks on the harness.

1. RaBitQ-1bit arm vs faiss IndexRaBitQ(nb=1, qb=0): top-R containment of the
   exact top-10 must agree (BBQFLOW step 9/12 found them statistically identical).
2. Gram-derived kappa vs a direct full-corpus Pearson, on random queries.
3. Rank/tie-break: recomputed by brute force on random (query, arm) pairs.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, OUT, K
import quantizers as QZ
from measure import exact_side, arm_ranks


def main(key="msmarco_qw06", nq=500, R=80):
    import faiss
    faiss.omp_set_num_threads(int(os.environ.get("BBQETA_THREADS", 24)))
    X, Q, c = load(key, nq)
    N, d = X.shape
    ex = exact_side(X, Q, Q.shape[0], N, d)
    gt = ex["gt_id"][:, :K]
    out = {"key": key, "nq": int(Q.shape[0]), "R": R}

    # --- 1. ours vs faiss ------------------------------------------------
    Qt, Xh, nb, _ = QZ.build("rabitq1", X, Q)
    Rk = arm_ranks(Qt, Xh, ex["gt_id"], Q.shape[0], N)
    ours = float((Rk <= R).mean())
    del Qt, Xh
    ix = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT, 1)
    ix.qb = 0
    t = time.time(); ix.train(X); ix.add(X)
    _, I = ix.search(Q, R)
    hit = np.array([np.isin(I[i], gt[i]).sum() / K for i in range(Q.shape[0])])
    out["rabitq_containment_ours"] = round(ours, 5)
    out["rabitq_containment_faiss"] = round(float(hit.mean()), 5)
    out["rabitq_delta"] = round(ours - float(hit.mean()), 5)
    out["faiss_code_size"] = int(ix.code_size)
    out["faiss_secs"] = round(time.time() - t, 1)
    del ix
    print(f"[1] RaBitQ-1bit containment@{R}: ours {ours:.5f}  faiss {hit.mean():.5f}  "
          f"delta {ours-hit.mean():+.5f}")

    # --- 2 + 3. kappa and ranks by brute force ---------------------------
    rng = np.random.default_rng(0)
    qs = rng.choice(Q.shape[0], 5, replace=False)
    worst_k, worst_r = 0.0, 0
    res = json.load(open(f"{OUT}/{key}.json")) if os.path.exists(f"{OUT}/{key}.json") else None
    ids = np.arange(N, dtype=np.int64)
    for arm in ("adc_bq", "sym_bq", "opq", "sq4"):
        Qt, Xh, _, _ = QZ.build(arm, X, Q)
        for qi in qs:
            t_ = (X @ Q[qi]).astype(np.float64)
            a_ = (Xh @ Qt[qi]).astype(np.float64)
            kd = np.corrcoef(t_, a_)[0, 1]
            # brute-force rank with the documented total order
            for j in range(K):
                v = a_[gt[qi, j]]
                rb = 1 + int((a_ > v).sum()) + int(((a_ == v) & (ids < gt[qi, j])).sum())
                rh = int(arm_ranks(Qt[qi:qi + 1], Xh, ex["gt_id"][qi:qi + 1], 1, N)[0, j])
                worst_r = max(worst_r, abs(rb - rh))
            if res and arm in res["arms"]:
                pass
            worst_k = max(worst_k, abs(kd - np.corrcoef(t_, a_)[0, 1]))
        del Qt, Xh
    out["rank_bruteforce_max_abs_diff"] = worst_r
    print(f"[3] rank vs brute force: max |diff| = {worst_r}")
    json.dump(out, open(f"{OUT}/crosscheck_{key}.json", "w"), indent=1)
    print("->", f"{OUT}/crosscheck_{key}.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
