"""MRL dimension sweep on one corpus and one encoder.

Qwen3-Embedding-8B is trained with Matryoshka loss and its documented way to
produce a d-dimensional vector is to truncate the 4096-d output to the first d
coordinates and renormalise, so the whole ladder comes from vectors we already
have -- no re-encoding.  This replaces the 1024/2560/4096 block of the old
Table 3 with a power-of-two ladder measured against per-d ground truth.

At every d the ground truth is that d's OWN full-precision top-10, so the
quantization loss is not mixed with the truncation loss.  The truncation loss
is reported separately as the overlap with the 1024-d top-10.

Also reports  f = ||x||_1 / (sqrt(d) ||x||_2),  the cosine between a vector and
its own sign code.  For isotropic x this tends to sqrt(2/pi) = 0.798 and does
NOT vanish with d: taking signs throws away a constant fraction of the
direction however large d gets.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import CONFIGS, OUT, NQ_MAX, K
import quantizers as QZ
import measure as MS

DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]   # powers of two (MRL)
ARMS = ["sym_bq", "adc_bq", "pq"]   # OPQ omitted: ~30 min of the ladder, and PQ
                                   # already gives a same-byte learned reference


def prep(X0, Q0, d):
    X = np.ascontiguousarray(X0[:, :d])
    Q = np.ascontiguousarray(Q0[:, :d])
    X /= np.linalg.norm(X, axis=1, keepdims=True).clip(1e-30)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True).clip(1e-30)
    return X, Q


def run(key="msmarco_qw8", nq=1000):
    c = CONFIGS[key]
    X0 = c["base"]().astype(np.float32)
    Q0 = c["query"]().astype(np.float32)[:nq]
    N = X0.shape[0]
    nq = Q0.shape[0]
    res = {"key": key, "N": int(N), "nq": int(nq), "dims": DIMS, "rows": []}
    gt_full = None
    for d in DIMS:
        t0 = time.time()
        X, Q = prep(X0, Q0, d)
        ex = MS.exact_side(X, Q, nq, N, d)
        gt = ex["gt_id"][:, :K]
        if d == DIMS[-1]:
            gt_full = gt.copy()
        # f = <x, sign(x)>/(sqrt(d)||x||): the sign code's own cosine
        sub = X[:200_000]
        f = float(np.mean(np.abs(sub).sum(1) / (np.sqrt(d) * np.linalg.norm(sub, axis=1))))
        row = dict(d=d, beta=float(ex["beta"].mean()), z10=float(ex["z10"].mean()),
                   f_sign_cos=f, arms={})
        for arm in ARMS:
            Qt, Xh, nb, _ = QZ.build(arm, X, Q)
            mx32 = ex["mx"].astype(np.float32)
            mh = MS.colmean(Xh); mh32 = mh.astype(np.float32)
            qd, qtd = Q.astype(np.float64), Qt.astype(np.float64)
            var_a = MS.qform(qtd, MS.grams_c(Xh, mh32, Xh, mh32), qtd) / N
            cov = MS.qform(qd, MS.grams_c(X, mx32, Xh, mh32), qtd) / N
            kap = np.clip(cov / (ex["sd_t"] * np.sqrt(var_a)), -1, 1)
            Rk = MS.arm_ranks(Qt, Xh, ex["gt_id"], nq, N)
            del Qt, Xh
            Rs = Rk.max(1)
            row["arms"][arm] = dict(
                bytes=float(nb), kappa=float(kap.mean()),
                c_Q=float((np.sqrt(np.maximum(1 - kap ** 2, 0)) / kap).mean()),
                Rstar_p95=float(np.quantile(Rs, .95)), Rstar_med=float(np.median(Rs)),
                candrec30=float((Rk <= 30).mean()), candrec80=float((Rk <= 80).mean()))
        row["gt"] = gt
        res["rows"].append(row)
        a = row["arms"]["adc_bq"]
        print(f"d={d:5d} beta={row['beta']:.3f} z10={row['z10']:.2f} f={f:.4f} | "
              f"ADC-BQ kappa={a['kappa']:.4f} c_Q={a['c_Q']:.3f} p95R*={a['Rstar_p95']:.0f} "
              f"CR@30={a['candrec30']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        del X, Q
    # truncation loss, measured against the 1024-d neighbours
    for row in res["rows"]:
        ov = np.mean([len(set(a.tolist()) & set(b.tolist())) / K
                      for a, b in zip(row.pop("gt"), gt_full)])
        row["overlap_with_1024"] = float(ov)
    json.dump(res, open(f"{OUT}/dimsweep_{key}.json", "w"), indent=1)
    print("\nd      beta   f      c_Q(ADC)  p95R*  CR@30   overlap with 1024-d top-10")
    for r in res["rows"]:
        a = r["arms"]["adc_bq"]
        print(f"{r['d']:5d} {r['beta']:6.3f} {r['f_sign_cos']:.4f} {a['c_Q']:9.3f} "
              f"{a['Rstar_p95']:6.0f} {a['candrec30']:7.4f} {r['overlap_with_1024']:10.3f}")


if __name__ == "__main__":
    run(*(sys.argv[1:] or []))
