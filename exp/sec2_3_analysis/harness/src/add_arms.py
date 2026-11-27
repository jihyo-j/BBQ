"""Measure extra arms and merge them into the existing out/<key>.json."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import ORDER, OUT, WORK, NQ_MAX, load, label
import quantizers as QZ
import measure as MS

NEW = sys.argv[1].split(",") if len(sys.argv) > 1 else ["adc_bq_c", "adc_bq_cb"]

for key in ORDER:
    p = f"{OUT}/{key}.json"
    if not os.path.exists(p):
        continue
    res = json.load(open(p))
    todo = [a for a in NEW if a not in res["arms"]]
    if not todo:
        continue
    X, Q, cfg = load(key, NQ_MAX)
    N, d = X.shape
    nq = Q.shape[0]
    ex = MS.exact_side(X, Q, nq, N, d)
    beta = ex["beta"]
    npz = dict(np.load(f"{WORK}/{key}_perq.npz"))
    MS.log(f"{key}: N={N} d={d} nq={nq} -> {todo}")
    for arm in todo:
        t0 = time.time()
        Qt, Xh, nbytes, note = QZ.build(arm, X, Q)
        mx32 = ex["mx"].astype(np.float32)
        mh = MS.colmean(Xh); mh32 = mh.astype(np.float32)
        qd, qtd = Q.astype(np.float64), Qt.astype(np.float64)
        var_a = MS.qform(qtd, MS.grams_c(Xh, mh32, Xh, mh32), qtd) / N
        cov = MS.qform(qd, MS.grams_c(X, mx32, Xh, mh32), qtd) / N
        ms_a = (qtd @ mh) ** 2 + var_a
        ok = var_a > 1e-12 * np.maximum(ms_a, 1e-30)
        kap = np.full(nq, np.nan)
        kap[ok] = cov[ok] / (ex["sd_t"][ok] * np.sqrt(var_a[ok]))
        kap = np.clip(kap, -1.0, 1.0)
        nu = np.sqrt(np.maximum(1 - kap ** 2, 0.0))
        eta = nu / np.maximum(kap, 1e-9) / beta
        Rk = MS.arm_ranks(Qt, Xh, ex["gt_id"], nq, N)
        del Qt, Xh
        Rs = Rk.max(1)
        res["arms"][arm] = dict(
            name=QZ.NICE[arm], bytes_per_vec=float(nbytes), note=note,
            kappa=float(np.nanmean(kap)), nu=float(np.nanmean(nu)),
            eta=float(np.nanmean(eta)), eta_med=float(np.nanmedian(eta)),
            Rstar_p95=float(np.quantile(Rs, .95)), Rstar_med=float(np.median(Rs)),
            Rstar_mean=float(Rs.mean()),
            candrec={int(g): float((Rk <= g).mean()) for g in MS.GRID},
            p95_Rstar_by_rank=[float(np.quantile(Rk[:, j], .95)) for j in range(MS.K)],
            spearman_z10_Rstar=float(MS._spearman(ex["z10"], Rs)),
            secs=round(time.time() - t0, 1))
        npz[f"kappa_{arm}"] = kap; npz[f"eta_{arm}"] = eta
        npz[f"Rstar_{arm}"] = Rs; npz[f"ranks_{arm}"] = Rk
        a = res["arms"][arm]
        MS.log(f"  {QZ.NICE[arm]:22s} {nbytes:6.1f}B kappa={a['kappa']:.4f} "
               f"p95R*={a['Rstar_p95']:.0f} CR@30={a['candrec'][30]:.4f} [{a['secs']}s]")
    json.dump(res, open(p, "w"), indent=1)
    np.savez_compressed(f"{WORK}/{key}_perq.npz", **npz)
    del X, Q
print("done")
