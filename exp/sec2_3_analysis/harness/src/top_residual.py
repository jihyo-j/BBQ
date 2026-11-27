"""Why centring lowers c_Q yet costs candidates.

Residual on the scale of the standardized exact score, e = a/kappa - t.
Measure its spread at the exact top-100 versus at random corpus documents.
If plain ADC-BQ scores the top ranks more precisely "for free", sigma_top /
sigma_rand < 1, and centring should push that ratio up even while lowering the
corpus-wide sigma that kappa and c_Q see.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, ORDER, CONFIGS, OUT, NQ_MAX
import quantizers as QZ
import measure as MS

NSAMP = 20_000
RTOP = 100


def run(key, nq=1000, arms=("adc_bq", "adc_bq_c")):
    X, Q, cfg = load(key, nq)
    N, d = X.shape
    nq = Q.shape[0]
    ex = MS.exact_side(X, Q, nq, N, d)
    rng = np.random.default_rng(7)
    samp = np.sort(rng.choice(N, min(NSAMP, N), replace=False))
    Xs = np.ascontiguousarray(X[samp])
    out = {}
    for arm in arms:
        Qt, Xh, nb, _ = QZ.build(arm, X, Q)
        mx32 = ex["mx"].astype(np.float32)
        mh = MS.colmean(Xh); mh32 = mh.astype(np.float32)
        qd, qtd = Q.astype(np.float64), Qt.astype(np.float64)
        var_a = MS.qform(qtd, MS.grams_c(Xh, mh32, Xh, mh32), qtd) / N
        cov = MS.qform(qd, MS.grams_c(X, mx32, Xh, mh32), qtd) / N
        kap = np.clip(cov / (ex["sd_t"] * np.sqrt(var_a)), -1, 1)
        mean_a = qtd @ mh
        sd_a = np.sqrt(var_a)
        Xhs = np.ascontiguousarray(Xh[samp])
        st, sr, bt, br = [], [], [], []
        for qi in range(nq):
            gt = ex["gt_id"][qi]
            tz = (ex["gt_sc"][qi] - ex["mean_t"][qi]) / ex["sd_t"][qi]
            az = (Xh[gt] @ Qt[qi] - mean_a[qi]) / sd_a[qi]
            e = az / kap[qi] - tz
            st.append(e.std()); bt.append(e.mean())
            tzr = (Xs @ Q[qi] - ex["mean_t"][qi]) / ex["sd_t"][qi]
            azr = (Xhs @ Qt[qi] - mean_a[qi]) / sd_a[qi]
            er = azr / kap[qi] - tzr
            sr.append(er.std()); br.append(er.mean())
        del Qt, Xh, Xhs
        out[arm] = dict(bytes=float(nb), kappa=float(kap.mean()),
                        c_Q=float((np.sqrt(np.maximum(1 - kap ** 2, 0)) / kap).mean()),
                        sigma_top=float(np.mean(st)), sigma_rand=float(np.mean(sr)),
                        ratio=float(np.mean(st) / np.mean(sr)),
                        bias_top=float(np.mean(bt)), bias_rand=float(np.mean(br)))
    return out


if __name__ == "__main__":
    keys = sys.argv[1:] or [k for k in ORDER if CONFIGS[k]["group"] != "classic"]
    res = {}
    print(f"{'config':22s} {'arm':12s} {'c_Q':>6s} {'sig_top':>8s} {'sig_rand':>9s} "
          f"{'top/rand':>9s} {'bias_top':>9s}")
    for k in keys:
        res[k] = run(k)
        for a, v in res[k].items():
            print(f"{k:22s} {a:12s} {v['c_Q']:6.3f} {v['sigma_top']:8.4f} "
                  f"{v['sigma_rand']:9.4f} {v['ratio']:9.3f} {v['bias_top']:+9.4f}", flush=True)
    json.dump(res, open(f"{OUT}/top_residual.json", "w"), indent=1)
