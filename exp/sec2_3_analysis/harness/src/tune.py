"""Section 3's diagnostic turned into three tuning decisions, tested out of sample.

The decisions, fixed before the held-out configurations were measured:

  D1  Is the target attainable at this width?  Read the pool ceiling
      A(ef) = 1 - L_visit from a 1-D scan of the ef ladder with ADC traversal
      only -- no FP32 reads, no refinement.  The target 1-rho is attainable at
      all only where A(ef) > 1-rho, and the calibration certificate of
      Theorem 1 returns a finite threshold at m calibration queries only where
      L_visit is comfortably below rho (Proposition 2).  We take
      ef_cert = min{ef : L_visit(ef) <= rho/2}, the margin m=1000 needs.

  D2  What will the refinement traffic be?  Predicted index-free from
      eta = c_Q/beta as log Rbar = a + b log eta, the two coefficients fitted
      once on the development grid of Section 3.  Rbar is the mean number of
      candidates the calibrated rule selects at ef_cert.

  D3  Is a more accurate code the lever?  Only when D1 is satisfied and D2's
      traffic is unaffordable; the reduction then available is c_Q's, which
      Section 4 prices.

Development grid: the 1024/1536-dimensional configurations Section 3 analyses.
Held-out: the 4096-dimensional encoders of Section 6 that Section 3 never
touches.  Cost model: ndis + lambda*R with lambda = 10 ADC evaluations per FP32
page read, from the paper's own 517.6 us / R=120 read wait against 0.45 us an
ADC evaluation; conclusions are reported for lambda in {3,10,30} too.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import OUT, K
import calib

DEV = ["msmarco_qw06", "nq_qw06", "quora_qw06", "dbpedia_qw06",
       "msmarco_bgem3", "nq_bgem3", "quora_bgem3", "fever_bgem3",
       "msmarco_ada2", "dbpedia_ada2", "dbpedia_bgel", "dbpedia_bgeb"]
HELD = ["nq_nem8", "quora_nem8", "nq_e5m", "quora_e5m", "dbpedia_e5m"]
EFS = (64, 128, 256, 512, 1024)
RGRID = tuple(list(range(10, 201, 5)) + [220, 250, 300, 400, 500, 700, 1024])
RHO, LAM, MCAL = 0.01, 10.0, 1000


def diag(key):
    """eta, beta, c_Q, p95 R* -- all from exhaustive ranking, no index."""
    d = json.load(open(f"{OUT}/{key}.json"))
    a = d["arms"]["adc_bq"]
    return dict(key=key, corpus=d["corpus"], encoder=d["encoder"], d=d["d"],
                beta=d["beta"], cQ=a["nu"] / a["kappa"], eta=a["eta"],
                Rstar_p95=a["Rstar_p95"])


def ladder(key, efs=EFS):
    out = {}
    for ef in efs:
        try:
            c = calib.fixed_curve(key, ef, Rs=RGRID)
        except FileNotFoundError:
            continue
        out[ef] = c
    return out


def cheapest_R(c, target):
    for R, v in sorted(c["curve"].items()):
        if v["recall"] >= target:
            return R
    return None


def profile(key, rho=RHO, lam=LAM, m=MCAL):
    """Everything the three decisions read, for one configuration."""
    g = ladder(key)
    if not g:
        return None
    efs = sorted(g)
    Lv = {ef: 1 - g[ef]["ceiling"] for ef in efs}
    ef_min = next((ef for ef in efs if Lv[ef] < rho), None)      # attainable
    ef_cert = next((ef for ef in efs if Lv[ef] <= rho / 2), None)  # certifiable
    o = dict(key=key, efs=efs, L_visit={int(k): v for k, v in Lv.items()},
             ndis={int(ef): g[ef]["ndis"] for ef in efs},
             ef_min=ef_min, ef_cert=ef_cert)
    try:
        o.update(diag(key))
    except (FileNotFoundError, KeyError):
        pass
    # tuned fixed budget at the cheapest attainable width
    if ef_min is not None:
        R = cheapest_R(g[ef_min], 1 - rho)
        o.update(R_fixed=R, cost_fixed=g[ef_min]["ndis"] + lam * (R or 0))
    # certified calibration at ef_cert
    if ef_cert is not None:
        c = calib.campaign(key, ef_cert, rho=rho, m=m, B=400)
        o.update(cal_recall=c["recall_mean"], cal_escal=c["fallback"],
                 cal_R=c["R_fin"], cal_Rp99=c["Rp99_fin"], cal_m=c["m"],
                 cost_cert=g[ef_cert]["ndis"] + lam * (c["R_fin"] or 0))
    # oracle over the whole (ef, R) grid
    best = None
    for ef in efs:
        R = cheapest_R(g[ef], 1 - rho)
        if R is None:
            continue
        cst = g[ef]["ndis"] + lam * R
        if best is None or cst < best[0]:
            best = (cst, ef, R)
    if best:
        o.update(cost_oracle=best[0], ef_oracle=best[1], R_oracle=best[2])
    return o


def fit_traffic(rows, k=K):
    """Two specifications of the traffic model, both fitted on the development
    grid before any held-out configuration was measured.

      M1  log Rbar      = a + b log eta      (the simpler form)
      M2  log (Rbar-k)  = a + b log eta      (R >= k is a hard floor, since
                                              C_tau always contains the top-k)

    They fit the development grid about equally well and extrapolate
    differently below its eta range, so the held-out set adjudicates.
    """
    ok = [r for r in rows if r.get("cal_R") and np.isfinite(r["cal_R"])
          and r.get("eta")]
    x = np.log([r["eta"] for r in ok])
    out = dict(n=len(ok), keys=[r["key"] for r in ok],
               eta_lo=float(np.exp(x.min())), eta_hi=float(np.exp(x.max())))
    for nm, y, off in (("M1", np.log([r["cal_R"] for r in ok]), 0.0),
                       ("M2", np.log([r["cal_R"] - k for r in ok]), float(k))):
        b, a = np.polyfit(x, y, 1)
        res = y - (a + b * x)
        pred = np.exp(a + b * x) + off
        act = np.array([r["cal_R"] for r in ok])
        out[nm] = dict(a=float(a), b=float(b), off=off,
                       r2=float(1 - (res ** 2).sum() /
                                ((y - y.mean()) ** 2).sum()),
                       mape=float(np.abs(pred / act - 1).mean()))
    return out


def predict_traffic(fit, eta, which="M1"):
    c = fit[which]
    return float(np.exp(c["a"] + c["b"] * np.log(eta)) + c["off"])


def run(rho=RHO, lam=LAM, m=MCAL):
    dev = [p for p in (profile(k, rho, lam, m) for k in DEV) if p]
    coef = fit_traffic(dev)
    held = [p for p in (profile(k, rho, lam, m) for k in HELD) if p]
    for h in held:
        if h.get("eta"):
            h["R_pred"] = float(np.exp(coef["a"] + coef["b"] * np.log(h["eta"])))
            if h.get("cal_R") and np.isfinite(h["cal_R"]):
                h["R_err"] = h["R_pred"] / h["cal_R"] - 1
    return dict(rho=rho, lam=lam, m=m, coef=coef, dev=dev, held=held)


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["coef"], indent=1))
    hdr = (f"{'key':16s} {'eta':>7s} {'ef_min':>7s} {'ef_cert':>8s} "
           f"{'Rfix':>5s} {'calR':>6s} {'esc':>6s} {'rec':>7s} "
           f"{'c_fix':>7s} {'c_cert':>7s} {'c_ora':>7s}")
    for tag, rows in (("DEV", r["dev"]), ("HELD", r["held"])):
        print(tag); print(hdr)
        for x in rows:
            f = lambda k, d="-": x.get(k, d)
            print(f"{x['key']:16s} {f('eta',0):7.3f} {str(f('ef_min')):>7s} "
                  f"{str(f('ef_cert')):>8s} {str(f('R_fixed')):>5s} "
                  f"{f('cal_R',float('nan')):6.1f} {f('cal_escal',float('nan')):6.3f} "
                  f"{f('cal_recall',float('nan')):7.5f} "
                  f"{f('cost_fixed',float('nan'))/1000:7.1f} "
                  f"{f('cost_cert',float('nan'))/1000:7.1f} "
                  f"{f('cost_oracle',float('nan'))/1000:7.1f}"
                  + (f"  pred={x['R_pred']:.1f} err={x['R_err']:+.2%}"
                     if 'R_err' in x else ""))


# ---------------------------------------------- G1: is the narrowest width the cheapest?
def width_check(keys, rho=RHO, lams=(3.0, 10.0, 30.0, 100.0)):
    """For each lambda, does argmin_{ef,R} cost coincide with the narrowest
    attainable width?  If it does, the 1-D ceiling scan of D1 already returns
    the cost-optimal point and the 2-D (ef, R) sweep buys nothing."""
    rows = []
    for k in keys:
        g = ladder(k)
        if not g:
            continue
        efs = sorted(g)
        Lv = {ef: 1 - g[ef]["ceiling"] for ef in efs}
        ef_min = next((ef for ef in efs if Lv[ef] < rho), None)
        if ef_min is None:
            rows.append(dict(key=k, ef_min=None))
            continue
        r = dict(key=k, ef_min=ef_min)
        for lam in lams:
            best, bef, bR = None, None, None
            for ef in efs:
                R = cheapest_R(g[ef], 1 - rho)
                if R is None:
                    continue
                c = g[ef]["ndis"] + lam * R
                if best is None or c < best:
                    best, bef, bR = c, ef, R
            Rm = cheapest_R(g[ef_min], 1 - rho)
            cm = g[ef_min]["ndis"] + lam * Rm
            r[f"lam{lam:g}"] = dict(ef=bef, R=bR, cost=best,
                                    cost_at_ef_min=cm, ratio=cm / best,
                                    agree=(bef == ef_min))
        rows.append(r)
    return rows


# ---------------------------------- D1 made index-free: predicting the ceiling
def fit_visit(keys, efs=EFS):
    """log L_visit(ef) = a + b log(p95 R*) + c log(ef).

    p95 R* is the exhaustive-ranking budget of Section 3 -- pure score geometry,
    no graph.  That it predicts what an HNSW traversal fails to reach is an
    empirical coupling, not an identity, so it is fitted on the development grid
    and tested on configurations the analysis never saw.
    """
    X, y, tag = [], [], []
    for k in keys:
        g = ladder(k, efs)
        try:
            d = diag(k)
        except (FileNotFoundError, KeyError):
            continue
        for ef in sorted(g):
            Lv = 1 - g[ef]["ceiling"]
            if Lv <= 0:
                continue
            X.append([1.0, np.log(d["Rstar_p95"]), np.log(ef)])
            y.append(np.log(Lv))
            tag.append((k, ef))
    X, y = np.array(X), np.array(y)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ c
    return dict(coef=[float(v) for v in c], n=len(y),
                r2=float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                mape=float(np.abs(np.exp(X @ c) / np.exp(y) - 1).mean()),
                rmse_log=float(np.sqrt((res ** 2).mean())))


def predict_visit(coef, Rstar_p95, ef):
    a, b, c = coef
    return float(np.exp(a + b * np.log(Rstar_p95) + c * np.log(ef)))


def recipe(key, vcoef, tcoef, rho=RHO, lam=LAM, m=MCAL, efs=EFS):
    """Recipe T end to end on one configuration: the width and the traffic are
    both predicted index-free, then checked against what the index does."""
    d = diag(key)
    g = ladder(key, efs)
    pred = {ef: predict_visit(vcoef, d["Rstar_p95"], ef) for ef in efs}
    ef_hat = next((ef for ef in efs if pred[ef] <= rho / 2), None)
    R_hat = predict_traffic(tcoef, d["eta"], "M1")
    R_hat2 = predict_traffic(tcoef, d["eta"], "M2")
    o = dict(key=key, eta=d["eta"], Rstar_p95=d["Rstar_p95"],
             ef_hat=ef_hat, R_hat=R_hat, R_hat2=R_hat2,
             L_visit_pred={int(e): pred[e] for e in efs},
             L_visit_true={int(e): 1 - g[e]["ceiling"] for e in sorted(g)})
    efs_m = sorted(g)
    o["ef_cert_true"] = next((e for e in efs_m
                              if 1 - g[e]["ceiling"] <= rho / 2), None)
    o["ef_min_true"] = next((e for e in efs_m if 1 - g[e]["ceiling"] < rho),
                            None)
    if ef_hat in g:
        c = calib.campaign(key, ef_hat, rho=rho, m=m, B=400)
        o.update(cal_recall=c["recall_mean"], cal_escal=c["fallback"],
                 cal_R=c["R_fin"], cal_Rp99=c["Rp99_fin"],
                 hit=bool(c["recall_mean"] >= 1 - rho and c["fallback"] < .5),
                 cost=g[ef_hat]["ndis"] + lam * (c["R_fin"] or 0))
        if np.isfinite(c["R_fin"]):
            o["R_err"] = R_hat / c["R_fin"] - 1
            o["R_err2"] = R_hat2 / c["R_fin"] - 1
    # cost of the *measured* certifiable width, to isolate what predicting it
    # adds over scanning for it
    if o["ef_cert_true"] in g:
        e = o["ef_cert_true"]
        cc = calib.campaign(key, e, rho=rho, m=m, B=400)
        o["cost_cert_true"] = g[e]["ndis"] + lam * (cc["R_fin"] or 0)
        o["cal_R_cert_true"] = cc["R_fin"]
    # the point a full 2-D sweep would pick, for the cost comparison
    best = None
    for ef in efs_m:
        R = cheapest_R(g[ef], 1 - rho)
        if R is None:
            continue
        cst = g[ef]["ndis"] + lam * R
        if best is None or cst < best[0]:
            best = (cst, ef, R)
    if best:
        o.update(cost_sweep=best[0], ef_sweep=best[1], R_sweep=best[2])
    return o


def run_recipe(rho=RHO, lam=LAM, m=MCAL):
    v = fit_visit(DEV)
    dev = [p for p in (profile(k, rho, lam, m) for k in DEV) if p]
    t = fit_traffic(dev)
    held = [recipe(k, v["coef"], t, rho, lam, m) for k in HELD]
    return dict(rho=rho, lam=lam, m=m, visit=v, traffic=t, dev=dev, held=held)
