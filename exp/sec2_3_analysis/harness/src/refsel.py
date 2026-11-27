"""Refinement-selection rules on a FIXED RaBitQ code, score and candidate pool.

Every arm reads the same arrays dumped by rbq.py -- the same pool L(q) in the
same dhat order, the same per-candidate bound radius -- and differs ONLY in
which pool members it sends to FP32.  That isolates the cost of the selection
rule from the cost of the code.

Target of every arm: H(q), the FP32 top-k INSIDE L(q).  An arm "preserves" a
query when every member of H(q) is refined; since H(q) is the pool's exact
top-k, that is the same statement as "the returned top-k equals H(q)".

Arms
  bound(eps0)   RaBitQ's own rule: scan L(q) in dhat order, keep the k-th best
                exact distance seen so far as theta, refine x iff
                dhat(x) - eps0*rad(x) < theta.  eps0 = 1.9 is RaBitQ's default;
                sweeping eps0 is the "tuned bound" arm.
  fixedR(R)     refine the top-R dhat prefix, R tuned per dataset.
  calib(rho)    conformal margin threshold on the SAME dhat: tau = the
                ceil((m+1)(1-rho))-th order statistic of the margins
                max(0, dhat(x_j) - dhat_k(q)) of one uniformly drawn answer per
                calibration query; refine {x : dhat(x) <= dhat_k(q) + tau}.
  calib+bound   the calibrated prefix, with the bound rule pruning inside it.
  oracle        per-query minimum prefix that preserves H(q) (a floor, not a
                deployable rule).

Global Recall@k is exact here: a global top-k answer that is refined is always
in the final top-k (fewer than k pool members beat it), so
Recall@k = #{global answers refined}/k.
"""
import os, sys, json, heapq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BBQETA_THREADS", "24")
import numpy as np
from common import WORK, OUT, CONFIGS, K, label
import rbq

EPS_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
            1.3, 1.4, 1.5, 1.7, 1.9, 2.2, 2.5, 3.0, 3.5, 4.0)
RHO_GRID = (0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03,
            0.025, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.0025, 0.001)


# ------------------------------------------------------------------ loading
def load_pool(key, ef, nq, M=rbq.M_DEFAULT):
    z = np.load(rbq.path(key, ef, nq, M))
    ids, dhat, rad, dex = z["ids"], z["dhat"], z["rad"], z["dex"]
    gt, dgt = z["gt"], z["dgt"]
    nq, efc = ids.shape
    # H(q): pool-internal exact top-k (dex ascending, id ascending)
    ordr = np.lexsort((ids, dex), axis=1)[:, :K]
    Hmask = np.zeros(ids.shape, dtype=bool)
    np.put_along_axis(Hmask, ordr, True, axis=1)
    # positions of the GLOBAL answers inside the pool (-1 when absent)
    gtpos = np.full(gt.shape, -1, dtype=np.int32)
    for r in range(nq):
        o = np.argsort(ids[r], kind="stable")
        s = ids[r][o]
        j = np.searchsorted(s, gt[r])
        j = np.clip(j, 0, len(s) - 1)
        hit = s[j] == gt[r]
        gtpos[r][hit] = o[j[hit]]
    return dict(ids=ids, dhat=dhat, rad=rad, dex=dex, Hmask=Hmask,
                Hord=ordr, gtpos=gtpos, ndis=z["ndis"], D=int(z["D"][0]),
                N=int(z["N"][0]), key=key, ef=int(ef))


# ------------------------------------------------------------ the bound rule
def bound_pass(P, eps0, collect_slack=False):
    """RaBitQ's refinement rule at confidence constant eps0, per query.

    Returns reads, preserved, n_gt_refined, and (optionally) the number of
    refined candidates that are not in H(q) -- the reads the bound could not
    avoid -- plus the tightest ratio |dhat-d|/rad seen among REFINED candidates.
    """
    dhat, rad, dex, Hm, gtpos = P["dhat"], P["rad"], P["dex"], P["Hmask"], P["gtpos"]
    nq, ef = dhat.shape
    reads = np.zeros(nq, dtype=np.int32)
    pres = np.zeros(nq, dtype=bool)
    ngt = np.zeros(nq, dtype=np.int32)
    waste = np.zeros(nq, dtype=np.int32)
    for r in range(nq):
        lb = (dhat[r] - eps0 * rad[r]).tolist()
        dx = dex[r].tolist()
        hm = Hm[r]
        heap = []            # max-heap (negated) of the k best exact distances
        theta = float("inf")
        nref = 0
        refined = np.zeros(ef, dtype=bool)
        for j in range(ef):
            if lb[j] < theta:
                d = dx[j]
                if d == float("inf"):
                    continue          # pool shorter than ef (padding)
                refined[j] = True
                nref += 1
                if len(heap) < K:
                    heapq.heappush(heap, -d)
                    if len(heap) == K:
                        theta = -heap[0]
                elif d < -heap[0]:
                    heapq.heapreplace(heap, -d)
                    theta = -heap[0]
        reads[r] = nref
        pres[r] = bool(refined[hm].all())
        g = gtpos[r]
        ngt[r] = int(refined[g[g >= 0]].sum())
        if collect_slack:
            waste[r] = nref - int((refined & hm).sum())
    o = dict(reads=reads, pres=pres, ngt=ngt)
    if collect_slack:
        o["waste"] = waste
    return o


# ----------------------------- batchable variants of the same bound
def bound_margin(P, T, eps0):
    """The per-candidate bound used as a ONE-PASS margin rule, no FP32 feedback:
    refine x iff dhat(x) - dhat_k(q) < eps0*rad(x).

    Same radius, same score, same pool as bound_pass, but theta is the estimated
    k-th distance instead of an exact one, so no read depends on another read.
    This is the arm that asks whether the bound's per-candidate WIDTH helps, with
    its sequential threshold updating taken away."""
    dhat, rad, Hm, gtpos = P["dhat"], P["rad"], P["Hmask"], P["gtpos"]
    fin = np.isfinite(P["dex"])
    ref = ((dhat - T["dk"]) < eps0 * rad) & fin
    reads = ref.sum(1).astype(np.int32)
    pres = (ref | ~Hm).all(1)
    g = gtpos
    ngt = np.where(g >= 0, np.take_along_axis(ref, np.maximum(g, 0), 1), False).sum(1)
    return dict(reads=reads, pres=pres, ngt=ngt.astype(np.int32))


def bound_2phase(P, T, eps0):
    """Two dependent rounds: read the top-k dhat prefix, then one batch.

    theta is the exact k-th distance among the first k candidates, so the batch
    decision uses a real threshold but the whole batch is still issued at once."""
    dhat, rad, dex, Hm, gtpos = P["dhat"], P["rad"], P["dex"], P["Hmask"], P["gtpos"]
    nq, ef = dhat.shape
    fin = np.isfinite(dex)
    theta = np.max(np.where(fin[:, :K], dex[:, :K], -np.inf), axis=1,
                   keepdims=True)
    ref = ((dhat - eps0 * rad) < theta) & fin
    ref[:, :K] |= fin[:, :K]
    reads = ref.sum(1).astype(np.int32)
    pres = (ref | ~Hm).all(1)
    ngt = np.where(gtpos >= 0,
                   np.take_along_axis(ref, np.maximum(gtpos, 0), 1), False).sum(1)
    return dict(reads=reads, pres=pres, ngt=ngt.astype(np.int32))


# ------------------------------------------- prefix rules (closed form in R)
def prefix_tables(P):
    """Per-query quantities that make every prefix rule O(1) to evaluate.

    Rneed[q]   smallest R with H(q) inside the top-R dhat prefix
    marg[q,j]  margin max(0, dhat(x_j) - dhat_k(q)) of the j-th member of H(q)
    sm[q,:]    sorted margins of the whole pool, so |C_tau(q)| = searchsorted
    gtR[q,j]   prefix length needed to include the j-th GLOBAL answer (inf if
               it is not in the pool)
    """
    dhat, Hord, gtpos = P["dhat"], P["Hord"], P["gtpos"]
    nq, ef = dhat.shape
    dk = dhat[:, K - 1:K]                      # k-th smallest dhat in the pool
    marg = np.maximum(0.0, np.take_along_axis(dhat, Hord, 1) - dk)
    Rneed = (Hord.max(1) + 1).astype(np.int32)
    pm = np.maximum(0.0, dhat - dk)
    sm = np.sort(pm, axis=1)
    gtR = np.where(gtpos >= 0, gtpos.astype(np.float64) + 1.0, np.inf)
    gtm = np.where(gtpos >= 0,
                   np.maximum(0.0, np.take_along_axis(
                       dhat, np.maximum(gtpos, 0), 1) - dk), np.inf)
    return dict(dk=dk, marg=marg, Rneed=Rneed, sm=sm, gtR=gtR, gtm=gtm,
                Hmaxmarg=marg.max(1))


def tau_cost(T, tau, rows=None):
    """|C_tau(q)| for a scalar tau."""
    sm = T["sm"] if rows is None else T["sm"][rows]
    return np.array([np.searchsorted(sm[i], tau, side="right")
                     for i in range(len(sm))], dtype=np.int32)


# ------------------------------------------------- calibrated prefix + bound
def calib_bound_pass(P, T, tau, eps0):
    """The calibrated prefix, with RaBitQ's bound pruning applied inside it."""
    dhat, rad, dex, Hm, gtpos = P["dhat"], P["rad"], P["dex"], P["Hmask"], P["gtpos"]
    nq, ef = dhat.shape
    lim = tau_cost(T, tau)
    reads = np.zeros(nq, dtype=np.int32); pres = np.zeros(nq, dtype=bool)
    ngt = np.zeros(nq, dtype=np.int32)
    for r in range(nq):
        n = int(lim[r])
        lb = (dhat[r, :n] - eps0 * rad[r, :n]).tolist()
        dx = dex[r, :n].tolist()
        heap = []; theta = float("inf"); nref = 0
        refined = np.zeros(ef, dtype=bool)
        for j in range(n):
            if lb[j] < theta:
                d = dx[j]
                if d == float("inf"):
                    continue
                refined[j] = True; nref += 1
                if len(heap) < K:
                    heapq.heappush(heap, -d)
                    if len(heap) == K:
                        theta = -heap[0]
                elif d < -heap[0]:
                    heapq.heapreplace(heap, -d); theta = -heap[0]
        reads[r] = nref
        pres[r] = bool(refined[Hm[r]].all())
        g = gtpos[r]
        ngt[r] = int(refined[g[g >= 0]].sum())
    return dict(reads=reads, pres=pres, ngt=ngt)


# ------------------------------------------------------------------ campaign
def _stats(reads, pres, ngt, D):
    pg = int(np.ceil(4.0 * D / 4096.0))          # pages per vector, 4 KiB aligned
    return dict(reads=float(reads.mean()), reads_p95=float(np.quantile(reads, .95)),
                reads_p99=float(np.quantile(reads, .99)),
                pages=float(reads.mean() * pg),
                pres=float(pres.mean()), recall=float(ngt.mean() / K))


def campaign(key, ef, nq, target=0.99, B=20, seed=20261120, eps_default=1.9,
             rho_theorem=0.01):
    """Tune every arm's knob on a calibration split, score it on the held-out half."""
    P = load_pool(key, ef, nq)
    T = prefix_tables(P)
    nqq, efc = P["dhat"].shape
    D = P["D"]
    BP = {e: bound_pass(P, e, collect_slack=True) for e in EPS_GRID}
    # monotonicity of the bound rule in eps0 (not guaranteed a priori)
    mono = all(BP[a]["reads"].mean() <= BP[b]["reads"].mean() + 1e-9
               for a, b in zip(EPS_GRID, EPS_GRID[1:]))
    rng = np.random.default_rng(seed)
    m = nqq // 2
    BM = {e: bound_margin(P, T, e) for e in EPS_GRID}
    B2 = {e: bound_2phase(P, T, e) for e in EPS_GRID}
    rows = {a: [] for a in ("bound_default", "bound_tuned", "boundM_tuned",
                            "bound2p_tuned", "fixedR", "calibQ",
                            "calib_theorem", "calib_bound", "oracle")}
    chosen = {a: [] for a in ("bound_tuned", "boundM_tuned", "bound2p_tuned",
                              "fixedR", "calibQ", "calib_theorem")}
    for b in range(B):
        perm = rng.permutation(nqq)
        cal, tst = perm[:m], perm[m:]
        # --- arm 1: RaBitQ's shipped confidence constant
        r = BP[eps_default]
        rows["bound_default"].append(_stats(r["reads"][tst], r["pres"][tst],
                                            r["ngt"][tst], D))
        # --- arm 2: the same rule with eps0 tuned on the calibration half
        feas = [e for e in EPS_GRID if BP[e]["pres"][cal].mean() >= target]
        e_sel = min(feas, key=lambda e: BP[e]["reads"][cal].mean()) if feas \
            else EPS_GRID[-1]
        r = BP[e_sel]
        rows["bound_tuned"].append(_stats(r["reads"][tst], r["pres"][tst],
                                          r["ngt"][tst], D))
        chosen["bound_tuned"].append(e_sel)
        # --- arm 2b/2c: the same bound without the read-to-read dependency
        for nm, tab in (("boundM_tuned", BM), ("bound2p_tuned", B2)):
            fe = [e for e in EPS_GRID if tab[e]["pres"][cal].mean() >= target]
            es = min(fe, key=lambda e: tab[e]["reads"][cal].mean()) if fe \
                else EPS_GRID[-1]
            t = tab[es]
            rows[nm].append(_stats(t["reads"][tst], t["pres"][tst],
                                   t["ngt"][tst], D))
            chosen[nm].append(es)
        # --- arm 3: one tuned fixed prefix (what the paper's baselines use)
        Rn = T["Rneed"][cal]
        ell = int(np.ceil((m + 1) * target))
        R_sel = int(np.sort(Rn)[min(ell, m) - 1])
        R_sel = min(R_sel, efc)
        reads = np.full(len(tst), R_sel, dtype=np.int32)
        pres = T["Rneed"][tst] <= R_sel
        ngt = (T["gtR"][tst] <= R_sel).sum(1).astype(np.int32)
        rows["fixedR"].append(_stats(reads, pres, ngt, D))
        chosen["fixedR"].append(R_sel)
        # --- arm 4: calibrated margin, calibrated on the per-query event
        tau = float(np.sort(T["Hmaxmarg"][cal])[min(ell, m) - 1])
        reads = tau_cost(T, tau, tst)
        pres = T["Hmaxmarg"][tst] <= tau
        ngt = (T["gtm"][tst] <= tau).sum(1).astype(np.int32)
        rows["calibQ"].append(_stats(reads, pres, ngt, D))
        chosen["calibQ"].append(tau)
        # --- arm 4b: the paper's theorem as written (per-answer marginal, rho)
        J = rng.integers(0, K, size=m)
        Z = T["marg"][cal, J]
        l2 = int(np.ceil((m + 1) * (1.0 - rho_theorem)))
        tau2 = float(np.sort(Z)[min(l2, m) - 1])
        reads2 = tau_cost(T, tau2, tst)
        pres2 = T["Hmaxmarg"][tst] <= tau2
        ngt2 = (T["gtm"][tst] <= tau2).sum(1).astype(np.int32)
        st = _stats(reads2, pres2, ngt2, D)
        st["ans_recall"] = float((T["marg"][tst] <= tau2).mean())
        rows["calib_theorem"].append(st)
        chosen["calib_theorem"].append(tau2)
        # --- arm 5: calibrated prefix + the bound pruning inside it
        cb = calib_bound_pass(P, T, tau, eps_default)
        rows["calib_bound"].append(_stats(cb["reads"][tst], cb["pres"][tst],
                                          cb["ngt"][tst], D))
        # --- floor: the per-query minimum prefix
        rows["oracle"].append(_stats(T["Rneed"][tst],
                                     np.ones(len(tst), bool),
                                     (T["gtR"][tst] <= T["Rneed"][tst, None]).sum(1)
                                     .astype(np.int32), D))
    agg = {}
    for a, lst in rows.items():
        keys = lst[0].keys()
        agg[a] = {f"{f}": float(np.mean([x[f] for x in lst])) for f in keys}
        agg[a].update({f"{f}_sd": float(np.std([x[f] for x in lst], ddof=1))
                       for f in keys})
    out = dict(key=key, label=label(key), ef=int(ef), nq=int(nqq), D=D,
               N=P["N"], target=target, B=B, m=m, eps_default=eps_default,
               rho_theorem=rho_theorem, arms=agg, mono_eps=bool(mono),
               chosen={a: (float(np.mean(v)), float(np.std(v)))
                       for a, v in chosen.items()},
               eps_sweep={str(e): dict(reads=float(BP[e]["reads"].mean()),
                                       pres=float(BP[e]["pres"].mean()),
                                       recall=float(BP[e]["ngt"].mean() / K),
                                       readsM=float(BM[e]["reads"].mean()),
                                       presM=float(BM[e]["pres"].mean()),
                                       reads2p=float(B2[e]["reads"].mean()),
                                       pres2p=float(B2[e]["pres"].mean()))
                          for e in EPS_GRID},
               io_rounds=dict(bound_default="per-read", bound_tuned="per-read",
                              boundM_tuned=1, bound2p_tuned=2, fixedR=1,
                              calibQ=1, calib_theorem=1,
                              calib_bound="per-read", oracle=1),
               ndis=float(P["ndis"].mean()),
               inpool=float((P["gtpos"] >= 0).mean()))
    return out, P, T, BP


# ---------------------------------------------------- is the bound too wide?
def tightness(P, T, BP, eps_default=1.9):
    """|dhat - d| / rad : the eps0 each candidate would actually have needed."""
    dhat, rad, dex, Hm = P["dhat"], P["rad"], P["dex"], P["Hmask"]
    nq, ef = dhat.shape
    fin = np.isfinite(dex) & (rad > 0)
    ratio = np.abs(dhat - dex) / np.where(rad > 0, rad, np.nan)
    rk = np.broadcast_to(np.arange(ef), (nq, ef))
    bands = dict(
        all=fin,
        inH=fin & Hm,
        boundary=fin & (rk >= K) & (rk < 10 * K),   # just outside the estimated top-k
        deep=fin & (rk >= 10 * K))
    o = {}
    for nm, msk in bands.items():
        v = ratio[msk]
        v = v[np.isfinite(v)]
        o[nm] = dict(n=int(v.size), mean=float(v.mean()),
                     p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95)),
                     p99=float(np.quantile(v, .99)),
                     p999=float(np.quantile(v, .999)), max=float(v.max()),
                     over_default=float((v > eps_default).mean()))
    r = BP[eps_default]
    o["waste"] = dict(mean=float(r["waste"].mean()),
                      share=float(r["waste"].sum() / max(r["reads"].sum(), 1)),
                      reads=float(r["reads"].mean()),
                      pres=float(r["pres"].mean()))
    # what the bound would cost if its width were only as wide as the data needs
    need = float(np.quantile(ratio[fin][np.isfinite(ratio[fin])], .999))
    e_near = min(EPS_GRID, key=lambda e: abs(e - need))
    o["tightest"] = dict(eps_p999=need, eps_grid=e_near,
                         reads=float(BP[e_near]["reads"].mean()),
                         pres=float(BP[e_near]["pres"].mean()))
    o["oracle_reads"] = float(T["Rneed"].mean())
    return o


def main(keys, efs=(128, 256, 512), nq=4000, target=0.99, B=20):
    res = {}
    for key in keys:
        for ef in efs:
            try:
                out, P, T, BP = campaign(key, ef, nq, target=target, B=B)
            except FileNotFoundError as e:
                print(f"  skip {key} ef={ef}: {e}", flush=True)
                continue
            out["tight"] = tightness(P, T, BP)
            res[f"{key}_ef{ef}"] = out
            a = out["arms"]
            print(f"{key} ef={ef}: bnd@1.9 {a['bound_default']['reads']:.1f}"
                  f"/{a['bound_default']['pres']:.3f} | bnd* "
                  f"{a['bound_tuned']['reads']:.1f}@{out['chosen']['bound_tuned'][0]:.2f}"
                  f" | bndM* {a['boundM_tuned']['reads']:.1f}"
                  f"@{out['chosen']['boundM_tuned'][0]:.2f} | bnd2p* "
                  f"{a['bound2p_tuned']['reads']:.1f} | fixedR {a['fixedR']['reads']:.1f}"
                  f" | calibQ {a['calibQ']['reads']:.1f}/{a['calibQ']['pres']:.3f}"
                  f" | oracle {a['oracle']['reads']:.1f}", flush=True)
            with open(f"{OUT}/refsel_{key}_ef{ef}_nq{nq}.json", "w") as f:
                json.dump(out, f, indent=1)
    return res


if __name__ == "__main__":
    nq = int(os.environ.get("REFSEL_NQ", 4000))
    efs = tuple(int(x) for x in os.environ.get("REFSEL_EFS", "128,256,512").split(","))
    main(sys.argv[1:], efs=efs, nq=nq)


# ------------------------------------------------- full rule-knob sweeps
def curves(key, ef, nq, eps_grid=EPS_GRID):
    """(reads, per-query preservation, global Recall@k) along each rule's knob.

    Lets the arms be compared at a matched preservation AND at a matched global
    recall, which are different axes: pruning a pool-top-k member costs recall
    only when that member is also a GLOBAL answer.
    """
    P = load_pool(key, ef, nq)
    T = prefix_tables(P)
    D, nqq, efc = P["D"], *P["dhat"].shape
    out = dict(key=key, ef=int(ef), D=D, nq=nqq,
               ceiling=float((P["gtpos"] >= 0).mean()), arms={})

    def pack(reads, pres, ngt):
        return dict(reads=float(np.mean(reads)),
                    reads_p95=float(np.quantile(reads, .95)),
                    pres=float(np.mean(pres)),
                    recall=float(np.mean(ngt) / K))
    BP = {e: bound_pass(P, e) for e in eps_grid}
    out["arms"]["bound_seq"] = [dict(knob=e, **pack(BP[e]["reads"], BP[e]["pres"],
                                                    BP[e]["ngt"])) for e in eps_grid]
    for nm, fn in (("bound_margin", bound_margin), ("bound_2phase", bound_2phase)):
        out["arms"][nm] = []
        for e in eps_grid:
            r = fn(P, T, e)
            out["arms"][nm].append(dict(knob=e, **pack(r["reads"], r["pres"], r["ngt"])))
    # calibrated margin: the knob is tau itself
    taus = np.unique(np.quantile(T["Hmaxmarg"], np.linspace(0.80, 1.0, 41)))
    out["arms"]["calib_margin"] = []
    for tau in taus:
        reads = tau_cost(T, float(tau))
        out["arms"]["calib_margin"].append(dict(
            knob=float(tau), **pack(reads, T["Hmaxmarg"] <= tau,
                                    (T["gtm"] <= tau).sum(1))))
    # fixed prefix
    Rs = sorted(set([int(x) for x in np.unique(np.quantile(T["Rneed"],
                    np.linspace(0.5, 1.0, 41)))] + [10, 20, 30, 40, 60, 80, 120,
                    160, 240, 320, 480]))
    out["arms"]["fixedR"] = [dict(knob=R, **pack(np.full(nqq, min(R, efc)),
                                                 T["Rneed"] <= R,
                                                 (T["gtR"] <= R).sum(1)))
                             for R in Rs if R <= efc]
    out["arms"]["oracle"] = [dict(knob=0, **pack(T["Rneed"], np.ones(nqq, bool),
                                                 (T["gtR"] <= T["Rneed"][:, None]).sum(1)))]
    return out


# =====================================================================
#  Per-query critical knob values: the clean form of the comparison.
#
#  Every rule here is a one-parameter family of selection thresholds, so for
#  each query there is a smallest knob value that keeps all of H(q):
#
#    bound_seq     eps_need(q)  (bisected; the rule is monotone in eps)
#    bound_margin  max_{j in H} (dhat_j - dhat_k)/rad_j      closed form
#    bound_2phase  max_{j in H, rank>=k} (dhat_j - theta)/rad_j
#    calib_margin  max_{j in H} (dhat_j - dhat_k)            closed form
#    fixedR        max_{j in H} rank_j + 1
#
#  Calibrating any of them is then the SAME conformal step -- the
#  ceil((m+1)p)-th order statistic of the critical values on the calibration
#  half -- so no arm is helped or hurt by the coarseness of a search grid.
# =====================================================================
def _seq_pres_one(lb, dx, hm, k=K):
    heap = []; theta = float("inf"); ok = True
    for j in range(len(lb)):
        if lb[j] < theta:
            d = dx[j]
            if d == float("inf"):
                continue
            if len(heap) < k:
                heapq.heappush(heap, -d)
                if len(heap) == k:
                    theta = -heap[0]
            elif d < -heap[0]:
                heapq.heapreplace(heap, -d); theta = -heap[0]
        elif hm[j]:
            ok = False                      # a member of H(q) was pruned
    return ok


def eps_need_seq(P, lo=0.0, hi=6.0, iters=16):
    """Smallest eps0 at which RaBitQ's sequential rule keeps all of H(q)."""
    dhat, rad, dex, Hm = P["dhat"], P["rad"], P["dex"], P["Hmask"]
    nq, ef = dhat.shape
    out = np.empty(nq, dtype=np.float64)
    for r in range(nq):
        dh, rd = dhat[r], rad[r]
        dx = dex[r].tolist(); hm = Hm[r].tolist()
        a, b = lo, hi
        if _seq_pres_one((dh - a * rd).tolist(), dx, hm):
            out[r] = a; continue
        if not _seq_pres_one((dh - b * rd).tolist(), dx, hm):
            out[r] = np.inf; continue
        for _ in range(iters):
            m = 0.5 * (a + b)
            if _seq_pres_one((dh - m * rd).tolist(), dx, hm):
                b = m
            else:
                a = m
        out[r] = b
    return out


def critical_values(P, T):
    dhat, rad, Hord, dex = P["dhat"], P["rad"], P["Hord"], P["dex"]
    nq, ef = dhat.shape
    rH = np.take_along_axis(rad, Hord, 1)
    mH = T["marg"]                                   # dhat_j - dhat_k, clipped at 0
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_m = np.where(rH > 0, (np.take_along_axis(dhat, Hord, 1) - T["dk"]) / rH,
                         0.0).max(1)
    theta2 = np.max(np.where(np.isfinite(dex[:, :K]), dex[:, :K], -np.inf), 1,
                    keepdims=True)
    far = Hord >= K
    with np.errstate(divide="ignore", invalid="ignore"):
        e2 = np.where(far & (rH > 0),
                      (np.take_along_axis(dhat, Hord, 1) - theta2) / np.where(rH > 0, rH, 1),
                      -np.inf).max(1)
    return dict(bound_seq=eps_need_seq(P),
                bound_margin=np.maximum(eps_m, 0.0),
                bound_2phase=np.maximum(e2, 0.0),
                calib_margin=T["Hmaxmarg"].astype(np.float64),
                fixedR=T["Rneed"].astype(np.float64))


def cost_at(P, T, arm, knob, rows=None, cache=None):
    """Mean/percentile FP32 reads of one arm at one knob value, on `rows`."""
    sl = slice(None) if rows is None else rows
    if arm == "calib_margin":
        reads = tau_cost(T, float(knob), rows)
    elif arm == "fixedR":
        n = P["dhat"].shape[0] if rows is None else len(rows)
        reads = np.full(n, min(int(np.ceil(knob)), P["dhat"].shape[1]), dtype=np.int32)
    elif arm == "bound_margin":
        reads = bound_margin(P, T, float(knob))["reads"][sl]
    elif arm == "bound_2phase":
        reads = bound_2phase(P, T, float(knob))["reads"][sl]
    elif arm == "bound_seq":
        kk = round(float(knob), 3)
        if cache is None or kk not in cache:
            r = bound_pass(P, kk)
            if cache is not None:
                cache[kk] = r
        else:
            r = cache[kk]
        reads = r["reads"][sl]
    else:
        raise ValueError(arm)
    return np.asarray(reads, dtype=np.float64)


def campaign2(key, ef, nq, target=0.99, B=20, seed=20261120, eps_default=1.9):
    """The conformal comparison: same calibration step for every rule family."""
    P = load_pool(key, ef, nq)
    T = prefix_tables(P)
    CV = critical_values(P, T)
    nqq = P["dhat"].shape[0]
    D = P["D"]
    rng = np.random.default_rng(seed)
    m = nqq // 2
    ell = lambda mm: int(np.ceil((mm + 1) * target))
    arms = list(CV.keys())
    acc = {a: dict(reads=[], p95=[], pres=[], knob=[]) for a in arms}
    acc["bound_default"] = dict(reads=[], p95=[], pres=[], knob=[])
    cache = {}
    for b in range(B):
        perm = rng.permutation(nqq)
        cal, tst = perm[:m], perm[m:]
        for a in arms:
            v = np.sort(CV[a][cal])
            knob = float(v[min(ell(m), m) - 1])
            if not np.isfinite(knob):
                knob = float(np.nanmax(CV[a][np.isfinite(CV[a])])) * 1.5
            rd = cost_at(P, T, a, knob, tst, cache)
            acc[a]["reads"].append(rd.mean())
            acc[a]["p95"].append(np.quantile(rd, .95))
            acc[a]["pres"].append(float((CV[a][tst] <= knob).mean()))
            acc[a]["knob"].append(knob)
        rd = cost_at(P, T, "bound_seq", eps_default, tst, cache)
        acc["bound_default"]["reads"].append(rd.mean())
        acc["bound_default"]["p95"].append(np.quantile(rd, .95))
        acc["bound_default"]["pres"].append(
            float((CV["bound_seq"][tst] <= eps_default).mean()))
        acc["bound_default"]["knob"].append(eps_default)
    out = dict(key=key, label=label(key), ef=int(ef), nq=nqq, D=D, target=target,
               B=B, m=m, oracle=float(T["Rneed"].mean()),
               oracle_p95=float(np.quantile(T["Rneed"], .95)),
               arms={a: {k: [float(np.mean(v)), float(np.std(v, ddof=1))]
                         for k, v in d.items()} for a, d in acc.items()},
               crit={a: dict(p50=float(np.nanquantile(CV[a], .5)),
                             p90=float(np.nanquantile(CV[a], .9)),
                             p99=float(np.nanquantile(CV[a], .99)),
                             inf_share=float((~np.isfinite(CV[a])).mean()))
                     for a in arms})
    return out, P, T, CV


# ============================================================================
#  Does the per-candidate radius carry information a single scalar does not?
#
#  One family interpolates the two rules:   refine x iff
#        dhat(x) - dhat_k(q)  <  t * (rad(x)/mean rad)^beta
#  beta = 0 is the calibrated scalar margin, beta = 1 is RaBitQ's bound used as
#  a margin (t then plays the role of eps0 * mean rad).  t is calibrated
#  conformally at every beta, beta is picked on the calibration half, and the
#  winner is scored on the test half.  If the optimum sits at an interior beta
#  and beats both ends, the radius is informative; if it does not, the bound's
#  per-candidate shape buys nothing that one workload scalar does not.
# ============================================================================
BETA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5)


def beta_family(P, T, betas=BETA_GRID):
    dhat, rad, Hord = P["dhat"], P["rad"], P["Hord"]
    fin = np.isfinite(P["dex"])
    rbar = float(rad[fin].mean())
    w = np.where(fin, rad / rbar, 1.0)
    marg = dhat - T["dk"]
    mH = np.take_along_axis(marg, Hord, 1)
    wH = np.take_along_axis(w, Hord, 1)
    out = {}
    for b in betas:
        out[b] = dict(crit=(mH / np.power(wH, b)).max(1),
                      score=marg / np.power(w, b), fin=fin)
    return out, rbar


def beta_campaign(key, ef, nq, target=0.99, B=20, seed=20261120):
    P = load_pool(key, ef, nq); T = prefix_tables(P)
    F, rbar = beta_family(P, T)
    nqq = P["dhat"].shape[0]; m = nqq // 2
    rng = np.random.default_rng(seed)
    ell = int(np.ceil((m + 1) * target))
    sortsc = {b: np.sort(np.where(F[b]["fin"], F[b]["score"], np.inf), axis=1)
              for b in F}
    res = {b: dict(reads=[], pres=[]) for b in F}
    sel = []
    for _ in range(B):
        perm = rng.permutation(nqq); cal, tst = perm[:m], perm[m:]
        best, bestc = None, np.inf
        for b in F:
            t = float(np.sort(F[b]["crit"][cal])[min(ell, m) - 1])
            rd = np.array([np.searchsorted(sortsc[b][i], t, side="right")
                           for i in tst], dtype=np.float64)
            res[b]["reads"].append(rd.mean())
            res[b]["pres"].append(float((F[b]["crit"][tst] <= t).mean()))
            cc = np.array([np.searchsorted(sortsc[b][i], t, side="right")
                           for i in cal], dtype=np.float64).mean()
            if cc < bestc:
                best, bestc = b, cc
        sel.append(best)
    return dict(key=key, ef=int(ef), nq=nqq, target=target, B=B, rbar=rbar,
                beta={str(b): dict(reads=[float(np.mean(v["reads"])),
                                          float(np.std(v["reads"], ddof=1))],
                                   pres=float(np.mean(v["pres"])))
                      for b, v in res.items()},
                beta_sel=[float(np.mean(sel)), float(np.std(sel))],
                beta_sel_mode=float(max(set(sel), key=sel.count)))
