"""Regenerate v16's tables and print every number the prose quotes."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import OUT
import calib, tune, tables_v16 as T

ANCHORS = ["msmarco_qw06", "nq_qw06", "quora_qw06", "msmarco_bgem3",
           "fever_bgem3"]
CAL_EFS = (128, 1024)
LOSS_EFS = (128, 1024)


def main():
    out = {}
    out["calibspec"] = T.tab_calibspec(ANCHORS, efs=CAL_EFS)
    out["recipe"] = T.tab_recipe()

    # ---- fallback identity: worst deviation from the binomial prediction
    d = [abs(r["fallback"] - r["fallback_pred"]) for r in out["calibspec"]]
    out["fallback_maxdev"] = max(d)
    out["fallback_signed"] = [round(r["fallback"] - r["fallback_pred"], 3)
                              for r in out["calibspec"]]

    # ---- D1: is the narrowest attainable width the cheapest cell?
    wc = tune.width_check(tune.DEV)
    agree, tot, crit = 0, 0, []
    for r in wc:
        if r.get("ef_min") is None:
            continue
        tot += 1
        if all(r[k]["agree"] for k in r if k.startswith("lam")):
            agree += 1
    for k in tune.DEV:
        g = tune.ladder(k)
        if not g:
            continue
        efs = sorted(g)
        em = next((e for e in efs if 1 - g[e]["ceiling"] < tune.RHO), None)
        if em is None or em == efs[-1]:
            continue
        nx = efs[efs.index(em) + 1]
        R0 = tune.cheapest_R(g[em], 1 - tune.RHO)
        R1 = tune.cheapest_R(g[nx], 1 - tune.RHO)
        if R1 is not None and R0 and R0 > R1:
            crit.append((k, (g[nx]["ndis"] - g[em]["ndis"]) / (R0 - R1)))
    out["width_agree"] = [agree, tot]
    out["lam_crit"] = sorted(round(c) for _, c in crit)
    out["lam_crit_by"] = [(k, round(c)) for k, c in crit]

    # ---- the two fitted models
    out["visit_fit"] = tune.fit_visit(tune.DEV)
    out["traffic_fit"] = tune.run_recipe()["traffic"] \
        if False else None       # already inside tab_recipe's json
    r = json.load(open(f"{OUT}/recipe.json"))
    out["visit_fit"] = r["visit"]
    out["traffic_fit"] = r["traffic"]
    out["held"] = r["held"]

    # ---- certified width vs attainable width, on the development grid
    rows = []
    for p in r["dev"]:
        if p.get("cost_cert") and p.get("cost_fixed"):
            rows.append((p["key"], p["ef_min"], p["ef_cert"],
                         p["cost_cert"] / p["cost_fixed"]))
    out["cert_price"] = rows

    json.dump(out, open(f"{OUT}/v16_numbers.json", "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    o = main()
    print("== fallback identity: max |meas - pred| =", round(o["fallback_maxdev"], 3))
    print("   signed devs:", o["fallback_signed"])
    print("== D1 width agreement:", o["width_agree"], " lam_crit:", o["lam_crit"])
    print("== visit fit:", json.dumps(o["visit_fit"]))
    print("== traffic fit:", json.dumps(o["traffic_fit"]))
    print("== cert price (key, ef_min, ef_cert, cost ratio):")
    for x in o["cert_price"]:
        print("   ", x[0], x[1], x[2], round(x[3], 2))
    print("== held out:")
    for h in o["held"]:
        print("   ", h["key"], "eta", round(float(h["eta"]), 3),
              "ef_hat", h["ef_hat"], "ef_cert_true", h["ef_cert_true"],
              "R_hat", round(float(h["R_hat"]), 1),
              "cal_R", h.get("cal_R"), "rec", h.get("cal_recall"),
              "escal", h.get("cal_escal"),
              "cost/sweep", (round(h["cost"]/h["cost_sweep"], 2)
                             if h.get("cost") and h.get("cost_sweep") else None))
