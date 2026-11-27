"""The three tables v16 adds, written straight from the measurements."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import OUT, CONFIGS, label
import calib, tune

NVEC = {"msmarco_qw06": 1_000_000, "nq_qw06": 1_000_000,
        "quora_qw06": 522_931, "msmarco_bgem3": 1_000_000,
        "fever_bgem3": 1_000_000}

PAPER = "/home/sieun/projects/BBQETA/paper"
NICE = {"msmarco_qw06": ("MS MARCO", "Qwen-1024"),
        "nq_qw06": ("NQ", "Qwen-1024"),
        "quora_qw06": ("Quora", "Qwen-1024"),
        "dbpedia_qw06": ("DBpedia", "Qwen-1024"),
        "msmarco_bgem3": ("MS MARCO", "BGE-M3"),
        "nq_bgem3": ("NQ", "BGE-M3"),
        "quora_bgem3": ("Quora", "BGE-M3"),
        "fever_bgem3": ("FEVER", "BGE-M3"),
        "msmarco_ada2": ("MS MARCO", "ada-002"),
        "dbpedia_ada2": ("DBpedia", "ada-002"),
        "dbpedia_bgel": ("DBpedia", "BGE-large"),
        "dbpedia_bgeb": ("DBpedia", "BGE-base"),
        "nq_nem8": ("NQ", "Nemotron-4096"),
        "quora_nem8": ("Quora", "Nemotron-4096"),
        "nq_e5m": ("NQ", "E5-Mistral-4096"),
        "quora_e5m": ("Quora", "E5-Mistral-4096"),
        "dbpedia_e5m": ("DBpedia", "E5-Mistral-4096")}


# ------------------------------------------------------------------ Table: calibration as specified
def tab_calibspec(keys, efs=(128, 512), rho=0.01, m=1000, B=400):
    rows = []
    for k in keys:
        for ef in efs:
            try:
                rows.append(calib.campaign(k, ef, rho=rho, m=m, B=B))
            except FileNotFoundError:
                pass
    L = [r"\begin{tabular*}{\columnwidth}"
         r"{@{\extracolsep{\fill}}lr|rrr|rr|rrr@{}}", r"\toprule",
         r"&& \multicolumn{3}{c|}{\textbf{achieved}}"
         r" & \multicolumn{2}{c|}{$\Pr[\hat\tau{=}\infty]$}"
         r" & \multicolumn{3}{c}{\textbf{reads / query}} \\",
         r"\textbf{Configuration} & $\boldsymbol{ef}$"
         r" & $\pi$ & $\overline{\mathrm{R@}10}$ & short"
         r" & meas. & pred. & $\bar R$ & p99 & $\mathbb{E}$ \\", r"\midrule"]
    prev = None
    for r in rows:
        c, e = NICE[r["key"]]
        if prev is not None and prev != r["key"]:
            L.append(r"\midrule")
        prev = r["key"]
        fin = np.isfinite(r["R_fin"])
        Rf = f"{r['R_fin']:.0f}" if fin else "---"
        Rp = f"{r['Rp99_fin']:.0f}" if fin else "---"
        # expected reads including the exhaustive branch the guarantee needs
        N = NVEC[r["key"]]
        pfb = r["fallback"]
        ER = (1 - pfb) * (r["R_fin"] if fin else 0.0) + pfb * N
        ERs = f"{ER/1e3:.0f}" if ER >= 1e4 else f"{ER/1e3:.1f}"
        L.append(f"{c} / {e} & {r['ef']} & {r['L_visit']:.4f} & "
                 f"{r['recall_mean']:.4f} & {r['shortfall']:.2f} & "
                 f"{r['fallback']:.3f} & {r['fallback_pred']:.3f} & "
                 f"{Rf} & {Rp} & {ERs} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}"]
    open(f"{PAPER}/tab_calibspec.tex", "w").write("\n".join(L) + "\n")
    json.dump(rows, open(f"{OUT}/calibspec.json", "w"), indent=1)
    return rows


# ------------------------------------------------------------------ Table: Recipe T out of sample
SHORT = {"Nemotron-4096": "Nem-4096", "E5-Mistral-4096": "E5M-4096"}


def tab_recipe(rho=0.01, lam=10.0, m=1000):
    r = tune.run_recipe(rho=rho, lam=lam, m=m)

    def g(v, f="{:.4f}"):
        try:
            return f.format(v) if v is not None and np.isfinite(v) else "---"
        except (TypeError, ValueError):
            return "---"

    L = [r"\begin{tabular*}{\columnwidth}"
         r"{@{\extracolsep{\fill}}lr|rr|rrr|rr@{}}", r"\toprule",
         r"&& \multicolumn{2}{c|}{$\pi(\hat{ef})$}"
         r" & \multicolumn{3}{c|}{\textbf{traffic} $\boldsymbol{\bar R}$}"
         r" & \multicolumn{2}{c}{\textbf{outcome}} \\",
         r"\textbf{Configuration} & $\hat{ef}$ & pred. & true"
         r" & M1 & M2 & meas."
         r" & $\overline{\mathrm{R@}10}$ & esc. \\", r"\midrule"]
    for h in r["held"]:
        c, e = NICE[h["key"]]
        e = SHORT.get(e, e)
        ef = h["ef_hat"]
        lvp = h["L_visit_pred"].get(ef) if ef else None
        lvt = h["L_visit_true"].get(ef) if ef else None
        L.append(f"{c} / {e} & {ef if ef else '$>$1024'} & "
                 f"{g(lvp)} & {g(lvt)} & "
                 f"{g(h.get('R_hat'), '{:.0f}')} & "
                 f"{g(h.get('R_hat2'), '{:.0f}')} & "
                 f"{g(h.get('cal_R'), '{:.1f}')} & "
                 f"{g(h.get('cal_recall'))} & "
                 f"{g(h.get('cal_escal'), '{:.3f}')} \\\\")
    L += [r"\bottomrule", r"\end{tabular*}"]
    open(f"{PAPER}/tab_recipe.tex", "w").write("\n".join(L) + "\n")
    json.dump(r, open(f"{OUT}/recipe.json", "w"), indent=1, default=str)
    return r


if __name__ == "__main__":
    ANCH = ["msmarco_qw06", "nq_qw06", "quora_qw06", "msmarco_bgem3",
            "fever_bgem3"]
    if "calib" in sys.argv:
        for x in tab_calibspec(ANCH):
            print(x["key"], x["ef"], round(x["L_visit"], 4),
                  round(x["recall_mean"], 4), round(x["fallback"], 3))
    if "recipe" in sys.argv:
        r = tab_recipe()
        print(json.dumps(r["visit"], indent=1))
        print(json.dumps(r["traffic"], indent=1))
        for h in r["held"]:
            print(json.dumps(h, default=str))
