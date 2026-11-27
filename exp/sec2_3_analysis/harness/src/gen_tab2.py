"""Emit tab:geometry for the representative setting set."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tables import load_all, eta_agg
R = load_all()
BLOCKS = [(r"\emph{\msmarco{}, encoder}",
           [("msmarco_ada2", "ada-002"), ("msmarco_bgem3", "BGE-M3"),
            ("msmarco_qw06", "Qwen3-0.6B"), ("msmarco_e5m", "E5-Mistral")]),
          (r"\emph{Qwen3-Embedding-0.6B, corpus}",
           [("nq_qw06", r"\nq{}"), ("quora_qw06", r"\quora{}"), ("dbpedia_qw06", r"\dbp{}")]),
          (r"\emph{Conventional}", [("glove", "GloVe"), ("gist", "GIST")])]
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{$\kappa$, top-rank decay $\beta$, $\eta=\nu/(\kappa\beta)$, standardized",
     r"rank-10 score $z_{10}$ and p95 $R^{*}$ under ADC-BQ, on one encoder axis and one",
     r"corpus axis. $\beta$ depends only on the exact scores. GIST gives every vector the",
     r"same sign code, so $\kappa$ and $\eta$ are undefined.}",
     r"\label{tab:geometry}", r"\scriptsize", r"\setlength{\tabcolsep}{3.0pt}",
     r"\renewcommand{\arraystretch}{1.05}",
     r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lrrrrr|r@{}}", r"\toprule",
     r"\textbf{Representation} & $d$ & $\kappa$ & $\beta$ & $\eta$ & $z_{10}$ & \textbf{p95 $R^{*}$} \\",
     r"\midrule"]
for hdr, items in BLOCKS:
    L.append(r"\multicolumn{6}{@{}l|}{%s} & \\" % hdr)
    for k, lab in items:
        if k not in R:
            L.append(f"{lab} & \\multicolumn{{6}}{{c}}{{\\emph{{pending}}}} \\\\"); continue
        a = R[k]["arms"]["adc_bq"]; e = eta_agg(R[k], a)
        es = "---" if not np.isfinite(e) else f"{e:.2f}"
        ks = "---" if not np.isfinite(a["kappa"]) else f"{a['kappa']:.3f}"
        L.append(f"{lab} & {R[k]['d']} & {ks} & {R[k]['beta']:.3f} & {es} & "
                 f"{R[k]['z10']:.2f} & {a['Rstar_p95']:,.0f} \\\\".replace(",", r"\,"))
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular*}", r"\end{table}"]
print("\n".join(L))
