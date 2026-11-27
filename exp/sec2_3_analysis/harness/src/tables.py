"""Markdown tables from out/<key>.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import ORDER, OUT, CONFIGS
import quantizers as QZ


def load_all():
    R = {}
    for k in ORDER:
        p = f"{OUT}/{k}.json"
        if os.path.exists(p):
            R[k] = json.load(open(p))
    return R


def eta_agg(r, a, window="1-100"):
    """eta as the paper computes it: from the aggregate kappa and beta of the
    table, not as a per-query mean. nu = sqrt(1-kappa^2) by eq. (1)."""
    k = a["kappa"]
    b = r["beta"] if window == "1-100" else r["beta_10_100"]
    if not np.isfinite(k) or k <= 0 or b <= 0:
        return np.nan
    return float(np.sqrt(max(1 - k * k, 0.0)) / (k * b))


def fmt(v, n=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:.{n}f}"


def rfmt(v):
    if v is None or not np.isfinite(v):
        return "--"
    return f"{v:,.0f}"


def table3(R):
    """Table 3 replacement: one row per configuration, ADC-BQ as the quantizer."""
    L = ["| Corpus | Encoder | d | N | nq | kappa | beta | beta(10-100) | eta | eta(10-100) | z10 | p95 R* | CR@30 | CR@80 |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in ORDER:
        if k not in R:
            continue
        r = R[k]
        a = r["arms"].get("adc_bq")
        if a is None:
            continue
        b, b2 = r["beta"], r["beta_10_100"]
        e1, eta2 = eta_agg(r, a), eta_agg(r, a, "10-100")
        L.append(f"| {r['corpus']} | {r['encoder']} | {r['d']} | {r['N']:,} | {r['nq']} | "
                 f"{fmt(a['kappa'])} | {fmt(b)} | {fmt(b2)} | {fmt(e1,2)} | "
                 f"{fmt(eta2,2)} | {fmt(r['z10'],2)} | {rfmt(a['Rstar_p95'])} | "
                 f"{fmt(a['candrec']['30'] if '30' in a['candrec'] else a['candrec'][30],3)} | "
                 f"{fmt(a['candrec']['80'] if '80' in a['candrec'] else a['candrec'][80],3)} |")
    return "\n".join(L)


def table_quant(R):
    """Per configuration x quantizer."""
    L = ["| Config | Quantizer | B/vec | kappa | nu | eta | p95 R* | median R* | CR@30 | CR@80 |",
         "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for k in ORDER:
        if k not in R:
            continue
        r = R[k]
        tag = f"{r['corpus']} / {r['encoder']}" if r["encoder"] != "-" else r["corpus"]
        for arm in QZ.ARMS:
            a = r["arms"].get(arm)
            if a is None:
                continue
            cr = a["candrec"]
            L.append(f"| {tag} | {a['name']} | {a['bytes_per_vec']:.0f} | {fmt(a['kappa'],4)} | "
                     f"{fmt(a['nu'],4)} | {fmt(eta_agg(r,a),3)} | {rfmt(a['Rstar_p95'])} | "
                     f"{rfmt(a['Rstar_med'])} | {fmt(cr['30'],4)} | {fmt(cr['80'],4)} |")
    return "\n".join(L)


def diagnostics(R):
    """Does eta still rank-predict the candidate budget, and does it do so ACROSS
    quantizers as well as across corpora?"""
    from scipy.stats import spearmanr
    out = []

    # (i) across configurations, ADC-BQ only -- the paper's claim
    ks = [k for k in ORDER if k in R and "adc_bq" in R[k]["arms"]]
    eta = np.array([eta_agg(R[k], R[k]["arms"]["adc_bq"]) for k in ks])
    r95 = np.array([R[k]["arms"]["adc_bq"]["Rstar_p95"] for k in ks])
    c80 = np.array([R[k]["arms"]["adc_bq"]["candrec"]["80"] for k in ks])
    fm = np.isfinite(eta)          # GIST: sign(x) is constant, kappa undefined
    out.append(("across configurations (ADC-BQ)", int(fm.sum()),
                spearmanr(eta[fm], r95[fm]).statistic,
                spearmanr(eta[fm], c80[fm]).statistic))
    lm = fm & np.array([R[k]["group"] != "classic" for k in ks])
    out.append(("across learned-embedding configurations (ADC-BQ)", int(lm.sum()),
                spearmanr(eta[lm], r95[lm]).statistic,
                spearmanr(eta[lm], c80[lm]).statistic))

    # (ii) across quantizers, within each configuration
    per = []
    for k in ks:
        e = [eta_agg(R[k], R[k]["arms"][a]) for a in QZ.ARMS if a in R[k]["arms"]]
        p = [R[k]["arms"][a]["Rstar_p95"] for a in QZ.ARMS if a in R[k]["arms"]]
        c = [R[k]["arms"][a]["candrec"]["80"] for a in QZ.ARMS if a in R[k]["arms"]]
        e, p, c = np.array(e), np.array(p), np.array(c)
        m = np.isfinite(e) & np.isfinite(p)
        if m.sum() >= 4:
            per.append((k, spearmanr(e[m], p[m]).statistic, spearmanr(e[m], c[m]).statistic))

    # (iii) pooled over every (config, quantizer) pair
    E, P, C = [], [], []
    for k in ks:
        for a in QZ.ARMS:
            if a in R[k]["arms"]:
                E.append(eta_agg(R[k], R[k]["arms"][a])); P.append(R[k]["arms"][a]["Rstar_p95"])
                C.append(R[k]["arms"][a]["candrec"]["80"])
    E, P, C = np.array(E), np.array(P), np.array(C)
    m = np.isfinite(E) & np.isfinite(P)
    out.append(("pooled over (config x quantizer)", int(m.sum()),
                spearmanr(E[m], P[m]).statistic, spearmanr(E[m], C[m]).statistic))

    L = ["| Axis | n | Spearman(eta, p95 R*) | Spearman(eta, CandRec@80) |", "|---|--:|--:|--:|"]
    for nm, n, a, b in out:
        L.append(f"| {nm} | {n} | {fmt(a)} | {fmt(b)} |")
    for k, a, b in per:
        L.append(f"| within {k} (8 quantizers) | 8 | {fmt(a)} | {fmt(b)} |")
    return "\n".join(L)


if __name__ == "__main__":
    R = load_all()
    print("## Table A -- Section 4 quantities on the evaluated datasets (ADC-BQ)\n")
    print(table3(R))
    print("\n## Table B -- per quantizer\n")
    print(table_quant(R))
    print("\n## Table C -- is eta still a diagnostic?\n")
    print(diagnostics(R))


def tex_table3(R):
    """Paste-ready replacement for Table 3, grouped by the factor that varies."""
    rows = {"qwen": [], "bge": [], "classic": []}
    for k in ORDER:
        if k not in R or "adc_bq" not in R[k]["arms"]:
            continue
        r = R[k]; a = r["arms"]["adc_bq"]
        nm = r["corpus"] if r["group"] != "bge" else f"{r['corpus']} ({r['encoder']})"
        rows[r["group"]].append(
            f"{nm} & {r['d']} & {r['N']:,} & {fmt(a['kappa'])} & {fmt(r['beta'])} & "
            f"{fmt(eta_agg(r,a),2)} & {fmt(r['z10'],2)} & {rfmt(a['Rstar_p95'])} \\\\")
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Representation & $d$ & $N$ & $\kappa$ & $\beta$ & $\eta$ & $z_{10}$ & p95 $R^*$ \\",
         r"\midrule"]
    for g, head in (("qwen", r"\multicolumn{8}{l}{\emph{Qwen3-0.6B 1024D, corpus}}\\"),
                    ("bge", r"\multicolumn{8}{l}{\emph{BGE, encoder}}\\"),
                    ("classic", r"\multicolumn{8}{l}{\emph{Conventional}}\\")):
        if rows[g]:
            L.append(head); L += rows[g]; L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    return "\n".join(L)
