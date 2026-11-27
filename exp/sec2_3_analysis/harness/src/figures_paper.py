"""Paper-ready figures in the draft's own style (see paperstyle.py).

F1  replaces Fig. 5  -- z(r) and the per-rank 1-bit cost, one encoder family
F2  replaces Fig. 6a -- per-query z10 vs R*
F3  NEW, the contribution -- eta factorises: c_Q is a quantizer constant, and
                            the price of 1 bit collapses as beta grows
F4  optional, talk/appendix -- per-quantizer eta and p95 R* bars
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import paperstyle as PS
from paperstyle import ARM_ORDER, ARM_STYLE, CFG_STYLE, INK, RED, BLUE
PS.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from common import WORK, FIGS, CONFIGS, ORDER, K
from tables import load_all, eta_agg
import quantizers as QZ

# the six series of the draft's Fig. 5, now on the evaluated configurations
F1_KEYS = [("quora_bgem3", "quora", "Quora"),
           ("dbpedia_bgel", "dbpedia", "DBpedia"),
           ("msmarco_bgem3", "msmarco", "MS MARCO"),
           ("nq_bgem3", "nq", "NQ"),
           ("fever_bgem3", "fever", "FEVER"),
           ("glove", "glove", "GloVe")]
QWEN = [("quora_qw06", "quora", "Quora"), ("dbpedia_qw06", "dbpedia", "DBpedia"),
        ("msmarco_qw06", "msmarco", "MS MARCO"), ("nq_qw06", "nq", "NQ")]


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/{name}.{ext}")
    plt.close(fig)
    print("  ->", name)


def f1_zrank(R, keys=F1_KEYS, name="paper_f1_zrank"):
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.55))
    rr = np.arange(1, 101)
    H, L = [], []
    for key, sty, lab in keys:
        if key not in R:
            continue
        s = CFG_STYLE[sty]
        z = np.load(f"{WORK}/{key}_perq.npz")["z_curve"]
        PS.line(ax[0], rr, z, s, markevery=[0, 1, 2, 4, 9, 19, 49, 99])
        p95 = np.array(R[key]["arms"]["adc_bq"]["p95_Rstar_by_rank"])
        PS.line(ax[1], np.arange(1, K + 1), p95 / np.arange(1, K + 1), s)
        H.append(Line2D([], [], color=s["color"], marker=s["marker"],
                        markerfacecolor="white", markeredgecolor=s["color"]))
        L.append(lab)
    ax[0].set_xscale("log"); ax[0].set_xlabel("rank $r$"); ax[0].set_ylabel("$z(r)$")
    ax[0].set_title("(a) Exact ranking")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("rank $r$"); ax[1].set_ylabel("p95 $R^*(r)/r$")
    ax[1].set_title("(b) 1-bit cost")
    fig.tight_layout()
    PS.legend_below(fig, H, L, ncol=3, y=.02)
    _save(fig, name)


def f2_z10(R, name="paper_f2_z10"):
    keys = [(k, s, l) for k, s, l in F1_KEYS if k in R and CONFIGS[k]["group"] != "classic"]
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    Z, S, H, L = [], [], [], []
    for key, sty, lab in keys:
        s = CFG_STYLE[sty]
        d = np.load(f"{WORK}/{key}_perq.npz")
        z, r = d["z10"], d["Rstar_adc_bq"].astype(float)
        ax.scatter(z, r, s=1.2, alpha=.16, color=s["color"], linewidths=0, rasterized=True)
        Z.append(z); S.append(r)
        H.append(Line2D([], [], ls="none", marker=s["marker"], color=s["color"],
                        markerfacecolor="white", markeredgecolor=s["color"]))
        L.append(f"{lab} {spearmanr(z, r).statistic:+.2f}")
    Z, S = np.concatenate(Z), np.concatenate(S)
    ax.set_yscale("log"); ax.set_xlabel("$z_{10}$"); ax.set_ylabel("$R^*$")
    ax.set_title(f"{len(Z):,} queries, pooled $\\rho$ = {spearmanr(Z, S).statistic:+.2f}")
    fig.tight_layout()
    PS.legend_below(fig, H, L, ncol=2, y=.02)
    _save(fig, name)


def f3_factorisation(R, name="paper_f3_factorisation"):
    """(a) nu/kappa normalised by ADC-BQ: every configuration on one curve.
       (b) what the extra noise costs, against beta."""
    arms = [a for a in ARM_ORDER if a != "sq8"]
    learned = [k for k in ORDER if k in R and CONFIGS[k]["group"] != "classic"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.55))

    ib = arms.index("adc_bq")
    x = np.arange(len(arms))
    for k in learned:
        v = np.array([R[k]["arms"][a]["nu"] / R[k]["arms"][a]["kappa"] for a in arms])
        ax[0].plot(x, v / v[ib], color="#BBBBBB", lw=.7, zorder=1)
    V = np.array([[R[k]["arms"][a]["nu"] / R[k]["arms"][a]["kappa"] for a in arms]
                  for k in learned])
    V = V / V[:, [ib]]
    for j, a in enumerate(arms):
        st = ARM_STYLE[a]
        ax[0].plot([x[j]], [np.exp(np.log(V[:, j]).mean())], marker=st["marker"],
                   color=st["color"], markerfacecolor="white",
                   markeredgecolor=st["color"], ls="none", ms=5.5, zorder=3)
        ax[0].vlines(x[j], V[:, j].min(), V[:, j].max(), color=st["color"], lw=1.1, zorder=2)
    if "glove" in R:                      # the control that breaks the collapse
        v = np.array([R["glove"]["arms"][a]["nu"] / R["glove"]["arms"][a]["kappa"] for a in arms])
        ax[0].plot(x, v / v[ib], color=CFG_STYLE["glove"]["color"], ls="--", lw=1.0,
                   marker="x", ms=4, zorder=4)
    ax[0].axhline(1.0, color=INK, lw=.6, ls=":")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([ARM_STYLE[a]["label"] for a in arms], rotation=38,
                          ha="right", fontsize=7)
    ax[0].set_yscale("log"); ax[0].set_ylim(.13, 2.0)
    ax[0].set_ylabel("$(\\nu_Q/\\kappa_Q)\;/\;(\\nu_{\\mathrm{ADC}}/\\kappa_{\\mathrm{ADC}})$")
    ax[0].set_title("(a) the quantizer factor is a constant")

    b = np.array([R[k]["beta"] for k in learned])
    ex = np.array([R[k]["arms"]["adc_bq"]["Rstar_p95"] - R[k]["arms"]["opq"]["Rstar_p95"]
                   for k in learned], dtype=float)
    sl, ic = np.polyfit(np.log(b), np.log(ex), 1)
    xs = np.linspace(b.min() * .96, b.max() * 1.04, 60)
    ax[1].plot(xs, np.exp(ic) * xs ** sl, color=INK, lw=.8, ls="--", zorder=1)
    for i, k in enumerate(learned):
        qw = CONFIGS[k]["encoder"].startswith("Qwen")
        ax[1].plot([b[i]], [ex[i]], marker="o" if qw else "s",
                   color=RED if qw else BLUE, markerfacecolor="white",
                   markeredgecolor=RED if qw else BLUE, ls="none", ms=5, zorder=3)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("$\\beta$")
    ax[1].set_ylabel("extra candidates vs OPQ\n(p95 $R^*$, same 128 B)")
    ax[1].set_title(f"(b) price of 1 bit $\\propto \\beta^{{{sl:.1f}}}$, "
                    f"$\\rho$={spearmanr(b, ex).statistic:+.2f}")
    fig.tight_layout()
    H = [Line2D([], [], ls="none", marker="o", color=RED, markerfacecolor="white"),
         Line2D([], [], ls="none", marker="s", color=BLUE, markerfacecolor="white"),
         Line2D([], [], ls="--", color=CFG_STYLE["glove"]["color"], marker="x",
                markerfacecolor="white")]
    PS.legend_below(fig, H, ["Qwen3-0.6B", "BGE", "GloVe (control, panel a)"], ncol=3, y=.02)
    _save(fig, name)


def f4_bars(R, name="paper_f4_bars"):
    """Per-quantizer eta and p95 R*. Dense -- recommended for the talk/appendix."""
    groups = [("Qwen3-0.6B 1024D", [k for k, _, _ in QWEN if k in R]),
              ("BGE", [k for k in ["msmarco_bgem3", "nq_bgem3", "quora_bgem3",
                                   "fever_bgem3", "dbpedia_bgel", "dbpedia_bgeb"] if k in R])]
    arms = ARM_ORDER
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.1),
                             gridspec_kw=dict(width_ratios=[4, 6]))
    x = np.arange(len(arms), dtype=float)
    SHADE = dict(color=RED, alpha=.07, zorder=0)
    for col, (title, keys) in enumerate(groups):
        w = .78 / max(len(keys), 1)
        for row, field in enumerate(["eta", "Rstar_p95"]):
            ax = axes[row][col]
            ax.axvspan(.5, 1.5, **SHADE)
            for i, k in enumerate(keys):
                v = np.array([(eta_agg(R[k], R[k]["arms"][a]) if field == "eta"
                               else R[k]["arms"][a][field]) for a in arms], dtype=float)
                g = .25 + .6 * i / max(len(keys) - 1, 1)
                ax.bar(x - .39 + w * (i + .5), v, w * .9, facecolor=str(1 - g * .8),
                       edgecolor=INK, linewidth=.4, zorder=2,
                       label=R[k]["corpus"] + ("" if col == 0 else
                             f" ({R[k]['encoder'].replace('-en-v1.5','')})") if row == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels([ARM_STYLE[a]["label"] for a in arms], rotation=38,
                               ha="right", fontsize=7)
            ax.set_xlim(-.6, len(arms) - .4)
            if row:
                ax.set_yscale("log")
                if col == 0:
                    ax.set_ylabel("p95 $R^*$")
            else:
                ax.set_title(title)
                if col == 0:
                    ax.set_ylabel("$\\eta_Q$")
                ax.legend(fontsize=6.2, ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, name)


if __name__ == "__main__":
    R = load_all()
    os.makedirs(FIGS, exist_ok=True)
    f1_zrank(R)
    f1_zrank(R, keys=QWEN + [("fever_bgem3", "fever", "FEVER (BGE-M3)"),
                             ("glove", "glove", "GloVe")],
             name="paper_f1_zrank_qwen")
    f2_z10(R)
    f3_factorisation(R)
    f4_bars(R)
