r"""One \textwidth figure with the three panels Section 4 needs."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import paperstyle as PS
from paperstyle import CFG_STYLE, INK, RED, BLUE
PS.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from common import WORK, FIGS, CONFIGS, ORDER, K
from tables import load_all

SERIES = [("quora_bgem3", "quora", "Quora"), ("dbpedia_ada2", "dbpedia", "DBpedia"),
          ("msmarco_bgem3", "msmarco", "MS MARCO"), ("nq_bgem3", "nq", "NQ"),
          ("fever_bgem3", "fever", "FEVER"), ("glove", "glove", "GloVe")]


def main():
    R = load_all()
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.72))
    rr = np.arange(1, 101)
    H, L = [], []
    for key, sty, lab in SERIES:
        s = CFG_STYLE[sty]
        PS.line(ax[0], rr, np.load(f"{WORK}/{key}_perq.npz")["z_curve"], s,
                markevery=[0, 1, 3, 9, 29, 99])
        if sty != "glove":                      # GloVe is 489-954 here; it would
            p = np.array(R[key]["arms"]["adc_bq"]["p95_Rstar_by_rank"])
            PS.line(ax[1], np.arange(1, K + 1), p / np.arange(1, K + 1), s,
                    markevery=[0, 1, 3, 6, 9])  # force a log axis and flatten the rest
        H.append(Line2D([], [], color=s["color"], marker=s["marker"],
                        markerfacecolor="white", markeredgecolor=s["color"]))
        L.append(lab)
    ax[0].set_xscale("log"); ax[0].set_xlabel("rank $r$"); ax[0].set_ylabel("$z(r)$")
    ax[0].set_title("(a) Exact ranking")
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 10.5)
    ax[1].set_xlabel("rank $r$"); ax[1].set_ylabel("p95 $R^*(r)/r$")
    ax[1].set_title("(b) 1-bit cost")

    learned = [k for k in ORDER if k in R and CONFIGS[k]["group"] != "classic"]
    b = np.array([R[k]["beta"] for k in learned])
    ex = np.array([R[k]["arms"]["adc_bq"]["Rstar_p95"] - R[k]["arms"]["opq"]["Rstar_p95"]
                   for k in learned], float)
    m = np.array([k != "dbpedia_ada2" for k in learned])
    sl, ic = np.polyfit(np.log(b[m]), np.log(ex[m]), 1)
    xs = np.linspace(b.min() * .96, b.max() * 1.04, 60)
    ax[2].plot(xs, np.exp(ic) * xs ** sl, color=INK, lw=.7, ls="--", zorder=1)
    for i, k in enumerate(learned):
        qw = CONFIGS[k]["encoder"].startswith("Qwen")
        c = RED if qw else BLUE
        ax[2].plot([b[i]], [ex[i]], marker="o" if qw else "s", color=c,
                   markerfacecolor="white", markeredgecolor=c, ls="none", ms=4, zorder=3)
    i0 = learned.index("dbpedia_ada2")
    ax[2].annotate("ada-002", (b[i0], ex[i0]), fontsize=5.8, color=INK,
                   xytext=(3, -8), textcoords="offset points")
    ax[2].set_yscale("log"); ax[2].set_xlabel(r"$\beta$")
    ax[2].set_ylabel("extra candidates vs OPQ")
    ax[2].set_title(r"(c) price of one bit")
    H += [Line2D([], [], ls="none", marker="o", color=RED, markerfacecolor="white"),
          Line2D([], [], ls="none", marker="s", color=BLUE, markerfacecolor="white"),
          Line2D([], [], ls="--", color=INK)]
    L += ["(c) Qwen3-0.6B", "(c) other enc.", r"(c) fit $\beta^{%.1f}$" % sl]
    fig.tight_layout(pad=.35)
    PS.legend_below(fig, H, L, ncol=6, y=.03)
    for e in ("pdf", "png"):
        fig.savefig(f"{FIGS}/profile3.{e}")
    plt.close(fig)
    print("profile3 written; slope", round(sl, 2))


if __name__ == "__main__":
    main()
