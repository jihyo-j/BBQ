r"""The two figures Section 4 ships, at \columnwidth (3.33in) in the draft's style."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import paperstyle as PS
from paperstyle import ARM_STYLE, CFG_STYLE, INK, RED, BLUE, GRAY
PS.use()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from common import WORK, FIGS, CONFIGS, ORDER, K
from tables import load_all
import quantizers as QZ

W = 3.33                                   # ACM column width
SERIES = [("quora_bgem3", "quora", "Quora"), ("dbpedia_ada2", "dbpedia", "DBpedia"),
          ("msmarco_bgem3", "msmarco", "MS MARCO"), ("nq_bgem3", "nq", "NQ"),
          ("fever_bgem3", "fever", "FEVER"), ("glove", "glove", "GloVe")]
LADDER = ["sym_bq", "adc_bq", "adc_bq_c", "adc_bq_cb", "rabitq1", "tq1", "pq", "opq"]


def save(fig, n):
    fig.savefig(f"{FIGS}/{n}.pdf"); fig.savefig(f"{FIGS}/{n}.png"); plt.close(fig)
    print("  ->", n)


def profile(R):
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.62))
    rr = np.arange(1, 101); H, L = [], []
    for key, sty, lab in SERIES:
        s = CFG_STYLE[sty]
        PS.line(ax[0], rr, np.load(f"{WORK}/{key}_perq.npz")["z_curve"], s,
                markevery=[0, 1, 3, 9, 29, 99])
        p = np.array(R[key]["arms"]["adc_bq"]["p95_Rstar_by_rank"])
        PS.line(ax[1], np.arange(1, K + 1), p / np.arange(1, K + 1), s, markevery=[0, 1, 3, 6, 9])
        H.append(Line2D([], [], color=s["color"], marker=s["marker"],
                        markerfacecolor="white", markeredgecolor=s["color"])); L.append(lab)
    ax[0].set_xscale("log"); ax[0].set_xlabel("rank $r$"); ax[0].set_ylabel("$z(r)$")
    ax[0].set_title("(a) Exact ranking")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("rank $r$"); ax[1].set_ylabel("p95 $R^*(r)/r$")
    ax[1].set_title("(b) 1-bit cost")
    fig.tight_layout(pad=.3)
    PS.legend_below(fig, H, L, ncol=3, y=.03)
    save(fig, "profile")


def factorisation(R):
    """Single panel: the candidate price of one bit against beta.  The c_Q
    ladder itself lives in Table 2, so plotting it again would be redundant."""
    learned = [k for k in ORDER if k in R and CONFIGS[k]["group"] != "classic"]
    fig, axs = plt.subplots(figsize=(W, 1.55))
    ax = [None, axs]
    b = np.array([R[k]["beta"] for k in learned])
    ex = np.array([R[k]["arms"]["adc_bq"]["Rstar_p95"] - R[k]["arms"]["opq"]["Rstar_p95"]
                   for k in learned], float)
    m = np.array([k != "dbpedia_ada2" for k in learned])
    sl, ic = np.polyfit(np.log(b[m]), np.log(ex[m]), 1)
    xs = np.linspace(b.min() * .96, b.max() * 1.04, 60)
    ax[1].plot(xs, np.exp(ic) * xs ** sl, color=INK, lw=.7, ls="--", zorder=1)
    for i, k in enumerate(learned):
        qw = CONFIGS[k]["encoder"].startswith("Qwen")
        col = RED if qw else BLUE
        ax[1].plot([b[i]], [ex[i]], marker="o" if qw else "s", color=col,
                   markerfacecolor="white", markeredgecolor=col, ls="none", ms=4, zorder=3)
    i0 = learned.index("dbpedia_ada2")
    ax[1].annotate("ada-002", (b[i0], ex[i0]), fontsize=5.6, color=INK,
                   xytext=(3, -7), textcoords="offset points")
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\beta$")
    ax[1].set_ylabel("extra candidates vs OPQ")
    ax[1].set_title(r"price of one bit $\propto\beta^{%.1f}$" % sl)
    fig.tight_layout(pad=.3)
    H = [Line2D([], [], ls="none", marker="o", color=RED, markerfacecolor="white"),
         Line2D([], [], ls="none", marker="s", color=BLUE, markerfacecolor="white")]
    PS.legend_below(fig, H, ["Qwen3-0.6B", "other encoders"], ncol=2, y=.03)
    save(fig, "factorisation")


if __name__ == "__main__":
    R = load_all(); profile(R); factorisation(R)
