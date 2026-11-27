"""Figures. Lab palette: INK #262626 for all text; BLUE #0000F5 highlights the
paper's own scorer (ADC-BQ). RED and GREEN never carry meaning on their own --
every series also has a distinct marker/hatch and a direct label.
All figure text is English (no Korean font on this box)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ORDER, OUT, FIGS, WORK, CONFIGS, K
import quantizers as QZ
from tables import load_all, eta_agg

INK = "#262626"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140, "font.size": 8,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.edgecolor": INK, "axes.linewidth": .8,
    "legend.frameon": False, "axes.grid": True, "grid.alpha": .25,
    "grid.linewidth": .5, "font.family": "DejaVu Sans",
})

ARM_C = {"sym_bq": "#9E9E9E", "adc_bq": "#0000F5", "pq": "#2E7D32",
         "opq": "#7B2CBF", "sq4": "#D97706", "sq8": "#8C5A00",
         "rabitq1": "#EA3323", "tq1": "#00838F"}
ARM_H = {"sym_bq": "", "adc_bq": "", "pq": "//", "opq": "\\\\",
         "sq4": "..", "sq8": "xx", "rabitq1": "++", "tq1": "--"}
ARM_M = {"sym_bq": "s", "adc_bq": "o", "pq": "^", "opq": "v",
         "sq4": "D", "sq8": "P", "rabitq1": "X", "tq1": "*"}

CFG_C = ["#0000F5", "#EA3323", "#2E7D32", "#D97706", "#7B2CBF",
         "#00838F", "#8C5A00", "#C2185B", "#455A64", "#9E9E9E",
         "#5D4037", "#1A237E"]
CFG_M = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "p"]

QWEN = ["msmarco_qw06", "nq_qw06", "quora_qw06", "dbpedia_qw06"]
OTHER = ["msmarco_bgem3", "nq_bgem3", "quora_bgem3", "fever_bgem3",
         "dbpedia_bgel", "dbpedia_bgeb", "glove", "gist"]


def short(r):
    return r["corpus"] if r["encoder"] == "-" else f"{r['corpus']}/{r['encoder'].replace('-en-v1.5','')}"


# ------------------------------------------------------------------ Fig 5
def fig_zrank(R):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    rr = np.arange(1, 101)
    for i, k in enumerate([x for x in ORDER if x in R]):
        d = np.load(f"{WORK}/{k}_perq.npz")
        z = d["z_curve"]
        ax[0].plot(rr, z, color=CFG_C[i], marker=CFG_M[i], ms=2.5, lw=1.1,
                   markevery=[0, 4, 14, 39, 99], label=short(R[k]))
        p95 = np.array(R[k]["arms"]["adc_bq"]["p95_Rstar_by_rank"])
        ax[1].plot(np.arange(1, K + 1), p95 / np.arange(1, K + 1),
                   color=CFG_C[i], marker=CFG_M[i], ms=2.5, lw=1.1)
    ax[0].set_xscale("log"); ax[0].set_xlabel("exact rank $r$")
    ax[0].set_ylabel("$z(r)$"); ax[0].set_title("(a) standardised cosine by exact rank", fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("exact rank $r$"); ax[1].set_ylabel("p95 $R^*(r)\\,/\\,r$")
    ax[1].set_title("(b) 1-bit cost (ADC-BQ)", fontsize=8)
    ax[0].legend(fontsize=5.6, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig_zrank.pdf"); fig.savefig(f"{FIGS}/fig_zrank.png")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 6
def fig_z10_rstar(R):
    from scipy.stats import spearmanr
    keys = [k for k in ORDER if k in R and CONFIGS[k]["group"] != "classic"]
    fig, ax = plt.subplots(figsize=(4.0, 3.1))
    Z, S = [], []
    for i, k in enumerate(keys):
        d = np.load(f"{WORK}/{k}_perq.npz")
        z10, rs = d["z10"], d["Rstar_adc_bq"].astype(float)
        ax.scatter(z10, rs, s=1.6, alpha=.18, color=CFG_C[i], linewidths=0, rasterized=True)
        rho = spearmanr(z10, rs).statistic
        ax.scatter([], [], s=14, color=CFG_C[i], marker=CFG_M[i],
                   label=f"{short(R[k])} {rho:+.2f}")
        Z.append(z10); S.append(rs)
    Z, S = np.concatenate(Z), np.concatenate(S)
    pooled = spearmanr(Z, S).statistic
    ax.set_yscale("log"); ax.set_xlabel("$z_{10}$"); ax.set_ylabel("$R^*$ (ADC-BQ)")
    ax.set_title(f"{len(Z):,} queries, pooled $\\rho$ = {pooled:+.2f}", fontsize=8)
    ax.legend(fontsize=5.6, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig_z10_rstar.pdf"); fig.savefig(f"{FIGS}/fig_z10_rstar.png")
    plt.close(fig)
    return pooled


# --------------------------------------------------- NEW: per-quantizer bars
def fig_eta_rstar_bars(R):
    """The requested chart: for every quantizer, how large is eta and how large is
    the candidate budget it buys.  Three column groups so that the conventional
    descriptors (eta 1.5-3.5) do not flatten the learned ones (eta 0.1-1.1)."""
    BGE = ["msmarco_bgem3", "nq_bgem3", "quora_bgem3", "fever_bgem3",
           "dbpedia_bgel", "dbpedia_bgeb"]
    CLS = ["glove", "gist"]
    groups = [("Qwen3-0.6B 1024D -- corpus axis", [k for k in QWEN if k in R], 0),
              ("BGE -- encoder axis", [k for k in BGE if k in R], 4),
              ("conventional", [k for k in CLS if k in R], 10)]
    arms = list(QZ.ARMS)
    x = np.arange(len(arms), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.2),
                             gridspec_kw=dict(width_ratios=[len(g[1]) + 1.5 for g in groups]))
    for col, (title, keys, c0) in enumerate(groups):
        w = .8 / max(len(keys), 1)
        for row, field in enumerate(["eta", "Rstar_p95"]):
            ax = axes[row][col]
            ax.axvspan(.5, 1.5, color="#0000F5", alpha=.06, zorder=0)
            for i, k in enumerate(keys):
                v = np.array([(eta_agg(R[k], R[k]["arms"][a]) if field == "eta"
                               else R[k]["arms"][a][field]) if a in R[k]["arms"] else np.nan
                              for a in arms], dtype=float)
                pos = x - .4 + w * (i + .5)
                ax.bar(pos, np.where(np.isfinite(v), v, 0.0), w * .9,
                       color=CFG_C[c0 + i], edgecolor=INK, linewidth=.35,
                       label=short(R[k]) if row == 0 else None, zorder=2)
                for xx, raw in zip(pos, v):
                    if not np.isfinite(raw):
                        ax.text(xx, 0, "n/a", rotation=90, fontsize=5.2,
                                va="bottom", ha="center", color=INK)
            ax.set_xticks(x)
            ax.set_xticklabels([QZ.NICE[a] for a in arms], rotation=40, ha="right", fontsize=6.3)
            ax.set_xlim(-.6, len(arms) - .4)
            if row == 1:
                ax.set_yscale("log")
                if col == 0:
                    ax.set_ylabel("p95 $R^*$  (candidates to hold the exact top-10)")
            else:
                ax.set_title(title, fontsize=8.5)
                if col == 0:
                    ax.set_ylabel("$\\eta_Q=\\nu_Q/(\\kappa_Q\\beta)$")
            ax.legend(fontsize=6, ncol=1, loc="upper right") if row == 0 else None
    # share only the R* row, so the 100x gap to the conventional corpora is visible;
    # the eta row keeps per-panel scales.
    lo = min(a.get_ylim()[0] for a in axes[1])
    hi = max(a.get_ylim()[1] for a in axes[1])
    for col, a in enumerate(axes[1]):
        a.set_ylim(max(lo, 5), hi)
        if col:
            a.set_yticklabels([])
    fig.suptitle("Per-quantizer residual-to-decay ratio $\\eta$ (top) and the candidate "
                 "budget it predicts (bottom).  Shaded = ADC-BQ, the scorer BBQ ships.  "
                 "n/a = sign(x) is constant, $\\kappa$ undefined.", fontsize=8.5)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(f"{FIGS}/fig_eta_rstar_bars.pdf"); fig.savefig(f"{FIGS}/fig_eta_rstar_bars.png")
    plt.close(fig)


def fig_eta_vs_rstar(R):
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    E, P = [], []
    for i, k in enumerate([x for x in ORDER if x in R]):
        for a in QZ.ARMS:
            if a not in R[k]["arms"]:
                continue
            e, p = eta_agg(R[k], R[k]["arms"][a]), R[k]["arms"][a]["Rstar_p95"]
            if not (np.isfinite(e) and np.isfinite(p)):
                continue
            ax.scatter(e, p, s=20, color=ARM_C[a], marker=ARM_M[a],
                       edgecolors=INK, linewidths=.3, alpha=.85)
            E.append(e); P.append(p)
    for a in QZ.ARMS:
        ax.scatter([], [], s=20, color=ARM_C[a], marker=ARM_M[a],
                   edgecolors=INK, linewidths=.3, label=QZ.NICE[a])
    rho = spearmanr(E, P).statistic
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$\\eta_Q$"); ax.set_ylabel("p95 $R^*$")
    ax.set_title(f"all (config x quantizer) pairs, n={len(E)}, $\\rho$={rho:+.3f}", fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig_eta_vs_rstar.pdf")
    fig.savefig(f"{FIGS}/fig_eta_vs_rstar.png")
    plt.close(fig)
    return rho


def fig_candrec(R):
    """Fig 2 style: CandidateRecall@R per quantizer, on two representative configs."""
    keys = [k for k in ("msmarco_qw06", "fever_bgem3") if k in R]
    fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 2.9), squeeze=False)
    for j, k in enumerate(keys):
        a_ = ax[0][j]
        for arm in QZ.ARMS:
            if arm not in R[k]["arms"]:
                continue
            cr = R[k]["arms"][arm]["candrec"]
            g = sorted(int(x) for x in cr)
            g = [x for x in g if x <= 1000]
            a_.plot(g, [cr[str(x)] for x in g], color=ARM_C[arm], marker=ARM_M[arm],
                    ms=3, lw=1.1, label=QZ.NICE[arm])
        a_.set_xscale("log"); a_.set_xlabel("candidate budget $R$")
        a_.set_ylabel("CandidateRecall@10" if j == 0 else "")
        a_.set_title(short(R[k]), fontsize=8)
        a_.set_ylim(.6, 1.005)
        if j == 0:
            a_.legend(fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{FIGS}/fig_candrec.pdf"); fig.savefig(f"{FIGS}/fig_candrec.png")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    R = load_all()
    print("configs:", list(R))
    fig_zrank(R)
    print("pooled rho(z10, R*) =", round(fig_z10_rstar(R), 4))
    fig_eta_rstar_bars(R)
    print("pooled rho(eta, p95R*) =", round(fig_eta_vs_rstar(R), 4))
    fig_candrec(R)
    print("figures ->", FIGS)


def fig_separability(R):
    """eta = c_Q / beta : the quantizer contributes a constant factor, the
    embedding contributes beta.  (a) nu/kappa normalised by each config's ADC-BQ
    value -- all configs collapse onto one curve.  (b) beta carries the spread."""
    arms = [a for a in QZ.ARMS if a != "sq8"]
    learned = [k for k in ORDER if k in R and CONFIGS[k]["group"] != "classic"]
    cls = [k for k in ORDER if k in R and CONFIGS[k]["group"] == "classic"]
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
    x = np.arange(len(arms))
    ib = arms.index("adc_bq")
    for i, k in enumerate(learned + cls):
        v = np.array([R[k]["arms"][a]["nu"] / R[k]["arms"][a]["kappa"] for a in arms])
        v = v / v[ib]
        dashed = CONFIGS[k]["group"] == "classic"
        ax[0].plot(x, v, color=CFG_C[i], marker=CFG_M[i], ms=3.2, lw=1.0,
                   ls="--" if dashed else "-", alpha=.9, label=short(R[k]))
    ax[0].axhline(1.0, color=INK, lw=.6, ls=":")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([QZ.NICE[a] for a in arms], rotation=40, ha="right", fontsize=6.3)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("$(\\nu_Q/\\kappa_Q)$ relative to ADC-BQ")
    ax[0].set_title("(a) the quantizer factor is a constant\n"
                    "(learned: 97.9% of variance, interaction 0.2%)", fontsize=7.5)
    ax[0].legend(fontsize=5.4, ncol=2, loc="lower left")

    keys = learned + cls
    b = [R[k]["beta"] for k in keys]
    ax[1].bar(np.arange(len(keys)), b,
              color=[CFG_C[i] for i in range(len(keys))], edgecolor=INK, linewidth=.4)
    ax[1].set_xticks(np.arange(len(keys)))
    ax[1].set_xticklabels([short(R[k]) for k in keys], rotation=55, ha="right", fontsize=6)
    ax[1].set_ylabel("$\\beta$  (top-rank score decay)")
    ax[1].set_title("(b) the embedding factor carries the spread\n"
                    "$\\beta$ 0.490 (FEVER) to 0.896 (Quora) = 1.83x", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_separability.pdf"); fig.savefig(f"{FIGS}/fig_separability.png")
    plt.close(fig)


def fig_bq_price(R):
    """What BQ's higher noise constant actually costs, against beta."""
    from scipy.stats import spearmanr
    learned = [k for k in ORDER if k in R and CONFIGS[k]["group"] != "classic"]
    b = np.array([R[k]["beta"] for k in learned])
    ex = np.array([R[k]["arms"]["adc_bq"]["Rstar_p95"] - R[k]["arms"]["opq"]["Rstar_p95"]
                   for k in learned], dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for i, k in enumerate(learned):
        qwen = CONFIGS[k]["encoder"].startswith("Qwen")
        ax.scatter(b[i], ex[i], s=46, color="#0000F5" if qwen else "#EA3323",
                   marker="o" if qwen else "s", edgecolors=INK, linewidths=.4, zorder=3)
        ax.annotate(short(R[k]), (b[i], ex[i]), fontsize=5.6, color=INK,
                    xytext=(4, 3), textcoords="offset points")
    sl, ic = np.polyfit(np.log(b), np.log(ex), 1)
    xs = np.linspace(b.min() * .97, b.max() * 1.03, 50)
    ax.plot(xs, np.exp(ic) * xs ** sl, color=INK, lw=.9, ls="--", zorder=1,
            label=f"$\\propto\\beta^{{{sl:.1f}}}$")
    ax.scatter([], [], s=40, color="#0000F5", marker="o", edgecolors=INK,
               linewidths=.4, label="Qwen3-0.6B")
    ax.scatter([], [], s=40, color="#EA3323", marker="s", edgecolors=INK,
               linewidths=.4, label="BGE")
    ax.set_yscale("log")
    ax.set_xlabel("$\\beta$  (how fast the exact scores decay)")
    ax.set_ylabel("extra candidates ADC-BQ pays over OPQ\n(p95 $R^*$ difference, same 128 B)")
    ax.set_title(f"$\\rho$ = {spearmanr(b, ex).statistic:+.2f}  --  the price of 1 bit "
                 f"collapses\nas the embedding separates its top ranks", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/fig_bq_price.pdf"); fig.savefig(f"{FIGS}/fig_bq_price.png")
    plt.close(fig)
