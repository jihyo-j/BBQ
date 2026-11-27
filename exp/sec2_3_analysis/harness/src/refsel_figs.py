"""Figures for the refinement-selection experiment (fixed RaBitQ code, score, pool).

Palette: the bound family is RED, with line style and marker carrying the I/O
structure (solid = sequential, dashed = 2 rounds, dotted = one batch); the
calibrated scalar margin is BLUE; the tuned fixed prefix ORANGE; the per-query
floor GRAY.  Colour never carries information alone.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, FIGS

INK, RED, BLUE, ORANGE, GRAY = "#262626", "#EA3323", "#0000F5", "#D97606", "#8C8C8C"
S = {
 "bound_seq":    dict(color=RED, ls="-", marker="o", ms=2.6,
                      label=r"RaBitQ bound, sequential ($\theta$ per read)"),
 "bound_2phase": dict(color=RED, ls=(0, (4, 1.5)), marker="^", ms=2.8,
                      label="RaBitQ bound, 2 rounds"),
 "bound_margin": dict(color=RED, ls=(0, (1, 1.2)), marker="s", ms=2.4,
                      label="RaBitQ bound as margin, 1 batch"),
 "calib_margin": dict(color=BLUE, ls="-", marker="D", ms=2.6,
                      label="Calibrated scalar margin, 1 batch"),
 "fixedR":       dict(color=ORANGE, ls="-.", marker="v", ms=2.8,
                      label=r"Tuned fixed prefix $R$, 1 batch"),
}
ORDER = ["bound_seq", "bound_2phase", "bound_margin", "calib_margin", "fixedR"]
LAB = {"msmarco_qw06": "MS MARCO / Qwen3-0.6B", "nq_qw06": "NQ / Qwen3-0.6B",
       "quora_bgem3": "Quora / BGE-M3", "fever_bgem3": "FEVER / BGE-M3",
       "msmarco_ada2": "MS MARCO / ada-002 (1536d)"}
KEYS = list(LAB)
NQ = {"nq_qw06": 3452}


def use():
    plt.rcParams.update({
        "figure.dpi": 170, "savefig.dpi": 170, "font.family": "DejaVu Sans",
        "font.size": 7.0, "axes.labelsize": 7.0, "axes.titlesize": 7.0,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
        "axes.linewidth": .7, "axes.grid": False, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "legend.frameon": False, "savefig.bbox": "tight"})


def save(fig, name):
    p = f"{FIGS}/{name}"
    fig.savefig(p + ".pdf"); fig.savefig(p + ".png"); plt.close(fig)
    return p + ".pdf"


def fig1(ef=256):
    """Cost of every rule at a matched preservation target."""
    use()
    fig, ax = plt.subplots(1, 5, figsize=(10.0, 2.0))
    for i, key in enumerate(KEYS):
        c = json.load(open(f"{OUT}/refccurve_{key}_ef{ef}.json"))
        a = ax[i]
        for nm in ORDER:
            p = c["arms"][nm]
            a.plot([x["target"] for x in p], [x["reads"] for x in p],
                   lw=.9, mfc="white", mew=.7, **S[nm])
        a.axhline(c["oracle"], color=GRAY, lw=.8, ls=(0, (1, 1.5)))
        a.annotate("per-query floor", (.03, c["oracle"]),
                   xycoords=("axes fraction", "data"), fontsize=5.4,
                   color=GRAY, va="bottom")
        a.axvline(.99, color=INK, lw=.6, ls=(0, (3, 2)))
        a.set_xlim(.90, 1.0); a.set_xticks([.90, .95, .99])
        a.set_xticklabels(["0.90", "0.95", "0.99"])
        a.set_yscale("log")
        a.set_title(f"({'abcde'[i]}) {LAB[key]}", pad=3)
        if i == 0:
            a.set_ylabel("FP32 vectors refined / query")
    fig.supxlabel(r"target share of queries with $H(q)$ fully refined"
                  "  (each rule calibrated on a held-out half)", y=.0, fontsize=6.8)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(.5, -.07), ncol=5,
               handlelength=2.3, columnspacing=1.1)
    fig.subplots_adjust(wspace=.3)
    return save(fig, f"refsel_f1_cost_ef{ef}")


def fig2(ef=256):
    """(a) how wide the bound is vs the error it covers, (b) what the workload needs,
    (c) the radius-vs-scalar interpolation."""
    use()
    fig, ax = plt.subplots(1, 3, figsize=(8.2, 2.1))
    qs = [.5, .95, .99, .999]
    gauss = [0.6745, 1.9600, 2.5758, 3.2905]
    mk = dict(zip(KEYS, ["o", "s", "D", "^", "v"]))
    for key in KEYS:
        nq = NQ.get(key, 4000)
        t = json.load(open(f"{OUT}/refsel_{key}_ef{ef}_nq{nq}.json"))["tight"]
        ax[0].plot(qs, [t["all"][k] for k in ("p50", "p95", "p99", "p999")],
                   lw=.9, marker=mk[key], ms=2.6, mfc="white", mew=.7,
                   color=BLUE, alpha=.85, label=LAB[key].split(" (")[0])
    ax[0].plot(qs, gauss, color=INK, lw=1.0, ls=(0, (3, 2)), label=r"$|N(0,1)|$")
    ax[0].axhline(1.9, color=RED, lw=.9)
    ax[0].annotate(r"RaBitQ default $\epsilon_0=1.9$", (.5, 1.96), color=RED,
                   fontsize=5.8)
    ax[0].set_xlabel("quantile over pool candidates")
    ax[0].set_ylabel(r"$|\hat d-d|\,/\,B_{\epsilon_0=1}$")
    ax[0].set_title(r"(a) $B$ is one standard deviation, not slack", pad=3)
    ax[0].legend(fontsize=5.4, handlelength=1.6, loc="upper left")
    for key in KEYS:
        nq = NQ.get(key, 4000)
        v = np.load(f"/home/smrc/samsung-nvme/sieun/bbqeta/crit_{key}_ef{ef}_nq{nq}.npz")
        e = np.sort(v["bound_seq"][np.isfinite(v["bound_seq"])])
        ax[1].plot(e, np.arange(1, len(e) + 1) / len(e), lw=1.0, color=BLUE,
                   alpha=.85, marker=mk[key], ms=2.2, markevery=max(1, len(e)//12),
                   mfc="white", mew=.7, label=LAB[key].split(" (")[0])
    ax[1].axvline(1.9, color=RED, lw=.9)
    ax[1].axhline(.99, color=INK, lw=.6, ls=(0, (3, 2)))
    ax[1].set_xlim(0, 4); ax[1].set_ylim(.8, 1.002)
    ax[1].set_xlabel(r"$\epsilon_0$ this query needs to keep all of $H(q)$")
    ax[1].set_ylabel("share of queries covered")
    ax[1].set_title(r"(b) the default $\epsilon_0$ under-covers", pad=3)
    for key in KEYS:
        nq = NQ.get(key, 4000)
        b = json.load(open(f"{OUT}/refbeta_{key}_ef{ef}_nq{nq}.json"))["beta"]
        x = sorted(float(k) for k in b)
        y = [b[str(k) if str(k) in b else f"{k}"]["reads"][0] for k in x]
        y0 = y[x.index(0.0)]
        ax[2].plot(x, [v / y0 for v in y], lw=1.0, color=BLUE, alpha=.85,
                   marker=mk[key], ms=2.4, mfc="white", mew=.7,
                   label=LAB[key].split(" (")[0])
    ax[2].axhline(1.0, color=GRAY, lw=.7, ls=(0, (1, 1.5)))
    ax[2].set_xlabel(r"$\beta$:  0 = one calibrated scalar, 1 = RaBitQ's radius")
    ax[2].set_ylabel(r"reads relative to $\beta=0$")
    ax[2].set_ylim(.9, 1.1)
    ax[2].set_title(r"(c) the per-candidate radius is worth $\leq 3\%$", pad=3)
    ax[2].legend(fontsize=5.4, handlelength=1.6, loc="lower left")
    fig.subplots_adjust(wspace=.33)
    return save(fig, f"refsel_f2_bound_ef{ef}")


def fig3(ef=256):
    """Part 3: reads are not the cost axis, rounds are."""
    use()
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.2))
    cols = {"msmarco_qw06": BLUE, "msmarco_ada2": ORANGE, "fever_bgem3": RED}
    for key, col in cols.items():
        o = json.load(open(f"{OUT}/refio_{key}_ef{ef}.json"))
        r = [v["rounds"] for v in o["arms"].values()]
        l = [v["lat_mean_us"] for v in o["arms"].values()]
        s = [1.0 + .35 * v["reads"] for v in o["arms"].values()]
        ax[0].scatter(r, l, s=s, facecolor="white", edgecolor=col, lw=.8,
                      label=LAB[key].split(" (")[0])
        X = np.array([[1.0, v["rounds"], v["reads"]] for v in o["arms"].values()])
        y = np.array(l)
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        xs = np.logspace(0, np.log10(max(r)), 50)
        ax[0].plot(xs, c[0] + c[1] * xs + c[2] * np.mean([v["reads"]
                   for v in o["arms"].values()]), color=col, lw=.7, alpha=.6)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("I/O rounds per query (marker area $\\propto$ FP32 reads)")
    ax[0].set_ylabel("refinement latency / query ($\\mu$s)")
    ax[0].set_title("(a) latency follows rounds, not reads", pad=3)
    ax[0].legend(fontsize=5.6, handlelength=1.0, loc="upper left")
    o = json.load(open(f"{OUT}/refio_msmarco_qw06_ef{ef}.json"))
    pick = [("RaBitQ bound tuned, sequential", "bound rule\nsequential I/O", RED),
            ("calibrated prefix, sequential reads", "calibrated rule\nsequential I/O", BLUE),
            ("calibrated prefix, one io_uring batch", "calibrated rule\nbatched I/O", BLUE),
            ("bound as margin, one batch", "bound as margin\nbatched I/O", RED),
            ("bound chunked at 16", "bound chunked (16)\nbatched I/O", RED)]
    lab, val, rds, cl = [], [], [], []
    for pre, nm, c in pick:
        v = [x for x in o["arms"].values() if x["note"].startswith(pre)][0]
        lab.append(nm); val.append(v["lat_mean_us"]); rds.append(v["reads"]); cl.append(c)
    x = np.arange(len(lab))
    ax[1].bar(x, val, .62, color=[c for c in cl], alpha=.8, edgecolor=INK, lw=.4)
    for i, (v, r) in enumerate(zip(val, rds)):
        ax[1].annotate(f"{v:.0f}$\\mu$s\n{r:.0f} reads", (i, v), ha="center",
                       va="bottom", fontsize=5.4)
    ax[1].set_yscale("log"); ax[1].set_ylim(100, 9000)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab, fontsize=5.4)
    ax[1].set_ylabel("refinement latency / query ($\\mu$s)")
    ax[1].set_title("(b) MS MARCO / Qwen3-0.6B, $efSearch$=256", pad=3)
    fig.subplots_adjust(wspace=.3)
    return save(fig, f"refsel_f3_io_ef{ef}")


if __name__ == "__main__":
    print(fig1()); print(fig2()); print(fig3())
