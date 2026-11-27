"""Match the figure style of the BBQ submission (gnuplot look).

Conventions read off Fig. 2/3/5/10 of the draft:
  * colour = family, marker = method  (lab rule: never colour alone)
  * open markers on a white face, thin lines, no grid, box on all four sides,
    ticks pointing in, legend in a horizontal block under the figure
  * panel captions as "(a) ..." titles above each axes
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#262626"
RED = "#CC2B5E"      # BQ family / Quora
BLUE = "#2B4CC0"     # RaBitQ family / DBpedia
ORANGE = "#E09B2D"   # SQ family / MS MARCO
PURPLE = "#8B5CC7"   # TurboQuant / NQ
GREEN = "#2E9E4F"    # PQ, OPQ / FEVER
GRAY = "#8C8C8C"     # conventional

# quantizers: exactly the colour/marker split the draft's Fig. 2-3 legend uses
ARM_STYLE = {
    "sym_bq":  dict(color=RED,    marker="o", label="SymBQ"),
    "adc_bq":  dict(color=RED,    marker="s", label="ADC-BQ"),
    "adc_bq_c":  dict(color=RED,  marker="^", label="ADC-BQ +centre"),
    "adc_bq_cb": dict(color=RED,  marker="v", label="ADC-BQ +centre+scale"),
    "pq":      dict(color=GREEN,  marker="o", label="PQ"),
    "opq":     dict(color=GREEN,  marker="v", label="OPQ"),
    "tq1":     dict(color=PURPLE, marker="o", label="TurboQuant 1bit"),
    "rabitq1": dict(color=BLUE,   marker="^", label="RaBitQ 1bit"),
    "sq4":     dict(color=ORANGE, marker="o", label="SQ 4bit"),
    "sq8":     dict(color=ORANGE, marker="s", label="SQ 8bit"),
}
ARM_ORDER = ["sym_bq", "adc_bq", "adc_bq_c", "adc_bq_cb", "pq", "opq",
             "tq1", "rabitq1", "sq4", "sq8"]

# configurations: same colour/marker roles as the draft's Fig. 5 legend
CFG_STYLE = {
    "quora":   dict(color=RED,    marker="o"),
    "dbpedia": dict(color=BLUE,   marker="s"),
    "msmarco": dict(color=ORANGE, marker="D"),
    "nq":      dict(color=PURPLE, marker="v"),
    "fever":   dict(color=GREEN,  marker="^"),
    "glove":   dict(color=GRAY,   marker="x"),
    "gist":    dict(color=GRAY,   marker="+"),
}


def use():
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 160,
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.labelsize": 7.0, "axes.titlesize": 7.0,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
        "axes.linewidth": .7, "axes.grid": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": .7, "ytick.major.width": .7,
        "lines.linewidth": .9, "lines.markersize": 3.4,
        "lines.markeredgewidth": .8, "lines.markerfacecolor": "white",
        "legend.frameon": False, "legend.handlelength": 2.2,
        "legend.columnspacing": 1.3, "legend.handletextpad": .5,
        "savefig.bbox": "tight", "savefig.pad_inches": .02,
    })


def line(ax, x, y, style, **kw):
    """A series in the draft's style: coloured thin line, open marker."""
    kw.setdefault("color", style["color"])
    kw.setdefault("marker", style["marker"])
    kw.setdefault("markerfacecolor", "white")
    kw.setdefault("markeredgecolor", style["color"])
    return ax.plot(x, y, **kw)


def legend_below(fig, handles, labels, ncol=3, y=0.0):
    """One horizontal legend block under the whole figure, as in Fig. 5 / Fig. 10."""
    return fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, y),
                      ncol=ncol, frameon=False)
