#!/usr/bin/env python3
"""Replace Figs 10/21 panels with the BEIR text-embedding-3-small / BGE-large measurements.

argv[1] = beir_recall_qps_t1.csv  (written by exp/scripts/collect_beir_recall_qps.py)

A panel is replaced only when all 12 methods have measurements for it; otherwise it is left unchanged and
reported, so a half-measured panel never mixes the old and new datasets. For each replaced panel:
  * every method's curve/marker blocks in recall-qps_main.dat and recall-qps_md.dat are rebuilt from the
    measured points (curve = all points in search-parameter order; markers = 5 evenly spaced points),
    matching update_recall-qps_main_tq_pq_opq.py;
  * the panel's plot command in recall-qps_main_panels.gp / recall-qps_md_panels.gp gains any method that
    now has data (e.g. RaBitQ-7bit, which the old data never brought into range), in the existing style;
  * the panel's y range is widened if the new points fall outside it (x range is fixed by the figure).
"""
import csv
import math
import re
import sys
from collections import defaultdict

SRC = sys.argv[1]
LETTER = {"msmarco_openai1536": "i", "dbpedia_openai1536": "m", "dbpedia_bge1024": "n"}
METHODS = ["adcbq", "esbbq", "tq", "rbq7", "rbqssd", "hnswrbq", "dannrbq", "pipeann", "spann", "dannpq", "fpq", "fopq"]
LS = {"adcbq": 1, "esbbq": 2, "tq": 3, "pipeann": 4, "rbq7": 5, "rbqssd": 6, "hnswrbq": 7, "dannrbq": 8,
      "spann": 9, "dannpq": 10, "fpq": 11, "fopq": 12}
CHAR = {"rbqssd": "CH_RTRI", "fopq": "CH_HEX"}

pts = defaultdict(list)
for r in csv.DictReader(open(SRC)):
    try:
        key = float(r["search_parameter_value"])
    except ValueError:
        key = 0.0
    pts[(r["dataset"], r["figure_method"])].append((key, float(r["recall_at_10"]), float(r["qps"])))

panels = {}
for ds, p in LETTER.items():
    have = [m for m in METHODS if pts.get((ds, m))]
    if len(have) == len(METHODS):
        panels[p] = ds
    else:
        print(f"panel ({p}) {ds}: kept unchanged, missing {sorted(set(METHODS) - set(have))}")


def blocks_for(ds, m):
    xs = sorted(pts[(ds, m)])
    curve = [f"{r:.6f} {q:.4f}" for _, r, q in xs]
    n = len(xs)
    idx = sorted({round(i * (n - 1) / 4) for i in range(5)}) if n > 5 else range(n)
    return curve, [curve[i] for i in idx]


def rewrite_dat(path):
    txt = open(path).read()
    blocks = {}
    order = []
    for chunk in re.split(r"(?m)^# (?=\S+_[a-p]_(?:pts|curve)\s*$)", txt)[1:]:
        name, _, body = chunk.partition("\n")
        blocks[name.strip()] = body.strip("\n")
        order.append(name.strip())
    header = re.split(r"(?m)^# (?=\S+_[a-p]_(?:pts|curve)\s*$)", txt)[0]
    for p, ds in panels.items():
        for m in METHODS:
            curve, marks = blocks_for(ds, m)
            for suf, body in (("pts", marks), ("curve", curve)):
                k = f"{m}_{p}_{suf}"
                if k not in blocks:
                    order.append(k)
                blocks[k] = "\n".join(body)
    note = "# panels %s: BEIR text-embedding-3-small / BGE-large measurements from %s (update_beir_panels.py)\n" % (
        ", ".join(f"({p})" for p in sorted(panels)), SRC.split("/")[-1])
    header = header.rstrip("\n") + "\n" + (note if note not in header else "")
    open(path, "w").write(header + "".join(f"\n\n# {k}\n{blocks[k]}\n" for k in order))


def rewrite_panels(path):
    txt = open(path).read()
    parts = re.split(r"(?m)^(?=# \([a-p]\)\n)", txt)
    for i, part in enumerate(parts):
        mo = re.match(r"# \(([a-p])\)\n", part)
        if not mo or mo.group(1) not in panels:
            continue
        p, ds = mo.group(1), panels[mo.group(1)]
        # add methods that now have data
        missing = [m for m in METHODS if f'"{m}_{p}_curve"' not in part]
        if missing:
            lines = part.split("\n")
            last = max(j for j, l in enumerate(lines) if f"_{p}_" in l)
            extra = []
            for m in missing:
                extra.append(f'     D index "{m}_{p}_curve" u 1:2 w l ls {LS[m]} notitle')
                if m in CHAR:
                    extra.append(f'     D index "{m}_{p}_pts" u 1:2:({CHAR[m]}) w labels center font ",".sprintf("%g",CHFS) tc ls {LS[m]} notitle')
                else:
                    extra.append(f'     D index "{m}_{p}_pts" u 1:2 w p ls {LS[m]} lw MLW notitle')
            lines[last] = lines[last].rstrip() + ", \\"
            lines[last + 1:last + 1] = [l + ", \\" for l in extra[:-1]] + [extra[-1]]
            part = "\n".join(lines)
        # widen y range (log axis) to the new data within the x range
        xr = re.search(r"set xrange \[([\d.]+):([\d.]+)\]", part)
        yr = re.search(r"set yrange \[([\d.]+):([\d.]+)\]", part)
        xlo = float(xr.group(1))
        qs = [q for m in METHODS for _, r, q in pts[(ds, m)] if r >= xlo]
        lo, hi = float(yr.group(1)), float(yr.group(2))
        nlo, nhi = min(lo, 10 ** (math.floor(math.log10(min(qs)) * 10) / 10)), max(hi, max(qs) * 1.25)
        if (nlo, nhi) != (lo, hi):
            part = part.replace(yr.group(0), f"set yrange [{nlo:.5g}:{nhi:.5g}]")
        parts[i] = part
    open(path, "w").write("".join(parts))


for f in ("recall-qps_main.dat", "recall-qps_md.dat"):
    rewrite_dat(f)
for f in ("recall-qps_main_panels.gp", "recall-qps_md_panels.gp"):
    rewrite_panels(f)
print("replaced panels:", ", ".join(f"({p}) {d}" for p, d in sorted(panels.items())) or "none")
