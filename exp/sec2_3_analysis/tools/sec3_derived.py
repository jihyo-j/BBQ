#!/usr/bin/env python3
"""Sec 3 numbers that the harness prints but does not store, recomputed with the harness's own code.

    sec3_derived.py [harness_dir] [results/sec3_derived.csv]

Reads only harness/out/*.json and harness/work/*_perq.npz (Sieun Kim's BBQETA outputs, copied verbatim).
Writes rows (source, key, value) in the same schema as sec3_values.csv:

* eta.spearman.*   Spearman(eta, p95 R*) from src/tables.py diagnostics(): eta_agg() over every
                   (configuration, quantizer) pair in out/<key>.json (pooled), and over ADC-BQ alone;
                   eta.paper.* the same over the configurations without ada-002;
* perq.<key>.*     median / p95 / p99 of the per-query R* (exact top-10, ADC-BQ) in work/<key>_perq.npz,
                   with numpy's default (linear) quantile, as src/export_perq.py prints them;
* levels.*         src/levels.py variance decomposition of log R* (its own 7-setting design, and
                   levels.paper.* the same design without its ada-002 setting).
"""
import csv, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HARNESS = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "harness")
OUTCSV = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "results/sec3_derived.csv")
sys.path.insert(0, str(HARNESS / "src"))
import common, quantizers as QZ, tables, levels  # noqa: E402  (harness code, unmodified)

levels.WORK = str(HARNESS / "work")
R = {k: json.load(open(HARNESS / "out" / f"{k}.json")) for k in common.ORDER if (HARNESS / "out" / f"{k}.json").exists()}
rows = []
add = lambda src, key, v: rows.append((src, key, repr(float(v))))

# eta diagnostic, exactly as tables.diagnostics() (i) and (iii).
# Two scopes: "eta.*" over every configuration of the sweep (20, including the two ada-002 ones), and "eta.paper.*" over
# the 18 configurations without ada-002, the set on which the manuscript's Sec 3 statistics reproduce.
ADA = ("msmarco_ada2", "dbpedia_ada2")


def eta_stats(tag, ks):
    E = np.array([tables.eta_agg(R[k], R[k]["arms"]["adc_bq"]) for k in ks])
    P = np.array([R[k]["arms"]["adc_bq"]["Rstar_p95"] for k in ks])
    m = np.isfinite(E)
    add("tables.py", f"{tag}.spearman.adc_bq", spearmanr(E[m], P[m]).statistic)
    add("tables.py", f"{tag}.n.adc_bq", m.sum())
    E, P = [], []
    for k in ks:
        for a in QZ.ARMS:
            if a in R[k]["arms"]:
                E.append(tables.eta_agg(R[k], R[k]["arms"][a])); P.append(R[k]["arms"][a]["Rstar_p95"])
    E, P = np.array(E), np.array(P)
    m = np.isfinite(E) & np.isfinite(P)
    add("tables.py", f"{tag}.spearman.pooled", spearmanr(E[m], P[m]).statistic)
    add("tables.py", f"{tag}.n.pooled", m.sum())
    add("tables.py", f"{tag}.n.pairs_total", len(E))
    add("tables.py", f"{tag}.n.configurations", len(ks))


ks = [k for k in common.ORDER if k in R and "adc_bq" in R[k]["arms"]]
eta_stats("eta", ks)
eta_stats("eta.paper", [k for k in ks if k not in ADA])

# per-query R* quantiles
for f in sorted((HARNESS / "work").glob("*_perq.npz")):
    key = f.name[: -len("_perq.npz")]
    z = np.load(f)
    for arm in [n[6:] for n in z.files if n.startswith("Rstar_")]:
        r = z[f"Rstar_{arm}"].astype(float)
        for q, tag in ((50, "median"), (95, "p95"), (99, "p99")):
            add(f.name, f"perq.{key}.{arm}.{tag}", np.percentile(r, q))
        add(f.name, f"perq.{key}.{arm}.n", len(r))

# variance decomposition: levels.py's own design (its 7 settings, one of them ada-002), and the same design without
# the ada-002 setting, which reproduces the manuscript's shares
for tag, keys in (("levels.7settings", levels.TAB), ("levels.paper", [k for k in levels.TAB if k not in ADA])):
    d = levels.decompose(keys, levels.DEPLOYABLE)
    for t in ("query", "setting", "arm", "interaction"):
        add("levels.py", f"{tag}.{t}", d[t])

OUTCSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUTCSV, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["source", "key", "value"]); w.writerows(rows)
print(f"{len(rows)} values -> {OUTCSV}")
