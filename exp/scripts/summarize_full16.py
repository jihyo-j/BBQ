#!/usr/bin/env python3
"""Summarise the 16-configuration x 11-method Recall@10-QPS table into the quantities the paper quotes.

    summarize_full16.py recall_qps_all_latest_t1.csv full16_summary_t1.csv [--t1 <t1 csv for cross-thread ratios>]

Rows written (kind / dataset / method / threshold / metric / value / search_parameter_value):
  best        best QPS among points with Recall@10 >= threshold (0.95, 0.99, 0.995) for every (dataset, method);
              also carries that point's search parameter, recall and FP32 reads per query
  max_recall  maximum Recall@10 reached by each (dataset, method)
  agg         corpus-wide aggregates used in Secs. 6.2 / 6.6 and Appendix B (definitions inline below)

Every aggregate is computed from the input table only; nothing is typed in.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict

BBQ, TQ, FP, R7, RSSD = ("BBQ (calibrated ADC-BQ)", "TurboQuant TQ1+", "FP32 HNSW RAM", "RaBitQ 7-bit (RAM)",
                         "RaBitQ-SSD B9")
PQ, OPQ = "Faiss PQ d/8", "Faiss OPQ-PQ d/8"
SSD_GRAPH = ("DiskANN-PQ d/8", "DiskANN-RaBitQ 1-bit", "PipeANN-RaBitQ 1-bit")
THRESHOLDS = (0.95, 0.99, 0.995)


def load(path):
    d = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        d[(r["dataset"], r["method"])].append(r)
    return d


def best(d, ds, m, th):
    c = [r for r in d.get((ds, m), []) if float(r["recall_at_10"]) >= th]
    return max(c, key=lambda r: float(r["qps"])) if c else None


def max_recall(d, ds, m):
    return max((float(r["recall_at_10"]) for r in d.get((ds, m), [])), default=None)


def main() -> None:
    src, out = sys.argv[1], sys.argv[2]
    t1 = load(sys.argv[sys.argv.index("--t1") + 1]) if "--t1" in sys.argv else None
    d = load(src)
    datasets = sorted({k[0] for k in d})
    methods = sorted({k[1] for k in d})
    rows = []

    def agg(metric, value, note=""):
        rows.append(dict(kind="agg", dataset="", method="", threshold="", metric=metric,
                         value=f"{value:.6g}" if isinstance(value, float) else str(value), note=note))

    for ds in datasets:
        for m in methods:
            for th in THRESHOLDS:
                b = best(d, ds, m, th)
                if b:
                    rows.append(dict(kind="best", dataset=ds, method=m, threshold=str(th), metric="qps", value=b["qps"],
                                     search_parameter_value=b["search_parameter_value"], recall_at_10=b["recall_at_10"],
                                     fp_vectors_per_query=b.get("fp_vectors_per_query", "")))
            mr = max_recall(d, ds, m)
            if mr is not None:
                rows.append(dict(kind="max_recall", dataset=ds, method=m, metric="recall_at_10", value=f"{mr:.6g}"))

    def q(r):
        return float(r["qps"])

    # --- BBQ vs TurboQuant-1bit
    both99 = [ds for ds in datasets if best(d, ds, BBQ, .99) and best(d, ds, TQ, .99)]
    agg("bbq_faster_than_tq_at_0.99", sum(q(best(d, ds, BBQ, .99)) > q(best(d, ds, TQ, .99)) for ds in both99),
        "configurations where both reach 0.99 and BBQ has the higher best QPS")
    agg("both_reach_0.99_bbq_tq", len(both99))
    agg("bbq_faster_than_tq_at_0.99_or_tq_misses",
        sum(1 for ds in datasets if best(d, ds, BBQ, .99) and (not best(d, ds, TQ, .99)
                                                               or q(best(d, ds, BBQ, .99)) > q(best(d, ds, TQ, .99)))))
    both995 = [ds for ds in datasets if best(d, ds, BBQ, .995) and best(d, ds, TQ, .995)]
    agg("both_reach_0.995_bbq_tq", len(both995))
    agg("bbq_faster_than_tq_at_0.995", sum(q(best(d, ds, BBQ, .995)) > q(best(d, ds, TQ, .995)) for ds in both995))
    reads = [(float(best(d, ds, BBQ, .99)["fp_vectors_per_query"]), float(best(d, ds, TQ, .99)["fp_vectors_per_query"]))
             for ds in both99 if best(d, ds, BBQ, .99).get("fp_vectors_per_query")
             and best(d, ds, TQ, .99).get("fp_vectors_per_query")]
    if reads:
        agg("fp_reads_recorded_configs_bbq_tq", len(reads))
        agg("mean_fp_reads_bbq_at_0.99", sum(a for a, _ in reads) / len(reads))
        agg("mean_fp_reads_tq_at_0.99", sum(b for _, b in reads) / len(reads))

    # --- BBQ vs in-memory FP32 HNSW
    both_fp = [ds for ds in datasets if best(d, ds, BBQ, .99) and best(d, ds, FP, .99)]
    ratios = {ds: q(best(d, ds, BBQ, .99)) / q(best(d, ds, FP, .99)) for ds in both_fp}
    agg("bbq_faster_than_fp_hnsw_at_0.99", sum(v > 1 for v in ratios.values()))
    wins = [v for v in ratios.values() if v > 1]
    if wins:
        agg("bbq_over_fp_hnsw_min_speedup_at_0.99", min(wins))
        agg("bbq_over_fp_hnsw_max_speedup_at_0.99", max(wins))
    gaps = [(max_recall(d, ds, FP) - max_recall(d, ds, BBQ)) * 100 for ds in datasets
            if max_recall(d, ds, FP) is not None and max_recall(d, ds, BBQ) is not None]
    if gaps:
        agg("max_recall_gap_fp_minus_bbq_max_pp", max(gaps))
        agg("max_recall_gap_fp_minus_bbq_mean_pp", sum(gaps) / len(gaps))

    # --- RaBitQ-7bit (RAM)
    agg("rabitq7_never_reaches_0.99", sum(1 for ds in datasets if not best(d, ds, R7, .99)))
    for th in (.95, .99):
        r = [q(best(d, ds, R7, th)) / q(best(d, ds, BBQ, th)) for ds in datasets
             if best(d, ds, R7, th) and best(d, ds, BBQ, th)]
        if r:
            agg(f"rabitq7_over_bbq_median_at_{th}", st.median(r))
            agg(f"rabitq7_over_bbq_configs_at_{th}", len(r))

    # --- RaBitQ-SSD
    g = [(max_recall(d, ds, RSSD) - max_recall(d, ds, BBQ)) * 100 for ds in datasets if max_recall(d, ds, RSSD) is not None and max_recall(d, ds, BBQ) is not None]
    if g:
        agg("rabitq_ssd_higher_max_recall_than_bbq", sum(x > 0 for x in g))
        agg("rabitq_ssd_minus_bbq_max_recall_mean_pp", sum(g) / len(g))
    nonfp = [m for m in methods if m != FP]
    agg("rabitq_ssd_highest_max_recall_excl_fp_hnsw",
        sum(max(nonfp, key=lambda m: max_recall(d, ds, m) or 0) == RSSD for ds in datasets))
    r = [q(best(d, ds, BBQ, .99)) / q(best(d, ds, RSSD, .99)) for ds in datasets
         if best(d, ds, BBQ, .99) and best(d, ds, RSSD, .99)]
    if r:
        agg("bbq_over_rabitq_ssd_median_at_0.99", st.median(r))
        agg("bbq_over_rabitq_ssd_min_at_0.99", min(r))
        agg("bbq_over_rabitq_ssd_max_at_0.99", max(r))

    # --- max-recall change of other d/8 codes relative to BBQ (mean over configurations, pp)
    for m, tag in ((OPQ, "opq"), (PQ, "pq"), (TQ, "tq")):
        g = [(max_recall(d, ds, m) - max_recall(d, ds, BBQ)) * 100 for ds in datasets if max_recall(d, ds, m) is not None and max_recall(d, ds, BBQ) is not None]
        if g:
            agg(f"{tag}_minus_bbq_max_recall_mean_pp", sum(g) / len(g))

    # --- SSD-resident graph systems (DiskANN-PQ, DiskANN-RaBitQ, PipeANN) at 0.99
    r = [q(best(d, ds, BBQ, .99)) / q(best(d, ds, m, .99)) for ds in datasets for m in SSD_GRAPH
         if best(d, ds, BBQ, .99) and best(d, ds, m, .99)]
    if r:
        agg("bbq_over_ssd_graph_min_at_0.99", min(r))
        agg("bbq_over_ssd_graph_max_at_0.99", max(r))
        agg("bbq_over_ssd_graph_pairs_at_0.99", len(r))
    for m in SSD_GRAPH:
        r = [q(best(d, ds, BBQ, .99)) / q(best(d, ds, m, .99)) for ds in datasets
             if best(d, ds, BBQ, .99) and best(d, ds, m, .99)]
        if r:
            tag = {"DiskANN-PQ d/8": "diskann_pq", "DiskANN-RaBitQ 1-bit": "diskann_rabitq",
                   "PipeANN-RaBitQ 1-bit": "pipeann"}[m]
            agg(f"bbq_over_{tag}_min_at_0.99", min(r))
            agg(f"bbq_over_{tag}_max_at_0.99", max(r))
            agg(f"bbq_over_{tag}_median_at_0.99", st.median(r))

    # --- method selection at a matched d/8-byte code budget (Sec 6.6, Table 7, Findings 8-9).
    # A cell is (configuration, recall target); its contenders are every method except FP32 HNSW and RaBitQ-7bit,
    # each at its best QPS with Recall@10 >= target. Definitions follow Sieun Kim's BBQ_evidence/02_verify_sec6.py.
    iso = [m for m in methods if m not in (FP, R7)]
    cells = []
    for th in (0.95, 0.98, 0.99, 0.995, 0.998):
        for ds in datasets:
            c = sorted(((q(best(d, ds, m, th)), m) for m in iso if best(d, ds, m, th)), reverse=True)
            if c:
                cells.append((ds, th, c))
    wins = [c for _, _, c in cells if c[0][1] == BBQ]
    agg("dbudget_cells", len(cells))
    agg("dbudget_bbq_fastest_cells", len(wins))
    agg("dbudget_bbq_losses", len(cells) - len(wins))
    agg("dbudget_bbq_margin_over_runner_up_median", st.median(c[0][0] / c[1][0] for c in wins if len(c) > 1))
    agg("dbudget_losses_above_bbq_ceiling",
        sum(1 for ds, th, c in cells if c[0][1] != BBQ and max_recall(d, ds, BBQ) < th))
    for pp in (0.3, 0.5):
        h = [(ds, th, c) for ds, th, c in cells if (max_recall(d, ds, BBQ) - th) * 100 >= pp]
        agg(f"dbudget_cells_bbq_headroom_ge_{pp}pp", len(h))
        agg(f"dbudget_bbq_fastest_headroom_ge_{pp}pp", sum(c[0][1] == BBQ for _, _, c in h))

    # --- scaling from T=1 to this table (Appendix B)
    if t1:
        for m, tag in ((BBQ, "bbq"), (FP, "fp_hnsw")):
            s = [q(best(d, ds, m, .99)) / q(best(t1, ds, m, .99)) for ds in datasets
                 if best(d, ds, m, .99) and best(t1, ds, m, .99) and best(d, ds, BBQ, .99) and best(t1, ds, BBQ, .99)]
            if s:
                agg(f"{tag}_thread_speedup_median_at_0.99", st.median(s))
                agg(f"{tag}_thread_speedup_configs", len(s))

    fields = ["kind", "dataset", "method", "threshold", "metric", "value", "search_parameter_value", "recall_at_10",
              "fp_vectors_per_query", "note"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
