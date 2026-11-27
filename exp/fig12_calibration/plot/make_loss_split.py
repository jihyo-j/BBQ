#!/usr/bin/env python3
"""Fig. 12(b) loss split at calibration target 0.99, written as CSV for verify.

    make_loss_split.py <results/legacy_gt> > loss_split_target099.csv

L_visit  = mean over test queries of (1 - recall of exact reranking of the entire pool)
L_select = mean over test queries of (full-pool recall - final recall)
so that L_visit + L_select = 1 - mean Recall@10. Query counts: queries with any loss of each kind."""
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] + "/per_query_target099.csv")))
n = len(rows)
pool = [float(r["full_pool_exact_recall_at_10"]) for r in rows]
fin = [float(r["final_recall_at_10"]) for r in rows]
lv = sum(1 - p for p in pool) / n
ls = sum(p - f for p, f in zip(pool, fin)) / n
w = csv.writer(sys.stdout)
w.writerow(["component", "loss", "queries"])
w.writerow(["L_visit", f"{lv:.6f}", sum(p < 1 for p in pool)])
w.writerow(["L_select", f"{ls:.6f}", sum(f < p for p, f in zip(pool, fin))])
w.writerow(["total", f"{lv + ls:.6f}", n])
w.writerow(["visit_share", f"{lv / (lv + ls):.6f}", ""])
