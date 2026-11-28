#!/usr/bin/env python3
"""Build index_quality.dat (Fig 8) from the graph-construction precision run.

Input : ../results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv
        (scripts/run_msmarco_quora_qwen1024_graph_construction_precision.py; single thread, efSearch 32..1024,
        identical packed 1-bit codes and ADC-BQ search, graph built with FP32 vs Hamming neighbour selection)
Output: index_quality.dat, gnuplot index blocks {fp32,onebit}_{a,b}_{curve,pts}; x = Recall@10, y = QPS.
        (a) MS MARCO Qwen-1024, (b) Quora Qwen-1024.  Curves join every measured efSearch; markers at MARK.
"""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "results/graph_construction_precision/msmarco_quora_qwen1024_graph_construction_precision_paired.csv"
MARK = (32, 64, 128, 256, 512, 1024)
rows = list(csv.DictReader(open(SRC)))
out = [f"# built by make_index_quality_dat.py from {SRC.relative_to(HERE.parent)}", "# Recall@10  QPS"]
for ds, t in (("msmarco", "a"), ("quora", "b")):
    for name, rk, qk in (("fp32", "fp_recall_at_10", "fp_qps"), ("onebit", "binary_recall_at_10", "binary_qps")):
        pts = sorted((int(r["efSearch"]), float(r[rk]), float(r[qk])) for r in rows if r["dataset"] == ds)
        out += ["", "", f"# {name}_{t}_curve"] + [f"{x:.6f} {q:.2f}" for _, x, q in pts]
        out += ["", "", f"# {name}_{t}_pts"] + [f"{x:.6f} {q:.2f}" for ef, x, q in pts if ef in MARK]
(HERE / "index_quality.dat").write_text("\n".join(out) + "\n")
print(f"wrote {HERE / 'index_quality.dat'}")
