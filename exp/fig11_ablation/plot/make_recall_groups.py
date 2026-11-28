#!/usr/bin/env python3
"""Fig 11(b): per-query outcomes of calibrated BBQ (MS MARCO Qwen-1024, efSearch 256, target 0.99, 5,980 test queries).

    make_recall_groups.py [per_query_target099.csv]
      default input: ../../fig12_calibration/results/legacy_gt/per_query_target099.csv
                     (scripts/run_msmarco_qwen1024_calibration_analysis.py, default result root = the paper's GT)

Groups the test queries by final Recall@10 exactly as the runner does (np.isclose 1.0 / 0.9, <= 0.8) and writes
  ../results/per_query_recall_groups_target099.legacy_gt.csv   same columns as the runner's per_query_recall_groups file
  recall_groups.gp                                              the $C datablock that recall_ablation.gp plots
FP reads per query = selected_r; latency = end_to_end_us.
"""
import csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1] / "fig12_calibration/results/legacy_gt/per_query_target099.csv"
rows = list(csv.DictReader(open(SRC)))
rec = [float(r["final_recall_at_10"]) for r in rows]
specs = (("1.0", lambda x: abs(x - 1.0) <= 1e-8), ("0.9", lambda x: abs(x - 0.9) <= 1e-8), ("<=0.8", lambda x: x <= 0.8 + 1e-12))
out, n = [], len(rows)
for label, f in specs:
    g = [r for r, x in zip(rows, rec) if f(x)]
    out.append(dict(per_query_recall_at_10=label, queries=len(g), query_fraction=len(g) / n, query_percent=100 * len(g) / n,
                    mean_fp_reads=sum(float(r["selected_r"]) for r in g) / len(g),
                    mean_latency_ms=sum(float(r["end_to_end_us"]) for r in g) / len(g) / 1000))
res = HERE.parent / "results/per_query_recall_groups_target099.legacy_gt.csv"
with open(res, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
lab = {"1.0": "1.0", "0.9": "0.9", "<=0.8": "≤0.8"}
blk = [f"# built by make_recall_groups.py from {SRC.name} ({n} queries); idx label queries %queries fp_reads latency_ms", "$C << EOD"]
blk += [f'{i} "{lab[o["per_query_recall_at_10"]]}" {o["queries"]} {o["query_percent"]:.2f} {o["mean_fp_reads"]:.1f} {o["mean_latency_ms"]:.3f}'
        for i, o in enumerate(out)]
(HERE / "recall_groups.gp").write_text("\n".join(blk + ["EOD"]) + "\n")
print(f"wrote {res.name} and recall_groups.gp")
