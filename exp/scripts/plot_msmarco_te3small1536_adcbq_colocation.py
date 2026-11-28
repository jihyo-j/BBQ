#!/usr/bin/env python3
"""Export paired comparisons and plots after the ADC-BQ campaign completes."""
import csv
import json
import math
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "msmarco_te3small1536_adcbq_colocation_ef256_t24"


def main():
    assert json.loads((ROOT / "validation.json").read_text())["status"] == "passed"
    with (ROOT / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    indexed = {(int(r["page_bytes"]), float(r["bq_ram_ratio_target"]), r["layout"]): r for r in rows}
    assert len(indexed) == 24
    comparison = []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row", constrained_layout=True)
    ratios = (0.1, 0.25, 0.5, 0.75)
    for column, page in enumerate((4096, 8192, 16384)):
        for ratio in ratios:
            baseline, coloc = [indexed[page, ratio, layout] for layout in ("baseline", "coloc")]
            for key in ("dataset", "model", "cache_mib", "efSearch", "threads", "calibration_tau"):
                assert baseline[key] == coloc[key], (page, ratio, key)
            assert math.isclose(float(baseline["recall_at_10"]), float(coloc["recall_at_10"]), abs_tol=1e-8)
            speedup = float(coloc["qps"]) / float(baseline["qps"])
            record = {key: baseline[key] for key in ("dataset", "model", "cache_mib", "efSearch", "threads", "calibration_tau")}
            record.update(page_bytes=page, ram_ratio=ratio, measured_trials=3,
                          baseline_qps_median=float(baseline["qps"]),
                          colocation_qps_median=float(coloc["qps"]),
                          qps_speedup=speedup, qps_gain_pct=100 * (speedup - 1),
                          recall_at_10=float(baseline["recall_at_10"]))
            for key in ("graph_physical_4k_reads_per_query", "total_physical_4k_reads_per_query"):
                left, right = float(baseline[key]), float(coloc[key])
                record[f"baseline_{key}"] = left
                record[f"colocation_{key}"] = right
                record[f"{key}_reduction_pct"] = 100 * (1 - right / left)
            comparison.append(record)
        for layout, label, color in (("baseline", "Baseline", "#666666"), ("coloc", "Colocation", "#2474b5")):
            group = [indexed[page, ratio, layout] for ratio in ratios]
            qps = [float(r["qps"]) for r in group]
            errors = [[v - float(r["qps_min"]) for v, r in zip(qps, group)],
                      [float(r["qps_max"]) - v for v, r in zip(qps, group)]]
            x = [100 * r for r in ratios]
            axes[0, column].errorbar(x, qps, yerr=errors, marker="o", capsize=3, label=label, color=color)
            axes[1, column].plot(x, [float(r["graph_physical_4k_reads_per_query"]) for r in group],
                                 marker="o", label=label, color=color)
        axes[0, column].set_title(f"{page // 1024} KiB graph pages")
        for axis in axes[:, column]:
            axis.set_xticks([10, 25, 50, 75])
            axis.grid(alpha=0.25)
        axes[0, column].set_ylabel("QPS (median; bars = min/max of 3 runs)")
        axes[1, column].set_ylabel("Graph SSD reads / query (4 KiB units)")
        axes[1, column].set_xlabel("Page-cache budget (% of 4 KiB baseline footprint)")
    # Set shared ranges after all pages are plotted, including their error bars.
    axes[0, 0].set_ylim(0, max(float(r["qps_max"]) for r in rows) * 1.08)
    axes[1, 0].set_ylim(0, max(float(r["graph_physical_4k_reads_per_query"]) for r in rows) * 1.08)
    axes[0, 0].legend()
    fig.suptitle("MSMARCO text-embedding-3-small, 1536D | ADC-BQ | ef=256 | 24 threads\n"
                 f"Fixed calibrated tau={float(rows[0]['calibration_tau']):.6f}; recall@10={float(rows[0]['recall_at_10']):.6f}")
    destination = ROOT / "comparison.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    assert len(comparison) == 12
    fig.savefig(ROOT / "comparison.png", dpi=160)
    fig.savefig(ROOT / "comparison.pdf")
    plt.close(fig)
    for source, target in ((destination, Path("/home/jihyo/msmarco_te3small1536_adcbq_colocation_comparison_t24.csv")),
                           (ROOT / "comparison.png", Path("/home/jihyo/msmarco_te3small1536_adcbq_colocation_t24.png"))):
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    print(f"Exported {len(comparison)} paired comparisons and PNG/PDF plots to {ROOT}")


if __name__ == "__main__":
    main()
