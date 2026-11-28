#!/usr/bin/env python3
"""MS MARCO SSD page reads: calibrated ADC-BQ, DiskANN, DiskANN RaBitQ-1, TurboQuant.

Extends the three-method DiskVecLab page-I/O comparison with TurboQuant
(Qdrant TQ1+, resident codes, exact FP rerank via O_DIRECT io_uring batch) for
the two MS MARCO embeddings.  TurboQuant rerank depth R is fixed per dataset:
OpenAI-1536 uses R=120 and Qwen-1024 uses R=60.  All methods share the same
seed11 held-out split at T=1.
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import plot_diskveclab_three_method_page_io as base  # re-execs into conda if needed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO = base.REPO
ROOT = base.ROOT
OUT = ROOT / "page_io_msmarco_4methods"

PANELS = (
    ("msmarco_qwen1024", "MS MARCO · Qwen-0.6B (1024D)"),
)
DATASETS = {key for key, _ in PANELS}

# Dataset -> TurboQuant rerank depth and the io_uring source CSV.
TQ_RERANK = {"msmarco_qwen1024": 60}
TQ_ROOT = ROOT / "tq_rerank_pareto_uring/raw"

METHODS = dict(base.METHODS)
METHODS["turboquant"] = {
    "label": "TurboQuant TQ1+ · RAM graph + FP@R (io_uring)",
    "short": "TurboQuant",
    "color": "#E69F00",
    "marker": "p",
    "linestyle": "-.",
    "component": "FP sidecar pages",
}
METHOD_ORDER = ("adcbq", "turboquant", "diskann_classic", "diskann_rabitq")


def tq_path(dataset: str) -> Path:
    return TQ_ROOT / dataset / f"qdranttq1.r{TQ_RERANK[dataset]}.csv"


def collect_turboquant() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    meta = {
        row["dataset"]: row
        for row in base.read_csv(ROOT / "split_manifest.csv")
    }
    for dataset, _ in PANELS:
        path = tq_path(dataset)
        for row in base.read_csv(path):
            if row["dataset"] != dataset or row["threads"] != "1":
                raise RuntimeError(f"unexpected TurboQuant row in {path}: {row}")
            if row["fp_io_mode"] != "uring-batch" or row["code_storage"] != "resident":
                raise RuntimeError(f"TurboQuant row is not resident+io_uring: {row}")
            rerank = int(row["rerank_n"])
            if rerank != TQ_RERANK[dataset]:
                raise RuntimeError(f"rerank mismatch in {path}: {rerank}")
            requests = float(row["fp_page_reads_per_query"])
            fp_pages = float(row["fp_read_bytes_per_query"]) / 4096.0
            graph_pages = float(row["page_read_bytes_per_query"]) / 4096.0
            total = fp_pages + graph_pages
            output.append(
                {
                    "dataset": dataset,
                    "corpus": meta[dataset]["corpus"],
                    "model": meta[dataset]["model"],
                    "dim": int(row["dim"]),
                    "test_queries": int(row["nq"]),
                    "method_key": "turboquant",
                    "method": f"{METHODS['turboquant']['label']} R={rerank}",
                    "search_parameter": "efSearch",
                    "search_parameter_value": int(row["efSearch"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "io_requests_per_query_mean": requests,
                    "graph_or_node_4k_pages_per_query_mean": graph_pages,
                    "fp_sidecar_4k_pages_per_query_mean": fp_pages,
                    "total_4k_pages_per_query_mean": total,
                    "pages_per_request": total / requests,
                    "io_component": METHODS["turboquant"]["component"],
                    "io_engine": "io_uring",
                    "source_csv": base.relative(path),
                }
            )
    return output


def collect() -> list[dict[str, object]]:
    rows = [row for row in base.collect() if row["dataset"] in DATASETS]
    rows += collect_turboquant()
    return rows


def validate(rows: list[dict[str, object]]) -> None:
    counts = Counter((str(r["dataset"]), str(r["method_key"])) for r in rows)
    for dataset, _ in PANELS:
        for method in METHOD_ORDER:
            if counts[(dataset, method)] == 0:
                raise RuntimeError(f"missing curve {dataset}/{method}")
    split = {
        row["dataset"]: int(row["test_queries"])
        for row in base.read_csv(ROOT / "split_manifest.csv")
    }
    for row in rows:
        if int(row["test_queries"]) != split[str(row["dataset"])]:
            raise RuntimeError(f"held-out split mismatch: {row}")
        dim = int(row["dim"])
        if row["method_key"] in ("adcbq", "turboquant"):
            expected = {1024: 1.0, 1536: 2.0}[dim]
        else:
            expected = 2.0
        if abs(float(row["pages_per_request"]) - expected) > 1e-3:
            raise RuntimeError(f"page alignment/accounting mismatch: {row}")


def grouped_rows(rows):
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method_key"]))].append(row)
    for values in grouped.values():
        values.sort(key=lambda r: int(r["search_parameter_value"]))
    return grouped


def label(method_key: str, dataset: str | None = None) -> str:
    text = METHODS[method_key]["label"]
    if method_key == "turboquant" and dataset is not None:
        text = text.replace("FP@R", f"FP@{TQ_RERANK[dataset]}")
    return text


def handles() -> list[Line2D]:
    out = []
    for key in METHOD_ORDER:
        text = METHODS[key]["label"]
        if key == "turboquant":
            text = text.replace("FP@R", "FP@R (OpenAI R=120, Qwen R=60)")
        out.append(Line2D([0], [0], color=METHODS[key]["color"],
                          marker=METHODS[key]["marker"],
                          linestyle=METHODS[key]["linestyle"],
                          linewidth=1.7, markersize=5.5, label=text))
    return out


def plot_recall_pages(rows, stem: Path, zoom90: bool) -> None:
    grouped = grouped_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for index, ((dataset, title), ax) in enumerate(zip(PANELS, axes.flat)):
        for key in METHOD_ORDER:
            points = grouped[(dataset, key)]
            if zoom90:
                points = [r for r in points if float(r["recall_at_10"]) >= 0.90]
            if not points:
                continue
            style = METHODS[key]
            ax.plot([float(r["recall_at_10"]) for r in points],
                    [float(r["total_4k_pages_per_query_mean"]) for r in points],
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=1.5, markersize=4)
        ax.axvline(0.99, color="#2CA02C", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_title(title, fontsize=11)
        if zoom90:
            ax.set_xlim(0.90, 1.001)
        ax.set_xlabel("Held-out Recall@10")
        if index == 0:
            ax.set_ylabel("Submitted 4 KiB pages/query (log)")
    fig.legend(handles=handles() + [Line2D([0], [0], color="#2CA02C", linestyle="--",
                                           label="Recall@10 = 0.99")],
               loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2,
               frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.80, wspace=0.18)
    base.save_figure(fig, stem)


def plot_qps_pages(rows, stem: Path) -> None:
    grouped = grouped_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for index, ((dataset, title), ax) in enumerate(zip(PANELS, axes.flat)):
        for key in METHOD_ORDER:
            points = grouped[(dataset, key)]
            style = METHODS[key]
            ax.plot([float(r["total_4k_pages_per_query_mean"]) for r in points],
                    [float(r["qps"]) for r in points],
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=1.5, markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Submitted 4 KiB pages/query (log)")
        if index == 0:
            ax.set_ylabel("QPS (log)")
    fig.legend(handles=handles(), loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.11, top=0.80, wspace=0.18)
    base.save_figure(fig, stem)


def select_common_target(rows) -> list[dict[str, object]]:
    grouped = grouped_rows(rows)
    selected = []
    for dataset, _ in PANELS:
        target = min(max(float(r["recall_at_10"]) for r in grouped[(dataset, m)])
                     for m in METHOD_ORDER)
        for method in METHOD_ORDER:
            feasible = [r for r in grouped[(dataset, method)]
                        if float(r["recall_at_10"]) + 1e-12 >= target]
            best = min(feasible, key=lambda r: (float(r["total_4k_pages_per_query_mean"]),
                                                -float(r["qps"])))
            selected.append({
                "dataset": dataset, "dim": best["dim"],
                "target_recall_at_10": target, "method_key": method,
                "method": best["method"],
                "selected_parameter": best["search_parameter"],
                "selected_parameter_value": best["search_parameter_value"],
                "measured_recall_at_10": best["recall_at_10"], "qps": best["qps"],
                "io_requests_per_query_mean": best["io_requests_per_query_mean"],
                "total_4k_pages_per_query_mean": best["total_4k_pages_per_query_mean"],
                "pages_per_request": best["pages_per_request"],
            })
    return selected


def select_at_recall(rows, target: float) -> list[dict[str, object]]:
    """Fewest-page point per method with Recall@10 >= target (if reachable)."""
    grouped = grouped_rows(rows)
    out = []
    for dataset, _ in PANELS:
        for method in METHOD_ORDER:
            feasible = [r for r in grouped[(dataset, method)]
                        if float(r["recall_at_10"]) + 1e-12 >= target]
            if not feasible:
                out.append({"dataset": dataset, "method_key": method,
                            "target_recall_at_10": target, "reachable": False})
                continue
            best = min(feasible, key=lambda r: (float(r["total_4k_pages_per_query_mean"]),
                                                -float(r["qps"])))
            out.append({
                "dataset": dataset, "method_key": method,
                "target_recall_at_10": target, "reachable": True,
                "selected_parameter": best["search_parameter"],
                "selected_parameter_value": best["search_parameter_value"],
                "measured_recall_at_10": best["recall_at_10"], "qps": best["qps"],
                "total_4k_pages_per_query_mean": best["total_4k_pages_per_query_mean"],
            })
    return out


def plot_bars(selected, stem: Path) -> None:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in selected:
        grouped[str(row["dataset"])][str(row["method_key"])] = row
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for index, ((dataset, title), ax) in enumerate(zip(PANELS, axes.flat)):
        rows = grouped[dataset]
        values = [float(rows[k]["total_4k_pages_per_query_mean"]) for k in METHOD_ORDER]
        bars = ax.bar(np.arange(len(METHOD_ORDER)), values,
                      color=[METHODS[k]["color"] for k in METHOD_ORDER], alpha=0.9)
        target = float(rows[METHOD_ORDER[0]]["target_recall_at_10"])
        ax.set_title(f"{title}\nminimum pages at Recall ≥ {target:.3f}", fontsize=10.5)
        ax.set_xticks(np.arange(len(METHOD_ORDER)))
        ax.set_xticklabels([
            f"{METHODS[k]['short']}\nR={TQ_RERANK[dataset]}" if k == "turboquant"
            else METHODS[k]["short"] for k in METHOD_ORDER], fontsize=9)
        ax.set_yscale("log")
        ax.grid(True, axis="y", which="both", alpha=0.2)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.04, f"{value:.0f}",
                    ha="center", va="bottom", fontsize=8.5)
        if index == 0:
            ax.set_ylabel("Submitted 4 KiB pages/query (log)")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.12, top=0.86, wspace=0.18)
    base.save_figure(fig, stem)


def write_summary(selected, at99) -> Path:
    lines = [
        "# MS MARCO four-method SSD page-I/O summary\n",
        "Held-out seed11 split (5,980 queries), T=1. Pages = O_DIRECT submitted bytes / 4096.",
        "TurboQuant: Qdrant TQ1+ codes resident in RAM, exact FP rerank of top-R via io_uring "
        "batch (OpenAI R=120, Qwen R=60).\n",
        "## Fewest pages at the highest recall reachable by all four methods\n",
        "| dataset | target R@10 | method | param | R@10 | pages/q | QPS |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in selected:
        lines.append(
            f"| {r['dataset']} | {float(r['target_recall_at_10']):.4f} | {r['method_key']} | "
            f"{r['selected_parameter']}={r['selected_parameter_value']} | "
            f"{float(r['measured_recall_at_10']):.4f} | "
            f"{float(r['total_4k_pages_per_query_mean']):.1f} | {float(r['qps']):.1f} |")
    lines += ["", "## Fewest pages at Recall@10 ≥ 0.99\n",
              "| dataset | method | param | R@10 | pages/q | QPS |",
              "|---|---|---:|---:|---:|---:|"]
    for r in at99:
        if not r["reachable"]:
            lines.append(f"| {r['dataset']} | {r['method_key']} | not reached | | | |")
            continue
        lines.append(
            f"| {r['dataset']} | {r['method_key']} | "
            f"{r['selected_parameter']}={r['selected_parameter_value']} | "
            f"{float(r['measured_recall_at_10']):.4f} | "
            f"{float(r['total_4k_pages_per_query_mean']):.1f} | {float(r['qps']):.1f} |")
    path = OUT / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    validate(rows)
    dataset_order = {k: i for i, (k, _) in enumerate(PANELS)}
    method_order = {k: i for i, k in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda r: (dataset_order[str(r["dataset"])],
                             method_order[str(r["method_key"])],
                             int(r["search_parameter_value"])))
    normalized = OUT / "normalized_page_io.csv"
    base.write_csv(normalized, rows, base.NORMALIZED_FIELDS)

    selected = select_common_target(rows)
    selected_path = OUT / "minimum_pages_at_highest_common_recall.csv"
    base.write_csv(selected_path, selected, tuple(selected[0]))
    at99 = select_at_recall(rows, 0.99)
    at99_path = OUT / "minimum_pages_at_recall099.csv"
    fields = tuple(dict.fromkeys(k for r in at99 for k in r))
    base.write_csv(at99_path, [{k: r.get(k, "") for k in fields} for r in at99], fields)

    plot_recall_pages(rows, OUT / "recall_vs_4k_pages_full", zoom90=False)
    plot_recall_pages(rows, OUT / "recall_vs_4k_pages_zoom90", zoom90=True)
    plot_qps_pages(rows, OUT / "qps_vs_4k_pages")
    plot_bars(selected, OUT / "minimum_pages_at_highest_common_recall")
    summary = write_summary(selected, at99)
    for path in (normalized, selected_path, at99_path, summary):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
