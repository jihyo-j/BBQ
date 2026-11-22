#!/usr/bin/env python3
"""Plot matched-memory T=1 HNSW and classic DiskANN adaptive-FP results."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
NATIVE_FRONTIER = (
    REPO
    / "experiment/figures/qwen1024_native_resident_uring_batch_t1"
    / "qwen1024_native_resident_pareto_frontiers_t1.csv"
)
DISKANN_ROOT = REPO / "experiment/qwen1024_diskann_pq128_opq128_adaptive_t1"
DISKANN_COMBINED = DISKANN_ROOT / "diskann_pq128_opq128_adaptive_t1.csv"
FIXED_TIMING = REPO / "experiment/qwen1024_fixed_rerank_io_timing_t1/summary.csv"
RABITQ_TIMING = REPO / "experiment/qwen1024_rabitq_adaptive_io_timing_t1/summary.csv"
MSM_PREP = (
    REPO
    / "experiment/query_preprocess_overhead_qwen1024_t1"
    / "clean_20261110_cpu20_sq_opq_tq/query_preprocess_overhead_derived.csv"
)
NQ_PREP = (
    REPO
    / "experiment/query_preprocess_overhead_qwen1024_t1"
    / "nq_cpu20/nq_qwen1024.method_totals.csv"
)
MSM_ADCBQ_PREP = (
    REPO
    / "experiment/query_preprocess_overhead_qwen1024_t1"
    / "clean_20261110_cpu20_sq_opq_tq/msmarco_qwen1024.adcbq_main_uncalibrated_optimized.csv"
)
NQ_ADCBQ_PREP = (
    REPO
    / "experiment/query_preprocess_overhead_qwen1024_t1"
    / "nq_cpu20/nq_qwen1024.adcbq_main_uncalibrated_optimized.csv"
)
OUTPUT_DIR = REPO / "experiment/figures/qwen1024_diskann_adaptive_compare_t1"
POINTS_CSV = OUTPUT_DIR / "qwen1024_five_method_points_t1.csv"
FRONTIER_CSV = OUTPUT_DIR / "qwen1024_five_method_qps_frontiers_t1.csv"
TARGET_CSV = OUTPUT_DIR / "qwen1024_five_method_recall099_targets_t1.csv"
TIMING_CSV = OUTPUT_DIR / "qwen1024_five_method_recall099_timing_t1.csv"
COMBINED_CSV = OUTPUT_DIR / "qwen1024_five_method_plot_data_t1.csv"
FIGURE = OUTPUT_DIR / "qwen1024_five_method_recall_qps_preprocess_io_t1.png"

DATASETS = (
    ("msmarco_qwen1024", "MSMARCO Qwen-1024"),
    ("nq_qwen1024", "NQ Qwen-1024"),
)
METHODS = (
    ("adcbq1", "ADC-BQ"),
    ("qdranttq1", "TurboQuant TQ1+"),
    ("rabitq1", "RaBitQ-1 C16 adaptive"),
    ("diskann_pq128", "DiskANN PQ128 adaptive"),
    ("diskann_opq128", "DiskANN OPQ128 adaptive"),
)
LABEL = dict(METHODS)
COLORS = {
    "adcbq1": "#1f77b4",
    "qdranttq1": "#8c564b",
    "rabitq1": "#9467bd",
    "diskann_pq128": "#2ca02c",
    "diskann_opq128": "#d62728",
}
MARKERS = dict(zip(LABEL, ("o", "D", "X", "s", "^")))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pareto(points: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for point in points:
        recall = float(point["recall_at_10"])
        qps = float(point["qps"])
        dominated = any(
            float(candidate["recall_at_10"]) >= recall
            and float(candidate["qps"]) >= qps
            and (
                float(candidate["recall_at_10"]) > recall
                or float(candidate["qps"]) > qps
            )
            for candidate in points
            if candidate is not point
        )
        if not dominated:
            output.append(point)
    return sorted(output, key=lambda row: float(row["recall_at_10"]))


def load_points() -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for row in read_csv(NATIVE_FRONTIER):
        if row["frontier"] != "qps" or row["method_key"] not in {
            "adcbq1", "qdranttq1", "rabitq1"
        }:
            continue
        points.append(
            {
                "dataset": row["dataset"],
                "method_key": row["method_key"],
                "method": LABEL[row["method_key"]],
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "search_parameter": f"ef={row['efSearch']}",
                "rerank_or_reads": row["rerank_n"],
                "source_csv": row["source_csv"],
            }
        )

    disk_rows: list[dict[str, str]] = []
    if DISKANN_COMBINED.is_file():
        disk_rows = read_csv(DISKANN_COMBINED)
    else:
        for dataset, _ in DATASETS:
            for method in ("pq128", "opq128"):
                disk_rows.extend(read_csv(DISKANN_ROOT / dataset / method / "search_t1.csv"))
    for row in disk_rows:
        method_key = row["method_key"]
        points.append(
            {
                "dataset": row["dataset"],
                "method_key": method_key,
                "method": LABEL[method_key],
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "search_parameter": f"L={row['Lsearch']}, beam={row['beam_width']}",
                "rerank_or_reads": float(row["adaptive_fp_node_reads_per_query"]),
                "mean_query_preprocess_us": float(row["mean_query_preprocess_us"]),
                "mean_io_wait_us": float(row["mean_io_wait_us"]),
                "adaptive_fp_node_reads_per_query": float(row["adaptive_fp_node_reads_per_query"]),
                "source_csv": str(DISKANN_COMBINED),
            }
        )
    return points


def load_prep() -> dict[tuple[str, str], float]:
    output: dict[tuple[str, str], float] = {}
    names = {
        "ADC-BQ int16 nibble": "adcbq1",
        "TurboQuant TQ+ 1b": "qdranttq1",
        "RaBitQ-1b C16": "rabitq1",
    }
    for row in read_csv(MSM_PREP):
        if row["dataset"] != "msmarco_qwen1024" or row["component"] != "method_total":
            continue
        if row["method"] in names:
            output[("msmarco_qwen1024", names[row["method"]])] = float(
                row["median_us_per_query"]
            )
    for row in read_csv(NQ_PREP):
        if row["method"] in names:
            output[("nq_qwen1024", names[row["method"]])] = float(
                row["median_us_per_query"]
            )
    # Override ADC-BQ with the exact optimized, uncalibrated set_query() path.
    # The learned-threshold query-scale scan is intentionally excluded.
    for dataset, path in (
        ("msmarco_qwen1024", MSM_ADCBQ_PREP),
        ("nq_qwen1024", NQ_ADCBQ_PREP),
    ):
        row = read_csv(path)[0]
        output[(dataset, "adcbq1")] = float(row["median_us_per_query"])
    return output


def select_targets(points: list[dict[str, object]]) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    for dataset, _ in DATASETS:
        for method_key, method_name in METHODS:
            eligible = [
                row for row in points
                if row["dataset"] == dataset
                and row["method_key"] == method_key
                and float(row["recall_at_10"]) >= 0.99
            ]
            if not eligible:
                raise RuntimeError(f"no Recall@10>=0.99 point for {dataset}/{method_key}")
            best = max(eligible, key=lambda row: float(row["qps"]))
            targets.append({"target_recall": 0.99, **best, "method": method_name})
    return targets


def timing_rows(targets: list[dict[str, object]]) -> list[dict[str, object]]:
    prep = load_prep()
    fixed = {
        (row["dataset"], row["method_key"]): row for row in read_csv(FIXED_TIMING)
    }
    rabitq = {
        (row["dataset"], "rabitq1"): row for row in read_csv(RABITQ_TIMING)
    }
    output: list[dict[str, object]] = []
    for target in targets:
        dataset = str(target["dataset"])
        method_key = str(target["method_key"])
        if method_key in {"diskann_pq128", "diskann_opq128"}:
            preprocess_us = float(target["mean_query_preprocess_us"])
            io_us = float(target["mean_io_wait_us"])
            reads = float(target["adaptive_fp_node_reads_per_query"])
            io_policy = "native-AIO FP graph-node reads interleaved in beam search"
        elif method_key == "rabitq1":
            timing = rabitq[(dataset, method_key)]
            preprocess_us = prep[(dataset, method_key)]
            io_us = float(timing["io_wait_us_per_query"])
            reads = float(timing["adaptive_fp_reads_per_query"])
            io_policy = "sync pread FP reads interleaved in HNSW search"
        elif method_key == "adcbq1":
            timing = fixed[(dataset, method_key)]
            preprocess_us = prep[(dataset, method_key)]
            io_us = float(timing["exact_fp_io_wait_us_per_query"])
            reads = float(timing["rerank_n"])
            io_policy = "one fixed final per-query io_uring FP rerank batch"
        else:
            timing = fixed[(dataset, method_key)]
            preprocess_us = prep[(dataset, method_key)]
            io_us = float(timing["exact_fp_io_wait_us_per_query"])
            reads = float(timing["rerank_n"])
            io_policy = "one final per-query io_uring FP rerank batch"
        output.append(
            {
                "dataset": dataset,
                "method_key": method_key,
                "method": target["method"],
                "recall_at_10": target["recall_at_10"],
                "qps": target["qps"],
                "query_preprocess_us": preprocess_us,
                "exact_fp_io_wait_us": io_us,
                "fixed_rerank_or_adaptive_reads_per_query": reads,
                "io_policy": io_policy,
                "search_parameter": target["search_parameter"],
            }
        )
    return output


def combined_plot_rows(
    frontiers: list[dict[str, object]],
    timings: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Emit one self-contained table for both panels of the figure."""
    timing_by_method = {
        (str(row["dataset"]), str(row["method_key"])): row
        for row in timings
    }
    dataset_order = {dataset: index for index, (dataset, _) in enumerate(DATASETS)}
    dataset_title = dict(DATASETS)
    method_order = {method: index for index, (method, _) in enumerate(METHODS)}
    output: list[dict[str, object]] = []
    for row in frontiers:
        dataset = str(row["dataset"])
        method_key = str(row["method_key"])
        timing = timing_by_method[(dataset, method_key)]
        is_target = (
            float(row["recall_at_10"]) == float(timing["recall_at_10"])
            and float(row["qps"]) == float(timing["qps"])
        )
        output.append(
            {
                "dataset_order": dataset_order[dataset],
                "dataset": dataset,
                "dataset_title": dataset_title[dataset],
                "method_order": method_order[method_key],
                "method_key": method_key,
                "method": row["method"],
                "color": COLORS[method_key],
                "marker": MARKERS[method_key],
                "line_style": "--" if "adaptive" in str(row["method"]).lower() else "-",
                "recall_at_10": row["recall_at_10"],
                "qps": row["qps"],
                "search_parameter": row["search_parameter"],
                "rerank_or_reads_per_query": row["rerank_or_reads"],
                "source_csv": row["source_csv"],
                "target_recall": 0.99 if is_target else "",
                "is_recall099_target": str(is_target).lower(),
                "query_preprocess_us": timing["query_preprocess_us"] if is_target else "",
                "exact_fp_io_wait_us": timing["exact_fp_io_wait_us"] if is_target else "",
                "target_reads_per_query": (
                    timing["fixed_rerank_or_adaptive_reads_per_query"]
                    if is_target
                    else ""
                ),
                "io_policy": timing["io_policy"] if is_target else "",
            }
        )
    target_count = sum(row["is_recall099_target"] == "true" for row in output)
    if target_count != len(DATASETS) * len(METHODS):
        raise RuntimeError(
            f"expected {len(DATASETS) * len(METHODS)} target rows, got {target_count}"
        )
    return output


def plot(frontiers: list[dict[str, object]], timings: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.2, 9.4))
    for column, (dataset, title) in enumerate(DATASETS):
        qps_ax = axes[0, column]
        timing_ax = axes[1, column]
        for method_key, method_name in METHODS:
            selected = [
                row for row in frontiers
                if row["dataset"] == dataset and row["method_key"] == method_key
            ]
            qps_ax.plot(
                [float(row["recall_at_10"]) for row in selected],
                [float(row["qps"]) for row in selected],
                color=COLORS[method_key],
                marker=MARKERS[method_key],
                markersize=4.7,
                linewidth=1.9,
                linestyle="--" if "adaptive" in method_name.lower() else "-",
                label=method_name,
            )
        qps_ax.axvline(0.99, color="0.4", linestyle=":", linewidth=1.1)
        qps_ax.set_xlim(0.90, 1.0005)
        qps_ax.set_yscale("log")
        qps_ax.set_title(title)
        qps_ax.set_xlabel("Recall@10")
        qps_ax.set_ylabel("QPS (higher is better, log scale)")
        qps_ax.grid(True, which="both", alpha=0.23)

        selected_timing = [row for row in timings if row["dataset"] == dataset]
        x = list(range(len(METHODS)))
        width = 0.37
        prep_bars = timing_ax.bar(
            [value - width / 2 for value in x],
            [float(row["query_preprocess_us"]) for row in selected_timing],
            width,
            color="#4c78a8",
            label="query preprocessing",
        )
        io_bars = timing_ax.bar(
            [value + width / 2 for value in x],
            [float(row["exact_fp_io_wait_us"]) for row in selected_timing],
            width,
            color="#f58518",
            label="exact-FP I/O wait",
        )
        timing_ax.set_xticks(
            x,
            ["ADC-BQ", "TQ1+", "RaBitQ", "DiskANN\nPQ128", "DiskANN\nOPQ128"],
        )
        timing_ax.set_yscale("log")
        timing_ax.set_ylabel("µs/query (log scale)")
        timing_ax.set_title("Fastest point with Recall@10 ≥ 0.99")
        timing_ax.grid(True, axis="y", which="both", alpha=0.23)
        for bars, field in (
            (prep_bars, "query_preprocess_us"),
            (io_bars, "exact_fp_io_wait_us"),
        ):
            for bar, row in zip(bars, selected_timing):
                value = float(row[field])
                timing_ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value * 1.08,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=90,
                )
        for position, row in zip(x, selected_timing):
            timing_ax.text(
                position,
                0.01,
                f"R/read={float(row['fixed_rerank_or_adaptive_reads_per_query']):.1f}",
                transform=timing_ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.4,
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=5, frameon=False)
    bar_handles, bar_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(bar_handles, bar_labels, loc="lower center", bbox_to_anchor=(0.5, 0.052), ncol=2, frameon=False)
    fig.suptitle(
        "Qwen-1024 matched 128 B/vector routing payload: Recall–QPS and query/I/O cost — T=1",
        fontsize=15.5,
        y=0.99,
    )
    fig.text(
        0.5,
        0.953,
        "1M vectors, first 1,000 queries, three-round means; compressed routing state resident; exact FP vectors on O_DIRECT SSD",
        ha="center",
        fontsize=9.5,
    )
    fig.text(
        0.5,
        0.018,
        (
            "Preprocessing excludes common normalization; ADC-BQ includes only optimized int16-nibble set_query "
            "without calibrated-threshold query-scale work. ADC-BQ/TQ1+: fixed final io_uring batch; "
            "RaBitQ: adaptive sync FP reads; DiskANN: adaptive native-AIO full-FP graph-node reads. "
            "PQ128/OPQ128 match the 1024-bit code payload; OPQ additionally owns a ~4 MiB rotation matrix."
        ),
        ha="center",
        fontsize=8.7,
    )
    fig.tight_layout(rect=(0.02, 0.09, 0.995, 0.895), h_pad=2.0, w_pad=1.25)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=210, bbox_inches="tight")
    fig.savefig(FIGURE.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = load_points()
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for point in points:
        grouped[(str(point["dataset"]), str(point["method_key"]))].append(point)
    frontiers = [point for group in grouped.values() for point in pareto(group)]
    targets = select_targets(points)
    timings = timing_rows(targets)
    write_csv(POINTS_CSV, points)
    write_csv(FRONTIER_CSV, frontiers)
    write_csv(TARGET_CSV, targets)
    write_csv(TIMING_CSV, timings)
    write_csv(COMBINED_CSV, combined_plot_rows(frontiers, timings))
    plot(frontiers, timings)
    print(f"WROTE {FIGURE}", flush=True)
    print(f"WROTE {COMBINED_CSV}", flush=True)


if __name__ == "__main__":
    main()
