#!/usr/bin/env python3
"""Plot independent QPS and SSD-read Pareto frontiers for Qwen-1024 arms."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
FIXED_INPUT = (
    REPO
    / "experiment/qwen1024_all_quantizers_joint_pareto_results"
    / "qwen1024_fixed_joint_ef_rerank_sweep_t24.csv"
)
RABITQ_INPUT = (
    REPO
    / "experiment/faiss_hnsw_bq_budget_paged_rerank80_results/rabitq_codepaged"
    / "qwen1024_rabitq1_c16_adaptive_codepaged128m_t24_combined.csv"
)
OUTPUT_DIR = REPO / "experiment/figures/qwen1024_all_quantizers_joint_pareto"
OUTPUT_STEM = OUTPUT_DIR / "qwen1024_joint_ef_rerank_pareto_qps_page_io_t24"
POINTS_CSV = OUTPUT_DIR / "qwen1024_joint_ef_rerank_mean_points_t24.csv"
FRONTIER_CSV = OUTPUT_DIR / "qwen1024_joint_ef_rerank_pareto_frontiers_t24.csv"
TARGET_CSV = OUTPUT_DIR / "qwen1024_joint_ef_rerank_target_summary_t24.csv"

DATASETS = (
    ("msmarco_qwen1024", "MSMARCO Qwen-1024"),
    ("nq_qwen1024", "NQ Qwen-1024"),
)
METHODS = (
    ("adcbq1", "ADC-BQ"),
    ("symbq1", "SymBQ"),
    ("pq8m128", "PQ8"),
    ("opq8m128", "OPQ-PQ8"),
    ("rabitq1", "RaBitQ"),
    ("qdranttq1", "TQ1+"),
    ("sq4", "SQ4"),
    ("sq8", "SQ8"),
)
DISPLAY = dict(METHODS)
COLORS = dict(zip(DISPLAY, plt.get_cmap("tab10").colors[: len(DISPLAY)]))
MARKERS = dict(zip(DISPLAY, ("o", "P", "s", "^", "X", "D", "v", "<")))
TARGETS = (0.95, 0.99)
FIGURE_TITLE = (
    "Joint efSearch × full-precision refinement Pareto frontiers "
    "under a 128 MB code-state budget"
)
SETTING_LINE = (
    "M=24, efC=200, T=24, three-round arithmetic mean, 1,000 queries, "
    "cold 4 KB O_DIRECT code cache"
)
POLICY_LINE = (
    "QPS and SSD-read frontiers are computed independently per method; "
    "fixed arms sweep rerank R, RaBitQ uses native adaptive FP"
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def read_fixed() -> list[dict[str, object]]:
    with FIXED_INPUT.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        key = (
            row["dataset"],
            row["sweep_method"],
            int(row["sweep_rerank_n"]),
            int(row["efSearch"]),
        )
        grouped[key].append(row)

    points: list[dict[str, object]] = []
    for (dataset, method, rerank_n, ef_search), rows in sorted(grouped.items()):
        if len(rows) != 3:
            raise RuntimeError(
                f"expected three rounds for {dataset}/{method}/R{rerank_n}/ef{ef_search}; "
                f"found {len(rows)}"
            )
        qps_values = [float(row["qps"]) for row in rows]
        qps_mean = mean(qps_values)
        qps_std = statistics.stdev(qps_values)
        points.append(
            {
                "dataset": dataset,
                "method_key": method,
                "method": DISPLAY[method],
                "rerank_policy": f"fixed_{rerank_n}",
                "rerank_n": rerank_n,
                "efSearch": ef_search,
                "rounds": 3,
                "recall_at_10": mean([float(row["recall_at_10"]) for row in rows]),
                "qps": qps_mean,
                "qps_std": qps_std,
                "qps_cv": qps_std / qps_mean if qps_mean else math.nan,
                "code_page_reads_per_query": mean(
                    [float(row["page_reads_per_query"]) for row in rows]
                ),
                "fp_page_reads_per_query": mean(
                    [float(row["fp_page_reads_per_query"]) for row in rows]
                ),
                "total_physical_page_reads_per_query": mean(
                    [float(row["total_physical_page_reads_per_query"]) for row in rows]
                ),
                "source_csv": rows[0]["source_csv"],
            }
        )
    return points


def read_rabitq() -> list[dict[str, object]]:
    with RABITQ_INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points: list[dict[str, object]] = []
    for row in rows:
        rounds = int(row["rounds"])
        if rounds != 3:
            raise RuntimeError(
                f"expected three RaBitQ rounds, found {rounds} for "
                f"{row['dataset']}/ef{row['efSearch']}"
            )
        points.append(
            {
                "dataset": row["dataset"],
                "method_key": "rabitq1",
                "method": DISPLAY["rabitq1"],
                "rerank_policy": "adaptive_c16",
                "rerank_n": "adaptive",
                "efSearch": int(row["efSearch"]),
                "rounds": rounds,
                "recall_at_10": float(row["recall_at_k"]),
                "qps": float(row["qps"]),
                "qps_std": "",
                "qps_cv": "",
                "code_page_reads_per_query": float(
                    row["index_page_requests_per_query"]
                ),
                "fp_page_reads_per_query": float(
                    row["fp_physical_4k_reads_per_query"]
                ),
                "total_physical_page_reads_per_query": float(
                    row["total_physical_4k_reads_per_query"]
                ),
                "source_csv": str(RABITQ_INPUT),
            }
        )
    return points


def pareto(
    points: list[dict[str, object]], metric: str, higher_is_better: bool
) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    for point in points:
        recall = float(point["recall_at_10"])
        value = float(point[metric])
        dominated = False
        for candidate in points:
            candidate_recall = float(candidate["recall_at_10"])
            candidate_value = float(candidate[metric])
            metric_no_worse = (
                candidate_value >= value
                if higher_is_better
                else candidate_value <= value
            )
            metric_strict = (
                candidate_value > value
                if higher_is_better
                else candidate_value < value
            )
            if (
                candidate_recall >= recall
                and metric_no_worse
                and (candidate_recall > recall or metric_strict)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(
        frontier,
        key=lambda row: (float(row["recall_at_10"]), float(row[metric])),
    )


def all_frontiers(points: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, _ in DATASETS:
        for method_key, _ in METHODS:
            method_points = [
                row
                for row in points
                if row["dataset"] == dataset and row["method_key"] == method_key
            ]
            if not method_points:
                raise RuntimeError(f"missing points for {dataset}/{method_key}")
            for frontier_name, metric, higher in (
                ("qps", "qps", True),
                ("ssd_reads", "total_physical_page_reads_per_query", False),
            ):
                for point in pareto(method_points, metric, higher):
                    rows.append({"frontier": frontier_name, **point})
    return rows


def target_summary(points: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset, _ in DATASETS:
        for method_key, method_name in METHODS:
            method_points = [
                row
                for row in points
                if row["dataset"] == dataset and row["method_key"] == method_key
            ]
            for target in TARGETS:
                eligible = [
                    row for row in method_points if float(row["recall_at_10"]) >= target
                ]
                reached = bool(eligible)
                candidates = eligible or [
                    row
                    for row in method_points
                    if float(row["recall_at_10"])
                    == max(float(item["recall_at_10"]) for item in method_points)
                ]
                for objective, selector in (
                    ("max_qps", lambda row: float(row["qps"])),
                    (
                        "min_ssd_reads",
                        lambda row: -float(row["total_physical_page_reads_per_query"]),
                    ),
                ):
                    best = max(candidates, key=selector)
                    output.append(
                        {
                            "dataset": dataset,
                            "method_key": method_key,
                            "method": method_name,
                            "target_recall": target,
                            "target_reached": reached,
                            "selection_objective": objective,
                            **{
                                key: value
                                for key, value in best.items()
                                if key not in {"dataset", "method_key", "method"}
                            },
                        }
                    )
    return output


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


def plot(frontiers: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.0), sharex="col")
    for column, (dataset, dataset_title) in enumerate(DATASETS):
        qps_axis = axes[0, column]
        io_axis = axes[1, column]
        for method_key, method_name in METHODS:
            common = {
                "color": COLORS[method_key],
                "marker": MARKERS[method_key],
                "markersize": 4.1,
                "linewidth": 1.75,
                "linestyle": "--" if method_key == "rabitq1" else "-",
            }
            qps_points = [
                row
                for row in frontiers
                if row["dataset"] == dataset
                and row["method_key"] == method_key
                and row["frontier"] == "qps"
            ]
            io_points = [
                row
                for row in frontiers
                if row["dataset"] == dataset
                and row["method_key"] == method_key
                and row["frontier"] == "ssd_reads"
            ]
            qps_axis.plot(
                [float(row["recall_at_10"]) for row in qps_points],
                [float(row["qps"]) for row in qps_points],
                label=method_name,
                **common,
            )
            io_axis.plot(
                [float(row["recall_at_10"]) for row in io_points],
                [float(row["total_physical_page_reads_per_query"]) for row in io_points],
                **common,
            )

        qps_axis.set_title(dataset_title, fontsize=12)
        qps_axis.set_ylabel("QPS (higher is better, log scale)")
        qps_axis.set_yscale("log")
        io_axis.set_ylabel("4 KB O_DIRECT reads/query (lower is better, log scale)")
        io_axis.set_xlabel("Recall@10")
        io_axis.set_yscale("log")
        for axis in (qps_axis, io_axis):
            axis.axvline(0.95, color="0.58", linestyle=":", linewidth=1.0)
            axis.axvline(0.99, color="0.36", linestyle=":", linewidth=1.0)
            axis.set_xlim(0.90, 1.001)
            axis.grid(True, which="both", alpha=0.23)
            axis.set_axisbelow(True)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(FIGURE_TITLE, y=0.985, fontsize=15)
    fig.text(
        0.5,
        0.946,
        POLICY_LINE,
        ha="center",
        fontsize=9.7,
    )
    fig.text(
        0.5,
        0.922,
        SETTING_LINE,
        ha="center",
        fontsize=9.3,
    )
    fig.tight_layout(rect=(0.02, 0.035, 0.995, 0.865), h_pad=1.15, w_pad=1.35)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=210, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    points = read_fixed() + read_rabitq()
    frontiers = all_frontiers(points)
    targets = target_summary(points)
    write_csv(POINTS_CSV, points)
    write_csv(FRONTIER_CSV, frontiers)
    write_csv(TARGET_CSV, targets)
    plot(frontiers)
    print(f"WROTE {OUTPUT_STEM.with_suffix('.png')}")
    print(f"WROTE {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"WROTE {POINTS_CSV}")
    print(f"WROTE {FRONTIER_CSV}")
    print(f"WROTE {TARGET_CSV}")


if __name__ == "__main__":
    main()
