#!/usr/bin/env python3
"""Plot the T=1 native-resident quantizer and native-adaptive RaBitQ sweep."""

import csv
from pathlib import Path

import plot_qwen1024_all_quantizers_joint_pareto as plotter


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiment/qwen1024_all_quantizers_resident_uring_batch_rerank_t1_results"
FIGURE_ROOT = REPO / "experiment/figures/qwen1024_native_resident_uring_batch_t1"

plotter.FIXED_INPUT = ROOT / "qwen1024_fixed_resident_ef_rerank_sweep_t1.csv"
plotter.RABITQ_INPUT = (
    ROOT / "adaptive_rabitq/qwen1024_rabitq1_c16_adaptive_t1_combined.csv"
)
plotter.OUTPUT_DIR = FIGURE_ROOT
plotter.OUTPUT_STEM = FIGURE_ROOT / "qwen1024_native_resident_uring_batch_pareto_t1"
plotter.POINTS_CSV = FIGURE_ROOT / "qwen1024_native_resident_mean_points_t1.csv"
plotter.FRONTIER_CSV = FIGURE_ROOT / "qwen1024_native_resident_pareto_frontiers_t1.csv"
plotter.TARGET_CSV = FIGURE_ROOT / "qwen1024_native_resident_target_summary_t1.csv"
plotter.FIGURE_TITLE = (
    "T=1 Recall-throughput and Recall-SSD-read Pareto frontiers: "
    "native compressed representations resident"
)
plotter.SETTING_LINE = (
    "M=24, efC=200, T=1, three-round arithmetic mean, 1,000 queries; "
    "fixed-R: one per-query io_uring batch with no inter-query overlap; "
    "RaBitQ: native adaptive sync O_DIRECT"
)
plotter.POLICY_LINE = (
    "Fixed methods jointly sweep efSearch and exact-rerank R; "
    "RaBitQ sweeps efSearch with native adaptive c16"
)


def read_adaptive_rabitq() -> list[dict[str, object]]:
    with plotter.RABITQ_INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points: list[dict[str, object]] = []
    for row in rows:
        rounds = int(row["rounds"])
        if rounds != 3 or int(row["threads"]) != 1:
            raise RuntimeError(
                f"expected T=1 and three rounds for {row['dataset']}/ef{row['efSearch']}"
            )
        fp_reads = float(row["fp_page_reads_per_query_mean"])
        points.append(
            {
                "dataset": row["dataset"],
                "method_key": "rabitq1",
                "method": plotter.DISPLAY["rabitq1"],
                "rerank_policy": "adaptive_c16",
                "rerank_n": "adaptive",
                "efSearch": int(row["efSearch"]),
                "rounds": rounds,
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "qps_std": "",
                "qps_cv": "",
                "code_page_reads_per_query": 0.0,
                "fp_page_reads_per_query": fp_reads,
                "total_physical_page_reads_per_query": fp_reads,
                "source_csv": str(plotter.RABITQ_INPUT),
            }
        )
    return points


_read_fixed = plotter.read_fixed


def read_and_validate_fixed() -> list[dict[str, object]]:
    points = _read_fixed()
    with plotter.FIXED_INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(int(row["threads"]) != 1 for row in rows):
        raise RuntimeError("fixed sweep is not uniformly T=1")
    return points


plotter.read_fixed = read_and_validate_fixed
plotter.read_rabitq = read_adaptive_rabitq


if __name__ == "__main__":
    plotter.main()
