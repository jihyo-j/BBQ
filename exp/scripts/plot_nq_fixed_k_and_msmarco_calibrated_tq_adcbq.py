#!/usr/bin/env python3
"""Plot the NQ fixed-k and paired MS MARCO calibrated follow-up runs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_nq() -> None:
    source = (
        REPO
        / "experiment/nq_qwen1024_turboquant_fixed_recall_k30_50_100_overlap_t1/summary_t1.csv"
    )
    rows = read_csv(source)
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for axis, k in zip(axes, (30, 50, 100)):
        selected = [row for row in rows if int(row["k"]) == k]
        selected.sort(key=lambda row: int(row["efSearch"]))
        fixed_r = int(selected[0]["fixed_r"])
        axis.plot(
            [float(row["recall_at_k"]) for row in selected],
            [float(row["qps"]) for row in selected],
            color="#8c564b",
            marker="D",
            markersize=4.2,
            linewidth=1.8,
        )
        for row in (selected[0], selected[-1]):
            axis.annotate(
                f"ef={row['efSearch']}",
                (float(row["recall_at_k"]), float(row["qps"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(f"Recall@{k} (fixed R={fixed_r})")
        axis.set_xlabel(f"Recall@{k}")
        axis.set_ylabel("QPS (T=1, log scale)")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle(
        "NQ Qwen-1024 — optimized TurboQuant fixed-R, inter-query overlap\n"
        "efSearch sweep; 2,452 held-out queries; median of 3 rounds",
        y=1.03,
    )
    figure.tight_layout()
    root = (
        REPO
        / "experiment/figures/nq_qwen1024_turboquant_fixed_recall_k30_50_100_overlap_t1"
    )
    root.mkdir(parents=True, exist_ok=True)
    stem = root / "nq_qwen1024_turboquant_fixed_recall_k30_50_100_overlap_t1"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"WROTE {stem}.png/.pdf")


def plot_msmarco() -> None:
    tq_source = (
        REPO
        / "experiment/turboquant_calibrated_target099_msmarco_optimized_inline_pair_t1/summary_t1.csv"
    )
    adc_source = (
        REPO
        / "experiment/msmarco_qwen1024_calibrated_adcbq_optimized_pair_t1/summary_t1.csv"
    )
    tq_rows = sorted(read_csv(tq_source), key=lambda row: int(row["efSearch"]))
    adc_rows = sorted(read_csv(adc_source), key=lambda row: int(row["efSearch"]))
    styles = (
        ("TurboQuant calibrated", tq_rows, "#8c564b", "D"),
        ("ADC-BQ calibrated", adc_rows, "#1f77b4", "o"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    recall_axis, reads_axis = axes
    for label, rows, color, marker in styles:
        recall_axis.plot(
            [float(row["recall_at_10"]) for row in rows],
            [float(row["qps"]) for row in rows],
            label=label,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.9,
        )
        reads_axis.plot(
            [int(row["efSearch"]) for row in rows],
            [float(row["rerank_count_mean"]) for row in rows],
            label=label,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.9,
        )
    for ef in (80, 144, 320, 1024):
        row = next(row for row in tq_rows if int(row["efSearch"]) == ef)
        recall_axis.annotate(
            f"ef={ef}",
            (float(row["recall_at_10"]), float(row["qps"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    recall_axis.axvline(0.99, color="0.35", linestyle="--", linewidth=1.1)
    recall_axis.set_xlabel("Recall@10")
    recall_axis.set_ylabel("QPS (T=1, log scale)")
    recall_axis.set_yscale("log")
    recall_axis.set_title("Recall–QPS")
    recall_axis.grid(True, which="both", alpha=0.25)
    recall_axis.legend(frameon=False)
    reads_axis.set_xlabel("efSearch")
    reads_axis.set_ylabel("Mean FP32 rerank reads/query")
    reads_axis.set_xscale("log", base=2)
    reads_axis.set_title("Calibration-selected rerank count")
    reads_axis.grid(True, which="both", alpha=0.25)
    reads_axis.legend(frameon=False)
    figure.suptitle(
        "MS MARCO Qwen-1024 — per-ef target-0.99 calibration\n"
        "optimized inline search; inter-query overlap; 5,980 test queries; T=1, 3 rounds",
        y=1.02,
    )
    figure.tight_layout()
    root = (
        REPO
        / "experiment/figures/msmarco_qwen1024_calibrated_adcbq_vs_turboquant_optimized_overlap_t1"
    )
    root.mkdir(parents=True, exist_ok=True)
    stem = root / "msmarco_qwen1024_calibrated_adcbq_vs_turboquant_optimized_overlap_t1"
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"WROTE {stem}.png/.pdf")


if __name__ == "__main__":
    plot_nq()
    plot_msmarco()
