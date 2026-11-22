#!/usr/bin/env python3
"""Plot latest per-ef target-0.99 ADC-BQ vs calibrated TurboQuant on NQ."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1]
ADC = (
    REPO
    / "experiment/nq_qwen_multidim_calibrated_adcbq_tq_t1/"
    "adcbq_pool_top10_target99_per_ef_t1.csv"
)
TQ = (
    REPO
    / "experiment/turboquant_calibrated_target099_main4_t1/summary_t1.csv"
)
OUT = REPO / "experiment/figures/nq_qwen1024_calibrated_adcbq_vs_tq_t1"
STEM = OUT / "nq_qwen1024_calibrated_adcbq_vs_turboquant_recall_qps_t1"
ADC_CURRENT = (
    REPO / "experiment/nq_qwen1024_calibrated_adcbq_current_load_t1/summary_t1.csv"
)
OUT_CURRENT = (
    REPO / "experiment/figures/"
    "nq_qwen1024_calibrated_adcbq_vs_tq_current_load_t1"
)
TQ_OPTIMIZED = (
    REPO
    / "experiment/turboquant_calibrated_target099_nq_optimized_inline_t1/"
    "summary_t1.csv"
)
ADC_OPTIMIZED = (
    REPO
    / "experiment/nq_qwen1024_calibrated_adcbq_optimized_pair_t1/summary_t1.csv"
)
OUT_OPTIMIZED = (
    REPO
    / "experiment/figures/"
    "nq_qwen1024_calibrated_adcbq_vs_tq_optimized_inline_t1"
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-load", action="store_true")
    parser.add_argument("--optimized-inline", action="store_true")
    args = parser.parse_args()
    if args.current_load and args.optimized_inline:
        parser.error("choose only one of --current-load and --optimized-inline")
    paired = args.current_load or args.optimized_inline
    adc_source = (
        ADC_OPTIMIZED
        if args.optimized_inline
        else (ADC_CURRENT if args.current_load else ADC)
    )
    tq_source = TQ_OPTIMIZED if args.optimized_inline else TQ
    output_root = (
        OUT_OPTIMIZED
        if args.optimized_inline
        else (OUT_CURRENT if args.current_load else OUT)
    )
    stem = output_root / (
        "nq_qwen1024_calibrated_adcbq_vs_turboquant_"
        + (
            "optimized_inline_"
            if args.optimized_inline
            else ("current_load_" if args.current_load else "")
        )
        + "recall_qps_t1"
    )
    adc = [
        row
        for row in rows(adc_source)
        if row["dataset"] == "nq_qwen06_1024"
        and row["method_key"] == "adc_bq_calibrated"
    ]
    tq = [
        row
        for row in rows(tq_source)
        if row["dataset"] == "nq_qwen06_1024"
        and row["method_key"] == "turboquant_tq1_calibrated"
    ]
    adc.sort(
        key=lambda row: int(
            row["efSearch"] if paired else row["search_parameter_value"]
        )
    )
    tq.sort(key=lambda row: int(row["efSearch"]))
    expected_adc = 20 if paired else 5
    if len(adc) != expected_adc:
        raise RuntimeError(
            f"expected {expected_adc} ADC-BQ points, found {len(adc)}"
        )
    if len(tq) != 20:
        raise RuntimeError(f"expected twenty calibrated TQ points, found {len(tq)}")

    combined: list[dict[str, object]] = []
    for row in adc:
        ef = int(row["efSearch"] if paired else row["search_parameter_value"])
        rerank_mean = float(
            row["rerank_count_mean"]
            if paired
            else row["fp_vectors_per_query"]
        )
        combined.append(
            {
                "method_key": "adc_bq_calibrated",
                "method": "ADC-BQ calibrated",
                "efSearch": ef,
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "rerank_count_mean": rerank_mean,
                "rerank_count_p95": (
                    float(row["rerank_count_p95"]) if paired else ""
                ),
                "tau": float(row["tau"]),
                "calibration_queries": int(row["calibration_queries"]),
                "test_queries": int(row["test_queries"]),
                "rounds": int(row["rounds"]),
                "io_engine": row["io_engine"],
                "first_stage_search_impl": row.get("first_stage_search_impl", ""),
                "first_stage_distance_evaluations_per_query": row.get(
                    "first_stage_distance_evaluations_per_query", ""
                ),
                "source_csv": str(adc_source.relative_to(REPO)),
            }
        )
    for row in tq:
        combined.append(
            {
                "method_key": "turboquant_tq1_calibrated",
                "method": "TurboQuant TQ1+ calibrated",
                "efSearch": int(row["efSearch"]),
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "rerank_count_mean": float(row["rerank_count_mean"]),
                "rerank_count_p95": float(row["rerank_count_p95"]),
                "tau": float(row["tau"]),
                "calibration_queries": int(row["calibration_queries"]),
                "test_queries": int(row["test_queries"]),
                "rounds": int(row["rounds"]),
                "io_engine": row["io_engine"],
                "first_stage_search_impl": row.get("first_stage_search_impl", ""),
                "first_stage_distance_evaluations_per_query": row.get(
                    "first_stage_distance_evaluations_per_query", ""
                ),
                "source_csv": str(tq_source.relative_to(REPO)),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 160,
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.18)
    ax.plot(
        [float(row["recall_at_10"]) for row in adc],
        [float(row["qps"]) for row in adc],
        color="#0072B2",
        marker="s",
        markersize=6.5,
        linewidth=2.2,
        label="ADC-BQ calibrated (per-ef)",
        zorder=3,
    )
    ax.plot(
        [float(row["recall_at_10"]) for row in tq],
        [float(row["qps"]) for row in tq],
        color="#D55E00",
        marker="o",
        markersize=4.8,
        linewidth=2.0,
        label=(
            "TurboQuant TQ1+ optimized (per-ef)"
            if args.optimized_inline
            else "TurboQuant TQ1+ calibrated (per-ef)"
        ),
        zorder=4,
    )
    ax.axvline(0.99, color="#555555", linestyle="--", linewidth=1.2, alpha=0.85)
    ax.text(
        0.99015,
        0.965,
        "target 0.99",
        rotation=90,
        transform=ax.get_xaxis_transform(),
        color="#555555",
        va="top",
        fontsize=9,
    )

    for row in adc:
        ef = int(row["efSearch"] if paired else row["search_parameter_value"])
        ax.annotate(
            f"{ef}",
            (float(row["recall_at_10"]), float(row["qps"])),
            xytext=(5, 5),
            textcoords="offset points",
            color="#0072B2",
            fontsize=7.5 if paired else 8,
        )
    for row in tq:
        ef = int(row["efSearch"])
        if ef not in {80, 120, 144, 256, 512, 1024}:
            continue
        ax.annotate(
            f"{ef}",
            (float(row["recall_at_10"]), float(row["qps"])),
            xytext=(5, -11 if ef in {144, 256} else 5),
            textcoords="offset points",
            color="#D55E00",
            fontsize=8,
        )

    all_recall = [float(row["recall_at_10"]) for row in adc + tq]
    ax.set_xlim(min(all_recall) - 0.001, 1.0003)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Queries per second (T1)")
    ax.set_title(
        "NQ Qwen-1024: calibrated ADC-BQ vs TurboQuant"
        + (
            " (optimized inline search)"
            if args.optimized_inline
            else (" (same server load)" if args.current_load else "")
        )
    )
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(loc="upper right", frameon=True)
    fig.text(
        0.5,
        0.025,
        "target=0.99 per ef; 1,000 calibration + 2,452 held-out queries; T1, 3 rounds"
        + ("; paired by ef under the same server load" if paired else ""),
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#555555",
    )
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
