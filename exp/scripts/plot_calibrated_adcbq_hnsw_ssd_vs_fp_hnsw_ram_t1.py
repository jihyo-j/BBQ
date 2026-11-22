#!/usr/bin/env python3
"""Plot calibrated ADC-BQ HNSW + SSD rerank against in-memory FP32 HNSW."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ADC_CSV = (
    ROOT
    / "experiment/adcbq_calibrated_threshold_overlap/"
    "comparison_single_tau_ef256_t1.csv"
)
FP_ROOT = (
    ROOT
    / "experiment/adcbq_vs_fp_hnsw_fullvec_ram/"
    "clean_t1_cpu20_20261109"
)
OUT = ROOT / "experiment/calibrated_adcbq_hnsw_ssd_vs_fp_hnsw_ram_t1"

DATASETS = {
    "msmarco_qwen1024": "MSMARCO Qwen-1024 (1M)",
    "dbpedia_qwen1024": "DBpedia Qwen-1024 (1M)",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    adc_all = read_rows(ADC_CSV)
    combined: list[dict[str, object]] = []

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)
    for axis, (dataset, title) in zip(axes, DATASETS.items()):
        adc = sorted(
            (row for row in adc_all if row["dataset"] == dataset),
            key=lambda row: int(row["efSearch"]),
        )
        fp_path = FP_ROOT / dataset / "fp_hnsw_t1.csv"
        fp = sorted(read_rows(fp_path), key=lambda row: int(row["efSearch"]))

        axis.plot(
            [float(row["recall_at_10"]) for row in adc],
            [float(row["qps"]) for row in adc],
            marker="o",
            linewidth=2.3,
            markersize=6.5,
            color="#d62728",
            label="Calibrated ADC-BQ HNSW + SSD FP32 rerank",
        )
        axis.plot(
            [float(row["recall_at_k"]) for row in fp],
            [float(row["qps"]) for row in fp],
            marker="s",
            linewidth=2.3,
            markersize=5.5,
            color="#1f77b4",
            label="Full FP32 HNSW (in-memory)",
        )

        for row in adc:
            combined.append(
                {
                    "dataset": dataset,
                    "method": "Calibrated ADC-BQ HNSW + SSD FP32 rerank",
                    "efSearch": int(row["efSearch"]),
                    "threads": int(row["search_threads"]),
                    "queries": int(row["test_queries"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "pages_4k_per_query": float(row["pages_4k_mean"]),
                    "full_vector_storage": "SSD O_DIRECT",
                    "source_csv": row["result_path"],
                }
            )
        for row in fp:
            combined.append(
                {
                    "dataset": dataset,
                    "method": "Full FP32 HNSW (in-memory)",
                    "efSearch": int(row["efSearch"]),
                    "threads": int(row["threads"]),
                    "queries": int(row["nq"]),
                    "recall_at_10": float(row["recall_at_k"]),
                    "qps": float(row["qps"]),
                    "pages_4k_per_query": 0.0,
                    "full_vector_storage": row["full_vector_storage"],
                    "source_csv": str(fp_path),
                }
            )

        axis.set_title(title)
        axis.set_xlabel("Recall@10")
        axis.set_xlim(0.96, 1.0)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.28)
        axis.legend(fontsize=9)

    axes[0].set_ylabel("End-to-end QPS (log scale)")
    fig.suptitle(
        "Calibrated ADC-BQ HNSW SSD vs in-memory FP32 HNSW — T=1",
        fontsize=16,
    )
    fig.text(
        0.5,
        0.005,
        (
            "Same FP-built HNSW topology (M=24, efConstruction=200) and held-out "
            "query split. ADC-BQ exact rerank: io_uring O_DIRECT SSD; FP-HNSW: "
            "process-owned FP32 vectors in RAM."
        ),
        ha="center",
        fontsize=9.5,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "recall_qps_t1_from_recall096.png"
    pdf = OUT / "recall_qps_t1_from_recall096.pdf"
    csv_path = OUT / "recall_qps_t1_from_recall096.csv"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    fields = list(combined[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    print(png)
    print(pdf)
    print(csv_path)


if __name__ == "__main__":
    main()
