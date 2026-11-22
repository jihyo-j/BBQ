#!/usr/bin/env python3
"""Combine the new 4 KiB ADC-BQ rerun with QD16 8/16 KiB reruns."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RERUN_4K = (
    REPO
    / "experiment/msmarco_adcbq_calibrated_4k_allratios_ef256_rerun2/"
    "page4k_memory_sweep_ef256_allratios_rerun_adjusted.csv"
)
RERUN_8K = (
    REPO
    / "experiment/msmarco_page8k_ef256_qps_rerun_alllevels_"
    "batchcache_allhitfast_async_qd16/qps_rerun_summary.csv"
)
RERUN_16K = (
    REPO
    / "experiment/msmarco_page16k_ef256_qps_rerun_alllevels_"
    "batchcache_allhitfast_async_qd16/qps_rerun_summary.csv"
)
OUTPUT_ROOT = (
    REPO
    / "experiment/msmarco_adcbq_calibrated_4k_qd48_with_8k16k_qd16_reruns"
)
FIGURE_ROOT = (
    REPO
    / "experiment/figures/"
    "msmarco_adcbq_calibrated_4k_qd48_with_8k16k_qd16_reruns"
)
OUTPUT_CSV = (
    OUTPUT_ROOT / "msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.csv"
)
OUTPUT_PNG = (
    FIGURE_ROOT / "msmarco_openai1536_adcbq_calibrated_4k8k16k_ef256.png"
)
OUTPUT_PDF = OUTPUT_PNG.with_suffix(".pdf")
METHOD = "ADC-BQ calibrated"
EF = 256
PAGES = (4096, 8192, 16384)
LAYOUTS = ("baseline", "coloc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def selected(rows: list[dict[str, str]], page: int) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row["method"] == METHOD
        and int(row["efSearch"]) == EF
        and int(row["page_bytes"]) == page
    ]


def compact_row(row: dict[str, str], page: int, source: Path) -> dict[str, object]:
    fresh_process = page == 4096
    return {
        "dataset": row["dataset"],
        "corpus": row["corpus"],
        "model": row["model"],
        "method": row["method"],
        "page_bytes": page,
        "page_kib": page // 1024,
        "layout": row["layout"],
        "bq_ram_ratio_target": float(row["bq_ram_ratio_target"]),
        "bq_ram_budget_mib": int(row["bq_ram_budget_mib"]),
        "page_cache_mib": int(row["page_cache_mib"]),
        "efSearch": int(row["efSearch"]),
        "threads": int(row["threads"]),
        "graph_async_qd": 48,
        "batch_cache": False,
        "calibration_queries": int(row["calibration_queries"]),
        "test_queries": int(row["test_queries"]),
        "calibration_reference_ef": int(row["calibration_reference_ef"]),
        "calibration_tau": float(row["calibration_tau"]),
        "recall_at_10": float(row["recall_at_10"]),
        "qps": float(row["qps"]),
        "qps_stdev": float(row["qps_stdev"]) if fresh_process else "",
        "independent_measured_processes": 3 if fresh_process else 1,
        "rounds_per_process": 1 if fresh_process else 3,
        "discarded_warmup_processes": 1 if fresh_process else 0,
        "measurement_protocol": (
            "fresh_process_trials_1_to_3_aggregate_qps_from_mean_seconds"
            if fresh_process
            else "prior_sweep_single_process_3_internal_rounds"
        ),
        "latency_ms_batch_average": float(row["latency_ms_batch_average"]),
        "graph_page_hit_ratio": float(row["graph_page_hit_ratio"]),
        "graph_page_accesses_per_query": float(
            row["graph_page_accesses_per_query"]
        ),
        "graph_physical_pages_per_query": float(
            row["graph_physical_pages_per_query"]
        ),
        "graph_physical_4k_reads_per_query": float(
            row["graph_physical_4k_reads_per_query"]
        ),
        "fp_vectors_mean": float(row["fp_vectors_mean"]),
        "fp_physical_4k_reads_per_query": float(
            row["fp_physical_4k_reads_per_query"]
        ),
        "total_physical_4k_reads_per_query": float(
            row["total_physical_4k_reads_per_query"]
        ),
        "source_csv": str(source),
    }


def compact_qd16_row(
    row: dict[str, str], page: int, source: Path
) -> dict[str, object]:
    ratio = float(row["bq_ram_ratio_target"])
    qps = float(row["qps_rerun_mean"])
    graph_reads = float(row["graph_physical_4k_reads_per_query"])
    total_reads = float(row["total_physical_4k_reads_per_query"])
    return {
        "dataset": "msmarco_openai1536",
        "corpus": "MSMARCO",
        "model": "OpenAI-1536",
        "method": "ADC-BQ calibrated",
        "page_bytes": page,
        "page_kib": page // 1024,
        "layout": row["layout"],
        "bq_ram_ratio_target": ratio,
        "bq_ram_budget_mib": {0.10: 46, 0.25: 113, 0.50: 226, 0.75: 339}[ratio],
        "page_cache_mib": {0.10: 46, 0.25: 113, 0.50: 226, 0.75: 339}[ratio],
        "efSearch": int(row["efSearch"]),
        "threads": int(row["threads"]),
        "graph_async_qd": 16,
        "batch_cache": True,
        "calibration_queries": 1000,
        "test_queries": 5980,
        "calibration_reference_ef": 256,
        "calibration_tau": 0.9290738475071337,
        "recall_at_10": float(row["recall_at_10"]),
        "qps": qps,
        "qps_stdev": float(row["qps_rerun_stdev"]),
        "independent_measured_processes": int(row["measured_trials"]),
        "rounds_per_process": 1,
        "discarded_warmup_processes": 1,
        "measurement_protocol": (
            "fresh_process_trials_1_to_3_arithmetic_qps_mean_"
            "batchcache_allhitfast_async_qd16"
        ),
        "latency_ms_batch_average": 1000.0 / qps,
        "graph_page_hit_ratio": float(row["graph_page_hit_ratio"]),
        "graph_page_accesses_per_query": "",
        "graph_physical_pages_per_query": graph_reads / (page / 4096),
        "graph_physical_4k_reads_per_query": graph_reads,
        "fp_vectors_mean": float(row["fp_vectors_mean"]),
        "fp_physical_4k_reads_per_query": total_reads - graph_reads,
        "total_physical_4k_reads_per_query": total_reads,
        "source_csv": str(source),
    }


def combine() -> list[dict[str, object]]:
    rerun = read_csv(RERUN_4K)
    rows: list[dict[str, object]] = []
    group = selected(rerun, 4096)
    if len(group) != 8:
        raise RuntimeError(f"expected 8 rerun rows for page=4096, found {len(group)}")
    rows.extend(compact_row(row, 4096, RERUN_4K) for row in group)
    for page, source in ((8192, RERUN_8K), (16384, RERUN_16K)):
        group = [
            row
            for row in read_csv(source)
            if row["method"].startswith("ADC-BQ calibrated")
        ]
        if len(group) != 8:
            raise RuntimeError(f"expected 8 QD16 rows for page={page}, found {len(group)}")
        rows.extend(compact_qd16_row(row, page, source) for row in group)
    rows.sort(
        key=lambda row: (
            int(row["page_bytes"]),
            float(row["bq_ram_ratio_target"]),
            LAYOUTS.index(str(row["layout"])),
        )
    )
    return rows


def plot(rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#0072B2", "coloc": "#D55E00"}
    markers = {"baseline": "s", "coloc": "o"}
    linestyles = {"baseline": "--", "coloc": "-"}
    labels = {"baseline": "Non-colocation", "coloc": "Colocation"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharex=True, sharey="row")

    for column, page in enumerate(PAGES):
        for layout in LAYOUTS:
            group = sorted(
                (
                    row
                    for row in rows
                    if int(row["page_bytes"]) == page and row["layout"] == layout
                ),
                key=lambda row: float(row["bq_ram_ratio_target"]),
            )
            x = [float(row["bq_ram_ratio_target"]) * 100 for row in group]
            qps = [float(row["qps"]) for row in group]
            reads = [float(row["total_physical_4k_reads_per_query"]) for row in group]
            axes[0, column].plot(
                x,
                qps,
                color=colors[layout],
                marker=markers[layout],
                linestyle=linestyles[layout],
                linewidth=2,
                label=labels[layout],
            )
            axes[1, column].plot(
                x,
                reads,
                color=colors[layout],
                marker=markers[layout],
                linestyle=linestyles[layout],
                linewidth=2,
                label=labels[layout],
            )
            axes[0, column].errorbar(
                x,
                qps,
                yerr=[float(row["qps_stdev"]) for row in group],
                fmt="none",
                ecolor=colors[layout],
                elinewidth=1.5,
                capsize=4,
                alpha=0.8,
            )

        axes[0, column].set_title(f"{page // 1024} KiB pages")
        axes[1, column].set_xlabel("BQ graph/code RAM budget (%)")
        for row in range(2):
            axes[row, column].grid(True, alpha=0.25)
            axes[row, column].set_xticks([10, 25, 50, 75])

    axes[0, 0].set_ylabel("QPS")
    axes[1, 0].set_ylabel("Physical 4 KiB-equivalent reads/query")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "MSMARCO OpenAI-1536 calibrated ADC-BQ page-size comparison\n"
        "efSearch=256, T=24, 5,980 held-out queries",
        y=0.985,
    )
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.925,
        "4 KiB: async QD48; 8/16 KiB: batch-cache all-hit-fast async QD16; "
        "all points are 3 fresh-process means (error bars = SD)",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=190, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = combine()
    expected = len(PAGES) * len(LAYOUTS) * 4
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} combined rows, found {len(rows)}")
    for page in PAGES:
        for layout in LAYOUTS:
            group = [
                row
                for row in rows
                if int(row["page_bytes"]) == page and row["layout"] == layout
            ]
            if [float(row["bq_ram_ratio_target"]) for row in group] != [
                0.10,
                0.25,
                0.50,
                0.75,
            ]:
                raise RuntimeError(f"incomplete ratio sweep: page={page} layout={layout}")
            if not all(math.isclose(float(row["recall_at_10"]), 0.985318) for row in group):
                raise RuntimeError(f"unexpected recall: page={page} layout={layout}")
    write_csv(OUTPUT_CSV, rows)
    plot(rows)
    print(f"CSV {OUTPUT_CSV}")
    print(f"PNG {OUTPUT_PNG}")
    print(f"PDF {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
