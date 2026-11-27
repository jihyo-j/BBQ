#!/usr/bin/env python3
"""Make the small HNSW-vs-IVF candidate-generation ablation.

This script intentionally compares candidate-generation work rather than QPS.
Both methods use the same 1-bit ADC representation and the same target-0.99
per-search-width calibration protocol.  HNSW work is ndis_mean (upper + base
layer distance evaluations); IVF work is mean_scanned_codes.  FP32 refinement
evaluations are reported separately and are not included in either count.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

HNSW_RESULT_ROOT = (
    REPO
    / "experiment/calibrated_adcbq_hnsw_ssd_vs_fp_hnsw_ram_t1"
    / "clean_rerun_20261110_cpu20/msmarco_qwen1024/adcbq_ssd_per_ef_target099"
)
HNSW_STATS_ROOT = (
    REPO
    / "experiment/msmarco_qwen1024_calibrated_adcbq_optimized_pair_t1"
    / "raw/msmarco_qwen1024"
)
IVF_ROOT = (
    REPO
    / "experiment/calibrated_adcbq_ivf_vs_rabitq_ssd"
    / "msmarco_qwen1024/adcbq"
)

HNSW_EFS = [128, 256, 512, 1024]
IVF_NPROBES = [768, 896, 977]


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def collect() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ef in HNSW_EFS:
        result_path = HNSW_RESULT_ROOT / f"ef{ef}.csv"
        stats_path = HNSW_STATS_ROOT / f"ef{ef}.search_stats.csv"
        result = read_one(result_path)
        stats = read_one(stats_path)
        assert result["nq"] == stats["nq"] == "5980"
        assert result["efSearch"] == stats["efSearch"] == str(ef)
        assert abs(float(result["recall_at_k"]) - float(stats["recall_at_k"])) < 1e-9
        rows.append(
            {
                "dataset": "msmarco_qwen1024_1m",
                "method": "HNSW",
                "parameter": "efSearch",
                "parameter_value": ef,
                "nlist": "",
                "recall_at_10": float(result["recall_at_k"]),
                "onebit_distance_evals_per_query": float(stats["ndis_mean"]),
                "fp32_refinements_per_query": float(
                    result["adaptive_rerank_count_mean"]
                ),
                "queries": int(result["nq"]),
                "threads": int(result["search_threads"]),
                "calibration_target": 0.99,
                "calibration_status": "target_reachable",
                "source_result": result_path.relative_to(REPO),
                "source_work": stats_path.relative_to(REPO),
            }
        )

    for nprobe in IVF_NPROBES:
        path = IVF_ROOT / (
            f"nprobe{nprobe}.calibrated.int16_pshufb_tau_direct_"
            "coarseopt_interquery_v6_fairmain.t1.numa0.csv"
        )
        result = read_one(path)
        assert result["nq"] == "5980"
        assert result["nprobe"] == str(nprobe)
        rows.append(
            {
                "dataset": "msmarco_qwen1024_1m",
                "method": "IVF",
                "parameter": "nprobe",
                "parameter_value": nprobe,
                "nlist": int(result["nlist"]),
                "recall_at_10": float(result["recall_at_k"]),
                "onebit_distance_evals_per_query": float(
                    result["mean_scanned_codes"]
                ),
                "fp32_refinements_per_query": float(
                    result["adaptive_rerank_count_mean"]
                ),
                "queries": int(result["nq"]),
                "threads": 1,
                "calibration_target": 0.99,
                "calibration_status": "target_reachable",
                "source_result": path.relative_to(REPO),
                "source_work": path.relative_to(REPO),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    with (OUT / "candidate_generation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first_at_99 = []
    for method in ("HNSW", "IVF"):
        candidates = [
            row
            for row in rows
            if row["method"] == method and float(row["recall_at_10"]) >= 0.99
        ]
        first_at_99.append(min(candidates, key=lambda row: int(row["parameter_value"])))

    hnsw, ivf = first_at_99
    ratio = float(ivf["onebit_distance_evals_per_query"]) / float(
        hnsw["onebit_distance_evals_per_query"]
    )
    summary_fields = [
        "method",
        "parameter",
        "parameter_value",
        "recall_at_10",
        "onebit_distance_evals_per_query",
        "fp32_refinements_per_query",
        "ivf_over_hnsw_eval_ratio",
    ]
    with (OUT / "first_measured_point_at_recall099.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in first_at_99:
            item = {key: row[key] for key in summary_fields[:-1]}
            item["ivf_over_hnsw_eval_ratio"] = ratio
            writer.writerow(item)


def plot(rows: list[dict[str, object]]) -> None:
    styles = {
        "HNSW": dict(color="#1677b8", marker="o"),
        "IVF": dict(color="#d95f02", marker="s"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))

    for method in ("HNSW", "IVF"):
        method_rows = [row for row in rows if row["method"] == method]
        style = styles[method]
        axes[0].plot(
            [row["recall_at_10"] for row in method_rows],
            [row["onebit_distance_evals_per_query"] for row in method_rows],
            linewidth=1.7,
            markersize=5,
            label=method,
            **style,
        )
        axes[1].plot(
            [row["parameter_value"] for row in method_rows],
            [row["recall_at_10"] for row in method_rows],
            linewidth=1.7,
            markersize=5,
            label=("HNSW ($efSearch$)" if method == "HNSW" else "IVF ($nprobe$)"),
            **style,
        )

    axes[0].set_xlabel("Recall@10")
    axes[0].set_ylabel("1-bit distance evaluations / query")
    axes[0].set_yscale("log")
    axes[0].axvline(0.99, color="0.55", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("Search-width parameter")
    axes[1].set_ylabel("Recall@10")
    axes[1].axhline(0.99, color="0.55", linestyle="--", linewidth=1)
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.grid(True, alpha=0.22, linewidth=0.6)
        axis.tick_params(labelsize=8)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "hnsw_vs_ivf_candidate_generation.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / "hnsw_vs_ivf_candidate_generation.png", dpi=220, bbox_inches="tight"
    )


def main() -> None:
    rows = collect()
    write_csv(rows)
    plot(rows)


if __name__ == "__main__":
    main()
