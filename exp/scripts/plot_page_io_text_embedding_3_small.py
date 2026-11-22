#!/usr/bin/env python3
"""Plot MS MARCO page I/O with the original FAISS-HNSW TurboQuant path.

TurboQuant uses resident TQ1+ codes while traversing the same FAISS HNSW
graph family used by the compressed in-memory baselines, followed by one
fixed-R O_DIRECT io_uring batch against the FP32 sidecar.  This is distinct
from the later native-Qdrant HNSW experiment.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
METHOD_ORDER = ("adcbq", "turboquant", "diskann_rabitq", "diskann_classic")
FAISS_TQ_ROOT = REPO / "experiment/diskveclab_full_matrix/tq_rerank_pareto_uring/raw"
FAISS_TQ_OPENAI = (
    REPO
    / "experiment/openai_beir_11methods_t1/four_methods"
    / "sieun_msmarco_openai1536/tq_selected_r_heldout_t1.csv"
)
STYLES = {
    "adcbq": ("BBQ", "#0072B2", "s", "-"),
    "turboquant": ("TurboQuant-1bit", "#E69F00", "p", "-."),
    "diskann_rabitq": ("DiskANN + RaBitQ-1bit", "#8C564B", "v", "--"),
    "diskann_classic": ("DiskANN", "#CC79A7", "D", "-"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect(campaign: Path, reuse_existing: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    old = REPO / "experiment/diskveclab_full_matrix/page_io_msmarco_4methods/normalized_page_io.csv"
    for row in read_csv(old):
        if row["dataset"] != "msmarco_qwen1024" or row["method_key"] != "adcbq":
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "model": "Qwen-1024",
                "method_key": row["method_key"],
                "method": row["method"],
                "parameter": row["search_parameter"],
                "parameter_value": int(row["search_parameter_value"]),
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "pages_per_query": float(row["total_4k_pages_per_query_mean"]),
                "test_queries": int(row["test_queries"]),
                "search_threads": 1,
                "source_csv": str(old.relative_to(REPO)),
            }
        )

    qwen_disk_root = REPO / "experiment/onebit_disk_baselines_t1/diskann/msmarco_qwen1024"
    for directory, key in (
        ("diskann_classic", "diskann_classic"),
        ("diskann_rabitq_1bit_adaptive_fp", "diskann_rabitq"),
    ):
        path = qwen_disk_root / directory / "search_t1.csv"
        for row in read_csv(path):
            rows.append(
                {
                    "dataset": "msmarco_qwen1024",
                    "model": "Qwen-1024",
                    "method_key": key,
                    "method": row["method"],
                    "parameter": "Lsearch",
                    "parameter_value": int(row["Lsearch"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "pages_per_query": float(row["total_4k_pages_per_query_mean"]),
                    "test_queries": int(row["nq"]),
                    "search_threads": int(row["search_threads"]),
                    "source_csv": str(path.relative_to(REPO)),
                }
            )

    adc = (
        REPO / "experiment/bbq_calibrated_io_modes_t1/sieun_msmarco_openai1536/summary_t1.csv"
        if reuse_existing
        else campaign / "adc_io_modes/sieun_msmarco_openai1536/summary_t1.csv"
    )
    for row in read_csv(adc):
        if row["method_key"] != "async_batch_overlap":
            continue
        rows.append(
            {
                "dataset": "sieun_msmarco_openai1536",
                "model": "OpenAI text-embedding-3-small (1536D)",
                "method_key": "adcbq",
                "method": "BBQ calibrated · RAM graph + adaptive FP",
                "parameter": "efSearch",
                "parameter_value": int(row["efSearch"]),
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "pages_per_query": float(row["fp_submitted_4k_pages_per_query"]),
                "test_queries": int(row["test_queries"]),
                "search_threads": int(row["search_threads"]),
                "source_csv": str(adc.relative_to(REPO)),
            }
        )

    # Use the established FAISS-HNSW + TurboQuant scorer experiment, not the
    # later native-Qdrant graph.  R was selected once per dataset from the
    # joint (efSearch, R) calibration Pareto sweep and is fixed on the test set.
    for output_dataset, tq, rerank_n in (
        ("sieun_msmarco_openai1536", FAISS_TQ_OPENAI, 60),
        ("msmarco_qwen1024", FAISS_TQ_ROOT / "msmarco_qwen1024/qdranttq1.r60.csv", 60),
    ):
        model = "OpenAI text-embedding-3-small (1536D)" if output_dataset.startswith("sieun_") else "Qwen-1024"
        for row in read_csv(tq):
            if int(row["rerank_n"]) != rerank_n:
                raise RuntimeError(f"unexpected TurboQuant R in {tq}: {row}")
            if output_dataset.startswith("sieun_"):
                pages_per_query = float(row["fp_4k_pages_per_query"])
                test_queries = int(row["test_queries"])
                search_threads = int(row["search_threads"])
            else:
                # Count submitted byte volume in 4-KiB equivalents.  A
                # row-aligned 1536D FP32 vector spans two pages, whereas
                # a 1024D vector spans one.
                pages_per_query = float(row["fp_read_bytes_per_query"]) / 4096.0
                test_queries = int(row["nq"])
                search_threads = int(row["threads"])
            rows.append(
                {
                    "dataset": output_dataset,
                    "model": model,
                    "method_key": "turboquant",
                    "method": f"TurboQuant-1bit · FAISS HNSW + fixed FP@{rerank_n}",
                    "parameter": "efSearch",
                    "parameter_value": int(row["efSearch"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "pages_per_query": pages_per_query,
                    "test_queries": test_queries,
                    "search_threads": search_threads,
                    "source_csv": str(tq.relative_to(REPO)),
                }
            )

    disk_root = (
        REPO / "experiment/onebit_disk_baselines_t1/diskann/sieun_msmarco_openai1536"
        if reuse_existing
        else campaign / "disk_matrix/diskann/sieun_msmarco_openai1536"
    )
    for directory, key in (
        ("diskann_classic", "diskann_classic"),
        ("diskann_rabitq_1bit_adaptive_fp", "diskann_rabitq"),
    ):
        path = disk_root / directory / "search_t1.csv"
        for row in read_csv(path):
            rows.append(
                {
                    "dataset": "sieun_msmarco_openai1536",
                    "model": "OpenAI text-embedding-3-small (1536D)",
                    "method_key": key,
                    "method": row["method"],
                    "parameter": "Lsearch",
                    "parameter_value": int(row["Lsearch"]),
                    "recall_at_10": float(row["recall_at_10"]),
                    "qps": float(row["qps"]),
                    "pages_per_query": float(row["total_4k_pages_per_query_mean"]),
                    "test_queries": int(row["nq"]),
                    "search_threads": int(row["search_threads"]),
                    "source_csv": str(path.relative_to(REPO)),
                }
            )

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method_key"]))].append(row)
    for dataset in ("sieun_msmarco_openai1536", "msmarco_qwen1024"):
        for method in METHOD_ORDER:
            curve = grouped[(dataset, method)]
            if not curve:
                raise RuntimeError(f"missing curve for {dataset}/{method}")
            if any(int(row["test_queries"]) != 5980 for row in curve):
                raise RuntimeError(f"held-out query mismatch for {dataset}/{method}")
            if any(int(row["search_threads"]) != 1 for row in curve):
                raise RuntimeError(f"thread mismatch for {dataset}/{method}")
            if max(float(row["recall_at_10"]) for row in curve) < 0.99:
                raise RuntimeError(f"curve does not reach Recall@10=0.99: {dataset}/{method}")
    return rows


def select(rows: list[dict[str, object]], target: float) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method_key"]))].append(row)
    output: list[dict[str, object]] = []
    for dataset in ("sieun_msmarco_openai1536", "msmarco_qwen1024"):
        for method in METHOD_ORDER:
            feasible = [
                row for row in grouped[(dataset, method)]
                if float(row["recall_at_10"]) + 1e-12 >= target
            ]
            if not feasible:
                output.append(
                    {"dataset": dataset, "method_key": method, "target_recall": target, "reachable": False}
                )
                continue
            best = min(feasible, key=lambda row: (float(row["pages_per_query"]), -float(row["qps"])))
            output.append(
                {
                    "dataset": dataset,
                    "method_key": method,
                    "target_recall": target,
                    "reachable": True,
                    "parameter": best["parameter"],
                    "parameter_value": best["parameter_value"],
                    "measured_recall_at_10": best["recall_at_10"],
                    "pages_per_query": best["pages_per_query"],
                    "qps": best["qps"],
                }
            )
    return output


def plot(rows: list[dict[str, object]], output: Path) -> None:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method_key"]))].append(row)
    panels = (
        ("sieun_msmarco_openai1536", "MS MARCO · OpenAI text-embedding-3-small (1536D)"),
        ("msmarco_qwen1024", "MS MARCO · Qwen-1024"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for axis, (dataset, title) in zip(axes, panels, strict=True):
        for method in METHOD_ORDER:
            points = sorted(grouped[(dataset, method)], key=lambda row: int(row["parameter_value"]))
            label, color, marker, linestyle = STYLES[method]
            axis.plot(
                [float(row["recall_at_10"]) for row in points],
                [float(row["pages_per_query"]) for row in points],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=4.2,
                label=label,
            )
        axis.axvline(0.99, color="0.45", linestyle="--", linewidth=0.9)
        axis.set_xlim(0.97, 1.0005)
        axis.set_yscale("log")
        axis.set_xlabel("Recall@10")
        axis.set_title(title, fontsize=10.5)
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("Submitted 4-KiB page equivalents/query (log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.14, top=0.78, wspace=0.20)
    fig.savefig(output / "page_io.pdf", bbox_inches="tight")
    fig.savefig(output / "page_io.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="reuse validated T=1, three-round, held-out results already on disk",
    )
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = campaign / "figure"
    output.mkdir(parents=True, exist_ok=True)
    rows = collect(campaign, reuse_existing=args.reuse_existing)
    rows.sort(key=lambda row: (str(row["dataset"]), METHOD_ORDER.index(str(row["method_key"])), int(row["parameter_value"])))
    write_csv(output / "normalized_page_io.csv", rows)
    selected = select(rows, 0.98) + select(rows, 0.99)
    write_csv(output / "operating_points.csv", selected)
    plot(rows, output)
    print(output / "normalized_page_io.csv")
    print(output / "operating_points.csv")
    print(output / "page_io.pdf")


if __name__ == "__main__":
    main()
