#!/usr/bin/env python3
"""Measure FP- vs Hamming-built HNSW graphs under identical ADC-BQ search.

The saved payload of both graph variants is the same packed sign code.  Only
the metric used during HNSW neighbor selection differs:

* fp: normalized FP32 inner product
* binary: Hamming distance between packed sign codes

Search always uses a full-precision query against the packed database code
(ADC-BQ), without full-precision reranking.  Besides recall and throughput,
the runner records FAISS HNSW traversal counters so graph quality can be
compared at equal efSearch and at equal numbers of scored nodes.
"""

from __future__ import annotations

import argparse
import csv
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "experiment/faiss_fpgraph_packed_asym_bq_hnsw"
RESULT_DIR = REPO / "experiment/msmarco_quora_qwen1024_graph_construction_precision"
FIGURE_DIR = REPO / "experiment/figures/msmarco_quora_qwen1024_graph_construction_precision"
INDEX_DIR = REPO / "indexes/msmarco_quora_qwen1024_graph_construction_precision"

DEFAULT_EFS = (
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    512,
    640,
    768,
    1024,
)


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    train: Path
    query: Path
    gt: Path
    fp_index: Path
    binary_index: Path


DATASETS = {
    "msmarco": Dataset(
        key="msmarco",
        title="MSMARCO Qwen-1024",
        train=Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_train.fbin"
        ),
        query=Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_query.fbin"
        ),
        gt=Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_neighbors.ibin"
        ),
        fp_index=Path(
            "/home/smrc/samsung-nvme/jihyo/msmarco_qwen1024_scale_v014/"
            "indexes/faiss_adcbq/msmarco-qwen1024-1m."
            "fpgraph_pgbit_binary_hnsw.n1000000.d1024.M24.efC200.faiss"
        ),
        binary_index=Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/indexes/faiss_packed_asym_bq/"
            "msmarco-1m__vanilla-qwen06b.pgbit_binary_hnsw."
            "n1000000.d1024.M24.efC200.faiss"
        ),
    ),
    "quora": Dataset(
        key="quora",
        title="Quora Qwen-1024",
        train=Path(
            "/home/smrc/samsung-nvme/jihyo/quora_sieun_adcbq_rabitq/"
            "quora_vanilla-qw06_train.fbin"
        ),
        query=Path(
            "/home/smrc/samsung-nvme/jihyo/quora_sieun_adcbq_rabitq/"
            "quora_vanilla-qw06_query.fbin"
        ),
        gt=Path(
            "/home/smrc/samsung-nvme/jihyo/quora_sieun_adcbq_rabitq/"
            "quora_vanilla-qw06_neighbors_top100.ibin"
        ),
        fp_index=Path(
            "/home/smrc/samsung-nvme/jihyo/rabitq_v051_ssd_multidataset_indexes/"
            "quora_qwen1024/quora_qwen1024.fpgraph_adcbq."
            "d1024.M24.efC200.faiss"
        ),
        binary_index=INDEX_DIR
        / "quora_qwen1024.binary_graph_bq_storage.n522931.d1024.M24.efC200.faiss",
    ),
}


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"invalid matrix header: {path}")
    return struct.unpack("<ii", header)


def validate(dataset: Dataset) -> None:
    for path in (dataset.train, dataset.query, dataset.gt, dataset.fp_index):
        if not path.exists():
            raise FileNotFoundError(path)
    nb, dim = matrix_shape(dataset.train)
    nq, qdim = matrix_shape(dataset.query)
    ngt, gt_width = matrix_shape(dataset.gt)
    if qdim != dim or ngt != nq or gt_width < 10:
        raise ValueError(
            f"inconsistent {dataset.key}: train={nb}x{dim}, "
            f"query={nq}x{qdim}, gt={ngt}x{gt_width}"
        )
    print(
        f"DATASET {dataset.key}: nb={nb}, nq={nq}, dim={dim}, gt_width={gt_width}",
        flush=True,
    )


def raw_path(dataset: Dataset, graph_build: str) -> Path:
    return RESULT_DIR / f"{dataset.key}.{graph_build}_graph.raw.csv"


def stats_path(dataset: Dataset, graph_build: str) -> Path:
    return RESULT_DIR / f"{dataset.key}.{graph_build}_graph.stats.csv"


def complete(path: Path, expected_efs: tuple[int, ...]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return sorted(int(row["efSearch"]) for row in rows) == sorted(expected_efs)
    except (KeyError, OSError, ValueError):
        return False


def run_one(
    dataset: Dataset,
    graph_build: str,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    out = raw_path(dataset, graph_build)
    stats = stats_path(dataset, graph_build)
    if (
        complete(out, args.efs)
        and complete(stats, args.efs)
        and not args.force_search
    ):
        print(f"SKIP complete: {dataset.key}/{graph_build}", flush=True)
        return out, stats

    index = dataset.fp_index if graph_build == "fp" else dataset.binary_index
    index.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(BIN),
        "--dataset",
        f"{dataset.key}_qwen1024_graph_construction_precision",
        "--train",
        str(dataset.train),
        "--query",
        str(dataset.query),
        "--gt",
        str(dataset.gt),
        "--index",
        str(index),
        "--out",
        str(out),
        "--stats-out",
        str(stats),
        "--m",
        "24",
        "--ef-construction",
        "200",
        "--k",
        "10",
        "--max-queries",
        str(args.max_queries),
        "--rounds",
        str(args.rounds),
        "--build-threads",
        str(args.build_threads),
        "--search-threads",
        "1",
        "--adc-mode",
        "int16",
        "--lut-mode",
        "nibble",
        "--graph-build",
        graph_build,
        "--distance-mode",
        "adcbq",
        "--search-impl",
        "inline",
        "--code-layout",
        "block32",
        "--no-rerank",
        "--efs",
        *map(str, args.efs),
    ]
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    return out, stats


def measure_build_times(
    datasets: list[Dataset], args: argparse.Namespace
) -> Path:
    """Fresh-build each graph once and record construction time and size."""
    build_dir = INDEX_DIR / "fresh_build_timing"
    result_dir = RESULT_DIR / "fresh_build_timing"
    build_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, object]] = []
    for dataset in datasets:
        nb, dim = matrix_shape(dataset.train)
        for graph_build in ("fp", "binary"):
            index = build_dir / (
                f"{dataset.key}.{graph_build}_graph_bq_storage."
                f"n{nb}.d{dim}.M24.efC200.faiss"
            )
            raw = result_dir / f"{dataset.key}.{graph_build}_graph.build.csv"
            existed_before = index.exists()
            if not (index.exists() and complete(raw, (32,))):
                if index.exists() and not raw.exists():
                    raise RuntimeError(
                        f"fresh timing index exists without result CSV: {index}"
                    )
                command = [
                    str(BIN),
                    "--dataset",
                    f"{dataset.key}_qwen1024_fresh_{graph_build}_build_timing",
                    "--train",
                    str(dataset.train),
                    "--query",
                    str(dataset.query),
                    "--gt",
                    str(dataset.gt),
                    "--index",
                    str(index),
                    "--out",
                    str(raw),
                    "--m",
                    "24",
                    "--ef-construction",
                    "200",
                    "--k",
                    "10",
                    "--max-queries",
                    "1",
                    "--rounds",
                    "1",
                    "--build-threads",
                    str(args.build_threads),
                    "--search-threads",
                    "1",
                    "--adc-mode",
                    "int16",
                    "--lut-mode",
                    "nibble",
                    "--graph-build",
                    graph_build,
                    "--distance-mode",
                    "adcbq",
                    "--search-impl",
                    "inline",
                    "--code-layout",
                    "block32",
                    "--no-rerank",
                    "--efs",
                    "32",
                ]
                print("BUILD-TIME RUN " + " ".join(command), flush=True)
                started = time.monotonic()
                subprocess.run(command, cwd=REPO, check=True)
                wall_seconds = time.monotonic() - started
                wall_file = raw.with_suffix(".wall_seconds.txt")
                wall_file.write_text(f"{wall_seconds:.9f}\n")
                existed_before = False
            else:
                wall_file = raw.with_suffix(".wall_seconds.txt")
                wall_seconds = (
                    float(wall_file.read_text().strip())
                    if wall_file.exists()
                    else float("nan")
                )

            with raw.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            output_rows.append(
                {
                    "dataset": dataset.key,
                    "dataset_title": dataset.title,
                    "graph_build": graph_build,
                    "graph_build_metric": (
                        "normalized FP32 inner product"
                        if graph_build == "fp"
                        else "sign-code Hamming distance"
                    ),
                    "nb": nb,
                    "dim": dim,
                    "M": 24,
                    "efConstruction": 200,
                    "build_threads": args.build_threads,
                    "fresh_build_verified": not existed_before,
                    "index_build_seconds": row["build_or_load_seconds"],
                    "command_wall_seconds": wall_seconds,
                    "index_bytes": index.stat().st_size,
                    "bytes_per_vector": index.stat().st_size / nb,
                    "index_path": str(index),
                    "source_csv": str(raw),
                }
            )

    out = RESULT_DIR / "msmarco_quora_qwen1024_graph_construction_build_times.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"WROTE {out}", flush=True)
    return out


def read_by_ef(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as handle:
        return {int(row["efSearch"]): row for row in csv.DictReader(handle)}


def build_summary(datasets: list[Dataset], efs: tuple[int, ...]) -> tuple[Path, Path]:
    tidy_path = RESULT_DIR / "msmarco_quora_qwen1024_graph_construction_precision.csv"
    paired_path = RESULT_DIR / "msmarco_quora_qwen1024_graph_construction_precision_paired.csv"
    fields = [
        "dataset",
        "dataset_title",
        "graph_build",
        "graph_build_metric",
        "search_metric",
        "M",
        "efConstruction",
        "efSearch",
        "search_threads",
        "rounds",
        "nq",
        "nb",
        "dim",
        "recall_at_10",
        "qps",
        "ndis_mean",
        "ndis_p50",
        "ndis_p95",
        "ndis_max",
        "base_distance_eval_count_mean",
        "base_visited_count_mean",
        "base_expand_count_mean",
        "index_path",
        "raw_csv",
        "stats_csv",
    ]
    rows_out: list[dict[str, object]] = []
    paired_out: list[dict[str, object]] = []
    for dataset in datasets:
        joined: dict[str, dict[int, dict[str, object]]] = {}
        for graph_build in ("fp", "binary"):
            raw = read_by_ef(raw_path(dataset, graph_build))
            stats = read_by_ef(stats_path(dataset, graph_build))
            per_ef: dict[int, dict[str, object]] = {}
            index = dataset.fp_index if graph_build == "fp" else dataset.binary_index
            for ef in efs:
                r = raw[ef]
                s = stats[ef]
                row = {
                    "dataset": dataset.key,
                    "dataset_title": dataset.title,
                    "graph_build": graph_build,
                    "graph_build_metric": (
                        "normalized FP32 inner product"
                        if graph_build == "fp"
                        else "sign-code Hamming distance"
                    ),
                    "search_metric": "ADC-BQ: FP32 query vs packed sign code",
                    "M": 24,
                    "efConstruction": 200,
                    "efSearch": ef,
                    "search_threads": r.get("search_threads", "1"),
                    "rounds": r["rounds"],
                    "nq": r["nq"],
                    "nb": r["nb"],
                    "dim": r["dim"],
                    "recall_at_10": r["recall_at_k"],
                    "qps": r["qps"],
                    "ndis_mean": s["ndis_mean"],
                    "ndis_p50": s["ndis_p50"],
                    "ndis_p95": s["ndis_p95"],
                    "ndis_max": s["ndis_max"],
                    "base_distance_eval_count_mean": s[
                        "base_distance_eval_count_mean"
                    ],
                    "base_visited_count_mean": s["base_visited_count_mean"],
                    "base_expand_count_mean": s["base_expand_count_mean"],
                    "index_path": str(index),
                    "raw_csv": str(raw_path(dataset, graph_build)),
                    "stats_csv": str(stats_path(dataset, graph_build)),
                }
                rows_out.append(row)
                per_ef[ef] = row
            joined[graph_build] = per_ef

        for ef in efs:
            fp = joined["fp"][ef]
            binary = joined["binary"][ef]
            fp_recall = float(fp["recall_at_10"])
            binary_recall = float(binary["recall_at_10"])
            fp_ndis = float(fp["ndis_mean"])
            binary_ndis = float(binary["ndis_mean"])
            paired_out.append(
                {
                    "dataset": dataset.key,
                    "dataset_title": dataset.title,
                    "efSearch": ef,
                    "fp_recall_at_10": fp_recall,
                    "binary_recall_at_10": binary_recall,
                    "fp_minus_binary_recall_percentage_points": 100.0
                    * (fp_recall - binary_recall),
                    "fp_ndis_mean": fp_ndis,
                    "binary_ndis_mean": binary_ndis,
                    "binary_ndis_overhead_percent": 100.0
                    * (binary_ndis / fp_ndis - 1.0),
                    "fp_qps": fp["qps"],
                    "binary_qps": binary["qps"],
                }
            )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with tidy_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)
    paired_fields = list(paired_out[0])
    with paired_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader()
        writer.writerows(paired_out)
    print(f"WROTE {tidy_path}", flush=True)
    print(f"WROTE {paired_path}", flush=True)
    return tidy_path, paired_path


def plot(tidy_path: Path, datasets: list[Dataset]) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(tidy_path.open(newline="")))
    styles = {
        "fp": {
            "label": "FP32-built graph",
            "color": "#2563eb",
            "marker": "o",
        },
        "binary": {
            "label": "Hamming-built graph",
            "color": "#dc2626",
            "marker": "s",
        },
    }
    fig, axes = plt.subplots(len(datasets), 3, figsize=(15.6, 8.3), squeeze=False)
    for row_idx, dataset in enumerate(datasets):
        dataset_rows = [r for r in rows if r["dataset"] == dataset.key]
        for graph_build in ("fp", "binary"):
            group = sorted(
                (r for r in dataset_rows if r["graph_build"] == graph_build),
                key=lambda r: int(r["efSearch"]),
            )
            style = styles[graph_build]
            efs = [int(r["efSearch"]) for r in group]
            recall = [float(r["recall_at_10"]) for r in group]
            ndis = [float(r["ndis_mean"]) for r in group]
            common = dict(
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=4.5,
                label=style["label"],
            )
            axes[row_idx, 0].plot(efs, recall, **common)
            axes[row_idx, 1].plot(efs, ndis, **common)
            axes[row_idx, 2].plot(ndis, recall, **common)

        axes[row_idx, 0].set_ylabel(f"{dataset.title}\nRecall@10")
        axes[row_idx, 0].set_xscale("log", base=2)
        axes[row_idx, 1].set_xscale("log", base=2)
        axes[row_idx, 0].set_xlabel("efSearch")
        axes[row_idx, 1].set_xlabel("efSearch")
        axes[row_idx, 2].set_xlabel("Scored nodes/query (ndis)")
        axes[row_idx, 1].set_ylabel("Scored nodes/query (ndis)")
        axes[row_idx, 2].set_ylabel("Recall@10")
        for axis in axes[row_idx]:
            axis.grid(True, which="both", alpha=0.25)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)

    axes[0, 0].set_title("Matched efSearch")
    axes[0, 1].set_title("Traversal work at matched efSearch")
    axes[0, 2].set_title("Recall at matched traversal work")
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=2,
        frameon=True,
    )
    fig.suptitle(
        "HNSW graph-construction precision with identical ADC-BQ search",
        y=0.99,
        fontsize=15,
    )
    fig.text(
        0.5,
        0.012,
        "Qwen-1024; M=24, efConstruction=200; no FP reranking; T=1; "
        "3 rounds. Both indexes store the same packed sign-code payload.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.89))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / "msmarco_quora_qwen1024_graph_construction_precision.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {png}", flush=True)
    print(f"WROTE {pdf}", flush=True)
    return png, pdf


def plot_recall_qps(
    paired_path: Path, datasets: list[Dataset]
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(paired_path.open(newline="")))
    styles = {
        "fp": {
            "label": "FP32-built graph",
            "color": "#2563eb",
            "marker": "o",
            "recall": "fp_recall_at_10",
            "qps": "fp_qps",
        },
        "binary": {
            "label": "Hamming-built graph",
            "color": "#dc2626",
            "marker": "s",
            "recall": "binary_recall_at_10",
            "qps": "binary_qps",
        },
    }
    fig, axes = plt.subplots(1, len(datasets), figsize=(11.2, 4.6), squeeze=False)
    for col_idx, dataset in enumerate(datasets):
        axis = axes[0, col_idx]
        dataset_rows = sorted(
            (row for row in rows if row["dataset"] == dataset.key),
            key=lambda row: int(row["efSearch"]),
        )
        for graph_build in ("fp", "binary"):
            style = styles[graph_build]
            recall = [float(row[style["recall"]]) for row in dataset_rows]
            qps = [float(row[style["qps"]]) for row in dataset_rows]
            axis.plot(
                recall,
                qps,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=4.5,
                label=style["label"],
            )
        axis.set_title(dataset.title)
        axis.set_xlabel("Recall@10")
        axis.set_ylabel("Queries per second (QPS)")
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=True,
    )
    fig.suptitle(
        "Recall-throughput impact of HNSW graph-construction precision",
        y=0.99,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.015,
        "Qwen-1024; M=24, efConstruction=200; identical ADC-BQ search; "
        "no FP reranking; T=1; 3 rounds.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.84))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURE_DIR / "msmarco_quora_qwen1024_graph_construction_recall_qps.png"
    pdf = png.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"WROTE {png}", flush=True)
    print(f"WROTE {pdf}", flush=True)
    return png, pdf


def write_report(paired_path: Path, datasets: list[Dataset]) -> Path:
    rows = list(csv.DictReader(paired_path.open(newline="")))
    lines = [
        "# HNSW graph-construction precision",
        "",
        "Both graph variants store identical packed one-bit codes and use identical "
        "ADC-BQ search. Only the construction-time neighbor-selection metric differs.",
        "",
    ]
    for dataset in datasets:
        group = [r for r in rows if r["dataset"] == dataset.key]
        lines += [f"## {dataset.title}", ""]
        lines.append(
            "| efSearch | FP recall | Hamming recall | FP-Hamming (pp) | "
            "FP ndis | Hamming ndis | Hamming work overhead |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for row in group:
            lines.append(
                f"| {row['efSearch']} | {float(row['fp_recall_at_10']):.6f} | "
                f"{float(row['binary_recall_at_10']):.6f} | "
                f"{float(row['fp_minus_binary_recall_percentage_points']):+.3f} | "
                f"{float(row['fp_ndis_mean']):.1f} | "
                f"{float(row['binary_ndis_mean']):.1f} | "
                f"{float(row['binary_ndis_overhead_percent']):+.1f}% |"
            )
        overhead = [float(r["binary_ndis_overhead_percent"]) for r in group]
        delta = [float(r["fp_minus_binary_recall_percentage_points"]) for r in group]
        lines += [
            "",
            f"Hamming graph work overhead range: {min(overhead):.1f}% to "
            f"{max(overhead):.1f}%. FP-minus-Hamming recall range: "
            f"{min(delta):+.3f} to {max(delta):+.3f} percentage points.",
            "",
        ]
    build_path = (
        RESULT_DIR / "msmarco_quora_qwen1024_graph_construction_build_times.csv"
    )
    if build_path.exists():
        build_rows = list(csv.DictReader(build_path.open(newline="")))
        lines += [
            "## Fresh index build time",
            "",
            "M=24, efConstruction=200, 24 build threads. `index_build_seconds` "
            "is the benchmark's construction-and-save timer; wall time also includes "
            "process setup and the one-query smoke search.",
            "",
            "| Dataset | Build metric | Index build (s) | Wall (s) | Index size (MiB) |",
            "|---|---|---:|---:|---:|",
        ]
        for row in build_rows:
            lines.append(
                f"| {row['dataset_title']} | {row['graph_build_metric']} | "
                f"{float(row['index_build_seconds']):.3f} | "
                f"{float(row['command_wall_seconds']):.3f} | "
                f"{int(row['index_bytes']) / (1024 ** 2):.2f} |"
            )
        lines.append("")
    out = RESULT_DIR / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"WROTE {out}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(DATASETS), default=tuple(DATASETS)
    )
    parser.add_argument("--efs", nargs="+", type=int, default=DEFAULT_EFS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--measure-build-times", action="store_true")
    args = parser.parse_args()
    args.efs = tuple(args.efs)

    if not BIN.exists():
        raise FileNotFoundError(BIN)
    datasets = [DATASETS[key] for key in args.datasets]
    for dataset in datasets:
        validate(dataset)
    if args.measure_build_times:
        measure_build_times(datasets, args)
        return
    if not args.plot_only:
        for dataset in datasets:
            for graph_build in ("fp", "binary"):
                run_one(dataset, graph_build, args)
    for dataset in datasets:
        for graph_build in ("fp", "binary"):
            if not complete(raw_path(dataset, graph_build), args.efs):
                raise FileNotFoundError(raw_path(dataset, graph_build))
            if not complete(stats_path(dataset, graph_build), args.efs):
                raise FileNotFoundError(stats_path(dataset, graph_build))
    tidy, paired = build_summary(datasets, args.efs)
    plot(tidy, datasets)
    plot_recall_qps(paired, datasets)
    write_report(paired, datasets)


if __name__ == "__main__":
    main()
