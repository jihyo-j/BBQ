#!/usr/bin/env python3
"""ADC-BQ external-memory HNSW colocation ablation for MSMARCO.

The original FAISS HNSW graph is identical for both layouts.  ``baseline``
serializes nodes in their original order, while ``coloc`` applies the LDG
neighbor permutation before serializing the graph and ADC-BQ codes.  Both use
the same 4 KiB O_DIRECT paged-index reader, an explicitly bounded graph-page
cache, and exact FP32 O_DIRECT/io_uring rerank@80.

The default full experiment uses a graph cache equal to 25% of each paged
index's data-page footprint.  Additional cache ratios can be requested for a
memory-sensitivity ablation.  T=24 overlaps independent graph reads and FP
rerank batches across query workers; T=1 still batches the 80 FP reads for a
query through io_uring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import run_v051_ssd_odirect_multidataset as base


REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw"
BASE_BUILD_CSV = (
    REPO / "experiment/msmarco_adcbq_colocation_4k/base_index_build_t24/build_times.csv"
)
RESULT_ROOT = REPO / "experiment/msmarco_adcbq_colocation_4k"
FIGURE_ROOT = REPO / "experiment/figures/msmarco_adcbq_colocation_4k"
ARTIFACT_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/msmarco_adcbq_colocation_4k"
)

PAGE_BYTES = 4096
FP_HEADER_BYTES = 4096
M = 24
EF_CONSTRUCTION = 200
RERANK_N = 80
QD = 80
COLOCATION_ITERATIONS = 20
COLOCATION_PARTITION_SIZE = 64
DEFAULT_EFS = (64, 128, 256, 512, 1024)
DEFAULT_THREADS = (1, 24)
DEFAULT_CACHE_RATIOS = (0.25,)
LAYOUTS = ("baseline", "coloc")


def datasets() -> list[base.Dataset]:
    return [
        base.DATASETS["msmarco_openai1536"],
        base.DATASETS["msmarco_bge768"],
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def matrix_shape(path: Path) -> tuple[int, int]:
    return base.matrix_shape(path)


def aligned_fp_path(dataset: base.Dataset, artifact_root: Path) -> Path:
    return artifact_root / dataset.key / f"{dataset.key}.fp32.aligned4k.fbin"


def validate_aligned_fp(path: Path, dataset: base.Dataset) -> bool:
    n, d = matrix_shape(dataset.train)
    return (
        path.exists()
        and path.stat().st_size == FP_HEADER_BYTES + n * d * np.dtype(np.float32).itemsize
        and matrix_shape(path) == (n, d)
    )


def prepare_aligned_fp(path: Path, dataset: base.Dataset) -> None:
    if validate_aligned_fp(path, dataset):
        print(f"REUSE aligned FP store: {path}", flush=True)
        return
    n, d = matrix_shape(dataset.train)
    payload_bytes = n * d * np.dtype(np.float32).itemsize
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    copied = 0
    started = time.perf_counter()
    try:
        with dataset.train.open("rb", buffering=0) as source, temporary.open(
            "wb", buffering=0
        ) as destination:
            header = source.read(8)
            if len(header) != 8:
                raise RuntimeError(f"short fbin header: {dataset.train}")
            destination.write(header)
            destination.write(bytes(FP_HEADER_BYTES - len(header)))
            while copied < payload_bytes:
                block = source.read(min(16 * 1024 * 1024, payload_bytes - copied))
                if not block:
                    raise RuntimeError(f"short fbin payload: {dataset.train}")
                destination.write(block)
                copied += len(block)
            destination.flush()
            os.fsync(destination.fileno())
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(destination.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if not validate_aligned_fp(path, dataset):
        raise RuntimeError(f"invalid aligned FP store after creation: {path}")
    print(
        f"PREPARED aligned FP store bytes={path.stat().st_size} "
        f"seconds={time.perf_counter() - started:.2f}: {path}",
        flush=True,
    )


def layout_dir(dataset: base.Dataset, artifact_root: Path) -> Path:
    return artifact_root / dataset.key


def paged_index_path(dataset: base.Dataset, layout: str, artifact_root: Path) -> Path:
    return layout_dir(dataset, artifact_root) / (
        f"{dataset.key}.adcbq_hnsw.{layout}.M{M}.efC{EF_CONSTRUCTION}."
        f"paged{PAGE_BYTES // 1024}k"
    )


def colocation_map_path(dataset: base.Dataset, artifact_root: Path) -> Path:
    return layout_dir(dataset, artifact_root) / (
        f"{dataset.key}.ldg{COLOCATION_ITERATIONS}.p{COLOCATION_PARTITION_SIZE}.map"
    )


def build_manifest_path(dataset: base.Dataset, layout: str) -> Path:
    return RESULT_ROOT / dataset.key / f"build_{layout}.json"


def base_build_rows() -> dict[str, dict[str, str]]:
    if BASE_BUILD_CSV.exists():
        rows = {row["dataset"]: row for row in read_csv(BASE_BUILD_CSV)}
        if all(dataset.key in rows for dataset in datasets()):
            return rows
    # Do not mislabel a sub-second existing-index load as build time.  The
    # dedicated 24-thread timing runner normally populates BASE_BUILD_CSV.
    return {
        dataset.key: {
            "build_seconds": "",
            "index_bytes": str(dataset.adcbq_index.stat().st_size),
            "timing_status": "fresh_build_time_unavailable",
        }
        for dataset in datasets()
    }


def common_command(
    dataset: base.Dataset,
    layout: str,
    artifact_root: Path,
    output: Path,
    cache_mib: int,
    threads: int,
    efs: tuple[int, ...],
    rounds: int,
    max_queries: int,
) -> list[str]:
    command = [
        str(BIN),
        "--dataset", dataset.key,
        "--train", str(dataset.train),
        "--query", str(dataset.query),
        "--gt", str(dataset.gt),
        "--index", str(dataset.adcbq_index),
        "--out", str(output),
        "--m", str(M),
        "--ef-construction", str(EF_CONSTRUCTION),
        "--k", "10",
        "--rerank-n", str(RERANK_N),
        "--max-queries", str(max_queries),
        "--rounds", str(rounds),
        "--build-threads", "24",
        "--search-threads", str(threads),
        "--adc-mode", "int16",
        "--lut-mode", "nibble",
        "--graph-build", "fp",
        "--distance-mode", "adcbq",
        "--search-impl", "inline",
        "--code-layout", "vector",
        "--prefetch-lookahead", "8",
        "--index-page-cache-mib", str(cache_mib),
        "--paged-index-path", str(paged_index_path(dataset, layout, artifact_root)),
        "--efs", *map(str, efs),
    ]
    if layout == "coloc":
        command.extend(
            [
                "--colocation-iterations", str(COLOCATION_ITERATIONS),
                "--colocation-partition-size", str(COLOCATION_PARTITION_SIZE),
                "--colocation-map-path", str(colocation_map_path(dataset, artifact_root)),
            ]
        )
    return command


def run_command(command: list[str], log: Path, threads: int) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
        }
    )
    print("RUN " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def validate_build_row(row: dict[str, str], layout: str) -> None:
    if int(row["index_page_bytes"]) != PAGE_BYTES:
        raise RuntimeError(f"unexpected graph page size: {row['index_page_bytes']}")
    if row["paged_index_status"] != "built":
        raise RuntimeError(f"expected newly built paged index, got {row['paged_index_status']}")
    expected_iterations = COLOCATION_ITERATIONS if layout == "coloc" else 0
    if int(row["colocation_iterations"]) != expected_iterations:
        raise RuntimeError("colocation mode mismatch in build result")


def build_layout(
    dataset: base.Dataset,
    layout: str,
    artifact_root: Path,
    base_build: dict[str, str],
    force: bool,
) -> dict[str, object]:
    manifest_path = build_manifest_path(dataset, layout)
    if manifest_path.exists() and not force:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            int(payload["page_bytes"]) == PAGE_BYTES
            and Path(payload["paged_index_path"]).exists()
        ):
            print(f"SKIP complete layout build: {dataset.key} {layout}", flush=True)
            return payload

    directory = RESULT_ROOT / dataset.key
    directory.mkdir(parents=True, exist_ok=True)
    paged = paged_index_path(dataset, layout, artifact_root)
    paged.parent.mkdir(parents=True, exist_ok=True)
    smoke = directory / f"build_{layout}_smoke.csv"
    log = directory / f"build_{layout}.log"
    command = common_command(
        dataset, layout, artifact_root, smoke, 1, 1, (64,), 1, 1
    )
    command.extend(["--no-rerank", "--rebuild-paged-index"])
    if layout == "coloc":
        command.append("--rebuild-colocation")
    started = time.perf_counter()
    run_command(command, log, 1)
    wall_seconds = time.perf_counter() - started
    rows = read_csv(smoke)
    if len(rows) != 1:
        raise RuntimeError(f"expected one layout-build row: {smoke}")
    row = rows[0]
    validate_build_row(row, layout)
    base_seconds = (
        float(base_build["build_seconds"])
        if base_build["build_seconds"]
        else None
    )
    map_seconds = float(row["colocation_prepare_seconds"])
    apply_seconds = float(row["colocation_apply_seconds"])
    serialize_seconds = float(row["paged_index_prepare_seconds"])
    payload: dict[str, object] = {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "layout": layout,
        "page_bytes": PAGE_BYTES,
        "M": M,
        "efConstruction": EF_CONSTRUCTION,
        "build_threads": 24,
        "base_index_build_seconds": base_seconds,
        "colocation_map_build_seconds": map_seconds,
        "colocation_apply_seconds": apply_seconds,
        "paged_index_serialize_seconds": serialize_seconds,
        "layout_materialization_seconds": map_seconds + apply_seconds + serialize_seconds,
        "build_end_to_end_seconds": (
            base_seconds + map_seconds + apply_seconds + serialize_seconds
            if base_seconds is not None
            else None
        ),
        "materialization_process_wall_seconds": wall_seconds,
        "base_index_timing_status": base_build.get(
            "timing_status", "fresh_build_measured_24_threads"
        ),
        "base_index_bytes": int(base_build["index_bytes"]),
        "paged_index_bytes": int(row["paged_index_file_bytes"]),
        "paged_index_data_pages": int(row["paged_index_data_pages"]),
        "paged_index_data_bytes": int(row["paged_index_data_pages"]) * PAGE_BYTES,
        "paged_index_path": str(paged),
        "colocation_map_path": str(colocation_map_path(dataset, artifact_root)) if layout == "coloc" else "",
        "timing_scope": f"fresh ADC-BQ graph build + LDG map + ID permutation + {PAGE_BYTES // 1024}KB paged serialization",
        "source_smoke_csv": str(smoke),
        "log_path": str(log),
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def cache_mib_for_ratio(
    build: dict[str, object], ratio: float, fixed_cache_mib: int | None = None
) -> int:
    if fixed_cache_mib is not None:
        return fixed_cache_mib
    data_bytes = int(build["paged_index_data_bytes"])
    return max(1, math.ceil(data_bytes * ratio / (1024 * 1024)))


def ratio_label(ratio: float) -> str:
    return f"p{round(ratio * 100):03d}"


def search_path(
    dataset: base.Dataset, layout: str, ratio: float, threads: int
) -> Path:
    return RESULT_ROOT / dataset.key / (
        f"search_{layout}_{ratio_label(ratio)}_t{threads}.csv"
    )


def search_complete(
    path: Path,
    layout: str,
    cache_mib: int,
    threads: int,
    efs: tuple[int, ...],
    rounds: int,
    expected_nq: int,
) -> bool:
    try:
        rows = read_csv(path)
        return (
            [int(row["efSearch"]) for row in rows] == list(efs)
            and all(int(row["search_threads"]) == threads for row in rows)
            and all(int(row["rounds"]) == rounds for row in rows)
            and all(int(row["nq"]) == expected_nq for row in rows)
            and all(int(row["rerank_n"]) == RERANK_N for row in rows)
            and all(int(row["index_page_bytes"]) == PAGE_BYTES for row in rows)
            and all(int(row["index_page_cache_mib"]) == cache_mib for row in rows)
            and all(row["fp_rerank_source"] == "odirect-uring" for row in rows)
            and all(int(float(row["direct_rows_mean"])) == expected_nq * RERANK_N for row in rows)
            and all(
                int(row["colocation_iterations"])
                == (COLOCATION_ITERATIONS if layout == "coloc" else 0)
                for row in rows
            )
        )
    except (OSError, KeyError, ValueError):
        return False


def run_search(
    dataset: base.Dataset,
    layout: str,
    ratio: float,
    threads: int,
    build: dict[str, object],
    sidecar: Path,
    args: argparse.Namespace,
) -> Path:
    cache_mib = cache_mib_for_ratio(
        build, ratio, args.fixed_cache_mib.get(dataset.key)
    )
    output = search_path(dataset, layout, ratio, threads)
    full_nq, _ = matrix_shape(dataset.query)
    expected_nq = min(full_nq, args.max_queries) if args.max_queries else full_nq
    if search_complete(
        output, layout, cache_mib, threads, args.efs, args.rounds, expected_nq
    ) and not args.force_search:
        print(f"SKIP complete search: {output}", flush=True)
        return output

    command = common_command(
        dataset,
        layout,
        args.artifact_root,
        output,
        cache_mib,
        threads,
        args.efs,
        args.rounds,
        args.max_queries,
    )
    command.extend(
        [
            "--rerank-dot", "fp32",
            "--fp-rerank-source", "odirect-uring",
            "--fp-rerank-norms", "onfetch",
            "--fp-rerank-path", str(sidecar),
            "--fp-rerank-qd", str(QD),
        ]
    )
    log = output.with_suffix(".log")
    run_command(command, log, threads)
    if not search_complete(
        output, layout, cache_mib, threads, args.efs, args.rounds, expected_nq
    ):
        raise RuntimeError(f"search output failed invariant checks: {output}")
    return output


def combine(
    selected: list[base.Dataset],
    builds: dict[tuple[str, str], dict[str, object]],
    args: argparse.Namespace,
) -> Path:
    rows: list[dict[str, object]] = []
    for dataset in selected:
        for ratio in args.cache_ratios:
            for threads in args.threads:
                groups: dict[str, list[dict[str, str]]] = {}
                for layout in LAYOUTS:
                    source = search_path(dataset, layout, ratio, threads)
                    groups[layout] = read_csv(source)
                    build = builds[(dataset.key, layout)]
                    for row in groups[layout]:
                        nq = int(row["nq"])
                        graph_read_bytes = float(row["index_page_read_bytes_mean"])
                        fp_read_bytes = float(row["direct_submitted_bytes_mean"])
                        rows.append(
                            {
                                "dataset": dataset.key,
                                "corpus": dataset.corpus,
                                "model": dataset.model,
                                "layout": layout,
                                "method": "ADC-BQ disk + colocation" if layout == "coloc" else "ADC-BQ disk baseline",
                                "graph_page_bytes": PAGE_BYTES,
                                "graph_cache_ratio": ratio,
                                "graph_cache_mib": int(row["index_page_cache_mib"]),
                                "graph_data_bytes": build["paged_index_data_bytes"],
                                "graph_cache_fraction_actual": (
                                    int(row["index_page_cache_mib"]) * 1024 * 1024
                                    / int(build["paged_index_data_bytes"])
                                ),
                                "threads": threads,
                                "efSearch": int(row["efSearch"]),
                                "recall_at_10": float(row["recall_at_k"]),
                                "qps": float(row["qps"]),
                                "latency_ms_batch_average": float(row["total_seconds_mean"]) * 1000.0 / nq,
                                "graph_page_accesses_per_query": float(row["index_page_accesses_mean"]) / nq,
                                "graph_page_hits_per_query": float(row["index_page_hits_mean"]) / nq,
                                "graph_page_misses_per_query": float(row["index_page_misses_mean"]) / nq,
                                "graph_page_hit_ratio": float(row["index_page_hit_ratio"]),
                                "graph_read_bytes_per_query": graph_read_bytes / nq,
                                "graph_physical_page_reads_per_query": graph_read_bytes / PAGE_BYTES / nq,
                                "graph_physical_4k_equivalent_reads_per_query": graph_read_bytes / 4096 / nq,
                                "fp_vectors_per_query": float(row["direct_rows_mean"]) / nq,
                                "fp_requests_per_query": float(row["direct_requests_mean"]) / nq,
                                "fp_read_bytes_per_query": fp_read_bytes / nq,
                                "fp_physical_4k_reads_per_query": fp_read_bytes / FP_HEADER_BYTES / nq,
                                "rounds": int(row["rounds"]),
                                "nq": nq,
                                "nb": int(row["nb"]),
                                "dim": int(row["dim"]),
                                "M": M,
                                "efConstruction": EF_CONSTRUCTION,
                                "rerank_n": RERANK_N,
                                "graph_storage": f"SSD O_DIRECT {PAGE_BYTES // 1024}KB pages with explicit LRU cache",
                                "fp_storage": "SSD io_uring O_DIRECT",
                                "cpu_ssd_overlap": "T24 independent query workers; per-query FP batch",
                                "source_csv": str(source),
                            }
                        )
                baseline = {int(row["efSearch"]): row for row in groups["baseline"]}
                coloc = {int(row["efSearch"]): row for row in groups["coloc"]}
                for ef in args.efs:
                    if abs(float(baseline[ef]["recall_at_k"]) - float(coloc[ef]["recall_at_k"])) > 1e-12:
                        raise RuntimeError(
                            f"recall changed after physical colocation: {dataset.key} "
                            f"ratio={ratio} T={threads} ef={ef}"
                        )
    output = RESULT_ROOT / "all_search_results.csv"
    write_csv(output, rows)
    write_csv(RESULT_ROOT / "build_times.csv", list(builds.values()))
    return output


def plot(combined: Path, selected: list[base.Dataset], args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_csv(combined)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    styles = {
        "baseline": ("#0072B2", "s", "--"),
        "coloc": ("#D55E00", "o", "-"),
    }
    corpora = list(dict.fromkeys(item.corpus for item in selected))
    for ratio in args.cache_ratios:
        for threads in args.threads:
            for corpus in corpora:
                corpus_sets = [item for item in selected if item.corpus == corpus]
                fig, axes = plt.subplots(2, len(corpus_sets), figsize=(5.2 * len(corpus_sets), 9.2), squeeze=False)
                for column, dataset in enumerate(corpus_sets):
                    selected_rows = [
                        row for row in rows
                        if row["dataset"] == dataset.key
                        and int(row["threads"]) == threads
                        and abs(float(row["graph_cache_ratio"]) - ratio) < 1e-12
                    ]
                    for layout, (color, marker, linestyle) in styles.items():
                        group = sorted(
                            (row for row in selected_rows if row["layout"] == layout),
                            key=lambda row: int(row["efSearch"]),
                        )
                        label = "ADC-BQ disk + colocation" if layout == "coloc" else "ADC-BQ disk baseline"
                        xs = [float(row["recall_at_10"]) for row in group]
                        qps = [float(row["qps"]) for row in group]
                        reads = [float(row["graph_physical_page_reads_per_query"]) for row in group]
                        axes[0][column].plot(xs, qps, color=color, marker=marker, linestyle=linestyle, label=label)
                        axes[1][column].plot(xs, reads, color=color, marker=marker, linestyle=linestyle, label=label)
                        for x, y, row in zip(xs, qps, group):
                            axes[0][column].annotate(row["efSearch"], (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color=color)
                    axes[0][column].set_title(dataset.model)
                    axes[0][column].set_yscale("log")
                    axes[0][column].set_ylabel("QPS (log scale)")
                    axes[1][column].set_ylabel(
                        f"Graph physical {PAGE_BYTES // 1024}KB reads/query"
                    )
                    for axis in (axes[0][column], axes[1][column]):
                        axis.set_xlabel("Recall@10")
                        axis.grid(True, alpha=0.25)
                handles, labels = axes[0][0].get_legend_handles_labels()
                fig.suptitle(
                    f"{corpus}: ADC-BQ {PAGE_BYTES // 1024}KB graph colocation, cache={ratio:.0%}, T={threads}",
                    fontsize=15,
                )
                fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=2, frameon=False)
                fig.tight_layout(rect=(0, 0, 1, 0.91))
                stem = f"{corpus.lower()}_cache_{ratio_label(ratio)}_t{threads}"
                fig.savefig(FIGURE_ROOT / f"{stem}.png", dpi=190, bbox_inches="tight")
                fig.savefig(FIGURE_ROOT / f"{stem}.pdf", bbox_inches="tight")
                plt.close(fig)


def main() -> None:
    global ARTIFACT_ROOT, BIN, FIGURE_ROOT, PAGE_BYTES, RESULT_ROOT
    available = datasets()
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(item.key for item in available), default=[item.key for item in available])
    parser.add_argument("--threads", type=int, nargs="+", default=list(DEFAULT_THREADS))
    parser.add_argument("--efs", type=int, nargs="+", default=list(DEFAULT_EFS))
    parser.add_argument("--cache-ratios", type=float, nargs="+", default=list(DEFAULT_CACHE_RATIOS))
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--figure-root", type=Path, default=FIGURE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--binary", type=Path, default=BIN)
    parser.add_argument("--page-bytes", type=int, choices=(4096, 8192, 16384), default=PAGE_BYTES)
    parser.add_argument(
        "--fixed-cache-mib",
        action="append",
        default=[],
        metavar="DATASET=MIB",
        help="override cache MiB for a dataset; repeat for multiple datasets",
    )
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--keep-fp-sidecar", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    args.threads = tuple(dict.fromkeys(args.threads))
    args.efs = tuple(dict.fromkeys(args.efs))
    args.cache_ratios = tuple(dict.fromkeys(args.cache_ratios))
    fixed_cache_mib: dict[str, int] = {}
    for item in args.fixed_cache_mib:
        try:
            key, raw_mib = item.split("=", 1)
            mib = int(raw_mib)
        except ValueError:
            parser.error(f"invalid --fixed-cache-mib value: {item!r}")
        if key not in {dataset.key for dataset in available} or mib <= 0:
            parser.error(f"invalid --fixed-cache-mib value: {item!r}")
        fixed_cache_mib[key] = mib
    args.fixed_cache_mib = fixed_cache_mib
    if any(thread <= 0 for thread in args.threads):
        parser.error("threads must be positive")
    if any(ratio <= 0 or ratio > 1 for ratio in args.cache_ratios):
        parser.error("cache ratios must be in (0, 1]")
    if args.rounds <= 0 or args.max_queries < 0:
        parser.error("rounds must be positive and max-queries nonnegative")

    RESULT_ROOT = args.result_root.resolve()
    FIGURE_ROOT = args.figure_root.resolve()
    ARTIFACT_ROOT = args.artifact_root.resolve()
    BIN = args.binary.resolve()
    PAGE_BYTES = args.page_bytes
    args.artifact_root = ARTIFACT_ROOT
    selected = [base.DATASETS[key] for key in args.datasets]
    if len(selected) == len(available) and len(selected) != 2:
        raise RuntimeError(f"expected two default MSMARCO datasets, got {len(selected)}")
    if not BIN.exists():
        raise FileNotFoundError(BIN)
    for dataset in selected:
        base.validate(dataset)
        if not dataset.adcbq_index.exists():
            raise FileNotFoundError(dataset.adcbq_index)

    base_build = base_build_rows()
    builds: dict[tuple[str, str], dict[str, object]] = {}
    for dataset in selected:
        for layout in LAYOUTS:
            builds[(dataset.key, layout)] = build_layout(
                dataset,
                layout,
                ARTIFACT_ROOT,
                base_build[dataset.key],
                args.force_build,
            )

    if not args.plot_only:
        for position, dataset in enumerate(selected, start=1):
            print(f"=== DATASET {position}/{len(selected)}: {dataset.key} ===", flush=True)
            sidecar = aligned_fp_path(dataset, ARTIFACT_ROOT)
            prepare_aligned_fp(sidecar, dataset)
            completed = False
            try:
                for ratio in args.cache_ratios:
                    for threads in args.threads:
                        for layout in LAYOUTS:
                            run_search(
                                dataset,
                                layout,
                                ratio,
                                threads,
                                builds[(dataset.key, layout)],
                                sidecar,
                                args,
                            )
                completed = True
            finally:
                if completed and not args.keep_fp_sidecar and sidecar.exists():
                    size = sidecar.stat().st_size
                    sidecar.unlink()
                    print(f"REMOVED transient FP sidecar bytes={size}: {sidecar}", flush=True)

    combined = combine(selected, builds, args)
    plot(combined, selected, args)
    manifest = {
        "datasets": [dataset.key for dataset in selected],
        "layouts": list(LAYOUTS),
        "graph_page_bytes": PAGE_BYTES,
        "fixed_cache_mib": args.fixed_cache_mib,
        "graph_cache_ratios": list(args.cache_ratios),
        "threads": list(args.threads),
        "efSearch": list(args.efs),
        "rounds": args.rounds,
        "max_queries": args.max_queries,
        "rerank_n": RERANK_N,
        "graph_io": "synchronous O_DIRECT per cache miss; T24 reads overlap across workers",
        "fp_io": "io_uring O_DIRECT batch per query",
        "cache_policy": "explicit shared LRU, recreated cold for each timed round",
    }
    (RESULT_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
