#!/usr/bin/env python3
"""Exhaustive compressed-ranking comparison at a BQ-sized RAM budget.

The exhaustive stage does not rerank and does not calibrate.  It computes the
rank, under each compressed distance and a deterministic (distance, id) total
order, of the same exact FP top-10 neighbors.  The BQ-sized budget is reported
as the number/fraction of each method's codes that can remain resident after
pinning its non-code metadata.  A later graph/SSD stage can consume that exact
resident capacity without changing the representation accounting.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiment/faiss_exact_matched_budget_compare.cpp"
BINARY = REPO / "experiment/faiss_exact_matched_budget_compare"
RESULT_ROOT = Path(
    os.environ.get(
        "FAISS_EXACT_MATCHED_BUDGET_ROOT",
        REPO / "experiment/faiss_exact_matched_budget_compare_results",
    )
)
INDEX_ROOT = Path(
    os.environ.get(
        "FAISS_EXACT_MATCHED_BUDGET_INDEX_ROOT",
        RESULT_ROOT / "indexes",
    )
)

FAISS_ROOT = Path("/home/jihyo/workspace/faiss")
FAISS_BUILD = FAISS_ROOT / "build_hnsw_conda/faiss"
CONDA_LIB = Path("/home/jihyo/anaconda3/envs/hnsw/lib")
QDRANT_TQ_FFI = REPO / "experiment/qdrant_tq_ffi"
QDRANT_TQ_TARGET = Path(
    os.environ.get("QDRANT_TQ_CARGO_TARGET_DIR", QDRANT_TQ_FFI / "target")
)


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    train: Path
    query: Path
    gt: Path
    centroids_c16: Path


DATASETS = {
    "msmarco_qwen1024": Dataset(
        key="msmarco_qwen1024",
        title="MSMARCO Qwen3-0.6B 1024D",
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
        centroids_c16=REPO
        / "experiment/correction_ablation_fullscan_results/centroids_c16/"
        "msmarco_qwen1024.c16.fbin",
    ),
    "nq_qwen1024": Dataset(
        key="nq_qwen1024",
        title="NQ Qwen3-0.6B 1024D",
        train=Path(
            "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
            "nq_vanilla-qw06_train.fbin"
        ),
        query=Path(
            "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
            "nq_vanilla-qw06_query.fbin"
        ),
        gt=Path(
            "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
            "nq_vanilla-qw06_neighbors_top100.ibin"
        ),
        centroids_c16=REPO
        / "experiment/nq_quora_encoding_preprocess/nq_qwen06_1024/"
        "centroids_c16.fbin",
    ),
}

METHODS = (
    "adcbq1",
    "symbq1",
    "pq8m128",
    "opq8m128",
    "sq4",
    "sq8",
    "rabitq1",
    "qdranttq1",
)


def compile_binary() -> None:
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(QDRANT_TQ_FFI / "Cargo.toml"),
        ],
        cwd=REPO,
        env={**os.environ, "CARGO_TARGET_DIR": str(QDRANT_TQ_TARGET)},
        check=True,
    )
    dependencies = [
        SOURCE,
        REPO / "experiment/faiss_hnsw_lowbit_compare.cpp",
        QDRANT_TQ_FFI / "qdrant_tq_ffi.h",
        QDRANT_TQ_TARGET / "release/libqdrant_tq_ffi.so",
    ]
    if BINARY.exists() and all(
        BINARY.stat().st_mtime_ns >= path.stat().st_mtime_ns for path in dependencies
    ):
        return
    command = [
        "g++",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-march=native",
        "-fopenmp",
        f"-I{REPO / 'include'}",
        f"-I{FAISS_ROOT}",
        f"-I{FAISS_BUILD}",
        str(SOURCE),
        f"-L{FAISS_BUILD}",
        f"-L{CONDA_LIB}",
        f"-L{QDRANT_TQ_TARGET / 'release'}",
        f"-Wl,-rpath,{FAISS_BUILD}:{CONDA_LIB}:{QDRANT_TQ_TARGET / 'release'}",
        "-lfaiss_avx2",
        "-lopenblas",
        "-lqdrant_tq_ffi",
        "-o",
        str(BINARY),
    ]
    print("COMPILE " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def method_paths(dataset: Dataset, method: str) -> tuple[Path, Path, Path, Path]:
    root = INDEX_ROOT / dataset.key
    suffix = "tqdata" if method == "qdranttq1" else "faiss"
    index = root / f"{dataset.key}.{method}.{suffix}"
    meta = root / f"{dataset.key}.{method}.meta"
    summary = RESULT_ROOT / dataset.key / f"{dataset.key}.{method}.summary.csv"
    per_query = RESULT_ROOT / dataset.key / f"{dataset.key}.{method}.per_query.csv"
    return index, meta, summary, per_query


def run_one(dataset: Dataset, method: str, args: argparse.Namespace) -> Path:
    for path in (dataset.train, dataset.query, dataset.gt):
        if not path.exists():
            raise FileNotFoundError(path)
    if method == "rabitq1" and not dataset.centroids_c16.exists():
        raise FileNotFoundError(dataset.centroids_c16)
    index, meta, summary, per_query = method_paths(dataset, method)
    index.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(BINARY),
        "--dataset",
        dataset.key,
        "--method",
        method,
        "--metric",
        "ip",
        "--train",
        str(dataset.train),
        "--query",
        str(dataset.query),
        "--gt",
        str(dataset.gt),
        "--index",
        str(index),
        "--summary-out",
        str(summary),
        "--per-query-out",
        str(per_query),
        "--max-queries",
        str(args.max_queries),
        "--threads",
        str(args.threads),
        "--train-size",
        str(args.train_size),
        "--add-batch-size",
        str(args.add_batch_size),
        "--opq-niter",
        str(args.opq_niter),
    ]
    if method in ("rabitq1", "qdranttq1"):
        command += ["--meta", str(meta)]
    if method == "rabitq1":
        command += ["--rabitq-centroids", str(dataset.centroids_c16)]
    if args.rebuild:
        command.append("--rebuild")
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=REPO,
        env={
            **os.environ,
            "OMP_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        check=True,
    )
    return summary


def combine(paths: list[Path]) -> Path:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    if not rows or fieldnames is None:
        raise RuntimeError("no summary rows produced")
    output = RESULT_ROOT / "faiss_exact_matched_budget_combined.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--add-batch-size", type=int, default=4096)
    parser.add_argument("--opq-niter", type=int, default=25)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.combine_only:
        compile_binary()
    if args.compile_only:
        return
    summaries: list[Path] = []
    for dataset_key in args.datasets:
        dataset = DATASETS[dataset_key]
        for method in args.methods:
            _, _, summary, _ = method_paths(dataset, method)
            if args.combine_only:
                if summary.exists():
                    summaries.append(summary)
            else:
                summaries.append(run_one(dataset, method, args))
    combine(summaries)


if __name__ == "__main__":
    main()
