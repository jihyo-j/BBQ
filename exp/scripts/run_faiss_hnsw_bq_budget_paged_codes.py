#!/usr/bin/env python3
"""Run a common M24/efC200 HNSW with BQ-sized paged code storage.

The graph is common and resident.  Only the compressed database-code budget is
matched to BQ (1,000,000 * 128 bytes).  Method metadata is pinned first; the
remaining budget is a 4 KiB LRU cache.  Cache misses are synchronous O_DIRECT
reads from the SSD code sidecar.  No calibration, FP reranking, or adaptive
exact verification is used in this stage.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "experiment/faiss_hnsw_bq_budget_paged_codes.cpp"
BINARY = REPO / "experiment/faiss_hnsw_bq_budget_paged_codes"
EXACT_ROOT = REPO / "experiment/faiss_exact_matched_budget_compare_results"
EXACT_SUMMARY = EXACT_ROOT / "faiss_exact_matched_budget_combined.csv"
RESULT_ROOT = REPO / "experiment/faiss_hnsw_bq_budget_paged_codes_results"
RERANK_RESULT_ROOT = (
    REPO / "experiment/faiss_hnsw_bq_budget_paged_rerank80_results"
)

CODEC_ROOT = Path(
    os.environ.get(
        "FAISS_EXACT_MATCHED_BUDGET_INDEX_ROOT",
        "/home/smrc/samsung-nvme/jihyo/"
        "faiss_exact_matched_budget_compare_results/indexes",
    )
)
SIDECAR_ROOT = Path(
    os.environ.get(
        "FAISS_BQ_BUDGET_SIDECAR_ROOT",
        "/home/smrc/samsung-nvme/jihyo/"
        "faiss_hnsw_bq_budget_paged_codes/sidecars",
    )
)

FAISS_ROOT = Path("/home/jihyo/workspace/faiss")
FAISS_BUILD = FAISS_ROOT / "build_hnsw_conda/faiss"
CONDA_LIB = Path("/home/jihyo/anaconda3/envs/hnsw/lib")
QDRANT_TQ_FFI = REPO / "experiment/qdrant_tq_ffi"
QDRANT_TQ_TARGET = QDRANT_TQ_FFI / "target"


def result_root(rerank_n: int) -> Path:
    if rerank_n <= 0:
        return RESULT_ROOT
    if rerank_n == 80:
        return RERANK_RESULT_ROOT
    return REPO / f"experiment/faiss_hnsw_bq_budget_paged_rerank{rerank_n}_results"


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    query: Path
    gt: Path
    graph: Path
    fp_sidecar: Path


DATASETS = {
    "msmarco_qwen1024": Dataset(
        key="msmarco_qwen1024",
        title="MSMARCO Qwen-1024",
        query=Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_query.fbin"
        ),
        gt=Path(
            "/home/smrc/samsung-nvme/jihyo/qwen/"
            "msmarco-1m__vanilla-qwen06b_neighbors.ibin"
        ),
        graph=Path(
            "/home/smrc/samsung-nvme/jihyo/msmarco_qwen1024_scale_v014/"
            "indexes/faiss_adcbq/"
            "msmarco-qwen1024-1m.fpgraph_pgbit_binary_hnsw."
            "n1000000.d1024.M24.efC200.faiss"
        ),
        fp_sidecar=Path(
            "/home/smrc/samsung-nvme/jihyo/diskveclab_nq_msmarco_qwen1024/"
            "msmarco_qwen1024/bbq/msmarco_qwen1024.fp32.aligned4k.fbin"
        ),
    ),
    "nq_qwen1024": Dataset(
        key="nq_qwen1024",
        title="NQ Qwen-1024",
        query=Path(
            "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
            "nq_vanilla-qw06_query.fbin"
        ),
        gt=Path(
            "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
            "nq_vanilla-qw06_neighbors_top100.ibin"
        ),
        graph=Path(
            "/home/smrc/samsung-nvme/jihyo/"
            "rabitq_v051_ssd_multidataset_indexes/nq_qwen06_1024/"
            "nq_qwen06_1024.fpgraph_adcbq.d1024.M24.efC200.faiss"
        ),
        fp_sidecar=Path(
            "/home/smrc/samsung-nvme/jihyo/diskveclab_nq_msmarco_qwen1024/"
            "nq_qwen06_1024/bbq/nq_qwen06_1024.fp32.aligned4k.fbin"
        ),
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
DISPLAY_TO_KEY = {
    "ADC-BQ1": "adcbq1",
    "Symmetric-BQ1": "symbq1",
    "Faiss-PQ8-M128": "pq8m128",
    "Faiss-OPQ-PQ8-M128": "opq8m128",
    "Faiss-SQ4": "sq4",
    "Faiss-SQ8": "sq8",
    "RaBitQ-1b-c16": "rabitq1",
    "Qdrant-TQ1-Plus": "qdranttq1",
}


def compile_binary() -> None:
    dependencies = [SOURCE, REPO / "experiment/faiss_hnsw_lowbit_compare.cpp"]
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
        "-luring",
        "-o",
        str(BINARY),
    ]
    print("COMPILE " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def exact_accounting() -> dict[tuple[str, str], dict[str, str]]:
    with EXACT_SUMMARY.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        result[(row["dataset"], DISPLAY_TO_KEY[row["method"]])] = row
    return result


def codec_paths(dataset: str, method: str) -> tuple[Path, Path]:
    root = CODEC_ROOT / dataset
    suffix = "tqdata" if method == "qdranttq1" else "faiss"
    return (
        root / f"{dataset}.{method}.{suffix}",
        root / f"{dataset}.{method}.meta",
    )


def run_one(
    dataset: Dataset,
    method: str,
    accounting: dict[tuple[str, str], dict[str, str]],
    args: argparse.Namespace,
) -> Path:
    codec, meta = codec_paths(dataset.key, method)
    sidecar = SIDECAR_ROOT / dataset.key / f"{dataset.key}.{method}.pages4k"
    output_root = result_root(args.rerank_n)
    output = output_root / dataset.key / f"{dataset.key}.{method}.csv"
    for path in (dataset.query, dataset.gt, dataset.graph, codec):
        if not path.exists():
            raise FileNotFoundError(path)
    if method in ("rabitq1", "qdranttq1") and not meta.exists():
        raise FileNotFoundError(meta)
    if args.rerank_n > 0 and not dataset.fp_sidecar.exists():
        raise FileNotFoundError(dataset.fp_sidecar)

    row = accounting[(dataset.key, method)]
    noncode = int(row["noncode_artifact_bytes"])
    # Serialization headers are not runtime metadata.  The stable page-sized
    # part is the codebook/rotation/correction state that must remain pinned.
    pinned_metadata = (noncode // 4096) * 4096
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(BINARY),
        "--dataset",
        dataset.key,
        "--method",
        method,
        "--metric",
        "ip",
        "--query",
        str(dataset.query),
        "--gt",
        str(dataset.gt),
        "--graph",
        str(dataset.graph),
        "--codec-index",
        str(codec),
        "--sidecar",
        str(sidecar),
        "--out",
        str(output),
        "--max-queries",
        str(args.max_queries),
        "--rounds",
        str(args.rounds),
        "--threads",
        str(args.threads),
        "--bq-budget-bytes",
        str(int(row["bq_budget_bytes"])),
        "--pinned-metadata-bytes",
        str(pinned_metadata),
        "--efs",
        *[str(value) for value in args.efs],
    ]
    if method in ("rabitq1", "qdranttq1"):
        command += ["--meta", str(meta)]
    if args.rerank_n > 0:
        command += [
            "--rerank-n",
            str(args.rerank_n),
            "--fp-sidecar",
            str(dataset.fp_sidecar),
        ]
    if args.rebuild_sidecars:
        command.append("--rebuild-sidecar")
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
    return output


def combine(paths: list[Path], rerank_n: int) -> Path:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for field in reader.fieldnames or []:
                if field not in fieldnames:
                    fieldnames.append(field)
            rows.extend(reader)
    if not rows or not fieldnames:
        raise RuntimeError("no rows to combine")
    root = result_root(rerank_n)
    name = (
        f"faiss_hnsw_bq_budget_paged_rerank{rerank_n}_combined.csv"
        if rerank_n > 0
        else "faiss_hnsw_bq_budget_paged_codes_combined.csv"
    )
    output = root / name
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {output}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--rerank-n", type=int, default=0)
    parser.add_argument(
        "--efs",
        nargs="+",
        type=int,
        default=[32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024],
    )
    parser.add_argument("--rebuild-sidecars", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--combine-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.combine_only:
        compile_binary()
    if args.compile_only:
        return
    accounting = exact_accounting()
    outputs: list[Path] = []
    for dataset_key in args.datasets:
        dataset = DATASETS[dataset_key]
        for method in args.methods:
            output_root = result_root(args.rerank_n)
            output = output_root / dataset.key / f"{dataset.key}.{method}.csv"
            if args.combine_only:
                if output.exists():
                    outputs.append(output)
            else:
                outputs.append(run_one(dataset, method, accounting, args))
    combine(outputs, args.rerank_n)


if __name__ == "__main__":
    main()
