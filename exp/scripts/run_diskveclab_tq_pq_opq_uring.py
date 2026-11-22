#!/usr/bin/env python3
"""Run resident TurboQuant/PQ/OPQ with per-query O_DIRECT io_uring FP batches.

The common DiskVecLab held-out split and FP-built M=24/efC=200 graph are used.
Compressed database codes stay resident.  Exact FP32 vectors are stored with a
4 KiB header and a per-vector 4 KiB-aligned stride, then read with O_DIRECT.
For every query, all fixed-R reads are submitted in one io_uring batch.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import diskveclab_full_matrix as matrix
import run_faiss_exact_matched_budget_compare as codec_builder
import run_faiss_hnsw_bq_budget_paged_codes as search_runner


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = matrix.RESULT_ROOT / "tq_pq_opq_uring"
EXTERNAL_ROOT = Path(
    "/home/smrc/samsung-nvme/jihyo/diskveclab_tq_pq_opq_uring"
)
INDEX_ROOT = EXTERNAL_ROOT / "indexes"
SIDECAR_ROOT = EXTERNAL_ROOT / "transient_aligned_fp"
SUMMARY = RESULT_ROOT / "summary_t1.csv"
STATUS = RESULT_ROOT / "queue_status.json"
LOCK = RESULT_ROOT / "queue.lock"
PAGE_BYTES = 4096
EXTERNAL_FLOOR_GIB = 25.0
ROOT_FLOOR_GIB = 3.0
DEFAULT_EFS = (
    80, 84, 88, 92, 96, 100, 104, 108, 112, 116,
    120, 128, 144, 160, 192, 224, 256, 320, 512, 1024,
)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    rerank_n: int


METHODS = (
    Method("qdranttq1", "TurboQuant TQ1+ + FP@80", 80),
    Method("pq8m128", "Faiss PQ8-M128 + FP@80", 80),
    Method("opq8m128", "Faiss OPQ-PQ8-M128 + FP@60", 60),
)

# These two datasets already have byte-identical resident codecs from the
# earlier native-code experiment.  Reuse them without copying.
PREBUILT_DATASET_NAMES = {
    "nq_qwen06_1024": "nq_qwen1024",
    "msmarco_qwen1024": "msmarco_qwen1024",
}

# The historical Qwen-1024 codecs are M=128, which is also exactly d/8 for
# 1024-dimensional vectors.  New exact-one-bit names may therefore reuse only
# those byte-identical files; other dimensions always get their own codec.
PREBUILT_METHOD_NAMES = {
    "qdranttq1": "qdranttq1",
    "pq8m128": "pq8m128",
    "opq8m128": "opq8m128",
    "pq8d8": "pq8m128",
    "opq8d8": "opq8m128",
}


class StorageRisk(RuntimeError):
    pass


def gib(value: int) -> float:
    return value / float(1024**3)


def free_gib(path: Path) -> float:
    return gib(shutil.disk_usage(path).free)


def disk_state() -> dict[str, float]:
    return {
        "external_free_gib": free_gib(EXTERNAL_ROOT.parent),
        "root_free_gib": free_gib(Path("/")),
    }


def check_disk(stage: str, additional_external_bytes: int = 0) -> None:
    state = disk_state()
    projected = state["external_free_gib"] - gib(additional_external_bytes)
    if state["root_free_gib"] < ROOT_FLOOR_GIB:
        raise StorageRisk(
            f"root free space below floor at {stage}: "
            f"{state['root_free_gib']:.2f} GiB < {ROOT_FLOOR_GIB:.2f} GiB"
        )
    if projected < EXTERNAL_FLOOR_GIB:
        raise StorageRisk(
            f"external projected free space below floor at {stage}: "
            f"{projected:.2f} GiB < {EXTERNAL_FLOOR_GIB:.2f} GiB"
        )


def write_status(**updates: object) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if STATUS.exists():
        try:
            payload = json.loads(STATUS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(updates)
    payload.update(disk_state())
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATUS)


def monitored_run(
    command: list[str],
    stage: str,
    *,
    threads: int,
    blas_threads: int = 1,
) -> None:
    print("RUN " + " ".join(command), flush=True)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(blas_threads),
        "GOTO_NUM_THREADS": str(blas_threads),
        "MKL_NUM_THREADS": str(blas_threads),
    }
    process = subprocess.Popen(command, cwd=REPO, env=environment)
    while process.poll() is None:
        time.sleep(10)
        try:
            check_disk(stage)
        except StorageRisk:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, command)


def prebuilt_codec_paths(
    dataset_key: str, method_key: str
) -> tuple[Path, Path] | None:
    old_name = PREBUILT_DATASET_NAMES.get(dataset_key)
    old_method = PREBUILT_METHOD_NAMES.get(method_key)
    if old_name is not None and old_method is not None:
        root = search_runner.CODEC_ROOT / old_name
        suffix = "tqdata" if method_key == "qdranttq1" else "faiss"
        return (
            root / f"{old_name}.{old_method}.{suffix}",
            root / f"{old_name}.{old_method}.meta",
        )
    return None


def codec_paths(dataset_key: str, method_key: str) -> tuple[Path, Path]:
    prebuilt = prebuilt_codec_paths(dataset_key, method_key)
    if prebuilt is not None:
        index, meta = prebuilt
        if index.is_file() and (
            method_key != "qdranttq1" or meta.is_file()
        ):
            return prebuilt
    root = INDEX_ROOT / dataset_key
    suffix = "tqdata" if method_key == "qdranttq1" else "faiss"
    return (
        root / f"{dataset_key}.{method_key}.{suffix}",
        root / f"{dataset_key}.{method_key}.meta",
    )


def codec_complete(dataset_key: str, method_key: str) -> bool:
    index, meta = codec_paths(dataset_key, method_key)
    return index.is_file() and index.stat().st_size > 4096 and (
        method_key != "qdranttq1" or (meta.is_file() and meta.stat().st_size > 0)
    )


def reclaim_codec(dataset_key: str, method: Method) -> int:
    """Remove an experiment-owned codec after its raw curve is durable."""
    index, meta = codec_paths(dataset_key, method.key)
    if prebuilt_codec_paths(dataset_key, method.key) == (index, meta):
        return 0
    reclaimed = 0
    for path in (index, meta):
        if path.is_file():
            reclaimed += path.stat().st_size
            path.unlink()
    if reclaimed:
        print(
            f"RECLAIMED codec {dataset_key}/{method.key} bytes={reclaimed}",
            flush=True,
        )
    return reclaimed


def build_codec(dataset_key: str, method: Method, args: argparse.Namespace) -> None:
    if codec_complete(dataset_key, method.key):
        print(f"REUSE codec {dataset_key} {method.key}", flush=True)
        return
    dataset = matrix.registry.DATASETS[dataset_key]
    split = matrix.split_for(dataset_key)
    index, meta = codec_paths(dataset_key, method.key)
    check_disk(f"before codec {dataset_key}/{method.key}", 2 * 1024**3)
    index.parent.mkdir(parents=True, exist_ok=True)
    for stale in index.parent.glob(f"{index.name}.tmp.*"):
        stale.unlink(missing_ok=True)
    for stale in index.parent.glob(f"{meta.name}.tmp.*"):
        stale.unlink(missing_ok=True)
    temporary_index = index.with_name(f"{index.name}.tmp.{os.getpid()}")
    temporary_meta = meta.with_name(f"{meta.name}.tmp.{os.getpid()}")
    dummy_summary = RESULT_ROOT / "build_only" / f"{dataset_key}.{method.key}.summary.csv"
    dummy_query = RESULT_ROOT / "build_only" / f"{dataset_key}.{method.key}.query.csv"
    dummy_summary.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(codec_builder.BINARY),
        "--dataset", dataset_key,
        "--method", method.key,
        "--metric", "ip",
        "--train", str(dataset.train),
        "--query", str(split.test_query),
        "--gt", str(split.test_gt),
        "--index", str(temporary_index),
        "--summary-out", str(dummy_summary),
        "--per-query-out", str(dummy_query),
        "--max-queries", "1",
        "--threads", str(args.build_threads),
        "--train-size", str(args.train_size),
        "--add-batch-size", "4096",
        "--opq-niter", str(args.opq_niter),
        "--build-only",
    ]
    if method.key == "qdranttq1":
        command += ["--meta", str(temporary_meta)]
    try:
        blas_threads = (
            args.opq_blas_threads
            if method.key in {"opq8m128", "opq8d8"}
            else 1
        )
        print(
            f"CODEC_THREADS {dataset_key}/{method.key}: "
            f"omp={args.build_threads} blas={blas_threads}",
            flush=True,
        )
        monitored_run(
            command,
            f"codec {dataset_key}/{method.key}",
            threads=args.build_threads,
            blas_threads=blas_threads,
        )
        if not temporary_index.is_file() or temporary_index.stat().st_size <= 4096:
            raise RuntimeError(f"codec builder did not create {temporary_index}")
        os.replace(temporary_index, index)
        if method.key == "qdranttq1":
            if not temporary_meta.is_file() or temporary_meta.stat().st_size == 0:
                raise RuntimeError(f"codec builder did not create {temporary_meta}")
            os.replace(temporary_meta, meta)
    finally:
        temporary_index.unlink(missing_ok=True)
        temporary_meta.unlink(missing_ok=True)


def sidecar_shape(dataset_key: str) -> tuple[int, int, int, int]:
    dataset = matrix.registry.DATASETS[dataset_key]
    rows, dim = matrix.registry.matrix_shape(dataset.train)
    row_bytes = dim * np.dtype(np.float32).itemsize
    stride = ((row_bytes + PAGE_BYTES - 1) // PAGE_BYTES) * PAGE_BYTES
    return rows, dim, row_bytes, stride


def sidecar_path(dataset_key: str) -> Path:
    return SIDECAR_ROOT / f"{dataset_key}.fp32.row_aligned4k.fbin"


def valid_sidecar(path: Path, dataset_key: str) -> bool:
    rows, dim, _, stride = sidecar_shape(dataset_key)
    if not path.is_file() or path.stat().st_size != PAGE_BYTES + rows * stride:
        return False
    with path.open("rb") as handle:
        header = handle.read(8)
    return header == struct.pack("<ii", rows, dim)


def build_sidecar(dataset_key: str) -> Path:
    path = sidecar_path(dataset_key)
    if valid_sidecar(path, dataset_key):
        print(f"REUSE aligned FP sidecar {path}", flush=True)
        return path
    dataset = matrix.registry.DATASETS[dataset_key]
    rows, dim, row_bytes, stride = sidecar_shape(dataset_key)
    expected = PAGE_BYTES + rows * stride
    check_disk(f"before FP sidecar {dataset_key}", expected)
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f"{path.name}.tmp.*"):
        stale.unlink(missing_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    stride_floats = stride // np.dtype(np.float32).itemsize
    chunk_rows = max(1, min(8192, (64 * 1024**2) // stride))
    print(
        f"BUILD aligned FP {dataset_key}: rows={rows} dim={dim} "
        f"stride={stride} bytes={expected}",
        flush=True,
    )
    try:
        with dataset.train.open("rb", buffering=0) as source, temporary.open(
            "wb", buffering=0
        ) as output:
            source_header = source.read(8)
            if source_header != struct.pack("<ii", rows, dim):
                raise RuntimeError(f"bad fbin header: {dataset.train}")
            payload_bytes = rows * row_bytes
            source_bytes = dataset.train.stat().st_size
            if source_bytes == 8 + payload_bytes:
                source_offset = 8
            elif source_bytes == PAGE_BYTES + payload_bytes:
                source_offset = PAGE_BYTES
            else:
                raise RuntimeError(
                    f"unexpected fbin size for {dataset.train}: "
                    f"{source_bytes} (payload={payload_bytes})"
                )
            # Some SSD-ready inputs use a 4 KiB header before otherwise
            # contiguous rows.  Reading immediately after the first 8 bytes
            # shifts every row and silently corrupts exact-FP reranking.
            source.seek(source_offset)
            output.write(source_header)
            output.write(bytes(PAGE_BYTES - 8))
            completed = 0
            while completed < rows:
                count = min(chunk_rows, rows - completed)
                values = np.fromfile(source, dtype=np.float32, count=count * dim)
                if values.size != count * dim:
                    raise RuntimeError(f"short fbin payload: {dataset.train}")
                values = values.reshape(count, dim)
                norms = np.linalg.norm(values, axis=1, keepdims=True)
                np.maximum(norms, np.float32(1e-12), out=norms)
                values /= norms
                if stride == row_bytes:
                    values.tofile(output)
                else:
                    padded = np.zeros((count, stride_floats), dtype=np.float32)
                    padded[:, :dim] = values
                    padded.tofile(output)
                completed += count
                if completed == rows or completed % 100000 < count:
                    print(f"  sidecar rows {completed}/{rows}", flush=True)
                check_disk(f"FP sidecar {dataset_key}")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if not valid_sidecar(path, dataset_key):
        raise RuntimeError(f"invalid aligned FP sidecar: {path}")
    return path


def raw_path(dataset_key: str, method: Method) -> Path:
    return RESULT_ROOT / "raw" / dataset_key / f"{method.key}.r{method.rerank_n}.csv"


def raw_row_compatible(row: dict[str, str], dataset_key: str, method: Method) -> bool:
    try:
        return (
            row["dataset"] == dataset_key
            and int(row["threads"]) == 1
            and int(row["round"]) == 1
            and int(row["rerank_n"]) == method.rerank_n
            and row["code_storage"] == "resident"
            and row["fp_io_mode"] == "uring-batch"
            and int(row["fp_rerank_qd"]) >= method.rerank_n
        )
    except (KeyError, ValueError):
        return False


def read_raw_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def write_raw_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    if not fieldnames:
        raise RuntimeError(f"missing raw CSV header for {path}")
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def raw_complete(path: Path, dataset_key: str, method: Method, efs: list[int]) -> bool:
    if not path.is_file():
        return False
    try:
        _, rows = read_raw_rows(path)
        return (
            len(rows) == len(efs)
            and {int(row["efSearch"]) for row in rows} == set(efs)
            and all(raw_row_compatible(row, dataset_key, method) for row in rows)
        )
    except (OSError, KeyError, ValueError):
        return False


def run_search(
    dataset_key: str,
    method: Method,
    fp_sidecar: Path,
    args: argparse.Namespace,
) -> Path:
    output = raw_path(dataset_key, method)
    if raw_complete(output, dataset_key, method, args.efs):
        print(f"REUSE result {dataset_key} {method.key}", flush=True)
        return output
    old_fieldnames, old_rows = read_raw_rows(output)
    target_efs = list(args.efs)
    target_set = set(target_efs)
    reusable: dict[int, dict[str, str]] = {}
    for row in old_rows:
        if not raw_row_compatible(row, dataset_key, method):
            continue
        try:
            ef = int(row["efSearch"])
        except (KeyError, ValueError):
            continue
        if ef in target_set:
            if ef in reusable:
                raise RuntimeError(f"duplicate reusable efSearch={ef} in {output}")
            reusable[ef] = row
    missing_efs = [ef for ef in target_efs if ef not in reusable]
    if not missing_efs:
        write_raw_rows(output, old_fieldnames, [reusable[ef] for ef in target_efs])
        if not raw_complete(output, dataset_key, method, target_efs):
            raise RuntimeError(f"failed to normalize reusable result: {output}")
        print(f"NORMALIZED reusable result {dataset_key} {method.key}", flush=True)
        return output
    dataset = matrix.registry.DATASETS[dataset_key]
    split = matrix.split_for(dataset_key)
    codec, meta = codec_paths(dataset_key, method.key)
    output.parent.mkdir(parents=True, exist_ok=True)
    supplement = output.with_name(f"{output.name}.supplement.{os.getpid()}.tmp")
    supplement.unlink(missing_ok=True)
    command = [
        str(search_runner.BINARY),
        "--dataset", dataset_key,
        "--method", method.key,
        "--metric", "ip",
        "--query", str(split.test_query),
        "--gt", str(split.test_gt),
        "--graph", str(dataset.adcbq_index),
        "--codec-index", str(codec),
        "--sidecar", str(EXTERNAL_ROOT / "unused_resident_code_sidecar"),
        "--fp-sidecar", str(fp_sidecar),
        "--out", str(supplement),
        "--max-queries", str(split.test_rows),
        "--rounds", "1",
        "--threads", "1",
        "--rerank-n", str(method.rerank_n),
        "--resident-codes",
        "--fp-io-mode", "uring-batch",
        "--fp-rerank-qd", str(method.rerank_n),
        "--efs", *[str(ef) for ef in missing_efs],
    ]
    if method.key == "qdranttq1":
        command += ["--meta", str(meta)]
    print(
        f"SUPPLEMENT {dataset_key} {method.key}: "
        f"reuse={len(reusable)} missing={missing_efs}",
        flush=True,
    )
    try:
        monitored_run(command, f"search {dataset_key}/{method.key}", threads=1)
        new_fieldnames, new_rows = read_raw_rows(supplement)
        fieldnames = old_fieldnames or new_fieldnames
        if old_fieldnames and new_fieldnames != old_fieldnames:
            raise RuntimeError(
                f"raw CSV schema mismatch for {dataset_key}/{method.key}"
            )
        for row in new_rows:
            if not raw_row_compatible(row, dataset_key, method):
                raise RuntimeError(f"incompatible supplemental row: {row}")
            ef = int(row["efSearch"])
            if ef not in set(missing_efs) or ef in reusable:
                raise RuntimeError(f"unexpected supplemental efSearch={ef}: {row}")
            reusable[ef] = row
        if set(reusable) != target_set:
            raise RuntimeError(
                f"supplement coverage mismatch for {dataset_key}/{method.key}: "
                f"expected={target_efs} got={sorted(reusable)}"
            )
        write_raw_rows(output, fieldnames, [reusable[ef] for ef in target_efs])
    finally:
        supplement.unlink(missing_ok=True)
    if not raw_complete(output, dataset_key, method, target_efs):
        raise RuntimeError(f"incomplete result after successful run: {output}")
    return output


def normalized_rows(path: Path, dataset_key: str, method: Method) -> list[dict[str, object]]:
    dataset = matrix.registry.DATASETS[dataset_key]
    split = matrix.split_for(dataset_key)
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    output: list[dict[str, object]] = []
    for row in raw:
        request_count = float(row["fp_page_reads_per_query"])
        submitted_pages = float(row["fp_read_bytes_per_query"]) / PAGE_BYTES
        pages_per_request = (
            int(row["dim"]) * 4 + PAGE_BYTES - 1
        ) // PAGE_BYTES
        expected_pages = request_count * pages_per_request
        if abs(submitted_pages - expected_pages) > 1e-6 * max(1.0, submitted_pages):
            raise RuntimeError(f"submitted page accounting mismatch: {row}")
        output.append(
            {
                "dataset": dataset_key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method_key": method.key,
                "method": method.label,
                "search_parameter": "efSearch",
                "search_parameter_value": int(row["efSearch"]),
                "efSearch": int(row["efSearch"]),
                "rerank_n": method.rerank_n,
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "search_seconds": float(row["search_seconds"]),
                "rerank_seconds": float(row["rerank_seconds"]),
                "rounds": 1,
                "search_threads": 1,
                "test_queries": split.test_rows,
                "nb": int(row["nb"]),
                "dim": int(row["dim"]),
                "io_requests_per_query_mean": request_count,
                "fp_4k_pages_per_query_mean": submitted_pages,
                "total_4k_pages_per_query_mean": submitted_pages,
                "pages_per_request": submitted_pages / request_count,
                "submitted_bytes_per_query": float(row["fp_read_bytes_per_query"]),
                "storage": "resident compressed HNSW codes + SSD FP32 sidecar",
                "io_engine": "O_DIRECT io_uring one fixed-R batch per query",
                "source_csv": str(path.resolve().relative_to(REPO)),
            }
        )
    return output


def rebuild_summary(dataset_keys: list[str], efs: list[int]) -> int:
    rows: list[dict[str, object]] = []
    for dataset_key in dataset_keys:
        for method in METHODS:
            path = raw_path(dataset_key, method)
            if raw_complete(path, dataset_key, method, efs):
                rows.extend(normalized_rows(path, dataset_key, method))
    if rows:
        dataset_order = {key: i for i, key in enumerate(matrix.CALIBRATION_ROWS)}
        method_order = {method.key: i for i, method in enumerate(METHODS)}
        rows.sort(
            key=lambda row: (
                dataset_order[str(row["dataset"])],
                method_order[str(row["method_key"])],
                int(row["efSearch"]),
            )
        )
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        temporary = SUMMARY.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, SUMMARY)
    return len(rows)


def retry(action, description: str, attempts: int = 3):
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except StorageRisk:
            raise
        except Exception as error:
            print(
                f"ERROR {description} attempt={attempt}/{attempts}: {error}",
                file=sys.stderr,
                flush=True,
            )
            if attempt == attempts:
                raise
            time.sleep(10)
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(matrix.CALIBRATION_ROWS),
        default=list(matrix.CALIBRATION_ROWS),
    )
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--opq-niter", type=int, default=25)
    parser.add_argument("--opq-blas-threads", type=int, default=8)
    parser.add_argument(
        "--reclaim-codecs",
        action="store_true",
        help="delete experiment-owned codecs after their complete raw curve is durable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (
        args.build_threads <= 0
        or args.train_size <= 0
        or args.opq_niter <= 0
        or args.opq_blas_threads <= 0
    ):
        raise SystemExit("positive build parameters required")
    if min(args.efs) < max(method.rerank_n for method in METHODS):
        raise SystemExit("every efSearch must be at least the largest rerank R")
    args.efs = list(dict.fromkeys(args.efs))
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another DiskVecLab TQ/PQ/OPQ queue is already running")

    total_methods = len(args.datasets) * len(METHODS)
    completed_methods = rebuild_summary(args.datasets, args.efs) // len(args.efs)
    write_status(
        phase="compiling",
        datasets=args.datasets,
        efs=args.efs,
        opq_blas_threads=args.opq_blas_threads,
        completed_methods=completed_methods,
        total_methods=total_methods,
    )
    try:
        check_disk("startup")
        codec_builder.compile_binary()
        search_runner.compile_binary()
        # Method-major order makes every TurboQuant curve available before any
        # new PQ work, and every PQ curve available before the expensive OPQ
        # phase.  Sidecars remain transient: one method/dataset at a time.
        for method_index, method in enumerate(METHODS, start=1):
            for dataset_index, dataset_key in enumerate(args.datasets, start=1):
                dataset = matrix.registry.DATASETS[dataset_key]
                matrix.registry.validate(dataset)
                matrix.split_for(dataset_key)
                fp_sidecar: Path | None = None
                try:
                    write_status(
                        phase="building_codecs",
                        current_dataset=dataset_key,
                        current_method=method.key,
                        method_index=method_index,
                        total_methods_per_dataset=len(METHODS),
                        dataset_index=dataset_index,
                        total_datasets=len(args.datasets),
                        completed_methods=completed_methods,
                        total_methods=total_methods,
                    )
                    # A complete raw curve is the durable experiment artifact.
                    # Its codec may have been reclaimed after measurement to
                    # preserve the SSD safety floor, so do not rebuild a codec
                    # merely to discover that no search work remains.
                    if raw_complete(raw_path(dataset_key, method), dataset_key, method, args.efs):
                        print(f"REUSE result {dataset_key} {method.key}", flush=True)
                    else:
                        retry(
                            lambda method=method: build_codec(dataset_key, method, args),
                            f"codec {dataset_key}/{method.key}",
                        )
                        if fp_sidecar is None:
                            write_status(
                                phase="building_fp_sidecar",
                                current_dataset=dataset_key,
                                current_method=method.key,
                            )
                            fp_sidecar = retry(
                                lambda: build_sidecar(dataset_key),
                                f"FP sidecar {dataset_key}",
                            )
                        write_status(
                            phase="searching",
                            current_dataset=dataset_key,
                            current_method=method.key,
                            rerank_n=method.rerank_n,
                            completed_methods=completed_methods,
                            total_methods=total_methods,
                        )
                        retry(
                            lambda method=method: run_search(
                                dataset_key, method, fp_sidecar, args
                            ),
                            f"search {dataset_key}/{method.key}",
                        )
                    completed_methods = rebuild_summary(args.datasets, args.efs) // len(args.efs)
                    write_status(
                        completed_methods=completed_methods,
                        total_methods=total_methods,
                    )
                    if args.reclaim_codecs and raw_complete(
                        raw_path(dataset_key, method), dataset_key, method, args.efs
                    ):
                        reclaim_codec(dataset_key, method)
                finally:
                    owned = sidecar_path(dataset_key)
                    if owned.exists():
                        size = owned.stat().st_size
                        owned.unlink()
                        print(
                            f"REMOVED transient FP sidecar bytes={size}: {owned}",
                            flush=True,
                        )

        rows = rebuild_summary(args.datasets, args.efs)
        expected_rows = total_methods * len(args.efs)
        if rows != expected_rows:
            raise RuntimeError(f"expected {expected_rows} summary rows, found {rows}")
        write_status(
            phase="complete",
            current_dataset=None,
            current_method=None,
            completed_methods=total_methods,
            total_methods=total_methods,
            summary=str(SUMMARY),
        )
        print(f"ALL COMPLETE: {SUMMARY} ({rows} rows)", flush=True)
        return 0
    except StorageRisk as error:
        write_status(phase="stopped_storage_floor", error=str(error))
        print(f"STOPPED FOR STORAGE SAFETY: {error}", file=sys.stderr, flush=True)
        return 75
    except Exception as error:
        write_status(phase="failed", error=repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
