#!/usr/bin/env python3
"""Build and benchmark c16 RaBitQ indexes for the NQ/Quora paper sets.

The build manifest separates input, normalization, centroid training,
centroid assignment, graph/quantization, save, and 4 KiB FP-sidecar time.
Search uses a configurable query-worker count and all queries.  The 1-bit path uses synchronous
O_DIRECT adaptive FP reads from a header-4096 contiguous-row fbin; the 7-bit
path is RAM-only.  Query-level FP-read and latency samples are retained.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import faiss  # type: ignore
import numpy as np

import run_nq_quora_aligned4k_rerank50_80 as previous
import run_v051_ssd_odirect_multidataset as base


REPO = Path(__file__).resolve().parents[1]
BINDING_DIR = REPO / "experiment/rabitq_v051_adaptive_odirect_aligned4k_build/rabitqlib"
if str(BINDING_DIR) not in sys.path:
    sys.path.insert(0, str(BINDING_DIR))
from _rabitqlib import HnswIndex  # type: ignore  # noqa: E402


EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq")
INDEX_ROOT = EXTERNAL_ROOT / "indexes"
FP_ROOT = EXTERNAL_ROOT / "aligned_fp"
RESULT_ROOT = REPO / "experiment/nq_quora_c16_rabitq"

EFS = (64, 128, 256, 512, 1024)
ROUNDS = 3
K = 10
M = 24
EF_CONSTRUCTION = 200
NUM_CLUSTERS = 16
PAGE_BYTES = 4096


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def selected_datasets(keys: list[str] | None) -> list[base.Dataset]:
    # Preserve the historical NQ/Quora default, but allow the runner to be
    # reused for explicitly requested datasets from the shared registry.
    datasets = previous.target_datasets() if not keys else list(base.DATASETS.values())
    if not keys:
        return datasets
    by_key = {dataset.key: dataset for dataset in datasets}
    missing = sorted(set(keys) - set(by_key))
    if missing:
        raise ValueError(f"unknown dataset keys: {missing}; choices={sorted(by_key)}")
    return [by_key[key] for key in keys]


def index_path(dataset: base.Dataset, bits: int) -> Path:
    n, d = base.matrix_shape(dataset.train)
    return INDEX_ROOT / (
        f"{dataset.key}.rabitq_b{bits}_c{NUM_CLUSTERS}.n{n}.d{d}."
        f"M{M}.efC{EF_CONSTRUCTION}.index"
    )


def fp_path(dataset: base.Dataset) -> Path:
    return FP_ROOT / f"{dataset.key}.fp32.header4096.fbin"


def result_dir(dataset: base.Dataset) -> Path:
    return RESULT_ROOT / dataset.key


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_aligned_fp(path: Path, dataset: base.Dataset) -> None:
    n, d = base.matrix_shape(dataset.train)
    expected = PAGE_BYTES + n * d * np.dtype(np.float32).itemsize
    if not path.exists() or path.stat().st_size != expected:
        raise RuntimeError(f"invalid aligned FP file {path}: expected {expected} bytes")
    if base.matrix_shape(path) != (n, d):
        raise RuntimeError(f"bad aligned FP header: {path}")


def create_aligned_fp(path: Path, dataset: base.Dataset, force: bool) -> tuple[str, float]:
    if path.exists() and not force:
        validate_aligned_fp(path, dataset)
        return "reused", 0.0
    if path.exists():
        raise RuntimeError(
            f"--force does not overwrite data destructively; move the old file first: {path}"
        )
    n, d = base.matrix_shape(dataset.train)
    payload_bytes = n * d * np.dtype(np.float32).itemsize
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    started = time.perf_counter()
    copied = 0
    try:
        with dataset.train.open("rb", buffering=0) as source, temporary.open("xb", buffering=0) as dest:
            header = source.read(8)
            if len(header) != 8:
                raise RuntimeError(f"short fbin header: {dataset.train}")
            dest.write(header)
            dest.write(bytes(PAGE_BYTES - 8))
            while copied < payload_bytes:
                block = source.read(min(16 * 1024 * 1024, payload_bytes - copied))
                if not block:
                    raise RuntimeError(f"short fbin payload: {dataset.train}")
                dest.write(block)
                copied += len(block)
            dest.flush()
            os.fsync(dest.fileno())
            if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
                os.posix_fadvise(dest.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    seconds = time.perf_counter() - started
    validate_aligned_fp(path, dataset)
    return "built", seconds


def normalize_rows(x: np.ndarray, batch: int = 65536) -> None:
    for start in range(0, len(x), batch):
        block = x[start : start + batch]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        block /= norms


def train_and_assign(
    train: np.ndarray, threads: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    faiss.omp_set_num_threads(threads)
    quantizer = faiss.index_factory(
        train.shape[1], f"IVF{NUM_CLUSTERS},Flat", faiss.METRIC_INNER_PRODUCT
    )
    quantizer.verbose = True
    quantizer.cp.seed = 100
    started = time.perf_counter()
    quantizer.train(train)
    train_seconds = time.perf_counter() - started
    centroids = np.ascontiguousarray(
        quantizer.quantizer.reconstruct_n(0, quantizer.nlist), dtype=np.float32
    )
    started = time.perf_counter()
    _, ids = quantizer.quantizer.search(train, 1)
    assign_seconds = time.perf_counter() - started
    cluster_ids = np.ascontiguousarray(ids.reshape(-1), dtype=np.uint32)
    return centroids, cluster_ids, train_seconds, assign_seconds


@dataclass
class BuildRecord:
    dataset: str
    corpus: str
    model: str
    nb: int
    dim: int
    num_clusters: int
    M: int
    efConstruction: int
    build_threads: int
    started_at: str
    finished_at: str
    input_read_seconds: float
    normalization_seconds: float
    centroid_train_seconds: float
    centroid_assign_seconds: float
    aligned_fp_status: str
    aligned_fp_seconds: float
    aligned_fp_bytes: int
    b1_graph_quantize_seconds: float
    b1_save_seconds: float
    b1_index_bytes: int
    b1_index_path: str
    b7_graph_quantize_seconds: float
    b7_save_seconds: float
    b7_index_bytes: int
    b7_index_path: str
    shared_preprocess_seconds: float
    total_pipeline_seconds: float


def build_dataset(dataset: base.Dataset, args: argparse.Namespace) -> BuildRecord | dict[str, object]:
    out = result_dir(dataset)
    manifest_path = out / "build_manifest.json"
    b1_path = index_path(dataset, 1)
    b7_path = index_path(dataset, 7)
    sidecar = fp_path(dataset)
    only_bits = tuple(getattr(args, "only_bits", (1, 7)))
    if not only_bits or any(bits not in (1, 7) for bits in only_bits):
        raise ValueError(f"invalid only_bits={only_bits}")
    selected_paths = tuple(index_path(dataset, bits) for bits in only_bits)
    if (
        only_bits == (1, 7)
        and manifest_path.exists()
        and b1_path.exists()
        and b7_path.exists()
        and sidecar.exists()
        and not args.force
    ):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"SKIP complete c16 build: {dataset.key}", flush=True)
        return BuildRecord(**payload)
    for path in selected_paths:
        if path.exists():
            raise RuntimeError(f"refusing to mix/rewrite an existing partial index: {path}")

    out.mkdir(parents=True, exist_ok=True)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    pipeline_started_at = now()
    pipeline_started = time.perf_counter()

    fp_status, fp_seconds = create_aligned_fp(sidecar, dataset, args.force)
    read_started = time.perf_counter()
    train = base.read_fbin(dataset.train)
    input_read_seconds = time.perf_counter() - read_started
    n, d = train.shape

    norm_started = time.perf_counter()
    normalize_rows(train)
    normalization_seconds = time.perf_counter() - norm_started
    centroids, cluster_ids, centroid_train_seconds, centroid_assign_seconds = train_and_assign(
        train, args.build_threads
    )
    shared_seconds = (
        fp_seconds
        + input_read_seconds
        + normalization_seconds
        + centroid_train_seconds
        + centroid_assign_seconds
    )

    method_times: dict[int, tuple[float, float, int, str]] = {}
    for bits, path in ((bits, index_path(dataset, bits)) for bits in only_bits):
        print(
            f"BUILD {dataset.key} RaBitQ b{bits} c16 n={n:,} d={d} "
            f"M={M} efC={EF_CONSTRUCTION} threads={args.build_threads}",
            flush=True,
        )
        index = HnswIndex(
            dim=d,
            max_elements=n,
            M=M,
            ef_construction=EF_CONSTRUCTION,
            nbits=bits,
            metric="ip",
            random_seed=100,
        )
        started = time.perf_counter()
        index.build(
            train,
            centroids,
            cluster_ids,
            num_threads=args.build_threads,
            fast_quantization=False,
        )
        graph_seconds = time.perf_counter() - started
        started = time.perf_counter()
        index.save(str(path))
        save_seconds = time.perf_counter() - started
        method_times[bits] = (graph_seconds, save_seconds, path.stat().st_size, str(path))
        print(
            f"BUILT {dataset.key} b{bits}: graph+quantize={graph_seconds:.3f}s "
            f"save={save_seconds:.3f}s bytes={path.stat().st_size}",
            flush=True,
        )
        del index
        gc.collect()

    total_seconds = time.perf_counter() - pipeline_started
    if only_bits != (1, 7):
        bits = only_bits[0]
        graph_seconds, save_seconds, index_bytes, saved_path = method_times[bits]
        payload: dict[str, object] = {
            "dataset": dataset.key,
            "corpus": dataset.corpus,
            "model": dataset.model,
            "nb": n,
            "dim": d,
            "num_clusters": NUM_CLUSTERS,
            "M": M,
            "efConstruction": EF_CONSTRUCTION,
            "build_threads": args.build_threads,
            "bits": bits,
            "started_at": pipeline_started_at,
            "finished_at": now(),
            "input_read_seconds": input_read_seconds,
            "normalization_seconds": normalization_seconds,
            "centroid_train_seconds": centroid_train_seconds,
            "centroid_assign_seconds": centroid_assign_seconds,
            "aligned_fp_status": fp_status,
            "aligned_fp_seconds": fp_seconds,
            "aligned_fp_bytes": sidecar.stat().st_size,
            "graph_quantize_seconds": graph_seconds,
            "save_seconds": save_seconds,
            "index_bytes": index_bytes,
            "index_path": saved_path,
            "shared_preprocess_seconds": shared_seconds,
            "total_pipeline_seconds": total_seconds,
        }
        (out / f"build_manifest.b{bits}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        del train, centroids, cluster_ids
        gc.collect()
        return payload

    record = BuildRecord(
        dataset=dataset.key,
        corpus=dataset.corpus,
        model=dataset.model,
        nb=n,
        dim=d,
        num_clusters=NUM_CLUSTERS,
        M=M,
        efConstruction=EF_CONSTRUCTION,
        build_threads=args.build_threads,
        started_at=pipeline_started_at,
        finished_at=now(),
        input_read_seconds=input_read_seconds,
        normalization_seconds=normalization_seconds,
        centroid_train_seconds=centroid_train_seconds,
        centroid_assign_seconds=centroid_assign_seconds,
        aligned_fp_status=fp_status,
        aligned_fp_seconds=fp_seconds,
        aligned_fp_bytes=sidecar.stat().st_size,
        b1_graph_quantize_seconds=method_times[1][0],
        b1_save_seconds=method_times[1][1],
        b1_index_bytes=method_times[1][2],
        b1_index_path=method_times[1][3],
        b7_graph_quantize_seconds=method_times[7][0],
        b7_save_seconds=method_times[7][1],
        b7_index_bytes=method_times[7][2],
        b7_index_path=method_times[7][3],
        shared_preprocess_seconds=shared_seconds,
        total_pipeline_seconds=total_seconds,
    )
    manifest_path.write_text(json.dumps(asdict(record), indent=2) + "\n", encoding="utf-8")
    del train, centroids, cluster_ids
    gc.collect()
    return record


def recall_at_10(ids: np.ndarray, gt: np.ndarray) -> float:
    return base.recall_at_10(np.asarray(ids), gt)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def distribution(prefix: str, values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p50": percentile(values, 50),
        f"{prefix}_p95": percentile(values, 95),
        f"{prefix}_p99": percentile(values, 99),
        f"{prefix}_max": float(np.max(values)),
    }


def search_complete(dataset: base.Dataset, rounds: int, search_threads: int) -> bool:
    search_path = result_dir(dataset) / "search_results.csv"
    detail_path = result_dir(dataset) / "per_query_fp_reads.csv"
    try:
        with search_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with detail_path.open(newline="", encoding="utf-8") as handle:
            detail_count = sum(1 for _ in csv.DictReader(handle))
        nq, _ = base.matrix_shape(dataset.query)
        expected_pairs = {
            (family, str(ef))
            for family in (
                "rabitq_b1_c16_adaptive_odirect_aligned4k",
                "rabitq_b7_c16_ram",
            )
            for ef in EFS
        }
        actual_pairs = {(row["family"], row["efSearch"]) for row in rows}
        return (
            len(rows) == len(expected_pairs)
            and actual_pairs == expected_pairs
            and all(int(row["rounds"]) == rounds for row in rows)
            and all(int(row["search_threads"]) == search_threads for row in rows)
            and all(int(row["nq"]) == nq for row in rows)
            and all(int(row["num_clusters"]) == NUM_CLUSTERS for row in rows)
            and detail_count == nq * len(EFS) * rounds
        )
    except (OSError, KeyError, ValueError):
        return False


def search_dataset(dataset: base.Dataset, args: argparse.Namespace) -> list[dict[str, object]]:
    only_bits = tuple(getattr(args, "only_bits", (1, 7)))
    if only_bits == (7,):
        return search_dataset_b7(dataset, args)
    if only_bits not in ((1,), (1, 7)):
        raise ValueError(f"invalid only_bits={only_bits}")
    if only_bits == (1, 7) and search_complete(dataset, args.rounds, args.search_threads) and not args.force:
        print(f"SKIP complete c16 search: {dataset.key}", flush=True)
        with (result_dir(dataset) / "search_results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    b1_path = index_path(dataset, 1)
    b7_path = index_path(dataset, 7)
    sidecar = fp_path(dataset)
    required = (b1_path, sidecar) if only_bits == (1,) else (b1_path, b7_path, sidecar)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    validate_aligned_fp(sidecar, dataset)
    queries = base.read_fbin(dataset.query)
    normalize_rows(queries)
    gt = base.read_ibin(dataset.gt, len(queries))
    n, d = base.matrix_shape(dataset.train)
    fp_pages_per_vector = math.ceil(d * 4 / PAGE_BYTES)
    rows: list[dict[str, object]] = []
    per_query: list[dict[str, object]] = []

    load_started = time.perf_counter()
    b1 = HnswIndex.load(str(b1_path))
    b1_load_seconds = time.perf_counter() - load_started
    if b1.nbits != 1 or b1.num_clusters != NUM_CLUSTERS:
        raise RuntimeError(f"unexpected 1-bit index metadata: {b1_path}")
    for ef in EFS:
        recalls: list[float] = []
        elapsed_values: list[float] = []
        fp_samples: list[np.ndarray] = []
        latency_samples: list[np.ndarray] = []
        bytes_values: list[int] = []
        requests_values: list[int] = []
        offsets: list[int] = []
        for round_id in range(1, args.rounds + 1):
            started = time.perf_counter()
            ids, _, stats = b1.search_1bit_adaptive_fp_odirect(
                queries, str(sidecar), k=K, ef=ef, num_threads=args.search_threads
            )
            elapsed = time.perf_counter() - started
            fp_reads = np.asarray(stats["fp_read_count"], dtype=np.uint64)
            latency_ns = np.asarray(stats["query_latency_ns"], dtype=np.uint64)
            if len(fp_reads) != len(queries) or len(latency_ns) != len(queries):
                raise RuntimeError("query-level stats length mismatch")
            if int(stats["fbin_data_offset"]) != PAGE_BYTES:
                raise RuntimeError(f"expected fbin offset 4096, got {stats['fbin_data_offset']}")
            recall = recall_at_10(np.asarray(ids), gt)
            recalls.append(recall)
            elapsed_values.append(elapsed)
            fp_samples.append(fp_reads)
            latency_samples.append(latency_ns)
            bytes_values.append(int(stats["odirect_read_bytes"]))
            requests_values.append(int(stats["odirect_read_requests"]))
            offsets.append(int(stats["fbin_data_offset"]))
            for query_id, (reads, latency) in enumerate(zip(fp_reads, latency_ns)):
                per_query.append(
                    {
                        "dataset": dataset.key,
                        "method": "rabitq_b1_c16_adaptive_fp_odirect_aligned4k",
                        "efSearch": ef,
                        "round": round_id,
                        "query_id": query_id,
                        "fp_vectors_read": int(reads),
                        "fp_4k_pages_read": int(reads) * fp_pages_per_vector,
                        "fp_payload_bytes": int(reads) * d * 4,
                        "query_latency_ns": int(latency),
                    }
                )
            print(
                f"SEARCH {dataset.key} b1 c16 ef={ef} round={round_id}/{args.rounds} "
                f"R@10={recall:.6f} QPS={len(queries)/elapsed:.2f} "
                f"FP/q={float(np.mean(fp_reads)):.2f}",
                flush=True,
            )
        all_fp = np.concatenate(fp_samples)
        all_fp_pages = all_fp * fp_pages_per_vector
        all_latency_ms = np.concatenate(latency_samples).astype(np.float64) / 1e6
        mean_seconds = statistics.fmean(elapsed_values)
        mean_bytes = statistics.fmean(bytes_values)
        row: dict[str, object] = {
            "dataset": dataset.key,
            "corpus": dataset.corpus,
            "model": dataset.model,
            "method": "RaBitQ 1-bit c16 + adaptive FP",
            "family": "rabitq_b1_c16_adaptive_odirect_aligned4k",
            "M": M,
            "efConstruction": EF_CONSTRUCTION,
            "num_clusters": NUM_CLUSTERS,
            "efSearch": ef,
            "k": K,
            "nq": len(queries),
            "nb": n,
            "dim": d,
            "bits": 1,
            "rounds": args.rounds,
            "search_threads": args.search_threads,
            "explicit_warmup": False,
            "rerank_policy": "adaptive_lower_bound",
            "storage": "SSD O_DIRECT",
            "io_engine": "sync pread O_DIRECT",
            "fp_store_layout": "header4096_rows_contiguous",
            "fbin_data_offset": min(offsets),
            "load_seconds": b1_load_seconds,
            "recall_at_10": statistics.fmean(recalls),
            "qps": len(queries) / mean_seconds,
            "total_seconds_mean": mean_seconds,
            "odirect_read_bytes_total_mean": mean_bytes,
            "odirect_4k_pages_per_query": mean_bytes / PAGE_BYTES / len(queries),
            "odirect_requests_per_query": statistics.fmean(requests_values) / len(queries),
            "fp_4k_pages_per_vector": fp_pages_per_vector,
            "index_bytes": b1_path.stat().st_size,
            "index_path": str(b1_path),
            "fp_path": str(sidecar),
        }
        row.update(distribution("fp_vectors_per_query", all_fp))
        row.update(distribution("fp_4k_pages_per_query", all_fp_pages))
        row.update(distribution("latency_ms", all_latency_ms))
        rows.append(row)
    del b1
    gc.collect()

    if only_bits == (1,):
        write_csv(result_dir(dataset) / "search_results.csv", rows)
        write_csv(result_dir(dataset) / "per_query_fp_reads.csv", per_query)
        del queries, gt
        gc.collect()
        return rows

    load_started = time.perf_counter()
    b7 = HnswIndex.load(str(b7_path))
    b7_load_seconds = time.perf_counter() - load_started
    if b7.nbits != 7 or b7.num_clusters != NUM_CLUSTERS:
        raise RuntimeError(f"unexpected 7-bit index metadata: {b7_path}")
    for ef in EFS:
        recalls: list[float] = []
        elapsed_values: list[float] = []
        for round_id in range(1, args.rounds + 1):
            started = time.perf_counter()
            ids, _ = b7.search(
                queries, k=K, ef=ef, num_threads=args.search_threads
            )
            elapsed = time.perf_counter() - started
            recall = recall_at_10(np.asarray(ids), gt)
            recalls.append(recall)
            elapsed_values.append(elapsed)
            print(
                f"SEARCH {dataset.key} b7 c16 ef={ef} round={round_id}/{args.rounds} "
                f"R@10={recall:.6f} QPS={len(queries)/elapsed:.2f}",
                flush=True,
            )
        mean_seconds = statistics.fmean(elapsed_values)
        rows.append(
            {
                "dataset": dataset.key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method": "RaBitQ 7-bit c16 RAM",
                "family": "rabitq_b7_c16_ram",
                "M": M,
                "efConstruction": EF_CONSTRUCTION,
                "num_clusters": NUM_CLUSTERS,
                "efSearch": ef,
                "k": K,
                "nq": len(queries),
                "nb": n,
                "dim": d,
                "bits": 7,
                "rounds": args.rounds,
                "search_threads": args.search_threads,
                "explicit_warmup": False,
                "rerank_policy": "none",
                "storage": "RAM-only",
                "io_engine": "none",
                "fp_vectors_per_query_mean": 0,
                "fp_vectors_per_query_min": 0,
                "fp_vectors_per_query_p50": 0,
                "fp_vectors_per_query_p95": 0,
                "fp_vectors_per_query_p99": 0,
                "fp_vectors_per_query_max": 0,
                "fp_4k_pages_per_query_mean": 0,
                "fp_4k_pages_per_query_min": 0,
                "fp_4k_pages_per_query_p50": 0,
                "fp_4k_pages_per_query_p95": 0,
                "fp_4k_pages_per_query_p99": 0,
                "fp_4k_pages_per_query_max": 0,
                "odirect_read_bytes_total_mean": 0,
                "odirect_4k_pages_per_query": 0,
                "odirect_requests_per_query": 0,
                "load_seconds": b7_load_seconds,
                "recall_at_10": statistics.fmean(recalls),
                "qps": len(queries) / mean_seconds,
                "total_seconds_mean": mean_seconds,
                "index_bytes": b7_path.stat().st_size,
                "index_path": str(b7_path),
            }
        )
    write_csv(result_dir(dataset) / "search_results.csv", rows)
    write_csv(result_dir(dataset) / "per_query_fp_reads.csv", per_query)
    del b7, queries, gt
    gc.collect()
    return rows


def search_dataset_b7(
    dataset: base.Dataset, args: argparse.Namespace
) -> list[dict[str, object]]:
    """Measure only the 7-bit c16 arm and merge it with a completed b1 arm."""
    b7_path = index_path(dataset, 7)
    if not b7_path.exists():
        raise FileNotFoundError(b7_path)
    queries = base.read_fbin(dataset.query)
    normalize_rows(queries)
    gt = base.read_ibin(dataset.gt, len(queries))
    n, d = base.matrix_shape(dataset.train)

    existing_rows: list[dict[str, object]] = []
    search_path = result_dir(dataset) / "search_results.csv"
    if search_path.exists():
        with search_path.open(newline="", encoding="utf-8") as handle:
            existing_rows = [
                dict(row)
                for row in csv.DictReader(handle)
                if row.get("family") != "rabitq_b7_c16_ram"
            ]

    rows: list[dict[str, object]] = []
    load_started = time.perf_counter()
    b7 = HnswIndex.load(str(b7_path))
    b7_load_seconds = time.perf_counter() - load_started
    if b7.nbits != 7 or b7.num_clusters != NUM_CLUSTERS:
        raise RuntimeError(f"unexpected 7-bit index metadata: {b7_path}")
    for ef in EFS:
        recalls: list[float] = []
        elapsed_values: list[float] = []
        for round_id in range(1, args.rounds + 1):
            started = time.perf_counter()
            ids, _ = b7.search(
                queries, k=K, ef=ef, num_threads=args.search_threads
            )
            elapsed = time.perf_counter() - started
            recall = recall_at_10(np.asarray(ids), gt)
            recalls.append(recall)
            elapsed_values.append(elapsed)
            print(
                f"SEARCH {dataset.key} b7 c16 ef={ef} round={round_id}/{args.rounds} "
                f"R@10={recall:.6f} QPS={len(queries)/elapsed:.2f}",
                flush=True,
            )
        mean_seconds = statistics.fmean(elapsed_values)
        rows.append(
            {
                "dataset": dataset.key,
                "corpus": dataset.corpus,
                "model": dataset.model,
                "method": "RaBitQ 7-bit c16 RAM",
                "family": "rabitq_b7_c16_ram",
                "M": M,
                "efConstruction": EF_CONSTRUCTION,
                "num_clusters": NUM_CLUSTERS,
                "efSearch": ef,
                "k": K,
                "nq": len(queries),
                "nb": n,
                "dim": d,
                "bits": 7,
                "rounds": args.rounds,
                "search_threads": args.search_threads,
                "explicit_warmup": False,
                "rerank_policy": "none",
                "storage": "RAM-only",
                "io_engine": "none",
                "fp_vectors_per_query_mean": 0,
                "fp_vectors_per_query_min": 0,
                "fp_vectors_per_query_p50": 0,
                "fp_vectors_per_query_p95": 0,
                "fp_vectors_per_query_p99": 0,
                "fp_vectors_per_query_max": 0,
                "fp_4k_pages_per_query_mean": 0,
                "fp_4k_pages_per_query_min": 0,
                "fp_4k_pages_per_query_p50": 0,
                "fp_4k_pages_per_query_p95": 0,
                "fp_4k_pages_per_query_p99": 0,
                "fp_4k_pages_per_query_max": 0,
                "odirect_read_bytes_total_mean": 0,
                "odirect_4k_pages_per_query": 0,
                "odirect_requests_per_query": 0,
                "load_seconds": b7_load_seconds,
                "recall_at_10": statistics.fmean(recalls),
                "qps": len(queries) / mean_seconds,
                "total_seconds_mean": mean_seconds,
                "index_bytes": b7_path.stat().st_size,
                "index_path": str(b7_path),
            }
        )
    combined = [*existing_rows, *rows]
    write_csv(search_path, combined)
    per_query_path = result_dir(dataset) / "per_query_fp_reads.csv"
    if not per_query_path.exists():
        raise FileNotFoundError(
            f"missing completed b1 query details before b7 merge: {per_query_path}"
        )
    del b7, queries, gt
    gc.collect()
    return combined


def combine(datasets: Iterable[base.Dataset]) -> None:
    build_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    for dataset in datasets:
        manifest = result_dir(dataset) / "build_manifest.json"
        search_csv = result_dir(dataset) / "search_results.csv"
        if manifest.exists():
            build_rows.append(json.loads(manifest.read_text(encoding="utf-8")))
        if search_csv.exists():
            with search_csv.open(newline="", encoding="utf-8") as handle:
                search_rows.extend(dict(row) for row in csv.DictReader(handle))
    if build_rows:
        write_csv(RESULT_ROOT / "build_times.csv", build_rows)
    if search_rows:
        write_csv(RESULT_ROOT / "all_results.csv", search_rows)


def main() -> None:
    global EXTERNAL_ROOT, FP_ROOT, INDEX_ROOT, RESULT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("build", "search", "report", "all"), default="all")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.search_threads <= 0:
        raise ValueError("--search-threads must be positive")
    RESULT_ROOT = args.result_root.resolve()
    EXTERNAL_ROOT = args.external_root.resolve()
    INDEX_ROOT = EXTERNAL_ROOT / "indexes"
    FP_ROOT = EXTERNAL_ROOT / "aligned_fp"
    datasets = selected_datasets(args.datasets)
    for dataset in datasets:
        print(f"[{now()}] DATASET {dataset.key} {dataset.model}", flush=True)
        if args.phase in {"build", "all"}:
            build_dataset(dataset, args)
        if args.phase in {"search", "all"}:
            search_dataset(dataset, args)
    combine(datasets)


if __name__ == "__main__":
    main()
