#!/usr/bin/env python3
"""Run official HNSW RaBitQ-1 with native adaptive FP on the held-out matrix.

The compressed RaBitQ index and HNSW graph are resident.  During search the
official adaptive lower-bound policy synchronously reads exact FP32 vectors
from the source fbin with O_DIRECT.  All searches use the common held-out
query split and one search worker.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


PYTHON311 = Path("/home/jihyo/anaconda3/envs/hnsw/bin/python3.11")
if sys.version_info[:2] != (3, 11) and PYTHON311.is_file():
    os.execv(str(PYTHON311), [str(PYTHON311), *sys.argv])

import numpy as np

import diskveclab_full_matrix as matrix
import run_v051_ssd_odirect_multidataset as rabitq


RESULT_ROOT = matrix.RESULT_ROOT / "rabitq1_c16_adaptive"
SUMMARY = RESULT_ROOT / "summary_t1.csv"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/diskveclab_rabitq_c16_adaptive")
INDEX_ROOT = EXTERNAL_ROOT / "indexes"
CLUSTER_ROOT = EXTERNAL_ROOT / "clustering"
CLUSTER_HELPER = Path(__file__).resolve().parent / "cluster_fbin_ivf256.py"
CLUSTER_PYTHON = Path("/home/jihyo/anaconda3/bin/python")
NUM_CLUSTERS = 16
M = 24
EF_CONSTRUCTION = 200
DEFAULT_EFS = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)

NQ_QUORA_C16 = Path("/home/smrc/samsung-nvme/jihyo/nq_quora_c16_rabitq/indexes")
PREBUILT_INDEXES = {
    "nq_bge_m3_vanilla": NQ_QUORA_C16
    / "nq_bge_m3_vanilla.rabitq_b1_c16.n1000000.d1024.M24.efC200.index",
    "nq_qwen06_1024": NQ_QUORA_C16
    / "nq_qwen06_1024.rabitq_b1_c16.n1000000.d1024.M24.efC200.index",
    "quora_bge_m3_vanilla": NQ_QUORA_C16
    / "quora_bge_m3_vanilla.rabitq_b1_c16.n522931.d1024.M24.efC200.index",
    "quora_qwen1024": NQ_QUORA_C16
    / "quora_qwen1024.rabitq_b1_c16.n522931.d1024.M24.efC200.index",
    "msmarco_qwen1024": NQ_QUORA_C16
    / "msmarco_qwen1024.rabitq_b1_c16.n1000000.d1024.M24.efC200.index",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def c16_index_path(dataset: object) -> Path:
    if dataset.key in PREBUILT_INDEXES:
        return PREBUILT_INDEXES[dataset.key]
    n, d = rabitq.matrix_shape(dataset.train)
    return INDEX_ROOT / (
        f"{dataset.key}.rabitq_b1_c16.n{n}.d{d}.M{M}.efC{EF_CONSTRUCTION}.index"
    )


def c16_dataset(dataset: object) -> object:
    return replace(dataset, b1_index=c16_index_path(dataset))


def c16_clusters(dataset: object, build_threads: int) -> tuple[np.ndarray, np.ndarray, float]:
    root = CLUSTER_ROOT / dataset.key
    centroids_path = root / "centroids_c16_seed100.npy"
    assignments_path = root / "assignments_c16_seed100.npy"
    n, d = rabitq.matrix_shape(dataset.train)
    if centroids_path.is_file() and assignments_path.is_file():
        centroids = np.ascontiguousarray(np.load(centroids_path), dtype=np.float32)
        assignments = np.ascontiguousarray(np.load(assignments_path), dtype=np.uint32)
        if centroids.shape == (NUM_CLUSTERS, d) and assignments.shape == (n,):
            return centroids, assignments, 0.0
        raise RuntimeError(f"invalid cached C16 clustering files under {root}")
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    subprocess.run(
        [
            str(CLUSTER_PYTHON),
            str(CLUSTER_HELPER),
            "--train",
            str(dataset.train),
            "--centroids",
            str(centroids_path),
            "--assignments",
            str(assignments_path),
            "--threads",
            str(build_threads),
            "--clusters",
            str(NUM_CLUSTERS),
            "--seed",
            "100",
        ],
        check=True,
    )
    elapsed = time.perf_counter() - started
    centroids = np.ascontiguousarray(np.load(centroids_path), dtype=np.float32)
    assignments = np.ascontiguousarray(np.load(assignments_path), dtype=np.uint32)
    if centroids.shape != (NUM_CLUSTERS, d) or assignments.shape != (n,):
        raise RuntimeError(f"bad generated C16 clustering files under {root}")
    return centroids, assignments, elapsed


def ensure_c16_index(dataset: object, build_threads: int) -> Path:
    path = c16_index_path(dataset)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    started_at = now()
    pipeline_started = time.perf_counter()
    centroids, cluster_ids, clustering_seconds = c16_clusters(dataset, build_threads)
    read_started = time.perf_counter()
    train = rabitq.read_fbin(dataset.train)
    read_seconds = time.perf_counter() - read_started
    normalize_started = time.perf_counter()
    rabitq.normalize_rows(train)
    normalize_seconds = time.perf_counter() - normalize_started

    HnswIndex = rabitq.import_rabitq()
    index = HnswIndex(
        dim=train.shape[1],
        max_elements=train.shape[0],
        M=M,
        ef_construction=EF_CONSTRUCTION,
        nbits=1,
        metric="ip",
        random_seed=100,
    )
    print(
        f"BUILD {dataset.key} RaBitQ b1 c16 n={len(train):,} d={train.shape[1]} "
        f"M={M} efC={EF_CONSTRUCTION} threads={build_threads}",
        flush=True,
    )
    build_started = time.perf_counter()
    index.build(
        train,
        centroids,
        cluster_ids,
        num_threads=build_threads,
        fast_quantization=False,
    )
    build_seconds = time.perf_counter() - build_started
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    save_started = time.perf_counter()
    index.save(str(temporary))
    os.replace(temporary, path)
    save_seconds = time.perf_counter() - save_started
    manifest = {
        "dataset": dataset.key,
        "nb": len(train),
        "dim": train.shape[1],
        "num_clusters": NUM_CLUSTERS,
        "M": M,
        "efConstruction": EF_CONSTRUCTION,
        "bits": 1,
        "build_threads": build_threads,
        "started_at": started_at,
        "finished_at": now(),
        "input_read_seconds": read_seconds,
        "normalization_seconds": normalize_seconds,
        "clustering_subprocess_seconds": clustering_seconds,
        "graph_quantize_seconds": build_seconds,
        "save_seconds": save_seconds,
        "total_seconds": time.perf_counter() - pipeline_started,
        "index_bytes": path.stat().st_size,
        "index_path": str(path),
    }
    manifest_path = RESULT_ROOT / "build_manifests" / f"{dataset.key}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"BUILT {path} bytes={path.stat().st_size}", flush=True)
    del index, train, centroids, cluster_ids
    gc.collect()
    return path


def result_path(dataset: object) -> Path:
    return RESULT_ROOT / f"{dataset.key}.rabitq_b1_adaptive_odirect.M24.efC200.csv"


def rebuild_summary(dataset_keys: list[str], efs: list[int]) -> int:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for key in dataset_keys:
        heldout = matrix.heldout_dataset(key)
        path = result_path(heldout)
        if not rabitq.complete_csv(path, efs, search_threads=1):
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    if not rows or fieldnames is None:
        return 0
    rows.sort(key=lambda row: (row["dataset"], int(row["efSearch"])))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    temporary = SUMMARY.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(SUMMARY)
    print(f"WROTE {SUMMARY} rows={len(rows)}", flush=True)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(matrix.CALIBRATION_ROWS),
        default=list(matrix.CALIBRATION_ROWS),
    )
    parser.add_argument("--efs", nargs="+", type=int, default=list(DEFAULT_EFS))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--build-threads", type=int, default=24)
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--search-cpu", type=int, default=46)
    parser.add_argument("--force-search", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.search_threads != 1:
        parser.error("rounds must be positive and the full matrix requires T=1")
    original_affinity = os.sched_getaffinity(0)
    if args.search_cpu not in original_affinity:
        parser.error(
            f"search CPU {args.search_cpu} is outside allowed affinity "
            f"{sorted(original_affinity)}"
        )
    efs = list(dict.fromkeys(args.efs))

    old_result = rabitq.RESULT_DIR
    rabitq.RESULT_DIR = RESULT_ROOT
    run_args = SimpleNamespace(
        efs=efs,
        rounds=args.rounds,
        max_queries=0,
        build_threads=args.build_threads,
        search_threads=args.search_threads,
        adc_rerank_n=80,
        parts=("b1",),
        force_search=args.force_search,
        rebuild=args.rebuild,
        num_clusters=NUM_CLUSTERS,
        resume_partial=True,
    )
    try:
        for key in args.datasets:
            original = rabitq.DATASETS[key]
            heldout = c16_dataset(matrix.heldout_dataset(key))
            rabitq.validate(heldout)
            output = result_path(heldout)
            if (
                rabitq.complete_csv(output, efs, args.search_threads, args.rounds)
                and not args.force_search
            ):
                print(f"SKIP complete {output}", flush=True)
                rebuild_summary(args.datasets, efs)
                continue
            print(
                f"DATASET {key}: heldout_queries={matrix.split_for(key).test_rows} "
                f"b1_c16_index={heldout.b1_index}",
                flush=True,
            )
            os.sched_setaffinity(0, original_affinity)
            ensure_c16_index(original, args.build_threads)
            os.sched_setaffinity(0, {args.search_cpu})
            produced = rabitq.run_rabitq(heldout, 1, run_args)
            if produced != output:
                raise RuntimeError(f"unexpected result path: {produced} != {output}")
            rebuild_summary(args.datasets, efs)
    finally:
        os.sched_setaffinity(0, original_affinity)
        rabitq.RESULT_DIR = old_result

    found = rebuild_summary(args.datasets, efs)
    expected = len(args.datasets) * len(efs)
    if found != expected:
        raise RuntimeError(f"incomplete summary: expected {expected}, found {found}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
