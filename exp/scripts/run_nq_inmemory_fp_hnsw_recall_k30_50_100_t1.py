#!/usr/bin/env python3
"""NQ Qwen-1024 in-memory FP32 HNSW Recall@30/50/100 benchmark.

The benchmark uses the exact same seed-11 held-out 2,452-query split and
top-100 ground truth as the existing ADC-BQ/RaBitQ/TurboQuant Recall@k
experiment.  A full-precision-built M=24, efConstruction=200 topology is
materialized once as a FAISS binary-HNSW graph.  For measurement, the graph
topology is copied into IndexHNSWFlat and all normalized FP32 vectors are kept
in process-owned RAM.  Each k is measured separately with one search thread
and three rounds.

To coexist with another experiment, graph construction is pinned to four
physical cores on NUMA node 1 and all measured searches are pinned to one of
those cores.  The lane is intentionally disjoint from the default unpinned
runner and is recorded in the manifest.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiment/nq_inmemory_fp_hnsw_recall_k30_50_100_t1"
SPLIT = (
    REPO
    / "experiment/nq_msmarco_qwen1024_recall_k30_50_100_t1/splits/"
    "nq_qwen06_1024"
)
TRAIN = Path(
    "/home/smrc/samsung-nvme/jihyo/nq_sieun_adcbq_rabitq/"
    "nq_vanilla-qw06_train.fbin"
)
QUERY = SPLIT / "seed11.cal1000.test2452.top100.test.query.fbin"
GT = SPLIT / "seed11.cal1000.test2452.top100.test.gt.ibin"
ADC_BINARY = REPO / "experiment/adcbq_optimized/faiss_adcbq_hnsw_same_pool_calibration"
FP_BINARY = REPO / "experiment/faiss_fp_hnsw_resident_from_binary_graph"
GRAPH = ROOT / "nq_qwen06_1024.fp_built_graph.M24.efC200.faiss"
GRAPH_BUILD_CSV = ROOT / "graph_build_probe.csv"
SUMMARY = ROOT / "summary_t1.csv"
MERGED = ROOT / "nq_recall_k30_50_100_with_inmemory_fp.csv"
STATUS = ROOT / "status.json"
MANIFEST = ROOT / "manifest.json"
EXISTING_SUMMARY = (
    REPO
    / "experiment/nq_msmarco_qwen1024_recall_k30_50_100_t1/"
    "nq_msmarco_qwen1024_recall_k30_50_100_t1.csv"
)

DATASET = "nq_qwen06_1024"
M = 24
EF_CONSTRUCTION = 200
ROUNDS = 3
SEARCH_THREADS = 1
BUILD_THREADS = 4
BUILD_CPUS = "13,15,23,27"
SEARCH_CPU = "13"
EFS = {
    30: (160, 168, 176, 200, 224, 248, 280, 320, 360, 400,
         448, 504, 552, 616, 672, 736, 808, 872, 952, 1024),
    50: (320, 328, 336, 352, 368, 392, 416, 448, 480, 520,
         560, 600, 640, 688, 736, 792, 848, 904, 960, 1024),
    100: (512, 513, 520, 536, 552, 568, 584, 608, 632, 656,
          680, 712, 744, 784, 816, 856, 896, 936, 976, 1024),
}


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def update_status(stage: str, **extra: object) -> None:
    payload: dict[str, object] = {}
    if STATUS.is_file():
        try:
            payload = json.loads(STATUS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
    payload.update(extra)
    if stage == "complete":
        payload.pop("error", None)
    payload["stage"] = stage
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(STATUS, payload)
    print(f"STATUS {stage} {extra}", flush=True)


def run(command: list[str], *, threads: int) -> None:
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": "1",
        "GOTO_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
    }
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=environment, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def graph_ready() -> bool:
    # A one-million-node binary HNSW graph is far larger than this floor.
    return GRAPH.is_file() and GRAPH.stat().st_size > 32 * 1024 * 1024


def build_graph() -> None:
    if graph_ready():
        print(f"REUSE_GRAPH {GRAPH}", flush=True)
        return
    update_status("building_fp_topology", build_cpus=BUILD_CPUS)
    command = [
        "taskset", "-c", BUILD_CPUS,
        str(ADC_BINARY),
        "--dataset", DATASET,
        "--train", str(TRAIN),
        "--query", str(QUERY),
        "--gt", str(GT),
        "--index", str(GRAPH),
        "--out", str(GRAPH_BUILD_CSV),
        "--m", str(M),
        "--ef-construction", str(EF_CONSTRUCTION),
        "--k", "100",
        "--max-queries", "1",
        "--rounds", "1",
        "--build-threads", str(BUILD_THREADS),
        "--search-threads", "1",
        "--adc-mode", "int16",
        "--lut-mode", "nibble",
        "--graph-build", "fp",
        "--distance-mode", "adcbq",
        "--search-impl", "inline",
        "--code-layout", "block32",
        "--no-rerank",
        "--efs", "100",
    ]
    run(command, threads=BUILD_THREADS)
    if not graph_ready():
        raise RuntimeError(f"graph build did not create a valid index: {GRAPH}")


def result_path(k: int) -> Path:
    return ROOT / f"fp_hnsw_recall_at_{k}_t1.csv"


def result_ready(k: int) -> bool:
    path = result_path(k)
    if not path.is_file():
        return False
    try:
        rows = read_csv(path)
        return (
            [int(row["efSearch"]) for row in rows] == list(EFS[k])
            and all(int(row["k"]) == k for row in rows)
            and all(int(row["nq"]) == 2452 for row in rows)
            and all(int(row["rounds"]) == ROUNDS for row in rows)
            and all(int(row["threads"]) == SEARCH_THREADS for row in rows)
            and all(row["full_vector_storage"] == "process_owned_ram" for row in rows)
        )
    except (OSError, KeyError, ValueError):
        return False


def benchmark_k(k: int) -> None:
    output = result_path(k)
    if result_ready(k):
        print(f"REUSE_RESULT k={k} {output}", flush=True)
        return
    update_status("measuring", k=k, search_cpu=SEARCH_CPU)
    command = [
        "taskset", "-c", SEARCH_CPU,
        str(FP_BINARY),
        "--dataset", DATASET,
        "--train", str(TRAIN),
        "--query", str(QUERY),
        "--gt", str(GT),
        "--graph", str(GRAPH),
        "--out", str(output),
        "--m", str(M),
        "--ef-construction", str(EF_CONSTRUCTION),
        "--k", str(k),
        "--rounds", str(ROUNDS),
        "--threads", str(SEARCH_THREADS),
        "--efs", *map(str, EFS[k]),
    ]
    run(command, threads=SEARCH_THREADS)
    if not result_ready(k):
        raise RuntimeError(f"incomplete Recall@{k} result: {output}")


def combine() -> None:
    rows: list[dict[str, object]] = []
    for k in EFS:
        source = result_path(k)
        for raw in read_csv(source):
            rows.append(
                {
                    "dataset": DATASET,
                    "corpus": "NQ",
                    "model": "Qwen3-Embedding-0.6B (1024D)",
                    "method_key": "fp32_hnsw_ram",
                    "method": "FP32 in-memory HNSW",
                    "k": k,
                    "efSearch": int(raw["efSearch"]),
                    "recall_at_k": float(raw["recall_at_k"]),
                    "qps": float(raw["qps"]),
                    "qps_median": float(raw["qps_median"]),
                    "rounds": int(raw["rounds"]),
                    "search_threads": int(raw["threads"]),
                    "test_queries": int(raw["nq"]),
                    "rerank_policy": "none",
                    "fp_reads_per_query": 0,
                    "io_engine": "none; FP32 vectors resident in process RAM",
                    "graph_topology": "same FP-built M24 efC200 topology as ADC-BQ",
                    "resident_vector_bytes": int(raw["resident_vector_bytes"]),
                    "process_rss_bytes": int(raw["process_rss_bytes"]),
                    # Older builds of the FP-HNSW runner did not export the
                    # optional HNSW statistics columns.  Recall/QPS and all
                    # protocol fields are unchanged, so leave this blank when
                    # the loaded binary predates that instrumentation.
                    "distance_evaluations_mean": (
                        float(raw["distance_evaluations_mean"])
                        if raw.get("distance_evaluations_mean")
                        else ""
                    ),
                    "source_csv": str(source.relative_to(REPO)),
                }
            )
    write_csv(SUMMARY, rows)

    merged_rows: list[dict[str, object]] = []
    if EXISTING_SUMMARY.is_file():
        merged_rows.extend(read_csv(EXISTING_SUMMARY))
    merged_rows.extend(rows)
    write_csv(MERGED, merged_rows)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN, QUERY, GT, ADC_BINARY, FP_BINARY):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.time()
    manifest_payload: dict[str, object] = {}
    if MANIFEST.is_file():
        try:
            manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest_payload = {}
    manifest_payload.update(
        {
            "protocol": "nq_qwen1024_inmemory_fp_hnsw_recall_k30_50_100",
            "dataset": DATASET,
            "train": str(TRAIN),
            "query": str(QUERY),
            "ground_truth": str(GT),
            "test_queries": 2452,
            "split_seed": 11,
            "ks": list(EFS),
            "efs_by_k": {str(k): list(values) for k, values in EFS.items()},
            "M": M,
            "efConstruction": EF_CONSTRUCTION,
            "rounds": ROUNDS,
            "search_threads": SEARCH_THREADS,
            "build_threads": BUILD_THREADS,
            "build_cpu_lane": BUILD_CPUS,
            "search_cpu_lane": SEARCH_CPU,
            "graph_build": "FP32 distances; topology reused by IndexHNSWFlat",
            "vector_storage": "complete normalized FP32 corpus in process-owned RAM",
            "recall_definition": "|ANN top-k intersect exact global FP32 top-k| / k",
        },
    )
    manifest_payload.setdefault(
        "started_at", time.strftime("%Y-%m-%dT%H:%M:%S%z")
    )
    manifest_payload.setdefault("load_average_at_start", list(os.getloadavg()))
    atomic_json(MANIFEST, manifest_payload)
    try:
        build_graph()
        for k in EFS:
            benchmark_k(k)
        combine()
        update_status(
            "complete",
            elapsed_seconds=time.time() - started,
            summary=str(SUMMARY),
            merged=str(MERGED),
        )
    except BaseException as error:
        update_status("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
