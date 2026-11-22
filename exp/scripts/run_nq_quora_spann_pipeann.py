#!/usr/bin/env python3
"""SPANN-PQ and native PipeANN-RaBitQ SSD benchmarks.

SPANN uses either PQ128 or a one-bit-equivalent ``dimension / 8``-byte PQ code
for routing/postings and fixed exact FP rerank@80 from a page-padded O_DIRECT
sidecar. PipeANN uses its native RaBitQ 1-bit (c=16) PipeSearch path without an
adaptive LB gate or a separate rerank stage.

Both indexes are built with 24 threads. The same index is searched at T=1 and
T=24, actual read requests/bytes/4-KiB pages are reported, and the large
per-dataset artifacts are removed after both searches unless --keep-index is
specified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

import run_nq_quora_aligned4k_rerank50_80 as datasets_module
import run_v051_ssd_odirect_multidataset as common
import diskveclab_full_matrix as heldout_matrix


REPO = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO / "experiment/nq_quora_spann_pipeann"
EXTERNAL_ROOT = Path("/home/smrc/samsung-nvme/jihyo/nq_quora_spann_pipeann")

SPTAG = Path("/home/jihyo/workspace/SPTAG")
SPTAG_QUANTIZER = SPTAG / "Release/quantizer"
SPTAG_BUILDER = SPTAG / "Release/ssdserving"
SPTAG_SEARCH = REPO / "experiment/spann_instrument/bin/spann_odirect_rerank_benchmark"

PIPEANN = Path("/home/jihyo/workspace/PipeANN")
PIPEANN_BUILD = PIPEANN / "build-uring-current/tests/build_disk_index"
PIPEANN_SEARCH = PIPEANN / "build-uring-current/tests/search_disk_index"

BUILD_THREADS = 24
SEARCH_THREADS = (1, 24)
ROUNDS = 3
K = 10
RERANK = 80
PAGE_BYTES = 4096

FIXED_PQ_BYTES = 128
PQ_CODE_BUDGET = "fixed128"
PQ_TRAIN_SAMPLES = 100_000
SPANN_HEAD_RATIO = 0.10
SPANN_HEAD_DEGREE = 24
SPANN_BUILD_INTERNAL = 64
SPANN_BUILD_POSTING_PAGES = 12
SPANN_SEARCH_HEADS = (
    32, 36, 40, 44, 48, 52, 56, 64, 72, 80,
    96, 112, 128, 144, 160, 192, 224, 256, 384, 512,
)
SPANN_SEARCH_POSTING_PAGES = 12

PIPEANN_R = 24
PIPEANN_LBUILD = 200
PIPEANN_MEMORY_GIB = 64
PIPEANN_BEAM = 2
PIPEANN_MEM_L = 0
PIPEANN_LSEARCH = (
    80, 88, 96, 104, 112, 120, 128, 144, 160, 176,
    192, 224, 256, 288, 320, 384, 448, 512, 768, 1024,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def pq_bytes_for_dim(dim: int) -> int:
    if PQ_CODE_BUDGET == "fixed128":
        return FIXED_PQ_BYTES
    if PQ_CODE_BUDGET != "one-bit":
        raise ValueError(f"unknown PQ code budget: {PQ_CODE_BUDGET}")
    if dim <= 0 or dim % 8:
        raise ValueError(f"one-bit-equivalent PQ requires dimension divisible by 8: {dim}")
    return dim // 8


def emit(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with (RESULT_ROOT / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def selected_datasets(keys: list[str] | None) -> list[common.Dataset]:
    datasets = (
        datasets_module.target_datasets()
        if not keys
        else list(common.DATASETS.values())
    )
    if not keys:
        return datasets
    by_key = {dataset.key: dataset for dataset in datasets}
    missing = sorted(set(keys) - set(by_key))
    if missing:
        raise ValueError(f"unknown datasets {missing}; choices={sorted(by_key)}")
    return [by_key[key] for key in keys]


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    emit(f"RUN {shlex.join(cmd)}")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"started_at={now()}\ncommand={shlex.join(cmd)}\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        returncode = process.wait()
        elapsed = time.perf_counter() - started
        log.write(f"\nfinished_at={now()}\nreturncode={returncode}\nelapsed_s={elapsed:.9f}\n")
    if returncode != 0:
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
        raise RuntimeError(f"command failed ({returncode}): {shlex.join(cmd)}\n{tail}")
    emit(f"DONE elapsed={elapsed:.1f}s log={log_path}")
    return elapsed


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def git_state(repo: Path) -> dict[str, object]:
    def output(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    return {
        "path": str(repo),
        "commit": output("rev-parse", "HEAD"),
        "diff": output("diff", "--", "."),
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def safe_cleanup(path: Path) -> None:
    target = path.resolve()
    root = EXTERNAL_ROOT.resolve()
    if target == root or root not in target.parents:
        raise RuntimeError(f"refusing unsafe cleanup: {target}")
    if target.exists():
        size = directory_bytes(target)
        shutil.rmtree(target)
        emit(f"CLEANED transient artifacts path={target} bytes={size}")


def aggregate_rows(rows: list[dict[str, object]], knob: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    values = sorted({int(row[knob]) for row in rows})
    for value in values:
        selected = [row for row in rows if int(row[knob]) == value]
        result = dict(selected[0])
        result.pop("round", None)
        result["rounds"] = len(selected)
        for key in list(result):
            if key in {knob, "threads", "nq", "nb", "dim", "k", "rerank_n", "rounds"}:
                continue
            numbers: list[float] = []
            for row in selected:
                item = row.get(key)
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    numbers.append(float(item))
            if len(numbers) == len(selected):
                result[key] = statistics.fmean(numbers)
        qps = [float(row["qps"]) for row in selected]
        result["qps_stddev"] = statistics.stdev(qps) if len(qps) > 1 else 0.0
        result["qps_min"] = min(qps)
        result["qps_max"] = max(qps)
        output.append(result)
    return output


def normalized_padded_fp_sidecar(dataset: common.Dataset, path: Path) -> tuple[float, int]:
    n, dim = common.matrix_shape(dataset.train)
    record_bytes = math.ceil(dim * 4 / PAGE_BYTES) * PAGE_BYTES
    expected = PAGE_BYTES + n * record_bytes
    if path.exists() and path.stat().st_size == expected:
        emit(f"REUSE aligned normalized FP sidecar {path}")
        return 0.0, record_bytes

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    started = time.perf_counter()
    emit(
        f"PREPARE normalized O_DIRECT FP sidecar dataset={dataset.key} "
        f"record_bytes={record_bytes} expected_bytes={expected}"
    )
    try:
        with dataset.train.open("rb", buffering=0) as source, temporary.open("wb", buffering=0) as out:
            raw_header = source.read(8)
            if len(raw_header) != 8:
                raise RuntimeError(f"short fbin header: {dataset.train}")
            src_n, src_dim = struct.unpack("<II", raw_header)
            if (src_n, src_dim) != (n, dim):
                raise RuntimeError("fbin header changed while preparing sidecar")
            metadata = struct.pack("<QQQQ", n, dim, record_bytes, PAGE_BYTES)
            out.write(metadata)
            out.write(bytes(PAGE_BYTES - len(metadata)))
            batch_rows = max(1, min(4096, (128 * 1024 * 1024) // record_bytes))
            record_floats = record_bytes // 4
            written = 0
            while written < n:
                count = min(batch_rows, n - written)
                raw = np.fromfile(source, dtype=np.float32, count=count * dim)
                if raw.size != count * dim:
                    raise RuntimeError(f"short fbin payload: {dataset.train}")
                raw = raw.reshape(count, dim)
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                np.maximum(norms, 1e-12, out=norms)
                block = np.zeros((count, record_floats), dtype=np.float32)
                block[:, :dim] = raw / norms
                block.tofile(out)
                written += count
                if written == n or written % 100_000 == 0:
                    emit(f"SIDECAR {dataset.key} {written}/{n}")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    if path.stat().st_size != expected:
        raise RuntimeError(f"bad sidecar size {path.stat().st_size}, expected {expected}")
    elapsed = time.perf_counter() - started
    emit(f"PREPARED sidecar dataset={dataset.key} elapsed={elapsed:.1f}s path={path}")
    return elapsed, record_bytes


def spann_paths(dataset: common.Dataset) -> tuple[Path, Path]:
    return EXTERNAL_ROOT / dataset.key / "spann", RESULT_ROOT / dataset.key / "spann"


def spann_config(dataset: common.Dataset, work: Path, result: Path) -> str:
    index = work / "index"
    _, dim = common.matrix_shape(dataset.train)
    pq_bytes = pq_bytes_for_dim(dim)
    quantized = work / f"base.pq{pq_bytes}.bin"
    quantizer = work / f"pq{pq_bytes}.quantizer.bin"
    tmp = work / "tmp"
    return f"""[Base]
ValueType=UInt8
DistCalcMethod=L2
IndexAlgoType=BKT
Dim={pq_bytes}
VectorPath={quantized}
VectorType=DEFAULT
IndexDirectory={index}
QuantizerFilePath={quantizer}
DeleteHeadVectors=true

[SelectHead]
isExecute=true
TreeNumber=1
BKTKmeansK=32
BKTLeafSize=8
SamplesNumber=1000
SelectThreshold=10
SplitFactor=6
SplitThreshold=25
Ratio={SPANN_HEAD_RATIO}
NumberOfThreads={BUILD_THREADS}
BKTLambdaFactor=1.0

[BuildHead]
isExecute=true
NeighborhoodSize={SPANN_HEAD_DEGREE}
TPTNumber=32
TPTLeafSize=2000
MaxCheck=4096
MaxCheckForRefineGraph=4096
RefineIterations=3
NumberOfThreads={BUILD_THREADS}
BKTLambdaFactor=-1.0

[BuildSSDIndex]
isExecute=true
BuildSsdIndex=true
InternalResultNum={SPANN_BUILD_INTERNAL}
ReplicaCount=8
PostingPageLimit={SPANN_BUILD_POSTING_PAGES}
NumberOfThreads={BUILD_THREADS}
MaxCheck=4096
TmpDir={tmp}
SearchInternalResultNum=64
SearchPostingPageLimit={SPANN_SEARCH_POSTING_PAGES}
UseDirectIO=true
IOThreadsPerHandler=4

[SearchSSDIndex]
isExecute=false
"""


SPANN_STAGE_RE = re.compile(
    r"select head time:\s*(?P<select>[0-9.]+)s\s+"
    r"build head time:\s*(?P<head>[0-9.]+)s\s+"
    r"build ssd time:\s*(?P<ssd>[0-9.]+)s"
)
SPANN_RESULT_RE = re.compile(r"^SPANN_RESULT (?P<body>.*)$")
SPANN_STATS_RE = re.compile(r"^SPANN_QUERY_STATS metric=(?P<metric>[a-z0-9_]+) (?P<body>.*)$")


def parse_key_values(body: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in body.split():
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def build_spann(dataset: common.Dataset, force: bool) -> dict[str, object]:
    work, result = spann_paths(dataset)
    index = work / "index"
    manifest_path = result / "build_manifest.json"
    _, dim = common.matrix_shape(dataset.train)
    pq_bytes = pq_bytes_for_dim(dim)
    if not force and (index / "indexloader.ini").exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("pq_bytes_per_vector", -1)) == pq_bytes:
            emit(f"REUSE SPANN build {dataset.key}")
            return manifest
    if work.exists():
        safe_cleanup(work)
    work.mkdir(parents=True)
    (work / "index").mkdir()
    (work / "tmp").mkdir()
    result.mkdir(parents=True, exist_ok=True)
    config_path = result / "build_config.ini"
    config_path.write_text(spann_config(dataset, work, result), encoding="utf-8")
    n, dim = common.matrix_shape(dataset.train)

    quantizer_elapsed = run_logged(
        [
            str(SPTAG_QUANTIZER),
            "-d", str(dim),
            "-v", "Float",
            "-f", "DEFAULT",
            "-i", str(dataset.train),
            "-o", str(work / f"base.pq{pq_bytes}.bin"),
            "-t", str(BUILD_THREADS),
            "-norm", "true",
            "-oq", str(work / f"pq{pq_bytes}.quantizer.bin"),
            "-qt", "PQQuantizer",
            "-qd", str(pq_bytes),
            "-ts", str(min(PQ_TRAIN_SAMPLES, n)),
            "-dist", "L2",
        ],
        result / "quantize.log",
    )
    build_elapsed = run_logged([str(SPTAG_BUILDER), str(config_path)], result / "build.log")
    text = (result / "build.log").read_text(encoding="utf-8", errors="replace")
    stages = list(SPANN_STAGE_RE.finditer(text))
    stage_values: dict[str, float] = {}
    if stages:
        raw = stages[-1].groupdict()
        stage_values = {f"spann_{key}_seconds_reported": float(value) for key, value in raw.items()}
    payload: dict[str, object] = {
        "dataset": dataset.key,
        "method": (
            f"SPANN PQ{pq_bytes} (1-bit-equivalent budget) + fixed exact FP O_DIRECT rerank@80"
            if PQ_CODE_BUDGET == "one-bit"
            else "SPANN PQ128 + fixed exact FP O_DIRECT rerank@80"
        ),
        "nb": n,
        "dim": dim,
        "build_threads": BUILD_THREADS,
        "pq_bytes_per_vector": pq_bytes,
        "pq_code_budget": PQ_CODE_BUDGET,
        "pq_nominal_bits_per_dimension": 8.0 * pq_bytes / dim,
        "resident_code_scope": "selected SPANN heads only; PQ posting codes remain on SSD",
        "pq_training_samples": min(PQ_TRAIN_SAMPLES, n),
        "head_ratio": SPANN_HEAD_RATIO,
        "head_graph_degree": SPANN_HEAD_DEGREE,
        "build_internal_result_num": SPANN_BUILD_INTERNAL,
        "build_posting_page_limit": SPANN_BUILD_POSTING_PAGES,
        "quantizer_train_and_encode_seconds": quantizer_elapsed,
        "spann_build_and_persist_seconds": build_elapsed,
        "build_end_to_end_seconds": quantizer_elapsed + build_elapsed,
        "build_timing_scope": "PQ train+encode, head selection, head graph, SSD postings/layout, top-level persist",
        "index_bytes": directory_bytes(index),
        "storage": "SSD O_DIRECT",
        "io_engine": "SPTAG liburing for postings; liburing O_DIRECT batch for FP rerank",
        "source": git_state(SPTAG),
        **stage_values,
    }
    write_json(manifest_path, payload)
    return payload


def parse_spann_search(
    dataset: common.Dataset,
    log_path: Path,
    threads: int,
    internal_heads: int,
    round_id: int,
) -> dict[str, object]:
    result_values: dict[str, str] | None = None
    stats: dict[str, dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SPANN_RESULT_RE.match(line.strip())
        if match:
            result_values = parse_key_values(match.group("body"))
        match = SPANN_STATS_RE.match(line.strip())
        if match:
            stats[match.group("metric")] = {
                key: float(value)
                for key, value in parse_key_values(match.group("body")).items()
            }
    if result_values is None:
        raise RuntimeError(f"missing SPANN_RESULT in {log_path}")
    required = {
        "latency_us", "head_us", "posting_us", "rerank_io_us", "rerank_cpu_us",
        "posting_requests", "posting_pages_4k", "posting_bytes", "rerank_requests",
        "rerank_pages_4k", "rerank_bytes", "total_requests", "total_pages_4k", "total_bytes",
    }
    if missing := sorted(required - set(stats)):
        raise RuntimeError(f"missing SPANN stats {missing} in {log_path}")
    n, dim = common.matrix_shape(dataset.train)
    nq, _ = common.matrix_shape(dataset.query)
    pq_bytes = pq_bytes_for_dim(dim)
    row: dict[str, object] = {
        "dataset": dataset.key,
        "corpus": dataset.corpus,
        "model": dataset.model,
        "method": (
            f"SPANN PQ{pq_bytes} (1-bit-equivalent budget) + fixed exact FP O_DIRECT rerank@80"
            if PQ_CODE_BUDGET == "one-bit"
            else "SPANN PQ128 + fixed exact FP O_DIRECT rerank@80"
        ),
        "family": "spann_pq_onebit_budget_fixed_fp_rerank80" if PQ_CODE_BUDGET == "one-bit" else "spann_pq128_fixed_fp_rerank80",
        "metric": "cosine (normalized L2 PQ; exact cosine rerank)",
        "threads": threads,
        "build_threads": BUILD_THREADS,
        "internal_heads": internal_heads,
        "posting_page_limit": SPANN_SEARCH_POSTING_PAGES,
        "pq_bytes": pq_bytes,
        "pq_code_budget": PQ_CODE_BUDGET,
        "pq_nominal_bits_per_dimension": 8.0 * pq_bytes / dim,
        "resident_code_scope": "selected SPANN heads only; PQ posting codes remain on SSD",
        "rerank_n": RERANK,
        "k": K,
        "nq": nq,
        "nb": n,
        "dim": dim,
        "round": round_id,
        "qps": float(result_values["qps"]),
        "recall_at_10": float(result_values["recall_at_k"]),
        "wall_seconds": float(result_values["wall_seconds"]),
        "storage": "SSD O_DIRECT",
        "io_engine": "io_uring postings + io_uring O_DIRECT FP batch",
        "cpu_ssd_overlap": (
            f"T{threads}: concurrent query workers overlap posting I/O, FP I/O, and CPU; "
            "no intra-query posting/rerank overlap"
            if threads > 1
            else "T1: no cross-query overlap and no intra-query posting/rerank overlap"
        ),
        "fp_record_layout": "normalized FP32, each vector padded to whole 4KiB pages",
    }
    for metric, distribution in stats.items():
        for statistic, value in distribution.items():
            row[f"{metric}_{statistic}"] = value
    return row


def search_spann(dataset: common.Dataset, rounds: int) -> None:
    work, result = spann_paths(dataset)
    index = work / "index"
    if not (index / "indexloader.ini").exists():
        raise FileNotFoundError(f"missing SPANN index for {dataset.key}")
    sidecar = work / "base.normalized.fp32.padded4k.bin"
    sidecar_elapsed, record_bytes = normalized_padded_fp_sidecar(dataset, sidecar)
    manifest_path = result / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fp_sidecar_prepare_seconds_outside_timed_search"] = sidecar_elapsed
    manifest["fp_sidecar_record_bytes"] = record_bytes
    manifest["fp_pages_per_reranked_vector"] = record_bytes // PAGE_BYTES
    write_json(manifest_path, manifest)

    for threads in SEARCH_THREADS:
        output_path = result / f"search_t{threads}.csv"
        all_rows: list[dict[str, object]] = []
        raw_dir = result / f"raw_t{threads}"
        for round_id in range(1, rounds + 1):
            for internal_heads in SPANN_SEARCH_HEADS:
                log = raw_dir / f"heads{internal_heads}.round{round_id}.log"
                run_logged(
                    [
                        str(SPTAG_SEARCH), str(index), str(dataset.query), str(dataset.gt), str(sidecar),
                        str(threads), str(internal_heads), str(SPANN_SEARCH_POSTING_PAGES),
                        str(RERANK), str(K),
                    ],
                    log,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                    },
                )
                all_rows.append(parse_spann_search(dataset, log, threads, internal_heads, round_id))
        write_csv(result / f"search_t{threads}_all_rounds.csv", all_rows)
        write_csv(output_path, aggregate_rows(all_rows, "internal_heads"))


PIPEANN_ROW_RE = re.compile(
    r"^\s*(?P<L>\d+)\s+(?P<beam>\d+)\s+(?P<qps>[0-9.]+)\s+"
    r"(?P<avg_latency>[0-9.]+)\s+(?P<p99_latency>[0-9.]+)\s+"
    r"(?P<mean_hops>[0-9.]+)\s+(?P<mean_pages>[0-9.]+)\s+(?P<recall>[0-9.]+)\s*$"
)
PIPEANN_STATS_RE = re.compile(
    r"^PIPEANN_QUERY_STATS block=(?P<block>\d+) metric=(?P<metric>[a-z0-9_]+) (?P<body>.*)$"
)

# Native PipeSearch currently leaves io_us/hops unset and stores the comparison
# count in QueryStats::cpu_us.  Keep those raw values in the logs for provenance,
# but never expose them as timings/counters in the parsed benchmark CSVs.
PIPEANN_UNAVAILABLE_METRICS = {"io_us", "cpu_us", "hops"}


def sanitize_pipeann_row(row: dict[str, object]) -> dict[str, object]:
    for metric in PIPEANN_UNAVAILABLE_METRICS:
        for key in list(row):
            if key == metric or key.startswith(f"{metric}_"):
                row.pop(key)
    row["internal_io_cpu_timing"] = (
        "not reported by native PipeSearch; latency_us is end-to-end query latency"
    )
    row["hops_status"] = "not reported by native PipeSearch"
    return row


def pipeann_paths(dataset: common.Dataset) -> tuple[Path, Path]:
    return EXTERNAL_ROOT / dataset.key / "pipeann", RESULT_ROOT / dataset.key / "pipeann"


def build_pipeann(dataset: common.Dataset, force: bool) -> dict[str, object]:
    work, result = pipeann_paths(dataset)
    prefix = work / "index" / "index"
    manifest_path = result / "build_manifest.json"
    if not force and Path(f"{prefix}_disk.index").exists() and manifest_path.exists():
        emit(f"REUSE PipeANN build {dataset.key}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if work.exists():
        safe_cleanup(work)
    prefix.parent.mkdir(parents=True)
    result.mkdir(parents=True, exist_ok=True)
    n, dim = common.matrix_shape(dataset.train)
    build_neighbor_bytes = pq_bytes_for_dim(dim)
    elapsed = run_logged(
        [
            str(PIPEANN_BUILD), "float", str(dataset.train), str(prefix),
            str(PIPEANN_R), str(PIPEANN_LBUILD), str(build_neighbor_bytes),
            str(PIPEANN_MEMORY_GIB), str(BUILD_THREADS), "cosine", "rabitq",
        ],
        result / "build.log",
        env={**os.environ, "OMP_NUM_THREADS": str(BUILD_THREADS), "MKL_NUM_THREADS": str(BUILD_THREADS)},
    )
    text = (result / "build.log").read_text(encoding="utf-8", errors="replace")
    vamana = re.findall(r"Vamana index built in:\s*([0-9.]+)s", text)
    indexing = re.findall(r"Indexing time:\s*([0-9.]+)", text)
    payload: dict[str, object] = {
        "dataset": dataset.key,
        "method": "PipeANN native RaBitQ 1-bit c=16 (no adaptive LB, no separate rerank)",
        "nb": n,
        "dim": dim,
        "R": PIPEANN_R,
        "Lbuild": PIPEANN_LBUILD,
        "build_threads": BUILD_THREADS,
        "build_memory_budget_GiB": PIPEANN_MEMORY_GIB,
        "rabitq_c": 16,
        "data_bits": 1,
        "query_bits": 4,
        "routing_code_bytes_per_vector": dim // 8,
        "routing_metadata_bytes_per_vector": 9,
        "routing_total_bytes_per_vector": dim // 8 + 9,
        "build_memory_estimator_bytes_per_neighbor": build_neighbor_bytes,
        "resident_code_scope": "all database vectors",
        "adaptive_lb": False,
        "separate_fp_rerank": False,
        "build_end_to_end_seconds": elapsed,
        "vamana_seconds_reported": float(vamana[-1]) if vamana else None,
        "indexing_seconds_reported": float(indexing[-1]) if indexing else None,
        "build_timing_scope": "normalization, RaBitQ pivot/code generation, Vamana graph, SSD layout",
        "index_bytes": directory_bytes(work / "index"),
        "storage": "SSD O_DIRECT",
        "io_engine": "PipeANN native io_uring PipeSearch",
        "source": git_state(PIPEANN),
    }
    write_json(manifest_path, payload)
    return payload


def parse_pipeann_search(
    dataset: common.Dataset,
    log_path: Path,
    threads: int,
    round_id: int,
) -> list[dict[str, object]]:
    stats: dict[int, dict[str, dict[str, float]]] = {}
    rows_by_l: dict[int, dict[str, str]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = PIPEANN_STATS_RE.match(line.strip())
        if match:
            block = int(match.group("block"))
            stats.setdefault(block, {})[match.group("metric")] = {
                key: float(value)
                for key, value in parse_key_values(match.group("body")).items()
            }
        match = PIPEANN_ROW_RE.match(line)
        if match:
            values = match.groupdict()
            rows_by_l[int(values["L"])] = values
    if tuple(rows_by_l) != PIPEANN_LSEARCH:
        raise RuntimeError(f"bad PipeANN L rows {tuple(rows_by_l)} in {log_path}")
    if len(stats) != len(PIPEANN_LSEARCH):
        raise RuntimeError(f"bad PipeANN stats blocks {sorted(stats)} in {log_path}")
    n, dim = common.matrix_shape(dataset.train)
    nq, _ = common.matrix_shape(dataset.query)
    output: list[dict[str, object]] = []
    for block, search_l in enumerate(PIPEANN_LSEARCH):
        raw = rows_by_l[search_l]
        required = {"latency_us", "page_reads_4k", "io_requests", "read_bytes", "comparisons"}
        if missing := sorted(required - set(stats[block])):
            raise RuntimeError(f"missing PipeANN stats block={block}: {missing}")
        row: dict[str, object] = {
            "dataset": dataset.key,
            "corpus": dataset.corpus,
            "model": dataset.model,
            "method": "PipeANN native RaBitQ 1-bit c=16",
            "family": "pipeann_native_rabitq_1bit_c16",
            "metric": "cosine",
            "threads": threads,
            "build_threads": BUILD_THREADS,
            "Lsearch": search_l,
            "beam_width": PIPEANN_BEAM,
            "R": PIPEANN_R,
            "Lbuild": PIPEANN_LBUILD,
            "rabitq_c": 16,
            "adaptive_lb": False,
            "separate_fp_rerank": False,
            "rerank_n": 0,
            "k": K,
            "nq": nq,
            "nb": n,
            "dim": dim,
            "round": round_id,
            "qps": float(raw["qps"]),
            "recall_at_10": float(raw["recall"]) / 100.0,
            "storage": "SSD O_DIRECT",
            "io_engine": "native io_uring PipeSearch",
            "read_scope": "combined native graph record: FP vector + neighbors",
            "routing_code_bytes_per_vector": dim // 8,
            "routing_metadata_bytes_per_vector": 9,
            "routing_total_bytes_per_vector": dim // 8 + 9,
            "resident_code_scope": "all database vectors",
            "cpu_ssd_overlap": (
                f"async intra-query PipeSearch plus {threads} concurrent query workers"
                if threads > 1
                else "async intra-query PipeSearch with one query worker"
            ),
        }
        for metric, distribution in stats[block].items():
            if metric in PIPEANN_UNAVAILABLE_METRICS:
                continue
            for statistic, value in distribution.items():
                row[f"{metric}_{statistic}"] = value
        output.append(sanitize_pipeann_row(row))
    return output


def search_pipeann(dataset: common.Dataset, rounds: int) -> None:
    work, result = pipeann_paths(dataset)
    prefix = work / "index" / "index"
    if not Path(f"{prefix}_disk.index").exists():
        raise FileNotFoundError(f"missing PipeANN index for {dataset.key}")
    for threads in SEARCH_THREADS:
        all_rows: list[dict[str, object]] = []
        raw_dir = result / f"raw_t{threads}"
        for round_id in range(1, rounds + 1):
            log = raw_dir / f"search.round{round_id}.log"
            run_logged(
                [
                    str(PIPEANN_SEARCH), "float", str(prefix), str(threads), str(PIPEANN_BEAM),
                    str(dataset.query), str(dataset.gt), str(K), "cosine", "rabitq", "2",
                    str(PIPEANN_MEM_L), *map(str, PIPEANN_LSEARCH),
                ],
                log,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": str(threads),
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
            all_rows.extend(parse_pipeann_search(dataset, log, threads, round_id))
        write_csv(result / f"search_t{threads}_all_rounds.csv", all_rows)
        write_csv(result / f"search_t{threads}.csv", aggregate_rows(all_rows, "Lsearch"))


def method_complete(dataset: common.Dataset, method: str) -> bool:
    result = RESULT_ROOT / dataset.key / method
    manifest_path = result / "build_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _, dim = common.matrix_shape(dataset.train)
        if method == "spann" and int(manifest.get("pq_bytes_per_vector", -1)) != pq_bytes_for_dim(dim):
            return False
        if method == "pipeann" and (
            int(manifest.get("data_bits", -1)) != 1
            or int(manifest.get("rabitq_c", -1)) != 16
        ):
            return False
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    expected = SPANN_SEARCH_HEADS if method == "spann" else PIPEANN_LSEARCH
    for threads in SEARCH_THREADS:
        path = result / f"search_t{threads}.csv"
        if not path.is_file():
            return False
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(expected):
            return False
        if any(int(row.get("rounds", 0)) != ROUNDS for row in rows):
            return False
    return True


def collect_reports(datasets: Iterable[common.Dataset]) -> None:
    build_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    for dataset in datasets:
        for method in ("spann", "pipeann"):
            result = RESULT_ROOT / dataset.key / method
            manifest = result / "build_manifest.json"
            if manifest.exists():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                payload.pop("source", None)
                build_rows.append(payload)
            for threads in SEARCH_THREADS:
                path = result / f"search_t{threads}.csv"
                if path.exists():
                    with path.open(newline="", encoding="utf-8") as handle:
                        for row in csv.DictReader(handle):
                            if row.get("family") == "pipeann_native_rabitq_1bit_c16":
                                sanitize_pipeann_row(row)
                            search_rows.append(row)
    if build_rows:
        write_csv(RESULT_ROOT / "build_times.csv", build_rows)
    if search_rows:
        write_csv(RESULT_ROOT / "all_search_results.csv", search_rows)


def verify_tools() -> None:
    required = [SPTAG_QUANTIZER, SPTAG_BUILDER, SPTAG_SEARCH, PIPEANN_BUILD, PIPEANN_SEARCH]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing benchmark tools: {missing}")
    cache = PIPEANN / "build-uring-current/CMakeCache.txt"
    if "IO_ENGINE:STRING=uring" not in cache.read_text(encoding="utf-8"):
        raise RuntimeError("PipeANN build is not IO_ENGINE=uring")


def main() -> None:
    global RESULT_ROOT, EXTERNAL_ROOT, ROUNDS, SEARCH_THREADS
    global SPANN_SEARCH_HEADS, PIPEANN_LSEARCH, PQ_CODE_BUDGET
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--method", choices=("all", "spann", "pipeann"), default="all")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument(
        "--search-threads",
        nargs="+",
        type=int,
        choices=(1, 24),
        default=list(SEARCH_THREADS),
        help="search thread counts to measure (default: 1 24)",
    )
    parser.add_argument(
        "--spann-heads",
        nargs="+",
        type=int,
        default=list(SPANN_SEARCH_HEADS),
    )
    parser.add_argument(
        "--pipeann-lsearch",
        nargs="+",
        type=int,
        default=list(PIPEANN_LSEARCH),
    )
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--external-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument(
        "--pq-code-budget",
        choices=("fixed128", "one-bit"),
        default=PQ_CODE_BUDGET,
        help="SPANN PQ bytes/vector: historical 128 or dimension/8; also used for PipeANN's build estimator",
    )
    parser.add_argument("--keep-index", action="store_true")
    parser.add_argument(
        "--heldout-matrix",
        action="store_true",
        help="evaluate on the seed-11 ADC-BQ held-out query/GT split",
    )
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    RESULT_ROOT = args.result_root.resolve()
    EXTERNAL_ROOT = args.external_root.resolve()
    PQ_CODE_BUDGET = args.pq_code_budget
    ROUNDS = args.rounds
    SEARCH_THREADS = tuple(dict.fromkeys(args.search_threads))
    SPANN_SEARCH_HEADS = tuple(dict.fromkeys(args.spann_heads))
    PIPEANN_LSEARCH = tuple(dict.fromkeys(args.pipeann_lsearch))
    datasets = selected_datasets(args.datasets)
    if args.heldout_matrix:
        datasets = [heldout_matrix.heldout_dataset(dataset.key) for dataset in datasets]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    verify_tools()
    if args.report_only:
        collect_reports(datasets)
        return

    methods = ("spann", "pipeann") if args.method == "all" else (args.method,)
    for dataset in datasets:
        emit(f"DATASET {dataset.key} corpus={dataset.corpus} model={dataset.model}")
        if "spann" in methods:
            work, _ = spann_paths(dataset)
            if method_complete(dataset, "spann") and not args.force_build:
                emit(f"SKIP complete SPANN {dataset.key}")
            else:
                build_spann(dataset, args.force_build)
                search_spann(dataset, args.rounds)
            if not args.keep_index and method_complete(dataset, "spann"):
                safe_cleanup(work)
        if "pipeann" in methods:
            work, _ = pipeann_paths(dataset)
            if method_complete(dataset, "pipeann") and not args.force_build:
                emit(f"SKIP complete PipeANN {dataset.key}")
            else:
                build_pipeann(dataset, args.force_build)
                search_pipeann(dataset, args.rounds)
            if not args.keep_index and method_complete(dataset, "pipeann"):
                safe_cleanup(work)
        collect_reports(datasets)
    emit("ALL REQUESTED SPANN/PIPEANN EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
